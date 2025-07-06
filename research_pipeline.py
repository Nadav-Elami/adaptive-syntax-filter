from __future__ import annotations

"""High-level orchestration for the Adaptive Syntax Filter analysis pipeline.

This module bundles together data generation, model fitting, visualization and
result archiving into a single *headless* entry-point so that external scripts
(`cli.py`, `batch_processing.py`, notebooks) can trigger complete runs without
knowing internal details.

The **current implementation** is a lightweight stub that wires up the minimum
functionality required for Phase 6 integration testing. It should be replaced
or extended with concrete logic once the full pipeline components are ready.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from config_cli import _read_config as _load_config  # Re-use YAML/JSON loader
from result_archiving import stash_artifact

# Data generation tools
from adaptive_syntax_filter.data.sequence_generator import (
    GenerationConfig,
    SequenceGenerator,
    sequences_to_observations,
    generate_dataset,
)

# EM algorithm
from adaptive_syntax_filter.core.state_space import StateSpaceManager
from adaptive_syntax_filter.core.em_algorithm import EMAlgorithm, EMParameters, EMResults

# Visualization
from adaptive_syntax_filter.viz.logit_evolution import create_logit_evolution_summary
from adaptive_syntax_filter.viz.probability_evolution import ProbabilityEvolutionAnalyzer, create_transition_heatmap_series
from adaptive_syntax_filter.viz.performance_assessment import PerformanceAnalyzer, create_fit_evaluation_plots, create_convergence_analysis
from adaptive_syntax_filter.viz.sequence_analysis import SequenceAnalyzer, analyze_sequence_lengths, analyze_symbol_usage, create_sequence_length_plot, create_symbol_usage_plot
from adaptive_syntax_filter.viz.publication_figures import PublicationFigureManager, create_main_results_figure

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------

_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_LOG_DIR.mkdir(exist_ok=True)


def _setup_logging(experiment_id: str) -> Path:
    """Initialise logging to console and file.

    Returns the path of the created log file so it can be archived later.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _DEFAULT_LOG_DIR / f"pipeline_{experiment_id}_{ts}.log"

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)
    logger.debug("Logging initialised – file: %s", log_path)
    return log_path


def _get_git_revision() -> str:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover – git may be unavailable
        commit = "unknown"
    return commit


# -----------------------------------------------------------------------------
# Public API – these functions are imported by ``cli.py`` and other scripts.
# -----------------------------------------------------------------------------

def run_pipeline(config: Dict[str, Any] | str | Path) -> bool:  # noqa: D401
    """Run the Adaptive Syntax Filter analysis pipeline.

    The pipeline consists of four stages: data generation, model fitting using
    the EM algorithm, visualisation, and result archiving. Recoverable errors
    in each stage are logged; processing continues with subsequent stages
    where possible.

    Parameters
    ----------
    config
        Either a path to a YAML/JSON configuration file **or** an already
        parsed dictionary.

    Returns
    -------
    bool
        ``True`` if the pipeline completed without critical errors, otherwise
        ``False``.

    Examples
    --------
    >>> from research_pipeline import run_pipeline
    >>> run_pipeline('configs/demo.yml')
    True
    """

    # ------------------------------------------------------------------
    # Load configuration ------------------------------------------------
    # ------------------------------------------------------------------
    if isinstance(config, (str, Path)):
        try:
            cfg: Dict[str, Any] = _load_config(Path(config))
        except Exception as exc:
            logger.error("Failed to load configuration: %s", exc)
            return False
    else:
        cfg = dict(config)  # shallow copy to avoid mutating caller data

    experiment_id = cfg.get("experiment_id", "exp")
    log_path = _setup_logging(experiment_id)

    logger.info("📌 Git revision: %s", _get_git_revision())
    logger.info("🚀 Starting pipeline – experiment_id=%s", experiment_id)

    # Ensure mandatory sections ------------------------------------------------
    if "data" not in cfg:
        logger.error("Configuration missing 'data' section – aborting")
        return False

    data_cfg: Dict[str, Any] = cfg["data"]

    required_data_keys = {"alphabet", "order", "n_sequences"}
    if not required_data_keys.issubset(data_cfg):
        logger.error("Data config missing keys: %s", required_data_keys - set(data_cfg))
        return False

    # ------------------------------------------------------------------
    # Prepare temporary workspace --------------------------------------
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="asf_run_") as tmp_str:
        tmp_dir = Path(tmp_str)
        logger.debug("Created temporary workspace: %s", tmp_dir)

        ###############################
        # Stage 1 – Data generation  #
        ###############################
        sequences: List[List[str]] = []
        parameter_traj = None  # type: ignore[assignment]
        stage_success = {
            "data": False,
            "em": False,
            "viz": False,
            "archive": False,
        }

        try:
            gen_cfg = GenerationConfig(**data_cfg)
            logger.info("[1/4] Generating data – %d sequences", gen_cfg.n_sequences)
            sequences, parameter_traj = generate_dataset(gen_cfg)
            stage_success["data"] = True
            # Persist raw data for archiving
            dataset_path = tmp_dir / "dataset.json"
            with open(dataset_path, "w", encoding="utf-8") as fp:
                json.dump({"sequences": sequences}, fp)
        except Exception as exc:
            logger.exception("Data generation failed: %s", exc)

        ################################
        # Stage 2 – EM algorithm fit   #
        ################################
        em_results: Optional[EMResults] = None
        try:
            if stage_success["data"]:
                observations = sequences_to_observations(sequences, gen_cfg.alphabet)
                state_mgr = StateSpaceManager(
                    alphabet_size=len(gen_cfg.alphabet),
                    markov_order=gen_cfg.order,
                    alphabet=gen_cfg.alphabet,
                )
                em_cfg = cfg.get("em", {})
                # Ensure numeric values are properly converted from YAML
                tolerance = em_cfg.get("tolerance", 1e-4)
                if isinstance(tolerance, str):
                    tolerance = float(tolerance)
                regularization_lambda = em_cfg.get("regularization_lambda", 1e-3)
                if isinstance(regularization_lambda, str):
                    regularization_lambda = float(regularization_lambda)
                max_iterations = em_cfg.get("max_iterations", 25)
                if isinstance(max_iterations, str):
                    max_iterations = int(max_iterations)
                damping_factor = em_cfg.get("damping_factor", 1.0)
                if isinstance(damping_factor, str):
                    damping_factor = float(damping_factor)
                adaptive_damping = em_cfg.get("adaptive_damping", True)
                if isinstance(adaptive_damping, str):
                    adaptive_damping = adaptive_damping.lower() == "true"
                
                em_algo = EMAlgorithm(
                    state_mgr,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    regularization_lambda=regularization_lambda,
                    damping_factor=damping_factor,
                    adaptive_damping=adaptive_damping,
                    verbose=False,
                )
                init_params = em_algo.initialize_parameters(observations)
                em_results = em_algo.fit(observations, init_params)
                stage_success["em"] = True
                
                # Log best iteration information
                logger.info(f"EM completed: {len(em_results.statistics_history)} iterations")
                logger.info(f"Best iteration: {em_results.best_iteration} (LL: {em_results.best_log_likelihood:.6f})")
                logger.info(f"Final iteration: {len(em_results.statistics_history)-1} (LL: {em_results.statistics_history[-1].log_likelihood:.6f})")
                
                # Save summary statistics
                def _to_serializable(obj):  # noqa: WPS430
                    """Convert numpy types to native for JSON."""
                    if isinstance(obj, (np.generic,)):
                        return obj.item()
                    return obj

                stats_path = tmp_dir / "em_stats.json"
                with open(stats_path, "w", encoding="utf-8") as fp:
                    json.dump([ {k: _to_serializable(v) for k, v in s.__dict__.items()} for s in em_results.statistics_history], fp, indent=2)
                
                # Save estimated logits trajectory from BEST iteration
                logits_estimated_mat = None
                if em_algo.kalman_filter is not None:
                    # Use best iteration parameters to get smoothed estimates
                    try:
                        # Temporarily set the Kalman filter to use best parameters
                        original_params = em_algo.current_params
                        em_algo.current_params = em_results.best_params
                        
                        # Get smoothed estimates with best parameters
                        x_smoothed, _, _ = em_algo.e_step(observations, em_results.best_params)
                        
                        # Restore original parameters
                        em_algo.current_params = original_params
                        
                        # x_smoothed: shape (n_sequences, state_dim) -> transpose for viz
                        logits_estimated_mat = x_smoothed.T
                        npy_path = tmp_dir / "logits_estimated.npy"
                        np.save(npy_path, logits_estimated_mat)
                        
                        logger.info(f"Saved logits from best iteration {em_results.best_iteration}")
                    except Exception as e:
                        logger.exception(f"Failed to extract smoothed state trajectory from best iteration: {e}")

            # After EM fitting and before visualization
            if em_results is not None:
                # Create figures directory first
                figures_dir = tmp_dir / "figures"
                figures_dir.mkdir(exist_ok=True)
                create_numerical_diagnostics(em_results, figures_dir)
        except Exception as exc:
            logger.exception("EM algorithm failed: %s", exc)

        ###############################
        # Stage 3 – Visualisation     #
        ###############################
        figures: Dict[str, Any] = {}
        try:
            if stage_success["em"] and "logits_estimated_mat" in locals() and logits_estimated_mat is not None:
                # Create figures directory
                figures_dir = tmp_dir / "figures"
                figures_dir.mkdir(exist_ok=True)
                
                results_dict = {"logits_estimated": logits_estimated_mat}
                if parameter_traj is not None:
                    results_dict["logits_true"] = parameter_traj
                
                # 1. Logit evolution analysis (existing)
                logger.info("[3/4] Creating logit evolution analysis")
                figures = create_logit_evolution_summary(
                    results_dict,
                    gen_cfg.alphabet,
                    gen_cfg.order,
                    save_dir=figures_dir,
                )
                
                # 2. Probability evolution analysis
                logger.info("[3/4] Creating probability evolution analysis")
                if parameter_traj is not None:
                    # Convert logits to transition matrices for visualization
                    from adaptive_syntax_filter.core.observation_model import softmax_observation_model
                    from src.adaptive_syntax_filter.data.sequence_generator import softmax_mc_higher_order
                    # Convert logits to probabilities for each time point
                    n_timepoints = parameter_traj.shape[1]
                    R = len(gen_cfg.alphabet)
                    order = gen_cfg.order
                    state_dim = R ** (order + 1)
                    true_probs = np.zeros((state_dim, n_timepoints))
                    estimated_probs = np.zeros((state_dim, n_timepoints))
                    for t in range(n_timepoints):
                        if order > 1:
                            true_probs[:, t] = softmax_mc_higher_order(parameter_traj[:, t], gen_cfg.alphabet, order)
                            estimated_probs[:, t] = softmax_mc_higher_order(logits_estimated_mat[:, t], gen_cfg.alphabet, order)
                        else:
                            true_probs[:, t] = softmax_observation_model(parameter_traj[:, t], R)
                            estimated_probs[:, t] = softmax_observation_model(logits_estimated_mat[:, t], R)
                    
                    # Create transition heatmap series
                    time_points = list(range(min(5, n_timepoints)))  # Show first 5 time points
                    heatmap_fig = create_transition_heatmap_series(
                        true_probs, gen_cfg.alphabet, time_points, gen_cfg.order
                    )
                    heatmap_fig.savefig(figures_dir / "transition_heatmap_series.png", dpi=300, bbox_inches='tight')
                    plt.close(heatmap_fig)
                    
                    # Analyze evolution patterns
                    prob_analyzer = ProbabilityEvolutionAnalyzer(gen_cfg.alphabet, gen_cfg.order)
                    evolution_analysis = prob_analyzer.analyze_evolution_patterns(true_probs)
                    logger.info(f"Found {len(evolution_analysis['significant_transitions'])} significant transitions")
                
                # 3. Performance assessment
                logger.info("[3/4] Creating performance assessment")
                if parameter_traj is not None:
                    # Convert logits to probabilities for comparison
                    from adaptive_syntax_filter.core.observation_model import softmax_observation_model
                    
                    # Convert logits to probabilities for each time point
                    n_timepoints = parameter_traj.shape[1]
                    R = len(gen_cfg.alphabet)
                    order = gen_cfg.order
                    state_dim = R ** (order + 1)
                    true_probs = np.zeros((state_dim, n_timepoints))
                    estimated_probs = np.zeros((state_dim, n_timepoints))
                    
                    for t in range(n_timepoints):
                        if order > 1:
                            true_probs[:, t] = softmax_mc_higher_order(parameter_traj[:, t], gen_cfg.alphabet, order)
                            estimated_probs[:, t] = softmax_mc_higher_order(logits_estimated_mat[:, t], gen_cfg.alphabet, order)
                        else:
                            true_probs[:, t] = softmax_observation_model(parameter_traj[:, t], R)
                            estimated_probs[:, t] = softmax_observation_model(logits_estimated_mat[:, t], R)
                    
                    # Create performance plots
                    from adaptive_syntax_filter.viz.performance_assessment import evaluate_model_fit
                    metrics = evaluate_model_fit(estimated_probs, true_probs)
                    perf_figures = create_fit_evaluation_plots(estimated_probs, true_probs, metrics)
                    perf_figures.savefig(figures_dir / "performance_assessment.png", dpi=300, bbox_inches='tight')
                    plt.close(perf_figures)
                    
                    # Create convergence analysis
                    if em_results is not None:
                        convergence_history = [stats.log_likelihood for stats in em_results.statistics_history]
                        iterations = list(range(1, len(convergence_history) + 1))
                        conv_figures = create_convergence_analysis(iterations, convergence_history)
                        conv_figures.savefig(figures_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
                        plt.close(conv_figures)
                        
                        logger.info(f"Performance metrics - RMSE: {metrics.rmse:.3f}%, R²: {metrics.r_squared:.3f}")
                else:
                    # Only convergence analysis if no true parameters
                    if em_results is not None:
                        convergence_history = [stats.log_likelihood for stats in em_results.statistics_history]
                        iterations = list(range(1, len(convergence_history) + 1))
                        conv_figures = create_convergence_analysis(iterations, convergence_history)
                        conv_figures.savefig(figures_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
                        plt.close(conv_figures)
                
                # 4. Sequence analysis
                logger.info("[3/4] Creating sequence analysis")
                seq_analyzer = SequenceAnalyzer(gen_cfg.alphabet)
                length_analysis = analyze_sequence_lengths(sequences)
                usage_analysis = analyze_symbol_usage(sequences, gen_cfg.alphabet)
                
                # Create sequence length plot
                length_fig = create_sequence_length_plot(length_analysis)
                length_fig.savefig(figures_dir / "sequence_length_analysis.png", dpi=300, bbox_inches='tight')
                plt.close(length_fig)
                
                # Create symbol usage plot
                usage_fig = create_symbol_usage_plot(usage_analysis, gen_cfg.alphabet)
                usage_fig.savefig(figures_dir / "symbol_usage_analysis.png", dpi=300, bbox_inches='tight')
                plt.close(usage_fig)
                
                # 5. Publication figures
                logger.info("[3/4] Creating publication figures")
                pub_manager = PublicationFigureManager()
                pub_figures = pub_manager.generate_complete_figure_set(
                    results_dict, gen_cfg.alphabet, save_dir=figures_dir
                )
                
                stage_success["viz"] = True
        except Exception as exc:
            logger.exception("Visualization failed: %s", exc)

        ################################
        # Stage 4 – Archiving          #
        ################################
        try:
            if any(stage_success.values()):
                for path in tmp_dir.rglob("*"):
                    if path.is_file():
                        stash_artifact(
                            path,
                            experiment_id=experiment_id,
                            stage="figures" if path.suffix in {".png", ".pdf"} else "artifacts",
                            tag=None,
                            overwrite=True,
                        )
                # Also stash log file
                stash_artifact(log_path, experiment_id=experiment_id, stage="logs", overwrite=True)
                stage_success["archive"] = True
        except Exception as exc:
            logger.exception("Archiving failed: %s", exc)

        logger.info("🏁 Pipeline finished – success matrix: %s", stage_success)
        overall_ok = stage_success["data"] and stage_success["em"]

    # Temporary directory cleaned up automatically
    return overall_ok


def evaluate_results(results_dir: str | Path) -> bool:  # noqa: D401
    """Evaluate outputs located in *results_dir*.

    This stub merely checks that the directory exists. Replace with real model
    evaluation once metrics are defined.
    """

    path = Path(results_dir)
    if not path.exists():
        logger.error("Results directory not found: %s", path)
        return False

    logger.info("[Pipeline] Evaluating results in %s", path)
    logger.warning(
        "evaluate_results is currently a stub – implement metrics and report generation later"
    )

    return True


def create_numerical_diagnostics(em_results, save_dir):
    import matplotlib.pyplot as plt
    stats = em_results.statistics_history
    iterations = [s.iteration for s in stats]
    log_likelihoods = [s.log_likelihood for s in stats]
    log_likelihood_changes = [s.log_likelihood_change for s in stats]
    parameter_changes = [s.parameter_change for s in stats]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Plot 1: Log-likelihood evolution
    ax1 = axes[0, 0]
    ax1.plot(iterations, log_likelihoods, 'b-o', linewidth=2, markersize=4)
    ax1.axvline(x=em_results.best_iteration, color='r', linestyle='--', label=f'Best iteration ({em_results.best_iteration})')
    ax1.set_xlabel('EM Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Log-Likelihood Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Plot 2: Log-likelihood changes
    ax2 = axes[0, 1]
    ax2.plot(iterations[1:], log_likelihood_changes[1:], 'r-o', linewidth=2, markersize=4)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.set_xlabel('EM Iteration')
    ax2.set_ylabel('Log-Likelihood Change')
    ax2.set_title('Log-Likelihood Changes')
    ax2.grid(True, alpha=0.3)
    # Plot 3: Parameter changes
    ax3 = axes[1, 0]
    ax3.plot(iterations[1:], parameter_changes[1:], 'g-o', linewidth=2, markersize=4)
    ax3.set_xlabel('EM Iteration')
    ax3.set_ylabel('Parameter Change Norm')
    ax3.set_title('Parameter Change Evolution')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    # Plot 4: Combined view
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    ll_normalized = (np.array(log_likelihoods) - np.min(log_likelihoods)) / (np.max(log_likelihoods) - np.min(log_likelihoods) + 1e-12)
    line1 = ax4.plot(iterations, ll_normalized, 'b-', linewidth=2, label='Log-Likelihood (normalized)')
    line2 = ax4_twin.plot(iterations[1:], parameter_changes[1:], 'g-', linewidth=2, label='Parameter Change')
    ax4.set_xlabel('EM Iteration')
    ax4.set_ylabel('Normalized Log-Likelihood', color='b')
    ax4_twin.set_ylabel('Parameter Change Norm', color='g')
    ax4.set_title('Combined Evolution')
    ax4.grid(True, alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_dir / 'numerical_instability_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'numerical_instability_diagnostics.pdf', bbox_inches='tight')
    plt.savefig(save_dir / 'numerical_instability_diagnostics.svg', bbox_inches='tight')
    plt.close() 