#!/usr/bin/env python3
"""Aggregate Statistics Analysis for Adaptive Syntax Filter.

This script performs comprehensive statistical analysis across many runs with different
random seeds to understand model behavior, convergence patterns, and estimate quality.

Implements the workplan from aggregate_analysis_workplan.md with:
- 4 experimental configurations (linear/sigmoid evolution, 1st/2nd order)
- Multiple estimation methods (EM, raw estimates, smoothed raw estimates)
- Cross-entropy analysis between estimates and true probabilities
- Statistical significance testing
- Comprehensive visualization and data storage

Usage:
    python aggregate_analysis.py --config-id 1 --n-seeds 100 --test-mode
    python aggregate_analysis.py --config-id all --n-seeds 20000 --parallel 8
"""

import argparse
import logging
import multiprocessing as mp
import pickle
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Local imports
from config_cli import _read_config as load_config
from research_pipeline import run_pipeline
from src.adaptive_syntax_filter.data.sequence_generator import (
    GenerationConfig, SequenceGenerator, sequences_to_observations, generate_dataset
)
from src.adaptive_syntax_filter.core.state_space import StateSpaceManager
from src.adaptive_syntax_filter.core.em_algorithm import EMAlgorithm, EMParameters, EMResults
from src.adaptive_syntax_filter.data.temporal_evolution import compute_evolution_trajectory


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aggregate_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AggregateConfig:
    """Configuration for aggregate analysis experiments."""
    config_id: int
    config_name: str
    config_file: str
    n_seeds: int
    description: str
    
    # Derived from config file
    alphabet: List[str] = field(default_factory=list)
    order: int = 1
    n_sequences: int = 50
    max_length: int = 100
    evolution_type: str = "linear"
    max_em_iterations: int = 300


@dataclass 
class SingleRunResults:
    """Results from a single seed run."""
    seed: int
    config_id: int
    
    # EM Algorithm Results
    em_results: EMResults
    convergence_iteration: int
    final_log_likelihood: float
    best_log_likelihood: float
    runtime_seconds: float
    
    # True probabilities (ground truth)
    true_probabilities: np.ndarray  # shape: (n_timesteps, n_transitions)
    true_logits: np.ndarray
    
    # Generated sequences
    sequences: List[List[str]]
    
    # Cross-entropy measures
    em_cross_entropy: np.ndarray  # shape: (n_timesteps,)
    raw_cross_entropy: np.ndarray
    smoothed_raw_cross_entropy: np.ndarray
    
    # Probability estimates
    raw_probabilities: np.ndarray  # shape: (n_timesteps, n_transitions)
    smoothed_raw_probabilities: np.ndarray
    em_probabilities: np.ndarray


class AggregateAnalysisManager:
    """Main manager for aggregate statistical analysis."""
    
    def __init__(self, output_dir: str = "results/aggregate_analysis"):
        """Initialize the aggregate analysis manager.
        
        Parameters
        ----------
        output_dir : str
            Base directory for storing results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure experimental conditions
        self.experimental_configs = {
            1: AggregateConfig(
                config_id=1,
                config_name="linear_1st_short", 
                config_file="configs/aggregate_config_1_linear_1st_short.yml",
                n_seeds=20000,
                description="Linear evolution, 1st order, short sequences (6 letters)"
            ),
            2: AggregateConfig(
                config_id=2,
                config_name="linear_2nd_long",
                config_file="configs/aggregate_config_2_linear_2nd_long.yml", 
                n_seeds=50000,
                description="Linear evolution, 2nd order, long sequences (5 letters)"
            ),
            3: AggregateConfig(
                config_id=3,
                config_name="sigmoid_1st_medium",
                config_file="configs/aggregate_config_3_sigmoid_1st_medium.yml",
                n_seeds=50000, 
                description="Sigmoid evolution, 1st order, medium sequences (6 letters)"
            ),
            4: AggregateConfig(
                config_id=4,
                config_name="sigmoid_2nd_medium",
                config_file="configs/aggregate_config_4_sigmoid_2nd_medium.yml",
                n_seeds=20000,
                description="Sigmoid evolution, 2nd order, medium sequences (6 letters)"
            )
        }
        
        # Load configuration details
        self._load_config_details()
        
        # Setup storage paths
        self._setup_storage_structure()
        
    def _load_config_details(self):
        """Load detailed configuration from YAML files."""
        for config_id, agg_config in self.experimental_configs.items():
            try:
                config_data = load_config(Path(agg_config.config_file))
                
                # Extract data section
                data_config = config_data.get('data', {})
                agg_config.alphabet = data_config.get('alphabet', ['<', 'a', '>'])
                agg_config.order = data_config.get('order', 1)
                agg_config.n_sequences = data_config.get('n_sequences', 50)
                agg_config.max_length = data_config.get('max_length', 100)
                agg_config.evolution_type = data_config.get('evolution_type', 'linear')
                
                # Extract EM section
                em_config = config_data.get('em', {})
                agg_config.max_em_iterations = em_config.get('max_iterations', 300)
                
                logger.info(f"Loaded config {config_id}: {agg_config.description}")
                
            except Exception as e:
                logger.error(f"Failed to load config {config_id}: {e}")
                raise
                
    def _load_single_config(self, config_id: int):
        """Load only a single configuration for efficiency in parallel processing."""
        if config_id not in self.experimental_configs:
            raise ValueError(f"Config {config_id} not found")
            
        agg_config = self.experimental_configs[config_id]
        
        try:
            config_data = load_config(Path(agg_config.config_file))
            
            # Extract data section
            data_config = config_data.get('data', {})
            agg_config.alphabet = data_config.get('alphabet', ['<', 'a', '>'])
            agg_config.order = data_config.get('order', 1)
            agg_config.n_sequences = data_config.get('n_sequences', 50)
            agg_config.max_length = data_config.get('max_length', 100)
            agg_config.evolution_type = data_config.get('evolution_type', 'linear')
            
            # Extract EM section
            em_config = config_data.get('em', {})
            agg_config.max_em_iterations = em_config.get('max_iterations', 300)
            
        except Exception as e:
            logger.error(f"Failed to load config {config_id}: {e}")
            raise
                
    def _setup_storage_structure(self):
        """Create directory structure for storing results."""
        for config_id, agg_config in self.experimental_configs.items():
            config_dir = self.output_dir / f"config_{config_id}_{agg_config.config_name}"
            config_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (config_dir / "example_plots").mkdir(exist_ok=True)
            
        # Summary plots directory
        (self.output_dir / "summary_plots").mkdir(exist_ok=True)
        
    def run_single_experiment(self, config_id: int, seed: int) -> SingleRunResults:
        """Run a single experiment with given config and seed.
        
        Parameters
        ----------
        config_id : int
            Configuration ID (1-4)
        seed : int
            Random seed for this run
            
        Returns
        -------
        SingleRunResults
            Complete results from this single run
        """
        agg_config = self.experimental_configs[config_id]
        
        try:
            start_time = time.time()
            
            # Set up generation config
            generation_config = GenerationConfig(
                alphabet=agg_config.alphabet,
                order=agg_config.order,
                n_sequences=agg_config.n_sequences,
                max_length=agg_config.max_length,
                evolution_type=agg_config.evolution_type,
                seed=seed
            )
            
            # Generate dataset
            sequences, true_logits = generate_dataset(generation_config)
            
            # Convert to probabilities for cross-entropy calculation  
            from src.adaptive_syntax_filter.data.sequence_generator import softmax_mc_higher_order
            
            # true_logits has shape (state_size, n_sequences), we need (n_sequences, state_size)
            true_logits = true_logits.T  # Transpose to get (n_sequences, state_size)
            n_timesteps = true_logits.shape[0]
            
            true_probabilities = np.zeros_like(true_logits)
            for t in range(n_timesteps):
                true_probabilities[t] = softmax_mc_higher_order(
                    true_logits[t], agg_config.alphabet, agg_config.order
                )
            
            # Convert sequences to observations format
            observations = sequences_to_observations(sequences, agg_config.alphabet)
            
            # Set up EM algorithm
            state_manager = StateSpaceManager(
                alphabet_size=len(agg_config.alphabet),
                markov_order=agg_config.order
            )
            
            em_algorithm = EMAlgorithm(
                state_space_manager=state_manager,
                max_iterations=agg_config.max_em_iterations,
                tolerance=1e-4,
                regularization_lambda=1e-3,
                damping_factor=0.5,
                adaptive_damping=True,
                verbose=False
            )
            
            # Fit EM algorithm
            em_results = em_algorithm.fit(observations)
            
            # Calculate EM probabilities from best parameters
            em_probabilities = self._extract_em_probabilities(
                em_results.best_params, agg_config, n_timesteps
            )
            
            # Calculate raw estimates from sequences
            raw_probabilities = self._calculate_raw_estimates(
                sequences, agg_config, n_timesteps
            )
            
            # Calculate smoothed raw estimates
            smoothed_raw_probabilities = self._calculate_smoothed_raw_estimates(
                raw_probabilities, window_size=5
            )
            
            # Calculate cross-entropies
            em_cross_entropy = self._calculate_cross_entropy(
                true_probabilities, em_probabilities
            )
            raw_cross_entropy = self._calculate_cross_entropy(
                true_probabilities, raw_probabilities
            )
            smoothed_raw_cross_entropy = self._calculate_cross_entropy(
                true_probabilities, smoothed_raw_probabilities
            )
            
            runtime = time.time() - start_time
            
            # Create results object
            results = SingleRunResults(
                seed=seed,
                config_id=config_id,
                em_results=em_results,
                convergence_iteration=em_results.best_iteration,
                final_log_likelihood=em_results.statistics_history[-1].log_likelihood if em_results.statistics_history else -np.inf,
                best_log_likelihood=em_results.best_log_likelihood,
                runtime_seconds=runtime,
                true_probabilities=true_probabilities,
                true_logits=true_logits,
                sequences=sequences,
                em_cross_entropy=em_cross_entropy,
                raw_cross_entropy=raw_cross_entropy,
                smoothed_raw_cross_entropy=smoothed_raw_cross_entropy,
                raw_probabilities=raw_probabilities,
                smoothed_raw_probabilities=smoothed_raw_probabilities,
                em_probabilities=em_probabilities
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed experiment config_id={config_id}, seed={seed}: {e}")
            raise
    
    def _extract_em_probabilities(self, em_params: EMParameters, 
                                 agg_config: AggregateConfig, 
                                 n_timesteps: int) -> np.ndarray:
        """Extract probability trajectories from EM parameters using state evolution."""
        alphabet_size = len(agg_config.alphabet)
        n_transitions = alphabet_size ** (agg_config.order + 1)
        
        # Get state evolution trajectory by evolving the initial state
        state_trajectory = np.zeros((n_timesteps, len(em_params.x0)))
        
        # Initialize with x0
        state_trajectory[0] = em_params.x0.copy()
        
        # Evolve state using F matrix and u vector
        for t in range(1, n_timesteps):
            state_trajectory[t] = (em_params.F @ state_trajectory[t-1] + em_params.u)
        
        # Convert states to probabilities using softmax
        from src.adaptive_syntax_filter.data.sequence_generator import softmax_mc_higher_order
        
        em_probabilities = np.zeros((n_timesteps, n_transitions))
        for t in range(n_timesteps):
            # Extract logits from state - need to be careful about state structure
            # For now, assume the state directly represents logits (this may need adjustment)
            if len(state_trajectory[t]) == n_transitions:
                logits = state_trajectory[t]
            else:
                # If dimensions don't match, take first n_transitions elements
                # This is a simplification that may need refinement based on actual state structure
                logits = state_trajectory[t][:n_transitions]
                if len(logits) < n_transitions:
                    # Pad with zeros if needed
                    padded_logits = np.zeros(n_transitions)
                    padded_logits[:len(logits)] = logits
                    logits = padded_logits
            
            try:
                em_probabilities[t] = softmax_mc_higher_order(
                    logits, agg_config.alphabet, agg_config.order
                )
            except Exception as e:
                # If softmax fails, use uniform probabilities
                logger.warning(f"Softmax failed for timestep {t}, using uniform: {e}")
                em_probabilities[t] = np.ones(n_transitions) / n_transitions
        
        return em_probabilities
    
    def _calculate_raw_estimates(self, sequences: List[List[str]], 
                               agg_config: AggregateConfig,
                               n_timesteps: int) -> np.ndarray:
        """Calculate raw transition probability estimates from sequences.
        
        Each 'timestep' corresponds to one sequence, and we calculate 
        transition probabilities within that sequence.
        """
        alphabet = agg_config.alphabet
        order = agg_config.order
        alphabet_size = len(alphabet)
        
        # Create symbol to index mapping
        symbol_to_idx = {symbol: i for i, symbol in enumerate(alphabet)}
        
        n_transitions = alphabet_size ** (order + 1)
        raw_estimates = np.zeros((n_timesteps, n_transitions))
        
        # n_timesteps should equal len(sequences) 
        assert n_timesteps == len(sequences), f"Expected {len(sequences)} timesteps, got {n_timesteps}"
        
        for seq_idx in range(min(n_timesteps, len(sequences))):
            sequence = sequences[seq_idx]
            
            # Count transitions within this specific sequence
            transition_counts = np.zeros(n_transitions)
            
            # Go through all valid positions in this sequence
            for pos in range(len(sequence) - order):
                # Get context and next symbol
                if order == 1:
                    context = [sequence[pos]]
                else:
                    context = sequence[pos:pos+order]
                next_symbol = sequence[pos + order]
                
                # Convert to transition index
                try:
                    context_indices = [symbol_to_idx[s] for s in context]
                    next_idx = symbol_to_idx[next_symbol]
                    
                    # Calculate flat transition index
                    transition_idx = 0
                    for i, ctx_idx in enumerate(context_indices):
                        transition_idx += ctx_idx * (alphabet_size ** (order - i - 1))
                    transition_idx = transition_idx * alphabet_size + next_idx
                    
                    transition_counts[transition_idx] += 1
                    
                except (KeyError, IndexError):
                    # Skip invalid transitions
                    continue
            
            # Normalize to probabilities within each context for this sequence
            for context_start in range(0, n_transitions, alphabet_size):
                context_end = context_start + alphabet_size
                context_counts = transition_counts[context_start:context_end]
                
                if context_counts.sum() > 0:
                    raw_estimates[seq_idx, context_start:context_end] = (
                        context_counts / context_counts.sum()
                    )
                else:
                    # Uniform if no data
                    raw_estimates[seq_idx, context_start:context_end] = 1.0 / alphabet_size
        
        return raw_estimates
    
    def _calculate_smoothed_raw_estimates(self, raw_probabilities: np.ndarray,
                                        window_size: int = 5) -> np.ndarray:
        """Apply smoothing window to raw probability estimates."""
        n_timesteps, n_transitions = raw_probabilities.shape
        smoothed = np.zeros_like(raw_probabilities)
        
        # Apply moving average with window_size
        half_window = window_size // 2
        
        for t in range(n_timesteps):
            start_idx = max(0, t - half_window)
            end_idx = min(n_timesteps, t + half_window + 1)
            
            smoothed[t] = np.mean(raw_probabilities[start_idx:end_idx], axis=0)
        
        return smoothed
    
    def _calculate_cross_entropy(self, true_probs: np.ndarray, 
                               estimated_probs: np.ndarray) -> np.ndarray:
        """Calculate cross-entropy between true and estimated probabilities."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        estimated_probs_safe = np.clip(estimated_probs, epsilon, 1.0)
        
        # Calculate cross-entropy for each timestep
        cross_entropy = -np.sum(true_probs * np.log(estimated_probs_safe), axis=1)
        
        return cross_entropy
    
    def run_sequential_batch(self, config_id: int, n_seeds: int) -> List[SingleRunResults]:
        """Run experiments sequentially for a given configuration.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        n_seeds : int
            Number of seeds to run
            
        Returns
        -------
        List[SingleRunResults]
            Results from successful runs
        """
        from aggregate_analysis_utils import HDF5DataManager
        
        # Set up HDF5 data manager for immediate saving
        data_manager = HDF5DataManager(self.output_dir)
        completed_seeds = []
        seeds = list(range(n_seeds))
        
        logger.info(f"Running {n_seeds} experiments sequentially for config {config_id}")
        logger.info(f"HDF5 files will be saved immediately after each seed completes")
        
        for i, seed in enumerate(tqdm(seeds, desc=f"Config {config_id}")):
            try:
                result = self.run_single_experiment(config_id, seed)
                
                # Save HDF5 file immediately
                try:
                    data_manager.save_single_run_results(result, config_id, result.seed)
                    completed_seeds.append(seed)
                    logger.debug(f"Saved HDF5 file for seed {seed}")
                except Exception as e:
                    logger.error(f"Failed to save HDF5 for seed {seed}: {e}")
                    continue
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i+1}/{n_seeds} experiments, {len(completed_seeds)} saved")
                    
            except Exception as e:
                logger.warning(f"Failed seed {seed} for config {config_id}: {e}")
                continue
        
        # Load completed results from HDF5 files for return
        if completed_seeds:
            logger.info(f"Loading {len(completed_seeds)} completed results from HDF5 files...")
            all_results = self._load_existing_results(config_id, completed_seeds)
            logger.info(f"Loaded {len(all_results)} results for aggregate analysis")
            return all_results
        else:
            return []
    
    def run_parallel_batch(self, config_id: int, n_seeds: int, n_processes: int = 4,
                          resume: bool = False) -> List[SingleRunResults]:
        """Run experiments in parallel for a given configuration.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        n_seeds : int
            Number of seeds to run
        n_processes : int
            Number of parallel processes
        resume : bool
            Whether to resume from checkpoint
            
        Returns
        -------
        List[SingleRunResults]
            Results from successful runs
        """
        from aggregate_analysis_utils import CheckpointManager
        
        # Set up checkpointing with HDF5 verification
        checkpoint_manager = CheckpointManager(
            self.output_dir / "checkpoints", 
            base_data_dir=self.output_dir
        )
        
        seeds = list(range(n_seeds))
        
        if resume:
            # Get remaining seeds from checkpoint
            remaining_seeds = checkpoint_manager.get_remaining_seeds(config_id, seeds)
            logger.info(f"Resuming: {len(remaining_seeds)} seeds remaining out of {n_seeds}")
        else:
            remaining_seeds = seeds
        
        if not remaining_seeds:
            logger.info(f"All seeds already completed for config {config_id}")
            # Load existing results
            return self._load_existing_results(config_id, seeds)
        
        results = []
        completed_seeds = []
        
        # Set up HDF5 data manager for immediate saving
        from aggregate_analysis_utils import HDF5DataManager
        data_manager = HDF5DataManager(self.output_dir)
        
        logger.info(f"Running {len(remaining_seeds)} experiments with {n_processes} processes for config {config_id}")
        logger.info(f"HDF5 files will be saved immediately after each seed completes")
        
        # Use multiprocessing to run experiments in parallel
        with mp.Pool(processes=n_processes) as pool:
            try:
                # Create argument tuples for parallel execution
                args_list = [(config_id, seed) for seed in remaining_seeds]
                
                # Use imap for better progress tracking
                with tqdm(total=len(remaining_seeds), desc=f"Config {config_id}") as pbar:
                    for result in pool.imap(self._run_single_experiment_wrapper, args_list):
                        if result is not None:
                            # CRITICAL FIX: Save HDF5 file immediately!
                            try:
                                data_manager.save_single_run_results(result, config_id, result.seed)
                                completed_seeds.append(result.seed)
                                logger.debug(f"Saved HDF5 file for seed {result.seed}")
                            except Exception as e:
                                logger.error(f"Failed to save HDF5 for seed {result.seed}: {e}")
                                # Don't add to completed_seeds if saving failed
                                continue
                        
                        pbar.update(1)
                        
                        # Save checkpoint every 50 completed experiments
                        if len(completed_seeds) % 50 == 0:
                            checkpoint_manager.save_progress(
                                config_id, completed_seeds, n_seeds
                            )
                            logger.info(f"Checkpoint saved: {len(completed_seeds)} completed, "
                                       f"{len(completed_seeds)} HDF5 files saved")
                            
            except KeyboardInterrupt:
                logger.warning("Interrupted by user. Saving progress...")
                pool.terminate()
                pool.join()
                
                # Save final checkpoint with completed seeds
                if completed_seeds:
                    checkpoint_manager.save_progress(config_id, completed_seeds, n_seeds)
                    logger.info(f"Final checkpoint saved: {len(completed_seeds)} seeds completed and saved")
                
        # Final checkpoint save
        if completed_seeds:
            checkpoint_manager.save_progress(config_id, completed_seeds, n_seeds)
        
        logger.info(f"Parallel batch completed: {len(completed_seeds)} successful out of {len(remaining_seeds)} attempted")
        logger.info(f"All HDF5 files saved immediately during processing")
        
        # Load the completed results from HDF5 files for aggregate analysis
        # Note: results list is empty since we save immediately, so load from files
        if completed_seeds:
            logger.info(f"Loading {len(completed_seeds)} completed results from HDF5 files...")
            all_results = self._load_existing_results(config_id, completed_seeds)
            logger.info(f"Loaded {len(all_results)} results for aggregate analysis")
            return all_results
        else:
            return []
    
    @staticmethod
    def _run_single_experiment_wrapper(args: Tuple[int, int]) -> Optional[SingleRunResults]:
        """Wrapper function for multiprocessing.
        
        Parameters
        ----------
        args : Tuple[int, int]
            (config_id, seed) tuple
            
        Returns
        -------
        Optional[SingleRunResults]
            Results if successful, None if failed
        """
        config_id, seed = args
        
        try:
            # Create a new manager instance for this process
            # Only load the specific config needed to avoid overhead
            manager = AggregateAnalysisManager()
            # Pre-load only the needed config to avoid loading all 4 configs
            manager._load_single_config(config_id)
            return manager.run_single_experiment(config_id, seed)
            
        except Exception as e:
            logger.error(f"Failed experiment config_id={config_id}, seed={seed}: {e}")
            return None
    
    def _load_existing_results(self, config_id: int, seeds: List[int]) -> List[SingleRunResults]:
        """Load existing results from saved data.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        seeds : List[int]
            List of seeds to load
            
        Returns
        -------
        List[SingleRunResults]
            Loaded results
        """
        from aggregate_analysis_utils import HDF5DataManager
        
        data_dir = self.output_dir / f"config_{config_id}_data"
        if not data_dir.exists():
            logger.warning(f"Data directory {data_dir} does not exist")
            return []
        
        data_manager = HDF5DataManager(self.output_dir)
        results = []
        
        for seed in seeds:
            try:
                result_data = data_manager.load_single_run_results(config_id, seed)
                if result_data:
                    # Create a mock EMResults object since it's not saved in HDF5
                    from src.adaptive_syntax_filter.core.em_algorithm import EMResults, EMParameters
                    
                    # Create a minimal EMParameters object
                    mock_params = EMParameters(
                        x0=np.array([]),
                        Sigma=np.array([]),
                        F=np.array([]),
                        u=np.array([])
                    )
                    
                    em_results = EMResults(
                        final_params=mock_params,
                        best_params=mock_params,
                        best_iteration=result_data.get('convergence_iteration', 0),
                        best_log_likelihood=result_data.get('best_log_likelihood', result_data['final_log_likelihood']),
                        statistics_history=[]
                    )
                    
                    # Create SingleRunResults object from loaded data
                    result = SingleRunResults(
                        seed=seed,
                        config_id=config_id,
                        true_probabilities=result_data['true_probabilities'],
                        em_probabilities=result_data['em_probabilities'],
                        raw_probabilities=result_data['raw_probabilities'],
                        smoothed_raw_probabilities=result_data['smoothed_raw_probabilities'],
                        em_cross_entropy=result_data['em_cross_entropy'],
                        raw_cross_entropy=result_data['raw_cross_entropy'],
                        smoothed_raw_cross_entropy=result_data['smoothed_raw_cross_entropy'],
                        final_log_likelihood=result_data['final_log_likelihood'],
                        runtime_seconds=result_data['runtime_seconds'],
                        em_results=em_results,
                        convergence_iteration=result_data.get('convergence_iteration', 0),
                        best_log_likelihood=result_data.get('best_log_likelihood', result_data['final_log_likelihood']),
                        true_logits=result_data.get('true_logits', np.array([])),
                        sequences=result_data.get('sequences', [])
                    )
                    results.append(result)
                    logger.debug(f"Loaded results for seed {seed}")
            except Exception as e:
                logger.warning(f"Failed to load results for seed {seed}: {e}")
        
        logger.info(f"Loaded {len(results)} results from {len(seeds)} requested seeds")
        return results
    
    def save_batch_results(self, config_id: int, results: List[SingleRunResults]):
        """Save batch results to storage.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        results : List[SingleRunResults]
            Results to save
        """
        from aggregate_analysis_utils import HDF5DataManager
        
        data_manager = HDF5DataManager(self.output_dir)
        
        logger.info(f"Saving {len(results)} results for config {config_id}")
        
        for result in tqdm(results, desc="Saving results"):
            try:
                data_manager.save_single_run_results(result, config_id, result.seed)
            except Exception as e:
                logger.error(f"Failed to save result for seed {result.seed}: {e}")
                continue
        
        logger.info(f"Results saved for config {config_id}")
    
    def compute_and_save_aggregates(self, config_id: int, results: List[SingleRunResults]):
        """Compute and save aggregate statistics.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        results : List[SingleRunResults]
            Individual run results
        """
        from aggregate_analysis_utils import StatisticalAnalyzer, HDF5DataManager
        import datetime
        
        logger.info(f"Computing aggregate statistics for config {config_id}")
        
        # Extract cross-entropy data
        cross_entropy_data = {
            'em_cross_entropy': np.array([r.em_cross_entropy for r in results]),
            'raw_cross_entropy': np.array([r.raw_cross_entropy for r in results]),
            'smoothed_raw_cross_entropy': np.array([r.smoothed_raw_cross_entropy for r in results])
        }
        
        # Compute statistics
        analyzer = StatisticalAnalyzer()
        
        ce_statistics = analyzer.compute_cross_entropy_statistics(cross_entropy_data)
        significance_tests = analyzer.perform_significance_tests(cross_entropy_data)
        
        # Extract EM data for convergence analysis
        em_data = []
        for result in results:
            em_data.append({
                'convergence_iteration': result.convergence_iteration,
                'final_log_likelihood': result.final_log_likelihood,
                'best_log_likelihood': result.best_log_likelihood,
                'runtime_seconds': result.runtime_seconds
            })
        
        convergence_stats = analyzer.compute_convergence_statistics(em_data)
        
        # Prepare summary data
        summary_data = {
            'n_runs': len(results),
            'timestamp': datetime.datetime.now().isoformat(),
            'cross_entropy_statistics': ce_statistics,
            'convergence_statistics': convergence_stats,
            'em_cross_entropy_all': cross_entropy_data['em_cross_entropy'],
            'raw_cross_entropy_all': cross_entropy_data['raw_cross_entropy'],
            'smoothed_raw_cross_entropy_all': cross_entropy_data['smoothed_raw_cross_entropy']
        }
        
        # Save aggregate data
        data_manager = HDF5DataManager(self.output_dir)
        data_manager.save_aggregate_summary(config_id, summary_data)
        
        # Save significance tests to CSV
        config_dir = self.output_dir / f"config_{config_id}_{self.experimental_configs[config_id].config_name}"
        significance_tests.to_csv(config_dir / "statistical_tests.csv", index=False)
        
        logger.info(f"Aggregate statistics saved for config {config_id}")
    
    def create_example_plots(self, config_id: int, results: List[SingleRunResults]):
        """Create example plots for a few representative runs.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        results : List[SingleRunResults]
            Results to create plots for
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from src.adaptive_syntax_filter.viz.performance_assessment import create_convergence_analysis
        from src.adaptive_syntax_filter.viz import setup_publication_style
        
        logger.info(f"Creating example plots for config {config_id}")
        
        agg_config = self.experimental_configs[config_id]
        config_dir = self.output_dir / f"config_{config_id}_{agg_config.config_name}"
        plots_dir = config_dir / "example_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set up publication style
        try:
            setup_publication_style()
        except:
            plt.style.use('seaborn-v0_8-paper')
        
        # Create example plots for first 5 seeds
        for i, result in enumerate(results[:5]):
            self._create_single_run_plot(result, agg_config, plots_dir)
            
            # Create convergence plot
            if result.em_results.statistics_history:
                self._create_convergence_plot(result, plots_dir)
        
        # Create aggregate cross-entropy comparison plot
        self._create_aggregate_cross_entropy_plot(config_id, results)
        
        logger.info(f"Example plots created for config {config_id}")
    
    def _create_single_run_plot(self, result: SingleRunResults, agg_config: AggregateConfig, plots_dir: Path):
        """Create probability evolution plot for a single run using research pipeline style."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
        import matplotlib.pyplot as plt
        from src.adaptive_syntax_filter.viz.logit_evolution import LogitEvolutionDashboard, LogitVisualizationConfig
        
        seed = result.seed
        n_timesteps = result.true_probabilities.shape[0]
        alphabet_size = len(agg_config.alphabet)
        n_contexts = alphabet_size ** agg_config.order
        
        # Create dashboard using research pipeline style
        config = LogitVisualizationConfig(
            figure_size=(15, 10),
            dpi=300,
            font_size=10,
            line_width=2.0,
            alpha_estimated=1.0,
            alpha_true=0.8,
            color_palette='Set2',
            show_legend=True
        )
        
        dashboard = LogitEvolutionDashboard(agg_config.alphabet, agg_config.order, config)
        
        # Create figure and setup layout like research pipeline
        fig = plt.figure(figsize=config.figure_size, dpi=config.dpi)
        
        # Calculate subplot layout  
        rows, cols = dashboard._calculate_subplot_layout(n_contexts)
        
        # Sequence indices for x-axis
        sequence_indices = np.arange(n_timesteps)
        
        # Plot each context block exactly like research pipeline
        for context_idx in range(n_contexts):
            ax = fig.add_subplot(rows, cols, context_idx + 1)
            
            # Calculate block indices
            start_idx = context_idx * alphabet_size
            end_idx = (context_idx + 1) * alphabet_size
            
            # Plot each transition probability
            for target_idx in range(alphabet_size):
                pos_idx = start_idx + target_idx
                
                if pos_idx >= result.true_probabilities.shape[1]:
                    continue
                
                # Get color for this target symbol
                color = dashboard.colors[target_idx]
                target_symbol = agg_config.alphabet[target_idx]
                
                # Extract probability data for this transition
                true_probs = result.true_probabilities[:, pos_idx]
                em_probs = result.em_probabilities[:, pos_idx]
                raw_probs = result.raw_probabilities[:, pos_idx]
                smoothed_raw_probs = result.smoothed_raw_probabilities[:, pos_idx]
                
                # Skip forbidden transitions (probability always 0)
                if np.allclose(true_probs, 0.0):
                    continue
                
                # True probabilities (solid line, research pipeline style)
                ax.plot(sequence_indices, true_probs, 
                       '-',
                       color=color,
                       linewidth=config.line_width * 0.75,
                       alpha=config.alpha_true,
                       label=f"True → {target_symbol}")
                
                # EM estimated probabilities (dashed line, research pipeline style)
                ax.plot(sequence_indices, em_probs, 
                       '--',
                       color=color,
                       linewidth=config.line_width,
                       alpha=config.alpha_estimated,
                       label=f"Est → {target_symbol}")
                
                # Smoothed raw estimates (dash-dot line - new style, different from solid/dashed)
                ax.plot(sequence_indices, smoothed_raw_probs, 
                       '-.',
                       color=color,
                       linewidth=config.line_width * 0.9,
                       alpha=0.25,
                       label=f"Smooth → {target_symbol}")
            
            # Customize subplot exactly like research pipeline
            context_label = dashboard.context_labels[context_idx]
            
            # Adjust font sizes based on number of subplots
            title_font_size = max(6, config.font_size - max(0, n_contexts // 10))
            label_font_size = max(6, config.font_size - max(0, n_contexts // 8))
            tick_font_size = max(5, config.font_size - max(0, n_contexts // 6))
            
            ax.set_title(f"From {context_label}", fontsize=title_font_size, pad=5)
            
            # Only show x-axis labels on bottom row
            if context_idx >= (rows-1)*cols:
                ax.set_xlabel('Song #', fontsize=label_font_size)
            else:
                ax.set_xticklabels([])
            
            # Only show y-axis labels on left column
            if context_idx % cols == 0:
                ax.set_ylabel('Probability', fontsize=label_font_size)
            else:
                ax.set_yticklabels([])
            
            # Reduce tick density for large state spaces
            if n_contexts > 16:
                ax.locator_params(nbins=3)
            
            ax.tick_params(labelsize=tick_font_size)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            # Create comprehensive legend (research pipeline style + new lines)
            if config.show_legend and n_contexts <= 16:
                if context_idx == n_contexts - 1:
                    from matplotlib.lines import Line2D
                    custom_handles = []
                    custom_labels = []
                    
                    # Add line style legend
                    custom_handles.append(Line2D([0], [0], linestyle='-', color='black', 
                                               linewidth=config.line_width * 0.75))
                    custom_labels.append('True')
                    
                    custom_handles.append(Line2D([0], [0], linestyle='--', color='black', 
                                               linewidth=config.line_width))
                    custom_labels.append('EM Estimated')
                    
                    custom_handles.append(Line2D([0], [0], linestyle='-.', color='black', 
                                               linewidth=config.line_width * 0.9))
                    custom_labels.append('Smoothed Raw')
                    
                    # Add color legend for target transitions
                    for target_idx in range(alphabet_size):
                        context_color = dashboard.colors[target_idx]
                        target_symbol = agg_config.alphabet[target_idx]
                        custom_handles.append(Line2D([0], [0], linestyle='-', color=context_color, 
                                                   linewidth=config.line_width))
                        custom_labels.append(f'→ {target_symbol}')
                    
                    ax.legend(custom_handles, custom_labels, 
                             bbox_to_anchor=(1.05, 1), loc='upper left', 
                             fontsize=max(5, config.font_size - 2))
        
        # Overall figure formatting (research pipeline style)
        font_size = config.font_size + 4
        fig.suptitle(f'Probability Evolution - Config {agg_config.config_name.replace("_", " ").title()}\n'
                    f'Seed {seed} (Order {agg_config.order}, {agg_config.evolution_type} evolution)', 
                    fontsize=font_size, fontweight='bold', y=0.98)
        
        # Improve spacing between subplots (research pipeline style)
        if n_contexts > 16:
            plt.tight_layout(rect=[0, 0, 0.95, 0.95], h_pad=2.0, w_pad=1.0)
        else:
            plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        
        # Save in multiple formats
        base_name = f"seed_{seed:06d}_evolution"
        for fmt in ['png', 'svg', 'pdf']:
            plt.savefig(plots_dir / f"{base_name}.{fmt}", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _create_convergence_plot(self, result: SingleRunResults, plots_dir: Path):
        """Create convergence analysis plot for a single run."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
        import matplotlib.pyplot as plt
        
        if not result.em_results.statistics_history:
            return
        
        seed = result.seed
        stats_history = result.em_results.statistics_history
        
        # Extract convergence data
        iterations = [stat.iteration for stat in stats_history]
        log_likelihoods = [stat.log_likelihood for stat in stats_history]
        likelihood_changes = [stat.log_likelihood_change for stat in stats_history[1:]]
        parameter_changes = [stat.parameter_change for stat in stats_history[1:]]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Log likelihood evolution
        ax1.plot(iterations, log_likelihoods, 'b-', linewidth=2)
        ax1.set_xlabel('EM Iteration')
        ax1.set_ylabel('Expected Log Likelihood')
        ax1.set_title('Likelihood Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Mark best iteration
        best_iter = result.em_results.best_iteration
        best_ll = result.em_results.best_log_likelihood
        ax1.axvline(x=best_iter, color='r', linestyle='--', alpha=0.7, label=f'Best (iter {best_iter})')
        ax1.legend()
        
        # Log likelihood changes
        if likelihood_changes:
            ax2.plot(iterations[1:], np.abs(likelihood_changes), 'g-', linewidth=2)
            ax2.set_xlabel('EM Iteration')
            ax2.set_ylabel('|Log Likelihood Change|')
            ax2.set_title('Convergence Rate')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        # Parameter changes
        if parameter_changes:
            ax3.plot(iterations[1:], parameter_changes, 'r-', linewidth=2)
            ax3.set_xlabel('EM Iteration')
            ax3.set_ylabel('Parameter Change Norm')
            ax3.set_title('Parameter Stability')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # Convergence summary
        ax4.axis('off')
        summary_text = f"""Convergence Summary:
        
Best Iteration: {best_iter}
Best Log-Likelihood: {best_ll:.2f}
Final Log-Likelihood: {result.final_log_likelihood:.2f}
Runtime: {result.runtime_seconds:.1f} seconds
Total Iterations: {len(stats_history)}
"""
        ax4.text(0.1, 0.8, summary_text, fontsize=11, 
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'EM Convergence Analysis - Seed {seed}', fontsize=14)
        plt.tight_layout()
        
        # Save in multiple formats
        base_name = f"seed_{seed:06d}_convergence"
        for fmt in ['png', 'svg', 'pdf']:
            plt.savefig(plots_dir / f"{base_name}.{fmt}", dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _create_aggregate_cross_entropy_plot(self, config_id: int, results: List[SingleRunResults]):
        """Create aggregate cross-entropy comparison plot across all seeds."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
        import matplotlib.pyplot as plt
        from aggregate_analysis_utils import StatisticalAnalyzer
        
        agg_config = self.experimental_configs[config_id]
        config_dir = self.output_dir / f"config_{config_id}_{agg_config.config_name}"
        
        # Extract cross-entropy data
        cross_entropy_data = {
            'em_cross_entropy': np.array([r.em_cross_entropy for r in results]),
            'raw_cross_entropy': np.array([r.raw_cross_entropy for r in results]),
            'smoothed_raw_cross_entropy': np.array([r.smoothed_raw_cross_entropy for r in results])
        }
        
        # Compute statistics
        analyzer = StatisticalAnalyzer()
        ce_statistics = analyzer.compute_cross_entropy_statistics(cross_entropy_data)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Mean cross-entropy over time with std bands
        n_timesteps = cross_entropy_data['em_cross_entropy'].shape[1]
        timesteps = np.arange(n_timesteps)
        
        colors = {'em_cross_entropy': 'blue', 'raw_cross_entropy': 'red', 'smoothed_raw_cross_entropy': 'green'}
        labels = {'em_cross_entropy': 'EM Estimates', 'raw_cross_entropy': 'Raw Estimates', 'smoothed_raw_cross_entropy': 'Smoothed Raw Estimates'}
        
        for method, stats in ce_statistics.items():
            color = colors[method]
            label = labels[method]
            
            # Plot mean with std bands
            ax1.plot(timesteps, stats['mean'], color=color, linewidth=2, label=label)
            ax1.fill_between(timesteps, 
                           stats['mean'] - stats['std'], 
                           stats['mean'] + stats['std'],
                           color=color, alpha=0.2)
        
        ax1.set_xlabel('Song Number')
        ax1.set_ylabel('Cross-Entropy')
        ax1.set_title(f'Cross-Entropy Evolution\n{agg_config.description}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plots for overall performance comparison with significance
        all_ce_data = []
        method_labels = []
        
        for method, data in cross_entropy_data.items():
            # Average cross-entropy across time for each run
            avg_ce_per_run = np.mean(data, axis=1)
            all_ce_data.extend(avg_ce_per_run)
            method_labels.extend([labels[method]] * len(avg_ce_per_run))
        
        import pandas as pd
        df = pd.DataFrame({'Cross-Entropy': all_ce_data, 'Method': method_labels})
        
        import seaborn as sns
        sns.boxplot(data=df, x='Method', y='Cross-Entropy', ax=ax2)
        ax2.set_title(f'Overall Performance Comparison\n({len(results)} runs)')
        ax2.set_ylabel('Mean Cross-Entropy')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add significance annotations
        # Get statistical test results for this configuration
        significance_tests = analyzer.perform_significance_tests(cross_entropy_data)
        
        # Add significance text annotations above the plot
        y_max = ax2.get_ylim()[1]
        y_text_start = y_max * 1.05
        
        annotation_text = "Statistical Significance:\n"
        for _, row in significance_tests.iterrows():
            method1_clean = row['method1'].replace('_cross_entropy', '').replace('_', ' ').title()
            method2_clean = row['method2'].replace('_cross_entropy', '').replace('_', ' ').title()
            t_sig = row['t_test_significance']
            
            annotation_text += f"{method1_clean} vs {method2_clean}: {t_sig}\n"
        
        # Add the annotation text above the plot
        ax2.text(0.02, 0.98, annotation_text.strip(), transform=ax2.transAxes, 
                verticalalignment='top', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save in multiple formats
        base_name = "aggregate_cross_entropy_comparison"
        for fmt in ['png', 'svg', 'pdf']:
            plt.savefig(config_dir / f"{base_name}.{fmt}", dpi=300, bbox_inches='tight')
        
        # Also save to summary plots directory
        summary_plots_dir = self.output_dir / "summary_plots"
        for fmt in ['png', 'svg', 'pdf']:
            plt.savefig(summary_plots_dir / f"config_{config_id}_{base_name}.{fmt}", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
        
        logger.info(f"Aggregate cross-entropy plot saved for config {config_id}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Aggregate statistical analysis for Adaptive Syntax Filter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config-id", 
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Configuration ID to run (1-4) or 'all' for all configs"
    )
    
    parser.add_argument(
        "--n-seeds",
        type=int,
        help="Number of random seeds to run (overrides config defaults)"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with 10 seeds per config"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel processes"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/aggregate_analysis",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if available"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main entry point for aggregate analysis."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Aggregate Analysis")
    logger.info(f"Configuration: {args}")
    
    # Initialize manager
    manager = AggregateAnalysisManager(output_dir=args.output_dir)
    
    # Determine which configs to run
    if args.config_id == "all":
        config_ids = [1, 2, 3, 4]
    else:
        config_ids = [int(args.config_id)]
    
    # Run analysis
    for config_id in config_ids:
        agg_config = manager.experimental_configs[config_id]
        
        # Determine number of seeds
        if args.test_mode:
            n_seeds = 10
        elif args.n_seeds:
            n_seeds = args.n_seeds
        else:
            n_seeds = agg_config.n_seeds
        
        logger.info(f"Running config {config_id}: {agg_config.description}")
        logger.info(f"  Seeds: {n_seeds}, Parallel: {args.parallel}")
        
        # Run batch processing with multiprocessing
        if args.test_mode or n_seeds <= 10:
            # For small runs, run sequentially for easier debugging
            results = manager.run_sequential_batch(config_id, n_seeds)
        else:
            # For large runs, use parallel processing
            results = manager.run_parallel_batch(config_id, n_seeds, args.parallel, 
                                               resume=args.resume)
        
        logger.info(f"Completed {len(results)} out of {n_seeds} runs for config {config_id}")
        
        if results:
            # Note: HDF5 files are already saved during parallel processing
            # Just compute aggregates and create plots
            logger.info(f"Creating aggregate analysis and plots for {len(results)} results...")
            manager.compute_and_save_aggregates(config_id, results)
            
            # Create example plots for a few seeds
            manager.create_example_plots(config_id, results[:5])  # First 5 successful runs
            
            logger.info(f"Config {config_id} analysis complete - all data saved to HDF5 files during processing")
    
    logger.info("Aggregate Analysis Complete")


if __name__ == "__main__":
    main() 