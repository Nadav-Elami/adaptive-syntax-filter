"""
Publication Figure Generation Pipeline.

Automated generation of publication-ready figures following Nature/Science standards.
Integrates all visualization modules to create coherent figure sets.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

# Import visualization modules
from .logit_evolution import LogitEvolutionDashboard, apply_block_softmax_to_trajectory
from .probability_evolution import create_transition_heatmap_series, ProbabilityEvolutionAnalyzer
from .performance_assessment import PerformanceAnalyzer
from .sequence_analysis import SequenceAnalyzer

# Publication standards
PUBLICATION_CONFIG = {
    'figure_width_single': 8.5,      # Nature single column (cm)
    'figure_width_double': 17.8,     # Nature double column (cm)
    'dpi': 300,
    'font_size': 8,
    'line_width': 1.0,
    'marker_size': 4
}

@dataclass
class PublicationConfig:
    """Configuration for publication figures."""
    journal_style: str = 'nature'  # 'nature', 'science', 'generic'
    figure_format: str = 'pdf'     # 'pdf', 'png', 'svg'
    dpi: int = 300
    font_family: str = 'sans-serif'
    color_palette: str = 'Set2'


def setup_publication_style():
    """Setup matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': PUBLICATION_CONFIG['font_size'],
        'figure.dpi': PUBLICATION_CONFIG['dpi'],
        'savefig.dpi': PUBLICATION_CONFIG['dpi'],
        'savefig.bbox': 'tight'
    })


def create_main_results_figure(results_data: Dict[str, Any],
                              alphabet: List[str]) -> plt.Figure:
    """
    Create main results figure for publication (Figure 1).
    
    Parameters
    ----------
    results_data : Dict[str, Any]
        Complete analysis results
    alphabet : List[str]
        Sequence alphabet
        
    Returns
    -------
    plt.Figure
        Main results figure
    """
    setup_publication_style()
    
    fig = plt.figure(figsize=(PUBLICATION_CONFIG['figure_width_double'] / 2.54, 10 / 2.54))
    
    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2)
    
    # Panel A: Model schematic
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.text(0.5, 0.5, f'Adaptive Kalman-EM\n{len(alphabet)} symbols', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightblue"))
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis('off')
    ax_a.text(-0.1, 1.1, 'A', transform=ax_a.transAxes, fontweight='bold', fontsize=12)
    
    # Panel B: Parameter evolution
    ax_b = fig.add_subplot(gs[0, 1])
    if 'logits_estimated' in results_data:
        logits = results_data['logits_estimated']
        x = np.arange(logits.shape[1])
        for i in range(min(3, logits.shape[0])):
            finite_mask = np.isfinite(logits[i, :])
            if np.any(finite_mask):
                ax_b.plot(x[finite_mask], logits[i, finite_mask], alpha=0.7)
        ax_b.set_xlabel('Song Number')
        ax_b.set_ylabel('Logit Value')
        ax_b.set_title('Parameter Evolution')
    ax_b.text(-0.1, 1.1, 'B', transform=ax_b.transAxes, fontweight='bold', fontsize=12)
    
    # Panel C: Performance metrics
    ax_c = fig.add_subplot(gs[1, 0])
    if 'performance_metrics' in results_data:
        metrics = results_data['performance_metrics']
        metric_names = ['RMSE', 'R²', 'Corr']
        metric_values = [
            getattr(metrics, 'rmse', 0),
            getattr(metrics, 'r_squared', 0),
            getattr(metrics, 'correlation_coefficient', 0)
        ]
        ax_c.bar(metric_names, metric_values, alpha=0.7)
        ax_c.set_ylabel('Value')
        ax_c.set_title('Performance')
    ax_c.text(-0.1, 1.1, 'C', transform=ax_c.transAxes, fontweight='bold', fontsize=12)
    
    # Panel D: Transition matrix (simplified view for publication)
    ax_d = fig.add_subplot(gs[1, 1])
    if 'transition_matrices' in results_data:
        matrices = results_data['transition_matrices']
        n_symbols = len(alphabet)
        
        # For higher-order models, create a simplified view by aggregating
        if matrices.shape[0] == n_symbols * n_symbols:
            # Simple first-order case
            final_matrix = matrices[:, -1].reshape(n_symbols, n_symbols)
        else:
            # Higher-order case: aggregate to simple transitions for visualization
            final_matrix = np.zeros((n_symbols, n_symbols))
            state_size = matrices.shape[0]
            
            # Sum probabilities across all contexts for each symbol-to-symbol transition
            for i in range(state_size):
                to_symbol_idx = i % n_symbols
                # Simple aggregation: average across contexts
                final_matrix[:, to_symbol_idx] += matrices[i, -1]
            
            # Normalize rows to make it a proper transition matrix
            row_sums = final_matrix.sum(axis=1)
            final_matrix = final_matrix / row_sums[:, np.newaxis]
            
        im = ax_d.imshow(final_matrix, cmap='viridis')
        ax_d.set_xticks(range(n_symbols))
        ax_d.set_yticks(range(n_symbols))
        ax_d.set_xticklabels(alphabet, fontsize=6)
        ax_d.set_yticklabels(alphabet, fontsize=6)
        ax_d.set_title('Final Syntax')
        plt.colorbar(im, ax=ax_d, fraction=0.046)
    ax_d.text(-0.1, 1.1, 'D', transform=ax_d.transAxes, fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    return fig


def create_supplementary_figure_s1(results_data: Dict[str, Any],
                                  alphabet: List[str],
                                  config: Optional[PublicationConfig] = None) -> plt.Figure:
    """Create supplementary figure S1: Detailed algorithm performance."""
    if config is None:
        config = PublicationConfig()
    
    setup_publication_style()
    
    fig = plt.figure(figsize=(PUBLICATION_CONFIG['figure_width_double'] / 2.54, 10 / 2.54))
    
    # Create detailed performance analysis
    gs = fig.add_gridspec(2, 2)
    
    # Convergence analysis
    if 'convergence_history' in results_data:
        ax1 = fig.add_subplot(gs[0, 0])
        convergence = results_data['convergence_history']
        ax1.plot(convergence, 'b-', linewidth=1.0)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('JS Divergence')
        ax1.set_title('Convergence')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
    
    # Likelihood evolution
    if 'likelihood_history' in results_data:
        ax2 = fig.add_subplot(gs[0, 1])
        likelihood = results_data['likelihood_history']
        ax2.plot(likelihood, 'r-', linewidth=1.0)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Log Likelihood')
        ax2.set_title('Likelihood Evolution')
        ax2.grid(True, alpha=0.3)
    
    # Parameter recovery (if available)
    if 'logits_estimated' in results_data and 'logits_true' in results_data:
        ax3 = fig.add_subplot(gs[1, :])
        estimated = results_data['logits_estimated'].flatten()
        true = results_data['logits_true'].flatten()
        
        # Filter finite values
        finite_mask = np.isfinite(estimated) & np.isfinite(true)
        estimated_finite = estimated[finite_mask]
        true_finite = true[finite_mask]
        
        ax3.scatter(true_finite, estimated_finite, alpha=0.5, s=1)
        min_val = min(np.min(true_finite), np.min(estimated_finite))
        max_val = max(np.max(true_finite), np.max(estimated_finite))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.0)
        ax3.set_xlabel('True Parameters')
        ax3.set_ylabel('Estimated Parameters')
        ax3.set_title('Parameter Recovery')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


class PublicationFigureManager:
    """Manager for creating publication figure sets."""
    
    def __init__(self, config: Optional[PublicationConfig] = None):
        self.config = config or PublicationConfig()
    
    def generate_figure_set(self, results_data: Dict[str, Any], 
                           alphabet: List[str]) -> Dict[str, plt.Figure]:
        """Generate complete set of publication figures."""
        return self.generate_complete_figure_set(results_data, alphabet)
    
    def generate_complete_figure_set(self,
                                   results_data: Dict[str, Any],
                                   alphabet: List[str],
                                   save_dir: Optional[Union[str, Path]] = None) -> Dict[str, plt.Figure]:
        """
        Generate complete set of publication figures.
        
        Parameters
        ----------
        results_data : Dict[str, Any]
            Complete analysis results
        alphabet : List[str]
            Sequence alphabet
        save_dir : Optional[Union[str, Path]]
            Directory to save figures
            
        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary of generated figures
        """
        figures = {}
        
        # Main results figure
        fig_main = create_main_results_figure(results_data, alphabet)
        figures['figure_1_main_results'] = fig_main
        
        # Supplementary figures
        fig_s1 = create_supplementary_figure_s1(results_data, alphabet, self.config)
        figures['figure_s1_algorithm_details'] = fig_s1
        
        # Save figures if directory provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            for name, fig in figures.items():
                # Save in multiple formats
                fig.savefig(save_path / f"{name}.pdf", dpi=self.config.dpi, bbox_inches='tight')
                fig.savefig(save_path / f"{name}.png", dpi=self.config.dpi, bbox_inches='tight')
                if self.config.figure_format == 'svg':
                    fig.savefig(save_path / f"{name}.svg", bbox_inches='tight')
        
        return figures
    
    def create_figure_captions(self, figures: Dict[str, plt.Figure]) -> Dict[str, str]:
        """Generate figure captions for publication."""
        captions = {}
        
        if 'figure_1_main_results' in figures:
            captions['figure_1_main_results'] = """
Figure 1. Adaptive syntax filter algorithm performance and results.
(A) Schematic of the adaptive Kalman-EM algorithm showing temporal evolution 
and constraint enforcement. (B) Example parameter evolution trajectories 
showing convergence over song sequences. (C) Model performance metrics 
including RMSE, R², and correlation coefficients. (D) Final syntax state 
showing transition probability matrix. (E) Sequence statistics including 
average length, diversity ratio, and total symbol usage.
"""
        
        if 'figure_s1_algorithm_details' in figures:
            captions['figure_s1_algorithm_details'] = """
Supplementary Figure S1. Detailed algorithm performance analysis.
Convergence analysis showing JS divergence reduction, likelihood evolution, 
and parameter recovery accuracy across iterations.
"""
        
        return captions


def demonstrate_publication_figures(alphabet: Optional[List[str]] = None,
                                  n_sequences: int = 100) -> Dict[str, Any]:
    """
    Demonstration of publication figure generation.
    
    Parameters
    ----------
    alphabet : Optional[List[str]]
        Sequence alphabet
    n_sequences : int
        Number of sequences to simulate
        
    Returns
    -------
    Dict[str, Any]
        Generated figures and captions
    """
    if alphabet is None:
        alphabet = ['<', 'A', 'B', 'C', '>']
    
    print(f"Generating publication figures for {len(alphabet)}-symbol alphabet")
    
    # Generate synthetic results data
    n_symbols = len(alphabet)
    n_params = n_symbols ** 2
    
    np.random.seed(42)
    
    # Create synthetic results
    results_data = {
        'logits_estimated': np.random.randn(n_params, n_sequences),
        'logits_true': np.random.randn(n_params, n_sequences),
        'transition_matrices': np.random.rand(n_params, n_sequences),
        'convergence_history': [0.1 * np.exp(-i/10) for i in range(50)],
        'likelihood_history': [-100 + 90 * (1 - np.exp(-i/15)) for i in range(50)],
        'performance_metrics': type('obj', (object,), {
            'rmse': 15.2,
            'r_squared': 0.85,
            'correlation_coefficient': 0.92
        })(),
        'sequence_analysis': {
            'length_analysis': {
                'statistics': {'mean': 6.5, 'std': 2.1}
            },
            'usage_analysis': {
                'diversity_ratio': 0.85,
                'total_symbols': 650
            }
        }
    }
    
    # Generate figures
    manager = PublicationFigureManager()
    figures = manager.generate_complete_figure_set(
        results_data, alphabet, save_dir="demo_publication_figures"
    )
    captions = manager.create_figure_captions(figures)
    
    print(f"Generated {len(figures)} publication-ready figures")
    print(f"Saved to demo_publication_figures/ directory")
    
    return {
        'figures': figures,
        'captions': captions,
        'results_data': results_data
    }


if __name__ == "__main__":
    # Run demonstration
    demo_results = demonstrate_publication_figures()
    plt.show() 
