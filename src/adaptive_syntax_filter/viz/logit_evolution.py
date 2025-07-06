"""
Advanced Logit Parameter Evolution Visualization.

This module provides real-time and static visualization of logit parameter evolution 
for the adaptive Kalman-EM algorithm. Designed for higher-order Markov models with 
any alphabet size (3-40+ symbols) and publication-ready output.

Priority: #1 module for Phase 3 visualization system.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import warnings
from typing import List, Tuple, Optional, Dict, Union, Any
from pathlib import Path
from dataclasses import dataclass
import matplotlib.cm as cm

# Publication-quality settings
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Publication configuration
PUBLICATION_CONFIG = {
    'figure_width': 8.5,          # Nature single column: 8.5cm
    'figure_width_double': 17.8,  # Nature double column: 17.8cm  
    'dpi': 300,                   # Publication DPI
    'font_size': 8,               # Nature font size
    'line_width': 1.0,
    'marker_size': 4,
    'color_palette': 'Set2'
}


@dataclass
class LogitVisualizationConfig:
    """Configuration for logit parameter visualization."""
    figure_size: Tuple[float, float] = (15, 10)
    dpi: int = 300
    font_size: int = 10
    line_width: float = 2.0
    alpha_estimated: float = 1.0
    alpha_true: float = 0.8
    color_palette: str = 'Set2'
    show_constraints: bool = True
    show_legend: bool = True
    publication_mode: bool = False


def encode_context_for_viz(context: List[str], alphabet: List[str]) -> int:
    """
    Encode context sequence to index for visualization purposes.
    Compatible with data generation encoding from constraint_system.py.
    """
    if not context:
        return 0
    
    n_symbols = len(alphabet)
    order = len(context)
    context_idx = 0
    
    for i, symbol in enumerate(context):
        if symbol not in alphabet:
            raise ValueError(f"Symbol '{symbol}' not in alphabet")
        
        symbol_idx = alphabet.index(symbol)
        power = order - 1 - i
        context_idx += symbol_idx * (n_symbols ** power)
    
    return context_idx


def decode_context_for_viz(context_idx: int, alphabet: List[str], order: int) -> List[str]:
    """
    Decode context index back to symbol sequence for visualization.
    
    Parameters
    ----------
    context_idx : int
        Context index
    alphabet : List[str]
        Symbol alphabet
    order : int
        Markov order
        
    Returns
    -------
    List[str]
        Context symbols
    """
    n_symbols = len(alphabet)
    context_symbols = []
    
    temp_idx = context_idx
    for i in range(order):
        symbol_idx = temp_idx % n_symbols
        context_symbols.insert(0, alphabet[symbol_idx])
        temp_idx //= n_symbols
    
    return context_symbols


def block_softmax_viz(logits: np.ndarray, n_symbols: int, 
                     handle_constraints: bool = True) -> np.ndarray:
    """
    Apply block-wise softmax to logit vector for visualization.
    Each block represents transitions FROM one context TO all symbols.
    
    Parameters
    ----------
    logits : np.ndarray
        Logit parameters arranged in blocks
    n_symbols : int
        Number of symbols in alphabet
    handle_constraints : bool
        Whether to handle -inf constraint values
        
    Returns
    -------
    np.ndarray
        Probability values after block softmax
    """
    n_blocks = len(logits) // n_symbols
    probs = np.zeros_like(logits)
    
    for block_idx in range(n_blocks):
        start_idx = block_idx * n_symbols
        end_idx = start_idx + n_symbols
        block_logits = logits[start_idx:end_idx]
        
        if handle_constraints:
            # Handle all -inf case (no valid transitions)
            finite_mask = np.isfinite(block_logits)
            if not np.any(finite_mask):
                probs[start_idx:end_idx] = 1.0 / n_symbols  # Uniform fallback
                continue
            
            # Numerically stable softmax
            max_val = np.max(block_logits[finite_mask])
            exp_logits = np.exp(block_logits - max_val)
            exp_logits[~finite_mask] = 0  # Set -inf to 0 probability
            sum_exp = np.sum(exp_logits)
            if sum_exp > 0:
                probs[start_idx:end_idx] = exp_logits / sum_exp
            else:
                probs[start_idx:end_idx] = 1.0 / n_symbols
        else:
            # Standard softmax
            max_val = np.max(block_logits)
            exp_logits = np.exp(block_logits - max_val)
            probs[start_idx:end_idx] = exp_logits / np.sum(exp_logits)
    
    return probs


class LogitEvolutionDashboard:
    """
    Real-time logit parameter evolution dashboard.
    
    Designed for higher-order Markov models with any alphabet size.
    Priority #1 visualization for Phase 3.
    """
    
    def __init__(self, 
                 alphabet: List[str], 
                 markov_order: int = 1, 
                 config: Optional[LogitVisualizationConfig] = None):
        """
        Initialize logit evolution dashboard.
        
        Parameters
        ----------
        alphabet : List[str]
            Sequence alphabet including start/end tokens
        markov_order : int
            Order of Markov model (1-5 supported)
        config : Optional[LogitVisualizationConfig]
            Visualization configuration
        """
        self.alphabet = alphabet
        self.n_symbols = len(alphabet)
        self.markov_order = markov_order
        self.config = config or LogitVisualizationConfig()
        
        # Calculate state space dimensions
        self.n_contexts = self.n_symbols ** markov_order
        self.state_space_size = self.n_symbols ** (markov_order + 1)
        
        # Generate context labels for subplots
        self.context_labels = self._generate_context_labels()
        
        # Color scheme: unique color for each target symbol
        palette = plt.get_cmap(self.config.color_palette)
        self.colors = palette(np.linspace(0, 1, self.n_symbols))
        
        # Initialize figures
        self.fig_logits = None
        self.fig_probs = None
        self.axes_logits = None
        self.axes_probs = None
        
        # Animation objects for real-time updates
        self.animation_logits = None
        self.animation_probs = None
        
    def _generate_context_labels(self) -> List[str]:
        """Generate human-readable context labels for higher-order models."""
        if self.markov_order == 1:
            return [f"'{symbol}'" for symbol in self.alphabet]
        else:
            contexts = []
            for context_idx in range(self.n_contexts):
                context_symbols = decode_context_for_viz(context_idx, self.alphabet, self.markov_order)
                context_str = "'" + " ".join(context_symbols) + "'"
                contexts.append(context_str)
            return contexts
    
    def _calculate_subplot_layout(self, n_subplots: int) -> Tuple[int, int]:
        """Calculate optimal subplot layout for given number of contexts."""
        if n_subplots <= 4:
            return (2, 2)
        elif n_subplots <= 6:
            return (2, 3)
        elif n_subplots <= 9:
            return (3, 3)
        elif n_subplots <= 12:
            return (3, 4)
        elif n_subplots <= 16:
            return (4, 4)
        elif n_subplots <= 20:
            return (4, 5)
        elif n_subplots <= 25:
            return (5, 5)
        else:
            # For very large contexts, use a more rectangular layout
            # Favor more columns than rows for better horizontal space usage
            target_aspect_ratio = 1.5  # Width to height ratio
            cols = max(5, int(np.ceil(np.sqrt(n_subplots * target_aspect_ratio))))
            rows = int(np.ceil(n_subplots / cols))
            return (rows, cols)
    
    def setup_static_figures(self) -> Tuple[plt.Figure, plt.Figure]:
        """
        Setup static figure layouts for logit and probability evolution.
        
        Returns
        -------
        Tuple[plt.Figure, plt.Figure]
            Logit and probability figures
        """
        # Setup logit evolution figure (PRIMARY)
        self.fig_logits = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Setup probability evolution figure (SECONDARY)  
        prob_fig_size = (self.config.figure_size[0] * 0.8, self.config.figure_size[1] * 0.8)
        self.fig_probs = plt.figure(figsize=prob_fig_size, dpi=self.config.dpi)
        
        return self.fig_logits, self.fig_probs
    
    def plot_logit_evolution_static(self, 
                                  logits_estimated: np.ndarray,
                                  logits_true: Optional[np.ndarray] = None,
                                  sequence_indices: Optional[np.ndarray] = None,
                                  title: str = "Logit Parameters Evolution") -> plt.Figure:
        """
        Create static plot of logit parameter evolution.
        
        PRIORITY VISUALIZATION: Shows raw logit values before softmax transformation.
        
        Parameters
        ----------
        logits_estimated : np.ndarray
            Estimated logit parameters (state_space_size x n_sequences)
        logits_true : Optional[np.ndarray]
            True logit parameters for comparison
        sequence_indices : Optional[np.ndarray]
            Custom sequence indices for x-axis
        title : str
            Figure title
            
        Returns
        -------
        plt.Figure
            Logit evolution figure
        """
        if self.fig_logits is None:
            self.setup_static_figures()
        
        self.fig_logits.clear()
        
        # Dynamic subplot layout based on number of contexts
        rows, cols = self._calculate_subplot_layout(self.n_contexts)
        
        # Adjust figure size based on number of subplots
        if self.n_contexts > 25:
            fig_height = max(12, rows * 2.5)
            self.fig_logits.set_size_inches(max(15, cols * 3), fig_height)
        
        # Sequence indices for x-axis
        if sequence_indices is None:
            sequence_indices = np.arange(logits_estimated.shape[1])
        
        # Plot each context block
        legend_handles = []
        legend_labels = []
        for target_idx in range(self.n_symbols):
            color = self.colors[target_idx]
            target_symbol = self.alphabet[target_idx]
            # Add legend entries for both estimated and true lines
            from matplotlib.lines import Line2D
            legend_handles.append(Line2D([0], [0], linestyle='--', color=color, linewidth=self.config.line_width))
            legend_labels.append(f"Est: →{target_symbol}")
            legend_handles.append(Line2D([0], [0], linestyle='-', color=color, linewidth=self.config.line_width * 0.75))
            legend_labels.append(f"True: →{target_symbol}")
        
        for context_idx in range(self.n_contexts):
            ax = self.fig_logits.add_subplot(rows, cols, context_idx + 1)
            
            # Calculate block indices for transitions FROM this context
            start_idx = context_idx * self.n_symbols
            end_idx = (context_idx + 1) * self.n_symbols
            
            # Plot each transition logit value
            for target_idx in range(self.n_symbols):
                pos_idx = start_idx + target_idx
                
                # Skip if out of bounds
                if pos_idx >= logits_estimated.shape[0]:
                    continue
                
                # Get color for this context (same color for all transitions from this context)
                color = self.colors[target_idx]
                target_symbol = self.alphabet[target_idx]
                
                # Check if this transition is constrained
                estimated_values = logits_estimated[pos_idx, :]
                is_forbidden = np.all(estimated_values == -np.inf)
                is_required = np.all(estimated_values >= 1e7)
                
                # Estimated logits (primary line)
                finite_mask = np.isfinite(estimated_values)
                if np.any(finite_mask):
                    if not is_forbidden:
                        line_style = '--' if logits_true is not None else '-'
                        line_width = self.config.line_width * 1.5 if is_required else self.config.line_width
                    ax.plot(sequence_indices[finite_mask], 
                           estimated_values[finite_mask], 
                           line_style,
                           color=color,
                           linewidth=line_width,
                           alpha=self.config.alpha_estimated)
                
                # True logits (if provided)
                if logits_true is not None:
                    true_values = logits_true[pos_idx, :]
                    finite_mask_true = np.isfinite(true_values)
                    if np.any(finite_mask_true) and not is_forbidden:
                        ax.plot(sequence_indices[finite_mask_true],
                               true_values[finite_mask_true], 
                               '-',
                               color=color,
                               linewidth=self.config.line_width * 0.75,
                               alpha=self.config.alpha_true)
            
            # Customize subplot
            context_label = self.context_labels[context_idx]
            
            # Adjust font sizes based on number of subplots
            title_font_size = max(6, self.config.font_size - max(0, self.n_contexts // 10))
            label_font_size = max(6, self.config.font_size - max(0, self.n_contexts // 8))
            tick_font_size = max(5, self.config.font_size - max(0, self.n_contexts // 6))
            
            ax.set_title(f"From {context_label}", fontsize=title_font_size, pad=5)
            
            # Only show x-axis labels on bottom row
            if context_idx >= (rows-1)*cols:
                ax.set_xlabel('Song #', fontsize=label_font_size)
            else:
                ax.set_xticklabels([])
            
            # Only show y-axis labels on left column
            if context_idx % cols == 0:
                ax.set_ylabel('Logit', fontsize=label_font_size)
            else:
                ax.set_yticklabels([])
            
            # Reduce tick density for large state spaces
            if self.n_contexts > 16:
                ax.locator_params(nbins=3)
            
            ax.tick_params(labelsize=tick_font_size)
            ax.grid(True, alpha=0.3)
            
            # Only show legend on last subplot
            if self.config.show_legend and self.n_contexts <= 16:
                if context_idx == self.n_contexts - 1:
                    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=max(5, self.config.font_size - 2), ncol=1)
        
        # Overall figure formatting
        font_size = self.config.font_size + 2 if self.config.publication_mode else self.config.font_size + 4
        self.fig_logits.suptitle(title, fontsize=font_size, fontweight='bold', y=0.98)
        
        # Improve spacing between subplots
        if self.n_contexts > 16:
            plt.tight_layout(rect=[0, 0, 0.95, 0.95], h_pad=2.0, w_pad=1.0)
        else:
            plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        
        return self.fig_logits
    
    def _highlight_constraint_violations(self, ax: plt.Axes, values: np.ndarray, 
                                        target_symbol: str, sequence_indices: np.ndarray, 
                                        context_idx: int):
        """Highlight constraint violations with background coloring."""
        # Get the context this subplot represents
        context_symbols = decode_context_for_viz(context_idx, self.alphabet, self.markov_order)
        
        # Check if this transition should be constrained
        should_be_forbidden = self._should_transition_be_forbidden(context_symbols, target_symbol)
        should_be_required = self._should_transition_be_required(context_symbols, target_symbol)
        
        # Check actual constraint status
        is_forbidden = np.all(values == -np.inf)
        is_required = np.all(values >= 1e7)
        
        # Highlight violations
        if should_be_forbidden and not is_forbidden:
            # Should be forbidden but isn't - red background
            ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.1, color='red', zorder=0)
            ax.text(0.02, 0.98, f"⚠ {target_symbol} should be forbidden", 
                   transform=ax.transAxes, fontsize=6, color='red', 
                   verticalalignment='top')
        elif should_be_required and not is_required:
            # Should be required but isn't - orange background
            ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.1, color='orange', zorder=0)
            ax.text(0.02, 0.95, f"⚠ {target_symbol} should be required", 
                   transform=ax.transAxes, fontsize=6, color='orange', 
                   verticalalignment='top')
    
    def _should_transition_be_forbidden(self, context_symbols: List[str], target_symbol: str) -> bool:
        """Check if a transition should be forbidden based on grammar rules."""
        # '<' can never be a transition target (except after '>')
        if target_symbol == '<' and '>' not in context_symbols:
            return True
        
        # Cannot transition to '>' if context contains '<' with insufficient content
        if target_symbol == '>' and '<' in context_symbols:
            if self.markov_order > 1:
                non_start_symbols = [s for s in context_symbols if s != '<']
                if len(non_start_symbols) < self.markov_order - 1:
                    return True
        
        return False
    
    def _should_transition_be_required(self, context_symbols: List[str], target_symbol: str) -> bool:
        """Check if a transition should be required based on grammar rules."""
        # After '>', must transition to '<'
        if '>' in context_symbols and target_symbol == '<':
            # Check if '>' is the rightmost symbol
            rightmost_end = max([i for i, s in enumerate(context_symbols) if s == '>'])
            if rightmost_end == len(context_symbols) - 1:
                return True
        
        return False
    
    def plot_probability_evolution_static(self,
                                        probs_estimated: np.ndarray,
                                        probs_true: Optional[np.ndarray] = None,
                                        sequence_indices: Optional[np.ndarray] = None,
                                        title: str = "Transition Probabilities (After Block Softmax)") -> plt.Figure:
        """
        Create static plot of transition probability evolution.
        
        Parameters
        ----------
        probs_estimated : np.ndarray
            Estimated probability parameters
        probs_true : Optional[np.ndarray]
            True probability parameters for comparison
        sequence_indices : Optional[np.ndarray]
            Custom sequence indices for x-axis
        title : str
            Figure title
            
        Returns
        -------
        plt.Figure
            Probability evolution figure
        """
        if self.fig_probs is None:
            self.setup_static_figures()
        
        self.fig_probs.clear()
        
        # Dynamic subplot layout
        rows, cols = self._calculate_subplot_layout(self.n_contexts)
        
        # Adjust figure size based on number of subplots
        if self.n_contexts > 25:
            fig_height = max(12, rows * 2.5)
            self.fig_probs.set_size_inches(max(15, cols * 3), fig_height)
        
        # Sequence indices for x-axis
        if sequence_indices is None:
            sequence_indices = np.arange(probs_estimated.shape[1])
        
        # Plot each context block
        for context_idx in range(self.n_contexts):
            ax = self.fig_probs.add_subplot(rows, cols, context_idx + 1)
            
            # Calculate block indices
            start_idx = context_idx * self.n_symbols
            end_idx = (context_idx + 1) * self.n_symbols
            
            # Plot each transition probability
            for target_idx in range(self.n_symbols):
                pos_idx = start_idx + target_idx
                
                if pos_idx >= probs_estimated.shape[0]:
                    continue
                
                # Get color for this context (same color for all transitions from this context)
                color = self.colors[target_idx]
                target_symbol = self.alphabet[target_idx]
                
                # Skip forbidden transitions (probability always 0)
                estimated_probs = probs_estimated[pos_idx, :]
                if np.allclose(estimated_probs, 0.0):
                    continue
                
                # Estimated probabilities
                ax.plot(sequence_indices, estimated_probs, 
                       '--' if probs_true is not None else '-',
                       color=color,
                       linewidth=self.config.line_width,
                       alpha=self.config.alpha_estimated,
                       label=f"{'Est ' if probs_true is not None else ''}→ {target_symbol}")
                
                # True probabilities (if provided)
                if probs_true is not None:
                    true_probs = probs_true[pos_idx, :]
                    if not np.allclose(true_probs, 0.0):
                        ax.plot(sequence_indices, true_probs, 
                           '-',
                           color=color,
                           linewidth=self.config.line_width * 0.75,
                           alpha=self.config.alpha_true,
                           label=f"True → {target_symbol}")
            
            # Customize subplot
            context_label = self.context_labels[context_idx]
            
            # Adjust font sizes based on number of subplots
            title_font_size = max(6, self.config.font_size - max(0, self.n_contexts // 10))
            label_font_size = max(6, self.config.font_size - max(0, self.n_contexts // 8))
            tick_font_size = max(5, self.config.font_size - max(0, self.n_contexts // 6))
            
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
            if self.n_contexts > 16:
                ax.locator_params(nbins=3)
            
            ax.tick_params(labelsize=tick_font_size)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            # Manage legend placement to avoid cluttering
            if self.config.show_legend and self.n_contexts <= 16:
                if context_idx == self.n_contexts - 1:
                    # Create a comprehensive legend that shows both estimated and true values
                    if probs_true is not None:
                        # Create custom legend entries for both estimated and true
                        from matplotlib.lines import Line2D
                        custom_handles = []
                        custom_labels = []
                        
                        # Add estimated line style
                        custom_handles.append(Line2D([0], [0], linestyle='--', color='black', 
                                                   linewidth=self.config.line_width))
                        custom_labels.append('Estimated')
                        
                        # Add true line style  
                        custom_handles.append(Line2D([0], [0], linestyle='-', color='black', 
                                                   linewidth=self.config.line_width * 0.75))
                        custom_labels.append('True')
                        
                        # Add color legend for target transitions
                        for target_idx in range(self.n_symbols):
                            context_color = self.colors[target_idx]
                            target_symbol = self.alphabet[target_idx]
                            custom_handles.append(Line2D([0], [0], linestyle='-', color=context_color, 
                                                       linewidth=self.config.line_width))
                            custom_labels.append(f'→ {target_symbol}')
                        
                        ax.legend(custom_handles, custom_labels, 
                                 bbox_to_anchor=(1.05, 1), loc='upper left', 
                                 fontsize=max(5, self.config.font_size - 2))
                    else:
                        # When no true values, still show color legend
                        from matplotlib.lines import Line2D
                        custom_handles = []
                        custom_labels = []
                        
                        for target_idx in range(self.n_symbols):
                            context_color = self.colors[target_idx]
                            target_symbol = self.alphabet[target_idx]
                            custom_handles.append(Line2D([0], [0], linestyle='-', color=context_color, 
                                                       linewidth=self.config.line_width))
                            custom_labels.append(f'→ {target_symbol}')
                        
                        ax.legend(custom_handles, custom_labels, 
                                 bbox_to_anchor=(1.05, 1), loc='upper left', 
                                 fontsize=max(5, self.config.font_size - 2))
        
        # Overall figure formatting
        font_size = self.config.font_size + 2 if self.config.publication_mode else self.config.font_size + 4
        self.fig_probs.suptitle(title, fontsize=font_size, fontweight='bold', y=0.98)
        
        # Improve spacing between subplots
        if self.n_contexts > 16:
            plt.tight_layout(rect=[0, 0, 0.95, 0.95], h_pad=2.0, w_pad=1.0)
        else:
            plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        
        return self.fig_probs
    
    def create_comparison_plot(self,
                             logits_estimated: np.ndarray,
                             logits_true: np.ndarray,
                             save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create publication-ready comparison plot of estimated vs true logits.
        
        Parameters
        ----------
        logits_estimated : np.ndarray
            Estimated logit parameters
        logits_true : np.ndarray
            True logit parameters
        save_path : Optional[Union[str, Path]]
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Comparison figure
        """
        # Use publication mode settings
        config = LogitVisualizationConfig(
            publication_mode=True,
            figure_size=(PUBLICATION_CONFIG['figure_width_double'] / 2.54,
                        8 / 2.54),  # Convert cm to inches
            font_size=PUBLICATION_CONFIG['font_size'],
            line_width=PUBLICATION_CONFIG['line_width']
        )
        
        # Create dashboard with publication config
        pub_dashboard = LogitEvolutionDashboard(self.alphabet, self.markov_order, config)
        
        # Generate comparison plot
        fig = pub_dashboard.plot_logit_evolution_static(
            logits_estimated, logits_true,
            title="Parameter Evolution: Estimated vs True"
        )
        
        if save_path:
            fig.savefig(save_path, dpi=PUBLICATION_CONFIG['dpi'], 
                       bbox_inches='tight', facecolor='white', edgecolor='none')
        
        return fig


def apply_block_softmax_to_trajectory(logit_trajectory: np.ndarray, 
                                    n_symbols: int) -> np.ndarray:
    """
    Apply block softmax to entire trajectory of logit parameters.
    Compatible with higher-order Markov models.
    
    Parameters
    ----------
    logit_trajectory : np.ndarray, shape (state_dim, n_sequences)
        Trajectory of logit parameters
    n_symbols : int
        Number of symbols in alphabet
        
    Returns
    -------
    np.ndarray, shape (state_dim, n_sequences)
        Trajectory of probability parameters
    """
    n_sequences = logit_trajectory.shape[1]
    prob_trajectory = np.zeros_like(logit_trajectory)
    
    for seq_idx in range(n_sequences):
        prob_trajectory[:, seq_idx] = block_softmax_viz(logit_trajectory[:, seq_idx], n_symbols)
    
    return prob_trajectory


def create_logit_evolution_summary(results_dict: Dict[str, Any],
                                 alphabet: List[str],
                                 markov_order: int = 1,
                                 save_dir: Optional[Union[str, Path]] = None) -> Dict[str, plt.Figure]:
    """
    Create comprehensive summary of logit evolution analysis.
    
    Parameters
    ----------
    results_dict : Dict[str, Any]
        Dictionary containing analysis results
    alphabet : List[str]
        Sequence alphabet
    markov_order : int
        Order of Markov model
    save_dir : Optional[Union[str, Path]]
        Directory to save figures
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of created figures
    """
    dashboard = LogitEvolutionDashboard(alphabet, markov_order)
    figures = {}
    
    # Extract data from results
    logits_estimated = results_dict.get('logits_estimated')
    logits_true = results_dict.get('logits_true')
    
    if logits_estimated is None:
        raise ValueError("results_dict must contain 'logits_estimated'")
    
    # 1. Logit evolution plot
    fig_logits = dashboard.plot_logit_evolution_static(
        logits_estimated, logits_true,
        title="Logit Parameter Evolution Analysis"
    )
    figures['logit_evolution'] = fig_logits
    
    # 2. Probability evolution plot
    probs_estimated = apply_block_softmax_to_trajectory(logits_estimated, len(alphabet))
    probs_true = None
    if logits_true is not None:
        probs_true = apply_block_softmax_to_trajectory(logits_true, len(alphabet))
    
    fig_probs = dashboard.plot_probability_evolution_static(
        probs_estimated, probs_true,
        title="Probability Evolution Analysis"
    )
    figures['probability_evolution'] = fig_probs
    
    # 3. Publication comparison (if true values available)
    if logits_true is not None:
        fig_comparison = dashboard.create_comparison_plot(logits_estimated, logits_true)
        figures['publication_comparison'] = fig_comparison
    
    # Save figures if directory provided
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            fig.savefig(save_path / f"{name}.pdf", dpi=300, bbox_inches='tight')
            fig.savefig(save_path / f"{name}.png", dpi=300, bbox_inches='tight')
            fig.savefig(save_path / f"{name}.svg", dpi=300, bbox_inches='tight')
    
    return figures


def demonstrate_logit_visualization(alphabet: Optional[List[str]] = None,
                                  markov_order: int = 2,
                                  n_sequences: int = 50) -> Dict[str, plt.Figure]:
    """
    Demonstration of logit evolution visualization capabilities.
    
    Parameters
    ----------
    alphabet : Optional[List[str]]
        Sequence alphabet, or None for default
    markov_order : int
        Order of Markov model
    n_sequences : int
        Number of sequences to simulate
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of demonstration figures
    """
    if alphabet is None:
        alphabet = ['<', 'A', 'B', 'C', '>']
    
    print(f"Demonstrating logit visualization for {len(alphabet)}-symbol alphabet, order {markov_order}")
    
    # Calculate dimensions
    n_symbols = len(alphabet)
    state_space_size = n_symbols ** (markov_order + 1)
    
    # Generate example data
    np.random.seed(42)
    
    # Simulate true logit trajectory
    logits_true = np.random.randn(state_space_size, n_sequences)
    
    # Apply constraints (set some transitions to -inf)
    # Simplified constraint example
    for i in range(0, state_space_size, n_symbols):
        # First transition in each block goes to -inf (forbidden)
        logits_true[i, :] = -np.inf
    
    # Simulate noisy estimates (what EM algorithm would produce)
    noise_level = 0.5
    logits_estimated = logits_true + noise_level * np.random.randn(state_space_size, n_sequences)
    
    # Apply same constraints to estimates
    for i in range(0, state_space_size, n_symbols):
        logits_estimated[i, :] = -np.inf
    
    # Create results dictionary
    results_dict = {
        'logits_estimated': logits_estimated,
        'logits_true': logits_true
    }
    
    # Generate visualization summary
    figures = create_logit_evolution_summary(
        results_dict, alphabet, markov_order,
        save_dir="demo_logit_evolution"
    )
    
    print(f"Generated {len(figures)} demonstration figures")
    return figures


if __name__ == "__main__":
    # Run demonstration
    demo_figures = demonstrate_logit_visualization()
    plt.show() 
