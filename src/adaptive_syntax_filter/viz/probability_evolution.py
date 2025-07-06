"""
Transition Probability Evolution Visualization.

This module provides visualization of transition probability dynamics over time,
including heatmaps, 3D surfaces, uncertainty bands, and comparative analysis.
Complements the logit evolution visualization system.

Secondary visualization module for Phase 3.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
from typing import List, Tuple, Optional, Dict, Union, Any
from pathlib import Path
from dataclasses import dataclass

# Import from our logit evolution module
from .logit_evolution import LogitVisualizationConfig, block_softmax_viz, decode_context_for_viz

# Publication settings
PUBLICATION_CONFIG = {
    'figure_width': 8.5,
    'figure_width_double': 17.8,  
    'dpi': 300,
    'font_size': 8,
    'line_width': 1.0,
    'marker_size': 4,
    'color_palette': 'viridis'
}


@dataclass
class ProbabilityVisualizationConfig:
    """Configuration for probability evolution visualization."""
    figure_size: Tuple[float, float] = (12, 8)
    dpi: int = 300
    font_size: int = 10
    line_width: float = 1.5
    alpha: float = 0.8
    colormap: str = 'viridis'
    show_uncertainty: bool = True
    confidence_level: float = 0.95
    publication_mode: bool = False


def create_transition_heatmap_series(transition_matrices: np.ndarray, 
                                   alphabet: List[str],
                                   time_points: List[int],
                                   markov_order: int = None) -> plt.Figure:
    """Create heatmap series showing syntax evolution."""
    n_timepoints = len(time_points)
    n_symbols = len(alphabet)
    
    # Auto-detect markov order if not provided
    if markov_order is None:
        # Calculate based on transition matrix size
        state_space_size = transition_matrices.shape[0]
        # For order k, state_space_size = n_symbols^(k+1)
        # So k+1 = log(state_space_size) / log(n_symbols)
        order_plus_one = np.log(state_space_size) / np.log(n_symbols)
        if not np.isclose(order_plus_one, round(order_plus_one)):
            raise ValueError(f"Cannot determine Markov order from transition matrix size {state_space_size} "
                           f"and alphabet size {n_symbols}. Please specify markov_order explicitly.")
        markov_order = int(round(order_plus_one)) - 1
        print(f"Auto-detected Markov order: {markov_order}")
    
    # Validate dimensions
    expected_state_space = n_symbols ** (markov_order + 1)
    if transition_matrices.shape[0] != expected_state_space:
        raise ValueError(f"Transition matrix size {transition_matrices.shape[0]} doesn't match "
                        f"expected size {expected_state_space} for order {markov_order} with {n_symbols} symbols. "
                        f"Expected: {n_symbols}^({markov_order}+1) = {expected_state_space}")
    
    # For higher order models, we need to be more careful about visualization
    if markov_order > 1:
        # For higher orders, create a grid showing context-based transitions
        n_contexts = n_symbols ** markov_order
        fig_width = min(20, 3 * n_timepoints)
        fig_height = min(15, 2 * min(n_contexts, 10))  # Limit to first 10 contexts
        
        fig, axes = plt.subplots(min(n_contexts, 10), n_timepoints, 
                                figsize=(fig_width, fig_height))
        if n_timepoints == 1:
            axes = axes.reshape(-1, 1)
        if min(n_contexts, 10) == 1:
            axes = axes.reshape(1, -1)
        
        # Get global min/max for consistent coloring
        vmin, vmax = np.min(transition_matrices), np.max(transition_matrices)
        
        for context_idx in range(min(n_contexts, 10)):  # Show only first 10 contexts
            for time_idx_pos, time_idx in enumerate(time_points):
                ax = axes[context_idx, time_idx_pos]
                
                # Extract transition probabilities for this context
                start_pos = context_idx * n_symbols
                end_pos = start_pos + n_symbols
                context_probs = transition_matrices[start_pos:end_pos, time_idx]
                
                # Create a horizontal bar chart for this context
                y_pos = np.arange(n_symbols)
                bars = ax.barh(y_pos, context_probs)
                
                # Color bars based on probability values
                for bar, prob in zip(bars, context_probs):
                    bar.set_color(plt.cm.viridis(prob / vmax if vmax > 0 else 0))
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(alphabet)
                ax.set_xlim([0, 1])
                
                # Add context label
                from ..data.constraint_system import decode_context
                try:
                    context_symbols = decode_context(context_idx, alphabet, markov_order)
                    context_label = "".join(context_symbols)
                except:
                    context_label = f"Ctx{context_idx}"
                
                if time_idx_pos == 0:
                    ax.set_ylabel(f'From {context_label}', fontsize=8)
                
                if context_idx == 0:
                    ax.set_title(f'Song {time_idx + 1}', fontsize=10)
                
                ax.tick_params(labelsize=7)
                
        plt.suptitle(f'Higher-Order Syntax Evolution (Order {markov_order})', 
                    fontsize=14, y=0.98)
    else:
        # Original first-order visualization
        fig, axes = plt.subplots(1, n_timepoints, figsize=(4 * n_timepoints, 4))
        if n_timepoints == 1:
            axes = [axes]
        
        vmin, vmax = np.min(transition_matrices), np.max(transition_matrices)
        
        for i, (ax, time_idx) in enumerate(zip(axes, time_points)):
            trans_matrix = transition_matrices[:, time_idx].reshape(n_symbols, n_symbols)
            im = ax.imshow(trans_matrix, cmap='viridis', vmin=vmin, vmax=vmax)
            
            ax.set_xticks(range(n_symbols))
            ax.set_yticks(range(n_symbols))
            ax.set_xticklabels(alphabet)
            ax.set_yticklabels(alphabet)
            ax.set_title(f'Song {time_idx + 1}')
            
            for j in range(n_symbols):
                for k in range(n_symbols):
                    ax.text(k, j, f'{trans_matrix[j, k]:.3f}',
                           ha='center', va='center', color='white', fontsize=8)
        
        # Create colorbar on the side instead of overlapping
        fig.colorbar(im, ax=axes, orientation='vertical', 
                     fraction=0.02, pad=0.04, label='Transition Probability')
        plt.suptitle('Syntax Evolution Across Songs', fontsize=14, y=1.05)
    
    plt.tight_layout()
    return fig


def create_3d_evolution_surface(transition_matrices: np.ndarray,
                              alphabet: List[str],
                              transition_pair: Tuple[str, str],
                              config: Optional[ProbabilityVisualizationConfig] = None) -> go.Figure:
    """
    Create 3D surface plot showing how specific transition evolves over time.
    Novel visualization not present in MATLAB code.
    
    Parameters
    ----------
    transition_matrices : np.ndarray
        Transition matrices over time
    alphabet : List[str]
        Sequence alphabet
    transition_pair : Tuple[str, str]
        (from_symbol, to_symbol) pair to visualize
    config : Optional[ProbabilityVisualizationConfig]
        Visualization configuration
        
    Returns
    -------
    go.Figure
        Plotly 3D surface figure
    """
    if config is None:
        config = ProbabilityVisualizationConfig()
    
    from_symbol, to_symbol = transition_pair
    from_idx = alphabet.index(from_symbol)
    to_idx = alphabet.index(to_symbol)
    n_symbols = len(alphabet)
    
    # Extract specific transition over time
    transition_idx = from_idx * n_symbols + to_idx
    evolution = transition_matrices[transition_idx, :]
    
    # Create time indices
    time_indices = np.arange(len(evolution))
    
    # Create 3D scatter plot with line
    fig = go.Figure()
    
    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=time_indices,
        y=[from_idx] * len(evolution),
        z=[to_idx] * len(evolution),
        mode='markers+lines',
        marker=dict(
            size=5,
            color=evolution,
            colorscale=config.colormap,
            showscale=True,
            colorbar=dict(title="Probability", titlefont=dict(size=config.font_size))
        ),
        line=dict(color='darkblue', width=6),
        name=f"'{from_symbol}' → '{to_symbol}'"
    ))
    
    # Add surface for context (optional enhancement)
    if len(evolution) > 10:  # Only for sufficient data points
        # Create a surface showing evolution context
        x_grid = np.linspace(0, len(evolution)-1, 20)
        y_grid = np.linspace(from_idx-0.5, from_idx+0.5, 5)
        z_grid = np.linspace(to_idx-0.5, to_idx+0.5, 5)
        
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.full_like(X, to_idx)
        
        # Interpolate probability values for surface coloring
        prob_surface = np.interp(X.flatten(), time_indices, evolution).reshape(X.shape)
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=prob_surface,
            colorscale=config.colormap,
            opacity=0.3,
            showscale=False,
            name="Evolution Surface"
        ))
    
    # Update layout
    fig.update_layout(
        title=f"3D Evolution: '{from_symbol}' → '{to_symbol}' Transition",
        scene=dict(
            xaxis_title="Song Number",
            yaxis_title="From Symbol Index",
            zaxis_title="To Symbol Index",
            xaxis=dict(titlefont=dict(size=config.font_size)),
            yaxis=dict(titlefont=dict(size=config.font_size)),
            zaxis=dict(titlefont=dict(size=config.font_size))
        ),
        font=dict(size=config.font_size),
        width=800,
        height=600
    )
    
    return fig


def plot_parameter_uncertainty(mean_params: np.ndarray,
                             covariance_matrices: np.ndarray,
                             alphabet: List[str],
                             config: Optional[ProbabilityVisualizationConfig] = None) -> plt.Figure:
    """
    Visualize parameter uncertainty from Kalman filtering.
    Enhanced version not implemented in original MATLAB code.
    
    Parameters
    ----------
    mean_params : np.ndarray
        Mean parameter estimates over time
    covariance_matrices : np.ndarray
        Covariance matrices over time
    alphabet : List[str]
        Sequence alphabet
    config : Optional[ProbabilityVisualizationConfig]
        Visualization configuration
        
    Returns
    -------
    plt.Figure
        Parameter uncertainty visualization
    """
    if config is None:
        config = ProbabilityVisualizationConfig()
    
    n_symbols = len(alphabet)
    n_sequences = mean_params.shape[1]
    
    # Calculate confidence intervals
    confidence_level = config.confidence_level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Create subplot grid
    fig, axes = plt.subplots(n_symbols, n_symbols, 
                            figsize=(15, 15), dpi=config.dpi)
    
    sequence_indices = np.arange(n_sequences)
    
    for from_idx in range(n_symbols):
        for to_idx in range(n_symbols):
            ax = axes[from_idx, to_idx]
            
            # Get parameter index
            param_idx = from_idx * n_symbols + to_idx
            
            # Extract mean and std
            mean_traj = mean_params[param_idx, :]
            std_traj = np.sqrt(covariance_matrices[param_idx, param_idx, :])
            
            # Plot trajectory with confidence bands
            ax.plot(sequence_indices, mean_traj, 'b-', 
                   linewidth=config.line_width, label='Mean')
            
            # Add confidence bands
            upper_bound = mean_traj + z_score * std_traj
            lower_bound = mean_traj - z_score * std_traj
            ax.fill_between(sequence_indices, lower_bound, upper_bound,
                          alpha=0.3, color='blue', 
                          label=f'{confidence_level*100:.0f}% CI')
            
            # Customize subplot
            ax.set_title(f"'{alphabet[from_idx]}' → '{alphabet[to_idx]}'",
                        fontsize=config.font_size)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=config.font_size - 2)
            
            # Add labels to edge subplots
            if from_idx == n_symbols - 1:
                ax.set_xlabel('Song Number', fontsize=config.font_size)
            if to_idx == 0:
                ax.set_ylabel('Parameter Value', fontsize=config.font_size)
            
            # Add legend to top-right subplot
            if from_idx == 0 and to_idx == n_symbols - 1:
                ax.legend(fontsize=config.font_size - 1)
    
    title_size = config.font_size + 4 if not config.publication_mode else config.font_size + 2
    plt.suptitle(f'Parameter Evolution with {confidence_level*100:.0f}% Confidence Intervals',
                fontsize=title_size)
    plt.tight_layout()
    
    return fig


def create_probability_flow_diagram(sequences: List[List[str]], 
                                  alphabet: List[str],
                                  config: Optional[ProbabilityVisualizationConfig] = None) -> go.Figure:
    """
    Create Sankey diagram showing sequence flow patterns.
    Novel visualization for understanding overall syntax patterns.
    
    Parameters
    ----------
    sequences : List[List[str]]
        Generated sequences
    alphabet : List[str]
        Sequence alphabet
    config : Optional[ProbabilityVisualizationConfig]
        Visualization configuration
        
    Returns
    -------
    go.Figure
        Sankey flow diagram
    """
    if config is None:
        config = ProbabilityVisualizationConfig()
    
    # Count transitions
    transition_counts = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            from_symbol = seq[i]
            to_symbol = seq[i + 1]
            key = (from_symbol, to_symbol)
            transition_counts[key] = transition_counts.get(key, 0) + 1
    
    # Prepare data for Sankey diagram
    symbols = list(alphabet)
    n_symbols = len(symbols)
    
    # Create node indices
    source_indices = []
    target_indices = []
    values = []
    labels = []
    
    # Add symbols to labels (nodes)
    for symbol in symbols:
        labels.append(f"From {symbol}")
    for symbol in symbols:
        labels.append(f"To {symbol}")
    
    # Add transitions (links)
    for (from_symbol, to_symbol), count in transition_counts.items():
        if count > 0:  # Only show non-zero transitions
            from_idx = symbols.index(from_symbol)
            to_idx = symbols.index(to_symbol) + n_symbols  # Offset for "to" nodes
            
            source_indices.append(from_idx)
            target_indices.append(to_idx)
            values.append(count)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="lightblue"
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color="rgba(0,100,200,0.4)"
        )
    )])
    
    fig.update_layout(
        title_text="Sequence Flow Patterns (Sankey Diagram)",
        font_size=config.font_size,
        width=1000,
        height=600
    )
    
    return fig


def create_evolution_comparison_plot(evolution_data: Dict[str, np.ndarray],
                                   alphabet: List[str],
                                   transition_pair: Tuple[str, str],
                                   config: Optional[ProbabilityVisualizationConfig] = None) -> plt.Figure:
    """
    Compare different evolution models for a specific transition.
    
    Parameters
    ----------
    evolution_data : Dict[str, np.ndarray]
        Dictionary of evolution model results
    alphabet : List[str]
        Sequence alphabet
    transition_pair : Tuple[str, str]
        Transition to compare
    config : Optional[ProbabilityVisualizationConfig]
        Visualization configuration
        
    Returns
    -------
    plt.Figure
        Evolution comparison figure
    """
    if config is None:
        config = ProbabilityVisualizationConfig()
    
    from_symbol, to_symbol = transition_pair
    from_idx = alphabet.index(from_symbol)
    to_idx = alphabet.index(to_symbol)
    n_symbols = len(alphabet)
    transition_idx = from_idx * n_symbols + to_idx
    
    fig, ax = plt.subplots(figsize=config.figure_size, dpi=config.dpi)
    
    # Plot each evolution model
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(evolution_data)))
    
    for i, (model_name, data) in enumerate(evolution_data.items()):
        if transition_idx < data.shape[0]:
            evolution = data[transition_idx, :]
            sequence_indices = np.arange(len(evolution))
            
            ax.plot(sequence_indices, evolution, 
                   color=colors[i], linewidth=config.line_width,
                   label=model_name.replace('_', ' ').title(),
                   alpha=config.alpha)
    
    # Customize plot
    ax.set_xlabel('Song Number', fontsize=config.font_size)
    ax.set_ylabel('Transition Probability', fontsize=config.font_size)
    ax.set_title(f"Evolution Model Comparison: '{from_symbol}' → '{to_symbol}'",
                fontsize=config.font_size + 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=config.font_size)
    ax.tick_params(labelsize=config.font_size - 1)
    
    plt.tight_layout()
    return fig


class ProbabilityEvolutionAnalyzer:
    """Analyzer for probability evolution patterns."""
    
    def __init__(self, alphabet: List[str], markov_order: int = 1):
        self.alphabet = alphabet
        self.n_symbols = len(alphabet)
        self.markov_order = markov_order
        self.n_contexts = self.n_symbols ** markov_order
        self.state_space_size = self.n_contexts * self.n_symbols
    
    def analyze_evolution_patterns(self, transition_matrices: np.ndarray) -> Dict[str, Any]:
        """Analyze evolution patterns in transition matrices."""
        variability = np.var(transition_matrices, axis=1)
        
        # Find significant transitions
        significant_transitions = []
        for i in range(transition_matrices.shape[0]):
            if np.var(transition_matrices[i, :]) > np.mean(variability):
                # For higher-order Markov models, we need to handle the state indexing properly
                if self.markov_order == 1:
                    # Simple case: i maps to (from_symbol, to_symbol)
                    from_idx = i // self.n_symbols
                    to_idx = i % self.n_symbols
                    if from_idx < self.n_symbols and to_idx < self.n_symbols:
                        significant_transitions.append((
                            self.alphabet[from_idx], 
                            self.alphabet[to_idx],
                            np.var(transition_matrices[i, :])
                        ))
                else:
                    # Higher-order case: more complex state mapping
                    context_idx = i // self.n_symbols
                    to_idx = i % self.n_symbols
                    
                    if context_idx < self.n_contexts and to_idx < self.n_symbols:
                        # Convert context index back to symbol sequence
                        context_symbols = []
                        temp_idx = context_idx
                        for _ in range(self.markov_order):
                            context_symbols.insert(0, self.alphabet[temp_idx % self.n_symbols])
                            temp_idx //= self.n_symbols
                        
                        context_str = ''.join(context_symbols)
                        to_symbol = self.alphabet[to_idx]
                        
                        significant_transitions.append((
                            context_str, 
                            to_symbol,
                            np.var(transition_matrices[i, :])
                        ))
        
        return {
            'parameter_variability': {
                'mean_variability': np.mean(variability),
                'max_variability': np.max(variability)
            },
            'significant_transitions': sorted(significant_transitions, 
                                            key=lambda x: x[2], reverse=True)
        }


def demonstrate_probability_visualization(alphabet: Optional[List[str]] = None,
                                        n_sequences: int = 100) -> Dict[str, Any]:
    """
    Demonstration of probability evolution visualization capabilities.
    
    Parameters
    ----------
    alphabet : Optional[List[str]]
        Sequence alphabet, or None for default
    n_sequences : int
        Number of sequences to simulate
        
    Returns
    -------
    Dict[str, Any]
        Demonstration results with figures
    """
    if alphabet is None:
        alphabet = ['<', 'A', 'B', 'C', '>']
    
    print(f"Demonstrating probability visualization for {len(alphabet)}-symbol alphabet")
    
    # Generate example data
    n_symbols = len(alphabet)
    n_params = n_symbols ** 2
    
    np.random.seed(42)
    
    # Simulate probability evolution
    base_probs = np.random.dirichlet([1] * n_symbols, n_params).T.flatten()
    
    # Add temporal variation
    time_variation = np.sin(np.linspace(0, 4*np.pi, n_sequences)) * 0.1
    transition_matrices = np.zeros((n_params, n_sequences))
    
    for t in range(n_sequences):
        # Apply block-wise normalization
        for block in range(n_symbols):
            start_idx = block * n_symbols
            end_idx = (block + 1) * n_symbols
            block_probs = base_probs[start_idx:end_idx]
            
            # Add variation and renormalize
            varied_probs = block_probs + time_variation[t] * np.random.randn(n_symbols) * 0.05
            varied_probs = np.maximum(varied_probs, 0.01)  # Ensure positive
            varied_probs = varied_probs / np.sum(varied_probs)  # Normalize
            
            transition_matrices[start_idx:end_idx, t] = varied_probs
    
    # Create analyzer and generate report
    analyzer = ProbabilityEvolutionAnalyzer(alphabet)
    report = analyzer.analyze_evolution_patterns(transition_matrices)
    
    print(f"Generated probability evolution analysis")
    print(f"Found {len(report['significant_transitions'])} significant transitions")
    
    return report


if __name__ == "__main__":
    # Run demonstration
    demo_report = demonstrate_probability_visualization()
    plt.show() 
