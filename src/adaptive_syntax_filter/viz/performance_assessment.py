"""
Model Performance Assessment Visualization.

Enhanced version of MATLAB evaluate_fit.m with additional metrics, 
comprehensive visualizations, and publication-ready output.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Tuple, Optional, Dict, Union, Any
from pathlib import Path
from dataclasses import dataclass

# Publication settings
plt.style.use('seaborn-v0_8-paper')

@dataclass
class ModelEvaluationMetrics:
    """Container for all model evaluation metrics."""
    rmse: float
    r_squared: float
    correlation_coefficient: float
    kl_divergence: float
    mae: float


@dataclass
class PerformanceVisualizationConfig:
    """Configuration for performance assessment visualization."""
    figure_size: Tuple[float, float] = (16, 12)
    dpi: int = 300
    font_size: int = 10
    publication_mode: bool = False


def evaluate_model_fit(predicted_probs: np.ndarray, 
                      true_probs: np.ndarray) -> ModelEvaluationMetrics:
    """Enhanced model fit evaluation with comprehensive metrics."""
    assert predicted_probs.shape == true_probs.shape
    
    # Core metrics
    mse = np.mean((predicted_probs - true_probs) ** 2)
    rmse = np.sqrt(mse) * 100
    
    # Additional metrics
    mae = np.mean(np.abs(predicted_probs - true_probs))
    corr_coef = np.corrcoef(predicted_probs.flatten(), true_probs.flatten())[0, 1]
    
    # R-squared
    ss_res = np.sum((true_probs - predicted_probs) ** 2)
    ss_tot = np.sum((true_probs - np.mean(true_probs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # KL divergence (with numerical stability)
    epsilon = 1e-10
    kl_div = np.sum(predicted_probs * np.log((predicted_probs + epsilon) / (true_probs + epsilon)))
    
    return ModelEvaluationMetrics(
        rmse=rmse, r_squared=r_squared, correlation_coefficient=corr_coef,
        kl_divergence=kl_div, mae=mae
    )


def create_fit_evaluation_plots(predicted_probs: np.ndarray,
                               true_probs: np.ndarray,
                               metrics: ModelEvaluationMetrics) -> plt.Figure:
    """Create comprehensive fit evaluation plots."""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Scatter plot: predicted vs true
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(true_probs.flatten(), predicted_probs.flatten(), alpha=0.6, s=10)
    min_val = min(np.min(true_probs), np.min(predicted_probs))
    max_val = max(np.max(true_probs), np.max(predicted_probs))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('True Probabilities')
    plt.ylabel('Predicted Probabilities')
    plt.title(f'Predicted vs True\nR² = {metrics.r_squared:.3f}')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    ax2 = plt.subplot(2, 3, 2)
    residuals = predicted_probs.flatten() - true_probs.flatten()
    plt.scatter(true_probs.flatten(), residuals, alpha=0.6, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Probabilities')
    plt.ylabel('Residuals')
    plt.title(f'Residuals\nRMSE = {metrics.rmse:.3f}%')
    plt.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=50, alpha=0.7, density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title(f'Error Distribution\nMAE = {metrics.mae:.3f}')
    plt.grid(True, alpha=0.3)
    
    # 4. Metrics summary
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    metrics_text = f"""Model Evaluation Metrics:

RMSE: {metrics.rmse:.3f}%
R²: {metrics.r_squared:.3f}
Correlation: {metrics.correlation_coefficient:.3f}
KL Divergence: {metrics.kl_divergence:.3f}
MAE: {metrics.mae:.3f}
"""
    ax4.text(0.1, 0.8, metrics_text, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def create_convergence_analysis(convergence_history: List[float], 
                              likelihood_history: List[float],
                              config: Optional[PerformanceVisualizationConfig] = None) -> plt.Figure:
    """
    Create convergence analysis plots.
    
    Parameters
    ----------
    convergence_history : List[float]
        JS divergence convergence measures
    likelihood_history : List[float]
        Log likelihood evolution
    config : Optional[PerformanceVisualizationConfig]
        Visualization configuration
        
    Returns
    -------
    plt.Figure
        Convergence analysis figure
    """
    if config is None:
        config = PerformanceVisualizationConfig()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=config.dpi)
    
    # Convergence measure (JS divergence)
    iterations = range(1, len(convergence_history) + 1)
    ax1.plot(iterations, convergence_history, 'b-', linewidth=2)
    ax1.set_xlabel('EM Iteration', fontsize=config.font_size)
    ax1.set_ylabel('JS Divergence', fontsize=config.font_size)
    ax1.set_title('Convergence Measure', fontsize=config.font_size)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.tick_params(labelsize=config.font_size - 1)
    
    # Log likelihood evolution
    ax2.plot(iterations, likelihood_history, 'r-', linewidth=2)
    ax2.set_xlabel('EM Iteration', fontsize=config.font_size)
    ax2.set_ylabel('Expected Log Likelihood', fontsize=config.font_size)
    ax2.set_title('Model Likelihood', fontsize=config.font_size)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=config.font_size - 1)
    
    plt.tight_layout()
    return fig


def create_parameter_recovery_analysis(true_params: np.ndarray,
                                     estimated_params: np.ndarray,
                                     alphabet: List[str],
                                     config: Optional[PerformanceVisualizationConfig] = None) -> plt.Figure:
    """
    Analyze parameter recovery across different transitions.
    
    Parameters
    ----------
    true_params : np.ndarray
        True parameter values
    estimated_params : np.ndarray
        Estimated parameter values
    alphabet : List[str]
        Sequence alphabet
    config : Optional[PerformanceVisualizationConfig]
        Visualization configuration
        
    Returns
    -------
    plt.Figure
        Parameter recovery analysis figure
    """
    if config is None:
        config = PerformanceVisualizationConfig()
    
    n_symbols = len(alphabet)
    n_params = n_symbols ** 2
    
    # Calculate recovery metrics for each parameter
    recovery_metrics = []
    for i in range(n_params):
        from_idx = i // n_symbols
        to_idx = i % n_symbols
        
        true_vals = true_params[i, :]
        est_vals = estimated_params[i, :]
        
        # Calculate metrics
        correlation = np.corrcoef(true_vals, est_vals)[0, 1] if len(true_vals) > 1 else 0
        rmse = np.sqrt(np.mean((true_vals - est_vals) ** 2))
        bias = np.mean(est_vals - true_vals)
        
        recovery_metrics.append({
            'from_symbol': alphabet[from_idx],
            'to_symbol': alphabet[to_idx],
            'correlation': correlation,
            'rmse': rmse,
            'bias': bias
        })
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=config.dpi)
    
    # Extract metrics for plotting
    correlations = [m['correlation'] for m in recovery_metrics]
    rmses = [m['rmse'] for m in recovery_metrics]
    biases = [m['bias'] for m in recovery_metrics]
    
    # 1. Correlation heatmap
    corr_matrix = np.array(correlations).reshape(n_symbols, n_symbols)
    im1 = axes[0].imshow(corr_matrix, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_xticks(range(n_symbols))
    axes[0].set_yticks(range(n_symbols))
    axes[0].set_xticklabels(alphabet)
    axes[0].set_yticklabels(alphabet)
    axes[0].set_title('Parameter Recovery Correlation')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # 2. RMSE heatmap
    rmse_matrix = np.array(rmses).reshape(n_symbols, n_symbols)
    im2 = axes[1].imshow(rmse_matrix, cmap='Reds')
    axes[1].set_xticks(range(n_symbols))
    axes[1].set_yticks(range(n_symbols))
    axes[1].set_xticklabels(alphabet)
    axes[1].set_yticklabels(alphabet)
    axes[1].set_title('Parameter Recovery RMSE')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # 3. Bias heatmap
    bias_matrix = np.array(biases).reshape(n_symbols, n_symbols)
    im3 = axes[2].imshow(bias_matrix, cmap='RdBu_r', 
                        vmin=-np.max(np.abs(bias_matrix)), 
                        vmax=np.max(np.abs(bias_matrix)))
    axes[2].set_xticks(range(n_symbols))
    axes[2].set_yticks(range(n_symbols))
    axes[2].set_xticklabels(alphabet)
    axes[2].set_yticklabels(alphabet)
    axes[2].set_title('Parameter Recovery Bias')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    return fig


def create_model_comparison_plot(model_results: Dict[str, ModelEvaluationMetrics],
                               config: Optional[PerformanceVisualizationConfig] = None) -> plt.Figure:
    """
    Compare performance across different models or conditions.
    
    Parameters
    ----------
    model_results : Dict[str, ModelEvaluationMetrics]
        Results for different models
    config : Optional[PerformanceVisualizationConfig]
        Visualization configuration
        
    Returns
    -------
    plt.Figure
        Model comparison figure
    """
    if config is None:
        config = PerformanceVisualizationConfig()
    
    model_names = list(model_results.keys())
    n_models = len(model_names)
    
    # Extract metrics
    metrics_names = ['rmse', 'r_squared', 'correlation_coefficient', 
                    'kl_divergence', 'mae']
    metrics_values = {name: [] for name in metrics_names}
    
    for model_name in model_names:
        metrics = model_results[model_name]
        for metric_name in metrics_names:
            metrics_values[metric_name].append(getattr(metrics, metric_name))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=config.dpi)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    for i, metric_name in enumerate(metrics_names):
        ax = axes[i]
        values = metrics_values[metric_name]
        
        bars = ax.bar(model_names, values, color=colors)
        ax.set_title(metric_name.replace('_', ' ').title(), fontsize=config.font_size)
        ax.tick_params(axis='x', rotation=45, labelsize=config.font_size - 2)
        ax.tick_params(axis='y', labelsize=config.font_size - 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom',
                   fontsize=config.font_size - 2)
    
    plt.tight_layout()
    return fig


class PerformanceAnalyzer:
    """Performance analysis and visualization."""
    
    def __init__(self):
        pass
    
    def analyze_performance(self, predicted_probs: np.ndarray,
                          true_probs: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        metrics = evaluate_model_fit(predicted_probs, true_probs)
        fig = create_fit_evaluation_plots(predicted_probs, true_probs, metrics)
        
        return {
            'metrics': metrics,
            'figure': fig
        }


def demonstrate_performance_assessment(alphabet: Optional[List[str]] = None,
                                     n_sequences: int = 100) -> Dict[str, Any]:
    """
    Demonstration of performance assessment capabilities.
    
    Parameters
    ----------
    alphabet : Optional[List[str]]
        Sequence alphabet
    n_sequences : int
        Number of sequences to simulate
        
    Returns
    -------
    Dict[str, Any]
        Demonstration results
    """
    if alphabet is None:
        alphabet = ['<', 'A', 'B', 'C', '>']
    
    print(f"Demonstrating performance assessment for {len(alphabet)}-symbol alphabet")
    
    # Generate synthetic data
    n_symbols = len(alphabet)
    n_params = n_symbols ** 2
    
    np.random.seed(42)
    
    # True probabilities (normalized blocks)
    true_probs = np.zeros((n_params, n_sequences))
    for i in range(0, n_params, n_symbols):
        for j in range(n_sequences):
            block_probs = np.random.dirichlet([1] * n_symbols)
            true_probs[i:i+n_symbols, j] = block_probs
    
    # Predicted probabilities (with noise)
    noise_level = 0.1
    predicted_probs = true_probs + noise_level * np.random.randn(n_params, n_sequences)
    
    # Renormalize blocks
    for i in range(0, n_params, n_symbols):
        for j in range(n_sequences):
            block_sum = np.sum(predicted_probs[i:i+n_symbols, j])
            if block_sum > 0:
                predicted_probs[i:i+n_symbols, j] /= block_sum
    
    # Generate convergence data
    convergence_history = [0.1 * np.exp(-i/10) for i in range(50)]
    likelihood_history = [-100 + 90 * (1 - np.exp(-i/15)) for i in range(50)]
    
    # Perform comprehensive analysis
    analyzer = PerformanceAnalyzer()
    results = analyzer.analyze_performance(predicted_probs, true_probs)
    
    print(f"Performance Analysis Results:")
    print(f"  RMSE: {results['metrics'].rmse:.3f}%")
    print(f"  R²: {results['metrics'].r_squared:.3f}")
    print(f"  Correlation: {results['metrics'].correlation_coefficient:.3f}")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    demo_results = demonstrate_performance_assessment()
    plt.show() 
