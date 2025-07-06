"""
Adaptive Syntax Filter Visualization Module.

This module provides comprehensive visualization capabilities for the adaptive
Kalman-EM algorithm analyzing canary song syntax evolution. Includes real-time
monitoring, publication-quality figures, and scientific analysis tools.

Modules:
- logit_evolution: Priority #1 logit parameter tracking
- probability_evolution: Transition probability dynamics  
- performance_assessment: Model fit evaluation metrics
- sequence_analysis: Song structure pattern analysis
- publication_figures: Automated figure generation pipeline
"""

# Core visualization classes
from .logit_evolution import (
    LogitEvolutionDashboard,
    LogitVisualizationConfig,
    block_softmax_viz,
    apply_block_softmax_to_trajectory,
    create_logit_evolution_summary
)

from .probability_evolution import (
    ProbabilityEvolutionAnalyzer,
    ProbabilityVisualizationConfig,
    create_transition_heatmap_series
)

from .performance_assessment import (
    PerformanceAnalyzer,
    ModelEvaluationMetrics,
    evaluate_model_fit,
    create_fit_evaluation_plots
)

from .sequence_analysis import (
    SequenceAnalyzer,
    analyze_sequence_lengths,
    analyze_symbol_usage,
    create_sequence_length_plot,
    create_symbol_usage_plot
)

from .publication_figures import (
    PublicationFigureManager,
    setup_publication_style,
    create_main_results_figure
)

# Export all public classes and functions
__all__ = [
    # Logit evolution (Priority #1)
    'LogitEvolutionDashboard',
    'LogitVisualizationConfig', 
    'block_softmax_viz',
    'apply_block_softmax_to_trajectory',
    'create_logit_evolution_summary',
    
    # Probability evolution
    'ProbabilityEvolutionAnalyzer',
    'ProbabilityVisualizationConfig',
    'create_transition_heatmap_series',
    
    # Performance assessment
    'PerformanceAnalyzer',
    'ModelEvaluationMetrics',
    'evaluate_model_fit',
    'create_fit_evaluation_plots',
    
    # Sequence analysis
    'SequenceAnalyzer',
    'analyze_sequence_lengths',
    'analyze_symbol_usage',
    'create_sequence_length_plot',
    'create_symbol_usage_plot',
    
    # Publication figures
    'PublicationFigureManager',
    'setup_publication_style',
    'create_main_results_figure'
] 
