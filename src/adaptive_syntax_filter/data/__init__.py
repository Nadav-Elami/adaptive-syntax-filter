"""Data generation system for adaptive syntax filter.

This package provides comprehensive tools for generating synthetic canary song
sequences using higher-order Markov models with temporal parameter evolution.

Key Components
--------------
- Constraint system: Enforces canary song grammar rules
- Temporal evolution: Six evolution models for parameter changes
- Sequence generation: Multi-order Markov sequence creation
- Alphabet management: Dynamic alphabet sizing and optimization
- Dataset building: Batch dataset creation and validation

Examples
--------
>>> from adaptive_syntax_filter.data import create_preset_alphabets, create_standard_configs
>>> from adaptive_syntax_filter.data import DatasetBuilder, SequenceGenerator
>>> 
>>> # Create dataset builder
>>> builder = DatasetBuilder()
>>> 
>>> # Generate bengalese finch scale dataset
>>> dataset = builder.create_from_preset('bengalese_finch')
>>> print(f"Generated {len(dataset.sequences)} sequences")
"""

# Core modules
from .constraint_system import (
    ConstraintManager,
    create_constraint_manager,
    encode_context,
    decode_context,
    get_constraint_positions_higher_order,
    apply_constraints_to_logits,
    validate_constraint_setup,
    analyze_constraint_structure
)

from .temporal_evolution import (
    EvolutionManager,
    EvolutionConfig, 
    compute_evolution_trajectory,
    get_batch_evolution_schedule,
    create_evolution_examples,
    validate_evolution_parameters
)

from .sequence_generator import (
    SequenceGenerator,
    GenerationConfig,
    generate_dataset,
    generate_single_sequence,
    generate_sequence_batch,
    initialize_generation_parameters,
    softmax_mc_higher_order,
    validate_generated_sequences,
    create_standard_configs,
    sequences_to_observations
)

from .alphabet_manager import (
    AlphabetManager,
    AlphabetStats,
    create_standard_alphabet,
    create_preset_alphabets,
    validate_alphabet,
    estimate_memory_requirements,
    get_recommended_order_limits,
    analyze_alphabet_scaling,
    compare_alphabets,
    optimize_alphabet_for_constraints
)

from .dataset_builder import (
    DatasetBuilder,
    Dataset,
    DatasetMetadata,
    analyze_sequence_statistics,
    validate_dataset_quality,
    create_research_datasets
)

# Convenience imports for common workflows
__all__ = [
    # Core classes
    'ConstraintManager',
    'EvolutionManager', 
    'SequenceGenerator',
    'AlphabetManager',
    'DatasetBuilder',
    
    # Configuration classes
    'EvolutionConfig',
    'GenerationConfig',
    'AlphabetStats',
    'Dataset',
    'DatasetMetadata',
    
    # Generation functions
    'generate_dataset',
    'generate_single_sequence',
    'generate_sequence_batch',
    'softmax_mc_higher_order',
    'sequences_to_observations',
    
    # Evolution functions
    'compute_evolution_trajectory',
    'get_batch_evolution_schedule',
    
    # Constraint functions
    'encode_context',
    'decode_context',
    'get_constraint_positions_higher_order',
    'apply_constraints_to_logits',
    
    # Alphabet functions
    'create_standard_alphabet',
    'validate_alphabet',
    'estimate_memory_requirements',
    'get_recommended_order_limits',
    
    # Analysis and validation
    'validate_generated_sequences',
    'validate_constraint_setup',
    'validate_dataset_quality',
    'analyze_sequence_statistics',
    'analyze_constraint_structure',
    'analyze_alphabet_scaling',
    'compare_alphabets',
    
    # Factory/preset functions
    'create_preset_alphabets',
    'create_standard_configs',
    'create_evolution_examples',
    'create_constraint_manager',
    'create_research_datasets',
    'optimize_alphabet_for_constraints',
    
    # Initialization
    'initialize_generation_parameters'
] 
