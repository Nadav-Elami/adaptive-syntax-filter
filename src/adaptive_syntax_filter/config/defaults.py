"""Default configuration parameters for different bird species and research scenarios."""

from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class DefaultConfig:
    """Base configuration structure for bird song analysis."""
    
    # Data scale parameters
    alphabet_size: int
    n_songs: int
    song_length_range: Tuple[int, int]
    
    # Model parameters
    markov_order: int
    evolution_model: str
    
    # Algorithm parameters
    max_em_iterations: int
    convergence_threshold: float
    regularization_lambda: float
    
    # Processing parameters
    batch_size: int
    memory_limit_gb: float
    
    # Visualization parameters
    figure_dpi: int
    export_formats: List[str]


# Bengalese Finch Configuration (~15 letters, ~300 songs)
BENGALESE_FINCH_CONFIG = DefaultConfig(
    # Data scale - considered "very large" 
    alphabet_size=15,
    n_songs=300,
    song_length_range=(20, 80),  # Typical phrase sequence lengths
    
    # Model complexity
    markov_order=2,  # 2nd order captures phrase dependencies well
    evolution_model="sigmoid",  # Gradual syntax changes
    
    # Algorithm settings
    max_em_iterations=50,
    convergence_threshold=1e-4,
    regularization_lambda=1e-3,
    
    # Processing efficiency
    batch_size=50,  # Process in smaller batches
    memory_limit_gb=1.0,  # Conservative memory usage
    
    # Visualization
    figure_dpi=300,
    export_formats=["pdf", "png"]
)

# Canary Configuration (~40 letters, ~10,000 songs)
CANARY_CONFIG = DefaultConfig(
    # Data scale - large research datasets
    alphabet_size=40,
    n_songs=10000,
    song_length_range=(30, 120),  # More complex phrase sequences
    
    # Model complexity
    markov_order=3,  # Higher order for complex syntax
    evolution_model="piecewise",  # Stepped syntax evolution
    
    # Algorithm settings
    max_em_iterations=100,  # More iterations for larger datasets
    convergence_threshold=1e-5,  # Tighter convergence
    regularization_lambda=1e-4,  # Less regularization for large data
    
    # Processing efficiency  
    batch_size=500,  # Larger batches for efficiency
    memory_limit_gb=2.0,  # Higher memory allowance
    
    # Visualization
    figure_dpi=300,
    export_formats=["pdf", "svg", "png"]
)

# Research scenario configurations
RESEARCH_CONFIGS = {
    "bengalese_finch": BENGALESE_FINCH_CONFIG,
    "canary": CANARY_CONFIG,
    "minimal": DefaultConfig(
        alphabet_size=5,
        n_songs=50,
        song_length_range=(10, 30),
        markov_order=1,
        evolution_model="constant",
        max_em_iterations=20,
        convergence_threshold=1e-3,
        regularization_lambda=1e-2,
        batch_size=10,
        memory_limit_gb=0.5,
        figure_dpi=150,
        export_formats=["png"]
    )
}

# Evolution model options
EVOLUTION_MODELS = [
    "linear",      # Linear parameter drift
    "exponential", # Exponential decay/growth
    "sigmoid",     # S-curve transitions
    "piecewise",   # Step-wise changes
    "oscillatory", # Periodic variations
    "constant"     # No evolution (baseline)
]

# Alphabet size constraints
MIN_ALPHABET_SIZE = 3  # Minimum: start, one phrase, end
MAX_ALPHABET_SIZE = 100  # Practical computational limit
RECOMMENDED_MAX_ALPHABET = 50  # For good performance

# Memory and performance guidelines
MEMORY_PER_SONG_MB = 0.1  # Approximate memory per song (MB)
SONGS_PER_GB = 10000  # Rough songs per GB of memory

def get_memory_estimate(config: DefaultConfig) -> float:
    """Estimate memory usage in GB for given configuration."""
    state_space_size = config.alphabet_size ** (config.markov_order + 1)
    song_memory = config.n_songs * MEMORY_PER_SONG_MB / 1024  # GB
    state_memory = state_space_size * 8 * 1e-9  # 8 bytes per float, convert to GB
    return song_memory + state_memory

def validate_config(config: DefaultConfig) -> List[str]:
    """Validate configuration parameters and return list of warnings."""
    warnings = []
    
    if config.alphabet_size < MIN_ALPHABET_SIZE:
        warnings.append(f"Alphabet size {config.alphabet_size} is below minimum {MIN_ALPHABET_SIZE}")
    
    if config.alphabet_size > RECOMMENDED_MAX_ALPHABET:
        warnings.append(f"Alphabet size {config.alphabet_size} may cause performance issues")
    
    if config.markov_order > 5:
        warnings.append(f"Markov order {config.markov_order} is very high, consider computational cost")
    
    if config.evolution_model not in EVOLUTION_MODELS:
        warnings.append(f"Evolution model '{config.evolution_model}' not recognized")
    
    estimated_memory = get_memory_estimate(config)
    if estimated_memory > config.memory_limit_gb:
        warnings.append(f"Estimated memory {estimated_memory:.2f}GB exceeds limit {config.memory_limit_gb}GB")
    
    return warnings 
