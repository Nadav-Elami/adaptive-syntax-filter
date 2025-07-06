"""Global random seed management for reproducible research across machines."""

import random
import numpy as np
from typing import Optional, Dict, Any
import os
import hashlib

# Global random state storage
_GLOBAL_SEED: Optional[int] = None
_RNG_STATE: Optional[Dict[str, Any]] = None

def set_global_seed(seed: int) -> None:
    """Set global random seed for all random number generators.
    
    Ensures reproducible results across different machines and platforms.
    Sets seeds for Python's random, NumPy, and any other RNG libraries.
    
    Parameters
    ----------
    seed : int
        Random seed value for reproducibility
        
    Examples
    --------
    >>> set_global_seed(42)
    >>> # All subsequent random operations will be reproducible
    """
    global _GLOBAL_SEED, _RNG_STATE
    
    _GLOBAL_SEED = seed
    
    # Set Python's built-in random module
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Store the initial state for reference
    _RNG_STATE = {
        'seed': seed,
        'python_state': random.getstate(),
        'numpy_state': np.random.get_state()
    }

def get_global_seed() -> Optional[int]:
    """Get the current global random seed.
    
    Returns
    -------
    Optional[int]
        Current global seed, or None if not set
    """
    return _GLOBAL_SEED

def get_random_state() -> Optional[Dict[str, Any]]:
    """Get current random number generator states.
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing RNG states, or None if not initialized
    """
    return _RNG_STATE

def create_deterministic_seed(base_string: str) -> int:
    """Create a deterministic seed from a string.
    
    Useful for creating reproducible seeds from experiment names,
    configuration hashes, or other identifiers.
    
    Parameters
    ----------
    base_string : str
        String to hash for seed generation
        
    Returns
    -------
    int
        Deterministic seed value
        
    Examples
    --------
    >>> seed = create_deterministic_seed("experiment_v1.2")
    >>> set_global_seed(seed)
    """
    # Use SHA-256 hash for deterministic seed generation
    hash_object = hashlib.sha256(base_string.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert first 8 hex characters to integer
    seed = int(hash_hex[:8], 16)
    
    # Ensure seed is within valid range for most RNGs
    return seed % (2**31 - 1)

def reset_random_state() -> None:
    """Reset all random number generators to their initial states.
    
    Only works if set_global_seed() was called previously.
    """
    global _RNG_STATE
    
    if _RNG_STATE is None:
        raise RuntimeError("Random state not initialized. Call set_global_seed() first.")
    
    # Restore Python's random state
    random.setstate(_RNG_STATE['python_state'])
    
    # Restore NumPy's random state
    np.random.set_state(_RNG_STATE['numpy_state'])

def get_environment_seed() -> int:
    """Get seed from environment variable if available.
    
    Checks for ADAPTIVE_SYNTAX_FILTER_SEED environment variable.
    
    Returns
    -------
    int
        Seed from environment, or a default value if not set
    """
    env_seed = os.environ.get('ADAPTIVE_SYNTAX_FILTER_SEED')
    
    if env_seed is not None:
        try:
            return int(env_seed)
        except ValueError:
            # If environment variable is not a valid integer,
            # create deterministic seed from the string value
            return create_deterministic_seed(env_seed)
    
    # Default seed for reproducibility
    return 42

def ensure_reproducibility() -> int:
    """Ensure reproducible random state is set.
    
    Sets global seed if not already set, using environment variable
    or default value.
    
    Returns
    -------
    int
        The seed that was set
    """
    if _GLOBAL_SEED is None:
        seed = get_environment_seed()
        set_global_seed(seed)
        return seed
    return _GLOBAL_SEED

# Automatically ensure reproducibility when module is imported
# This can be disabled by setting environment variable to empty string
if os.environ.get('ADAPTIVE_SYNTAX_FILTER_SEED') != '':
    ensure_reproducibility() 
