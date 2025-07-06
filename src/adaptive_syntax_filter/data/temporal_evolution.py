"""Temporal evolution models for adaptive syntax parameter changes.

Implements the six evolution models from data_generation.md section 2.3:
linear, exponential, sigmoid, piecewise, oscillatory, and constant evolution
with support for both per-sequence and batch-wise parameter changes.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import warnings


@dataclass
class EvolutionConfig:
    """Configuration for temporal evolution models.
    
    Attributes
    ----------
    evolution_type : str
        Type: 'linear', 'exponential', 'sigmoid', 'piecewise', 'oscillatory', 'constant'
    batch_size : Optional[int]
        If provided, parameters evolve per batch instead of per sequence
    evolution_params : Dict[str, Any]
        Additional parameters specific to evolution type
    """
    evolution_type: str
    batch_size: Optional[int] = None
    evolution_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.evolution_params is None:
            self.evolution_params = {}


def compute_evolution_trajectory(x_init: np.ndarray,
                               x_final: np.ndarray, 
                               n_sequences: int,
                               evolution_type: str,
                               batch_size: Optional[int] = None,
                               **evolution_params) -> np.ndarray:
    """
    Compute temporal evolution trajectory for parameter vectors.
    
    Implements all six evolution models from data_generation.md with support
    for both per-sequence and batch-wise evolution patterns.
    
    Parameters
    ----------
    x_init, x_final : np.ndarray
        Initial and final logit vectors
    n_sequences : int
        Total number of sequences to generate
    evolution_type : str
        'linear' | 'exponential' | 'sigmoid' | 'piecewise' | 'oscillatory' | 'constant'
    batch_size : int, optional
        If provided, parameters evolve per batch instead of per sequence
    evolution_params : dict
        Additional parameters specific to evolution type
        
    Returns
    -------
    np.ndarray, shape (state_size, n_sequences)
        Parameter trajectory for each sequence
    """
    if len(x_init) != len(x_final):
        raise ValueError("Initial and final vectors must have same length")
    if n_sequences < 1:
        raise ValueError("Number of sequences must be positive")
    if evolution_type not in ['linear', 'exponential', 'sigmoid', 'piecewise', 
                             'oscillatory', 'constant']:
        raise ValueError(f"Unknown evolution type: {evolution_type}")
    
    if batch_size is not None:
        return _compute_batched_evolution(x_init, x_final, n_sequences, 
                                        evolution_type, batch_size, **evolution_params)
    else:
        return _compute_sequence_evolution(x_init, x_final, n_sequences, 
                                         evolution_type, **evolution_params)


def _compute_sequence_evolution(x_init: np.ndarray, 
                              x_final: np.ndarray, 
                              n_sequences: int, 
                              evolution_type: str, 
                              **params) -> np.ndarray:
    """Compute evolution parameters for each individual sequence."""
    state_size = len(x_init)
    trajectory = np.zeros((state_size, n_sequences))
    
    # Create time vector
    t = np.linspace(0, 1, n_sequences) if n_sequences > 1 else np.array([0])
    
    if evolution_type == 'linear':
        # Linear interpolation: x(t) = x_init + t * (x_final - x_init)
        # Handle infinite values carefully
        with np.errstate(invalid='ignore'):
            # Compute difference, handling infinities
            diff = x_final - x_init
            # Set NaN and infinite differences to 0 (no evolution for constrained positions)
            diff = np.where(np.isfinite(diff), diff, 0.0)
            
            for i, time_point in enumerate(t):
                trajectory[:, i] = x_init + time_point * diff
            
    elif evolution_type == 'exponential':
        # Exponential evolution: x(t) = x_init * exp(rate * t * log(x_final/x_init))
        rate = params.get('rate', 2.0)
        
        # Handle special cases and numerical stability
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # Compute ratio, handling zeros and infinities
            # For exponential evolution, we need to be more careful with -inf and finite values
            finite_mask = np.isfinite(x_init) & np.isfinite(x_final)
            inf_mask = (x_init == -np.inf) | (x_final == -np.inf)
            
            # Initialize log_ratio
            log_ratio = np.zeros_like(x_init)
            
            # For finite values, compute log ratio safely
            finite_indices = finite_mask & (x_init != 0)
            if np.any(finite_indices):
                safe_ratio = x_final[finite_indices] / x_init[finite_indices]
                log_ratio[finite_indices] = np.where(safe_ratio > 0, np.log(safe_ratio), 0.0)
            
            # For infinite values, set log_ratio to 0 (no evolution)
            log_ratio[inf_mask] = 0.0
        
        for i, time_point in enumerate(t):
            exp_factor = np.exp(rate * time_point * log_ratio)
            trajectory[:, i] = x_init * exp_factor
            
    elif evolution_type == 'sigmoid':
        # Sigmoid evolution: smooth S-curve transition
        steepness = params.get('steepness', 6.0)
        midpoint = params.get('midpoint', 0.5)
        
        # Pre-compute difference, handling infinities
        with np.errstate(invalid='ignore'):
            diff = x_final - x_init
            diff = np.where(np.isfinite(diff), diff, 0.0)
        
            for i, time_point in enumerate(t):
                sigmoid_weight = 1 / (1 + np.exp(-steepness * (time_point - midpoint)))
                trajectory[:, i] = x_init + sigmoid_weight * diff
            
    elif evolution_type == 'piecewise':
        # Piecewise linear with multiple segments
        breakpoints = params.get('breakpoints', [0.33, 0.67])
        intermediate_values = params.get('intermediate_values', None)
        
        if intermediate_values is None:
            # Create default intermediate values with non-linear distribution
            # This creates more obvious changes at breakpoints
            n_segments = len(breakpoints) + 1
            diff = x_final - x_init
            diff = np.where(np.isfinite(diff), diff, 0.0)
            intermediate_values = []
            
            # Create non-uniform intermediate values for more obvious piecewise behavior
            for i in range(len(breakpoints)):
                # Use a non-linear progression to make breakpoints more obvious
                # Alternate between faster and slower progress
                if i % 2 == 0:
                    # Fast progress in even segments
                    weight = 0.8 * (i + 1) / (n_segments)
                else:
                    # Slower progress in odd segments  
                    weight = 0.3 * (i + 1) / (n_segments) + 0.4
                
                # Ensure weight stays in [0, 1] bounds
                weight = min(1.0, max(0.0, weight))
                intermediate_values.append(x_init + weight * diff)
        
        # Validate intermediate values
        if len(intermediate_values) != len(breakpoints):
            raise ValueError("Number of intermediate values must equal number of breakpoints")
        
        # Create piecewise trajectory
        all_points = [0.0] + sorted(breakpoints) + [1.0]
        all_values = [x_init] + intermediate_values + [x_final]
        
        for i, time_point in enumerate(t):
            # Find which segment we're in
            segment_idx = np.searchsorted(all_points[1:], time_point)
            t_start, t_end = all_points[segment_idx], all_points[segment_idx + 1]
            x_start, x_end = all_values[segment_idx], all_values[segment_idx + 1]
            
            # Linear interpolation within segment
            if t_end > t_start:
                local_t = (time_point - t_start) / (t_end - t_start)
            else:
                local_t = 0
            
            # Handle infinite values in interpolation
            with np.errstate(invalid='ignore'):
                segment_diff = x_end - x_start
                segment_diff = np.where(np.isfinite(segment_diff), segment_diff, 0.0)
                trajectory[:, i] = x_start + local_t * segment_diff
            
    elif evolution_type == 'oscillatory':
        # Oscillatory evolution with trend
        frequency = params.get('frequency', 2.0)
        amplitude = params.get('amplitude', 0.1)
        trend_weight = params.get('trend_weight', 0.8)
        
        # Pre-compute difference, handling infinities
        with np.errstate(invalid='ignore'):
            diff = x_final - x_init
            diff = np.where(np.isfinite(diff), diff, 0.0)
        
            for i, time_point in enumerate(t):
                # Base linear trend
                linear_component = x_init + time_point * diff
                # Oscillatory component
                oscillation = amplitude * diff * np.sin(2 * np.pi * frequency * time_point)
                # Combine with trend dominance
                trajectory[:, i] = trend_weight * linear_component + (1 - trend_weight) * oscillation
            
    elif evolution_type == 'constant':
        # No evolution - all sequences use same parameters
        for i in range(n_sequences):
            trajectory[:, i] = x_init.copy()
            
    # Apply constraints to all trajectory points
    return _apply_constraints_to_trajectory(trajectory, x_init)


def _apply_constraints_to_trajectory(trajectory: np.ndarray, x_init: np.ndarray) -> np.ndarray:
    """Apply constraints to entire parameter trajectory."""
    # Preserve forbidden transitions (-∞) and required transitions (1e8)
    forbidden_mask = x_init == -np.inf
    required_mask = x_init >= 1e7  # Large positive values for required transitions
    
    # Apply constraints to all time points
    trajectory = trajectory.copy()  # Ensure we don't modify original
    for i in range(trajectory.shape[1]):
        trajectory[forbidden_mask, i] = -np.inf
        trajectory[required_mask, i] = 1e8
        
    return trajectory


def _compute_batched_evolution(x_init: np.ndarray, 
                             x_final: np.ndarray, 
                             n_sequences: int, 
                             evolution_type: str,
                             batch_size: int, 
                             **evolution_params) -> np.ndarray:
    """
    Compute parameter evolution in batches rather than per sequence.
    
    This creates step-wise parameter changes where all sequences within 
    a batch share the same parameters, and evolution occurs between batches.
    
    Parameters
    ----------
    x_init, x_final : np.ndarray
        Initial and final parameter vectors
    n_sequences : int
        Total number of sequences
    evolution_type : str
        Evolution model to use for batch transitions
    batch_size : int
        Number of sequences per batch
    evolution_params : dict
        Parameters for the evolution model
        
    Returns
    -------
    np.ndarray, shape (state_size, n_sequences)
        Parameter trajectory with batch-wise evolution
    """
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    state_size = len(x_init)
    trajectory = np.zeros((state_size, n_sequences))
    
    # Calculate number of batches
    n_batches = int(np.ceil(n_sequences / batch_size))
    
    # Compute parameter values for each batch
    if n_batches == 1:
        # Only one batch - use initial parameters for all sequences
        batch_params = [x_init]
    else:
        # Multiple batches - evolve parameters between batches
        batch_trajectory = _compute_sequence_evolution(x_init, x_final, n_batches, 
                                                     evolution_type, **evolution_params)
        batch_params = [batch_trajectory[:, i] for i in range(n_batches)]
    
    # Assign parameters to sequences within each batch
    for batch_idx in range(n_batches):
        start_seq = batch_idx * batch_size
        end_seq = min((batch_idx + 1) * batch_size, n_sequences)
        
        # All sequences in this batch use the same parameters
        for seq_idx in range(start_seq, end_seq):
            trajectory[:, seq_idx] = batch_params[batch_idx]
    
    return trajectory


def get_batch_evolution_schedule(n_sequences: int, 
                               batch_size: int, 
                               evolution_type: str = 'linear') -> Dict[str, Any]:
    """
    Create a schedule showing when parameter evolution occurs.
    
    Parameters
    ----------
    n_sequences : int
        Total number of sequences
    batch_size : int
        Sequences per batch
    evolution_type : str
        Type of evolution between batches
        
    Returns
    -------
    dict
        Schedule information including batch boundaries and evolution points
    """
    n_batches = int(np.ceil(n_sequences / batch_size))
    
    # Batch boundaries
    batch_starts = [i * batch_size for i in range(n_batches)]
    batch_ends = [min((i + 1) * batch_size, n_sequences) for i in range(n_batches)]
    batch_sizes = [end - start for start, end in zip(batch_starts, batch_ends)]
    
    # Evolution occurs between batches
    evolution_points = batch_starts[1:]  # Parameter changes at start of each new batch
    
    return {
        'n_batches': n_batches,
        'batch_boundaries': list(zip(batch_starts, batch_ends)),
        'batch_sizes': batch_sizes,
        'evolution_points': evolution_points,
        'evolution_type': evolution_type,
        'sequences_per_evolution': batch_size
    }


def visualize_evolution_schedule(n_sequences: int = 100, 
                               batch_size: int = 20, 
                               show_details: bool = True) -> None:
    """Create visual representation of batched evolution schedule."""
    schedule = get_batch_evolution_schedule(n_sequences, batch_size)
    
    print(f"Evolution Schedule for {n_sequences} sequences:")
    print(f"├─ Batch size: {batch_size}")
    print(f"├─ Number of batches: {schedule['n_batches']}")
    print(f"├─ Evolution points: {schedule['evolution_points']}")
    
    if show_details:
        print("└─ Batch structure:")
        for i, (start, end) in enumerate(schedule['batch_boundaries']):
            size = schedule['batch_sizes'][i]
            evolution_marker = " ← EVOLUTION" if i > 0 else ""
            print(f"   Batch {i+1}: sequences {start:3d}-{end-1:3d} ({size:2d} sequences){evolution_marker}")


def create_evolution_trajectory_from_config(config: EvolutionConfig,
                                          x_init: np.ndarray,
                                          x_final: np.ndarray, 
                                          n_sequences: int) -> np.ndarray:
    """
    Create evolution trajectory from configuration object.
    
    Parameters
    ----------
    config : EvolutionConfig
        Evolution configuration
    x_init, x_final : np.ndarray
        Initial and final parameter vectors
    n_sequences : int
        Number of sequences
        
    Returns
    -------
    np.ndarray
        Parameter trajectory
    """
    return compute_evolution_trajectory(
        x_init, x_final, n_sequences,
        config.evolution_type,
        config.batch_size,
        **config.evolution_params
    )


class EvolutionManager:
    """Manager class for handling temporal evolution of parameters."""
    
    def __init__(self, evolution_type: str = 'linear', **evolution_params):
        """
        Initialize evolution manager.
        
        Parameters
        ----------
        evolution_type : str
            Type of evolution model
        **evolution_params
            Parameters specific to evolution type
        """
        self.evolution_type = evolution_type
        self.evolution_params = evolution_params
        self.last_trajectory = None
        
        # Validate evolution type
        valid_types = ['linear', 'exponential', 'sigmoid', 'piecewise', 
                      'oscillatory', 'constant']
        if evolution_type not in valid_types:
            raise ValueError(f"Evolution type must be one of {valid_types}")
    
    def compute_trajectory(self, 
                         x_init: np.ndarray,
                         x_final: np.ndarray, 
                         n_sequences: int,
                         batch_size: Optional[int] = None) -> np.ndarray:
        """Compute evolution trajectory."""
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, self.evolution_type,
            batch_size, **self.evolution_params
        )
        self.last_trajectory = trajectory
        return trajectory
    
    def update_params(self, **new_params):
        """Update evolution parameters."""
        self.evolution_params.update(new_params)
    
    def get_config(self, batch_size: Optional[int] = None) -> EvolutionConfig:
        """Get configuration object."""
        return EvolutionConfig(
            evolution_type=self.evolution_type,
            batch_size=batch_size,
            evolution_params=self.evolution_params.copy()
        )
    
    def __repr__(self) -> str:
        return f"EvolutionManager(type='{self.evolution_type}', params={self.evolution_params})"


def create_evolution_examples() -> Dict[str, EvolutionManager]:
    """Create example evolution managers for different models."""
    return {
        'linear': EvolutionManager('linear'),
        'exponential': EvolutionManager('exponential', rate=2.0),
        'sigmoid': EvolutionManager('sigmoid', steepness=6.0, midpoint=0.5),
        'piecewise': EvolutionManager('piecewise', breakpoints=[0.33, 0.67]),
        'oscillatory': EvolutionManager('oscillatory', frequency=2.0, amplitude=0.1, trend_weight=0.8),
        'constant': EvolutionManager('constant')
    }


def validate_evolution_parameters(evolution_type: str, **params) -> Dict[str, Any]:
    """
    Validate evolution parameters for given type.
    
    Parameters
    ----------
    evolution_type : str
        Evolution model type
    **params
        Parameters to validate
        
    Returns
    -------
    dict
        Validation results
    """
    valid = True
    warnings_list = []
    
    if evolution_type == 'exponential':
        rate = params.get('rate', 2.0)
        if rate <= 0:
            valid = False
            warnings_list.append("Exponential rate must be positive")
        elif rate > 10:
            warnings_list.append("Large exponential rate may cause numerical instability")
    
    elif evolution_type == 'sigmoid':
        steepness = params.get('steepness', 6.0)
        midpoint = params.get('midpoint', 0.5)
        if steepness <= 0:
            valid = False
            warnings_list.append("Sigmoid steepness must be positive")
        if not 0 <= midpoint <= 1:
            valid = False
            warnings_list.append("Sigmoid midpoint must be in [0, 1]")
    
    elif evolution_type == 'piecewise':
        breakpoints = params.get('breakpoints', [0.33, 0.67])
        if not all(0 < bp < 1 for bp in breakpoints):
            valid = False
            warnings_list.append("Piecewise breakpoints must be in (0, 1)")
        if breakpoints != sorted(breakpoints):
            valid = False
            warnings_list.append("Piecewise breakpoints must be sorted")
    
    elif evolution_type == 'oscillatory':
        frequency = params.get('frequency', 2.0)
        amplitude = params.get('amplitude', 0.1)
        trend_weight = params.get('trend_weight', 0.8)
        
        if frequency <= 0:
            valid = False
            warnings_list.append("Oscillatory frequency must be positive")
        if amplitude < 0:
            valid = False  
            warnings_list.append("Oscillatory amplitude must be non-negative")
        if not 0 <= trend_weight <= 1:
            valid = False
            warnings_list.append("Oscillatory trend weight must be in [0, 1]")
    
    return {
        'valid': valid,
        'evolution_type': evolution_type,
        'warnings': warnings_list,
        'parameters': params
    } 
