"""
Core functionality for adaptive syntax filtering.
"""

# Core module imports will go here 

"""Core mathematical algorithms for adaptive syntax filtering.

This module contains the fundamental mathematical components:
- Kalman filtering and RTS smoothing
- Observation models (softmax transformations)
- EM algorithm for parameter learning
- Higher-order Markov state space management
"""

from .kalman import KalmanFilter, KalmanState
from .observation_model import (
    softmax_observation_model,
    log_softmax_jacobian,
    log_softmax_hessian,
    compute_observation_likelihood,
    compute_finite_rmse,
    validate_transition_probabilities
)
from .state_space import StateSpaceManager, StateSpaceConfig
from .em_algorithm import EMAlgorithm, EMParameters, EMStatistics

__all__ = [
    # Kalman filtering
    'KalmanFilter',
    'KalmanState',
    
    # Observation models
    'softmax_observation_model',
    'log_softmax_jacobian', 
    'log_softmax_hessian',
    'compute_observation_likelihood',
    'compute_finite_rmse',
    'validate_transition_probabilities',
    
    # State space management
    'StateSpaceManager',
    'StateSpaceConfig',
    
    # EM algorithm
    'EMAlgorithm',
    'EMParameters',
    'EMStatistics'
] 
