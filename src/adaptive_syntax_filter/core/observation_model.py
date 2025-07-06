"""Observation model for categorical distributions with block-wise softmax.

Implements the softmax observation model and related functions for adaptive
Kalman filtering of evolving canary song syntax rules. Based on equations 1 and 7.
"""

import numpy as np
from typing import Tuple
import warnings
from src.adaptive_syntax_filter.data.sequence_generator import softmax_mc_higher_order
from src.adaptive_syntax_filter.data.constraint_system import encode_context


def softmax_observation_model(x: np.ndarray, R: int) -> np.ndarray:
    """Block-wise softmax transformation for categorical observation model.
    
    Implements equation 1: P(y_m = a_i | y_{m-1} = a_j; x_k) = exp(x_{k,p}) / Σ_i' exp(x_{k,i'+(j-1)R})
    where p = i + (j-1)R, transforming logit parameters to transition probabilities.
    
    Parameters
    ----------
    x : np.ndarray, shape (R²,)
        Logit vector x_k ∈ ℝ^{R²} representing transition log-odds
    R : int
        Alphabet size (number of symbols)
        
    Returns
    -------
    np.ndarray, shape (R²,)
        Probability vector with block-wise normalization
        
    Notes
    -----
    The logit vector x is organized in R blocks of size R each:
    - Block j represents transitions FROM symbol j TO all symbols
    - Element x[j*R + i] is logit for transition from j to i  
    - Each block is softmax-normalized to ensure Σ_i P(· → i | · → j) = 1
    
    Examples
    --------
    >>> x = np.array([1.0, 2.0, 0.5, 1.5])  # R=2, transitions: 0→0, 0→1, 1→0, 1→1
    >>> probs = softmax_observation_model(x, R=2)
    >>> # probs[0:2] sum to 1 (transitions from symbol 0)
    >>> # probs[2:4] sum to 1 (transitions from symbol 1)
    """
    if x.shape[0] != R * R:
        raise ValueError(f"x must have length R²={R*R}, got {x.shape[0]}")
    
    probs = np.zeros_like(x)
    
    # Apply softmax block-wise for each source symbol
    for j in range(R):
        start_idx = j * R
        end_idx = (j + 1) * R
        logits_block = x[start_idx:end_idx]
        
        # Numerical stability: subtract max before exponential
        max_logit = np.max(logits_block)
        exp_logits = np.exp(logits_block - max_logit)
        sum_exp = np.sum(exp_logits)
        
        if sum_exp == 0:
            warnings.warn(f"Numerical underflow in softmax block {j}, using uniform distribution")
            probs[start_idx:end_idx] = 1.0 / R
        else:
            probs[start_idx:end_idx] = exp_logits / sum_exp
    
    return probs


def log_softmax_jacobian(x_block: np.ndarray) -> np.ndarray:
    """Jacobian of log-softmax function for a single block.
    
    Implements equation 7.1: J_{ij} = ∂log f_i / ∂x_j = δ_{ij} - f_j
    where f is the softmax function and δ_{ij} is the Kronecker delta.
    
    Parameters
    ----------
    x_block : np.ndarray, shape (R,)
        Logit values for single transition block
        
    Returns
    -------
    np.ndarray, shape (R, R)
        Jacobian matrix J where J[i,j] = ∂log f_i / ∂x_j
        
    Notes
    -----
    For softmax f_i = exp(x_i) / Σ_k exp(x_k), the log-softmax Jacobian is:
    J_{ij} = δ_{ij} - f_j where δ_{ij} = 1 if i=j, 0 otherwise
    """
    R = len(x_block)
    
    # Compute softmax probabilities
    max_logit = np.max(x_block)
    exp_logits = np.exp(x_block - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    
    # J[i,j] = δ_{ij} - f_j
    J = np.eye(R) - probs[np.newaxis, :]
    
    return J


def log_softmax_hessian(x_block: np.ndarray) -> np.ndarray:
    """Hessian of log-softmax function for a single block.
    
    Implements equation 7.2: H_{ij} = ∂²log f_i / ∂x_i ∂x_j = -f_i(δ_{ij} - f_j)
    where f is the softmax function and δ_{ij} is the Kronecker delta.
    
    Parameters
    ----------
    x_block : np.ndarray, shape (R,)
        Logit values for single transition block
        
    Returns
    -------
    np.ndarray, shape (R, R)
        Hessian matrix H where H[i,j] = ∂²log f_i / ∂x_i ∂x_j
        
    Notes
    -----
    The Hessian is used in the Kalman filter update step (equation 4.5) to compute
    the information matrix contribution from categorical observations.
    
    For log-softmax log f_i = x_i - log(Σ_k exp(x_k)), the Hessian is:
    H_{ij} = -f_i(δ_{ij} - f_j) = -f_i δ_{ij} + f_i f_j
    """
    R = len(x_block)
    
    # Compute softmax probabilities  
    max_logit = np.max(x_block)
    exp_logits = np.exp(x_block - max_logit)
    probs = exp_logits / np.sum(exp_logits)
    
    # H[i,j] = -f_i(δ_{ij} - f_j) = -f_i δ_{ij} + f_i f_j
    H = np.outer(probs, probs) - np.diag(probs)
    
    return H


def compute_observation_likelihood(observations: np.ndarray, 
                                 x: np.ndarray, 
                                 R: int) -> float:
    """Compute log-likelihood of observation sequence given logit parameters.
    
    Computes the log-likelihood term from equation 3.2:
    Σ_m log f(x_{k,y_m,y_{m-1}}) for observation sequence y^{(k)}
    
    Parameters
    ----------
    observations : np.ndarray, shape (n,)
        Sequence of symbol indices y^{(k)} = (y_1, ..., y_n)
    x : np.ndarray, shape (R²,)
        Logit parameter vector x_k
    R : int
        Alphabet size
        
    Returns
    -------
    float
        Log-likelihood of observation sequence
        
    Notes
    -----
    For each transition y_{m-1} → y_m, contributes log P(y_m | y_{m-1}; x)
    to the total log-likelihood using the softmax observation model.
    """
    if len(observations) < 2:
        return 0.0  # Need at least 2 symbols for a transition
    
    probs = softmax_observation_model(x, R)
    log_likelihood = 0.0
    
    for m in range(1, len(observations)):
        y_prev = int(observations[m-1])
        y_curr = int(observations[m])
        
        # Validate indices
        if not (0 <= y_prev < R and 0 <= y_curr < R):
            raise ValueError(f"Invalid symbol indices: y_prev={y_prev}, y_curr={y_curr}")
        
        # Position in probability vector
        pos = y_prev * R + y_curr
        prob = probs[pos]
        
        if prob <= 0:
            warnings.warn(f"Zero probability for transition {y_prev}→{y_curr}")
            log_likelihood += -np.inf
        else:
            log_likelihood += np.log(prob)
    
    return log_likelihood


def compute_finite_rmse(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute RMSE only for finite values in both vectors.
    
    This is essential when comparing constrained logit vectors where
    forbidden transitions are set to -∞ and required transitions to +∞.
    
    Parameters
    ----------
    x1, x2 : np.ndarray
        Vectors to compare, possibly containing ±∞ values
        
    Returns
    -------
    float
        RMSE computed only over finite values in both vectors
        
    Examples
    --------
    >>> x1 = np.array([1.0, -np.inf, 2.0, np.inf])
    >>> x2 = np.array([1.1, -np.inf, 2.1, np.inf])  
    >>> compute_finite_rmse(x1, x2)
    0.1
    """
    finite_mask = np.isfinite(x1) & np.isfinite(x2)
    
    if not np.any(finite_mask):
        # No finite values to compare - either all constrained identically (perfect)
        # or incompatible constraint patterns (check if constraints match)
        x1_inf_pattern = np.isinf(x1)
        x2_inf_pattern = np.isinf(x2)
        if np.array_equal(x1_inf_pattern, x2_inf_pattern):
            return 0.0  # Identical constraint patterns
        else:
            return np.inf  # Incompatible constraint patterns
    
    finite_diff = x1[finite_mask] - x2[finite_mask]
    return np.sqrt(np.mean(finite_diff**2))


def validate_transition_probabilities(probs: np.ndarray, R: int, atol: float = 1e-6) -> bool:
    """Validate that transition probabilities form valid distributions.
    
    Checks that each block (transitions from one symbol) sums to 1.
    
    Parameters
    ----------
    probs : np.ndarray, shape (R²,)
        Probability vector from softmax_observation_model
    R : int
        Alphabet size
    atol : float, default=1e-6
        Absolute tolerance for sum-to-one check
        
    Returns
    -------
    bool
        True if all blocks sum to 1 within tolerance
        
    Raises
    ------
    AssertionError
        If any block doesn't sum to 1 within tolerance
    """
    for i in range(R):
        block_start = i * R
        block_end = (i + 1) * R
        block_sum = np.sum(probs[block_start:block_end])
        
        if not np.isclose(block_sum, 1.0, atol=atol):
            raise AssertionError(
                f"Transition probabilities from symbol {i} sum to {block_sum:.8f}, "
                f"expected 1.0 ± {atol}"
            )
    
    return True 


def compute_observation_likelihood_higher_order(observations: np.ndarray, x: np.ndarray, alphabet: list, order: int) -> float:
    """Compute log-likelihood of observation sequence given logit parameters for higher-order Markov models."""
    if len(observations) < order + 1:
        return 0.0  # Need at least order+1 symbols for a transition
    n_symbols = len(alphabet)
    probs = softmax_mc_higher_order(x, alphabet, order)
    log_likelihood = 0.0
    for m in range(order, len(observations)):
        # Convert integer indices to symbol strings for context
        context = [alphabet[int(observations[m - order + j])] for j in range(order)]
        context_idx = encode_context(context, alphabet, order)
        y_curr = int(observations[m])
        pos = context_idx * n_symbols + y_curr
        prob = probs[pos]
        if prob <= 0:
            warnings.warn(f"Zero probability for transition context {context}→{y_curr}")
            log_likelihood += -np.inf
        else:
            log_likelihood += np.log(prob)
    return log_likelihood 
