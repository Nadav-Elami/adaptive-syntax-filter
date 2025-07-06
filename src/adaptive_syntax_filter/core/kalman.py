"""Kalman filtering for adaptive syntax evolution using block-diagonal state transitions.

Implements forward filtering and RTS backward smoothing as described in equations 4-5
of the adaptive Kalman-EM algorithm for evolving canary song syntax rules.
"""

import numpy as np
from scipy import linalg
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import warnings

from .observation_model import softmax_observation_model, log_softmax_hessian
from src.adaptive_syntax_filter.data.sequence_generator import softmax_mc_higher_order
from src.adaptive_syntax_filter.data.constraint_system import encode_context


@dataclass
class KalmanState:
    """Container for Kalman filter state variables.
    
    Stores the mean and covariance estimates for both filtered and smoothed states
    following the notation in equations 4-5.
    
    Attributes
    ----------
    x_filtered : np.ndarray, shape (R²,)
        Filtered state mean x_{k|k} (Equation 4.6)
    W_filtered : np.ndarray, shape (R², R²)  
        Filtered state covariance W_{k|k} (Equation 4.5)
    x_predicted : np.ndarray, shape (R²,)
        Predicted state mean x_{k|k-1} (Equation 4.2)
    W_predicted : np.ndarray, shape (R², R²)
        Predicted state covariance W_{k|k-1} (Equation 4.3)
    x_smoothed : Optional[np.ndarray], shape (R²,)
        Smoothed state mean x_{k|K} (Equation 5.2)
    W_smoothed : Optional[np.ndarray], shape (R², R²)
        Smoothed state covariance W_{k|K} (Equation 5.3)
    W_lag_one : Optional[np.ndarray], shape (R², R²)
        Lag-one smoothed covariance W_{k,k+1|K} (Equation 5.4)
    """
    x_filtered: np.ndarray
    W_filtered: np.ndarray
    x_predicted: np.ndarray
    W_predicted: np.ndarray
    x_smoothed: Optional[np.ndarray] = None
    W_smoothed: Optional[np.ndarray] = None
    W_lag_one: Optional[np.ndarray] = None


class KalmanFilter:
    """Adaptive Kalman filter for evolving syntax rules with block-diagonal transitions.
    
    Implements the forward filtering and RTS backward smoothing algorithms described
    in equations 4-5 for learning time-varying logit parameters in canary song syntax.
    
    The state vector x_k ∈ ℝ^{R²} represents logit parameters for transition probabilities
    between R symbols, where transitions are organized in R blocks of size R each.
    
    Parameters
    ----------
    alphabet_size : int
        Number of symbols R in the alphabet
    F : np.ndarray, shape (R², R²)
        State transition matrix (Equation 2.1)
    u : np.ndarray, shape (R²,)
        Control vector (Equation 2.1)  
    Sigma : np.ndarray, shape (R², R²)
        Process noise covariance matrix (Equation 2.1)
    regularization_eps : float, default=1e-8
        Regularization parameter for numerical stability
    markov_order : int, default=1
        Markov order for higher-order context encoding
    alphabet : list, default=None
        Alphabet for higher-order context encoding
        
    Notes
    -----
    The implementation follows the exact notation from equations 4-5:
    - Forward filtering: Equations 4.1-4.6
    - RTS smoothing: Equations 5.1-5.4
    - Block-wise softmax ensures ∑_j P(y_m = a_j | y_{m-1} = a_i; x_k) = 1
    """
    
    def __init__(self, 
                 alphabet_size: int,
                 F: np.ndarray,
                 u: np.ndarray, 
                 Sigma: np.ndarray,
                 regularization_eps: float = 1e-8,
                 markov_order: int = 1,
                 alphabet: list = None):
        
        # For higher-order models, we need to extract the actual alphabet size
        # The state dimension should be R^(order+1) where R is alphabet size
        # We can estimate R from the state dimension
        state_dim = F.shape[0]
        self.R = int(round(state_dim ** (1.0 / (self._estimate_markov_order(state_dim) + 1))))
        self.state_dim = state_dim
        self.regularization_eps = regularization_eps
        self.markov_order = markov_order
        self.alphabet = alphabet
        
        # Validate dimensions
        self._validate_parameters(F, u, Sigma)
        
        self.F = F.copy()
        self.u = u.copy()
        self.Sigma = Sigma.copy()
        
        # Storage for filter states
        self.states: List[KalmanState] = []
        self.K = 0  # Number of processed sequences
    
    def _estimate_markov_order(self, state_dim: int) -> int:
        """Estimate Markov order from state dimension.
        
        For a Markov model of order k with alphabet size R:
        state_dim = R^(k+1)
        
        We can estimate k by trying different values and finding the best fit.
        """
        # Try different orders and find the one that gives integer R
        for order in range(1, 6):  # Reasonable range for Markov orders
            R_estimate = state_dim ** (1.0 / (order + 1))
            if abs(R_estimate - round(R_estimate)) < 0.1:  # Close to integer
                return order
        return 1  # Default to first order if we can't determine
    
    def _validate_parameters(self, F: np.ndarray, u: np.ndarray, Sigma: np.ndarray) -> None:
        """Validate parameter dimensions and properties."""
        expected_dim = self.state_dim
        
        if F.shape != (expected_dim, expected_dim):
            raise ValueError(f"F must be ({expected_dim}, {expected_dim}), got {F.shape}")
        
        if u.shape != (expected_dim,):
            raise ValueError(f"u must be ({expected_dim},), got {u.shape}")
            
        if Sigma.shape != (expected_dim, expected_dim):
            raise ValueError(f"Sigma must be ({expected_dim}, {expected_dim}), got {Sigma.shape}")
            
        # Check Sigma is positive definite
        try:
            linalg.cholesky(Sigma)
        except linalg.LinAlgError:
            warnings.warn("Sigma is not positive definite, adding regularization")
    
    def forward_filter(self, 
                      observations: List[np.ndarray],
                      x0: np.ndarray) -> List[KalmanState]:
        """Forward Kalman filtering for sequence of observations.
        
        Implements equations 4.1-4.6 for forward filtering of logit parameters.
        Each observation sequence corresponds to one song with categorical observations.
        
        Parameters
        ----------
        observations : List[np.ndarray]
            List of K observation sequences, where observations[k] has shape (n(k),)
            representing symbol indices for song k
        x0 : np.ndarray, shape (R²,)
            Initial state estimate x_0^{(i)} (Equation 4.1)
            
        Returns
        -------
        List[KalmanState]
            Filtered states for each sequence k=1,...,K
            
        Notes
        -----
        Forward filtering equations implemented:
        - Initialization: x_{1|0} = F*x_0 + u, W_{0|0} = 0 (Equation 4.1)
        - Prediction: x_{k|k-1} = F*x_{k-1|k-1} + u (Equation 4.2)
        - Prediction: W_{k|k-1} = F*W_{k-1|k-1}*F^T + Σ (Equation 4.3)
        - Update: δ_k = Σ_m [e_{y_m,y_{m-1}} - f(x_{k|k-1})] (Equation 4.4)
        - Update: W_{k|k}^{-1} = W_{k|k-1}^{-1} + H_k (Equation 4.5)
        - Update: x_{k|k} = x_{k|k-1} + W_{k|k}*δ_k (Equation 4.6)
        """
        if x0.shape != (self.state_dim,):
            raise ValueError(f"x0 must have shape ({self.state_dim},), got {x0.shape}")
        
        self.K = len(observations)
        self.states = []
        
        # Initialize: x_{1|0} = F*x_0 + u, W_{0|0} = 0 (Equation 4.1)
        x_prev_filtered = x0.copy()
        W_prev_filtered = np.zeros((self.state_dim, self.state_dim))
        
        for k, y_k in enumerate(observations):
            # Prediction step (Equations 4.2-4.3)
            x_predicted = self._apply_state_transition(x_prev_filtered)  # Equation 4.2
            W_predicted = (self.F @ W_prev_filtered @ self.F.T + 
                          self.Sigma)  # Equation 4.3
            
            # Add regularization for numerical stability
            W_predicted += self.regularization_eps * np.eye(self.state_dim)
            
            # Update step (Equations 4.4-4.6)
            delta_k, H_k = self._compute_update_terms(y_k, x_predicted)
            
            # W_{k|k}^{-1} = W_{k|k-1}^{-1} + H_k (Equation 4.5)
            try:
                W_predicted_inv = linalg.inv(W_predicted)
            except linalg.LinAlgError:
                # Use pseudoinverse if singular
                W_predicted_inv = linalg.pinv(W_predicted)
                warnings.warn(f"Singular W_predicted at k={k}, using pseudoinverse")
            
            W_filtered_inv = W_predicted_inv + H_k
            
            try:
                W_filtered = linalg.inv(W_filtered_inv)
            except linalg.LinAlgError:
                W_filtered = linalg.pinv(W_filtered_inv)
                warnings.warn(f"Singular W_filtered_inv at k={k}, using pseudoinverse")
            
            # x_{k|k} = x_{k|k-1} + W_{k|k}*δ_k (Equation 4.6)
            x_filtered = self._apply_state_update(x_predicted, W_filtered, delta_k)
            
            # Store state
            state = KalmanState(
                x_filtered=x_filtered,
                W_filtered=W_filtered,
                x_predicted=x_predicted,
                W_predicted=W_predicted
            )
            self.states.append(state)
            
            # Prepare for next iteration
            x_prev_filtered = x_filtered
            W_prev_filtered = W_filtered
            
            # Validate transition probabilities sum to 1 (assertion check)
            self._assert_probability_normalization(x_filtered, k)
        
        return self.states
    
    def _apply_state_transition(self, x_prev: np.ndarray) -> np.ndarray:
        """Apply state transition while preserving constraint values.
        
        For constrained values (±∞), preserve them rather than applying transition.
        For finite values, apply normal transition: F*x + u
        
        Parameters
        ----------
        x_prev : np.ndarray
            Previous state, possibly containing constraint values
            
        Returns
        -------
        np.ndarray
            Predicted state with constraints preserved
        """
        # Identify constraint positions
        inf_mask = np.isinf(x_prev)
        finite_mask = ~inf_mask
        
        # Apply transition to finite values only
        x_predicted = np.zeros_like(x_prev)
        
        if np.any(finite_mask):
            # For simplicity, apply transition to all and then restore constraints
            x_predicted = self.F @ x_prev + self.u
            
            # Replace any non-finite results with original constraint values
            invalid_mask = ~np.isfinite(x_predicted)
            x_predicted[invalid_mask] = x_prev[invalid_mask]
        
        # Preserve original constraint values
        x_predicted[inf_mask] = x_prev[inf_mask]
        
        return x_predicted
    
    def _apply_state_update(self, x_predicted: np.ndarray, 
                           W_filtered: np.ndarray, 
                           delta_k: np.ndarray) -> np.ndarray:
        """Apply state update while preserving constraint values.
        
        Parameters
        ----------
        x_predicted : np.ndarray
            Predicted state
        W_filtered : np.ndarray
            Filtered covariance
        delta_k : np.ndarray
            Update vector
            
        Returns
        -------
        np.ndarray
            Updated state with constraints preserved
        """
        # Identify constraint positions
        inf_mask = np.isinf(x_predicted)
        
        # Apply update
        x_filtered = x_predicted + W_filtered @ delta_k
        
        # Restore constraint values
        x_filtered[inf_mask] = x_predicted[inf_mask]
        
        return x_filtered
    
    def _compute_update_terms(self, 
                            y_k: np.ndarray, 
                            x_predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute update terms δ_k and H_k for Kalman filtering.
        
        Implements equations 4.4 and 4.5 for the update step.
        
        Parameters
        ----------
        y_k : np.ndarray, shape (n(k),)
            Observation sequence for song k (symbol indices)
        x_predicted : np.ndarray, shape (R²,)
            Predicted state x_{k|k-1}
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            delta_k (shape R²,) and H_k (shape R², R²) from equations 4.4-4.5
        """
        if self.markov_order > 1 and self.alphabet is not None:
            n_symbols = len(self.alphabet)
            order = self.markov_order
            state_dim = self.state_dim
            probs = softmax_mc_higher_order(x_predicted, self.alphabet, order)
            delta_k = np.zeros(state_dim)
            H_k = np.zeros((state_dim, state_dim))
            n_k = len(y_k)
            for m in range(order, n_k):
                try:
                    context = [self.alphabet[int(y_k[m - order + j])] for j in range(order)]
                    context_idx = encode_context(context, self.alphabet, order)
                    y_curr = int(y_k[m])
                    if not (0 <= y_curr < n_symbols):
                        print(f"[ERROR] y_curr={y_curr} out of range for alphabet of size {n_symbols}")
                        print(f"  context_idx={context_idx}, context={context}, order={order}")
                        print(f"  y_k={y_k}")
                        raise ValueError(f"y_curr={y_curr} out of range for alphabet of size {n_symbols}")
                    pos = context_idx * n_symbols + y_curr
                    if pos >= state_dim or pos < 0:
                        print(f"[ERROR] pos={pos} out of bounds for state_dim={state_dim}")
                        print(f"  context_idx={context_idx}, y_curr={y_curr}, n_symbols={n_symbols}")
                        print(f"  context={context}, order={order}")
                        print(f"  expected max context_idx = {n_symbols**order - 1}")
                        print(f"  y_k={y_k}")
                        raise ValueError(f"Position {pos} out of bounds for state dimension {state_dim}")
                    e_indicator = np.zeros(state_dim)
                    e_indicator[pos] = 1.0
                    block_start = context_idx * n_symbols
                    block_end = block_start + n_symbols
                    if block_end > state_dim or block_start < 0:
                        print(f"[ERROR] block indices out of bounds: block_start={block_start}, block_end={block_end}, state_dim={state_dim}")
                        print(f"  context_idx={context_idx}, n_symbols={n_symbols}")
                        print(f"  context={context}, order={order}")
                        print(f"  y_k={y_k}")
                        raise ValueError(f"Block indices out of bounds for state dimension {state_dim}")
                    prob_block = probs[block_start:block_end]
                    if prob_block.shape[0] != n_symbols:
                        print(f"[ERROR] prob_block shape mismatch: got {prob_block.shape}, expected {n_symbols}")
                        print(f"  block_start={block_start}, block_end={block_end}, state_dim={state_dim}")
                        print(f"  context_idx={context_idx}, n_symbols={n_symbols}")
                        print(f"  context={context}, order={order}")
                        print(f"  y_k={y_k}")
                        raise ValueError(f"prob_block shape mismatch")
                    delta_k += (e_indicator - np.zeros(state_dim))
                    delta_k[block_start:block_end] -= prob_block
                    H_block = log_softmax_hessian(x_predicted[block_start:block_end])
                    if H_block.shape != (n_symbols, n_symbols):
                        print(f"[ERROR] H_block shape mismatch: got {H_block.shape}, expected ({n_symbols}, {n_symbols})")
                        print(f"  block_start={block_start}, block_end={block_end}, state_dim={state_dim}")
                        print(f"  context_idx={context_idx}, n_symbols={n_symbols}")
                        print(f"  context={context}, order={order}")
                        print(f"  y_k={y_k}")
                        raise ValueError(f"H_block shape mismatch")
                    H_k[block_start:block_end, block_start:block_end] -= H_block
                except Exception as e:
                    print(f"[EXCEPTION] Error in _compute_update_terms at m={m}")
                    print(f"  Exception: {e}")
                    print(f"  y_k={y_k}")
                    print(f"  x_predicted.shape={x_predicted.shape}")
                    print(f"  state_dim={state_dim}, n_symbols={n_symbols}, order={order}")
                    raise
            return delta_k, H_k
        else:
            n_k = len(y_k)
            
            # Initialize delta_k = Σ_m [e_{y_m,y_{m-1}} - f(x_{k|k-1})] (Equation 4.4)
            delta_k = np.zeros(self.state_dim)
            H_k = np.zeros((self.state_dim, self.state_dim))
            
            # Convert logits to probabilities using block-wise softmax
            probs = softmax_observation_model(x_predicted, self.R)
            
            for m in range(1, n_k):  # Start from m=1 since we need y_{m-1}
                y_prev = int(y_k[m-1])
                y_curr = int(y_k[m])
                
                # Validate symbol indices
                if not (0 <= y_prev < self.R and 0 <= y_curr < self.R):
                    raise ValueError(f"Invalid symbol indices: y_prev={y_prev}, y_curr={y_curr}, R={self.R}")
                
                # Create one-hot indicator e_{y_m,y_{m-1}}
                # Position in state vector for transition from y_prev to y_curr
                pos = y_prev * self.R + y_curr
                
                e_indicator = np.zeros(self.state_dim)
                e_indicator[pos] = 1.0
                
                # Extract predicted probabilities for this context (previous symbol)
                prob_block = probs[y_prev * self.R:(y_prev + 1) * self.R]
                
                # Update delta_k
                delta_k += (e_indicator - np.zeros(self.state_dim))
                delta_k[y_prev * self.R:(y_prev + 1) * self.R] -= prob_block
                
                # Compute Hessian contribution for this observation
                H_block = log_softmax_hessian(x_predicted[y_prev * self.R:(y_prev + 1) * self.R])
                H_k[y_prev * self.R:(y_prev + 1) * self.R, 
                    y_prev * self.R:(y_prev + 1) * self.R] -= H_block
            
            return delta_k, H_k
    
    def _assert_probability_normalization(self, x_filtered: np.ndarray, k: int) -> None:
        """Assert that transition probabilities sum to 1 for each context.
        
        This is a validation check to ensure the softmax observation model
        is working correctly and producing valid probability distributions.
        
        Parameters
        ----------
        x_filtered : np.ndarray
            Filtered state vector
        k : int
            Sequence index for error reporting
        """
        # Use correct softmax for higher-order models
        if self.markov_order > 1 and self.alphabet is not None:
            from src.adaptive_syntax_filter.data.sequence_generator import softmax_mc_higher_order
            probs = softmax_mc_higher_order(x_filtered, self.alphabet, self.markov_order)
        else:
            probs = softmax_observation_model(x_filtered, self.R)
        
        for i in range(self.R):
            prob_block = probs[i * self.R:(i + 1) * self.R]
            prob_sum = np.sum(prob_block)
            
            if not np.isclose(prob_sum, 1.0, atol=1e-6):
                warnings.warn(f"Probability normalization violated at k={k}, "
                            f"context {i}: sum={prob_sum:.6f}")
    
    def rts_smoother(self) -> List[KalmanState]:
        """RTS backward smoothing for filtered states.
        
        Implements equations 5.1-5.4 for backward smoothing of filtered estimates.
        Must be called after forward_filter().
        
        Returns
        -------
        List[KalmanState]
            States with smoothed estimates added (x_smoothed, W_smoothed, W_lag_one)
            
        Notes
        -----
        RTS smoothing equations implemented:
        - Smoother gain: M_k = W_{k|k}*F^T*(W_{k+1|k})^{-1} (Equation 5.1)
        - Smoothed mean: x_{k|K} = x_{k|k} + M_k*(x_{k+1|K} - x_{k+1|k}) (Equation 5.2)
        - Smoothed cov: W_{k|K} = W_{k|k} + M_k*(W_{k+1|K} - W_{k+1|k})*M_k^T (Equation 5.3)
        - Lag-one cov: W_{k,k+1|K} = M_k*W_{k+1|K} (Equation 5.4)
        """
        if not self.states:
            raise RuntimeError("Must run forward_filter() before smoothing")
        
        # Initialize backward pass with final filtered estimates
        # x_{K|K} = x_{K|K}, W_{K|K} = W_{K|K}
        self.states[-1].x_smoothed = self.states[-1].x_filtered.copy()
        self.states[-1].W_smoothed = self.states[-1].W_filtered.copy()
        
        # Backward pass: k = K-1, K-2, ..., 1
        for k in range(self.K - 2, -1, -1):
            # Smoother gain: M_k = W_{k|k}*F^T*(W_{k+1|k})^{-1} (Equation 5.1)
            W_next_pred = self.states[k + 1].W_predicted
            
            try:
                W_next_pred_inv = linalg.inv(W_next_pred)
            except linalg.LinAlgError:
                W_next_pred_inv = linalg.pinv(W_next_pred)
                warnings.warn(f"Singular W_{{k+1|k}} at k={k}, using pseudoinverse")
            
            M_k = self.states[k].W_filtered @ self.F.T @ W_next_pred_inv
            
            # Smoothed mean: x_{k|K} = x_{k|k} + M_k*(x_{k+1|K} - x_{k+1|k}) (Equation 5.2)
            x_diff = (self.states[k + 1].x_smoothed - 
                     self.states[k + 1].x_predicted)
            self.states[k].x_smoothed = self.states[k].x_filtered + M_k @ x_diff
            
            # Smoothed covariance: W_{k|K} = W_{k|k} + M_k*(W_{k+1|K} - W_{k+1|k})*M_k^T (Equation 5.3)
            W_diff = (self.states[k + 1].W_smoothed - 
                     self.states[k + 1].W_predicted)
            self.states[k].W_smoothed = (self.states[k].W_filtered + 
                                       M_k @ W_diff @ M_k.T)
            
            # Lag-one covariance: W_{k,k+1|K} = M_k*W_{k+1|K} (Equation 5.4)
            self.states[k].W_lag_one = M_k @ self.states[k + 1].W_smoothed
            
            # Validate transition probabilities for smoothed state
            self._assert_probability_normalization(self.states[k].x_smoothed, k)
        
        return self.states
    
    def get_smoothed_estimates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract smoothed state estimates and lag-one covariances.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x_smoothed: shape (K, R²) - smoothed state means x_{k|K}
            W_smoothed: shape (K, R², R²) - smoothed covariances W_{k|K}  
            W_lag_one: shape (K-1, R², R²) - lag-one covariances W_{k,k+1|K}
        """
        if not all(state.x_smoothed is not None for state in self.states):
            raise RuntimeError("Must run rts_smoother() before extracting estimates")
        
        x_smoothed = np.array([state.x_smoothed for state in self.states])
        W_smoothed = np.array([state.W_smoothed for state in self.states])
        W_lag_one = np.array([state.W_lag_one for state in self.states[:-1] 
                             if state.W_lag_one is not None])
        
        return x_smoothed, W_smoothed, W_lag_one
    
    def update_parameters(self, F: np.ndarray, u: np.ndarray, Sigma: np.ndarray) -> None:
        """Update filter parameters for next EM iteration.
        
        Parameters
        ----------
        F : np.ndarray, shape (R², R²)
            New state transition matrix
        u : np.ndarray, shape (R²,)
            New control vector
        Sigma : np.ndarray, shape (R², R²)
            New process noise covariance
        """
        self._validate_parameters(F, u, Sigma)
        self.F = F.copy()
        self.u = u.copy()
        self.Sigma = Sigma.copy()
    
    def reset(self) -> None:
        """Reset filter state for new sequence processing."""
        self.states = []
        self.K = 0 
