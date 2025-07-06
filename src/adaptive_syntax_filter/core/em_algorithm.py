"""EM algorithm implementation for adaptive Kalman filtering of syntax evolution.

Implements the complete M-step parameter updates and EM iteration structure
described in sections 6 and 8 of the equations document for learning
time-varying logit parameters in canary song syntax.
"""

import numpy as np
from scipy import linalg
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import warnings

from .kalman import KalmanFilter, KalmanState
from .state_space import StateSpaceManager
from .observation_model import compute_observation_likelihood, compute_observation_likelihood_higher_order
from src.adaptive_syntax_filter.data.constraint_system import encode_context


@dataclass
class EMParameters:
    """Container for EM algorithm parameters.
    
    Represents Θ = {x_0, Σ, F, u} as described in equation 8.1.
    
    Attributes
    ----------
    x0 : np.ndarray, shape (R²,)
        Initial state estimate x_0
    Sigma : np.ndarray, shape (R², R²)
        Process noise covariance matrix Σ (diagonal)
    F : np.ndarray, shape (R², R²)
        State transition matrix F (block-diagonal)
    u : np.ndarray, shape (R²,)
        Control vector u
    """
    x0: np.ndarray
    Sigma: np.ndarray
    F: np.ndarray
    u: np.ndarray


@dataclass
class EMStatistics:
    """Container for EM algorithm statistics and convergence tracking.
    
    Attributes
    ----------
    iteration : int
        Current EM iteration number
    log_likelihood : float
        Current expected log-likelihood Q(Θ, Θ^{(i)})
    log_likelihood_change : float
        Change in log-likelihood from previous iteration
    parameter_change : float
        Norm of parameter change from previous iteration
    converged : bool
        Whether algorithm has converged
    """
    iteration: int
    log_likelihood: float
    log_likelihood_change: float
    parameter_change: float
    converged: bool


@dataclass
class EMResults:
    """Container for EM algorithm results including best iteration tracking.
    
    Attributes
    ----------
    final_params : EMParameters
        Parameters from the final iteration
    best_params : EMParameters
        Parameters from the best iteration (highest log-likelihood)
    best_iteration : int
        Iteration number with the best log-likelihood
    best_log_likelihood : float
        Best log-likelihood achieved
    statistics_history : List[EMStatistics]
        Complete history of EM statistics
    """
    final_params: EMParameters
    best_params: EMParameters
    best_iteration: int
    best_log_likelihood: float
    statistics_history: List[EMStatistics]


class EMAlgorithm:
    """EM algorithm for adaptive Kalman filtering with block-diagonal dynamics.
    
    Implements Algorithm 1 from section 8 of the equations document, performing
    iterative E-step (Kalman filtering + RTS smoothing) and M-step (parameter updates)
    to learn evolving syntax rules in birdsong sequences.
    
    Parameters
    ----------
    state_space_manager : StateSpaceManager
        Manager for higher-order Markov state space
    max_iterations : int, default=50
        Maximum number of EM iterations
    tolerance : float, default=1e-4
        Convergence tolerance for log-likelihood change
    regularization_lambda : float, default=1e-3
        Ridge regularization parameter λ for equation 6.3
    damping_factor : float, default=1.0
        Damping factor for parameter updates
    adaptive_damping : bool, default=True
        Whether to use adaptive damping
    verbose : bool, default=False
        Whether to print progress information
        
    Notes
    -----
    Implements the complete EM algorithm structure:
    - E-step: Equations 4.1-4.6 (forward) + 5.1-5.4 (backward) 
    - M-step: Equations 6.1-6.5 (parameter updates)
    - Convergence: Monitor change in expected log-likelihood Q
    """
    
    def __init__(self,
                 state_space_manager: StateSpaceManager,
                 max_iterations: int = 50,
                 tolerance: float = 1e-4,
                 regularization_lambda: float = 1e-3,
                 damping_factor: float = 1.0,
                 adaptive_damping: bool = True,
                 verbose: bool = False):
        
        self.state_manager = state_space_manager
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization_lambda = regularization_lambda
        self.damping_factor = damping_factor
        self.adaptive_damping = adaptive_damping
        self.verbose = verbose
        
        # Initialize containers
        self.current_params: Optional[EMParameters] = None
        self.kalman_filter: Optional[KalmanFilter] = None
        self.statistics_history: List[EMStatistics] = []
        
        # Best iteration tracking
        self.best_params: Optional[EMParameters] = None
        self.best_log_likelihood: float = -np.inf
        self.best_iteration: int = -1
        
        # Adaptive damping tracking
        self.current_damping_factor = damping_factor
        self.damping_history = []
        self.consecutive_decreases = 0
        self.consecutive_increases = 0
        
    def initialize_parameters(self, 
                            observations: List[np.ndarray],
                            random_state: Optional[int] = None,
                            constraint_manager = None) -> EMParameters:
        """Initialize EM parameters using data-driven approach.
        
        Implements improved initialization using empirical transition counts
        and spectral properties for better convergence.
        
        Parameters
        ----------
        observations : List[np.ndarray]
            List of observation sequences
        random_state : Optional[int]
            Random seed for reproducibility
        constraint_manager : Optional[ConstraintManager]
            Constraint manager for applying syntax constraints
            
        Returns
        -------
        EMParameters
            Initialized parameters
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        R = self.state_manager.alphabet_size
        state_dim = self.state_manager.state_dim
        
        # 1. Data-driven initialization using empirical transition counts
        x0 = self._initialize_from_empirical_counts(observations, R)
        
        # 2. Initialize F as block-diagonal with identity blocks
        F = self._initialize_block_diagonal_F(R)
        
        # 3. Initialize u as zero vector
        u = np.zeros(state_dim)
        
        # 4. Initialize Sigma as diagonal with moderate variance
        Sigma = self._initialize_diagonal_sigma(state_dim)
        
        # Create initial parameters
        initial_params = EMParameters(x0=x0, Sigma=Sigma, F=F, u=u)
        
        # Apply constraints if provided
        if constraint_manager is not None:
            initial_params = self._apply_constraints_to_parameters(initial_params, constraint_manager)
        
        return initial_params
    
    def _initialize_from_empirical_counts(self, observations: List[np.ndarray], R: int) -> np.ndarray:
        """Initialize x0 using empirical transition counts from data for arbitrary Markov order."""
        order = self.state_manager.markov_order
        state_dim = self.state_manager.state_dim
        # Count transitions for all contexts
        transition_counts = np.zeros(state_dim)
        context_counts = np.zeros(R ** order)
        for obs in observations:
            if len(obs) <= order:
                continue
            for i in range(order, len(obs)):
                # Convert integer indices to symbol strings for context
                context = [self.state_manager.alphabet[int(obs[i - order + j])] for j in range(order)]
                context_idx = encode_context(context, self.state_manager.alphabet, order)
                target_symbol = int(obs[i])
                pos = context_idx * R + target_symbol
                transition_counts[pos] += 1
                context_counts[context_idx] += 1
        # Convert to logits with regularization
        x0 = np.zeros(state_dim)
        epsilon = 1e-6
        for context_idx in range(R ** order):
            row_sum = context_counts[context_idx]
            for j in range(R):
                pos = context_idx * R + j
                count = transition_counts[pos]
                if row_sum > 0:
                    prob = (count + epsilon) / (row_sum + R * epsilon)
                    x0[pos] = np.log(prob)
                else:
                    x0[pos] = np.log(1.0 / R)
        return x0
    
    def _initialize_block_diagonal_F(self, R: int) -> np.ndarray:
        """Initialize F as block-diagonal matrix with identity blocks.
        
        Parameters
        ----------
        R : int
            Alphabet size
            
        Returns
        -------
        np.ndarray
            Block-diagonal F matrix
        """
        state_dim = self.state_manager.state_dim
        F = np.zeros((state_dim, state_dim))
        
        # Create identity blocks for each context
        for context_idx in range(self.state_manager.n_contexts):
            start_idx, end_idx = self.state_manager.get_block_indices(context_idx)
            block_size = end_idx - start_idx
            
            # Identity block with slight regularization
            F[start_idx:end_idx, start_idx:end_idx] = 0.9 * np.eye(block_size)
        
        return F
    
    def _initialize_diagonal_sigma(self, state_dim: int) -> np.ndarray:
        """Initialize Sigma as diagonal matrix with moderate variance.
        
        Parameters
        ----------
        state_dim : int
            State dimension
            
        Returns
        -------
        np.ndarray
            Diagonal covariance matrix
        """
        # Start with moderate variance, will be updated by EM
        base_variance = 0.1
        Sigma = base_variance * np.eye(state_dim)
        
        return Sigma
    
    def e_step(self, 
               observations: List[np.ndarray], 
               parameters: EMParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """E-step: Forward filtering and backward smoothing.
        
        Implements equations 4.1-4.6 and 5.1-5.4 to obtain smoothed state estimates
        and lag-one covariances needed for the M-step.
        
        Parameters
        ----------
        observations : List[np.ndarray]
            List of K observation sequences
        parameters : EMParameters
            Current parameter estimates Θ^{(i)}
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x_smoothed: shape (K, R²) - smoothed states x_{k|K}
            W_smoothed: shape (K, R², R²) - smoothed covariances W_{k|K}
            W_lag_one: shape (K-1, R², R²) - lag-one covariances W_{k,k+1|K}
        """
        # Initialize Kalman filter with current parameters
        self.kalman_filter = KalmanFilter(
            alphabet_size=self.state_manager.state_dim,
            F=parameters.F,
            u=parameters.u,
            Sigma=parameters.Sigma,
            markov_order=self.state_manager.markov_order,
            alphabet=self.state_manager.alphabet
        )
        
        # Forward filtering (equations 4.1-4.6)
        states = self.kalman_filter.forward_filter(observations, parameters.x0)
        
        # Backward smoothing (equations 5.1-5.4)
        states = self.kalman_filter.rts_smoother()
        
        # Extract smoothed estimates
        x_smoothed, W_smoothed, W_lag_one = self.kalman_filter.get_smoothed_estimates()
        
        return x_smoothed, W_smoothed, W_lag_one
    
    def m_step(self,
               observations: List[np.ndarray],
               x_smoothed: np.ndarray,
               W_smoothed: np.ndarray,
               W_lag_one: np.ndarray,
               current_params: EMParameters) -> EMParameters:
        """M-step: Parameter updates using smoothed state estimates.
        
        Implements equations 6.1-6.5 for updating all parameters Θ^{(i+1)}.
        
        Parameters
        ----------
        observations : List[np.ndarray]
            List of K observation sequences
        x_smoothed : np.ndarray, shape (K, R²)
            Smoothed state means x_{k|K}
        W_smoothed : np.ndarray, shape (K, R², R²)
            Smoothed state covariances W_{k|K}
        W_lag_one : np.ndarray, shape (K-1, R², R²)
            Lag-one covariances W_{k,k+1|K}
        current_params : EMParameters
            Current parameter estimates Θ^{(i)}
            
        Returns
        -------
        EMParameters
            Updated parameter estimates Θ^{(i+1)}
        """
        K = len(observations)
        state_dim = self.state_manager.state_dim
        
        # 1. Initial state update: x_0^* = x_{1|K} (Equation 6.1)
        x0_new = x_smoothed[0].copy()
        
        # 2. Block-wise dynamics updates (Equations 6.2-6.4)
        F_new, u_new = self._update_block_diagonal_dynamics(
            x_smoothed, W_smoothed, W_lag_one, current_params
        )
        
        # 3. Process noise variance update (Equation 6.5)
        Sigma_new = self._update_process_noise(
            x_smoothed, F_new, u_new, current_params
        )
        
        return EMParameters(x0=x0_new, Sigma=Sigma_new, F=F_new, u=u_new)
    
    def _update_block_diagonal_dynamics(self,
                                       x_smoothed: np.ndarray,
                                       W_smoothed: np.ndarray,
                                       W_lag_one: np.ndarray,
                                       current_params: EMParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Update block-diagonal dynamics F and control vector u.
        
        Implements equations 6.2-6.4 for block-wise parameter updates.
        
        Parameters
        ----------
        x_smoothed : np.ndarray, shape (K, R²)
            Smoothed state means
        W_smoothed : np.ndarray, shape (K, R², R²)
            Smoothed state covariances
        W_lag_one : np.ndarray, shape (K-1, R², R²)
            Lag-one covariances
        current_params : EMParameters
            Current parameter estimates
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Updated F and u matrices
        """
        K = x_smoothed.shape[0]
        R = self.state_manager.alphabet_size
        
        # Initialize new F as block-diagonal
        F_blocks_new = []
        u_blocks_new = []
        
        # Process each context block separately
        for context_idx in range(self.state_manager.n_contexts):
            start_idx, end_idx = self.state_manager.get_block_indices(context_idx)
            
            # Extract smoothed states for this block: Z_k^{(b)} = x_{k|K}^{(b)}
            Z_current = x_smoothed[:, start_idx:end_idx]  # Shape: (K, R)
            Z_prev = np.vstack([current_params.x0[start_idx:end_idx].reshape(1, -1),
                               x_smoothed[:-1, start_idx:end_idx]])  # Shape: (K, R)
            
            # Sufficient statistics (Equation 6.2):
            # S_1^{(b)} = Σ_k Z_k^{(b)} Z_{k-1}^{(b)T}
            # S_0^{(b)} = Σ_k Z_{k-1}^{(b)} Z_{k-1}^{(b)T}
            S1_block = Z_current.T @ Z_prev  # Shape: (R, R)
            S0_block = Z_prev.T @ Z_prev     # Shape: (R, R)
            
            # Add covariance contributions from smoothed estimates
            for k in range(K):
                if k == 0:
                    # For k=0, previous state is x_0 (deterministic)
                    continue
                else:
                    # Add W_{k,k-1|K} contribution
                    W_k_lag = W_lag_one[k-1, start_idx:end_idx, start_idx:end_idx]
                    S1_block += W_k_lag
                    
                    # Add W_{k-1|K} contribution  
                    W_k_prev = W_smoothed[k-1, start_idx:end_idx, start_idx:end_idx]
                    S0_block += W_k_prev
            
            # State transition matrix update (Equation 6.3):
            # F^{(b)(i+1)} = S_1^{(b)} (S_0^{(b)} + λI_d)^{-1}
            regularized_S0 = S0_block + self.regularization_lambda * np.eye(R)
            
            try:
                F_block_new = S1_block @ linalg.inv(regularized_S0)
            except linalg.LinAlgError:
                warnings.warn(f"Singular S0 matrix for block {context_idx}, using pseudoinverse")
                F_block_new = S1_block @ linalg.pinv(regularized_S0)
            
            # Control vector update (Equation 6.4):
            # u^{(b)(i+1)} = (1/K) Σ_k (Z_k^{(b)} - F^{(b)(i+1)} Z_{k-1}^{(b)})
            residuals = Z_current - Z_prev @ F_block_new.T
            u_block_new = np.mean(residuals, axis=0)
            
            F_blocks_new.append(F_block_new)
            u_blocks_new.append(u_block_new)
        
        # Construct full F matrix and u vector
        F_new = self.state_manager.create_block_diagonal_structure(F_blocks_new)
        u_new = np.concatenate(u_blocks_new)
        
        return F_new, u_new
    
    def _update_process_noise(self,
                            x_smoothed: np.ndarray,
                            F_new: np.ndarray,
                            u_new: np.ndarray,
                            current_params: EMParameters) -> np.ndarray:
        """Update process noise covariance matrix.
        
        Implements equation 6.5: σ_j^{2(i+1)} = (1/K) Σ_k [x_{k|K,j} - (F*x_{k-1|K} + u)_j]²
        
        Parameters
        ----------
        x_smoothed : np.ndarray, shape (K, R²)
            Smoothed state means
        F_new : np.ndarray, shape (R², R²)
            Updated state transition matrix
        u_new : np.ndarray, shape (R²,)
            Updated control vector
        current_params : EMParameters
            Current parameter estimates for accessing x0
            
        Returns
        -------
        np.ndarray, shape (R², R²)
            Updated diagonal process noise covariance matrix
        """
        K = x_smoothed.shape[0]
        state_dim = self.state_manager.state_dim
        
        # Prepare previous states (include x_0 for k=1)
        x_prev = np.vstack([current_params.x0.reshape(1, -1),
                           x_smoothed[:-1, :]])  # Shape: (K, R²)
        
        # Compute predicted states: F*x_{k-1|K} + u
        x_predicted = x_prev @ F_new.T + u_new  # Shape: (K, R²)
        
        # Compute residuals: x_{k|K} - (F*x_{k-1|K} + u)
        residuals = x_smoothed - x_predicted  # Shape: (K, R²)
        
        # Compute diagonal variances (Equation 6.5)
        variances = np.mean(residuals**2, axis=0)  # Shape: (R²,)
        
        # Ensure minimum variance for numerical stability
        min_variance = 1e-6
        variances = np.maximum(variances, min_variance)
        
        # Return diagonal covariance matrix
        return np.diag(variances)
    
    def compute_expected_log_likelihood(self,
                                      observations: List[np.ndarray],
                                      x_smoothed: np.ndarray,
                                      W_smoothed: np.ndarray,
                                      parameters: EMParameters) -> float:
        """Compute expected log-likelihood Q(Θ, Θ^{(i)}).
        
        Implements equation 8.1 for monitoring convergence.
        
        Parameters
        ----------
        observations : List[np.ndarray]
            List of observation sequences
        x_smoothed : np.ndarray, shape (K, R²)
            Smoothed state means
        W_smoothed : np.ndarray, shape (K, R², R²)
            Smoothed state covariances
        parameters : EMParameters
            Current parameter estimates
            
        Returns
        -------
        float
            Expected log-likelihood value
        """
        K = len(observations)
        total_log_likelihood = 0.0
        
        # Observation likelihood terms
        alphabet = self.state_manager.alphabet
        order = self.state_manager.markov_order
        for k, y_k in enumerate(observations):
            if order > 1:
                obs_ll = compute_observation_likelihood_higher_order(y_k, x_smoothed[k], alphabet, order)
            else:
                obs_ll = compute_observation_likelihood(y_k, x_smoothed[k], self.state_manager.alphabet_size)
            total_log_likelihood += obs_ll
        
        # State dynamics likelihood terms
        x_prev = np.vstack([parameters.x0.reshape(1, -1), x_smoothed[:-1, :]])
        
        for k in range(K):
            # Predicted state
            x_pred = parameters.F @ x_prev[k] + parameters.u
            
            # Residual
            residual = x_smoothed[k] - x_pred
            
            # Quadratic form: -0.5 * residual^T * Σ^{-1} * residual
            try:
                Sigma_inv = linalg.inv(parameters.Sigma)
            except linalg.LinAlgError:
                Sigma_inv = linalg.pinv(parameters.Sigma)
                
            quadratic_term = -0.5 * residual.T @ Sigma_inv @ residual
            
            # Add trace term for covariance: -0.5 * tr(Σ^{-1} * W_{k|K})
            trace_term = -0.5 * np.trace(Sigma_inv @ W_smoothed[k])
            
            total_log_likelihood += quadratic_term + trace_term
        
        # Normalization constant: -(K/2) * log|2π Σ|
        try:
            log_det_Sigma = np.linalg.slogdet(parameters.Sigma)[1]
        except:
            log_det_Sigma = np.sum(np.log(np.diag(parameters.Sigma)))
            
        normalization = -0.5 * K * (self.state_manager.state_dim * np.log(2 * np.pi) + log_det_Sigma)
        total_log_likelihood += normalization
        
        return total_log_likelihood
    
    def fit(self, 
            observations: List[np.ndarray],
            initial_params: Optional[EMParameters] = None) -> EMResults:
        """Run complete EM algorithm to fit parameters.
        
        Implements Algorithm 1 from section 8 with convergence monitoring and
        best iteration tracking.
        
        Parameters
        ----------
        observations : List[np.ndarray]
            List of K observation sequences to fit
        initial_params : Optional[EMParameters]
            Initial parameter estimates. If None, will be initialized automatically.
            
        Returns
        -------
        EMResults
            Complete EM results including best iteration tracking
        """
        if initial_params is None:
            initial_params = self.initialize_parameters(observations)
        
        self.current_params = initial_params
        self.statistics_history = []
        self.best_params = None
        self.best_log_likelihood = -np.inf
        self.best_iteration = -1
        prev_log_likelihood = -np.inf
        
        if self.verbose:
            print(f"Starting EM algorithm with {len(observations)} sequences")
            print(f"State space dimension: {self.state_manager.state_dim}")
        
        for iteration in range(self.max_iterations):
            # E-step: Forward filtering + RTS smoothing
            x_smoothed, W_smoothed, W_lag_one = self.e_step(observations, self.current_params)
            
            # Compute expected log-likelihood
            current_log_likelihood = self.compute_expected_log_likelihood(
                observations, x_smoothed, W_smoothed, self.current_params
            )
            
            # M-step: Parameter updates
            new_params = self.m_step(
                observations, x_smoothed, W_smoothed, W_lag_one, self.current_params
            )
            
            # Apply damping to parameter updates
            if self.adaptive_damping:
                self.current_damping_factor = self._adaptive_damping(current_log_likelihood, prev_log_likelihood)
            new_params = self._apply_damping(new_params, self.current_params)
            
            # Track best iteration
            if current_log_likelihood > self.best_log_likelihood:
                self.best_log_likelihood = current_log_likelihood
                self.best_iteration = iteration
                # Deep copy the best parameters
                self.best_params = EMParameters(
                    x0=new_params.x0.copy(),
                    Sigma=new_params.Sigma.copy(),
                    F=new_params.F.copy(),
                    u=new_params.u.copy()
                )
                if self.verbose:
                    print(f"🏆 NEW BEST ITERATION: {iteration} (LL: {current_log_likelihood:.6f})")
            
            # Compute parameter change
            param_change = (
                np.linalg.norm(new_params.x0 - self.current_params.x0) +
                np.linalg.norm(new_params.F - self.current_params.F) +
                np.linalg.norm(new_params.u - self.current_params.u) +
                np.linalg.norm(new_params.Sigma - self.current_params.Sigma)
            )
            
            # Check convergence
            log_likelihood_change = current_log_likelihood - prev_log_likelihood
            converged = abs(log_likelihood_change) < self.tolerance
            
            # Record statistics
            stats = EMStatistics(
                iteration=iteration,
                log_likelihood=current_log_likelihood,
                log_likelihood_change=log_likelihood_change,
                parameter_change=param_change,
                converged=converged
            )
            self.statistics_history.append(stats)
            
            if self.verbose:
                print(f"Iteration {iteration}: LL={current_log_likelihood:.6f}, "
                      f"ΔLL={log_likelihood_change:.2e}, "
                      f"Δparams={param_change:.2e}")
            
            # Update parameters for next iteration
            self.current_params = new_params
            prev_log_likelihood = current_log_likelihood
            
            # Check convergence
            if converged:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        else:
            if self.verbose:
                print(f"Reached maximum iterations ({self.max_iterations})")
        
        # Ensure we have best parameters (in case first iteration was best)
        if self.best_params is None:
            self.best_params = self.current_params
        
        return EMResults(
            final_params=self.current_params,
            best_params=self.best_params,
            best_iteration=self.best_iteration,
            best_log_likelihood=self.best_log_likelihood,
            statistics_history=self.statistics_history
        )

    def _apply_damping(self, new_params: EMParameters, current_params: EMParameters) -> EMParameters:
        """Apply damping to parameter updates.
        
        Implements weighted average: θ_new = (1-α) * θ_old + α * θ_M_step
        
        Parameters
        ----------
        new_params : EMParameters
            Parameters from M-step
        current_params : EMParameters
            Current parameters before M-step
            
        Returns
        -------
        EMParameters
            Damped parameters
        """
        alpha = self.current_damping_factor
        
        return EMParameters(
            x0=(1 - alpha) * current_params.x0 + alpha * new_params.x0,
            Sigma=(1 - alpha) * current_params.Sigma + alpha * new_params.Sigma,
            F=(1 - alpha) * current_params.F + alpha * new_params.F,
            u=(1 - alpha) * current_params.u + alpha * new_params.u
        )

    def _adaptive_damping(self, current_log_likelihood: float, prev_log_likelihood: float) -> float:
        """Adaptive damping factor adjustment based on convergence behavior.
        
        Parameters
        ----------
        current_log_likelihood : float
            Current expected log-likelihood Q(Θ, Θ^{(i)})
        prev_log_likelihood : float
            Previous expected log-likelihood Q(Θ, Θ^{(i-1)})
            
        Returns
        -------
        float
            Updated damping factor
        """
        log_likelihood_change = current_log_likelihood - prev_log_likelihood
        if log_likelihood_change > 0:
            self.consecutive_increases += 1
            self.consecutive_decreases = 0
        else:
            self.consecutive_decreases += 1
            self.consecutive_increases = 0
        
        if self.consecutive_increases > 3:
            self.current_damping_factor = min(self.current_damping_factor + 0.1, 1.0)
            self.consecutive_increases = 0
        elif self.consecutive_decreases > 3:
            self.current_damping_factor = max(self.current_damping_factor - 0.1, 0.1)
            self.consecutive_decreases = 0
        
        return self.current_damping_factor

    def _apply_constraints_to_parameters(self, parameters: EMParameters, 
                                       constraint_manager = None) -> EMParameters:
        """Apply constraints to parameter estimates.
        
        Parameters
        ----------
        parameters : EMParameters
            Parameter estimates to constrain
        constraint_manager : Optional[ConstraintManager]
            Constraint manager to apply syntax constraints
            
        Returns
        -------
        EMParameters
            Constrained parameter estimates
        """
        if constraint_manager is None:
            return parameters
        
        # Apply constraints to initial state
        x0_constrained = constraint_manager.apply_constraints(parameters.x0)
        
        return EMParameters(
            x0=x0_constrained,
            Sigma=parameters.Sigma,
            F=parameters.F, 
            u=parameters.u
        ) 
