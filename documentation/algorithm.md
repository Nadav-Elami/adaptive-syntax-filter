# Algorithm Documentation

## Overview

The Adaptive Syntax Filter implements a novel **adaptive Kalman–EM algorithm** for learning time-varying syntax rules in behavioral sequences. This algorithm unifies Bayesian inference, state-space modeling, and expectation-maximization to provide real-time parameter estimation with sub-song latency.

**Research Group**: [Neural Syntax Lab](https://github.com/NeuralSyntaxLab) at the Weizmann Institute of Science

## Mathematical Foundations

### Problem Formulation

Given a sequence of observations $y_1, y_2, \ldots, y_T$ where each $y_t$ is a symbol from alphabet $\mathcal{A}$, we model the underlying syntax as a time-varying Markov process with parameters that evolve according to a state-space model.

### State-Space Model

The algorithm uses a block-diagonal state-transition matrix $F$ and control vector $u$ to model parameter evolution:

$$\mathbf{x}_t = F\mathbf{x}_{t-1} + \mathbf{u} + \mathbf{w}_t$$

where:
- $\mathbf{x}_t$ is the logit vector at time $t$
- $F$ is the block-diagonal state-transition matrix
- $\mathbf{u}$ is the control vector
- $\mathbf{w}_t \sim \mathcal{N}(0, \Sigma)$ is process noise

### Observation Model

The observation model uses a soft-max function to convert logits to transition probabilities:

$$P(y_t | y_{t-1}, \ldots, y_{t-k}) = \frac{\exp(\mathbf{x}_t^T \mathbf{e}_{y_t})}{\sum_{a \in \mathcal{A}} \exp(\mathbf{x}_t^T \mathbf{e}_a)}$$

where:
- $\mathbf{e}_a$ is the one-hot encoding of symbol $a$
- $k$ is the Markov order
- $\mathbf{x}_t$ contains the logit parameters

### EM Algorithm

The algorithm alternates between:

1. **E-step**: Forward filtering and backward smoothing using the Kalman filter
2. **M-step**: Parameter updates using maximum likelihood estimation

#### E-step: Kalman Filtering

**Forward Pass (Filtering)**:
$$\hat{\mathbf{x}}_t = F\hat{\mathbf{x}}_{t-1} + \mathbf{u}$$
$$P_t = FP_{t-1}F^T + \Sigma$$

**Backward Pass (Smoothing)**:
$$\mathbf{x}_t^s = \hat{\mathbf{x}}_t + K_t(\mathbf{x}_{t+1}^s - F\hat{\mathbf{x}}_t)$$
$$P_t^s = P_t + K_t(P_{t+1}^s - FP_tF^T - \Sigma)K_t^T$$

where $K_t$ is the Kalman gain.

#### M-step: Parameter Updates

The M-step updates parameters using weighted averaging with damping:

$$\theta^{(i+1)} = \alpha \theta^{(i)} + (1-\alpha) \theta_{\text{ML}}$$

where:
- $\alpha$ is the damping factor
- $\theta_{\text{ML}}$ is the maximum likelihood estimate
- $\theta^{(i)}$ is the current parameter estimate

## Implementation Details

### State Space Management

The `StateSpaceManager` class handles the conversion between:
- **Symbol sequences**: Raw behavioral observations
- **State vectors**: Internal state representations
- **Parameter vectors**: Logit parameters for transition probabilities

```python
class StateSpaceManager:
    def __init__(self, alphabet_size: int, markov_order: int):
        self.alphabet_size = alphabet_size
        self.markov_order = markov_order
        self.state_dim = alphabet_size ** markov_order
```

### EM Algorithm Implementation

The core EM algorithm is implemented in the `EMAlgorithm` class:

```python
class EMAlgorithm:
    def __init__(self, state_manager: StateSpaceManager, 
                 max_iterations: int = 20,
                 tolerance: float = 1e-4,
                 damping_factor: float = 0.1):
        self.state_manager = state_manager
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping_factor = damping_factor
```

### Kalman Filter Implementation

The Kalman filter is implemented with numerical stability considerations:

```python
class KalmanFilter:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        
    def forward_pass(self, observations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward filtering pass."""
        # Implementation details...
        
    def backward_pass(self, filtered_means: List[np.ndarray], 
                     filtered_covs: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward smoothing pass."""
        # Implementation details...
```

## Numerical Stability

### Adaptive Damping

The algorithm uses adaptive damping to improve convergence:

```python
def compute_adaptive_damping(self, current_ll: float, previous_ll: float) -> float:
    """Compute adaptive damping factor based on likelihood improvement."""
    if previous_ll is None:
        return self.damping_factor
    
    improvement = current_ll - previous_ll
    if improvement > 0:
        # Good improvement, reduce damping
        return max(0.05, self.damping_factor * 0.9)
    else:
        # Poor improvement, increase damping
        return min(0.5, self.damping_factor * 1.1)
```

### Regularization

Regularization is applied to prevent overfitting:

```python
def apply_regularization(self, params: EMParameters) -> EMParameters:
    """Apply regularization to parameters."""
    # Add small diagonal to covariance matrices
    params.Sigma += self.regularization_lambda * np.eye(params.Sigma.shape[0])
    return params
```

## Higher-Order Models

### Context Encoding

For higher-order Markov models, context is encoded using a sliding window:

```python
def encode_context(self, sequence: np.ndarray, order: int) -> np.ndarray:
    """Encode context for higher-order model."""
    context_vectors = []
    for i in range(order, len(sequence)):
        context = sequence[i-order:i]
        context_vector = self.encode_single_context(context)
        context_vectors.append(context_vector)
    return np.array(context_vectors)
```

### State Space Expansion

The state space grows exponentially with order:

```python
def compute_state_dimension(self, alphabet_size: int, order: int) -> int:
    """Compute state space dimension for given order."""
    return alphabet_size ** order
```

## Performance Optimizations

### Block-Diagonal Structure

The state-transition matrix $F$ is constrained to be block-diagonal:

```python
def create_block_diagonal_F(self, block_sizes: List[int]) -> np.ndarray:
    """Create block-diagonal state-transition matrix."""
    total_dim = sum(block_sizes)
    F = np.zeros((total_dim, total_dim))
    
    start_idx = 0
    for block_size in block_sizes:
        end_idx = start_idx + block_size
        # Create random block (in practice, learned from data)
        F[start_idx:end_idx, start_idx:end_idx] = np.random.randn(block_size, block_size)
        start_idx = end_idx
    
    return F
```

### Sparse Computations

Sparse matrix operations are used where possible:

```python
def sparse_matrix_multiply(self, sparse_matrix: scipy.sparse.csr_matrix, 
                           dense_vector: np.ndarray) -> np.ndarray:
    """Efficient sparse matrix multiplication."""
    return sparse_matrix.dot(dense_vector)
```

## Validation and Testing

### Parameter Recovery

The algorithm is validated using synthetic data with known parameters:

```python
def validate_parameter_recovery(self, true_params: EMParameters, 
                               estimated_params: EMParameters, 
                               tolerance: float = 1e-3) -> bool:
    """Validate parameter recovery accuracy."""
    # Compare true vs estimated parameters
    f_error = np.linalg.norm(true_params.F - estimated_params.F)
    sigma_error = np.linalg.norm(true_params.Sigma - estimated_params.Sigma)
    x0_error = np.linalg.norm(true_params.x0 - estimated_params.x0)
    
    return (f_error < tolerance and 
            sigma_error < tolerance and 
            x0_error < tolerance)
```

### Convergence Monitoring

Convergence is monitored using multiple criteria:

```python
def check_convergence(self, current_ll: float, previous_ll: float, 
                     iteration: int) -> bool:
    """Check if EM algorithm has converged."""
    # Likelihood improvement
    ll_improvement = abs(current_ll - previous_ll)
    
    # Parameter change
    param_change = self.compute_parameter_change()
    
    # Maximum iterations
    max_iter_reached = iteration >= self.max_iterations
    
    return (ll_improvement < self.tolerance or 
            param_change < self.tolerance or 
            max_iter_reached)
```

## Research Applications

### Birdsong Analysis

The algorithm is specifically designed for birdsong analysis:

1. **Real-time Processing**: Sub-song latency for live monitoring
2. **Multi-timescale Dynamics**: Detects changes from minutes to weeks
3. **Context Dependence**: Models history-dependent transitions
4. **Adaptive Learning**: Continuously updates parameter estimates

### General Sequence Analysis

The algorithm can be applied to any time-varying Markov process:

1. **Behavioral Sequences**: Animal behavior patterns
2. **Language Processing**: Text sequence analysis
3. **Financial Time Series**: Market behavior modeling
4. **Biological Sequences**: DNA/RNA sequence analysis

## Limitations and Future Work

### Current Limitations

1. **Computational Complexity**: Exponential growth with Markov order
2. **Memory Requirements**: Large state spaces for high-order models
3. **Convergence Guarantees**: No theoretical convergence guarantees
4. **Model Assumptions**: Linear state-space model assumption

### Future Improvements

1. **Nonlinear Dynamics**: Extend to nonlinear state-space models
2. **Online Learning**: Implement true online parameter updates
3. **Model Selection**: Automatic order selection
4. **Parallel Processing**: GPU acceleration for large datasets

## References

1. Dempster, A.P., Laird, N.M., & Rubin, D.B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
2. Kalman, R.E. (1960). A new approach to linear filtering and prediction problems.
3. Bishop, C.M. (2006). Pattern Recognition and Machine Learning.
4. Murphy, K.P. (2012). Machine Learning: A Probabilistic Perspective.

## Contact

For questions about the algorithm implementation:

- **Maintainer**: [Nadav Elami](mailto:nadav.elami@weizmann.ac.il)
- **GitHub**: [Nadav-Elami](https://github.com/Nadav-Elami)
- **Research Group**: [Neural Syntax Lab](https://github.com/NeuralSyntaxLab)
- **Repository**: https://github.com/Nadav-Elami/adaptive-syntax-filter 