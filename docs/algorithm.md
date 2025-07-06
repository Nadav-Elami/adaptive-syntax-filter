# Algorithm Documentation

## Mathematical Foundations

### Overview

The Adaptive Syntax Filter implements a state-space filtering approach for learning time-varying syntax rules in behavioral sequences. The core algorithm combines:

1. **Expectation-Maximization (EM)** for parameter estimation
2. **Kalman Filtering** for state estimation
3. **Higher-Order Markov Models** for context-dependent transitions
4. **Soft-Max Observation Model** for probability computation

### State Space Model

The state vector **x_k** represents logit parameters for transition probabilities:

```
x_k ∈ ℝ^{R^(order+1)}
```

where:
- **R** = alphabet size
- **order** = Markov order (1, 2, 3, ...)

For a first-order model with 3 symbols: x_k ∈ ℝ^9
For a second-order model with 3 symbols: x_k ∈ ℝ^27

### State Transition Model

The state evolves according to:

```
x_k = F * x_{k-1} + u + w_k
```

where:
- **F** = block-diagonal state transition matrix
- **u** = control vector
- **w_k** ~ N(0, Σ) = process noise

### Block-Diagonal Structure

The state transition matrix F has block-diagonal structure:

```
F = [F₁  0   0  ]
    [0   F₂  0  ]
    [0   0   F₃ ]
```

Each block F_i corresponds to a context and governs transitions from that context.

### Observation Model

Observations are categorical symbols from the alphabet. The likelihood is computed using soft-max probabilities:

```
P(y_k | x_k) = ∏_m softmax(x_k[context_m])
```

where context_m represents the context for observation m.

## EM Algorithm

### E-Step: Kalman Filtering

The E-step computes smoothed state estimates using forward filtering and backward smoothing:

#### Forward Filtering (Equations 4.1-4.6)

1. **Prediction Step**:
   ```
   x_{k|k-1} = F * x_{k-1|k-1} + u
   W_{k|k-1} = F * W_{k-1|k-1} * F^T + Σ
   ```

2. **Update Step**:
   ```
   δ_k = Σ_m [e_{y_m,y_{m-1}} - f(x_{k|k-1})]
   W_{k|k}^{-1} = W_{k|k-1}^{-1} + H_k
   x_{k|k} = x_{k|k-1} + W_{k|k} * δ_k
   ```

where:
- e_{y_m,y_{m-1}} = one-hot indicator for observed transition
- f(x) = soft-max probabilities
- H_k = Hessian of log-likelihood

#### Backward Smoothing (Equations 5.1-5.4)

1. **Smoothing Equations**:
   ```
   x_{k|K} = x_{k|k} + W_{k|k} * F^T * W_{k+1|k}^{-1} * (x_{k+1|K} - x_{k+1|k})
   W_{k|K} = W_{k|k} - W_{k|k} * F^T * W_{k+1|k}^{-1} * F * W_{k|k}
   ```

### M-Step: Parameter Updates

The M-step updates all parameters using smoothed estimates:

#### Initial State Update (Equation 6.1)
```
x_0^* = x_{1|K}
```

#### Dynamics Updates (Equations 6.2-6.4)
```
F^* = (Σ_k x_{k|K} * x_{k-1|K}^T) * (Σ_k W_{k-1|K} + x_{k-1|K} * x_{k-1|K}^T)^{-1}
u^* = (1/K) * Σ_k (x_{k|K} - F^* * x_{k-1|K})
```

#### Process Noise Update (Equation 6.5)
```
Σ^* = (1/K) * Σ_k [W_{k|K} + (x_{k|K} - F^* * x_{k-1|K} - u^*) * (x_{k|K} - F^* * x_{k-1|K} - u^*)^T]
```

## Higher-Order Models

### Context Encoding

For order k, contexts are encoded as integers:

```
context_idx = Σ_{i=0}^{k-1} symbol_idx[i] * R^(k-1-i)
```

### State Space Organization

The state vector is organized in blocks:

```
x_k = [x_k^0, x_k^1, ..., x_k^{R^k-1}]
```

where each block x_k^i contains R logit parameters for context i.

### Transition Probabilities

For context c and target symbol s:

```
P(s | c) = softmax(x_k[c*R + s])
```

## Implementation Details

### Numerical Stability

1. **Regularization**: Small diagonal terms added to covariance matrices
2. **Damping**: Parameter updates are damped to prevent oscillations
3. **Adaptive Damping**: Damping factor adjusted based on convergence

### Constraint Handling

1. **Forbidden Transitions**: Set to -∞ in logit space
2. **Context Validation**: Invalid contexts are excluded
3. **Probability Normalization**: Ensured within each context block

### Memory Management

1. **Block Operations**: Matrix operations performed block-wise
2. **Sparse Storage**: Zero blocks not stored explicitly
3. **Efficient Indexing**: Context-to-index mapping optimized

## Convergence Criteria

The EM algorithm converges when:

1. **Log-likelihood change** < tolerance
2. **Parameter change norm** < tolerance
3. **Maximum iterations** reached

## Performance Considerations

### Computational Complexity

- **E-step**: O(K * R^(2*order+2))
- **M-step**: O(K * R^(2*order+2))
- **Memory**: O(K * R^(2*order+2))

### Optimization Strategies

1. **Block-wise computation** reduces memory usage
2. **Early stopping** for convergence
3. **Parallel processing** for independent contexts
4. **Caching** of frequently computed values

## Validation

### Synthetic Data Tests

1. **Parameter Recovery**: Known parameters recovered from synthetic data
2. **Convergence**: Algorithm converges to reasonable solutions
3. **Robustness**: Performance under various noise levels

### Real Data Validation

1. **Birdsong Analysis**: Applied to canary song sequences
2. **Syntax Detection**: Identifies known syntax rules
3. **Evolution Tracking**: Detects temporal changes

## References

1. Kalman, R.E. (1960). A new approach to linear filtering and prediction problems.
2. Dempster, A.P. et al. (1977). Maximum likelihood from incomplete data via the EM algorithm.
3. Rauch, H.E. et al. (1965). Maximum likelihood estimates of linear dynamic systems. 