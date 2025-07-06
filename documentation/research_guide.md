# Research Guide

## Overview

This guide explains how to conduct research experiments using the Adaptive Syntax Filter package. The package provides a complete framework for studying time-varying syntax rules in behavioral sequences.

**Research Group**: [Neural Syntax Lab](https://github.com/NeuralSyntaxLab) at the Weizmann Institute of Science

## Experimental Design

### Research Questions

The package is designed to address questions such as:

1. **Syntax Evolution**: How do syntax rules change over time?
2. **Context Dependence**: How do transition probabilities depend on history?
3. **Multi-timescale Dynamics**: What are the characteristic timescales of syntax changes?
4. **Real-time Detection**: Can syntax changes be detected in real-time?

### Experimental Parameters

Key parameters to consider:

- **Alphabet Size**: Number of distinct symbols/phrases
- **Markov Order**: How much history influences transitions
- **Sequence Length**: Length of individual sequences
- **Number of Sequences**: Total number of observations
- **Evolution Type**: How parameters change over time
- **Noise Level**: Amount of observational noise

## Running Experiments

### Basic Experiment

```bash
# Run a minimal experiment
python cli.py run --config configs/minimal.yml

# Run with custom parameters
python cli.py run --config configs/higher_order.yml
```

### Configuration Files

Create custom configuration files:

```yaml
experiment_id: my_experiment

data:
  alphabet: ['<', 'a', 'b', 'c', '>']
  order: 2
  n_sequences: 100
  max_length: 25
  evolution_type: linear

em:
  max_iterations: 20
  tolerance: 1e-4
  regularization_lambda: 0.001
  damping_factor: 0.1
  adaptive_damping: true
```

### Experiment Types

#### 1. Parameter Recovery

Test if the algorithm can recover known parameters:

```python
from adaptive_syntax_filter.data import SequenceGenerator
from adaptive_syntax_filter.core import EMAlgorithm, StateSpaceManager

# Generate data with known parameters
generator = SequenceGenerator(alphabet, order=2)
sequences = generator.generate_sequences(n_sequences=50)

# Run EM algorithm
state_manager = StateSpaceManager(alphabet_size=len(alphabet), markov_order=2)
em = EMAlgorithm(state_manager, max_iterations=20)
results = em.fit(sequences)

# Compare recovered vs true parameters
print(f"Converged: {results.converged}")
print(f"Final log-likelihood: {results.best_log_likelihood}")
```

#### 2. Evolution Tracking

Study how syntax rules evolve over time:

```python
# Generate data with temporal evolution
generator = SequenceGenerator(alphabet, order=1)
sequences = generator.generate_sequences_with_evolution(
    n_sequences=100,
    evolution_type="linear",
    evolution_rate=0.1
)

# Run analysis
results = em.fit(sequences)

# Analyze evolution
from adaptive_syntax_filter.viz import LogitEvolutionDashboard
dashboard = LogitEvolutionDashboard(alphabet, order=1)
dashboard.plot_evolution(results.final_params)
```

#### 3. Higher-Order Analysis

Study context-dependent transitions:

```python
# Higher-order model
state_manager = StateSpaceManager(alphabet_size=len(alphabet), markov_order=3)
em = EMAlgorithm(state_manager, max_iterations=30)
results = em.fit(sequences)

# Analyze context-specific transitions
from adaptive_syntax_filter.viz import ProbabilityEvolutionAnalyzer
analyzer = ProbabilityEvolutionAnalyzer(alphabet, order=3)
analyzer.analyze_transitions(results.final_params)
```

## Data Analysis

### Convergence Analysis

```python
# Check convergence
stats = results.statistics_history
for stat in stats:
    print(f"Iteration {stat.iteration}: LL={stat.log_likelihood:.4f}")

# Plot convergence
import matplotlib.pyplot as plt
iterations = [s.iteration for s in stats]
log_likelihoods = [s.log_likelihood for s in stats]
plt.plot(iterations, log_likelihoods)
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('EM Convergence')
plt.show()
```

### Parameter Analysis

```python
# Analyze final parameters
final_params = results.final_params

# State transition matrix
print("F matrix shape:", final_params.F.shape)
print("F matrix structure:", np.linalg.matrix_rank(final_params.F))

# Process noise
print("Sigma diagonal:", np.diag(final_params.Sigma))

# Initial state
print("x0 norm:", np.linalg.norm(final_params.x0))
```

### Visualization

#### Logit Evolution

```python
from adaptive_syntax_filter.viz import LogitEvolutionDashboard

dashboard = LogitEvolutionDashboard(alphabet, order=1)
fig = dashboard.create_evolution_plot(results.final_params)
fig.savefig('logit_evolution.png', dpi=300, bbox_inches='tight')
```

#### Probability Evolution

```python
from adaptive_syntax_filter.viz import ProbabilityEvolutionAnalyzer

analyzer = ProbabilityEvolutionAnalyzer(alphabet, order=1)
fig = analyzer.create_evolution_plot(results.final_params)
fig.savefig('probability_evolution.png', dpi=300, bbox_inches='tight')
```

#### Performance Assessment

```python
from adaptive_syntax_filter.viz import PerformanceAssessment

assessor = PerformanceAssessment()
metrics = assessor.assess_performance(results, sequences)
print("Performance metrics:", metrics)
```

## Advanced Experiments

### Real Data Analysis

```python
# Load real birdsong data
sequences = load_birdsong_data('path/to/data.csv')

# Preprocess data
processed_sequences = preprocess_sequences(sequences)

# Run analysis
results = em.fit(processed_sequences)
```

### Comparative Studies

```python
# Compare different orders
orders = [1, 2, 3]
results_by_order = {}

for order in orders:
    state_manager = StateSpaceManager(alphabet_size=len(alphabet), markov_order=order)
    em = EMAlgorithm(state_manager, max_iterations=20)
    results_by_order[order] = em.fit(sequences)

# Compare performance
for order, results in results_by_order.items():
    print(f"Order {order}: LL={results.best_log_likelihood:.4f}")
```

### Robustness Analysis

```python
# Test with different noise levels
noise_levels = [0.0, 0.1, 0.2, 0.5]
results_by_noise = {}

for noise in noise_levels:
    noisy_sequences = add_noise(sequences, noise)
    results_by_noise[noise] = em.fit(noisy_sequences)

# Analyze robustness
for noise, results in results_by_noise.items():
    print(f"Noise {noise}: LL={results.best_log_likelihood:.4f}")
```

## Best Practices

### 1. Parameter Selection

- **Start Simple**: Begin with first-order models
- **Gradual Complexity**: Increase order gradually
- **Convergence Monitoring**: Always check convergence
- **Parameter Validation**: Validate recovered parameters

### 2. Data Quality

- **Sufficient Data**: Ensure enough sequences for reliable estimation
- **Data Preprocessing**: Clean and validate input data
- **Outlier Detection**: Identify and handle outliers
- **Missing Data**: Handle missing observations appropriately

### 3. Computational Considerations

- **Memory Usage**: Monitor memory for high-order models
- **Convergence Time**: Set reasonable iteration limits
- **Numerical Stability**: Use appropriate regularization
- **Reproducibility**: Set random seeds for reproducible results

### 4. Validation

- **Synthetic Tests**: Validate with known parameters
- **Cross-validation**: Use multiple data splits
- **Robustness Tests**: Test with different noise levels
- **Interpretability**: Ensure results are interpretable

## Troubleshooting

### Common Issues

1. **Non-convergence**: Increase iterations or adjust damping
2. **Numerical Issues**: Add regularization or check data quality
3. **Memory Problems**: Reduce model complexity or use smaller datasets
4. **Poor Performance**: Check data preprocessing and parameter selection

### Debugging

```python
# Enable verbose output
em = EMAlgorithm(state_manager, verbose=True)

# Check intermediate results
for i, stat in enumerate(results.statistics_history):
    if i % 5 == 0:  # Print every 5th iteration
        print(f"Iteration {stat.iteration}: LL={stat.log_likelihood:.4f}")
```

## Publication Guidelines

### Reproducible Research

1. **Version Control**: Use git for code versioning
2. **Environment**: Document Python and package versions
3. **Random Seeds**: Set seeds for reproducible results
4. **Configuration**: Save all experiment configurations

### Documentation

1. **Method Description**: Document experimental setup
2. **Parameter Justification**: Explain parameter choices
3. **Results Interpretation**: Provide clear interpretation
4. **Limitations**: Discuss limitations and assumptions

### Data Sharing

1. **Synthetic Data**: Provide code to generate synthetic data
2. **Real Data**: Share anonymized data when possible
3. **Results**: Share processed results and visualizations
4. **Code**: Make code publicly available

## Contact

For research questions and collaboration:

- **Maintainer**: [Nadav Elami](mailto:nadav.elami@weizmann.ac.il)
- **GitHub**: [Nadav-Elami](https://github.com/Nadav-Elami)
- **Research Group**: [Neural Syntax Lab](https://github.com/NeuralSyntaxLab)
- **Repository**: https://github.com/Nadav-Elami/adaptive-syntax-filter

## References

1. Dempster, A.P. et al. (1977). Maximum likelihood from incomplete data via the EM algorithm.
2. Kalman, R.E. (1960). A new approach to linear filtering and prediction problems.
3. Bishop, C.M. (2006). Pattern Recognition and Machine Learning. 