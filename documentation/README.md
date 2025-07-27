# Adaptive Syntax Filter

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Scientific-orange.svg)](https://github.com/Nadav-Elami/adaptive-syntax-filter)

A Python research package implementing an **adaptive Kalman–EM algorithm** for learning time-varying syntax rules in behavioral sequences, with applications to canary song analysis and real-time behavioral monitoring.

**Research Group**: [Neural Syntax Lab](https://github.com/NeuralSyntaxLab) at the Weizmann Institute of Science

## 🎯 Research Overview

Canary songs exhibit rich, history-dependent phrase sequencing whose syntax rules drift on timescales from a few songs to several weeks. This package presents an adaptive Kalman–EM algorithm that learns:

- **Block-diagonal state-transition matrix F** and control vector **u**
- **Process variance Sigma** and initial logit vector **x0**
- **Time-varying logit vectors** that govern soft-max transition probabilities between song phrases

By unifying Bayes' rule, the Chapman–Kolmogorov equation, and a soft-max observation model, the filter–smoother achieves sub-song latency and detects syntax changes across multiple timescales—an essential step toward real-time behavioral monitoring in songbirds.

**Core Research Question:** Can a state-space filtering approach accurately characterize canary-syntax dynamics across timescales from minutes to weeks?

## 🚀 Features

- **Adaptive Kalman Filtering**: Real-time parameter estimation with block-diagonal dynamics
- **Higher-Order Markov Models**: Support for 1st through 5th order Markov chains
- **Comprehensive Testing**: Full test suite with 300+ tests
- **Research Pipeline**: Complete experimental framework with visualization
- **Jupyter Integration**: Notebooks for experimentation and analysis
- **Modular Architecture**: Clean separation of core algorithms, data processing, and visualization

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Nadav-Elami/adaptive-syntax-filter.git
cd adaptive-syntax-filter

# Install in development mode
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Quick Verification

```bash
# Run the test suite
pytest

# Run a quick demo
python cli.py run --config configs/minimal.yml
```

## 🎮 Quick Start

### Basic Usage

```python
import numpy as np
from adaptive_syntax_filter.core import EMAlgorithm, StateSpaceManager
from adaptive_syntax_filter.data import SequenceGenerator

# Create a simple first-order Markov model
alphabet = ['<', 'a', 'b', '>']
state_manager = StateSpaceManager(alphabet_size=len(alphabet), markov_order=1)

# Generate synthetic data
generator = SequenceGenerator(alphabet, order=1)
sequences = generator.generate_sequences(n_sequences=10, max_length=20)

# Initialize and run EM algorithm
em = EMAlgorithm(state_manager, max_iterations=10)
results = em.fit(sequences)

print(f"Converged: {results.final_params}")
```

### Command Line Interface

```bash
# Run with default configuration
python cli.py run --config configs/minimal.yml

# Run with custom parameters
python cli.py run --config configs/higher_order.yml

# Export results
python export_cli.py --experiment_id my_experiment
```

## 📁 Project Structure

```
adaptive-syntax-filter/
├── src/adaptive_syntax_filter/     # Core package
│   ├── core/                       # EM algorithm, Kalman filter
│   ├── data/                       # Data generation, constraints
│   ├── viz/                        # Visualization tools
│   └── config/                     # Configuration management
├── tests/                          # Comprehensive test suite
├── configs/                        # Experiment configurations
├── notebooks/                      # Jupyter notebooks
├── documentation/                  # Complete documentation
├── results/                        # Experiment results
└── cli.py                         # Command line interface
```

## 🔬 Research Components

### Core Algorithms

- **EM Algorithm**: Expectation-maximization for parameter estimation
- **Kalman Filter**: Forward filtering and RTS smoothing
- **State Space Management**: Higher-order Markov model support
- **Observation Model**: Soft-max probability computation

### Data Processing

- **Sequence Generation**: Synthetic birdsong data generation
- **Constraint System**: Syntax rule enforcement
- **Temporal Evolution**: Time-varying parameter models
- **Dataset Builder**: Research dataset creation

### Visualization

- **Logit Evolution**: Parameter trajectory visualization
- **Probability Evolution**: Transition probability analysis
- **Performance Assessment**: Algorithm evaluation tools
- **Publication Figures**: Publication-ready plots

## 📊 Configuration

The package uses YAML configuration files for experiments:

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

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/adaptive_syntax_filter

# Run specific test categories
pytest tests/test_core/
pytest tests/test_data/
pytest tests/test_viz/
```

## 📚 Documentation

- **[Algorithm Documentation](algorithm.md)**: Mathematical foundations and implementation details
- **[Research Guide](research_guide.md)**: How to conduct experiments
- **[Development Guide](development.md)**: Contributing guidelines

## 🎯 Research Applications

### Birdsong Analysis
- Real-time syntax rule detection
- Multi-timescale evolution tracking
- Behavioral monitoring systems

### General Sequence Analysis
- Time-varying Markov models
- Adaptive filtering applications
- Behavioral pattern recognition

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](development.md) for details on:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Research contribution process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- [Neural Syntax Lab](https://github.com/NeuralSyntaxLab) at the Weizmann Institute of Science
- Research collaborators and advisors
- Open source community

## 📞 Contact

- **Maintainer**: [Nadav Elami](mailto:nadav.elami@weizmann.ac.il)
- **GitHub**: [Nadav-Elami](https://github.com/Nadav-Elami)
- **Repository**: https://github.com/Nadav-Elami/adaptive-syntax-filter
- **Issues**: https://github.com/Nadav-Elami/adaptive-syntax-filter/issues
- **Research Group**: [Neural Syntax Lab](https://github.com/NeuralSyntaxLab)

---

**Note**: This is a research package. For production use, additional validation and testing may be required. 
