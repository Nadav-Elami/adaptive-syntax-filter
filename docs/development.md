# Development Guide

## Overview

This guide provides information for developers who want to contribute to the Adaptive Syntax Filter project. We welcome contributions from the research community!

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- pip package manager

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-syntax-filter.git
cd adaptive-syntax-filter

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
pytest
```

### Development Dependencies

The project uses several development tools:

- **pytest**: Testing framework
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **pytest-cov**: Coverage reporting

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (black default)
- **Import sorting**: isort profile "black"
- **Type hints**: Encouraged but not required
- **Docstrings**: NumPy style for functions and classes

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/
```

### Pre-commit Hooks

Set up pre-commit hooks for automatic formatting:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/adaptive_syntax_filter

# Run specific test files
pytest tests/test_core/test_em_algorithm.py

# Run with verbose output
pytest -v
```

### Writing Tests

Follow these guidelines for writing tests:

1. **Test Structure**: Use descriptive test names
2. **Test Isolation**: Each test should be independent
3. **Test Coverage**: Aim for high coverage of core functionality
4. **Test Data**: Use synthetic data for testing

Example test:

```python
def test_em_algorithm_convergence():
    """Test that EM algorithm converges for simple case."""
    # Arrange
    state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
    em = EMAlgorithm(state_manager, max_iterations=10)
    observations = [np.array([0, 1, 2, 0])]
    
    # Act
    results = em.fit(observations)
    
    # Assert
    assert results.converged
    assert results.best_log_likelihood > -np.inf
```

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test computational efficiency
- **Edge Case Tests**: Test boundary conditions

## Documentation

### Code Documentation

All public functions and classes should have docstrings:

```python
def compute_evolution_trajectory(x_init, x_final, n_sequences, evolution_type):
    """Compute parameter evolution trajectory.
    
    Parameters
    ----------
    x_init : np.ndarray
        Initial parameter vector
    x_final : np.ndarray
        Final parameter vector
    n_sequences : int
        Number of sequences in trajectory
    evolution_type : str
        Type of evolution ('linear', 'exponential', etc.)
        
    Returns
    -------
    np.ndarray
        Parameter trajectory of shape (n_params, n_sequences)
    """
    # Implementation here
```

### API Documentation

Update API documentation when adding new features:

1. **Function signatures**: Document all parameters and return values
2. **Examples**: Provide usage examples
3. **Type hints**: Use type hints for clarity
4. **Error handling**: Document possible exceptions

## Contributing Process

### 1. Issue Reporting

Before making changes:

1. **Check existing issues**: Search for similar issues
2. **Create detailed issue**: Include reproduction steps
3. **Label appropriately**: Use appropriate labels
4. **Provide context**: Include system information

### 2. Feature Development

For new features:

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Implement changes**: Follow coding standards
3. **Add tests**: Include comprehensive tests
4. **Update documentation**: Update relevant docs
5. **Test thoroughly**: Run full test suite

### 3. Bug Fixes

For bug fixes:

1. **Reproduce the bug**: Create minimal reproduction
2. **Fix the issue**: Implement the fix
3. **Add regression test**: Prevent future regressions
4. **Test the fix**: Verify the fix works

### 4. Pull Request Process

1. **Fork the repository**: Create your own fork
2. **Create feature branch**: Work on a dedicated branch
3. **Make changes**: Implement your changes
4. **Run tests**: Ensure all tests pass
5. **Update documentation**: Update relevant docs
6. **Submit PR**: Create pull request with description

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] Code comments added

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests pass locally
- [ ] Documentation is clear
```

## Research Contributions

### Algorithm Improvements

We welcome contributions to:

1. **Numerical Stability**: Improve convergence properties
2. **Performance**: Optimize computational efficiency
3. **New Models**: Add new evolution types
4. **Validation**: Add new validation methods

### Example: Adding New Evolution Type

```python
class OscillatoryEvolution(EvolutionManager):
    """Oscillatory evolution with frequency and amplitude."""
    
    def __init__(self, frequency=1.0, amplitude=0.1, trend_weight=0.8):
        self.frequency = frequency
        self.amplitude = amplitude
        self.trend_weight = trend_weight
    
    def compute_trajectory(self, x_init, x_final, n_sequences):
        """Compute oscillatory trajectory."""
        # Implementation here
        pass
```

### Validation Contributions

Add new validation methods:

```python
def validate_parameter_recovery(estimated_params, true_params, tolerance=1e-3):
    """Validate parameter recovery accuracy."""
    # Implementation here
    pass
```

## Code Review Process

### Review Criteria

1. **Functionality**: Does the code work correctly?
2. **Performance**: Is the implementation efficient?
3. **Readability**: Is the code clear and well-documented?
4. **Testing**: Are there adequate tests?
5. **Documentation**: Is the documentation updated?

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive
- [ ] Documentation is clear
- [ ] Performance is acceptable
- [ ] Error handling is appropriate
- [ ] Security considerations addressed

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. **Update version**: Update version in pyproject.toml
2. **Update changelog**: Document changes
3. **Run full tests**: Ensure all tests pass
4. **Update documentation**: Update relevant docs
5. **Create release**: Tag and release on GitHub

## Troubleshooting

### Common Issues

1. **Import errors**: Check virtual environment activation
2. **Test failures**: Check dependencies and Python version
3. **Performance issues**: Profile code and optimize
4. **Memory issues**: Check for memory leaks

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pdb for debugging
import pdb; pdb.set_trace()

# Profile performance
import cProfile
cProfile.run('your_function()')
```

## Community Guidelines

### Code of Conduct

1. **Be respectful**: Treat others with respect
2. **Be constructive**: Provide constructive feedback
3. **Be inclusive**: Welcome diverse perspectives
4. **Be professional**: Maintain professional communication

### Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for general questions
- **Pull Requests**: Use PRs for code contributions
- **Documentation**: Contribute to documentation improvements

## Resources

### Documentation

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [NumPy Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [pytest Documentation](https://docs.pytest.org/)

### Tools

- [Black Code Formatter](https://black.readthedocs.io/)
- [isort Import Sorter](https://pycqa.github.io/isort/)
- [flake8 Linter](https://flake8.pycqa.org/)

### Research Resources

- [Kalman Filter Tutorial](https://www.kalmanfilter.net/)
- [EM Algorithm Tutorial](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
- [Markov Models](https://en.wikipedia.org/wiki/Markov_model)

## Contact

For development questions:

- **Issues**: https://github.com/yourusername/adaptive-syntax-filter/issues
- **Discussions**: https://github.com/yourusername/adaptive-syntax-filter/discussions
- **Email**: [your.email@example.com](mailto:your.email@example.com)

Thank you for contributing to the Adaptive Syntax Filter project! 