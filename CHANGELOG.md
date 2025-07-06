# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation structure
- Development guide with contributing guidelines
- Research guide for experimental design
- Algorithm documentation with mathematical foundations
- MIT license and proper project structure
- GitHub-ready configuration files

### Changed
- Reorganized documentation into new `documentation/` folder
- Updated project structure for GitHub publication
- Improved test coverage and organization
- Enhanced CLI interface and configuration management

### Fixed
- Higher-order model context encoding issues
- Numerical stability improvements in EM algorithm
- Probability normalization for higher-order models
- Memory management for large state spaces

## [0.1.0] - 2025-07-06

### Added
- Core EM algorithm implementation
- Kalman filtering with forward/backward smoothing
- Higher-order Markov model support (1st-5th order)
- State space management for context-dependent transitions
- Soft-max observation model for probability computation
- Comprehensive test suite (300+ tests)
- Research pipeline with visualization tools
- Command-line interface for experiments
- Configuration management with YAML files
- Data generation and constraint system
- Temporal evolution models (linear, exponential, sigmoid, oscillatory)
- Visualization tools for logit and probability evolution
- Performance assessment and validation tools
- Jupyter notebook integration
- Publication-ready figure generation

### Features
- Adaptive Kalman-EM algorithm for time-varying syntax rules
- Block-diagonal state transition matrix support
- Real-time parameter estimation
- Multi-timescale evolution tracking
- Birdsong sequence analysis capabilities
- Modular architecture for research extensibility
- Comprehensive error handling and validation
- Numerical stability improvements with damping
- Memory-efficient implementation for large datasets

### Technical Details
- Python 3.11+ compatibility
- NumPy/SciPy for numerical computations
- Matplotlib for visualization
- pytest for testing framework
- YAML for configuration management
- Modular design for easy extension

## [0.0.1] - 2025-07-01

### Added
- Initial project structure
- Basic EM algorithm implementation
- Simple Kalman filter
- Basic test framework
- Project configuration files

---

## Version History

- **0.0.1**: Initial development version with basic functionality
- **0.1.0**: First stable release with comprehensive features
- **Unreleased**: Documentation and GitHub preparation

## Contributing

Please see [CONTRIBUTING.md](documentation/development.md) for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 