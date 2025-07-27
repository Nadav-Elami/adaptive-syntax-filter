# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-07-27

### Added
- Comprehensive aggregate analysis tools with statistical testing
- Support for 20,000+ seed experiments with parallel processing
- Cross-entropy analysis between EM, raw, and smoothed estimates
- Statistical significance testing with p-values and effect sizes
- Comprehensive visualization tools for evolution and convergence plots
- Aggregate analysis configuration files for 4 experimental conditions
- Result archiving and data management utilities
- Batch processing capabilities for large-scale experiments

### Changed
- **Major project reorganization**: Moved core scripts to organized directories
  - `aggregate_analysis/` - Core aggregate analysis tools
  - `scripts/` - Main research pipeline scripts
  - `diagnostics/` - Archived diagnostic scripts (removed from git)
- Improved project structure and maintainability
- Enhanced documentation with README files for each directory
- Updated version to 0.2.0 to reflect major reorganization

### Fixed
- Deadlock issues in parallel processing
- Memory management for large-scale experiments
- Unicode encoding issues in logging
- Import path organization after restructuring

### Removed
- Obsolete diagnostic scripts and temporary files
- External machine setup files (no longer needed)
- One-time analysis scripts and planning documents
- Diagnostic log files (preserved main aggregate_analysis.log)
- Redundant notebook files and workplan documents

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
- **0.2.0**: Major reorganization with aggregate analysis tools

## Contributing

Please see [CONTRIBUTING.md](documentation/development.md) for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 