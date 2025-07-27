# Research Pipeline Scripts

This directory contains the main research pipeline scripts for the Adaptive Syntax Filter project.

## Files

### Core Research Pipeline
- `research_pipeline.py` - Main research pipeline orchestrating data generation, EM fitting, and visualization
- `cli.py` - Main command-line interface for running experiments
- `config_cli.py` - Configuration loading utilities (YAML/JSON)
- `export_cli.py` - Export functionality for figures and results

## Usage

### Main Research Pipeline
```bash
# Run with configuration file
python research_pipeline.py configs/config3.yml

# Run with custom parameters
python research_pipeline.py --config configs/my_config.yml --output-dir results/my_experiment
```

### Command Line Interface
```bash
# Run experiment
python cli.py run configs/config3.yml

# Export results
python export_cli.py --experiment-id my_experiment

# Load configuration
python config_cli.py load configs/config3.yml
```

## Configuration

Scripts use YAML configuration files in the `configs/` directory:
- `configs/minimal.yml` - Minimal configuration for testing
- `configs/medium.yml` - Medium complexity configuration
- `configs/higher_order.yml` - Higher-order Markov model configuration
- `configs/aggregate_config_*.yml` - Aggregate analysis configurations

## Dependencies

- Core library: `src/adaptive_syntax_filter/`
- Configuration: YAML/JSON files
- Visualization: matplotlib, seaborn
- Data processing: numpy, pandas

## Integration

These scripts integrate with:
- Core algorithms in `src/adaptive_syntax_filter/core/`
- Data generation in `src/adaptive_syntax_filter/data/`
- Visualization in `src/adaptive_syntax_filter/viz/`
- Aggregate analysis tools in `../aggregate_analysis/` 