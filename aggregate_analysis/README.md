# Aggregate Analysis Tools

This directory contains the core aggregate analysis tools for the Adaptive Syntax Filter project.

## Files

### Core Analysis Scripts
- `aggregate_analysis.py` - Main aggregate analysis script with comprehensive statistical analysis
- `aggregate_analysis_utils.py` - Utilities for aggregate analysis (HDF5 data management, statistical analysis)
- `aggregate_analysis_optimized.py` - Optimized version with deadlock fixes and improved performance
- `aggregate_analysis_clean.py` - Clean version without emoji characters for better compatibility

### Supporting Tools
- `generate_aggregate_analysis.py` - Script for generating aggregate analysis for specific configurations
- `batch_processing.py` - Batch processing utilities for large-scale experiments
- `create_comprehensive_plots.py` - Plot generation script for comprehensive visualizations
- `result_archiving.py` - Result archiving utilities

## Usage

### Basic Aggregate Analysis
```bash
# Run with default configuration
python aggregate_analysis.py --config-id 1 --n-seeds 100 --test-mode

# Run with custom parameters
python aggregate_analysis.py --config-id all --n-seeds 20000 --parallel 8

# Run optimized version
python aggregate_analysis_optimized.py --config-id 3 --n-seeds 1000 --parallel 4

# Run clean version
python aggregate_analysis_clean.py --config-id 4 --n-seeds 500 --parallel 2
```

### Generate Specific Analysis
```bash
# Generate analysis for config 3
python generate_aggregate_analysis.py --config-id 3

# Generate with custom output directory
python generate_aggregate_analysis.py --config-id 4 --output-dir results/my_analysis
```

### Create Plots
```bash
# Create comprehensive plots for config 3
python create_comprehensive_plots.py --config-id 3
```

## Configuration

The scripts support 4 experimental configurations:
- Config 1: Linear evolution, 1st order, short sequences
- Config 2: Linear evolution, 2nd order, long sequences  
- Config 3: Sigmoid evolution, 1st order, medium sequences
- Config 4: Sigmoid evolution, 2nd order, medium sequences

## Output

Results are saved in `results/aggregate_analysis/` with:
- Statistical tests CSV files
- Aggregate summary HDF5 files
- Example plots (evolution and convergence)
- Comprehensive cross-entropy comparison plots

## Dependencies

- numpy, pandas, matplotlib, seaborn
- h5py for data storage
- scipy for statistical analysis
- tqdm for progress tracking 