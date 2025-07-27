#!/usr/bin/env python3
"""Generate aggregate analysis from existing HDF5 files.

This script loads the individual HDF5 files that were created by the optimized
aggregate analysis and generates the aggregate statistics and plots.
SINGLE-THREADED VERSION - Safe for running alongside other processes.
"""

import logging
import sys
from pathlib import Path
from typing import List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generate_aggregate_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def generate_aggregate_analysis_safe(config_id: int, output_dir: str = "results/aggregate_analysis"):
    """Generate aggregate analysis from existing HDF5 files - SAFE VERSION.
    
    Parameters
    ----------
    config_id : int
        Configuration ID (1-4)
    output_dir : str
        Base directory containing the results
    """
    from aggregate_analysis import AggregateAnalysisManager
    from aggregate_analysis_utils import HDF5DataManager
    
    logger.info(f"Generating aggregate analysis for config {config_id} (SAFE MODE)")
    logger.info(f"   Single-threaded processing - no parallel jobs")
    
    # Initialize managers
    analysis_manager = AggregateAnalysisManager(output_dir)
    data_manager = HDF5DataManager(output_dir)
    
    # Find all available seeds
    config_data_dir = Path(output_dir) / f"config_{config_id}_data"
    if not config_data_dir.exists():
        logger.error(f"Data directory not found: {config_data_dir}")
        return
    
    # Get all seed files
    seed_files = list(config_data_dir.glob("seed_*.h5"))
    if not seed_files:
        logger.error(f"No seed files found in {config_data_dir}")
        return
    
    # Extract seeds from filenames
    seeds = []
    for seed_file in seed_files:
        try:
            seed_str = seed_file.stem.replace("seed_", "")
            seed = int(seed_str)
            seeds.append(seed)
        except ValueError:
            logger.warning(f"Could not parse seed from filename: {seed_file}")
            continue
    
    seeds.sort()
    logger.info(f"Found {len(seeds)} completed seeds for config {config_id}")
    logger.info(f"   Seed range: {min(seeds)} to {max(seeds)}")
    
    # Load all results (single-threaded)
    logger.info("Loading results from HDF5 files...")
    results = analysis_manager._load_existing_results(config_id, seeds)
    
    if not results:
        logger.error("No results could be loaded")
        return
    
    logger.info(f"Successfully loaded {len(results)} results")
    
    # Generate aggregate analysis
    logger.info("Computing aggregate statistics...")
    try:
        analysis_manager.compute_and_save_aggregates(config_id, results)
        logger.info("Aggregate statistics computed and saved")
    except Exception as e:
        logger.error(f"Error computing aggregate statistics: {e}")
        return
    
    # Generate example plots (following original logic: first 10 seeds)
    logger.info("Creating example plots...")
    try:
        # Create plots for first 10 results (not 3) - following original logic
        analysis_manager.create_example_plots(config_id, results[:10])
        logger.info("Example plots created for first 10 seeds")
    except Exception as e:
        logger.error(f"Error creating example plots: {e}")
        # Continue anyway - plots are optional
    
    # Create comprehensive aggregate cross-entropy plot using ALL results
    logger.info("Creating comprehensive aggregate cross-entropy plot using ALL seeds...")
    try:
        analysis_manager._create_aggregate_cross_entropy_plot(config_id, results)
        logger.info("Comprehensive aggregate cross-entropy plot created using all seeds")
    except Exception as e:
        logger.error(f"Error creating comprehensive aggregate plot: {e}")
    
    logger.info(f"Aggregate analysis complete for config {config_id}!")
    logger.info(f"   Results saved in: {output_dir}")
    logger.info(f"   Processed {len(results)} seeds")
    logger.info(f"   Created 10 example plots + convergence plots")
    logger.info(f"   Created comprehensive aggregate plot using all {len(results)} seeds")
    logger.info(f"   Safe mode: No parallel processing used")

def main():
    """Main entry point - SAFE VERSION."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate aggregate analysis from existing HDF5 files (SAFE MODE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config-id", type=int, choices=[1, 2, 3, 4], default=3,
                       help="Configuration ID to process (default: 3)")
    parser.add_argument("--output-dir", type=str, default="results/aggregate_analysis",
                       help="Base directory containing results")
    
    args = parser.parse_args()
    
    logger.info("SAFE MODE: Single-threaded processing only")
    logger.info("   No parallel jobs will be started")
    logger.info("   Safe to run alongside other terminals")
    
    # Process single configuration
    generate_aggregate_analysis_safe(args.config_id, args.output_dir)

if __name__ == "__main__":
    main() 