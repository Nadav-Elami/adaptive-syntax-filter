#!/usr/bin/env python3
"""Create comprehensive plots for config 3 with exactly 10 example plots.

This script creates 10 example evolution plots and 10 convergence plots
for config 3, following the original plotting logic from aggregate_analysis.py.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_comprehensive_plots.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def create_comprehensive_plots(config_id: int = 3, output_dir: str = "results/aggregate_analysis"):
    """Create comprehensive plots for config 3 with exactly 10 example plots."""
    from aggregate_analysis import AggregateAnalysisManager
    
    logger.info(f"Creating comprehensive plots for config {config_id}")
    
    # Initialize manager
    analysis_manager = AggregateAnalysisManager(output_dir)
    
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
    
    # Load first 10 results
    logger.info("Loading first 10 results for example plots...")
    results = analysis_manager._load_existing_results(config_id, seeds[:10])
    
    if not results:
        logger.error("No results could be loaded")
        return
    
    logger.info(f"Successfully loaded {len(results)} results for plotting")
    
    # Get config details
    agg_config = analysis_manager.experimental_configs[config_id]
    config_dir = analysis_manager.output_dir / f"config_{config_id}_{agg_config.config_name}"
    plots_dir = config_dir / "example_plots"
    plots_dir.mkdir(exist_ok=True)
    
    logger.info(f"Creating plots in: {plots_dir}")
    
    # Create exactly 10 example plots with convergence plots
    for i, result in enumerate(results):
        logger.info(f"Creating plots for seed {result.seed} ({i+1}/10)")
        
        try:
            # Create evolution plot
            analysis_manager._create_single_run_plot(result, agg_config, plots_dir)
            logger.info(f"  Created evolution plot for seed {result.seed}")
            
            # Create convergence plot if EM statistics are available
            if hasattr(result.em_results, 'statistics_history') and result.em_results.statistics_history:
                analysis_manager._create_convergence_plot(result, plots_dir)
                logger.info(f"  Created convergence plot for seed {result.seed}")
            else:
                logger.info(f"  No convergence plot for seed {result.seed} (no EM statistics)")
                
        except Exception as e:
            logger.error(f"Error creating plots for seed {result.seed}: {e}")
            continue
    
    logger.info(f"Comprehensive plots complete!")
    logger.info(f"  Created evolution plots for {len(results)} seeds")
    logger.info(f"  Created convergence plots where EM statistics were available")
    logger.info(f"  All plots saved in: {plots_dir}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create comprehensive plots for config 3 with exactly 10 example plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config-id", type=int, default=3,
                       help="Configuration ID to process (default: 3)")
    parser.add_argument("--output-dir", type=str, default="results/aggregate_analysis",
                       help="Base directory containing results")
    
    args = parser.parse_args()
    
    logger.info("Creating comprehensive plots...")
    logger.info(f"  Config: {args.config_id}")
    logger.info(f"  Output: {args.output_dir}")
    
    # Create comprehensive plots
    create_comprehensive_plots(args.config_id, args.output_dir)

if __name__ == "__main__":
    main() 