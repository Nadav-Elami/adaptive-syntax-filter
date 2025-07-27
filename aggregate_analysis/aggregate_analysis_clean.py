#!/usr/bin/env python3
"""Clean Optimized Aggregate Statistics Analysis for Adaptive Syntax Filter.

FIXES:
1. Removed emoji characters to fix Unicode encoding issues
2. Optimized worker initialization to reduce I/O contention  
3. Enhanced progress tracking and error handling
4. Support for high worker counts (32-128 workers)

Usage:
    python aggregate_analysis_clean.py --config-id 2 --n-seeds 50000 --parallel 64
"""

import argparse
import logging
import multiprocessing as mp
import pickle
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Local imports
from config_cli import _read_config as load_config
from research_pipeline import run_pipeline
from src.adaptive_syntax_filter.data.sequence_generator import (
    GenerationConfig, SequenceGenerator, sequences_to_observations, generate_dataset
)
from src.adaptive_syntax_filter.core.state_space import StateSpaceManager
from src.adaptive_syntax_filter.core.em_algorithm import EMAlgorithm, EMParameters, EMResults
from src.adaptive_syntax_filter.data.temporal_evolution import compute_evolution_trajectory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aggregate_analysis_clean.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

# Global config cache to avoid repeated loading in workers
_CONFIG_CACHE = {}
_DIRECTORIES_CREATED = False

def _initialize_worker_process(config_id: int, output_dir: str):
    """Initialize worker process with pre-loaded config."""
    global _CONFIG_CACHE, _DIRECTORIES_CREATED
    
    try:
        # Load and cache config once per worker
        if config_id not in _CONFIG_CACHE:
            from aggregate_analysis import AggregateAnalysisManager
            manager = AggregateAnalysisManager(output_dir=output_dir)
            manager._load_single_config(config_id)
            _CONFIG_CACHE[config_id] = {
                'config': manager.experimental_configs[config_id],
                'output_dir': manager.output_dir
            }
        
        # Ensure directories exist (thread-safe)
        if not _DIRECTORIES_CREATED:
            output_path = Path(output_dir)
            (output_path / f"config_{config_id}_data").mkdir(parents=True, exist_ok=True)
            _DIRECTORIES_CREATED = True
            
    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")


def _optimized_single_experiment_wrapper(args: Tuple[int, int, str]) -> Optional:
    """Optimized wrapper that uses cached config."""
    config_id, seed, output_dir = args
    
    try:
        # Use cached config instead of creating new manager
        if config_id not in _CONFIG_CACHE:
            _initialize_worker_process(config_id, output_dir)
        
        # Import here to avoid pickle issues
        from aggregate_analysis import AggregateAnalysisManager, SingleRunResults
        
        # Create minimal manager instance
        manager = AggregateAnalysisManager(output_dir=output_dir)
        manager.experimental_configs[config_id] = _CONFIG_CACHE[config_id]['config']
        
        return manager.run_single_experiment(config_id, seed)
        
    except Exception as e:
        logger.error(f"Failed experiment config_id={config_id}, seed={seed}: {e}")
        return None


class CleanOptimizedAggregateAnalysisManager:
    """Clean optimized version that fixes multiprocessing deadlocks and encoding issues."""
    
    def __init__(self, output_dir: str = "results/aggregate_analysis"):
        # Import the original manager
        from aggregate_analysis import AggregateAnalysisManager
        self._manager = AggregateAnalysisManager(output_dir)
        self.output_dir = Path(output_dir)
        
    def run_parallel_batch_clean(self, config_id: int, n_seeds: int, n_processes: int = 4,
                               resume: bool = False) -> List:
        """Clean optimized parallel processing that prevents deadlocks."""
        from aggregate_analysis_utils import CheckpointManager, HDF5DataManager
        
        # Set up checkpointing with HDF5 verification
        checkpoint_manager = CheckpointManager(
            self.output_dir / "checkpoints", 
            base_data_dir=self.output_dir
        )
        
        seeds = list(range(n_seeds))
        
        if resume:
            remaining_seeds = checkpoint_manager.get_remaining_seeds(config_id, seeds)
            logger.info(f"Resuming: {len(remaining_seeds)} seeds remaining out of {n_seeds}")
        else:
            remaining_seeds = seeds
        
        if not remaining_seeds:
            logger.info(f"All seeds already completed for config {config_id}")
            return self._manager._load_existing_results(config_id, seeds)
        
        completed_seeds = []
        data_manager = HDF5DataManager(self.output_dir)
        
        logger.info(f"OPTIMIZED PARALLEL PROCESSING")
        logger.info(f"   Config: {config_id} | Seeds: {len(remaining_seeds)} | Workers: {n_processes}")
        logger.info(f"   Using unordered processing to prevent deadlocks")
        
        # Pre-create directories to avoid worker contention
        (self.output_dir / f"config_{config_id}_data").mkdir(parents=True, exist_ok=True)
        
        # Use optimized multiprocessing with worker initialization
        with mp.Pool(processes=n_processes, 
                     initializer=_initialize_worker_process,
                     initargs=(config_id, str(self.output_dir))) as pool:
            try:
                # Create argument tuples - include output_dir for workers
                args_list = [(config_id, seed, str(self.output_dir)) for seed in remaining_seeds]
                
                # CRITICAL FIX: Use imap_unordered instead of imap
                # This prevents ordering deadlocks!
                logger.info("Starting unordered parallel processing...")
                
                with tqdm(total=len(remaining_seeds), desc=f"Config {config_id}") as pbar:
                    for result in pool.imap_unordered(_optimized_single_experiment_wrapper, args_list):
                        if result is not None:
                            try:
                                # Save immediately
                                data_manager.save_single_run_results(result, config_id, result.seed)
                                completed_seeds.append(result.seed)
                                
                                # Update progress
                                pbar.set_postfix({
                                    'completed': len(completed_seeds),
                                    'success_rate': f"{len(completed_seeds)/len([r for r in [result] if r is not None])*100:.1f}%"
                                })
                                
                            except Exception as e:
                                logger.error(f"Failed to save seed {result.seed}: {e}")
                                continue
                        
                        pbar.update(1)
                        
                        # Checkpoint every 50 completed
                        if len(completed_seeds) % 50 == 0 and len(completed_seeds) > 0:
                            checkpoint_manager.save_progress(config_id, completed_seeds, n_seeds)
                            logger.info(f"Checkpoint: {len(completed_seeds)} completed")
                            
            except KeyboardInterrupt:
                logger.warning("Interrupted by user. Saving progress...")
                pool.terminate()
                pool.join()
                
                if completed_seeds:
                    checkpoint_manager.save_progress(config_id, completed_seeds, n_seeds)
                    logger.info(f"Final checkpoint: {len(completed_seeds)} seeds saved")
        
        # Final checkpoint
        if completed_seeds:
            checkpoint_manager.save_progress(config_id, completed_seeds, n_seeds)
        
        logger.info(f"Optimized processing complete: {len(completed_seeds)}/{len(remaining_seeds)} successful")
        
        # Load completed results
        if completed_seeds:
            logger.info(f"Loading {len(completed_seeds)} results from HDF5 files...")
            return self._manager._load_existing_results(config_id, completed_seeds)
        else:
            return []


def main():
    """Main entry point with clean optimized processing."""
    parser = argparse.ArgumentParser(
        description="Clean Optimized Aggregate Analysis (fixes multiprocessing deadlocks and encoding issues)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config-id", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results/aggregate_analysis")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("STARTING CLEAN OPTIMIZED AGGREGATE ANALYSIS")
    logger.info(f"   Config: {args.config_id} | Seeds: {args.n_seeds} | Workers: {args.parallel}")
    logger.info(f"   Optimizations: unordered processing, worker caching, reduced I/O contention")
    
    # Use clean optimized manager
    manager = CleanOptimizedAggregateAnalysisManager(output_dir=args.output_dir)
    
    start_time = time.time()
    
    # Run clean optimized parallel processing
    results = manager.run_parallel_batch_clean(
        config_id=args.config_id,
        n_seeds=args.n_seeds,
        n_processes=args.parallel,
        resume=args.resume
    )
    
    end_time = time.time()
    runtime_hours = (end_time - start_time) / 3600
    
    logger.info(f"ANALYSIS COMPLETE!")
    logger.info(f"   Completed: {len(results)} seeds")
    logger.info(f"   Runtime: {runtime_hours:.2f} hours")
    logger.info(f"   Speed: {runtime_hours*3600/len(results):.2f} seconds/seed")
    logger.info(f"   Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 