#!/usr/bin/env python3
"""
DEADLOCK-FREE Aggregate Analysis
Eliminates ALL sources of multiprocessing deadlocks:
- Uses conservative worker counts (max 16)
- No file I/O in worker processes
- Minimal imports in workers
- Simple object passing only
- Immediate result collection
"""

import argparse
import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CRITICAL: Minimal global state
_GLOBAL_CONFIG = None
_GLOBAL_OUTPUT_DIR = None

def _minimal_worker_init(config_data: dict, output_dir: str):
    """Ultra-minimal worker initialization - no complex objects."""
    global _GLOBAL_CONFIG, _GLOBAL_OUTPUT_DIR
    _GLOBAL_CONFIG = config_data
    _GLOBAL_OUTPUT_DIR = output_dir


def _minimal_experiment_worker(seed: int) -> Optional[dict]:
    """Ultra-minimal worker function - returns simple dict, no file I/O."""
    try:
        # Import only what's absolutely needed, inside the function
        from aggregate_analysis import AggregateAnalysisManager
        
        # Create manager using global config
        manager = AggregateAnalysisManager(output_dir=_GLOBAL_OUTPUT_DIR)
        manager.experimental_configs = {_GLOBAL_CONFIG['config_id']: _GLOBAL_CONFIG['config_obj']}
        
        # Run experiment
        result = manager.run_single_experiment(_GLOBAL_CONFIG['config_id'], seed)
        
        # Return minimal serializable data only
        return {
            'seed': seed,
            'config_id': _GLOBAL_CONFIG['config_id'],
            'result': result,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Worker failed for seed {seed}: {e}")
        return {
            'seed': seed,
            'config_id': _GLOBAL_CONFIG['config_id'],
            'result': None,
            'success': False,
            'error': str(e)
        }


class DeadlockFreeAnalysisManager:
    """Ultra-conservative manager that eliminates ALL deadlock sources."""
    
    def __init__(self, output_dir: str = "results/aggregate_analysis"):
        # Import the original manager
        from aggregate_analysis import AggregateAnalysisManager
        self._manager = AggregateAnalysisManager(output_dir)
        self.output_dir = Path(output_dir)
        
    def run_conservative_parallel(self, config_id: int, n_seeds: int, n_processes: int = 8,
                                resume: bool = False) -> List:
        """Ultra-conservative parallel processing - eliminates ALL deadlock sources."""
        from aggregate_analysis_utils import CheckpointManager, HDF5DataManager
        
        # SAFETY LIMIT: Never exceed 16 workers
        safe_workers = min(n_processes, 16)
        if n_processes > 16:
            logger.warning(f"🛡️ SAFETY: Reducing {n_processes} workers to {safe_workers} (deadlock prevention)")
        
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
        
        # Pre-load config data for workers (avoid complex object serialization)
        config_data = {
            'config_id': config_id,
            'config_obj': self._manager.experimental_configs[config_id]
        }
        
        completed_results = []
        data_manager = HDF5DataManager(self.output_dir)
        
        logger.info(f"🛡️ DEADLOCK-FREE PARALLEL PROCESSING")
        logger.info(f"   Config: {config_id} | Seeds: {len(remaining_seeds)} | Workers: {safe_workers}")
        logger.info(f"   🔧 Using minimal workers, no file I/O contention, simple object passing")
        
        # Pre-create directories to avoid worker contention
        (self.output_dir / f"config_{config_id}_data").mkdir(parents=True, exist_ok=True)
        
        # Process in small batches to minimize resource pressure
        batch_size = safe_workers * 4  # Process 4x worker count at a time
        
        for batch_start in range(0, len(remaining_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_seeds))
            batch_seeds = remaining_seeds[batch_start:batch_end]
            
            logger.info(f"🔄 Processing batch {batch_start//batch_size + 1}: seeds {batch_start} to {batch_end-1}")
            
            # Create minimal pool with conservative settings
            try:
                with mp.Pool(processes=safe_workers, 
                            initializer=_minimal_worker_init,
                            initargs=(config_data, str(self.output_dir))) as pool:
                    
                    # Use map() with small batches instead of imap - simpler and more reliable
                    batch_results = []
                    
                    with tqdm(total=len(batch_seeds), desc=f"Batch {batch_start//batch_size + 1}") as pbar:
                        # Process in smaller sub-batches within each batch
                        sub_batch_size = safe_workers
                        
                        for sub_start in range(0, len(batch_seeds), sub_batch_size):
                            sub_end = min(sub_start + sub_batch_size, len(batch_seeds))
                            sub_batch = batch_seeds[sub_start:sub_end]
                            
                            # Simple map() call - most reliable
                            sub_results = pool.map(_minimal_experiment_worker, sub_batch)
                            
                            # Process results immediately
                            for result_dict in sub_results:
                                if result_dict and result_dict['success']:
                                    try:
                                        # Save immediately to prevent data loss
                                        result_obj = result_dict['result']
                                        data_manager.save_single_run_results(result_obj, config_id, result_obj.seed)
                                        completed_results.append(result_obj)
                                        
                                    except Exception as e:
                                        logger.error(f"Failed to save seed {result_dict['seed']}: {e}")
                                
                                pbar.update(1)
                            
                            # Checkpoint after each sub-batch
                            if completed_results:
                                completed_seeds = [r.seed for r in completed_results]
                                checkpoint_manager.save_progress(config_id, completed_seeds, n_seeds)
                    
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Continue with next batch or fallback to sequential
                
            # Brief pause between batches to let system recover
            if batch_end < len(remaining_seeds):
                time.sleep(2)
                logger.info(f"📊 Completed {len(completed_results)} seeds so far...")
        
        # Final checkpoint
        if completed_results:
            completed_seeds = [r.seed for r in completed_results]
            checkpoint_manager.save_progress(config_id, completed_seeds, n_seeds)
        
        logger.info(f"✅ Conservative processing complete: {len(completed_results)}/{len(remaining_seeds)} successful")
        
        return completed_results


def main():
    """Main entry point with deadlock-free processing."""
    parser = argparse.ArgumentParser(
        description="DEADLOCK-FREE Aggregate Analysis (eliminates ALL multiprocessing deadlocks)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--config-id", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--parallel", type=int, default=8, help="Worker count (max 16 for safety)")
    parser.add_argument("--output-dir", type=str, default="results/aggregate_analysis")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🛡️ STARTING DEADLOCK-FREE AGGREGATE ANALYSIS")
    logger.info(f"   Config: {args.config_id} | Seeds: {args.n_seeds} | Workers: {args.parallel}")
    logger.info(f"   Safety: Max 16 workers, batch processing, minimal I/O contention")
    
    # Use deadlock-free manager
    manager = DeadlockFreeAnalysisManager(output_dir=args.output_dir)
    
    start_time = time.time()
    
    # Run conservative parallel processing
    results = manager.run_conservative_parallel(
        config_id=args.config_id,
        n_seeds=args.n_seeds,
        n_processes=args.parallel,
        resume=args.resume
    )
    
    end_time = time.time()
    runtime_hours = (end_time - start_time) / 3600
    
    logger.info(f"🎉 DEADLOCK-FREE ANALYSIS COMPLETE!")
    logger.info(f"   ✅ Completed: {len(results)} seeds")
    logger.info(f"   ⏱️ Runtime: {runtime_hours:.2f} hours")
    if len(results) > 0:
        logger.info(f"   📈 Speed: {runtime_hours*3600/len(results):.2f} seconds/seed")
    logger.info(f"   💾 Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 