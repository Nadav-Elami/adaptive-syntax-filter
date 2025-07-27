"""Utility functions for aggregate analysis of Adaptive Syntax Filter.

Provides data storage, statistical analysis, and visualization utilities
for the comprehensive aggregate analysis system.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class HDF5DataManager:
    """Manager for efficient HDF5 data storage and retrieval."""
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize HDF5 data manager.
        
        Parameters
        ----------
        base_path : Union[str, Path]
            Base directory for HDF5 files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_single_run_results(self, results, config_id: int, seed: int):
        """Save results from a single run to HDF5 format.
        
        Parameters
        ----------
        results : SingleRunResults
            Results object from a single experiment run
        config_id : int
            Configuration ID
        seed : int
            Random seed used
        """
        config_dir = self.base_path / f"config_{config_id}_data"
        config_dir.mkdir(exist_ok=True)
        
        # Save to HDF5 with hierarchical structure
        hdf5_path = config_dir / f"seed_{seed:06d}.h5"
        
        with h5py.File(hdf5_path, 'w') as f:
            # Metadata
            f.attrs['seed'] = seed
            f.attrs['config_id'] = config_id
            f.attrs['runtime_seconds'] = results.runtime_seconds
            f.attrs['convergence_iteration'] = results.convergence_iteration
            f.attrs['final_log_likelihood'] = results.final_log_likelihood
            f.attrs['best_log_likelihood'] = results.best_log_likelihood
            
            # True probabilities and logits
            f.create_dataset('true_probabilities', data=results.true_probabilities, compression='gzip')
            f.create_dataset('true_logits', data=results.true_logits, compression='gzip')
            
            # Probability estimates
            f.create_dataset('em_probabilities', data=results.em_probabilities, compression='gzip')
            f.create_dataset('raw_probabilities', data=results.raw_probabilities, compression='gzip')
            f.create_dataset('smoothed_raw_probabilities', data=results.smoothed_raw_probabilities, compression='gzip')
            
            # Cross-entropy measures
            f.create_dataset('em_cross_entropy', data=results.em_cross_entropy, compression='gzip')
            f.create_dataset('raw_cross_entropy', data=results.raw_cross_entropy, compression='gzip')
            f.create_dataset('smoothed_raw_cross_entropy', data=results.smoothed_raw_cross_entropy, compression='gzip')
            
            # EM statistics history
            if results.em_results.statistics_history:
                em_stats = results.em_results.statistics_history
                iterations = [stat.iteration for stat in em_stats]
                log_likelihoods = [stat.log_likelihood for stat in em_stats]
                log_likelihood_changes = [stat.log_likelihood_change for stat in em_stats]
                parameter_changes = [stat.parameter_change for stat in em_stats]
                converged_flags = [stat.converged for stat in em_stats]
                f.create_dataset('em_statistics_iterations', data=iterations)
                f.create_dataset('em_statistics_log_likelihoods', data=log_likelihoods)
                f.create_dataset('em_statistics_log_likelihood_changes', data=log_likelihood_changes)
                f.create_dataset('em_statistics_parameter_changes', data=parameter_changes)
                f.create_dataset('em_statistics_converged_flags', data=converged_flags)
            
            # Save sequences as string data
            sequence_group = f.create_group('sequences')
            for i, sequence in enumerate(results.sequences):
                sequence_group.create_dataset(f'sequence_{i:04d}', 
                                            data=[s.encode('utf-8') for s in sequence])
        
        logger.debug(f"Saved results for config {config_id}, seed {seed} to {hdf5_path}")
    
    def load_single_run_results(self, config_id: int, seed: int) -> Dict[str, Any]:
        """Load results from a single run from HDF5 format.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        seed : int
            Random seed
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the loaded results
        """
        config_dir = self.base_path / f"config_{config_id}_data"
        hdf5_path = config_dir / f"seed_{seed:06d}.h5"
        
        if not hdf5_path.exists():
            raise FileNotFoundError(f"Results file not found: {hdf5_path}")
        
        results = {}
        
        with h5py.File(hdf5_path, 'r') as f:
            # Metadata
            results['seed'] = f.attrs['seed']
            results['config_id'] = f.attrs['config_id']
            results['runtime_seconds'] = f.attrs['runtime_seconds']
            results['convergence_iteration'] = f.attrs['convergence_iteration']
            results['final_log_likelihood'] = f.attrs['final_log_likelihood']
            results['best_log_likelihood'] = f.attrs['best_log_likelihood']
            
            # Arrays
            for key in ['true_probabilities', 'true_logits', 'em_probabilities', 
                       'raw_probabilities', 'smoothed_raw_probabilities',
                       'em_cross_entropy', 'raw_cross_entropy', 'smoothed_raw_cross_entropy']:
                if key in f:
                    results[key] = f[key][...]
            
            # EM statistics (legacy, for backward compatibility)
            for key in ['em_iterations', 'em_log_likelihoods', 'em_log_likelihood_changes',
                       'em_parameter_changes', 'em_converged_flags']:
                if key in f:
                    results[key] = f[key][...]
            # Full EM statistics history for convergence plots
            em_stats_keys = [
                'em_statistics_iterations',
                'em_statistics_log_likelihoods',
                'em_statistics_log_likelihood_changes',
                'em_statistics_parameter_changes',
                'em_statistics_converged_flags',
            ]
            if all(key in f for key in em_stats_keys):
                iterations = f['em_statistics_iterations'][...]
                log_likelihoods = f['em_statistics_log_likelihoods'][...]
                log_likelihood_changes = f['em_statistics_log_likelihood_changes'][...]
                parameter_changes = f['em_statistics_parameter_changes'][...]
                converged_flags = f['em_statistics_converged_flags'][...]
                # Reconstruct EMStatistics objects
                from src.adaptive_syntax_filter.core.em_algorithm import EMStatistics
                statistics_history = []
                for i in range(len(iterations)):
                    statistics_history.append(EMStatistics(
                        iteration=int(iterations[i]),
                        log_likelihood=float(log_likelihoods[i]),
                        log_likelihood_change=float(log_likelihood_changes[i]),
                        parameter_change=float(parameter_changes[i]),
                        converged=bool(converged_flags[i])
                    ))
                results['em_statistics_history'] = statistics_history
            
            # Sequences
            if 'sequences' in f:
                sequences = []
                sequence_group = f['sequences']
                for seq_name in sorted(sequence_group.keys()):
                    seq_data = sequence_group[seq_name][...]
                    sequence = [s.decode('utf-8') for s in seq_data]
                    sequences.append(sequence)
                results['sequences'] = sequences
        
        return results
    
    def aggregate_cross_entropy_data(self, config_id: int, seeds: List[int]) -> Dict[str, np.ndarray]:
        """Aggregate cross-entropy data across multiple seeds.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        seeds : List[int]
            List of seeds to aggregate
            
        Returns
        -------
        Dict[str, np.ndarray]
            Aggregated cross-entropy data with shape (n_seeds, n_timesteps)
        """
        em_cross_entropies = []
        raw_cross_entropies = []
        smoothed_raw_cross_entropies = []
        
        for seed in seeds:
            try:
                results = self.load_single_run_results(config_id, seed)
                em_cross_entropies.append(results['em_cross_entropy'])
                raw_cross_entropies.append(results['raw_cross_entropy'])
                smoothed_raw_cross_entropies.append(results['smoothed_raw_cross_entropy'])
            except FileNotFoundError:
                logger.warning(f"Missing results for config {config_id}, seed {seed}")
                continue
        
        return {
            'em_cross_entropy': np.array(em_cross_entropies),
            'raw_cross_entropy': np.array(raw_cross_entropies),
            'smoothed_raw_cross_entropy': np.array(smoothed_raw_cross_entropies)
        }
    
    def save_aggregate_summary(self, config_id: int, summary_data: Dict[str, Any]):
        """Save aggregate summary data to HDF5.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        summary_data : Dict[str, Any]
            Summary statistics and aggregated data
        """
        config_dir = self.base_path / f"config_{config_id}_data"
        summary_path = config_dir / "aggregate_summary.h5"
        
        with h5py.File(summary_path, 'w') as f:
            f.attrs['config_id'] = config_id
            f.attrs['n_runs'] = summary_data.get('n_runs', 0)
            f.attrs['creation_timestamp'] = summary_data.get('timestamp', '')
            
            # Save all array data
            for key, value in summary_data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    if isinstance(value[0], (int, float)):
                        f.create_dataset(key, data=np.array(value), compression='gzip')


class StatisticalAnalyzer:
    """Statistical analysis utilities for aggregate results."""
    
    @staticmethod
    def compute_cross_entropy_statistics(cross_entropy_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute statistics for cross-entropy data.
        
        Parameters
        ----------
        cross_entropy_data : Dict[str, np.ndarray]
            Cross-entropy data with shape (n_seeds, n_timesteps) for each method
            
        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Statistics including mean, std, median, quantiles for each method
        """
        statistics = {}
        
        for method, data in cross_entropy_data.items():
            if len(data) == 0:
                continue
                
            method_stats = {
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0),
                'median': np.median(data, axis=0),
                'q25': np.percentile(data, 25, axis=0),
                'q75': np.percentile(data, 75, axis=0),
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0),
                'n_runs': data.shape[0]
            }
            
            statistics[method] = method_stats
        
        return statistics
    
    @staticmethod
    def perform_significance_tests(cross_entropy_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Perform statistical significance tests between methods.
        
        Parameters
        ----------
        cross_entropy_data : Dict[str, np.ndarray]
            Cross-entropy data for each method
            
        Returns
        -------
        pd.DataFrame
            Test results with p-values and effect sizes
        """
        methods = list(cross_entropy_data.keys())
        n_methods = len(methods)
        n_timesteps = next(iter(cross_entropy_data.values())).shape[1] if cross_entropy_data else 0
        
        results = []
        
        # Pairwise comparisons
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method1, method2 = methods[i], methods[j]
                data1 = cross_entropy_data[method1]
                data2 = cross_entropy_data[method2]
                
                # Average cross-entropy across timesteps for each run
                avg_ce1 = np.mean(data1, axis=1)
                avg_ce2 = np.mean(data2, axis=1)
                
                # Paired t-test (since same seeds used)
                t_stat, p_value = stats.ttest_rel(avg_ce1, avg_ce2)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(avg_ce1) + np.var(avg_ce2)) / 2)
                cohens_d = (np.mean(avg_ce1) - np.mean(avg_ce2)) / pooled_std if pooled_std > 0 else 0
                
                # Wilcoxon signed-rank test (non-parametric)
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(avg_ce1, avg_ce2)
                
                # Determine significance levels
                def get_significance_marker(p_val):
                    if p_val < 0.001:
                        return "***"
                    elif p_val < 0.01:
                        return "**"
                    elif p_val < 0.05:
                        return "*"
                    else:
                        return "ns"
                
                t_test_significance = get_significance_marker(p_value)
                wilcoxon_significance = get_significance_marker(wilcoxon_p)
                
                results.append({
                    'method1': method1,
                    'method2': method2,
                    'mean_diff': np.mean(avg_ce1) - np.mean(avg_ce2),
                    't_statistic': t_stat,
                    't_test_p_value': p_value,
                    't_test_significance': t_test_significance,
                    'cohens_d': cohens_d,
                    'wilcoxon_statistic': wilcoxon_stat,
                    'wilcoxon_p_value': wilcoxon_p,
                    'wilcoxon_significance': wilcoxon_significance,
                    'n_pairs': len(avg_ce1)
                })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compute_convergence_statistics(em_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute convergence statistics across runs.
        
        Parameters
        ----------
        em_data : List[Dict[str, Any]]
            EM algorithm data from multiple runs
            
        Returns
        -------
        Dict[str, Any]
            Convergence statistics
        """
        convergence_iterations = []
        final_log_likelihoods = []
        best_log_likelihoods = []
        runtimes = []
        
        for run_data in em_data:
            convergence_iterations.append(run_data.get('convergence_iteration', -1))
            final_log_likelihoods.append(run_data.get('final_log_likelihood', -np.inf))
            best_log_likelihoods.append(run_data.get('best_log_likelihood', -np.inf))
            runtimes.append(run_data.get('runtime_seconds', 0))
        
        convergence_iterations = np.array(convergence_iterations)
        final_log_likelihoods = np.array(final_log_likelihoods)
        best_log_likelihoods = np.array(best_log_likelihoods)
        runtimes = np.array(runtimes)
        
        # Remove invalid values
        valid_mask = (convergence_iterations >= 0) & np.isfinite(final_log_likelihoods)
        
        if np.sum(valid_mask) == 0:
            return {}
        
        return {
            'convergence_rate': np.sum(valid_mask) / len(convergence_iterations),
            'mean_convergence_iteration': np.mean(convergence_iterations[valid_mask]),
            'std_convergence_iteration': np.std(convergence_iterations[valid_mask]),
            'mean_final_log_likelihood': np.mean(final_log_likelihoods[valid_mask]),
            'std_final_log_likelihood': np.std(final_log_likelihoods[valid_mask]),
            'mean_best_log_likelihood': np.mean(best_log_likelihoods[valid_mask]),
            'std_best_log_likelihood': np.std(best_log_likelihoods[valid_mask]),
            'mean_runtime': np.mean(runtimes[valid_mask]),
            'std_runtime': np.std(runtimes[valid_mask]),
            'n_valid_runs': np.sum(valid_mask),
            'n_total_runs': len(convergence_iterations)
        }


class CrossEntropyCalculator:
    """Utilities for cross-entropy calculations."""
    
    @staticmethod
    def calculate_cross_entropy(true_probs: np.ndarray, 
                              estimated_probs: np.ndarray,
                              epsilon: float = 1e-12) -> np.ndarray:
        """Calculate cross-entropy between true and estimated probabilities.
        
        Parameters
        ----------
        true_probs : np.ndarray
            True probability distributions
        estimated_probs : np.ndarray  
            Estimated probability distributions
        epsilon : float
            Small value to avoid log(0)
            
        Returns
        -------
        np.ndarray
            Cross-entropy values
        """
        # Ensure probabilities are valid
        estimated_probs_safe = np.clip(estimated_probs, epsilon, 1.0)
        
        # Normalize to ensure they sum to 1 (in case of numerical errors)
        if estimated_probs_safe.ndim > 1:
            estimated_probs_safe = estimated_probs_safe / np.sum(estimated_probs_safe, axis=-1, keepdims=True)
        
        # Calculate cross-entropy
        cross_entropy = -np.sum(true_probs * np.log(estimated_probs_safe), axis=-1)
        
        return cross_entropy
    
    @staticmethod
    def calculate_kl_divergence(true_probs: np.ndarray,
                               estimated_probs: np.ndarray,
                               epsilon: float = 1e-12) -> np.ndarray:
        """Calculate KL divergence between true and estimated probabilities.
        
        Parameters
        ----------
        true_probs : np.ndarray
            True probability distributions
        estimated_probs : np.ndarray
            Estimated probability distributions
        epsilon : float
            Small value to avoid log(0)
            
        Returns
        -------
        np.ndarray
            KL divergence values
        """
        # Ensure probabilities are valid
        true_probs_safe = np.clip(true_probs, epsilon, 1.0)
        estimated_probs_safe = np.clip(estimated_probs, epsilon, 1.0)
        
        # Calculate KL divergence
        kl_div = np.sum(true_probs_safe * np.log(true_probs_safe / estimated_probs_safe), axis=-1)
        
        return kl_div


class CheckpointManager:
    """Manager for checkpointing and resuming large-scale experiments."""
    
    def __init__(self, checkpoint_dir: Union[str, Path], base_data_dir: Union[str, Path] = None):
        """Initialize checkpoint manager.
        
        Parameters
        ----------
        checkpoint_dir : Union[str, Path]
            Directory for storing checkpoint files
        base_data_dir : Union[str, Path], optional
            Base directory where HDF5 data files are stored (for verification)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.base_data_dir = Path(base_data_dir) if base_data_dir else None
    
    def save_progress(self, config_id: int, completed_seeds: List[int], 
                     total_seeds: int, metadata: Dict[str, Any] = None):
        """Save progress checkpoint.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        completed_seeds : List[int]
            List of completed seeds
        total_seeds : int
            Total number of seeds to process
        metadata : Dict[str, Any], optional
            Additional metadata to save
        """
        checkpoint_file = self.checkpoint_dir / f"config_{config_id}_checkpoint.npz"
        
        save_data = {
            'completed_seeds': np.array(completed_seeds),
            'total_seeds': total_seeds,
            'config_id': config_id
        }
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str)):
                    save_data[f"meta_{key}"] = value
        
        np.savez_compressed(checkpoint_file, **save_data)
        logger.info(f"Saved checkpoint for config {config_id}: {len(completed_seeds)}/{total_seeds} seeds completed")
    
    def load_progress(self, config_id: int) -> Tuple[List[int], int, Dict[str, Any]]:
        """Load progress checkpoint.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
            
        Returns
        -------
        Tuple[List[int], int, Dict[str, Any]]
            Completed seeds, total seeds, and metadata
        """
        checkpoint_file = self.checkpoint_dir / f"config_{config_id}_checkpoint.npz"
        
        if not checkpoint_file.exists():
            return [], 0, {}
        
        data = np.load(checkpoint_file)
        completed_seeds = data['completed_seeds'].tolist()
        total_seeds = int(data['total_seeds'])
        
        metadata = {}
        for key in data.keys():
            if key.startswith('meta_'):
                metadata[key[5:]] = data[key].item()
        
        logger.info(f"Loaded checkpoint for config {config_id}: {len(completed_seeds)}/{total_seeds} seeds completed")
        return completed_seeds, total_seeds, metadata
    
    def get_remaining_seeds(self, config_id: int, all_seeds: List[int]) -> List[int]:
        """Get list of remaining seeds to process.
        
        This method now properly verifies that HDF5 files actually exist,
        not just checking the checkpoint data.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
        all_seeds : List[int]
            Complete list of seeds to process
            
        Returns
        -------
        List[int]
            Seeds that still need to be processed
        """
        # Get seeds marked as completed in checkpoint
        checkpoint_completed_seeds, _, _ = self.load_progress(config_id)
        checkpoint_completed_set = set(checkpoint_completed_seeds)
        
        # If no base_data_dir provided, fall back to old behavior with warning
        if self.base_data_dir is None:
            logger.warning(f"No base_data_dir provided to CheckpointManager - "
                          f"cannot verify HDF5 files exist! Using checkpoint data only.")
            remaining = [seed for seed in all_seeds if seed not in checkpoint_completed_set]
            return remaining
        
        # Check which seeds actually have HDF5 files
        data_dir = self.base_data_dir / f"config_{config_id}_data"
        actually_completed = set()
        missing_files = []
        
        for seed in checkpoint_completed_seeds:
            hdf5_file = data_dir / f"seed_{seed:06d}.h5"
            if hdf5_file.exists():
                actually_completed.add(seed)
            else:
                missing_files.append(seed)
        
        # Log any discrepancies
        if missing_files:
            logger.warning(f"Config {config_id}: {len(missing_files)} seeds in checkpoint "
                          f"but missing HDF5 files: {missing_files[:10]}...")
            logger.info(f"These seeds will be re-processed to ensure data integrity")
        
        # Also check for HDF5 files not in checkpoint (shouldn't happen, but let's be safe)
        if data_dir.exists():
            existing_h5_files = list(data_dir.glob("seed_*.h5"))
            existing_seeds = set()
            for h5_file in existing_h5_files:
                # Extract seed number from filename: seed_000042.h5 -> 42
                try:
                    seed_str = h5_file.stem.split('_')[1]  # Remove 'seed_' prefix
                    seed = int(seed_str)
                    existing_seeds.add(seed)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse seed from filename: {h5_file}")
            
            # Check for files not in checkpoint
            orphaned_files = existing_seeds - checkpoint_completed_set
            if orphaned_files:
                logger.info(f"Config {config_id}: Found {len(orphaned_files)} HDF5 files "
                           f"not in checkpoint: {list(orphaned_files)[:10]}...")
                # Add these to actually_completed since they exist
                actually_completed.update(orphaned_files)
        
        # Return seeds that are not actually completed
        remaining = [seed for seed in all_seeds if seed not in actually_completed]
        
        logger.info(f"Config {config_id}: {len(actually_completed)} seeds verified complete, "
                   f"{len(remaining)} remaining to process")
        
        return remaining
    
    def verify_and_repair_checkpoint(self, config_id: int) -> Tuple[int, int, int]:
        """Verify checkpoint integrity and repair if needed.
        
        Parameters
        ----------
        config_id : int
            Configuration ID
            
        Returns
        -------
        Tuple[int, int, int]
            (seeds_in_checkpoint, seeds_with_files, seeds_repaired)
        """
        if self.base_data_dir is None:
            logger.error("Cannot verify checkpoint without base_data_dir")
            return 0, 0, 0
        
        # Load current checkpoint
        checkpoint_completed, total_seeds, metadata = self.load_progress(config_id)
        
        # Check which ones actually have files
        data_dir = self.base_data_dir / f"config_{config_id}_data"
        actually_completed = []
        
        if data_dir.exists():
            existing_h5_files = list(data_dir.glob("seed_*.h5"))
            for h5_file in existing_h5_files:
                try:
                    seed_str = h5_file.stem.split('_')[1]
                    seed = int(seed_str)
                    actually_completed.append(seed)
                except (ValueError, IndexError):
                    continue
        
        # Update checkpoint to match reality
        if set(actually_completed) != set(checkpoint_completed):
            logger.info(f"Config {config_id}: Repairing checkpoint - "
                       f"was {len(checkpoint_completed)}, now {len(actually_completed)} seeds")
            self.save_progress(config_id, actually_completed, total_seeds, metadata)
            return len(checkpoint_completed), len(actually_completed), len(actually_completed)
        else:
            logger.info(f"Config {config_id}: Checkpoint is consistent with HDF5 files")
            return len(checkpoint_completed), len(actually_completed), 0 