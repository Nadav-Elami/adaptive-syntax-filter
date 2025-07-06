"""Dataset builder for batch creation and management of synthetic canary song data.

Provides high-level interfaces for creating research-scale datasets with
validation, export capabilities, and memory-efficient batch processing
for both bengalese finch and canary scale analyses.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import warnings
from datetime import datetime

from .sequence_generator import (
    SequenceGenerator, GenerationConfig, generate_dataset,
    validate_generated_sequences, create_standard_configs
)
from .alphabet_manager import AlphabetManager, create_preset_alphabets
from .temporal_evolution import EvolutionManager, create_evolution_examples
from .constraint_system import ConstraintManager


@dataclass
class DatasetMetadata:
    """Metadata for generated datasets.
    
    Attributes
    ----------
    name : str
        Dataset name/identifier
    creation_time : str
        ISO timestamp of creation
    config : Dict
        Generation configuration used
    alphabet_info : Dict
        Alphabet properties and statistics
    sequence_stats : Dict
        Statistics about generated sequences
    validation_results : Dict
        Validation results
    """
    name: str
    creation_time: str
    config: Dict
    alphabet_info: Dict
    sequence_stats: Dict
    validation_results: Dict


@dataclass
class Dataset:
    """Complete dataset with sequences, parameters, and metadata.
    
    Attributes
    ----------
    sequences : List[List[str]]
        Generated sequences
    parameter_trajectory : np.ndarray
        Evolution trajectory of parameters
    metadata : DatasetMetadata
        Dataset metadata and statistics
    """
    sequences: List[List[str]]
    parameter_trajectory: np.ndarray
    metadata: DatasetMetadata


def analyze_sequence_statistics(sequences: List[List[str]], 
                              alphabet: List[str]) -> Dict[str, Any]:
    """
    Analyze statistical properties of generated sequences.
    
    Parameters
    ----------
    sequences : List[List[str]]
        Generated sequences to analyze
    alphabet : List[str]
        Symbol alphabet
        
    Returns
    -------
    dict
        Comprehensive sequence statistics
    """
    if not sequences:
        return {'error': 'No sequences to analyze'}
    
    # Basic statistics
    lengths = [len(seq) for seq in sequences]
    
    # Symbol usage counts
    symbol_counts = {symbol: 0 for symbol in alphabet}
    total_symbols = 0
    
    for seq in sequences:
        for symbol in seq:
            if symbol in symbol_counts:
                symbol_counts[symbol] += 1
                total_symbols += 1
    
    # Symbol frequencies
    symbol_frequencies = {symbol: count / total_symbols if total_symbols > 0 else 0 
                         for symbol, count in symbol_counts.items()}
    
    # Phrase symbols only (excluding start/end)
    phrase_symbols = alphabet[1:-1]
    phrase_counts = {symbol: symbol_counts[symbol] for symbol in phrase_symbols}
    phrase_total = sum(phrase_counts.values())
    phrase_frequencies = {symbol: count / phrase_total if phrase_total > 0 else 0 
                         for symbol, count in phrase_counts.items()}
    
    # Transition analysis (first-order)
    transitions = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            transition = (seq[i], seq[i + 1])
            transitions[transition] = transitions.get(transition, 0) + 1
    
    # Most common patterns
    common_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'n_sequences': len(sequences),
        'length_stats': {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'median': np.median(lengths)
        },
        'symbol_usage': {
            'counts': symbol_counts,
            'frequencies': symbol_frequencies,
            'total_symbols': total_symbols
        },
        'phrase_usage': {
            'counts': phrase_counts,
            'frequencies': phrase_frequencies,
            'total_phrases': phrase_total
        },
        'transition_stats': {
            'total_transitions': len(transitions),
            'unique_transitions': len(set(transitions.keys())),
            'most_common': common_transitions[:5]
        },
        'diversity_metrics': {
            'unique_sequences': len(set(tuple(seq) for seq in sequences)),
            'repetition_rate': 1 - len(set(tuple(seq) for seq in sequences)) / len(sequences)
        }
    }


def validate_dataset_quality(dataset: Dataset, 
                           quality_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Validate dataset quality against research standards.
    
    Parameters
    ----------
    dataset : Dataset
        Dataset to validate
    quality_thresholds : Optional[Dict[str, float]]
        Custom quality thresholds, or None for defaults
        
    Returns
    -------
    dict
        Quality validation results
    """
    if quality_thresholds is None:
        quality_thresholds = {
            'min_validation_rate': 0.95,
            'min_unique_sequences': 0.8,
            'max_repetition_rate': 0.2,
            'min_symbol_diversity': 0.1,  # Each phrase symbol should appear at least 10% expected frequency
            'max_length_cv': 2.0  # Coefficient of variation for sequence lengths
        }
    
    results = {
        'overall_quality': 'unknown',
        'passed_checks': 0,
        'total_checks': 0,
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    try:
        # Basic validation
        validation = dataset.metadata.validation_results
        results['total_checks'] += 1
        if validation['validation_rate'] >= quality_thresholds['min_validation_rate']:
            results['passed_checks'] += 1
        else:
            results['issues'].append(f"Low validation rate: {validation['validation_rate']:.3f}")
        
        # Sequence diversity
        stats = dataset.metadata.sequence_stats
        diversity = stats['diversity_metrics']
        
        results['total_checks'] += 1
        unique_rate = diversity['unique_sequences'] / stats['n_sequences']
        if unique_rate >= quality_thresholds['min_unique_sequences']:
            results['passed_checks'] += 1
        else:
            results['issues'].append(f"Low sequence diversity: {unique_rate:.3f}")
        
        results['total_checks'] += 1
        if diversity['repetition_rate'] <= quality_thresholds['max_repetition_rate']:
            results['passed_checks'] += 1
        else:
            results['issues'].append(f"High repetition rate: {diversity['repetition_rate']:.3f}")
        
        # Symbol usage balance
        phrase_freqs = stats['phrase_usage']['frequencies']
        alphabet_size = len(dataset.metadata.alphabet_info['phrase_symbols'])
        expected_freq = 1.0 / alphabet_size
        min_acceptable_freq = expected_freq * quality_thresholds['min_symbol_diversity']
        
        results['total_checks'] += 1
        low_usage_symbols = [symbol for symbol, freq in phrase_freqs.items() 
                           if freq < min_acceptable_freq]
        if not low_usage_symbols:
            results['passed_checks'] += 1
        else:
            results['warnings'].append(f"Low usage symbols: {low_usage_symbols}")
        
        # Length variation
        length_stats = stats['length_stats']
        cv = length_stats['std'] / length_stats['mean'] if length_stats['mean'] > 0 else float('inf')
        
        results['total_checks'] += 1
        if cv <= quality_thresholds['max_length_cv']:
            results['passed_checks'] += 1
        else:
            results['warnings'].append(f"High length variation (CV={cv:.2f})")
        
        # Overall quality assessment
        pass_rate = results['passed_checks'] / results['total_checks']
        if pass_rate >= 0.8 and not results['issues']:
            results['overall_quality'] = 'excellent'
        elif pass_rate >= 0.6 and len(results['issues']) <= 1:
            results['overall_quality'] = 'good'
        elif pass_rate >= 0.4:
            results['overall_quality'] = 'acceptable'
        else:
            results['overall_quality'] = 'poor'
        
        results['metrics'] = {
            'validation_rate': validation['validation_rate'],
            'unique_rate': unique_rate,
            'repetition_rate': diversity['repetition_rate'],
            'length_cv': cv,
            'pass_rate': pass_rate
        }
        
    except Exception as e:
        results['issues'].append(f"Quality validation error: {e}")
        results['overall_quality'] = 'error'
    
    return results


class DatasetBuilder:
    """Builder class for creating and managing research datasets."""
    
    def __init__(self, workspace_dir: Optional[Union[str, Path]] = None):
        """
        Initialize dataset builder.
        
        Parameters
        ----------
        workspace_dir : Optional[Union[str, Path]]
            Directory for saving datasets, or None for current directory
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.workspace_dir.mkdir(exist_ok=True)
        
        # Initialize preset managers
        self.preset_alphabets = create_preset_alphabets()
        self.preset_configs = create_standard_configs()
        self.evolution_examples = create_evolution_examples()
        
        # Track created datasets
        self.created_datasets = []
    
    def create_dataset(self, 
                      config: GenerationConfig,
                      name: Optional[str] = None,
                      validate_quality: bool = True) -> Dataset:
        """
        Create complete dataset from configuration.
        
        Parameters
        ----------
        config : GenerationConfig
            Generation configuration
        name : Optional[str]
            Dataset name, or None for auto-generated
        validate_quality : bool
            Whether to perform quality validation
            
        Returns
        -------
        Dataset
            Complete dataset with metadata
        """
        if name is None:
            name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate sequences and parameters
        sequences, parameter_trajectory = generate_dataset(config)
        
        # Create alphabet manager for analysis
        alphabet_manager = AlphabetManager(config.alphabet)
        
        # Validate sequences
        validation_results = validate_generated_sequences(sequences, config.alphabet)
        
        # Analyze sequence statistics
        sequence_stats = analyze_sequence_statistics(sequences, config.alphabet)
        
        # Create metadata
        metadata = DatasetMetadata(
            name=name,
            creation_time=datetime.now().isoformat(),
            config=asdict(config),
            alphabet_info={
                'alphabet': config.alphabet,
                'size': len(config.alphabet),
                'n_phrases': len(config.alphabet) - 2,
                'phrase_symbols': config.alphabet[1:-1]
            },
            sequence_stats=sequence_stats,
            validation_results=validation_results
        )
        
        # Create dataset
        dataset = Dataset(
            sequences=sequences,
            parameter_trajectory=parameter_trajectory,
            metadata=metadata
        )
        
        # Quality validation
        if validate_quality:
            quality_results = validate_dataset_quality(dataset)
            dataset.metadata.validation_results['quality_assessment'] = quality_results
            
            # Issue warnings for quality problems
            if quality_results['overall_quality'] in ['poor', 'error']:
                warnings.warn(f"Dataset quality is {quality_results['overall_quality']}: "
                            f"{quality_results['issues']}")
        
        # Track dataset
        self.created_datasets.append(name)
        
        return dataset
    
    def create_from_preset(self, 
                          preset_name: str,
                          name: Optional[str] = None,
                          modifications: Optional[Dict[str, Any]] = None) -> Dataset:
        """
        Create dataset from preset configuration.
        
        Parameters
        ----------
        preset_name : str
            Name of preset configuration
        name : Optional[str]
            Dataset name
        modifications : Optional[Dict[str, Any]]
            Modifications to apply to preset config
            
        Returns
        -------
        Dataset
            Generated dataset
        """
        if preset_name not in self.preset_configs:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.preset_configs.keys())}")
        
        # Get base config
        config = self.preset_configs[preset_name]
        
        # Apply modifications
        if modifications:
            for key, value in modifications.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    warnings.warn(f"Unknown config parameter: {key}")
        
        # Use preset name if no custom name provided
        if name is None:
            name = f"{preset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self.create_dataset(config, name)
    
    def create_batch_datasets(self, 
                            base_config: GenerationConfig,
                            variations: List[Dict[str, Any]],
                            name_prefix: str = "batch") -> List[Dataset]:
        """
        Create multiple datasets with systematic variations.
        
        Parameters
        ----------
        base_config : GenerationConfig
            Base configuration
        variations : List[Dict[str, Any]]
            List of modifications to apply
        name_prefix : str
            Prefix for dataset names
            
        Returns
        -------
        List[Dataset]
            List of generated datasets
        """
        datasets = []
        
        for i, variation in enumerate(variations):
            # Copy base config
            config = GenerationConfig(**asdict(base_config))
            
            # Apply variation
            for key, value in variation.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    warnings.warn(f"Unknown config parameter: {key}")
            
            # Create dataset
            name = f"{name_prefix}_{i:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset = self.create_dataset(config, name, validate_quality=False)
            datasets.append(dataset)
        
        return datasets
    
    def save_dataset(self, 
                    dataset: Dataset,
                    format: str = 'pickle',
                    include_parameters: bool = True) -> Path:
        """
        Save dataset to file.
        
        Parameters
        ----------
        dataset : Dataset
            Dataset to save
        format : str
            'pickle', 'json', or 'txt'
        include_parameters : bool
            Whether to include parameter trajectory
            
        Returns
        -------
        Path
            Path to saved file
        """
        filename = f"{dataset.metadata.name}.{format}"
        filepath = self.workspace_dir / filename
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(dataset, f)
                
        elif format == 'json':
            # Convert to JSON-serializable format
            data = {
                'sequences': dataset.sequences,
                'metadata': asdict(dataset.metadata)
            }
            
            if include_parameters:
                data['parameter_trajectory'] = dataset.parameter_trajectory.tolist()
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format == 'txt':
            # Simple text format for sequences only
            with open(filepath, 'w') as f:
                f.write(f"# Dataset: {dataset.metadata.name}\n")
                f.write(f"# Created: {dataset.metadata.creation_time}\n")
                f.write(f"# Sequences: {len(dataset.sequences)}\n")
                f.write(f"# Alphabet: {dataset.metadata.alphabet_info['alphabet']}\n\n")
                
                for i, seq in enumerate(dataset.sequences):
                    f.write(f"{i:4d}: {' '.join(seq)}\n")
                    
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return filepath
    
    def load_dataset(self, filepath: Union[str, Path]) -> Dataset:
        """
        Load dataset from file.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Path to dataset file
            
        Returns
        -------
        Dataset
            Loaded dataset
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
                
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct dataset
            metadata = DatasetMetadata(**data['metadata'])
            parameter_trajectory = np.array(data.get('parameter_trajectory', []))
            
            return Dataset(
                sequences=data['sequences'],
                parameter_trajectory=parameter_trajectory,
                metadata=metadata
            )
            
        else:
            raise ValueError(f"Cannot load format: {filepath.suffix}")
    
    def list_presets(self) -> Dict[str, Dict[str, Any]]:
        """List available preset configurations."""
        return {
            'configs': {name: asdict(config) for name, config in self.preset_configs.items()},
            'alphabets': {name: mgr.to_dict() for name, mgr in self.preset_alphabets.items()},
            'evolution_models': {name: mgr.evolution_type for name, mgr in self.evolution_examples.items()}
        }
    
    def benchmark_performance(self, 
                            config: GenerationConfig,
                            n_trials: int = 3) -> Dict[str, Any]:
        """
        Benchmark generation performance for given configuration.
        
        Parameters
        ----------
        config : GenerationConfig
            Configuration to benchmark
        n_trials : int
            Number of trials for timing
            
        Returns
        -------
        dict
            Performance benchmark results
        """
        import time
        
        times = []
        memory_estimates = []
        
        for trial in range(n_trials):
            start_time = time.time()
            
            # Generate dataset
            sequences, trajectory = generate_dataset(config)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Estimate memory usage
            sequence_memory = sum(len(seq) for seq in sequences) * 4  # Rough estimate
            trajectory_memory = trajectory.nbytes
            memory_estimates.append((sequence_memory + trajectory_memory) / (1024**2))
        
        return {
            'config': asdict(config),
            'timing': {
                'mean_seconds': np.mean(times),
                'std_seconds': np.std(times),
                'min_seconds': min(times),
                'max_seconds': max(times)
            },
            'memory': {
                'mean_mb': np.mean(memory_estimates),
                'std_mb': np.std(memory_estimates),
                'trajectory_mb': trajectory.nbytes / (1024**2),
                'estimated_peak_mb': max(memory_estimates)
            },
            'throughput': {
                'sequences_per_second': config.n_sequences / np.mean(times),
                'symbols_per_second': sum(len(seq) for seq in sequences) / np.mean(times)
            }
        }
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary of workspace and created datasets."""
        return {
            'workspace_dir': str(self.workspace_dir),
            'created_datasets': self.created_datasets,
            'available_presets': list(self.preset_configs.keys()),
            'available_alphabets': list(self.preset_alphabets.keys())
        }


def create_research_datasets(output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dataset]:
    """
    Create standard research datasets for adaptive syntax filter development.
    
    Parameters
    ----------
    output_dir : Optional[Union[str, Path]]
        Output directory, or None for current directory
        
    Returns
    -------
    Dict[str, Dataset]
        Dictionary of created datasets
    """
    builder = DatasetBuilder(output_dir)
    
    datasets = {}
    
    # Create standard research datasets
    for preset_name in ['bengalese_finch', 'canary', 'minimal']:
        print(f"Creating {preset_name} dataset...")
        dataset = builder.create_from_preset(preset_name)
        datasets[preset_name] = dataset
        
        # Save dataset
        filepath = builder.save_dataset(dataset, format='pickle')
        print(f"Saved to: {filepath}")
        
        # Print summary
        stats = dataset.metadata.sequence_stats
        validation = dataset.metadata.validation_results
        print(f"  Sequences: {stats['n_sequences']}")
        print(f"  Validation rate: {validation['validation_rate']:.3f}")
        print(f"  Avg length: {stats['length_stats']['mean']:.1f}")
        print()
    
    return datasets 
