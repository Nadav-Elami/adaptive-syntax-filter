"""Alphabet management for dynamic sizing and symbol handling.

Provides utilities for creating, validating, and managing symbol alphabets
for different scales of birdsong analysis, from minimal 3-symbol alphabets
up to large 40+ symbol alphabets for complex canary song analysis.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
import string


@dataclass
class AlphabetStats:
    """Statistics and properties of an alphabet configuration.
    
    Attributes
    ----------
    size : int
        Total alphabet size (including start/end tokens)
    n_phrases : int
        Number of phrase symbols (excluding start/end)
    memory_per_sequence_mb : float
        Memory requirement per sequence for given Markov order
    max_recommended_order : int
        Maximum recommended Markov order for memory efficiency
    state_space_size : int
        Total state space size for given order
    """
    size: int
    n_phrases: int
    memory_per_sequence_mb: float
    max_recommended_order: int
    state_space_size: int


def create_standard_alphabet(size: int, naming_scheme: str = 'letters') -> List[str]:
    """
    Create standard alphabet with specified size and naming scheme.
    
    Parameters
    ----------
    size : int
        Total alphabet size (minimum 3: start + 1 phrase + end)
    naming_scheme : str
        'letters' (a,b,c...), 'numbers' (p0,p1,p2...), or 'custom'
        
    Returns
    -------
    List[str]
        Alphabet with format ['<', phrase1, phrase2, ..., phraseN, '>']
        
    Examples
    --------
    >>> create_standard_alphabet(5, 'letters')
    ['<', 'a', 'b', 'c', '>']
    >>> create_standard_alphabet(5, 'numbers')
    ['<', 'p0', 'p1', 'p2', '>']
    """
    if size < 3:
        raise ValueError("Alphabet size must be at least 3 (start + phrase + end)")
    
    n_phrases = size - 2
    
    if naming_scheme == 'letters':
        if n_phrases <= 26:
            # Use single letters a-z
            phrase_symbols = [string.ascii_lowercase[i] for i in range(n_phrases)]
        else:
            # Use double letters aa, ab, ac, ... ba, bb, bc, ...
            phrase_symbols = []
            for i in range(n_phrases):
                if i < 26:
                    phrase_symbols.append(string.ascii_lowercase[i])
                else:
                    # aa, ab, ac, ... ba, bb, bc, ...
                    first = string.ascii_lowercase[(i - 26) // 26]
                    second = string.ascii_lowercase[i % 26]
                    phrase_symbols.append(first + second)
                    
    elif naming_scheme == 'numbers':
        phrase_symbols = [f'p{i}' for i in range(n_phrases)]
        
    else:
        raise ValueError("Naming scheme must be 'letters' or 'numbers'")
    
    return ['<'] + phrase_symbols + ['>']


def validate_alphabet(alphabet: List[str]) -> Dict[str, any]:
    """
    Validate alphabet format and provide diagnostic information.
    
    Parameters
    ----------
    alphabet : List[str]
        Alphabet to validate
        
    Returns
    -------
    dict
        Validation results with diagnostics
    """
    results = {
        'valid': True,
        'size': len(alphabet),
        'errors': [],
        'warnings': [],
        'properties': {}
    }
    
    try:
        # Basic size check
        if len(alphabet) < 3:
            results['valid'] = False
            results['errors'].append("Alphabet must have at least 3 symbols")
            return results
        
        # Check start/end tokens
        if alphabet[0] != '<':
            results['valid'] = False
            results['errors'].append("Alphabet must start with '<'")
        
        if alphabet[-1] != '>':
            results['valid'] = False
            results['errors'].append("Alphabet must end with '>'")
        
        # Check for duplicates
        if len(set(alphabet)) != len(alphabet):
            results['valid'] = False
            results['errors'].append("Alphabet contains duplicate symbols")
        
        # Check phrase symbols
        phrase_symbols = alphabet[1:-1]
        results['properties']['n_phrases'] = len(phrase_symbols)
        
        # Check for invalid characters in phrase symbols
        for symbol in phrase_symbols:
            if symbol in ['<', '>']:
                results['warnings'].append(f"Phrase symbol '{symbol}' conflicts with start/end tokens")
            if len(symbol) == 0:
                results['valid'] = False
                results['errors'].append("Empty phrase symbol found")
        
        # Size warnings
        if len(alphabet) > 50:
            results['warnings'].append("Very large alphabet (>50) may cause memory issues")
        elif len(alphabet) > 20:
            results['warnings'].append("Large alphabet (>20) may be slow for high Markov orders")
        
        # Calculate properties
        results['properties']['phrase_symbols'] = phrase_symbols
        results['properties']['naming_pattern'] = _detect_naming_pattern(phrase_symbols)
        
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Validation error: {e}")
    
    return results


def _detect_naming_pattern(phrase_symbols: List[str]) -> str:
    """Detect naming pattern in phrase symbols."""
    if not phrase_symbols:
        return 'empty'
    
    # Check for letter pattern
    if all(len(s) == 1 and s.islower() and s.isalpha() for s in phrase_symbols):
        return 'single_letters'
    
    # Check for numbered pattern
    if all(s.startswith('p') and s[1:].isdigit() for s in phrase_symbols):
        return 'numbered'
    
    # Check for double letter pattern
    if all(len(s) == 2 and s.islower() and s.isalpha() for s in phrase_symbols):
        return 'double_letters'
    
    return 'custom'


def estimate_memory_requirements(alphabet_size: int, 
                               markov_order: int = 1,
                               n_sequences: int = 1000) -> Dict[str, float]:
    """
    Estimate memory requirements for given alphabet and model configuration.
    
    Parameters
    ----------
    alphabet_size : int
        Size of alphabet
    markov_order : int
        Markov model order
    n_sequences : int
        Number of sequences to generate
        
    Returns
    -------
    dict
        Memory estimates in different units
    """
    # State space size grows exponentially with order
    state_space_size = alphabet_size ** (markov_order + 1)
    
    # Memory per sequence (float64 = 8 bytes)
    bytes_per_sequence = state_space_size * 8
    
    # Total memory for trajectory
    total_bytes = bytes_per_sequence * n_sequences
    
    # Additional memory for sequences themselves (rough estimate)
    avg_sequence_length = 10  # Estimated average
    sequence_memory = n_sequences * avg_sequence_length * 4  # Rough estimate
    
    return {
        'state_space_size': state_space_size,
        'bytes_per_sequence': bytes_per_sequence,
        'kb_per_sequence': bytes_per_sequence / 1024,
        'mb_per_sequence': bytes_per_sequence / (1024**2),
        'total_trajectory_mb': total_bytes / (1024**2),
        'total_sequences_mb': sequence_memory / (1024**2),
        'total_estimated_mb': (total_bytes + sequence_memory) / (1024**2)
    }


def get_recommended_order_limits(alphabet_size: int, 
                               memory_limit_mb: float = 100.0) -> Dict[str, int]:
    """
    Get recommended Markov order limits based on memory constraints.
    
    Parameters
    ----------
    alphabet_size : int
        Size of alphabet
    memory_limit_mb : float
        Memory limit in megabytes
        
    Returns
    -------
    dict
        Recommended order limits for different use cases
    """
    recommendations = {}
    
    for order in range(1, 6):  # Test orders 1-5
        memory_est = estimate_memory_requirements(alphabet_size, order, 1000)
        
        if memory_est['total_estimated_mb'] <= memory_limit_mb:
            recommendations['safe_order'] = order
        
        if memory_est['total_estimated_mb'] <= memory_limit_mb * 0.1:  # Very conservative
            recommendations['conservative_order'] = order
            
        if memory_est['total_estimated_mb'] <= memory_limit_mb * 10:  # Liberal
            recommendations['max_feasible_order'] = order
    
    # Provide defaults if no limits found
    recommendations.setdefault('safe_order', 1)
    recommendations.setdefault('conservative_order', 1) 
    recommendations.setdefault('max_feasible_order', 2)
    
    return recommendations


def analyze_alphabet_scaling(alphabet_sizes: List[int], 
                           markov_orders: List[int]) -> Dict[str, any]:
    """
    Analyze how memory requirements scale with alphabet size and order.
    
    Parameters
    ----------
    alphabet_sizes : List[int]
        List of alphabet sizes to analyze
    markov_orders : List[int]
        List of Markov orders to analyze
        
    Returns
    -------
    dict
        Scaling analysis results
    """
    results = {
        'alphabet_sizes': alphabet_sizes,
        'markov_orders': markov_orders,
        'memory_grid': {},
        'recommendations': {}
    }
    
    # Compute memory requirements for all combinations
    for size in alphabet_sizes:
        results['memory_grid'][size] = {}
        for order in markov_orders:
            memory_est = estimate_memory_requirements(size, order, 1000)
            results['memory_grid'][size][order] = memory_est['total_estimated_mb']
    
    # Find sweet spots
    for size in alphabet_sizes:
        size_recs = get_recommended_order_limits(size)
        results['recommendations'][size] = size_recs
    
    return results


class AlphabetManager:
    """Manager for alphabet creation, validation, and optimization."""
    
    def __init__(self, alphabet: Optional[List[str]] = None, 
                 alphabet_size: Optional[int] = None,
                 naming_scheme: str = 'letters'):
        """
        Initialize alphabet manager.
        
        Parameters
        ----------
        alphabet : Optional[List[str]]
            Existing alphabet to manage, or None to create new
        alphabet_size : Optional[int]  
            Size for new alphabet creation (if alphabet is None)
        naming_scheme : str
            Naming scheme for new alphabet creation
        """
        if alphabet is not None:
            self.alphabet = alphabet
        elif alphabet_size is not None:
            self.alphabet = create_standard_alphabet(alphabet_size, naming_scheme)
        else:
            raise ValueError("Must provide either alphabet or alphabet_size")
        
        # Validate alphabet
        self.validation = validate_alphabet(self.alphabet)
        if not self.validation['valid']:
            raise ValueError(f"Invalid alphabet: {self.validation['errors']}")
        
        # Store properties
        self.size = len(self.alphabet)
        self.n_phrases = self.size - 2
        self.phrase_symbols = self.alphabet[1:-1]
        
        # Issue warnings if any
        for warning in self.validation['warnings']:
            warnings.warn(warning)
    
    def get_stats(self, markov_order: int = 1, n_sequences: int = 1000) -> AlphabetStats:
        """Get alphabet statistics for given configuration."""
        memory_est = estimate_memory_requirements(self.size, markov_order, n_sequences)
        order_limits = get_recommended_order_limits(self.size)
        
        return AlphabetStats(
            size=self.size,
            n_phrases=self.n_phrases,
            memory_per_sequence_mb=memory_est['mb_per_sequence'],
            max_recommended_order=order_limits['safe_order'],
            state_space_size=memory_est['state_space_size']
        )
    
    def recommend_order(self, memory_limit_mb: float = 100.0) -> Dict[str, int]:
        """Recommend Markov orders based on memory constraints."""
        return get_recommended_order_limits(self.size, memory_limit_mb)
    
    def create_subset(self, n_phrases: int) -> 'AlphabetManager':
        """Create subset alphabet with fewer phrase symbols."""
        if n_phrases >= self.n_phrases:
            return self
        if n_phrases < 1:
            raise ValueError("Must have at least 1 phrase symbol")
        
        subset_alphabet = ['<'] + self.phrase_symbols[:n_phrases] + ['>']
        return AlphabetManager(subset_alphabet)
    
    def extend(self, additional_phrases: int, naming_scheme: str = 'numbers') -> 'AlphabetManager':
        """Extend alphabet with additional phrase symbols."""
        if additional_phrases <= 0:
            return self
        
        # Generate new symbols
        if naming_scheme == 'numbers':
            new_symbols = [f'p{self.n_phrases + i}' for i in range(additional_phrases)]
        elif naming_scheme == 'letters':
            start_idx = self.n_phrases
            new_symbols = []
            for i in range(additional_phrases):
                idx = start_idx + i
                if idx < 26:
                    new_symbols.append(string.ascii_lowercase[idx])
                else:
                    # Use double letters for overflow
                    first = string.ascii_lowercase[(idx - 26) // 26]
                    second = string.ascii_lowercase[idx % 26]
                    new_symbols.append(first + second)
        else:
            raise ValueError("Naming scheme must be 'letters' or 'numbers'")
        
        extended_alphabet = self.alphabet[:-1] + new_symbols + ['>']
        return AlphabetManager(extended_alphabet)
    
    def analyze_scaling(self, max_order: int = 5) -> Dict[str, any]:
        """Analyze memory scaling for this alphabet."""
        orders = list(range(1, max_order + 1))
        return analyze_alphabet_scaling([self.size], orders)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary representation."""
        return {
            'alphabet': self.alphabet,
            'size': self.size,
            'n_phrases': self.n_phrases,
            'phrase_symbols': self.phrase_symbols,
            'validation': self.validation
        }
    
    def __repr__(self) -> str:
        return f"AlphabetManager(size={self.size}, phrases={self.n_phrases})"
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index: int) -> str:
        return self.alphabet[index]
    
    def __iter__(self):
        return iter(self.alphabet)


def create_preset_alphabets() -> Dict[str, AlphabetManager]:
    """Create preset alphabet configurations for common use cases."""
    presets = {
        # Research-scale configurations
        'bengalese_finch': AlphabetManager(alphabet_size=16, naming_scheme='letters'),  # 14 phrases
        'canary': AlphabetManager(alphabet_size=40, naming_scheme='numbers'),           # 38 phrases
        
        # Development and testing
        'minimal': AlphabetManager(alphabet_size=3, naming_scheme='letters'),          # 1 phrase
        'small': AlphabetManager(alphabet_size=5, naming_scheme='letters'),            # 3 phrases
        'medium': AlphabetManager(alphabet_size=10, naming_scheme='letters'),          # 8 phrases
        'large': AlphabetManager(alphabet_size=20, naming_scheme='letters'),           # 18 phrases
        
        # Specific test cases
        'test_3sym': AlphabetManager(['<', 'a', '>']),
        'test_4sym': AlphabetManager(['<', 'x', 'y', '>']),
        'test_5sym': AlphabetManager(['<', 'a', 'b', 'c', '>']),
    }
    
    return presets


def compare_alphabets(alphabets: Dict[str, AlphabetManager], 
                     markov_order: int = 2) -> Dict[str, any]:
    """
    Compare memory requirements and properties of different alphabets.
    
    Parameters
    ----------
    alphabets : Dict[str, AlphabetManager]
        Dictionary of alphabet managers to compare
    markov_order : int
        Markov order for memory calculations
        
    Returns
    -------
    dict
        Comparison results
    """
    comparison = {
        'markov_order': markov_order,
        'alphabets': {},
        'ranking': {}
    }
    
    # Analyze each alphabet
    for name, manager in alphabets.items():
        stats = manager.get_stats(markov_order)
        comparison['alphabets'][name] = {
            'size': stats.size,
            'n_phrases': stats.n_phrases,
            'state_space_size': stats.state_space_size,
            'memory_per_sequence_mb': stats.memory_per_sequence_mb,
            'max_recommended_order': stats.max_recommended_order
        }
    
    # Create rankings
    by_size = sorted(alphabets.items(), key=lambda x: x[1].size)
    by_memory = sorted(alphabets.items(), 
                      key=lambda x: x[1].get_stats(markov_order).memory_per_sequence_mb)
    
    comparison['ranking'] = {
        'by_size': [name for name, _ in by_size],
        'by_memory_efficiency': [name for name, _ in by_memory]
    }
    
    return comparison


def optimize_alphabet_for_constraints(target_phrases: int,
                                    memory_limit_mb: float = 100.0,
                                    min_order: int = 1,
                                    preferred_naming: str = 'letters') -> AlphabetManager:
    """
    Create optimized alphabet given constraints.
    
    Parameters
    ----------
    target_phrases : int
        Desired number of phrase symbols
    memory_limit_mb : float
        Memory limit in MB
    min_order : int
        Minimum required Markov order support
    preferred_naming : str
        Preferred naming scheme
        
    Returns
    -------
    AlphabetManager
        Optimized alphabet manager
    """
    alphabet_size = target_phrases + 2
    
    # Check if target is feasible
    memory_est = estimate_memory_requirements(alphabet_size, min_order, 1000)
    if memory_est['total_estimated_mb'] > memory_limit_mb:
        # Reduce alphabet size to fit memory constraint
        for size in range(3, target_phrases + 3):
            memory_est = estimate_memory_requirements(size, min_order, 1000)
            if memory_est['total_estimated_mb'] <= memory_limit_mb:
                alphabet_size = size
                break
        else:
            warnings.warn(f"Cannot satisfy constraints, using minimal alphabet")
            alphabet_size = 3
    
    return AlphabetManager(alphabet_size=alphabet_size, naming_scheme=preferred_naming) 
