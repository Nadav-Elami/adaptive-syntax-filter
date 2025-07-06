"""Multi-order Markov sequence generation for canary song syntax modeling.

Implements the complete sequence generation pipeline from data_generation.md,
including higher-order softmax, constraint enforcement, and batch processing
for synthetic canary song data creation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings

from .constraint_system import ConstraintManager, apply_constraints_to_logits
from .temporal_evolution import compute_evolution_trajectory


def softmax_mc_higher_order(x: np.ndarray, alphabet: List[str], order: int) -> np.ndarray:
    """
    Multi-class softmax applied block-wise for higher-order Markov models.
    
    Each block represents transitions FROM one context TO all symbols.
    Softmax is applied within each block to ensure valid probabilities.
    
    Parameters
    ----------
    x : np.ndarray, shape (n_symbols^(order+1),)
        Logit parameter vector with blocks for each context
    alphabet : List[str]
        Symbol alphabet
    order : int
        Markov order
        
    Returns
    -------
    np.ndarray, shape (n_symbols^(order+1),)
        Probability vector with valid transitions (sums to 1 within each block)
        
    Notes
    -----
    For n_symbols=4, order=2: x has 64 elements arranged as 16 blocks of 4.
    Each block represents transitions from one 2-symbol context to all 4 symbols.
    Softmax is applied separately to each 4-element block.
    """
    n_symbols = len(alphabet)
    n_contexts = n_symbols ** order
    block_size = n_symbols
    
    if len(x) != n_symbols ** (order + 1):
        raise ValueError(f"Logit vector length {len(x)} != expected {n_symbols**(order+1)}")
    
    # Reshape into blocks: (n_contexts, block_size)
    x_blocks = x.reshape(n_contexts, block_size)
    p_blocks = np.zeros_like(x_blocks)
    
    # Apply softmax to each block separately
    for i in range(n_contexts):
        block = x_blocks[i]
        
        # Handle special constraint values
        if np.any(np.isinf(block)):
            # Block contains forbidden (-∞) or required (+∞) transitions
            finite_mask = np.isfinite(block)
            
            if not np.any(finite_mask):
                # All transitions are constrained
                if np.any(block > 1e7):
                    # Required transitions present - set to probability 1
                    required_mask = block > 1e7
                    p_blocks[i, required_mask] = 1.0 / np.sum(required_mask)
                else:
                    # All forbidden - uniform over feasible (shouldn't happen)
                    p_blocks[i] = 1.0 / block_size
            else:
                # Mix of finite and infinite values
                if np.any(block > 1e7):
                    # Required transitions present
                    required_mask = block > 1e7
                    p_blocks[i, required_mask] = 1.0 / np.sum(required_mask)
                else:
                    # Only forbidden transitions and finite values
                    finite_block = block[finite_mask]
                    finite_block_stable = finite_block - np.max(finite_block)
                    exp_vals = np.exp(finite_block_stable)
                    probabilities = exp_vals / np.sum(exp_vals)
                    p_blocks[i, finite_mask] = probabilities
        else:
            # Standard softmax - no constraints
            block_stable = block - np.max(block)
            exp_vals = np.exp(block_stable)
            p_blocks[i] = exp_vals / np.sum(exp_vals)
    
    # Flatten back to original shape
    return p_blocks.flatten()


def generate_single_sequence(x: np.ndarray,
                           alphabet: List[str], 
                           order: int,
                           max_length: int = 50,
                           min_phrase_length: int = 2,
                           seed: Optional[int] = None) -> List[str]:
    """
    Generate a single sequence using higher-order Markov model.
    
    Parameters
    ----------
    x : np.ndarray
        Logit parameter vector  
    alphabet : List[str]
        Symbol alphabet with format ['<', phrase1, ..., phraseN, '>']
    order : int
        Markov order
    max_length : int
        Maximum sequence length (safety limit)
    min_phrase_length : int
        Minimum number of phrase symbols (excluding start/end tokens)
        Default: 2 to prevent meaningless empty songs
    seed : Optional[int]
        Random seed for reproducible generation
        
    Returns
    -------
    List[str]
        Generated sequence starting with '<' and ending with '>'
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_symbols = len(alphabet)
    
    # Convert logits to probabilities
    probabilities = softmax_mc_higher_order(x, alphabet, order)
    
    # Initialize sequence with start token
    sequence = ['<']
    
    # Generate sequence
    phrase_count = 0  # Count actual phrase symbols (not start/end)
    for step in range(max_length):
        # Handle context construction for higher-order models
        if order == 1:
            # For first-order, use the last symbol
            current_context = [sequence[-1]]
        else:
            # For higher-order models, we need to be very careful about context
            if len(sequence) == 1:
                # First transition: from '<' we can only go to phrase symbols or '>' (with restrictions)
                # For higher orders, we should NOT use ['<', '<', ...] as context
                # Instead, use a special "start context" approach
                
                # Create a pseudo-context that represents "start of sequence"
                # This is effectively ['<'] but we need to find the right encoding
                # For higher orders, we can use the context ['<'] padded with special handling
                
                # Use first-order-like behavior for the initial transition
                start_context_idx = alphabet.index('<')
                base_context_idx = start_context_idx * (n_symbols ** (order - 1))
                
                # Find a valid context that starts with '<' but doesn't have invalid patterns
                # Try using ['<', first_phrase_symbol] type contexts for the initial transition
                phrase_symbols = [s for s in alphabet if s not in ['<', '>']]
                
                if phrase_symbols:
                    # Use the context ['<', first_phrase] for the initial transition
                    # This represents the state after we've transitioned to the first phrase
                    temp_context = ['<'] + [phrase_symbols[0]] * (order - 1)
                    current_context = temp_context
                else:
                    # Fallback to first order behavior
                    current_context = ['<']
                    
            elif len(sequence) < order:
                # We have some history but not enough for full context
                # Pad with the first symbol only, not repeated '<'
                needed = order - len(sequence)
                current_context = [sequence[0]] + sequence[1:]
                
                # Make sure we don't create invalid contexts like ['<', '<']
                if len(current_context) < order:
                    # Add phrase symbols instead of repeating '<'
                    phrase_symbols = [s for s in alphabet if s not in ['<', '>']]
                    if phrase_symbols:
                        # Pad with a phrase symbol
                        current_context.extend([phrase_symbols[0]] * (order - len(current_context)))
                    else:
                        # Fallback - shouldn't happen
                        current_context.extend([alphabet[1]] * (order - len(current_context)))
                        
            else:
                # We have enough history - use the last `order` symbols
                current_context = sequence[-order:]
        
        # Validate that we don't have invalid contexts
        if len(current_context) == order and order > 1:
            # Check for problematic patterns
            if current_context.count('<') > 1:
                # Multiple start symbols - fix this
                phrase_symbols = [s for s in alphabet if s not in ['<', '>']]
                if phrase_symbols:
                    # Replace extra '<' with phrase symbols
                    fixed_context = []
                    start_count = 0
                    for sym in current_context:
                        if sym == '<':
                            if start_count == 0:
                                fixed_context.append(sym)
                                start_count += 1
                            else:
                                fixed_context.append(phrase_symbols[0])
                        else:
                            fixed_context.append(sym)
                    current_context = fixed_context
        
        # Encode context to get block index
        context_idx = 0
        for i, symbol in enumerate(current_context):
            if symbol in alphabet:
                symbol_idx = alphabet.index(symbol)
            else:
                # Fallback for invalid symbols
                symbol_idx = 1  # Use first phrase symbol
            context_idx += symbol_idx * (n_symbols ** (order - 1 - i))
        
        # Get transition probabilities for this context
        start_pos = context_idx * n_symbols
        end_pos = start_pos + n_symbols
        
        # Bounds check
        if start_pos >= len(probabilities):
            # Fallback to first context
            start_pos = 0
            end_pos = n_symbols
            
        transition_probs = probabilities[start_pos:end_pos].copy()
        
        # Additional constraints to prevent invalid sequences
        # Never allow '<' as a transition target after the first symbol
        if len(sequence) >= 1:
            start_token_idx = alphabet.index('<')
            transition_probs[start_token_idx] = 0.0
        
        # Enforce minimum length constraint
        if phrase_count < min_phrase_length:
            # Haven't reached minimum length yet - forbid end token
            end_token_idx = alphabet.index('>')
            transition_probs[end_token_idx] = 0.0
        
        # Renormalize probabilities if we modified them
        prob_sum = np.sum(transition_probs)
        if prob_sum > 0:
            transition_probs = transition_probs / prob_sum
        else:
            # All forbidden - force a phrase symbol
            phrase_symbols = [i for i, sym in enumerate(alphabet) if sym not in ['<', '>']]
            if phrase_symbols:
                transition_probs = np.zeros(n_symbols)
                transition_probs[phrase_symbols[0]] = 1.0
            else:
                # Fallback - shouldn't happen with valid alphabet
                transition_probs = np.ones(n_symbols) / n_symbols
        
        # Sample next symbol
        try:
            next_idx = np.random.choice(n_symbols, p=transition_probs)
            next_symbol = alphabet[next_idx]
        except ValueError:
            # Invalid probabilities - default to end token if we have minimum length
            if phrase_count >= min_phrase_length:
                next_symbol = '>'
            else:
                # Force a phrase symbol
                phrase_symbols = [sym for sym in alphabet if sym not in ['<', '>']]
                if phrase_symbols:
                    next_symbol = phrase_symbols[0]
                else:
                    warnings.warn("Invalid transition probabilities and no phrase symbols available")
                    next_symbol = '>'
        
        sequence.append(next_symbol)
        
        # Count phrase symbols (not start/end tokens)
        if next_symbol not in ['<', '>']:
            phrase_count += 1
        
        # Stop if we hit the end token (and have minimum length)
        if next_symbol == '>' and phrase_count >= min_phrase_length:
            break
    
    # Ensure sequence ends properly
    if sequence[-1] != '>':
        sequence.append('>')
    
    return sequence


def generate_sequence_batch(x_trajectory: np.ndarray,
                          alphabet: List[str],
                          order: int,
                          n_sequences: int,
                          max_length: int = 50,
                          min_phrase_length: int = 2,
                          seed: Optional[int] = None) -> List[List[str]]:
    """
    Generate a batch of sequences with evolving parameters.
    
    Parameters
    ----------
    x_trajectory : np.ndarray, shape (state_size, n_sequences)
        Parameter evolution trajectory
    alphabet : List[str]
        Symbol alphabet
    order : int
        Markov order
    n_sequences : int
        Number of sequences to generate
    max_length : int
        Maximum sequence length
    min_phrase_length : int
        Minimum number of phrase symbols (excluding start/end tokens)
    seed : Optional[int]
        Random seed for reproducibility
        
    Returns
    -------
    List[List[str]]
        List of generated sequences
    """
    if seed is not None:
        np.random.seed(seed)
    
    if x_trajectory.shape[1] != n_sequences:
        raise ValueError(f"Trajectory has {x_trajectory.shape[1]} columns, expected {n_sequences}")
    
    sequences = []
    
    for i in range(n_sequences):
        # Get parameters for this sequence
        x_current = x_trajectory[:, i]
        
        # Generate sequence with current parameters
        sequence = generate_single_sequence(
            x_current, alphabet, order, max_length, min_phrase_length,
            seed=None  # Use global random state for batch consistency
        )
        
        sequences.append(sequence)
    
    return sequences


@dataclass
class GenerationConfig:
    """Configuration for sequence generation."""
    alphabet: List[str]
    order: int
    n_sequences: int
    max_length: int = 50
    min_phrase_length: int = 2
    evolution_type: str = 'linear'
    batch_size: Optional[int] = None
    evolution_params: Dict = None
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.evolution_params is None:
            self.evolution_params = {}


def initialize_generation_parameters(alphabet: List[str], 
                                   order: int = 1, 
                                   seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize parameters for higher-order Markov data generation.
    
    Parameters
    ----------
    alphabet : List[str]
        Symbol alphabet including start/end tokens
        Must have format: ['<', phrase1, phrase2, ..., phraseN, '>']
        Minimum size: 3 (start + 1 phrase + end)
    order : int
        Markov order (1=first order, 2=second order, etc.)
        Higher orders require exponentially more memory
    seed : Optional[int]
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (x_init, x_final) - initial and final parameter vectors
        
    Raises
    ------
    ValueError
        If alphabet size < 3, improper format, or order < 1
    """
    # Validate alphabet format
    if len(alphabet) < 3:
        raise ValueError("Alphabet must have at least 3 symbols: ['<', phrase, '>']")
    if alphabet[0] != '<' or alphabet[-1] != '>':
        raise ValueError("Alphabet must start with '<' and end with '>'")
    if order < 1:
        raise ValueError("Markov order must be at least 1")
    
    if seed is not None:
        np.random.seed(seed)
    
    n_symbols = len(alphabet)
    state_space_size = n_symbols ** (order + 1)
    
    # Initialize parameters with small random values
    x_init = np.random.normal(0, 0.5, state_space_size)
    x_final = np.random.normal(0, 0.5, state_space_size)
    
    # Apply constraints using constraint manager
    constraint_manager = ConstraintManager(alphabet, order)
    x_init = constraint_manager.apply_constraints(x_init)
    x_final = constraint_manager.apply_constraints(x_final)
    
    return x_init, x_final


def generate_dataset(config: GenerationConfig) -> Tuple[List[List[str]], np.ndarray]:
    """
    Generate complete dataset according to configuration.
    
    Parameters
    ----------
    config : GenerationConfig
        Generation configuration
        
    Returns
    -------
    Tuple[List[List[str]], np.ndarray]
        (sequences, parameter_trajectory)
    """
    # Initialize parameters
    x_init, x_final = initialize_generation_parameters(
        config.alphabet, config.order, config.seed
    )
    
    # Compute evolution trajectory
    x_trajectory = compute_evolution_trajectory(
        x_init, x_final, config.n_sequences, config.evolution_type,
        config.batch_size, **config.evolution_params
    )
    
    # Generate sequences
    sequences = generate_sequence_batch(
        x_trajectory, config.alphabet, config.order, config.n_sequences,
        config.max_length, config.min_phrase_length, config.seed
    )
    
    return sequences, x_trajectory


class SequenceGenerator:
    """Main class for sequence generation with higher-order Markov models."""
    
    def __init__(self, 
                 alphabet: List[str], 
                 order: int = 1,
                 max_length: int = 50):
        """
        Initialize sequence generator.
        
        Parameters
        ----------
        alphabet : List[str]
            Symbol alphabet
        order : int
            Markov order
        max_length : int
            Maximum sequence length
        """
        self.alphabet = alphabet
        self.order = order
        self.max_length = max_length
        self.n_symbols = len(alphabet)
        self.state_space_size = self.n_symbols ** (order + 1)
        
        # Initialize constraint manager
        self.constraint_manager = ConstraintManager(alphabet, order)
        
        # Validate setup
        if self.state_space_size > 10000:
            warnings.warn(f"Large state space ({self.state_space_size}) may be slow")
    
    def initialize_parameters(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize random parameters."""
        return initialize_generation_parameters(self.alphabet, self.order, seed)
    
    def generate_single(self, 
                       x: np.ndarray, 
                       min_phrase_length: int = 2,
                       seed: Optional[int] = None) -> List[str]:
        """Generate single sequence."""
        return generate_single_sequence(x, self.alphabet, self.order, 
                                      self.max_length, min_phrase_length, seed)
    
    def generate_batch(self,
                      x_trajectory: np.ndarray,
                      min_phrase_length: int = 2,
                      seed: Optional[int] = None) -> List[List[str]]:
        """Generate batch of sequences."""
        n_sequences = x_trajectory.shape[1]
        return generate_sequence_batch(x_trajectory, self.alphabet, self.order,
                                     n_sequences, self.max_length, min_phrase_length, seed)
    
    def generate_dataset_from_config(self, config: GenerationConfig) -> Tuple[List[List[str]], np.ndarray]:
        """Generate dataset from configuration."""
        # Override config alphabet and order with instance values
        config.alphabet = self.alphabet
        config.order = self.order
        return generate_dataset(config)
    
    def get_probabilities(self, x: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities."""
        return softmax_mc_higher_order(x, self.alphabet, self.order)
    
    def analyze_setup(self) -> Dict:
        """Analyze generator setup."""
        return self.constraint_manager.analyze(verbose=False)
    
    def __repr__(self) -> str:
        return (f"SequenceGenerator(alphabet_size={self.n_symbols}, "
                f"order={self.order}, state_space={self.state_space_size})")


def validate_generated_sequences(sequences: List[List[str]], 
                               alphabet: List[str]) -> Dict[str, any]:
    """
    Validate generated sequences for structural correctness.
    
    Parameters
    ----------
    sequences : List[List[str]]
        Generated sequences to validate
    alphabet : List[str]
        Symbol alphabet
        
    Returns
    -------
    dict
        Validation results
    """
    results = {
        'total_sequences': len(sequences),
        'valid_sequences': 0,
        'errors': []
    }
    
    for i, seq in enumerate(sequences):
        try:
            # Check start and end tokens
            if len(seq) == 0:
                results['errors'].append(f"Sequence {i}: Empty sequence")
                continue
            
            if seq[0] != '<':
                results['errors'].append(f"Sequence {i}: Does not start with '<'")
                continue
                
            if seq[-1] != '>':
                results['errors'].append(f"Sequence {i}: Does not end with '>'")
                continue
            
            # Check all symbols are in alphabet
            for symbol in seq:
                if symbol not in alphabet:
                    results['errors'].append(f"Sequence {i}: Unknown symbol '{symbol}'")
                    break
            else:
                # All checks passed
                results['valid_sequences'] += 1
        
        except Exception as e:
            results['errors'].append(f"Sequence {i}: Validation error {e}")
    
    results['validation_rate'] = results['valid_sequences'] / results['total_sequences']
    return results


def create_standard_configs() -> Dict[str, GenerationConfig]:
    """Create standard generation configurations for different scales."""
    return {
        'bengalese_finch': GenerationConfig(
            alphabet=['<', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', '>'],
            order=2,
            n_sequences=300,
            evolution_type='sigmoid',
            batch_size=15,
            evolution_params={'steepness': 5.0, 'midpoint': 0.6}
        ),
        
        'canary': GenerationConfig(
            alphabet=['<'] + [f'p{i}' for i in range(38)] + ['>'],
            order=3,
            n_sequences=10000,
            evolution_type='piecewise',
            batch_size=25,
            evolution_params={'breakpoints': [0.3, 0.7]}
        ),
        
        'minimal': GenerationConfig(
            alphabet=['<', 'a', 'b', '>'],
            order=1,
            n_sequences=100,
            evolution_type='linear'
        ),
        
        'test': GenerationConfig(
            alphabet=['<', 'x', 'y', 'z', '>'],
            order=2,
            n_sequences=50,
            evolution_type='constant',
            max_length=20
        )
    } 


def sequences_to_observations(sequences: List[List[str]], alphabet: List[str]) -> List[np.ndarray]:
    """Convert list of sequences to numerical observation format for EM algorithm.
    
    Converts symbolic sequences to numerical arrays where each symbol is replaced
    by its index in the alphabet. This is the required format for Kalman filtering
    and EM algorithm processing.
    
    Parameters
    ----------
    sequences : List[List[str]]
        List of symbolic sequences (e.g., [['<', 'A', 'B', '>'], ['<', 'C', '>']])
    alphabet : List[str]
        Symbol alphabet defining the mapping from symbols to indices
        
    Returns
    -------
    List[np.ndarray]
        List of numerical observation sequences where each sequence is an array
        of symbol indices
        
    Examples
    --------
    >>> alphabet = ['<', 'A', 'B', '>']
    >>> sequences = [['<', 'A', 'B', '>'], ['<', 'B', '>']]
    >>> obs = sequences_to_observations(sequences, alphabet)
    >>> print(obs[0])  # [0, 1, 2, 3]
    >>> print(obs[1])  # [0, 2, 3]
    """
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(alphabet)}
    observations = []
    
    for seq in sequences:
        obs_seq = [symbol_to_idx[symbol] for symbol in seq]
        observations.append(np.array(obs_seq))
    
    return observations
