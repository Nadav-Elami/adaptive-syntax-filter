"""Constraint system for canary song grammar rules in higher-order Markov models.

Implements the comprehensive constraint logic from data_generation.md section 1.3,
supporting 1st through 5th order Markov models with automatic context encoding
and validation for proper syntax enforcement.
"""

import numpy as np
from typing import List, Tuple, Dict, Set
import warnings
from itertools import product


def encode_context(context_symbols: List[str], alphabet: List[str], order: int) -> int:
    """
    Encode a context sequence of symbols into a single index.
    
    Converts a context (sequence of symbols) into a single integer index
    for addressing the appropriate block in the state vector.
    
    Parameters
    ----------
    context_symbols : List[str]
        Last `order` symbols forming the context
        E.g., ['a', 'b'] for 2nd order, ['<', 'a', 'b'] for 3rd order
    alphabet : List[str]
        Symbol alphabet with format ['<', phrase1, ..., phraseN, '>']
    order : int
        Markov order (1, 2, 3, ...)
        
    Returns
    -------
    int
        Context index in range [0, n_symbols^order - 1]
        
    Examples
    --------
    >>> alphabet = ['<', 'a', 'b', '>']
    >>> encode_context(['<', 'a'], alphabet, 2)  # Context '<a'
    1
    >>> encode_context(['a', 'b'], alphabet, 2)   # Context 'ab'
    6
    """
    if len(context_symbols) != order:
        raise ValueError(f"Context must have exactly {order} symbols, got {len(context_symbols)}")
    
    n_symbols = len(alphabet)
    context_idx = 0
    
    for i, symbol in enumerate(context_symbols):
        if symbol not in alphabet:
            raise ValueError(f"Symbol '{symbol}' not in alphabet {alphabet}")
        
        symbol_idx = alphabet.index(symbol)
        power = order - 1 - i
        context_idx += symbol_idx * (n_symbols ** power)
    
    return context_idx


def decode_context(context_idx: int, alphabet: List[str], order: int) -> List[str]:
    """
    Decode context index back to sequence of symbols.
    
    Parameters
    ----------
    context_idx : int
        Context index in range [0, n_symbols^order - 1]
    alphabet : List[str]
        Symbol alphabet
    order : int
        Markov order
        
    Returns
    -------
    List[str]
        Sequence of symbol names forming the context
        
    Examples
    --------
    >>> alphabet = ['<', 'a', 'b', '>']
    >>> decode_context(1, alphabet, 2)
    ['<', 'a']
    >>> decode_context(6, alphabet, 2)
    ['a', 'b']
    """
    n_symbols = len(alphabet)
    if not (0 <= context_idx < n_symbols ** order):
        raise ValueError(f"Context index {context_idx} out of range [0, {n_symbols**order})")
    
    context_symbols = []
    
    for i in range(order):
        power = order - 1 - i
        symbol_idx = (context_idx // (n_symbols ** power)) % n_symbols
        context_symbols.append(alphabet[symbol_idx])
    
    return context_symbols


def generate_all_contexts(alphabet: List[str], order: int) -> List[List[str]]:
    """
    Generate all possible context sequences of given order.
    
    Parameters
    ----------
    alphabet : List[str]
        Symbol alphabet
    order : int
        Markov order
        
    Returns
    -------
    List[List[str]]
        List of all possible context sequences of length `order`
    """
    n_symbols = len(alphabet)
    contexts = []
    
    # Generate all combinations using itertools.product
    for indices in product(range(n_symbols), repeat=order):
        context = [alphabet[idx] for idx in indices]
        contexts.append(context)
    
    return contexts


def is_valid_context(context_symbols: List[str], alphabet: List[str]) -> bool:
    """
    Check if a context sequence is theoretically possible in valid songs.
    
    A context is valid if:
    1. '<' only appears at the beginning of sequences (not after other symbols)
    2. '>' only appears at the end of sequences 
    3. No '<' appears after any other symbol
    4. No other symbol appears after '>'
    
    Parameters
    ----------
    context_symbols : List[str]
        Context sequence to validate
    alphabet : List[str]
        Symbol alphabet
        
    Returns
    -------
    bool
        True if context could appear in valid sequences
    """
    # Check for '<' in non-initial positions (within this context)
    if '<' in context_symbols[1:]:
        return False
    
    # Check for symbols after '>'
    try:
        end_idx = context_symbols.index('>')
        if end_idx < len(context_symbols) - 1:
            return False
    except ValueError:
        pass  # '>' not in context, which is fine
    
    return True


def get_constraint_positions_higher_order(alphabet: List[str], order: int) -> Tuple[List[int], List[int]]:
    """
    Get constraint positions for higher-order Markov models.
    
    Implements the complete constraint logic from data_generation.md section 1.3,
    handling all cases for start tokens, end tokens, and invalid context formations.
    
    Parameters
    ----------
    alphabet : List[str]
        Symbol alphabet with format ['<', phrase1, ..., phraseN, '>']
    order : int
        Markov order (1, 2, 3, ...)
        
    Returns
    -------
    Tuple[List[int], List[int]]
        (forbidden_positions, required_positions) in state space
        
    Notes
    -----
    Forbidden positions get logit value -∞ (impossible transitions)
    Required positions get logit value +∞ (forced transitions)
    """
    if len(alphabet) < 3:
        raise ValueError("Alphabet must have at least 3 symbols: ['<', phrase, '>']")
    if alphabet[0] != '<' or alphabet[-1] != '>':
        raise ValueError("Alphabet must start with '<' and end with '>'")
    if order < 1:
        raise ValueError("Markov order must be at least 1")
    
    n_symbols = len(alphabet)
    forbidden = []
    required = []
    
    # Generate all possible contexts of length `order`
    all_contexts = generate_all_contexts(alphabet, order)
    
    for context_symbols in all_contexts:
        context_idx = encode_context(context_symbols, alphabet, order)
        base_position = context_idx * n_symbols
        
        # Apply constraints based on context content
        if '>' in context_symbols:
            # Contexts containing '>' (end symbol)
            end_positions = [i for i, s in enumerate(context_symbols) if s == '>']
            rightmost_end = max(end_positions)
            
            if rightmost_end < order - 1:
                # '>' is not the rightmost symbol in context
                # This should never happen in valid sequences
                # All transitions from this context are forbidden
                for target_idx in range(n_symbols):
                    forbidden.append(base_position + target_idx)
            else:
                # '>' is the rightmost symbol - can only transition to '<'
                for target_idx in range(n_symbols):
                    if alphabet[target_idx] == '<':
                        required.append(base_position + target_idx)
                    else:
                        forbidden.append(base_position + target_idx)
        
        elif '<' in context_symbols[1:]:
            # '<' appears after the first position in context
            # Invalid context: '<' should only appear at sequence start
            for target_idx in range(n_symbols):
                forbidden.append(base_position + target_idx)
        
        else:
            # Normal context without end tokens or misplaced start tokens
            # Apply standard transition constraints
            for target_idx in range(n_symbols):
                target_symbol = alphabet[target_idx]
                
                if target_symbol == '<':
                    # '<' can never be a transition target (except after '>')
                    # This prevents sequences like: a → < or < → <
                    forbidden.append(base_position + target_idx)
                
                elif target_symbol == '>' and context_symbols.count('<') > 0:
                    # Prevent premature ending when context contains start tokens
                    # For higher orders, check if we have enough content
                    if order > 1:
                        # Count non-start symbols in context
                        non_start_symbols = [s for s in context_symbols if s != '<']
                        if len(non_start_symbols) < order - 1:
                            # Not enough phrase content - forbid ending
                            forbidden.append(base_position + target_idx)
    
    return forbidden, required


def apply_constraints_to_logits(x: np.ndarray, 
                               forbidden_positions: List[int], 
                               required_positions: List[int]) -> np.ndarray:
    """
    Apply constraint positions to logit vector.
    
    Parameters
    ----------
    x : np.ndarray, shape (n_symbols^(order+1),)
        Logit parameter vector
    forbidden_positions : List[int]
        Positions that should be set to -∞ (impossible transitions)
    required_positions : List[int]
        Positions that should be set to +∞ (forced transitions)  
        
    Returns
    -------
    np.ndarray
        Constrained logit vector
    """
    x_constrained = x.copy()
    
    # Apply forbidden constraints (logit = -∞)
    for pos in forbidden_positions:
        x_constrained[pos] = -np.inf
    
    # Apply required constraints (logit = large positive value)
    for pos in required_positions:
        x_constrained[pos] = 1e8  # Large but finite for numerical stability
    
    return x_constrained


def validate_constraint_setup(alphabet: List[str], order: int) -> Dict[str, any]:
    """
    Validate constraint setup and provide diagnostic information.
    
    Parameters
    ----------
    alphabet : List[str]
        Symbol alphabet
    order : int
        Markov order
        
    Returns
    -------
    Dict[str, any]
        Validation results and statistics
    """
    try:
        # Basic validation
        if len(alphabet) < 3:
            raise ValueError("Alphabet too small")
        if alphabet[0] != '<' or alphabet[-1] != '>':
            raise ValueError("Invalid alphabet format")
        if order < 1 or order > 5:
            raise ValueError("Order must be 1-5")
        
        # Generate constraints
        forbidden, required = get_constraint_positions_higher_order(alphabet, order)
        
        # Calculate statistics
        n_symbols = len(alphabet)
        state_space_size = n_symbols ** (order + 1)
        n_contexts = n_symbols ** order
        
        # Count valid vs invalid contexts
        all_contexts = generate_all_contexts(alphabet, order)
        valid_contexts = [ctx for ctx in all_contexts if is_valid_context(ctx, alphabet)]
        
        # Memory estimation
        memory_per_sequence = state_space_size * 8  # float64
        
        return {
            'valid': True,
            'alphabet_size': n_symbols,
            'order': order,
            'state_space_size': state_space_size,
            'n_contexts': n_contexts,
            'forbidden_positions': len(forbidden),
            'required_positions': len(required),
            'constraint_ratio': (len(forbidden) + len(required)) / state_space_size,
            'valid_contexts': len(valid_contexts),
            'invalid_contexts': len(all_contexts) - len(valid_contexts),
            'memory_per_sequence_bytes': memory_per_sequence,
            'estimated_constraint_coverage': len(forbidden) / state_space_size,
            'warnings': []
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'alphabet_size': len(alphabet) if alphabet else 0,
            'order': order
        }


def analyze_constraint_structure(alphabet: List[str], order: int, verbose: bool = True) -> None:
    """
    Analyze and display constraint structure for debugging.
    
    Parameters
    ----------
    alphabet : List[str]
        Symbol alphabet
    order : int
        Markov order  
    verbose : bool
        Whether to print detailed analysis
    """
    validation = validate_constraint_setup(alphabet, order)
    
    if not validation['valid']:
        print(f"❌ Invalid constraint setup: {validation['error']}")
        return
    
    forbidden, required = get_constraint_positions_higher_order(alphabet, order)
    
    if verbose:
        print(f"Constraint Analysis for {order}-order Markov model")
        print(f"Alphabet: {alphabet} (size: {validation['alphabet_size']})")
        print("=" * 60)
        print(f"State space size: {validation['state_space_size']}")
        print(f"Number of contexts: {validation['n_contexts']}")
        print(f"Forbidden transitions: {validation['forbidden_positions']}")
        print(f"Required transitions: {validation['required_positions']}")
        print(f"Constraint coverage: {validation['constraint_ratio']:.2%}")
        print(f"Valid contexts: {validation['valid_contexts']} / {validation['n_contexts']}")
        print(f"Memory per sequence: {validation['memory_per_sequence_bytes'] / 1024:.1f} KB")
        
        # Show example contexts and their constraints
        print(f"\nExample contexts (showing first 10):")
        all_contexts = generate_all_contexts(alphabet, order)
        for i, context in enumerate(all_contexts[:10]):
            context_idx = encode_context(context, alphabet, order)
            is_valid = is_valid_context(context, alphabet)
            base_pos = context_idx * len(alphabet)
            
            # Count constraints for this context
            context_forbidden = [pos for pos in forbidden 
                               if base_pos <= pos < base_pos + len(alphabet)]
            context_required = [pos for pos in required 
                              if base_pos <= pos < base_pos + len(alphabet)]
            
            print(f"  {context} (idx={context_idx:2d}): "
                  f"{'✓' if is_valid else '✗'} "
                  f"({len(context_forbidden)} forbidden, {len(context_required)} required)")


class ConstraintManager:
    """Manager class for handling constraints in higher-order Markov models."""
    
    def __init__(self, alphabet: List[str], order: int):
        """
        Initialize constraint manager.
        
        Parameters
        ----------
        alphabet : List[str]
            Symbol alphabet with format ['<', phrase1, ..., phraseN, '>']
        order : int
            Markov order (1, 2, 3, ...)
        """
        self.alphabet = alphabet
        self.order = order
        self.n_symbols = len(alphabet)
        self.state_space_size = self.n_symbols ** (order + 1)
        
        # Validate setup
        validation = validate_constraint_setup(alphabet, order)
        if not validation['valid']:
            raise ValueError(f"Invalid constraint setup: {validation['error']}")
        
        # Generate constraints
        self.forbidden_positions, self.required_positions = \
            get_constraint_positions_higher_order(alphabet, order)
        
        # Store validation info
        self.validation_info = validation
        
        # Warn about large state spaces
        if self.state_space_size > 10000:
            warnings.warn(f"Large state space ({self.state_space_size} transitions) "
                         f"may cause memory issues")
    
    def apply_constraints(self, x: np.ndarray) -> np.ndarray:
        """Apply constraints to logit vector."""
        return apply_constraints_to_logits(x, self.forbidden_positions, self.required_positions)
    
    def encode_context(self, context_symbols: List[str]) -> int:
        """Encode context sequence to index."""
        return encode_context(context_symbols, self.alphabet, self.order)
    
    def decode_context(self, context_idx: int) -> List[str]:
        """Decode context index to symbol sequence."""
        return decode_context(context_idx, self.alphabet, self.order)
    
    def is_valid_context(self, context_symbols: List[str]) -> bool:
        """Check if context is valid."""
        return is_valid_context(context_symbols, self.alphabet)
    
    def get_transition_position(self, context_symbols: List[str], target_symbol: str) -> int:
        """Get position in state vector for specific transition."""
        context_idx = self.encode_context(context_symbols)
        if target_symbol not in self.alphabet:
            raise ValueError(f"Target symbol '{target_symbol}' not in alphabet")
        target_idx = self.alphabet.index(target_symbol)
        return context_idx * self.n_symbols + target_idx
    
    def analyze(self, verbose: bool = True) -> Dict[str, any]:
        """Analyze constraint structure."""
        if verbose:
            analyze_constraint_structure(self.alphabet, self.order, verbose=True)
        return self.validation_info
    
    def __repr__(self) -> str:
        return (f"ConstraintManager(alphabet={self.alphabet}, order={self.order}, "
                f"state_space={self.state_space_size})")


def create_constraint_manager(alphabet_size: int, order: int = 1) -> ConstraintManager:
    """
    Convenience function to create constraint manager with standard alphabet.
    
    Parameters
    ----------
    alphabet_size : int
        Total size of alphabet (including start/end tokens)
        Minimum: 3 (start + 1 phrase + end)
    order : int
        Markov order
        
    Returns
    -------
    ConstraintManager
        Configured constraint manager
    """
    if alphabet_size < 3:
        raise ValueError("Alphabet size must be at least 3")
    
    # Generate standard alphabet: ['<', 'a', 'b', ..., '>']
    n_phrases = alphabet_size - 2
    if n_phrases <= 26:
        # Use letters for small alphabets
        phrase_symbols = [chr(ord('a') + i) for i in range(n_phrases)]
    else:
        # Use numbered symbols for large alphabets
        phrase_symbols = [f'p{i}' for i in range(n_phrases)]
    
    alphabet = ['<'] + phrase_symbols + ['>']
    
    return ConstraintManager(alphabet, order) 
