"""Higher-order Markov state space management for adaptive syntax filtering.

Provides utilities for managing state spaces in higher-order Markov models
where the state vector represents logit parameters for symbol transitions
with varying context lengths.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class StateSpaceConfig:
    """Configuration for higher-order Markov state space.
    
    Attributes
    ----------
    alphabet_size : int
        Number of symbols R in alphabet
    markov_order : int
        Order of Markov model (1 = first-order, 2 = second-order, etc.)
    state_dim : int
        Total dimension of state vector (R^{order+1})
    n_contexts : int
        Number of possible contexts (R^order)
    block_size : int
        Size of each transition block (R)
    """
    alphabet_size: int
    markov_order: int
    state_dim: int
    n_contexts: int
    block_size: int


class StateSpaceManager:
    """Manager for higher-order Markov state spaces in adaptive syntax filtering.
    
    Handles the encoding and organization of state vectors for different Markov orders,
    where the state represents logit parameters for context-dependent transitions.
    
    For a Markov model of order k with alphabet size R:
    - Context length: k symbols  
    - Number of contexts: R^k
    - State dimension: R^{k+1} (each context has R possible transitions)
    - Block structure: R^k blocks of size R each
    
    Parameters
    ----------
    alphabet_size : int
        Number of symbols R in the alphabet
    markov_order : int, default=1
        Order of the Markov model (1, 2, 3, ...)
    alphabet : Optional[List[str]]
        Optional list of symbol names for display purposes
        
    Examples
    --------
    >>> # First-order model with 3 symbols
    >>> mgr = StateSpaceManager(alphabet_size=3, markov_order=1)
    >>> mgr.state_dim  # 3^2 = 9 transitions
    9
    
    >>> # Second-order model with 3 symbols  
    >>> mgr = StateSpaceManager(alphabet_size=3, markov_order=2)
    >>> mgr.state_dim  # 3^3 = 27 transitions
    27
    """
    
    def __init__(self, 
                 alphabet_size: int, 
                 markov_order: int = 1,
                 alphabet: Optional[List[str]] = None):
        
        if alphabet_size < 2:
            raise ValueError("Alphabet size must be at least 2")
        if markov_order < 1:
            raise ValueError("Markov order must be at least 1")
        if markov_order > 5:
            warnings.warn(f"Markov order {markov_order} is very high, consider computational cost")
        
        self.alphabet_size = alphabet_size
        self.markov_order = markov_order
        self.alphabet = alphabet or [f"s{i}" for i in range(alphabet_size)]
        
        # Validate alphabet size matches provided alphabet
        if len(self.alphabet) != alphabet_size:
            raise ValueError(f"Alphabet length {len(self.alphabet)} != alphabet_size {alphabet_size}")
        
        # Calculate dimensions
        self.n_contexts = alphabet_size ** markov_order
        self.block_size = alphabet_size
        self.state_dim = alphabet_size ** (markov_order + 1)
        
        # Create configuration object
        self.config = StateSpaceConfig(
            alphabet_size=alphabet_size,
            markov_order=markov_order,
            state_dim=self.state_dim,
            n_contexts=self.n_contexts,
            block_size=self.block_size
        )
        
        # Warn about large state spaces
        if self.state_dim > 10000:
            warnings.warn(f"Large state space ({self.state_dim} dimensions) may cause memory issues")
    
    def encode_context(self, context_symbols: List[Union[str, int]]) -> int:
        """Encode a sequence of symbols into a context index.
        
        Converts a context (sequence of symbols) into a single integer index
        for addressing the appropriate block in the state vector.
        
        Parameters
        ----------
        context_symbols : List[Union[str, int]]
            Sequence of symbols forming the context. Length must equal markov_order.
            Can be symbol names (strings) or indices (integers).
            
        Returns
        -------
        int
            Context index in range [0, n_contexts-1]
            
        Examples
        --------
        >>> mgr = StateSpaceManager(3, markov_order=2, alphabet=['a', 'b', 'c'])
        >>> mgr.encode_context(['a', 'b'])  # Context 'ab'
        1
        >>> mgr.encode_context([0, 1])     # Same as above using indices
        1
        """
        if len(context_symbols) != self.markov_order:
            raise ValueError(f"Context length {len(context_symbols)} != markov_order {self.markov_order}")
        
        context_idx = 0
        
        for i, symbol in enumerate(context_symbols):
            # Convert symbol to index if needed
            if isinstance(symbol, str):
                try:
                    symbol_idx = self.alphabet.index(symbol)
                except ValueError:
                    raise ValueError(f"Unknown symbol '{symbol}' in alphabet {self.alphabet}")
            else:
                symbol_idx = int(symbol)
                if not (0 <= symbol_idx < self.alphabet_size):
                    raise ValueError(f"Symbol index {symbol_idx} out of range [0, {self.alphabet_size})")
            
            # Compute positional contribution to context index
            power = self.markov_order - 1 - i
            context_idx += symbol_idx * (self.alphabet_size ** power)
        
        return context_idx
    
    def decode_context(self, context_idx: int) -> List[str]:
        """Decode a context index back to a sequence of symbols.
        
        Parameters
        ----------
        context_idx : int
            Context index in range [0, n_contexts-1]
            
        Returns
        -------
        List[str]
            Sequence of symbol names forming the context
            
        Examples
        --------
        >>> mgr = StateSpaceManager(3, markov_order=2, alphabet=['a', 'b', 'c'])
        >>> mgr.decode_context(1)
        ['a', 'b']
        """
        if not (0 <= context_idx < self.n_contexts):
            raise ValueError(f"Context index {context_idx} out of range [0, {self.n_contexts})")
        
        context_symbols = []
        
        for i in range(self.markov_order):
            power = self.markov_order - 1 - i
            symbol_idx = (context_idx // (self.alphabet_size ** power)) % self.alphabet_size
            context_symbols.append(self.alphabet[symbol_idx])
        
        return context_symbols
    
    def get_transition_position(self, 
                              context: Union[List[Union[str, int]], int], 
                              target_symbol: Union[str, int]) -> int:
        """Get position in state vector for a specific transition.
        
        Parameters
        ----------
        context : Union[List[Union[str, int]], int]
            Either a context sequence or context index
        target_symbol : Union[str, int]
            Target symbol (name or index) for the transition
            
        Returns
        -------
        int
            Position in state vector for this transition
            
        Examples
        --------
        >>> mgr = StateSpaceManager(3, markov_order=2)
        >>> mgr.get_transition_position([0, 1], 2)  # Context [0,1] -> symbol 2
        5
        """
        # Encode context if needed
        if isinstance(context, list):
            context_idx = self.encode_context(context)
        else:
            context_idx = int(context)
            if not (0 <= context_idx < self.n_contexts):
                raise ValueError(f"Context index {context_idx} out of range")
        
        # Convert target symbol to index if needed
        if isinstance(target_symbol, str):
            try:
                target_idx = self.alphabet.index(target_symbol)
            except ValueError:
                raise ValueError(f"Unknown target symbol '{target_symbol}'")
        else:
            target_idx = int(target_symbol)
            if not (0 <= target_idx < self.alphabet_size):
                raise ValueError(f"Target index {target_idx} out of range")
        
        # Calculate position: context_idx * block_size + target_idx
        return context_idx * self.block_size + target_idx
    
    def get_block_indices(self, context: Union[List[Union[str, int]], int]) -> Tuple[int, int]:
        """Get start and end indices for a context block in the state vector.
        
        Parameters
        ----------
        context : Union[List[Union[str, int]], int]
            Either a context sequence or context index
            
        Returns
        -------
        Tuple[int, int]
            (start_idx, end_idx) where end_idx is exclusive
            
        Examples
        --------
        >>> mgr = StateSpaceManager(3, markov_order=2)
        >>> mgr.get_block_indices([0, 1])  # Context [0,1]
        (3, 6)
        """
        # Encode context if needed
        if isinstance(context, list):
            context_idx = self.encode_context(context)
        else:
            context_idx = int(context)
        
        start_idx = context_idx * self.block_size
        end_idx = (context_idx + 1) * self.block_size
        
        return start_idx, end_idx
    
    def extract_block(self, state_vector: np.ndarray, 
                     context: Union[List[Union[str, int]], int]) -> np.ndarray:
        """Extract the block corresponding to a specific context.
        
        Parameters
        ----------
        state_vector : np.ndarray, shape (state_dim,)
            Full state vector with logit parameters
        context : Union[List[Union[str, int]], int]
            Either a context sequence or context index
            
        Returns
        -------
        np.ndarray, shape (block_size,)
            Block of logit parameters for transitions from this context
        """
        if state_vector.shape[0] != self.state_dim:
            raise ValueError(f"State vector shape {state_vector.shape} != expected ({self.state_dim},)")
        
        start_idx, end_idx = self.get_block_indices(context)
        return state_vector[start_idx:end_idx].copy()
    
    def set_block(self, state_vector: np.ndarray, 
                  context: Union[List[Union[str, int]], int],
                  block_values: np.ndarray) -> None:
        """Set the block corresponding to a specific context.
        
        Parameters
        ----------
        state_vector : np.ndarray, shape (state_dim,)
            Full state vector to modify in-place
        context : Union[List[Union[str, int]], int]
            Either a context sequence or context index  
        block_values : np.ndarray, shape (block_size,)
            New values for this context block
        """
        if state_vector.shape[0] != self.state_dim:
            raise ValueError(f"State vector shape {state_vector.shape} != expected ({self.state_dim},)")
        if block_values.shape[0] != self.block_size:
            raise ValueError(f"Block values shape {block_values.shape} != expected ({self.block_size},)")
        
        start_idx, end_idx = self.get_block_indices(context)
        state_vector[start_idx:end_idx] = block_values
    
    def generate_all_contexts(self) -> List[List[str]]:
        """Generate all possible context sequences.
        
        Returns
        -------
        List[List[str]]
            List of all possible context sequences of length markov_order
        """
        contexts = []
        
        for context_idx in range(self.n_contexts):
            context_symbols = self.decode_context(context_idx)
            contexts.append(context_symbols)
        
        return contexts
    
    def create_block_diagonal_structure(self, block_matrices: List[np.ndarray]) -> np.ndarray:
        """Create block-diagonal matrix from individual context blocks.
        
        Used for constructing state transition matrices F where each context
        has its own dynamics block.
        
        Parameters
        ----------
        block_matrices : List[np.ndarray]
            List of matrices, one for each context. Each should be (block_size, block_size).
            
        Returns
        -------
        np.ndarray, shape (state_dim, state_dim)
            Block-diagonal matrix with the provided blocks
            
        Examples
        --------
        >>> mgr = StateSpaceManager(2, markov_order=1)
        >>> blocks = [np.eye(2), 0.5 * np.eye(2)]  # Two 2x2 blocks
        >>> F = mgr.create_block_diagonal_structure(blocks)
        >>> F.shape
        (4, 4)
        """
        if len(block_matrices) != self.n_contexts:
            raise ValueError(f"Need {self.n_contexts} blocks, got {len(block_matrices)}")
        
        # Initialize block-diagonal matrix
        F = np.zeros((self.state_dim, self.state_dim))
        
        for context_idx, block_matrix in enumerate(block_matrices):
            if block_matrix.shape != (self.block_size, self.block_size):
                raise ValueError(f"Block {context_idx} has shape {block_matrix.shape}, "
                               f"expected ({self.block_size}, {self.block_size})")
            
            start_idx, end_idx = self.get_block_indices(context_idx)
            F[start_idx:end_idx, start_idx:end_idx] = block_matrix
        
        return F
    
    def extract_block_diagonal_blocks(self, matrix: np.ndarray) -> List[np.ndarray]:
        """Extract individual blocks from a block-diagonal matrix.
        
        Parameters
        ----------
        matrix : np.ndarray, shape (state_dim, state_dim)
            Block-diagonal matrix
            
        Returns
        -------
        List[np.ndarray]
            List of extracted blocks, each of shape (block_size, block_size)
        """
        if matrix.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Matrix shape {matrix.shape} != expected ({self.state_dim}, {self.state_dim})")
        
        blocks = []
        
        for context_idx in range(self.n_contexts):
            start_idx, end_idx = self.get_block_indices(context_idx)
            block = matrix[start_idx:end_idx, start_idx:end_idx].copy()
            blocks.append(block)
        
        return blocks
    
    def validate_state_vector(self, state_vector: np.ndarray, 
                            check_probabilities: bool = True) -> bool:
        """Validate a state vector and optionally check probability normalization.
        
        Parameters
        ----------
        state_vector : np.ndarray
            State vector to validate
        check_probabilities : bool, default=True
            Whether to check that softmax of each block sums to 1
            
        Returns
        -------
        bool
            True if valid
            
        Raises
        ------
        ValueError
            If state vector is invalid
        AssertionError
            If probabilities don't sum to 1 (when check_probabilities=True)
        """
        if state_vector.shape[0] != self.state_dim:
            raise ValueError(f"State vector shape {state_vector.shape} != expected ({self.state_dim},)")
        
        if not np.all(np.isfinite(state_vector)):
            raise ValueError("State vector contains non-finite values")
        
        if check_probabilities:
            from .observation_model import softmax_observation_model, validate_transition_probabilities
            probs = softmax_observation_model(state_vector, self.alphabet_size)
            validate_transition_probabilities(probs, self.alphabet_size)
        
        return True
    
    def get_memory_estimate(self) -> Dict[str, float]:
        """Estimate memory usage for this state space configuration.
        
        Returns
        -------
        Dict[str, float]
            Memory estimates in MB for different components
        """
        # Estimate memory usage (8 bytes per float64)
        state_vector_mb = self.state_dim * 8 / (1024 * 1024)
        covariance_mb = self.state_dim * self.state_dim * 8 / (1024 * 1024)
        transition_matrix_mb = covariance_mb  # F has same size as covariance
        
        return {
            'state_vector_mb': state_vector_mb,
            'covariance_matrix_mb': covariance_mb,
            'transition_matrix_mb': transition_matrix_mb,
            'total_per_sequence_mb': state_vector_mb + covariance_mb,
            'total_with_parameters_mb': state_vector_mb + 2 * covariance_mb
        }
    
    def __repr__(self) -> str:
        """String representation of the state space manager."""
        return (f"StateSpaceManager(alphabet_size={self.alphabet_size}, "
                f"markov_order={self.markov_order}, state_dim={self.state_dim})")
    
    def summary(self) -> str:
        """Detailed summary of the state space configuration."""
        memory_est = self.get_memory_estimate()
        
        summary_lines = [
            f"State Space Configuration:",
            f"  Alphabet size: {self.alphabet_size}",
            f"  Markov order: {self.markov_order}",
            f"  Number of contexts: {self.n_contexts}",
            f"  Block size: {self.block_size}",
            f"  Total state dimension: {self.state_dim}",
            f"",
            f"Memory estimates:",
            f"  State vector: {memory_est['state_vector_mb']:.2f} MB",
            f"  Covariance matrix: {memory_est['covariance_matrix_mb']:.2f} MB", 
            f"  Total per sequence: {memory_est['total_per_sequence_mb']:.2f} MB",
            f"",
            f"Context examples:",
        ]
        
        # Show first few contexts as examples
        contexts = self.generate_all_contexts()
        for i, context in enumerate(contexts[:min(5, len(contexts))]):
            summary_lines.append(f"  Context {i}: {' -> '.join(context)}")
        
        if len(contexts) > 5:
            summary_lines.append(f"  ... and {len(contexts) - 5} more contexts")
        
        return "\n".join(summary_lines) 
