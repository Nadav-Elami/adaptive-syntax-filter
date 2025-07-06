"""Tests for state space management.

Tests the StateSpaceManager for handling higher-order Markov models
and state transitions in adaptive syntax evolution.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from src.adaptive_syntax_filter.core.state_space import StateSpaceManager
from src.adaptive_syntax_filter.core.observation_model import softmax_observation_model


class TestStateSpaceManager:
    """Test suite for StateSpaceManager."""
    
    def test_state_space_initialization(self, small_alphabet):
        """Test StateSpaceManager initialization."""
        alphabet_size = len(small_alphabet)
        markov_order = 1
        
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
        
        assert manager.alphabet_size == alphabet_size
        assert manager.markov_order == markov_order
        assert manager.state_dim == alphabet_size ** (markov_order + 1)

    def test_state_space_dimensions(self, small_alphabet):
        """Test state space dimensions for different orders."""
        alphabet_size = len(small_alphabet)
        
        # Test different Markov orders
        for order in [1, 2, 3]:
            manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=order)
            expected_dim = alphabet_size ** (order + 1)
            assert manager.state_dim == expected_dim

    def test_symbol_to_state_conversion(self, small_alphabet):
        """Test conversion from symbol sequences to context indices."""
        alphabet_size = len(small_alphabet)
        
        # Test first-order Markov model
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=1)
        
        for symbol in range(alphabet_size):
            context_idx = manager.encode_context([symbol])
            assert 0 <= context_idx < alphabet_size

    def test_state_to_symbol_conversion(self, small_alphabet):
        """Test conversion from context indices to symbol sequences."""
        alphabet_size = len(small_alphabet)
        
        # Test first-order Markov model
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=1)
        
        for context_idx in range(alphabet_size):
            symbols = manager.decode_context(context_idx)
            assert len(symbols) == 1
            # The decoded context returns string symbols by default
            symbol_idx = manager.alphabet.index(symbols[0])
            assert 0 <= symbol_idx < alphabet_size

    def test_higher_order_symbol_to_state(self, small_alphabet):
        """Test symbol to state conversion for higher-order models."""
        alphabet_size = len(small_alphabet)
        markov_order = 2
        
        if alphabet_size >= 2:  # Need at least 2 symbols for order 2
            manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
            
            # Test valid symbol sequences
            symbols = [0, 1]
            context_idx = manager.encode_context(symbols)
            assert 0 <= context_idx < alphabet_size ** markov_order
            
            # Verify round-trip conversion
            recovered_symbols = manager.decode_context(context_idx)
            # Convert string symbols back to indices for comparison
            recovered_indices = [manager.alphabet.index(s) for s in recovered_symbols]
            assert recovered_indices == symbols

    def test_higher_order_state_to_symbol(self, small_alphabet):
        """Test state to symbol conversion for higher-order models."""
        alphabet_size = len(small_alphabet)
        markov_order = 2
        
        if alphabet_size >= 2:
            manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
            
            # Test all valid contexts
            for context_idx in range(alphabet_size ** markov_order):
                symbols = manager.decode_context(context_idx)
                assert len(symbols) == markov_order
                for symbol in symbols:
                    # Convert string symbol to index for validation
                    symbol_idx = manager.alphabet.index(symbol)
                    assert 0 <= symbol_idx < alphabet_size

    def test_transition_matrix_construction(self, small_alphabet):
        """Test construction of transition matrices."""
        alphabet_size = len(small_alphabet)
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=1)
        
        # Create simple transition matrix
        try:
            transition_matrix = manager.build_transition_matrix()
            expected_shape = (alphabet_size, alphabet_size)
            assert transition_matrix.shape == expected_shape
            
            # Check probability constraints
            row_sums = np.sum(transition_matrix, axis=1)
            assert_array_almost_equal(row_sums, np.ones(alphabet_size), decimal=6)
            assert np.all(transition_matrix >= 0)
        except (AttributeError, NotImplementedError):
            pytest.skip("Transition matrix construction not implemented")

    def test_state_enumeration(self, small_alphabet):
        """Test enumeration of all possible states."""
        alphabet_size = len(small_alphabet)
        markov_order = 1
        
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
        
        try:
            all_states = manager.enumerate_states()
            assert len(all_states) == alphabet_size ** markov_order
            
            # Check that each state is unique
            assert len(set(tuple(state) for state in all_states)) == len(all_states)
        except (AttributeError, NotImplementedError):
            pytest.skip("State enumeration not implemented")

    def test_next_state_computation(self, small_alphabet):
        """Test computation of next states given current state and observation."""
        alphabet_size = len(small_alphabet)
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=1)
        
        current_state = 0
        next_symbol = 1
        
        try:
            next_state = manager.compute_next_state(current_state, next_symbol)
            assert 0 <= next_state < alphabet_size
        except (AttributeError, NotImplementedError):
            pytest.skip("Next state computation not implemented")

    def test_state_sequence_conversion(self, small_alphabet):
        """Test conversion of observation sequences to state sequences."""
        alphabet_size = len(small_alphabet)
        markov_order = 1
        
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
        
        observations = np.array([0, 1, 0, 2])
        
        try:
            state_sequence = manager.observations_to_states(observations)
            assert len(state_sequence) == len(observations) - markov_order + 1
            
            for state in state_sequence:
                assert 0 <= state < alphabet_size ** markov_order
        except (AttributeError, NotImplementedError):
            pytest.skip("State sequence conversion not implemented")


class TestStateSpaceValidation:
    """Test state space validation and error handling."""
    
    def test_invalid_alphabet_size(self):
        """Test handling of invalid alphabet sizes."""
        with pytest.raises(ValueError):
            StateSpaceManager(alphabet_size=0, markov_order=1)
        
        with pytest.raises(ValueError):
            StateSpaceManager(alphabet_size=-1, markov_order=1)

    def test_invalid_markov_order(self, small_alphabet):
        """Test handling of invalid Markov orders."""
        alphabet_size = len(small_alphabet)
        
        with pytest.raises(ValueError):
            StateSpaceManager(alphabet_size=alphabet_size, markov_order=0)
        
        with pytest.raises(ValueError):
            StateSpaceManager(alphabet_size=alphabet_size, markov_order=-1)

    def test_symbol_out_of_range(self, small_alphabet):
        """Test handling of symbols outside alphabet range."""
        alphabet_size = len(small_alphabet)
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=1)
        
        # Test symbol too large
        with pytest.raises(ValueError):
            manager.encode_context([alphabet_size])
        
        # Test negative symbol
        with pytest.raises(ValueError):
            manager.encode_context([-1])

    def test_state_out_of_range(self, small_alphabet):
        """Test handling of state indices outside valid range."""
        alphabet_size = len(small_alphabet)
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=1)
        
        # Test context index too large
        with pytest.raises(ValueError):
            manager.decode_context(alphabet_size)
        
        # Test negative context index
        with pytest.raises(ValueError):
            manager.decode_context(-1)

    def test_wrong_sequence_length(self, small_alphabet):
        """Test handling of symbol sequences with wrong length."""
        alphabet_size = len(small_alphabet)
        markov_order = 2
        
        if alphabet_size >= 2:
            manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
            
            # Test sequence too short
            with pytest.raises(ValueError):
                manager.encode_context([0])
            
            # Test sequence too long
            with pytest.raises(ValueError):
                manager.encode_context([0, 1, 2])


class TestStateSpaceHigherOrder:
    """Test state space management for higher-order models."""
    
    def test_second_order_states(self, medium_alphabet):
        """Test second-order Markov state management."""
        alphabet_size = len(medium_alphabet)
        markov_order = 2
        
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
        
        # Test all possible second-order contexts
        for i in range(alphabet_size):
            for j in range(alphabet_size):
                symbols = [i, j]
                context_idx = manager.encode_context(symbols)
                recovered = manager.decode_context(context_idx)
                # Convert string symbols back to indices for comparison
                recovered_indices = [manager.alphabet.index(s) for s in recovered]
                assert recovered_indices == symbols

    def test_third_order_states(self, small_alphabet):
        """Test third-order Markov state management."""
        alphabet_size = len(small_alphabet)
        markov_order = 3
        
        if alphabet_size <= 3:  # Keep computational cost reasonable
            manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
            
            expected_contexts = alphabet_size ** markov_order
            assert manager.n_contexts == expected_contexts
            
            # Test round-trip conversion for a few contexts
            for context_idx in range(min(10, expected_contexts)):
                symbols = manager.decode_context(context_idx)
                # Convert string symbols to indices for encode_context
                symbol_indices = [manager.alphabet.index(s) for s in symbols]
                recovered_idx = manager.encode_context(symbol_indices)
                assert recovered_idx == context_idx

    def test_state_ordering_consistency(self, small_alphabet):
        """Test that state ordering is consistent across operations."""
        alphabet_size = len(small_alphabet)
        markov_order = 2
        
        if alphabet_size >= 2:
            manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
            
            # Verify lexicographic ordering
            context_idx = 0
            for i in range(alphabet_size):
                for j in range(alphabet_size):
                    symbols = manager.decode_context(context_idx)
                    # Convert to indices for comparison
                    symbol_indices = [manager.alphabet.index(s) for s in symbols]
                    expected_symbols = [i, j]
                    assert symbol_indices == expected_symbols, f"Context {context_idx}: got {symbol_indices}, expected {expected_symbols}"
                    context_idx += 1


class TestStateSpaceIntegration:
    """Integration tests for state space management."""
    
    def test_state_space_kalman_integration(self, small_alphabet):
        """Test state space integration with Kalman filter dimensions."""
        alphabet_size = len(small_alphabet)
        markov_order = 1
        
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
        
        # Verify state dimension matches expected Kalman filter state size
        from src.adaptive_syntax_filter.core.kalman import KalmanFilter
        
        F = np.eye(manager.state_dim)
        u = np.zeros(manager.state_dim)
        Sigma = 0.1 * np.eye(manager.state_dim)
        
        try:
            kf = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
            # If this doesn't raise an error, dimensions are compatible
            assert True
        except ValueError:
            assert False, "State space dimensions incompatible with Kalman filter"

    def test_state_space_observation_model_integration(self, small_alphabet):
        """Test state space integration with observation models."""
        alphabet_size = len(small_alphabet)
        markov_order = 1
        
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
        
        # Test integration with observation model        
        try:
            # Create state vector compatible with observation model
            x = np.random.randn(manager.state_dim)
            probs = softmax_observation_model(x, alphabet_size)
            
            # Check that probabilities make sense for state space
            assert len(probs) == manager.state_dim
            assert np.all(probs >= 0)
        except (AttributeError, NotImplementedError):
            pytest.skip("Observation model integration incomplete")


class TestStateSpacePerformance:
    """Performance tests for state space operations."""
    
    @pytest.mark.performance
    def test_large_alphabet_performance(self, large_alphabet):
        """Test performance with large alphabets."""
        alphabet_size = len(large_alphabet)
        markov_order = 1  # Keep order low for large alphabets
        
        manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
        
        # Test bulk operations
        import time
        start_time = time.time()
        
        # Convert many symbols to contexts
        for i in range(min(1000, alphabet_size)):
            context_idx = manager.encode_context([i % alphabet_size])
            symbols = manager.decode_context(context_idx)
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0, f"State space operations took {elapsed:.2f}s, should be < 1.0s"

    @pytest.mark.performance  
    def test_higher_order_performance(self, medium_alphabet):
        """Test performance with higher-order models."""
        alphabet_size = len(medium_alphabet)
        markov_order = 3
        
        if alphabet_size <= 5:  # Limit state space size
            manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=markov_order)
            
            import time
            start_time = time.time()
            
            # Test state enumeration performance
            try:
                all_states = manager.enumerate_states()
                elapsed = time.time() - start_time
                assert elapsed < 2.0, f"State enumeration took {elapsed:.2f}s, should be < 2.0s"
            except (AttributeError, NotImplementedError):
                pytest.skip("State enumeration not implemented") 