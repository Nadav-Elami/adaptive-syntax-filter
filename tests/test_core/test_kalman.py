"""Tests for Kalman filtering functionality.

Tests the core Kalman filter implementation for adaptive syntax evolution,
including forward filtering, RTS smoothing, and numerical stability.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import time

from adaptive_syntax_filter.core import KalmanFilter, KalmanState


class TestKalmanFilter:
    """Test suite for KalmanFilter."""
    
    def test_filter_initialization(self, small_alphabet):
        """Test KalmanFilter initialization."""
        alphabet_size = len(small_alphabet)
        state_dim = alphabet_size ** 2
        
        F = 0.9 * np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = 0.1 * np.eye(state_dim)
        
        kf = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        assert kf.R == alphabet_size
        assert kf.state_dim == state_dim
        assert np.array_equal(kf.F, F)
        assert np.array_equal(kf.u, u)
        assert np.array_equal(kf.Sigma, Sigma)
    
    def test_filter_invalid_dimensions(self, small_alphabet):
        """Test KalmanFilter with invalid dimensions."""
        alphabet_size = len(small_alphabet)
        state_dim = alphabet_size ** 2
        
        # Wrong F dimensions
        with pytest.raises(ValueError):
            F = np.eye(state_dim + 1)
            u = np.zeros(state_dim)
            Sigma = np.eye(state_dim)
            KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        # Wrong u dimensions
        with pytest.raises(ValueError):
            F = np.eye(state_dim)
            u = np.zeros(state_dim + 1)
            Sigma = np.eye(state_dim)
            KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
    
    def test_forward_filter_basic(self, small_alphabet):
        """Test basic forward filtering."""
        alphabet_size = len(small_alphabet)
        state_dim = alphabet_size ** 2
        
        # Create simple system
        F = 0.95 * np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = 0.05 * np.eye(state_dim)
        
        kf = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        # Create simple observations (transition pairs)
        observations = [
            np.array([0, 1, 2, 0]),  # Simple sequence of symbol indices
            np.array([1, 2, 0, 1])   # Another sequence
        ]
        
        x0 = np.zeros(state_dim)
        
        # Test forward filtering
        states = kf.forward_filter(observations, x0)
        
        assert len(states) == len(observations)
        for state in states:
            assert isinstance(state, KalmanState)
            assert state.x_filtered.shape == (state_dim,)
            assert state.W_filtered.shape == (state_dim, state_dim)
            assert state.x_predicted.shape == (state_dim,)
            assert state.W_predicted.shape == (state_dim, state_dim)

    def test_rts_smoother_basic(self, small_alphabet):
        """Test basic RTS smoothing."""
        alphabet_size = len(small_alphabet)
        state_dim = alphabet_size ** 2
        
        # Create simple system
        F = 0.95 * np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = 0.05 * np.eye(state_dim)
        
        kf = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        # Create simple observations
        observations = [
            np.array([0, 1, 2]),
            np.array([1, 2, 0])
        ]
        
        x0 = np.zeros(state_dim)
        
        # First do forward filtering
        forward_states = kf.forward_filter(observations, x0)
        
        # Then test backward smoothing
        smoothed_states = kf.rts_smoother()
        
        assert len(smoothed_states) == len(observations)
        for state in smoothed_states:
            assert isinstance(state, KalmanState)
            assert state.x_smoothed is not None
            assert state.W_smoothed is not None
            assert state.x_smoothed.shape == (state_dim,)
            assert state.W_smoothed.shape == (state_dim, state_dim)

    def test_filter_numerical_stability(self, small_alphabet):
        """Test numerical stability with extreme parameters."""
        alphabet_size = len(small_alphabet)
        state_dim = alphabet_size ** 2
        
        # Test with very small noise
        F = 0.99 * np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = 1e-8 * np.eye(state_dim)
        
        kf = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        # Create simple observation
        observations = [np.array([0, 1])]
        x0 = np.zeros(state_dim)
        
        # Should not raise numerical errors
        states = kf.forward_filter(observations, x0)
        assert len(states) == 1

    def test_filter_reproducibility(self, small_alphabet, global_test_seed):
        """Test that filtering is reproducible."""
        np.random.seed(global_test_seed)
        
        alphabet_size = len(small_alphabet)
        state_dim = alphabet_size ** 2
        
        F = 0.9 * np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = 0.1 * np.eye(state_dim)
        
        observations = [np.array([0, 1, 2])]
        x0 = np.zeros(state_dim)
        
        # Run twice with same setup
        kf1 = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        kf2 = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        states1 = kf1.forward_filter(observations, x0)
        states2 = kf2.forward_filter(observations, x0)
        
        # Results should be identical
        for s1, s2 in zip(states1, states2):
            assert_array_almost_equal(s1.x_filtered, s2.x_filtered)
            assert_array_almost_equal(s1.x_predicted, s2.x_predicted)


@pytest.mark.performance
class TestKalmanFilterPerformance:
    """Performance tests for KalmanFilter."""
    
    @pytest.mark.performance
    def test_filter_large_dataset(self, medium_alphabet, performance_test_config):
        """Test filtering performance with large dataset."""
        alphabet_size = len(medium_alphabet)
        state_dim = alphabet_size ** 2
        
        F = 0.9 * np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = 0.1 * np.eye(state_dim)
        
        kf = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        # Create large dataset
        n_sequences = performance_test_config['batch_size']
        observations = []
        
        for _ in range(n_sequences):
            # Simple sequences
            obs = np.array([0, 1, 2])
            observations.append(obs)
        
        x0 = np.zeros(state_dim)
        
        # Measure performance
        start_time = time.time()
        states = kf.forward_filter(observations, x0)
        end_time = time.time()
        
        execution_time = end_time - start_time
        time_limit = performance_test_config['time_limit_seconds']
        
        assert execution_time < time_limit, f"Filtering took {execution_time:.2f}s, limit is {time_limit}s"
        assert len(states) == n_sequences


class TestKalmanFilterEdgeCases:
    """Edge case tests for KalmanFilter."""
    
    def test_empty_observations(self, small_alphabet):
        """Test with empty observation sequences."""
        alphabet_size = len(small_alphabet)
        state_dim = alphabet_size ** 2
        
        F = 0.9 * np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = 0.1 * np.eye(state_dim)
        
        kf = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        # Empty observations
        observations = []
        x0 = np.zeros(state_dim)
        
        states = kf.forward_filter(observations, x0)
        assert len(states) == 0

    def test_single_observation(self, small_alphabet):
        """Test with single observation sequence."""
        alphabet_size = len(small_alphabet)
        state_dim = alphabet_size ** 2
        
        F = 0.9 * np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = 0.1 * np.eye(state_dim)
        
        kf = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        # Single sequence
        observations = [np.array([0, 1])]
        x0 = np.zeros(state_dim)
        
        states = kf.forward_filter(observations, x0)
        assert len(states) == 1
        assert isinstance(states[0], KalmanState)

    def test_minimal_alphabet(self):
        """Test with minimal alphabet size."""
        alphabet_size = 2  # Just 2 symbols
        state_dim = alphabet_size ** 2
        
        F = 0.9 * np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = 0.1 * np.eye(state_dim)
        
        kf = KalmanFilter(alphabet_size=alphabet_size, F=F, u=u, Sigma=Sigma)
        
        observations = [np.array([0, 1, 0])]
        x0 = np.zeros(state_dim)
        
        states = kf.forward_filter(observations, x0)
        assert len(states) == 1


class TestKalmanStateManagement:
    """Test suite for Kalman state management functionality."""
    
    def test_kalman_state_creation(self, small_alphabet):
        """Test KalmanState creation and attributes."""
        alphabet_size = len(small_alphabet)
        state_dim = alphabet_size ** 2
        
        # Create mock state data
        x_filtered = np.random.randn(state_dim)
        W_filtered = 0.1 * np.eye(state_dim)
        x_predicted = np.random.randn(state_dim)
        W_predicted = 0.2 * np.eye(state_dim)
        log_likelihood = -100.5
        
        state = KalmanState(
            x_filtered=x_filtered,
            W_filtered=W_filtered,
            x_predicted=x_predicted,
            W_predicted=W_predicted
        )
        
        assert np.array_equal(state.x_filtered, x_filtered)
        assert np.array_equal(state.W_filtered, W_filtered)
        assert np.array_equal(state.x_predicted, x_predicted)
        assert np.array_equal(state.W_predicted, W_predicted) 
