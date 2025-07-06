"""Tests for EM algorithm implementation.

Tests the Expectation-Maximization algorithm for parameter learning
in adaptive syntax evolution models.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from src.adaptive_syntax_filter.core.em_algorithm import EMAlgorithm, EMParameters, EMStatistics
from src.adaptive_syntax_filter.core.kalman import KalmanFilter
from src.adaptive_syntax_filter.core.state_space import StateSpaceManager
from src.adaptive_syntax_filter.core.observation_model import softmax_observation_model


class TestEMAlgorithm:
    """Test suite for EMAlgorithm."""
    
    def test_em_algorithm_initialization(self):
        """Test EM algorithm can be initialized with correct parameters."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager, max_iterations=10, tolerance=1e-6)
        
        assert em.max_iterations == 10
        assert em.tolerance == 1e-6
        assert em.state_manager.alphabet_size == 3
        assert em.state_manager.state_dim == 9
        
    def test_em_parameters_initialization(self):
        """Test EMParameters can be created with correct fields."""
        state_dim = 9
        F = np.eye(state_dim)
        u = np.zeros(state_dim)
        Sigma = np.eye(state_dim) * 0.1
        x0 = np.random.normal(0, 0.1, size=state_dim)
        
        # Use the correct field names from the actual EMParameters class
        params = EMParameters(F=F, u=u, Sigma=Sigma, x0=x0)
        
        assert params.F.shape == (state_dim, state_dim)
        assert params.u.shape == (state_dim,)
        assert params.Sigma.shape == (state_dim, state_dim)
        assert params.x0.shape == (state_dim,)
        
    def test_em_statistics_initialization(self):
        """Test EMStatistics can be created with correct fields."""
        # Use the correct field names from the actual EMStatistics class
        stats = EMStatistics(
            iteration=1,
            log_likelihood=-100.5,
            log_likelihood_change=2.3,
            parameter_change=0.1,
            converged=False
        )
        
        assert stats.iteration == 1
        assert stats.log_likelihood == -100.5
        assert stats.log_likelihood_change == 2.3
        assert stats.parameter_change == 0.1
        assert stats.converged == False

    def test_e_step_basic(self):
        """Test E-step produces valid output dimensions."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager)
        
        # Create test observations  
        obs1 = np.array([0, 1, 2, 0])
        obs2 = np.array([1, 0, 2, 1])
        observations = [obs1, obs2]
        
        # Initialize parameters
        params = em.initialize_parameters(observations, random_state=42)
        
        # Run E-step
        x_smoothed, W_smoothed, W_lag_one = em.e_step(observations, params)
        
        # Check dimensions
        K = len(observations)
        R_squared = state_manager.state_dim
        
        assert x_smoothed.shape == (K, R_squared)
        assert W_smoothed.shape == (K, R_squared, R_squared)
        assert W_lag_one.shape == (K-1, R_squared, R_squared)

    def test_m_step_basic(self):
        """Test M-step produces valid parameter updates."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager)
        
        # Create test data
        obs1 = np.array([0, 1, 2, 0])
        obs2 = np.array([1, 0, 2, 1])
        observations = [obs1, obs2]
        
        # Initialize parameters and run E-step
        params = em.initialize_parameters(observations, random_state=42)
        em.current_params = params  # Set current params for M-step
        x_smoothed, W_smoothed, W_lag_one = em.e_step(observations, params)
        
        # Run M-step
        new_params = em.m_step(observations, x_smoothed, W_smoothed, W_lag_one, params)
        
        # Check that parameters have correct dimensions
        R_squared = state_manager.state_dim
        assert new_params.F.shape == (R_squared, R_squared)
        assert new_params.u.shape == (R_squared,)
        assert new_params.Sigma.shape == (R_squared, R_squared)
        assert new_params.x0.shape == (R_squared,)

    def test_em_iteration_basic(self):
        """Test single EM iteration completes successfully."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager, max_iterations=1)
        
        # Create test observations
        obs1 = np.array([0, 1, 2, 0])
        obs2 = np.array([1, 0, 2, 1])
        observations = [obs1, obs2]
        
        # Run single iteration
        final_params, stats_history = em.fit(observations)
        
        assert len(stats_history) == 1
        assert isinstance(final_params, EMParameters)
        assert stats_history[0].iteration == 0  # EM iterations are 0-indexed

    def test_em_convergence_monitoring(self):
        """Test EM algorithm convergence monitoring."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager, max_iterations=5, tolerance=1e-4)
        
        # Create test observations
        obs1 = np.array([0, 1, 2, 0])
        obs2 = np.array([1, 0, 2, 1])
        observations = [obs1, obs2]
        
        # Run EM
        final_params, stats_history = em.fit(observations)
        
        assert len(stats_history) <= 5
        assert all(isinstance(stat, EMStatistics) for stat in stats_history)
        
        # Check final convergence status
        if len(stats_history) > 1:
            final_stat = stats_history[-1]
            assert isinstance(final_stat.converged, (bool, np.bool_))

    def test_em_parameter_validation(self):
        """Test EM parameter validation."""
        with pytest.raises(ValueError):
            # Invalid alphabet size
            StateSpaceManager(alphabet_size=1, markov_order=1)
        
        with pytest.raises(ValueError):
            # Invalid markov order
            StateSpaceManager(alphabet_size=3, markov_order=0)

    def test_em_numerical_stability(self):
        """Test EM algorithm numerical stability with extreme values."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager, max_iterations=2)
        
        # Create observations with potential numerical issues
        obs1 = np.array([0, 0, 0, 0])  # Repeated symbols
        obs2 = np.array([1, 1, 1, 1])
        observations = [obs1, obs2]
        
        # Should not crash or produce NaN/inf values
        final_params, stats_history = em.fit(observations)
        
        assert np.isfinite(final_params.F).all()
        assert np.isfinite(final_params.u).all()
        assert np.isfinite(final_params.Sigma).all()
        assert np.isfinite(final_params.x0).all()


class TestEMIntegration:
    """Test EM algorithm integration with other components."""
    
    def test_em_kalman_integration(self):
        """Test EM algorithm integrates correctly with Kalman filter."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager, max_iterations=3)
        
        # Create test observations
        obs1 = np.array([0, 1, 2, 0])
        obs2 = np.array([1, 0, 2, 1])
        observations = [obs1, obs2]
        
        # Run EM - should create and use Kalman filter internally
        final_params, stats_history = em.fit(observations)
        
        # Verify Kalman filter was created and used
        assert em.kalman_filter is not None
        assert isinstance(em.kalman_filter, KalmanFilter)
        
        # Verify states were computed
        assert len(em.kalman_filter.states) == len(observations)

    def test_em_observation_model_integration(self):
        """Test EM algorithm integrates with observation model."""
        alphabet_size = 3
        state_manager = StateSpaceManager(alphabet_size=alphabet_size, markov_order=1)
        
        # Create test logit vector
        x = np.random.normal(0, 1, size=alphabet_size**2)
        
        # Test softmax observation model (correct API: only x and R parameters)
        probs = softmax_observation_model(x, alphabet_size)
        
        assert probs.shape == (alphabet_size**2,)
        assert np.allclose(np.sum(probs.reshape(alphabet_size, alphabet_size), axis=1), 1.0)


class TestEMEdgeCases:
    """Test EM algorithm edge cases and error handling."""
    
    def test_em_empty_observations(self):
        """Test EM algorithm with empty observations."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager)
        
        # The EM algorithm should handle empty observations gracefully
        # or raise an informative error
        try:
            em.fit([])
            assert False, "Should have raised an error for empty observations"
        except (ValueError, RuntimeError) as e:
            # Either ValueError or RuntimeError is acceptable
            assert len(str(e)) > 0

    def test_em_single_observation(self):
        """Test EM algorithm with single observation sequence."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager, max_iterations=2)
        
        # Single observation sequence
        observations = [np.array([0, 1, 2])]
        
        # Should handle gracefully
        final_params, stats_history = em.fit(observations)
        
        assert isinstance(final_params, EMParameters)
        assert len(stats_history) >= 1

    def test_em_zero_iterations(self):
        """Test EM algorithm with zero iterations."""
        state_manager = StateSpaceManager(alphabet_size=3, markov_order=1)
        em = EMAlgorithm(state_space_manager=state_manager, max_iterations=0)
        
        observations = [np.array([0, 1, 2])]
        
        # Should return initial parameters
        final_params, stats_history = em.fit(observations)
        
        assert isinstance(final_params, EMParameters)
        assert len(stats_history) == 0 