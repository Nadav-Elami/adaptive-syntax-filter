"""Tests for logit evolution visualization.

Tests the priority #1 visualization system for logit parameter evolution,
including dashboard functionality, static plots, and block softmax operations.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

from adaptive_syntax_filter.viz import (
    LogitEvolutionDashboard, LogitVisualizationConfig,
    block_softmax_viz, apply_block_softmax_to_trajectory,
    create_logit_evolution_summary
)


class TestLogitEvolutionDashboard:
    """Test suite for LogitEvolutionDashboard."""
    
    def test_dashboard_initialization(self, small_alphabet):
        """Test LogitEvolutionDashboard initialization."""
        dashboard = LogitEvolutionDashboard(
            alphabet=small_alphabet,
            markov_order=1
        )
        
        assert dashboard.alphabet == small_alphabet
        assert dashboard.n_symbols == len(small_alphabet)
        assert dashboard.markov_order == 1
        assert len(dashboard.context_labels) == len(small_alphabet)
    
    def test_dashboard_with_custom_config(self, small_alphabet):
        """Test dashboard with custom configuration."""
        config = LogitVisualizationConfig(
            figure_size=(12, 8),
            dpi=150,
            font_size=12
        )
        
        dashboard = LogitEvolutionDashboard(
            alphabet=small_alphabet,
            markov_order=1,
            config=config
        )
        
        assert dashboard.config.figure_size == (12, 8)
        assert dashboard.config.dpi == 150
        assert dashboard.config.font_size == 12
    
    def test_higher_order_dashboard(self, medium_alphabet):
        """Test dashboard with higher-order Markov model."""
        dashboard = LogitEvolutionDashboard(
            alphabet=medium_alphabet,
            markov_order=2
        )
        
        assert dashboard.markov_order == 2
        assert dashboard.n_contexts == len(medium_alphabet) ** 2
        assert len(dashboard.context_labels) == len(medium_alphabet) ** 2
    
    @patch('matplotlib.pyplot.show')
    def test_static_logit_plot(self, mock_show, small_alphabet):
        """Test static logit evolution plotting."""
        dashboard = LogitEvolutionDashboard(
            alphabet=small_alphabet,
            markov_order=1
        )
        
        # Create test data
        n_sequences = 10
        state_dim = len(small_alphabet) ** 2
        logits_estimated = np.random.randn(state_dim, n_sequences)
        
        # Test plot creation
        fig = dashboard.plot_logit_evolution_static(logits_estimated)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_static_logit_plot_with_true_values(self, mock_show, small_alphabet):
        """Test static logit plotting with true values comparison."""
        dashboard = LogitEvolutionDashboard(
            alphabet=small_alphabet,
            markov_order=1
        )
        
        # Create test data
        n_sequences = 10
        state_dim = len(small_alphabet) ** 2
        logits_estimated = np.random.randn(state_dim, n_sequences)
        logits_true = np.random.randn(state_dim, n_sequences)
        
        # Test plot with comparison
        fig = dashboard.plot_logit_evolution_static(
            logits_estimated=logits_estimated,
            logits_true=logits_true
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_static_probability_plot(self, mock_show, small_alphabet):
        """Test static probability evolution plotting."""
        dashboard = LogitEvolutionDashboard(
            alphabet=small_alphabet,
            markov_order=1
        )
        
        # Create test data
        n_sequences = 10
        state_dim = len(small_alphabet) ** 2
        probs_estimated = np.random.rand(state_dim, n_sequences)
        
        # Test plot creation
        fig = dashboard.plot_probability_evolution_static(probs_estimated)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestBlockSoftmaxOperations:
    """Test suite for block softmax operations."""
    
    def test_block_softmax_basic(self, small_alphabet):
        """Test basic block softmax functionality."""
        n_symbols = len(small_alphabet)
        state_dim = n_symbols ** 2
        
        # Create test logits
        logits = np.random.randn(state_dim)
        
        # Apply block softmax
        probs = block_softmax_viz(logits, n_symbols)
        
        assert probs.shape == logits.shape
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        
        # Check that each block sums to 1
        n_blocks = state_dim // n_symbols
        for block_idx in range(n_blocks):
            start_idx = block_idx * n_symbols
            end_idx = start_idx + n_symbols
            block_sum = np.sum(probs[start_idx:end_idx])
            assert np.isclose(block_sum, 1.0, rtol=1e-10)
    
    def test_block_softmax_with_constraints(self, small_alphabet):
        """Test block softmax with constraint handling."""
        n_symbols = len(small_alphabet)
        state_dim = n_symbols ** 2
        
        # Create logits with some -inf constraints
        logits = np.random.randn(state_dim)
        logits[0] = -np.inf  # Constrained transition
        logits[3] = -np.inf  # Another constraint
        
        # Apply block softmax with constraint handling
        probs = block_softmax_viz(logits, n_symbols, handle_constraints=True)
        
        assert probs.shape == logits.shape
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert probs[0] == 0  # Constrained probability should be 0
        assert probs[3] == 0  # Constrained probability should be 0
    
    def test_apply_block_softmax_to_trajectory(self, small_alphabet):
        """Test applying block softmax to full trajectory."""
        n_symbols = len(small_alphabet)
        state_dim = n_symbols ** 2
        n_sequences = 20
        
        # Create trajectory of logits
        logit_trajectory = np.random.randn(state_dim, n_sequences)
        
        # Apply block softmax
        prob_trajectory = apply_block_softmax_to_trajectory(logit_trajectory, n_symbols)
        
        assert prob_trajectory.shape == logit_trajectory.shape
        assert np.all(prob_trajectory >= 0)
        assert np.all(prob_trajectory <= 1)
        
        # Check block normalization for each time step
        n_blocks = state_dim // n_symbols
        for seq_idx in range(n_sequences):
            for block_idx in range(n_blocks):
                start_idx = block_idx * n_symbols
                end_idx = start_idx + n_symbols
                block_sum = np.sum(prob_trajectory[start_idx:end_idx, seq_idx])
                assert np.isclose(block_sum, 1.0, rtol=1e-10)


class TestVisualizationUtilities:
    """Test suite for visualization utility functions."""
    
    @patch('matplotlib.pyplot.show')
    def test_create_logit_evolution_summary(self, mock_show, small_alphabet):
        """Test creation of logit evolution summary figures."""
        n_sequences = 15
        state_dim = len(small_alphabet) ** 2
        
        results_dict = {
            'logits_estimated': np.random.randn(state_dim, n_sequences),
            'logits_true': np.random.randn(state_dim, n_sequences),
            'log_likelihood_trajectory': np.random.randn(n_sequences)
        }
        
        # Create summary
        figures = create_logit_evolution_summary(
            results_dict=results_dict,
            alphabet=small_alphabet,
            markov_order=1
        )
        
        assert isinstance(figures, dict)
        assert len(figures) > 0
        
        # Clean up figures
        for fig in figures.values():
            if isinstance(fig, plt.Figure):
                plt.close(fig)


@pytest.mark.performance
class TestVisualizationPerformance:
    """Performance tests for visualization."""
    
    @pytest.mark.performance
    @patch('matplotlib.pyplot.show')
    def test_large_alphabet_visualization(self, mock_show, large_alphabet, performance_test_config):
        """Test visualization performance with large alphabet."""
        import time
        
        dashboard = LogitEvolutionDashboard(
            alphabet=large_alphabet,
            markov_order=1
        )
        
        # Create large test dataset
        n_sequences = performance_test_config['batch_size']
        state_dim = len(large_alphabet) ** 2
        logits_estimated = np.random.randn(state_dim, n_sequences)
        
        # Measure visualization performance
        start_time = time.time()
        fig = dashboard.plot_logit_evolution_static(logits_estimated)
        end_time = time.time()
        
        execution_time = end_time - start_time
        time_limit = performance_test_config['time_limit_seconds']
        
        assert execution_time < time_limit, f"Visualization took {execution_time:.2f}s, limit is {time_limit}s"
        
        plt.close(fig)
    
    @pytest.mark.performance
    @patch('matplotlib.pyplot.show')
    def test_large_trajectory_visualization(self, mock_show, medium_alphabet, performance_test_config):
        """Test visualization performance with long trajectories."""
        import time
        
        dashboard = LogitEvolutionDashboard(
            alphabet=medium_alphabet,
            markov_order=1
        )
        
        # Create long trajectory
        n_sequences = performance_test_config['batch_size'] * 2
        state_dim = len(medium_alphabet) ** 2
        logits_estimated = np.random.randn(state_dim, n_sequences)
        
        # Measure performance
        start_time = time.time()
        fig = dashboard.plot_logit_evolution_static(logits_estimated)
        end_time = time.time()
        
        execution_time = end_time - start_time
        time_limit = performance_test_config['time_limit_seconds']
        
        assert execution_time < time_limit, f"Long trajectory visualization took {execution_time:.2f}s, limit is {time_limit}s"
        
        plt.close(fig)


class TestVisualizationEdgeCases:
    """Edge case tests for visualization."""
    
    @patch('matplotlib.pyplot.show')
    def test_minimal_alphabet_visualization(self, mock_show):
        """Test visualization with minimal alphabet."""
        alphabet = ['<', 'A', '>']
        
        dashboard = LogitEvolutionDashboard(
            alphabet=alphabet,
            markov_order=1
        )
        
        # Test with minimal data
        state_dim = len(alphabet) ** 2
        logits = np.random.randn(state_dim, 3)
        
        fig = dashboard.plot_logit_evolution_static(logits)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @patch('matplotlib.pyplot.show')
    def test_single_sequence_visualization(self, mock_show, small_alphabet):
        """Test visualization with single sequence."""
        dashboard = LogitEvolutionDashboard(
            alphabet=small_alphabet,
            markov_order=1
        )
        
        # Single time step
        state_dim = len(small_alphabet) ** 2
        logits = np.random.randn(state_dim, 1)
        
        fig = dashboard.plot_logit_evolution_static(logits)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_extreme_logit_values(self, small_alphabet):
        """Test block softmax with extreme logit values."""
        n_symbols = len(small_alphabet)
        state_dim = n_symbols ** 2
        
        # Test with extreme values
        logits = np.array([1000, -1000, 0, 500] * (state_dim // 4))[:state_dim]
        
        probs = block_softmax_viz(logits, n_symbols)
        
        assert np.all(np.isfinite(probs))
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        
        # Check normalization still holds
        n_blocks = state_dim // n_symbols
        for block_idx in range(n_blocks):
            start_idx = block_idx * n_symbols
            end_idx = start_idx + n_symbols
            block_sum = np.sum(probs[start_idx:end_idx])
            assert np.isclose(block_sum, 1.0, rtol=1e-6) 
