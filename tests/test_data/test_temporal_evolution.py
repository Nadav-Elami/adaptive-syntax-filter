"""Tests for temporal evolution models and parameter trajectories."""

import pytest
import numpy as np
from unittest.mock import patch
from typing import Dict, Any

from src.adaptive_syntax_filter.data.temporal_evolution import (
    EvolutionConfig,
    compute_evolution_trajectory,
    get_batch_evolution_schedule,
    visualize_evolution_schedule,
    create_evolution_trajectory_from_config,
    EvolutionManager,
    create_evolution_examples,
    validate_evolution_parameters
)


class TestEvolutionConfig:
    """Test suite for EvolutionConfig dataclass."""
    
    def test_evolution_config_basic_creation(self):
        """Test basic EvolutionConfig creation."""
        config = EvolutionConfig(evolution_type="linear")
        
        assert config.evolution_type == "linear"
        assert config.batch_size is None
        assert config.evolution_params == {}
    
    def test_evolution_config_with_parameters(self):
        """Test EvolutionConfig with custom parameters."""
        params = {"rate": 2.0, "steepness": 6.0}
        config = EvolutionConfig(
            evolution_type="exponential",
            batch_size=50,
            evolution_params=params
        )
        
        assert config.evolution_type == "exponential"
        assert config.batch_size == 50
        assert config.evolution_params == params
    
    def test_evolution_config_post_init(self):
        """Test that __post_init__ initializes empty evolution_params."""
        config = EvolutionConfig(evolution_type="sigmoid", evolution_params=None)
        
        assert config.evolution_params == {}


class TestComputeEvolutionTrajectory:
    """Test suite for compute_evolution_trajectory function."""
    
    def test_linear_evolution(self):
        """Test linear evolution model."""
        x_init = np.array([0.0, 1.0, 2.0])
        x_final = np.array([3.0, 4.0, 5.0])
        n_sequences = 5
        
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "linear"
        )
        
        # Check shape
        assert trajectory.shape == (3, 5)
        
        # Check endpoints
        np.testing.assert_array_almost_equal(trajectory[:, 0], x_init)
        np.testing.assert_array_almost_equal(trajectory[:, -1], x_final)
        
        # Check linear progression
        expected_mid = x_init + 0.5 * (x_final - x_init)
        np.testing.assert_array_almost_equal(trajectory[:, 2], expected_mid)
    
    def test_exponential_evolution(self):
        """Test exponential evolution model."""
        x_init = np.array([1.0, 2.0])
        x_final = np.array([3.0, 8.0])
        n_sequences = 3
        
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "exponential", rate=2.0
        )
        
        # Check shape
        assert trajectory.shape == (2, 3)
        
        # Check endpoints
        np.testing.assert_array_almost_equal(trajectory[:, 0], x_init)
        np.testing.assert_array_almost_equal(trajectory[:, -1], x_final)
        
        # All values should be finite
        assert np.all(np.isfinite(trajectory))
    
    def test_sigmoid_evolution(self):
        """Test sigmoid evolution model."""
        x_init = np.array([0.0, 1.0])
        x_final = np.array([1.0, 0.0])
        n_sequences = 5
        
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "sigmoid", 
            steepness=6.0, midpoint=0.5
        )
        
        # Check shape
        assert trajectory.shape == (2, 5)
        
        # Check endpoints
        np.testing.assert_array_almost_equal(trajectory[:, 0], x_init, decimal=3)
        np.testing.assert_array_almost_equal(trajectory[:, -1], x_final, decimal=3)
        
        # Check S-curve property: middle should be close to average
        mid_expected = (x_init + x_final) / 2
        np.testing.assert_array_almost_equal(trajectory[:, 2], mid_expected, decimal=1)
    
    def test_piecewise_evolution(self):
        """Test piecewise evolution model."""
        x_init = np.array([0.0])
        x_final = np.array([3.0])
        n_sequences = 7
        
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "piecewise",
            breakpoints=[0.33, 0.67]
        )
        
        # Check shape
        assert trajectory.shape == (1, 7)
        
        # Check endpoints
        np.testing.assert_array_almost_equal(trajectory[:, 0], x_init)
        np.testing.assert_array_almost_equal(trajectory[:, -1], x_final)
        
        # Should be monotonic for this simple case
        assert np.all(np.diff(trajectory[0, :]) >= 0)
    
    def test_oscillatory_evolution(self):
        """Test oscillatory evolution model."""
        x_init = np.array([0.0, 1.0])
        x_final = np.array([1.0, 0.0])
        n_sequences = 10
        
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "oscillatory",
            frequency=1.0, amplitude=0.1, trend_weight=0.8
        )
        
        # Check shape
        assert trajectory.shape == (2, 10)
        
        # Check that trajectory has oscillatory behavior
        # (not strictly monotonic due to oscillations)
        diffs = np.diff(trajectory[0, :])
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        assert sign_changes > 0  # Should have some direction changes
    
    def test_constant_evolution(self):
        """Test constant evolution model."""
        x_init = np.array([1.0, 2.0, 3.0])
        x_final = np.array([4.0, 5.0, 6.0])  # Should be ignored
        n_sequences = 5
        
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "constant"
        )
        
        # Check shape
        assert trajectory.shape == (3, 5)
        
        # All sequences should have initial values
        for i in range(n_sequences):
            np.testing.assert_array_equal(trajectory[:, i], x_init)
    
    def test_evolution_with_infinite_values(self):
        """Test evolution with infinite initial values."""
        x_init = np.array([-np.inf, 0.0, np.inf])
        x_final = np.array([0.0, 1.0, np.inf])
        n_sequences = 3
        
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "linear"
        )
        
        # Check shape
        assert trajectory.shape == (3, 3)
        
        # Infinite values should remain unchanged
        assert trajectory[0, 0] == -np.inf
        assert trajectory[2, 0] == np.inf
        assert trajectory[2, -1] == np.inf
        
        # Finite evolution should work normally
        assert trajectory[1, 0] == 0.0
        assert trajectory[1, -1] == 1.0
    
    def test_input_validation(self):
        """Test input validation for compute_evolution_trajectory."""
        x_init = np.array([1.0, 2.0])
        x_final = np.array([3.0])  # Wrong size
        
        # Mismatched vector sizes
        with pytest.raises(ValueError, match="must have same length"):
            compute_evolution_trajectory(x_init, x_final, 5, "linear")
        
        # Invalid number of sequences
        with pytest.raises(ValueError, match="must be positive"):
            compute_evolution_trajectory(x_init, x_init, 0, "linear")
        
        # Invalid evolution type
        with pytest.raises(ValueError, match="Unknown evolution type"):
            compute_evolution_trajectory(x_init, x_init, 5, "invalid")
    
    def test_batch_evolution(self):
        """Test batched evolution mode."""
        x_init = np.array([0.0, 1.0])
        x_final = np.array([1.0, 0.0])
        n_sequences = 10
        batch_size = 3
        
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "linear", batch_size=batch_size
        )
        
        # Check shape
        assert trajectory.shape == (2, 10)
        
        # First batch should be constant
        np.testing.assert_array_equal(
            trajectory[:, 0], trajectory[:, 1]
        )
        np.testing.assert_array_equal(
            trajectory[:, 1], trajectory[:, 2]
        )
        
        # Parameters should change between batches
        assert not np.array_equal(trajectory[:, 2], trajectory[:, 3])


class TestBatchEvolutionSchedule:
    """Test suite for batch evolution scheduling functions."""
    
    def test_get_batch_evolution_schedule_basic(self):
        """Test basic batch evolution schedule."""
        schedule = get_batch_evolution_schedule(
            n_sequences=10, batch_size=3, evolution_type="linear"
        )
        
        # Check structure
        required_keys = ['n_batches', 'batch_boundaries', 'batch_sizes', 
                        'evolution_points', 'evolution_type', 'sequences_per_evolution']
        for key in required_keys:
            assert key in schedule
        
        # Check values
        assert schedule['n_batches'] == 4  # ceil(10/3)
        assert schedule['evolution_type'] == "linear"
        assert schedule['sequences_per_evolution'] == 3
        
        # Check batch boundaries
        expected_boundaries = [(0, 3), (3, 6), (6, 9), (9, 10)]
        assert schedule['batch_boundaries'] == expected_boundaries
        
        # Check batch sizes
        expected_sizes = [3, 3, 3, 1]
        assert schedule['batch_sizes'] == expected_sizes
        
        # Check evolution points (starts of batches after first)
        expected_evolution_points = [3, 6, 9]
        assert schedule['evolution_points'] == expected_evolution_points
    
    def test_get_batch_evolution_schedule_exact_division(self):
        """Test batch schedule when sequences divide evenly."""
        schedule = get_batch_evolution_schedule(n_sequences=9, batch_size=3)
        
        assert schedule['n_batches'] == 3
        assert schedule['batch_sizes'] == [3, 3, 3]
        assert schedule['batch_boundaries'] == [(0, 3), (3, 6), (6, 9)]
    
    def test_get_batch_evolution_schedule_single_batch(self):
        """Test batch schedule with single batch."""
        schedule = get_batch_evolution_schedule(n_sequences=5, batch_size=10)
        
        assert schedule['n_batches'] == 1
        assert schedule['batch_sizes'] == [5]
        assert schedule['evolution_points'] == []  # No evolution with single batch
    
    def test_visualize_evolution_schedule(self):
        """Test evolution schedule visualization."""
        # Test without errors (output goes to stdout)
        visualize_evolution_schedule(n_sequences=15, batch_size=4, show_details=True)
        visualize_evolution_schedule(n_sequences=15, batch_size=4, show_details=False)


class TestEvolutionManager:
    """Test suite for EvolutionManager class."""
    
    def test_evolution_manager_initialization(self):
        """Test EvolutionManager initialization."""
        manager = EvolutionManager("linear")
        
        assert manager.evolution_type == "linear"
        assert manager.evolution_params == {}
        assert manager.last_trajectory is None
    
    def test_evolution_manager_with_params(self):
        """Test EvolutionManager with parameters."""
        manager = EvolutionManager("exponential", rate=3.0, custom_param="test")
        
        assert manager.evolution_type == "exponential"
        assert manager.evolution_params == {"rate": 3.0, "custom_param": "test"}
    
    def test_evolution_manager_invalid_type(self):
        """Test EvolutionManager with invalid evolution type."""
        with pytest.raises(ValueError, match="Evolution type must be one of"):
            EvolutionManager("invalid_type")
    
    def test_evolution_manager_compute_trajectory(self):
        """Test trajectory computation through manager."""
        manager = EvolutionManager("linear")
        
        x_init = np.array([0.0, 1.0])
        x_final = np.array([1.0, 0.0])
        n_sequences = 5
        
        trajectory = manager.compute_trajectory(x_init, x_final, n_sequences)
        
        # Check shape and endpoints
        assert trajectory.shape == (2, 5)
        np.testing.assert_array_almost_equal(trajectory[:, 0], x_init)
        np.testing.assert_array_almost_equal(trajectory[:, -1], x_final)
        
        # Should store trajectory
        assert manager.last_trajectory is not None
        np.testing.assert_array_equal(manager.last_trajectory, trajectory)
    
    def test_evolution_manager_update_params(self):
        """Test parameter updates."""
        manager = EvolutionManager("sigmoid", steepness=4.0)
        
        assert manager.evolution_params["steepness"] == 4.0
        
        manager.update_params(steepness=8.0, midpoint=0.3)
        
        assert manager.evolution_params["steepness"] == 8.0
        assert manager.evolution_params["midpoint"] == 0.3
    
    def test_evolution_manager_get_config(self):
        """Test configuration object generation."""
        manager = EvolutionManager("piecewise", breakpoints=[0.25, 0.75])
        
        config = manager.get_config(batch_size=20)
        
        assert isinstance(config, EvolutionConfig)
        assert config.evolution_type == "piecewise"
        assert config.batch_size == 20
        assert config.evolution_params == {"breakpoints": [0.25, 0.75]}
    
    def test_evolution_manager_repr(self):
        """Test string representation."""
        manager = EvolutionManager("oscillatory", frequency=2.0)
        
        repr_str = repr(manager)
        
        assert "EvolutionManager" in repr_str
        assert "oscillatory" in repr_str
        assert "frequency" in repr_str


class TestCreateEvolutionExamples:
    """Test suite for create_evolution_examples function."""
    
    def test_create_evolution_examples_basic(self):
        """Test creation of example evolution managers."""
        examples = create_evolution_examples()
        
        # Check that all standard types are included
        expected_types = ['linear', 'exponential', 'sigmoid', 'piecewise', 
                         'oscillatory', 'constant']
        
        assert set(examples.keys()) == set(expected_types)
        
        # Check that all are EvolutionManager instances
        for name, manager in examples.items():
            assert isinstance(manager, EvolutionManager)
            assert manager.evolution_type == name
    
    def test_create_evolution_examples_parameters(self):
        """Test that examples have appropriate parameters."""
        examples = create_evolution_examples()
        
        # Check specific parameters
        assert "rate" in examples["exponential"].evolution_params
        assert "steepness" in examples["sigmoid"].evolution_params
        assert "midpoint" in examples["sigmoid"].evolution_params
        assert "breakpoints" in examples["piecewise"].evolution_params
        assert "frequency" in examples["oscillatory"].evolution_params
        assert "amplitude" in examples["oscillatory"].evolution_params
        assert "trend_weight" in examples["oscillatory"].evolution_params
    
    def test_evolution_examples_functionality(self):
        """Test that example managers can compute trajectories."""
        examples = create_evolution_examples()
        
        x_init = np.array([0.0, 1.0])
        x_final = np.array([1.0, 0.0])
        n_sequences = 5
        
        for name, manager in examples.items():
            trajectory = manager.compute_trajectory(x_init, x_final, n_sequences)
            
            assert trajectory.shape == (2, 5)
            assert np.all(np.isfinite(trajectory))


class TestValidateEvolutionParameters:
    """Test suite for validate_evolution_parameters function."""
    
    def test_validate_linear_parameters(self):
        """Test validation of linear evolution (no specific parameters)."""
        result = validate_evolution_parameters("linear")
        
        assert result["valid"] is True
        assert result["evolution_type"] == "linear"
        assert result["warnings"] == []
        assert result["parameters"] == {}
    
    def test_validate_exponential_parameters_valid(self):
        """Test validation of valid exponential parameters."""
        result = validate_evolution_parameters("exponential", rate=2.0)
        
        assert result["valid"] is True
        assert result["evolution_type"] == "exponential"
        assert result["warnings"] == []
        assert result["parameters"] == {"rate": 2.0}
    
    def test_validate_exponential_parameters_invalid(self):
        """Test validation of invalid exponential parameters."""
        # Negative rate
        result = validate_evolution_parameters("exponential", rate=-1.0)
        
        assert result["valid"] is False
        assert "positive" in result["warnings"][0]
        
        # Very large rate (warning but valid)
        result = validate_evolution_parameters("exponential", rate=15.0)
        
        assert result["valid"] is True
        assert "instability" in result["warnings"][0]
    
    def test_validate_sigmoid_parameters_valid(self):
        """Test validation of valid sigmoid parameters."""
        result = validate_evolution_parameters(
            "sigmoid", steepness=6.0, midpoint=0.5
        )
        
        assert result["valid"] is True
        assert result["warnings"] == []
    
    def test_validate_sigmoid_parameters_invalid(self):
        """Test validation of invalid sigmoid parameters."""
        # Invalid steepness
        result = validate_evolution_parameters("sigmoid", steepness=-2.0)
        
        assert result["valid"] is False
        assert "positive" in result["warnings"][0]
        
        # Invalid midpoint
        result = validate_evolution_parameters("sigmoid", midpoint=1.5)
        
        assert result["valid"] is False
        assert "[0, 1]" in result["warnings"][0]
    
    def test_validate_piecewise_parameters_valid(self):
        """Test validation of valid piecewise parameters."""
        result = validate_evolution_parameters(
            "piecewise", breakpoints=[0.3, 0.7]
        )
        
        assert result["valid"] is True
        assert result["warnings"] == []
    
    def test_validate_piecewise_parameters_invalid(self):
        """Test validation of invalid piecewise parameters."""
        # Breakpoints outside (0,1)
        result = validate_evolution_parameters(
            "piecewise", breakpoints=[0.0, 0.5, 1.0]
        )
        
        assert result["valid"] is False
        assert "(0, 1)" in result["warnings"][0]
        
        # Unsorted breakpoints
        result = validate_evolution_parameters(
            "piecewise", breakpoints=[0.7, 0.3]
        )
        
        assert result["valid"] is False
        assert "sorted" in result["warnings"][0]
    
    def test_validate_oscillatory_parameters_valid(self):
        """Test validation of valid oscillatory parameters."""
        result = validate_evolution_parameters(
            "oscillatory", frequency=2.0, amplitude=0.1, trend_weight=0.8
        )
        
        assert result["valid"] is True
        assert result["warnings"] == []
    
    def test_validate_oscillatory_parameters_invalid(self):
        """Test validation of invalid oscillatory parameters."""
        # Invalid frequency
        result = validate_evolution_parameters("oscillatory", frequency=-1.0)
        
        assert result["valid"] is False
        assert "positive" in result["warnings"][0]
        
        # Invalid amplitude
        result = validate_evolution_parameters("oscillatory", amplitude=-0.1)
        
        assert result["valid"] is False
        assert "non-negative" in result["warnings"][0]
        
        # Invalid trend weight
        result = validate_evolution_parameters("oscillatory", trend_weight=1.5)
        
        assert result["valid"] is False
        assert "[0, 1]" in result["warnings"][0]
    
    def test_validate_constant_parameters(self):
        """Test validation of constant evolution (no specific parameters)."""
        result = validate_evolution_parameters("constant")
        
        assert result["valid"] is True
        assert result["warnings"] == []


class TestCreateEvolutionTrajectoryFromConfig:
    """Test suite for create_evolution_trajectory_from_config function."""
    
    def test_create_trajectory_from_config_basic(self):
        """Test trajectory creation from config object."""
        config = EvolutionConfig(
            evolution_type="linear",
            batch_size=None,
            evolution_params={}
        )
        
        x_init = np.array([0.0, 1.0])
        x_final = np.array([1.0, 0.0])
        n_sequences = 5
        
        trajectory = create_evolution_trajectory_from_config(
            config, x_init, x_final, n_sequences
        )
        
        assert trajectory.shape == (2, 5)
        np.testing.assert_array_almost_equal(trajectory[:, 0], x_init)
        np.testing.assert_array_almost_equal(trajectory[:, -1], x_final)
    
    def test_create_trajectory_from_config_with_params(self):
        """Test trajectory creation with evolution parameters."""
        config = EvolutionConfig(
            evolution_type="sigmoid",
            batch_size=None,
            evolution_params={"steepness": 8.0, "midpoint": 0.3}
        )
        
        x_init = np.array([0.0])
        x_final = np.array([1.0])
        n_sequences = 7
        
        trajectory = create_evolution_trajectory_from_config(
            config, x_init, x_final, n_sequences
        )
        
        assert trajectory.shape == (1, 7)
        assert np.all(np.isfinite(trajectory))
    
    def test_create_trajectory_from_config_with_batching(self):
        """Test trajectory creation with batch configuration."""
        config = EvolutionConfig(
            evolution_type="linear",
            batch_size=3,
            evolution_params={}
        )
        
        x_init = np.array([0.0, 1.0])
        x_final = np.array([1.0, 0.0])
        n_sequences = 8
        
        trajectory = create_evolution_trajectory_from_config(
            config, x_init, x_final, n_sequences
        )
        
        assert trajectory.shape == (2, 8)
        
        # First batch should be constant
        np.testing.assert_array_equal(trajectory[:, 0], trajectory[:, 1])
        np.testing.assert_array_equal(trajectory[:, 1], trajectory[:, 2])


class TestIntegrationScenarios:
    """Integration tests for temporal evolution functionality."""
    
    def test_full_evolution_workflow(self):
        """Test complete evolution workflow from config to trajectory."""
        # Create manager
        manager = EvolutionManager("sigmoid", steepness=4.0, midpoint=0.4)
        
        # Get config
        config = manager.get_config(batch_size=5)
        
        # Validate parameters
        validation = validate_evolution_parameters(
            config.evolution_type, **config.evolution_params
        )
        assert validation["valid"]
        
        # Create trajectory using multiple methods
        x_init = np.array([0.0, 2.0, -1.0])
        x_final = np.array([1.0, 0.0, 1.0])
        n_sequences = 10
        
        # Method 1: Through manager
        trajectory1 = manager.compute_trajectory(
            x_init, x_final, n_sequences, batch_size=5
        )
        
        # Method 2: Through config
        trajectory2 = create_evolution_trajectory_from_config(
            config, x_init, x_final, n_sequences
        )
        
        # Method 3: Direct function call
        trajectory3 = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "sigmoid", 
            batch_size=5, steepness=4.0, midpoint=0.4
        )
        
        # All methods should produce same result
        np.testing.assert_array_almost_equal(trajectory1, trajectory2)
        np.testing.assert_array_almost_equal(trajectory2, trajectory3)
    
    def test_all_evolution_types_consistency(self):
        """Test that all evolution types produce consistent results."""
        x_init = np.array([0.0, 1.0, -0.5])
        x_final = np.array([1.0, 0.0, 0.5])
        n_sequences = 6
        
        evolution_types = ['linear', 'exponential', 'sigmoid', 'piecewise', 
                          'oscillatory', 'constant']
        
        for evolution_type in evolution_types:
            # Get example manager
            examples = create_evolution_examples()
            manager = examples[evolution_type]
            
            # Compute trajectory
            trajectory = manager.compute_trajectory(x_init, x_final, n_sequences)
            
            # Basic consistency checks
            assert trajectory.shape == (3, 6)
            assert np.all(np.isfinite(trajectory))
            
            # Check endpoints (except for constant which ignores x_final)
            if evolution_type != 'constant':
                np.testing.assert_array_almost_equal(
                    trajectory[:, 0], x_init, decimal=2
                )
                np.testing.assert_array_almost_equal(
                    trajectory[:, -1], x_final, decimal=2
                )
            else:
                # Constant should be all x_init
                for i in range(n_sequences):
                    np.testing.assert_array_equal(trajectory[:, i], x_init)
    
    def test_batch_evolution_detailed(self):
        """Test detailed batch evolution behavior."""
        x_init = np.array([0.0])
        x_final = np.array([1.0])
        n_sequences = 13
        batch_size = 4
        
        # Get batch schedule
        schedule = get_batch_evolution_schedule(n_sequences, batch_size)
        
        # Compute trajectory
        trajectory = compute_evolution_trajectory(
            x_init, x_final, n_sequences, "linear", batch_size=batch_size
        )
        
        # Verify batch structure
        assert schedule['n_batches'] == 4
        assert schedule['batch_sizes'] == [4, 4, 4, 1]
        
        # Check that values are constant within batches
        for start, end in schedule['batch_boundaries']:
            batch_values = trajectory[0, start:end]
            assert np.all(batch_values == batch_values[0])
        
        # Check that values change between batches (except last might equal previous)
        for i in range(len(schedule['evolution_points'])):
            evolution_point = schedule['evolution_points'][i]
            before_value = trajectory[0, evolution_point - 1]
            after_value = trajectory[0, evolution_point]
            # Values should be different (allowing for floating point precision)
            assert abs(before_value - after_value) > 1e-10
    
    def test_parameter_validation_integration(self):
        """Test parameter validation integration with managers."""
        # Test all evolution types with their default parameters
        examples = create_evolution_examples()
        
        for evolution_type, manager in examples.items():
            validation = validate_evolution_parameters(
                evolution_type, **manager.evolution_params
            )
            
            # All examples should have valid parameters
            assert validation["valid"], f"Invalid parameters for {evolution_type}: {validation['warnings']}"
            
            # Test that manager can actually compute trajectories
            x_init = np.array([0.0, 1.0])
            x_final = np.array([1.0, 0.0])
            
            trajectory = manager.compute_trajectory(x_init, x_final, 5)
            assert trajectory.shape == (2, 5)
            assert np.all(np.isfinite(trajectory)) 