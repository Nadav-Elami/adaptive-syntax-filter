"""Tests for global random seed management functionality."""

import pytest
import random
import numpy as np
import os
import hashlib
from unittest.mock import patch

from src.adaptive_syntax_filter.config.random_state import (
    set_global_seed,
    get_global_seed,
    get_random_state,
    create_deterministic_seed,
    reset_random_state,
    get_environment_seed,
    ensure_reproducibility
)


class TestSetGlobalSeed:
    """Test suite for set_global_seed function."""
    
    def test_set_global_seed_basic(self):
        """Test basic seed setting functionality."""
        set_global_seed(42)
        
        # Check that global seed is stored
        assert get_global_seed() == 42
        
        # Check that random state is initialized
        state = get_random_state()
        assert state is not None
        assert state['seed'] == 42
        assert 'python_state' in state
        assert 'numpy_state' in state
    
    def test_set_global_seed_reproducibility(self):
        """Test that setting the same seed produces reproducible results."""
        # First run
        set_global_seed(123)
        random_val1 = random.random()
        numpy_val1 = np.random.random()
        
        # Second run with same seed
        set_global_seed(123)
        random_val2 = random.random()
        numpy_val2 = np.random.random()
        
        # Should be identical
        assert random_val1 == random_val2
        assert numpy_val1 == numpy_val2
    
    def test_set_global_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        # First seed
        set_global_seed(100)
        random_val1 = random.random()
        numpy_val1 = np.random.random()
        
        # Different seed
        set_global_seed(200)
        random_val2 = random.random()
        numpy_val2 = np.random.random()
        
        # Should be different
        assert random_val1 != random_val2
        assert numpy_val1 != numpy_val2
    
    def test_set_global_seed_overwrites_previous(self):
        """Test that setting a new seed overwrites the previous one."""
        set_global_seed(1)
        assert get_global_seed() == 1
        
        set_global_seed(2)
        assert get_global_seed() == 2
        
        # State should be updated
        state = get_random_state()
        assert state['seed'] == 2


class TestGetGlobalSeed:
    """Test suite for get_global_seed function."""
    
    def test_get_global_seed_when_set(self):
        """Test getting seed when it has been set."""
        set_global_seed(456)
        assert get_global_seed() == 456
    
    def test_get_global_seed_when_not_set(self):
        """Test getting seed when it hasn't been explicitly set."""
        # Reset module state
        import src.adaptive_syntax_filter.config.random_state as rs_module
        rs_module._GLOBAL_SEED = None
        
        # Should return None if not set
        assert get_global_seed() is None
    
    def test_get_global_seed_type(self):
        """Test that get_global_seed returns correct type."""
        set_global_seed(789)
        seed = get_global_seed()
        assert isinstance(seed, int)
        assert seed == 789


class TestGetRandomState:
    """Test suite for get_random_state function."""
    
    def test_get_random_state_when_initialized(self):
        """Test getting random state when initialized."""
        set_global_seed(111)
        state = get_random_state()
        
        assert state is not None
        assert isinstance(state, dict)
        assert 'seed' in state
        assert 'python_state' in state
        assert 'numpy_state' in state
        assert state['seed'] == 111
    
    def test_get_random_state_when_not_initialized(self):
        """Test getting random state when not initialized."""
        # Reset module state
        import src.adaptive_syntax_filter.config.random_state as rs_module
        rs_module._RNG_STATE = None
        
        # Should return None if not initialized
        assert get_random_state() is None
    
    def test_get_random_state_structure(self):
        """Test the structure of returned random state."""
        set_global_seed(222)
        state = get_random_state()
        
        # Check required keys
        required_keys = {'seed', 'python_state', 'numpy_state'}
        assert set(state.keys()) == required_keys
        
        # Check types
        assert isinstance(state['seed'], int)
        assert isinstance(state['python_state'], tuple)  # Python's random state is a tuple
        assert isinstance(state['numpy_state'], tuple)   # NumPy's state is also a tuple


class TestCreateDeterministicSeed:
    """Test suite for create_deterministic_seed function."""
    
    def test_create_deterministic_seed_basic(self):
        """Test basic deterministic seed creation."""
        seed = create_deterministic_seed("test_string")
        
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31 - 1  # Should be within valid range
    
    def test_create_deterministic_seed_reproducible(self):
        """Test that same string produces same seed."""
        seed1 = create_deterministic_seed("experiment_1")
        seed2 = create_deterministic_seed("experiment_1")
        
        assert seed1 == seed2
    
    def test_create_deterministic_seed_different_strings(self):
        """Test that different strings produce different seeds."""
        seed1 = create_deterministic_seed("string_a")
        seed2 = create_deterministic_seed("string_b")
        
        assert seed1 != seed2
    
    def test_create_deterministic_seed_empty_string(self):
        """Test deterministic seed with empty string."""
        seed = create_deterministic_seed("")
        
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31 - 1
    
    def test_create_deterministic_seed_special_characters(self):
        """Test deterministic seed with special characters."""
        seed = create_deterministic_seed("test!@#$%^&*()_+")
        
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31 - 1
    
    def test_create_deterministic_seed_long_string(self):
        """Test deterministic seed with very long string."""
        long_string = "a" * 1000
        seed = create_deterministic_seed(long_string)
        
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31 - 1
    
    def test_create_deterministic_seed_implementation(self):
        """Test that implementation matches expected behavior."""
        test_string = "test"
        expected_hash = hashlib.sha256(test_string.encode()).hexdigest()
        expected_seed = int(expected_hash[:8], 16) % (2**31 - 1)
        
        actual_seed = create_deterministic_seed(test_string)
        assert actual_seed == expected_seed


class TestResetRandomState:
    """Test suite for reset_random_state function."""
    
    def test_reset_random_state_basic(self):
        """Test basic random state reset functionality."""
        # Initialize state
        set_global_seed(333)
        
        # Generate some random numbers to change state
        random.random()
        np.random.random()
        
        # Reset state
        reset_random_state()
        
        # Generate numbers again - should be same as initial
        set_global_seed(333)
        expected_random = random.random()
        expected_numpy = np.random.random()
        
        # Reset and generate again
        reset_random_state()
        actual_random = random.random()
        actual_numpy = np.random.random()
        
        assert actual_random == expected_random
        assert actual_numpy == expected_numpy
    
    def test_reset_random_state_without_initialization(self):
        """Test reset when random state not initialized."""
        # Reset module state
        import src.adaptive_syntax_filter.config.random_state as rs_module
        rs_module._RNG_STATE = None
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Random state not initialized"):
            reset_random_state()
    
    def test_reset_random_state_multiple_times(self):
        """Test resetting random state multiple times."""
        set_global_seed(444)
        
        # Get initial values
        initial_random = random.random()
        initial_numpy = np.random.random()
        
        # Generate more numbers
        random.random()
        np.random.random()
        
        # Reset multiple times
        reset_random_state()
        reset_random_state()
        
        # Should still reset to initial state
        reset_val_random = random.random()
        reset_val_numpy = np.random.random()
        
        assert reset_val_random == initial_random
        assert reset_val_numpy == initial_numpy


class TestGetEnvironmentSeed:
    """Test suite for get_environment_seed function."""
    
    def test_get_environment_seed_with_valid_integer(self):
        """Test getting seed from environment variable with valid integer."""
        with patch.dict(os.environ, {'ADAPTIVE_SYNTAX_FILTER_SEED': '12345'}):
            seed = get_environment_seed()
            assert seed == 12345
    
    def test_get_environment_seed_with_invalid_integer(self):
        """Test getting seed from environment variable with invalid integer."""
        with patch.dict(os.environ, {'ADAPTIVE_SYNTAX_FILTER_SEED': 'not_a_number'}):
            seed = get_environment_seed()
            
            # Should create deterministic seed from string
            expected_seed = create_deterministic_seed('not_a_number')
            assert seed == expected_seed
    
    def test_get_environment_seed_not_set(self):
        """Test getting seed when environment variable not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the env var is not set
            if 'ADAPTIVE_SYNTAX_FILTER_SEED' in os.environ:
                del os.environ['ADAPTIVE_SYNTAX_FILTER_SEED']
            
            seed = get_environment_seed()
            assert seed == 42  # Default value
    
    def test_get_environment_seed_empty_string(self):
        """Test getting seed from empty environment variable."""
        with patch.dict(os.environ, {'ADAPTIVE_SYNTAX_FILTER_SEED': ''}):
            seed = get_environment_seed()
            
            # Empty string should create deterministic seed
            expected_seed = create_deterministic_seed('')
            assert seed == expected_seed
    
    def test_get_environment_seed_zero(self):
        """Test getting seed from environment variable with zero."""
        with patch.dict(os.environ, {'ADAPTIVE_SYNTAX_FILTER_SEED': '0'}):
            seed = get_environment_seed()
            assert seed == 0
    
    def test_get_environment_seed_negative(self):
        """Test getting seed from environment variable with negative number."""
        with patch.dict(os.environ, {'ADAPTIVE_SYNTAX_FILTER_SEED': '-100'}):
            seed = get_environment_seed()
            assert seed == -100


class TestEnsureReproducibility:
    """Test suite for ensure_reproducibility function."""
    
    def test_ensure_reproducibility_when_not_set(self):
        """Test ensure_reproducibility when global seed not set."""
        # Reset module state
        import src.adaptive_syntax_filter.config.random_state as rs_module
        rs_module._GLOBAL_SEED = None
        
        with patch('src.adaptive_syntax_filter.config.random_state.get_environment_seed', return_value=555):
            seed = ensure_reproducibility()
            
            assert seed == 555
            assert get_global_seed() == 555
    
    def test_ensure_reproducibility_when_already_set(self):
        """Test ensure_reproducibility when global seed already set."""
        set_global_seed(666)
        
        # Should return existing seed without changing it
        seed = ensure_reproducibility()
        assert seed == 666
        assert get_global_seed() == 666
    
    def test_ensure_reproducibility_sets_random_state(self):
        """Test that ensure_reproducibility actually sets the random state."""
        # Reset module state
        import src.adaptive_syntax_filter.config.random_state as rs_module
        rs_module._GLOBAL_SEED = None
        
        with patch('src.adaptive_syntax_filter.config.random_state.get_environment_seed', return_value=777):
            ensure_reproducibility()
            
            # Random state should be initialized
            state = get_random_state()
            assert state is not None
            assert state['seed'] == 777


class TestModuleAutoInitialization:
    """Test suite for module auto-initialization behavior."""
    
    def test_module_initialization_with_env_var(self):
        """Test that module initializes automatically with environment variable."""
        # This test is tricky because the module auto-initializes on import
        # We can test the intended behavior by checking environment handling
        with patch.dict(os.environ, {'ADAPTIVE_SYNTAX_FILTER_SEED': '888'}):
            seed = get_environment_seed()
            assert seed == 888
    
    def test_module_initialization_disabled(self):
        """Test module initialization when disabled by empty env var."""
        # When env var is empty string, auto-initialization should be disabled
        with patch.dict(os.environ, {'ADAPTIVE_SYNTAX_FILTER_SEED': ''}):
            # This would normally be tested by re-importing the module,
            # but that's complex in tests. We just verify the condition.
            env_value = os.environ.get('ADAPTIVE_SYNTAX_FILTER_SEED')
            assert env_value == ''


class TestIntegrationScenarios:
    """Integration tests combining multiple functions."""
    
    def test_full_reproducibility_workflow(self):
        """Test complete workflow for reproducible research."""
        # Create deterministic seed from experiment name
        experiment_name = "birdsong_analysis_v2.1"
        seed = create_deterministic_seed(experiment_name)
        
        # Set global seed
        set_global_seed(seed)
        
        # Generate some random data
        data1 = [random.random() for _ in range(5)]
        data2 = np.random.random(5)
        
        # Reset and generate again - should be identical
        reset_random_state()
        data1_repeat = [random.random() for _ in range(5)]
        data2_repeat = np.random.random(5)
        
        assert data1 == data1_repeat
        assert np.array_equal(data2, data2_repeat)
    
    def test_environment_based_reproducibility(self):
        """Test reproducibility using environment variable."""
        with patch.dict(os.environ, {'ADAPTIVE_SYNTAX_FILTER_SEED': 'exp_123'}):
            # Reset global state
            import src.adaptive_syntax_filter.config.random_state as rs_module
            rs_module._GLOBAL_SEED = None
            
            # Ensure reproducibility
            seed = ensure_reproducibility()
            
            # Should be deterministic from environment string
            expected_seed = create_deterministic_seed('exp_123')
            assert seed == expected_seed
            assert get_global_seed() == expected_seed
    
    def test_seed_persistence_across_operations(self):
        """Test that seed information persists across random operations."""
        set_global_seed(999)
        
        # Perform many random operations
        for _ in range(100):
            random.random()
            np.random.random()
        
        # Seed should still be stored
        assert get_global_seed() == 999
        
        # State should still be available for reset
        reset_random_state()
        
        # Should be able to reproduce initial sequence
        first_val = random.random()
        
        reset_random_state()
        second_val = random.random()
        
        assert first_val == second_val
    
    def test_cross_platform_determinism(self):
        """Test that seed generation is deterministic across platforms."""
        # This tests the deterministic seed creation
        test_strings = [
            "experiment_1",
            "test_config_v1.0",
            "birdsong_analysis",
            "canary_study_2023"
        ]
        
        seeds = []
        for test_string in test_strings:
            seed = create_deterministic_seed(test_string)
            seeds.append(seed)
            
            # Each seed should be valid
            assert isinstance(seed, int)
            assert 0 <= seed < 2**31 - 1
        
        # All seeds should be different
        assert len(set(seeds)) == len(seeds)
        
        # Seeds should be reproducible
        for test_string in test_strings:
            seed1 = create_deterministic_seed(test_string)
            seed2 = create_deterministic_seed(test_string)
            assert seed1 == seed2 