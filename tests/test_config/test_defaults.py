"""Tests for configuration defaults functionality.

Tests the default configuration system that provides preset configurations,
validation functions, and memory estimation utilities.
"""

import pytest

from src.adaptive_syntax_filter.config.defaults import (
    DefaultConfig,
    BENGALESE_FINCH_CONFIG,
    CANARY_CONFIG,
    RESEARCH_CONFIGS,
    EVOLUTION_MODELS,
    MIN_ALPHABET_SIZE,
    MAX_ALPHABET_SIZE,
    RECOMMENDED_MAX_ALPHABET,
    MEMORY_PER_SONG_MB,
    SONGS_PER_GB,
    get_memory_estimate,
    validate_config
)


class TestDefaultConfig:
    """Test suite for the DefaultConfig dataclass."""
    
    def test_default_config_creation(self):
        """Test DefaultConfig creation with all parameters."""
        config = DefaultConfig(
            alphabet_size=20,
            n_songs=100,
            song_length_range=(15, 50),
            markov_order=2,
            evolution_model="sigmoid",
            max_em_iterations=30,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=25,
            memory_limit_gb=1.5,
            figure_dpi=200,
            export_formats=["pdf", "png"]
        )
        
        assert config.alphabet_size == 20
        assert config.n_songs == 100
        assert config.song_length_range == (15, 50)
        assert config.markov_order == 2
        assert config.evolution_model == "sigmoid"
        assert config.max_em_iterations == 30
        assert config.convergence_threshold == 1e-4
        assert config.regularization_lambda == 1e-3
        assert config.batch_size == 25
        assert config.memory_limit_gb == 1.5
        assert config.figure_dpi == 200
        assert config.export_formats == ["pdf", "png"]


class TestPresetConfigurations:
    """Test suite for preset configurations."""
    
    def test_bengalese_finch_config(self):
        """Test Bengalese Finch configuration values."""
        config = BENGALESE_FINCH_CONFIG
        
        # Check that it's a DefaultConfig instance
        assert isinstance(config, DefaultConfig)
        
        # Check specific values
        assert config.alphabet_size == 15
        assert config.n_songs == 300
        assert config.song_length_range == (20, 80)
        assert config.markov_order == 2
        assert config.evolution_model == "sigmoid"
        assert config.max_em_iterations == 50
        assert config.convergence_threshold == 1e-4
        assert config.regularization_lambda == 1e-3
        assert config.batch_size == 50
        assert config.memory_limit_gb == 1.0
        assert config.figure_dpi == 300
        assert config.export_formats == ["pdf", "png"]
    
    def test_canary_config(self):
        """Test Canary configuration values."""
        config = CANARY_CONFIG
        
        # Check that it's a DefaultConfig instance
        assert isinstance(config, DefaultConfig)
        
        # Check specific values for canary (larger dataset)
        assert config.alphabet_size == 40
        assert config.n_songs == 10000
        assert config.song_length_range == (30, 120)
        assert config.markov_order == 3
        assert config.evolution_model == "piecewise"
        assert config.max_em_iterations == 100
        assert config.convergence_threshold == 1e-5
        assert config.regularization_lambda == 1e-4
        assert config.batch_size == 500
        assert config.memory_limit_gb == 2.0
        assert config.figure_dpi == 300
        assert config.export_formats == ["pdf", "svg", "png"]
    
    def test_research_configs_dictionary(self):
        """Test RESEARCH_CONFIGS dictionary."""
        # Check that all expected presets are present
        assert "bengalese_finch" in RESEARCH_CONFIGS
        assert "canary" in RESEARCH_CONFIGS
        assert "minimal" in RESEARCH_CONFIGS
        
        # Check that they point to correct configurations
        assert RESEARCH_CONFIGS["bengalese_finch"] is BENGALESE_FINCH_CONFIG
        assert RESEARCH_CONFIGS["canary"] is CANARY_CONFIG
        
        # Check minimal configuration
        minimal_config = RESEARCH_CONFIGS["minimal"]
        assert isinstance(minimal_config, DefaultConfig)
        assert minimal_config.alphabet_size == 5
        assert minimal_config.n_songs == 50
        assert minimal_config.song_length_range == (10, 30)
        assert minimal_config.markov_order == 1
        assert minimal_config.evolution_model == "constant"
        assert minimal_config.max_em_iterations == 20
        assert minimal_config.convergence_threshold == 1e-3
        assert minimal_config.regularization_lambda == 1e-2
        assert minimal_config.batch_size == 10
        assert minimal_config.memory_limit_gb == 0.5
        assert minimal_config.figure_dpi == 150
        assert minimal_config.export_formats == ["png"]


class TestEvolutionModels:
    """Test suite for evolution model constants."""
    
    def test_evolution_models_list(self):
        """Test EVOLUTION_MODELS contains expected models."""
        expected_models = [
            "linear",
            "exponential", 
            "sigmoid",
            "piecewise",
            "oscillatory",
            "constant"
        ]
        
        assert EVOLUTION_MODELS == expected_models
        
        # Check that all models are strings
        for model in EVOLUTION_MODELS:
            assert isinstance(model, str)
            assert len(model) > 0
    
    def test_evolution_models_coverage(self):
        """Test that preset configs use valid evolution models."""
        # All preset configurations should use valid evolution models
        assert BENGALESE_FINCH_CONFIG.evolution_model in EVOLUTION_MODELS
        assert CANARY_CONFIG.evolution_model in EVOLUTION_MODELS
        assert RESEARCH_CONFIGS["minimal"].evolution_model in EVOLUTION_MODELS


class TestConstraints:
    """Test suite for constraint constants."""
    
    def test_alphabet_size_constraints(self):
        """Test alphabet size constraint values."""
        assert MIN_ALPHABET_SIZE == 3
        assert MAX_ALPHABET_SIZE == 100
        assert RECOMMENDED_MAX_ALPHABET == 50
        
        # Check logical relationships
        assert MIN_ALPHABET_SIZE < RECOMMENDED_MAX_ALPHABET
        assert RECOMMENDED_MAX_ALPHABET <= MAX_ALPHABET_SIZE
    
    def test_memory_constants(self):
        """Test memory-related constants."""
        assert MEMORY_PER_SONG_MB == 0.1
        assert SONGS_PER_GB == 10000
        
        # Check that constants are positive
        assert MEMORY_PER_SONG_MB > 0
        assert SONGS_PER_GB > 0
    
    def test_preset_configs_within_constraints(self):
        """Test that preset configurations respect constraints."""
        for config_name, config in RESEARCH_CONFIGS.items():
            # All alphabet sizes should be within bounds
            assert config.alphabet_size >= MIN_ALPHABET_SIZE, f"{config_name} alphabet too small"
            assert config.alphabet_size <= MAX_ALPHABET_SIZE, f"{config_name} alphabet too large"
            
            # All evolution models should be valid
            assert config.evolution_model in EVOLUTION_MODELS, f"{config_name} invalid evolution model"
            
            # Basic sanity checks
            assert config.n_songs > 0, f"{config_name} must have positive songs"
            assert config.batch_size > 0, f"{config_name} must have positive batch size"
            assert config.memory_limit_gb > 0, f"{config_name} must have positive memory limit"


class TestMemoryEstimation:
    """Test suite for memory estimation functionality."""
    
    def test_get_memory_estimate_basic(self):
        """Test basic memory estimation."""
        config = DefaultConfig(
            alphabet_size=10,
            n_songs=100,
            song_length_range=(20, 50),
            markov_order=1,
            evolution_model="linear",
            max_em_iterations=30,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=20,
            memory_limit_gb=1.0,
            figure_dpi=300,
            export_formats=["png"]
        )
        
        memory_estimate = get_memory_estimate(config)
        
        # Should return a positive float
        assert isinstance(memory_estimate, float)
        assert memory_estimate > 0
        
        # Should be reasonable for small config
        assert memory_estimate < 1.0  # Should be less than 1GB for small config
    
    def test_get_memory_estimate_scaling(self):
        """Test that memory estimate scales correctly."""
        # Small configuration
        small_config = DefaultConfig(
            alphabet_size=5,
            n_songs=50,
            song_length_range=(10, 20),
            markov_order=1,
            evolution_model="linear",
            max_em_iterations=20,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=10,
            memory_limit_gb=0.5,
            figure_dpi=150,
            export_formats=["png"]
        )
        
        # Large configuration
        large_config = DefaultConfig(
            alphabet_size=20,
            n_songs=1000,
            song_length_range=(30, 100),
            markov_order=3,
            evolution_model="sigmoid",
            max_em_iterations=50,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=100,
            memory_limit_gb=5.0,
            figure_dpi=300,
            export_formats=["pdf", "png"]
        )
        
        small_memory = get_memory_estimate(small_config)
        large_memory = get_memory_estimate(large_config)
        
        # Large config should require more memory
        assert large_memory > small_memory
    
    def test_get_memory_estimate_preset_configs(self):
        """Test memory estimation for preset configurations."""
        # Test all preset configurations
        bengalese_memory = get_memory_estimate(BENGALESE_FINCH_CONFIG)
        canary_memory = get_memory_estimate(CANARY_CONFIG)
        minimal_memory = get_memory_estimate(RESEARCH_CONFIGS["minimal"])
        
        # All should be positive
        assert bengalese_memory > 0
        assert canary_memory > 0
        assert minimal_memory > 0
        
        # Canary should require most memory (largest dataset)
        assert canary_memory > bengalese_memory
        assert bengalese_memory > minimal_memory


class TestValidation:
    """Test suite for configuration validation."""
    
    def test_validate_config_valid_configuration(self):
        """Test validation with valid configuration."""
        valid_config = DefaultConfig(
            alphabet_size=15,
            n_songs=200,
            song_length_range=(20, 60),
            markov_order=2,
            evolution_model="sigmoid",
            max_em_iterations=40,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=30,
            memory_limit_gb=2.0,
            figure_dpi=300,
            export_formats=["pdf", "png"]
        )
        
        warnings = validate_config(valid_config)
        
        # Should return a list (may be empty for valid config)
        assert isinstance(warnings, list)
        
        # For this reasonable configuration, should have no warnings
        assert len(warnings) == 0
    
    def test_validate_config_alphabet_size_too_small(self):
        """Test validation with alphabet size below minimum."""
        invalid_config = DefaultConfig(
            alphabet_size=2,  # Below MIN_ALPHABET_SIZE (3)
            n_songs=100,
            song_length_range=(20, 50),
            markov_order=1,
            evolution_model="linear",
            max_em_iterations=30,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=20,
            memory_limit_gb=1.0,
            figure_dpi=300,
            export_formats=["png"]
        )
        
        warnings = validate_config(invalid_config)
        
        assert len(warnings) > 0
        assert any("below minimum" in warning for warning in warnings)
    
    def test_validate_config_alphabet_size_too_large(self):
        """Test validation with alphabet size above recommended maximum."""
        large_config = DefaultConfig(
            alphabet_size=60,  # Above RECOMMENDED_MAX_ALPHABET (50)
            n_songs=100,
            song_length_range=(20, 50),
            markov_order=1,
            evolution_model="linear",
            max_em_iterations=30,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=20,
            memory_limit_gb=1.0,
            figure_dpi=300,
            export_formats=["png"]
        )
        
        warnings = validate_config(large_config)
        
        assert len(warnings) > 0
        assert any("performance issues" in warning for warning in warnings)
    
    def test_validate_config_high_markov_order(self):
        """Test validation with very high Markov order."""
        high_order_config = DefaultConfig(
            alphabet_size=15,
            n_songs=100,
            song_length_range=(20, 50),
            markov_order=6,  # Above 5
            evolution_model="linear",
            max_em_iterations=30,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=20,
            memory_limit_gb=1.0,
            figure_dpi=300,
            export_formats=["png"]
        )
        
        warnings = validate_config(high_order_config)
        
        assert len(warnings) > 0
        assert any("very high" in warning for warning in warnings)
    
    def test_validate_config_invalid_evolution_model(self):
        """Test validation with invalid evolution model."""
        invalid_model_config = DefaultConfig(
            alphabet_size=15,
            n_songs=100,
            song_length_range=(20, 50),
            markov_order=2,
            evolution_model="invalid_model",  # Not in EVOLUTION_MODELS
            max_em_iterations=30,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=20,
            memory_limit_gb=1.0,
            figure_dpi=300,
            export_formats=["png"]
        )
        
        warnings = validate_config(invalid_model_config)
        
        assert len(warnings) > 0
        assert any("not recognized" in warning for warning in warnings)
    
    def test_validate_config_memory_exceeded(self):
        """Test validation when estimated memory exceeds limit."""
        # Create configuration that will definitely exceed memory limit
        memory_config = DefaultConfig(
            alphabet_size=50,  # Large alphabet
            n_songs=50000,     # Many songs
            song_length_range=(50, 200),
            markov_order=4,    # High order = huge state space
            evolution_model="linear",
            max_em_iterations=50,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=100,
            memory_limit_gb=0.1,  # Very low limit
            figure_dpi=300,
            export_formats=["png"]
        )
        
        warnings = validate_config(memory_config)
        
        assert len(warnings) > 0
        assert any("exceeds limit" in warning for warning in warnings)
    
    def test_validate_config_multiple_warnings(self):
        """Test validation with multiple issues."""
        problematic_config = DefaultConfig(
            alphabet_size=2,     # Too small
            n_songs=100,
            song_length_range=(20, 50),
            markov_order=7,      # Too high
            evolution_model="bad_model",  # Invalid
            max_em_iterations=30,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=20,
            memory_limit_gb=1.0,
            figure_dpi=300,
            export_formats=["png"]
        )
        
        warnings = validate_config(problematic_config)
        
        # Should have multiple warnings
        assert len(warnings) >= 3
        
        # Check for specific warning types
        warning_text = " ".join(warnings)
        assert "below minimum" in warning_text
        assert "very high" in warning_text
        assert "not recognized" in warning_text
    
    def test_validate_config_preset_configurations(self):
        """Test validation of all preset configurations."""
        # All preset configurations should be valid (no warnings or acceptable warnings)
        for config_name, config in RESEARCH_CONFIGS.items():
            warnings = validate_config(config)
            
            # Print warnings for debugging if any
            if warnings:
                print(f"Warnings for {config_name}: {warnings}")
            
            # For preset configurations, we expect them to be well-designed
            # Canary might have performance warnings due to large alphabet size
            if config_name == "canary":
                # Canary might have alphabet size warning, but should be manageable
                assert len(warnings) <= 1
            else:
                # Other presets should have no warnings
                assert len(warnings) == 0


class TestIntegration:
    """Integration tests for defaults functionality."""
    
    def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow."""
        # Create a configuration
        config = DefaultConfig(
            alphabet_size=20,
            n_songs=500,
            song_length_range=(25, 75),
            markov_order=2,
            evolution_model="sigmoid",
            max_em_iterations=50,
            convergence_threshold=1e-4,
            regularization_lambda=1e-3,
            batch_size=50,
            memory_limit_gb=3.0,
            figure_dpi=300,
            export_formats=["pdf", "png"]
        )
        
        # Estimate memory
        memory_estimate = get_memory_estimate(config)
        
        # Validate configuration
        warnings = validate_config(config)
        
        # Results should be reasonable
        assert memory_estimate > 0
        assert memory_estimate < config.memory_limit_gb  # Should fit in memory limit
        assert isinstance(warnings, list)
        assert len(warnings) == 0  # Should be a valid configuration
    
    def test_all_evolution_models_are_testable(self):
        """Test that we can create configurations with all evolution models."""
        base_config_params = {
            "alphabet_size": 10,
            "n_songs": 100,
            "song_length_range": (20, 50),
            "markov_order": 1,
            "max_em_iterations": 30,
            "convergence_threshold": 1e-4,
            "regularization_lambda": 1e-3,
            "batch_size": 20,
            "memory_limit_gb": 1.0,
            "figure_dpi": 300,
            "export_formats": ["png"]
        }
        
        for evolution_model in EVOLUTION_MODELS:
            config = DefaultConfig(
                evolution_model=evolution_model,
                **base_config_params
            )
            
            # Should create successfully
            assert config.evolution_model == evolution_model
            
            # Should validate without evolution model warnings
            warnings = validate_config(config)
            evolution_warnings = [w for w in warnings if "not recognized" in w]
            assert len(evolution_warnings) == 0 