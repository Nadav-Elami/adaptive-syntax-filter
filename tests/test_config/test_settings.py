"""Tests for configuration settings functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.adaptive_syntax_filter.config.settings import (
    Settings,
    get_config,
    set_config
)


class TestSettingsDataclass:
    """Test suite for the Settings dataclass."""
    
    def test_settings_default_initialization(self):
        """Test Settings initialization with default values."""
        settings = Settings()
        
        # Check default values
        assert settings.alphabet_size == 15
        assert settings.n_songs == 300
        assert settings.song_length_range == (20, 80)
        assert settings.markov_order == 2
        assert settings.evolution_model == "sigmoid"
        assert settings.max_em_iterations == 50
        assert settings.convergence_threshold == 1e-4
        assert settings.regularization_lambda == 1e-3
        assert settings.batch_size == 50
        assert settings.memory_limit_gb == 1.0
        assert settings.n_parallel == 1
        assert settings.figure_dpi == 300
        assert settings.export_formats == ["pdf", "png"]
        assert settings.output_dir == "output"
        assert settings.random_seed is None
        assert settings.numerical_precision == "float64"
        assert settings.cache_enabled is True
        assert settings.verbose is False
    
    def test_settings_custom_initialization(self):
        """Test Settings initialization with custom values."""
        settings = Settings(
            alphabet_size=20,
            n_songs=500,
            song_length_range=(10, 100),
            markov_order=3,
            evolution_model="linear",
            random_seed=42,
            verbose=True
        )
        
        # Check custom values
        assert settings.alphabet_size == 20
        assert settings.n_songs == 500
        assert settings.song_length_range == (10, 100)
        assert settings.markov_order == 3
        assert settings.evolution_model == "linear"
        assert settings.random_seed == 42
        assert settings.verbose is True
        
        # Check defaults are preserved
        assert settings.max_em_iterations == 50
        assert settings.convergence_threshold == 1e-4
    
    def test_settings_post_init_output_dir_creation(self):
        """Test that __post_init__ creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output"
            settings = Settings(output_dir=str(output_path))
            
            # Directory should be created
            assert output_path.exists()
            assert output_path.is_dir()


class TestSettingsFromPreset:
    """Test suite for Settings.from_preset() method."""
    
    def test_from_preset_bengalese_finch(self):
        """Test loading bengalese finch preset."""
        settings = Settings.from_preset("bengalese_finch")
        
        # Should be a Settings instance
        assert isinstance(settings, Settings)
        assert settings.alphabet_size > 0
        assert settings.n_songs > 0
        assert settings.song_length_range[0] > 0
        assert settings.song_length_range[1] > settings.song_length_range[0]
        assert settings.markov_order > 0
        assert settings.evolution_model in ["sigmoid", "linear", "exponential"]
    
    def test_from_preset_canary(self):
        """Test loading canary preset."""
        settings = Settings.from_preset("canary")
        
        # Should be a Settings instance
        assert isinstance(settings, Settings)
        assert settings.alphabet_size > 0
        assert settings.n_songs > 0
        assert isinstance(settings.evolution_model, str)
    
    def test_from_preset_minimal(self):
        """Test loading minimal preset."""
        settings = Settings.from_preset("minimal")
        
        # Should be a Settings instance
        assert isinstance(settings, Settings)
        assert settings.alphabet_size > 0
        assert settings.n_songs > 0
    
    def test_from_preset_invalid_preset(self):
        """Test error handling for invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset 'invalid'"):
            Settings.from_preset("invalid")
    
    def test_from_preset_list_available_presets(self):
        """Test that error message lists available presets."""
        try:
            Settings.from_preset("nonexistent")
        except ValueError as e:
            error_msg = str(e)
            # Should mention available presets
            assert "bengalese_finch" in error_msg or "canary" in error_msg or "minimal" in error_msg


class TestSettingsFromToml:
    """Test suite for Settings.from_toml() method."""
    
    def test_from_toml_missing_tomllib(self):
        """Test error when tomllib is not available."""
        with patch('src.adaptive_syntax_filter.config.settings.tomllib', None):
            with pytest.raises(ImportError, match="tomllib not available"):
                Settings.from_toml("dummy.toml")
    
    def test_from_toml_file_not_found(self):
        """Test error when TOML file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            Settings.from_toml("nonexistent.toml")
    
    def test_from_toml_nested_structure(self):
        """Test loading TOML with nested structure."""
        toml_content = '''
        [data]
        alphabet_size = 20
        n_songs = 400
        song_length_range = [30, 90]
        
        [model]
        markov_order = 3
        evolution_model = "linear"
        
        [algorithm]
        max_em_iterations = 100
        convergence_threshold = 1e-5
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_path = f.name
        
        try:
            settings = Settings.from_toml(temp_path)
            
            # Check loaded values
            assert settings.alphabet_size == 20
            assert settings.n_songs == 400
            assert settings.song_length_range == [30, 90]  # TOML arrays become lists
            assert settings.markov_order == 3
            assert settings.evolution_model == "linear"
            assert settings.max_em_iterations == 100
            assert settings.convergence_threshold == 1e-5
            
            # Check defaults are preserved
            assert settings.batch_size == 50  # Not in TOML, should use default
            
        finally:
            Path(temp_path).unlink()
    
    def test_from_toml_flat_structure(self):
        """Test loading TOML with flat structure."""
        toml_content = '''
        alphabet_size = 25
        n_songs = 200
        markov_order = 1
        evolution_model = "exponential"
        verbose = true
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_path = f.name
        
        try:
            settings = Settings.from_toml(temp_path)
            
            # Check loaded values
            assert settings.alphabet_size == 25
            assert settings.n_songs == 200
            assert settings.markov_order == 1
            assert settings.evolution_model == "exponential"
            assert settings.verbose is True
            
        finally:
            Path(temp_path).unlink()


class TestSettingsToToml:
    """Test suite for Settings.to_toml() method."""
    
    def test_to_toml_missing_tomli_w(self):
        """Test error when tomli_w is not available."""
        with patch('src.adaptive_syntax_filter.config.settings.tomli_w', None):
            settings = Settings()
            with pytest.raises(ImportError, match="tomli_w not available"):
                settings.to_toml("dummy.toml")
    
    def test_to_toml_creates_file(self):
        """Test that to_toml creates a valid TOML file."""
        settings = Settings(
            alphabet_size=20,
            n_songs=400,
            markov_order=3,
            evolution_model="linear",
            verbose=True
        )
        
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
            temp_path = f.name
        
        try:
            settings.to_toml(temp_path)
            
            # File should exist
            assert Path(temp_path).exists()
            
            # Should be loadable back
            loaded_settings = Settings.from_toml(temp_path)
            
            # Key values should match
            assert loaded_settings.alphabet_size == 20
            assert loaded_settings.n_songs == 400
            assert loaded_settings.markov_order == 3
            assert loaded_settings.evolution_model == "linear"
            assert loaded_settings.verbose is True
            
        finally:
            Path(temp_path).unlink()


class TestSettingsUpdate:
    """Test suite for Settings.update() method."""
    
    def test_update_single_field(self):
        """Test updating a single field."""
        original = Settings(alphabet_size=15)
        updated = original.update(alphabet_size=20)
        
        # Original should be unchanged
        assert original.alphabet_size == 15
        
        # Updated should have new value
        assert updated.alphabet_size == 20
        
        # Other fields should be the same
        assert updated.n_songs == original.n_songs
        assert updated.markov_order == original.markov_order
    
    def test_update_multiple_fields(self):
        """Test updating multiple fields."""
        original = Settings()
        updated = original.update(
            alphabet_size=25,
            n_songs=500,
            verbose=True,
            random_seed=42
        )
        
        # Original should be unchanged
        assert original.alphabet_size == 15
        assert original.n_songs == 300
        assert original.verbose is False
        assert original.random_seed is None
        
        # Updated should have new values
        assert updated.alphabet_size == 25
        assert updated.n_songs == 500
        assert updated.verbose is True
        assert updated.random_seed == 42
        
        # Other fields should be preserved
        assert updated.markov_order == original.markov_order
        assert updated.evolution_model == original.evolution_model
    
    def test_update_returns_new_instance(self):
        """Test that update returns a new Settings instance."""
        original = Settings()
        updated = original.update(alphabet_size=20)
        
        # Should be different objects
        assert original is not updated
        assert isinstance(updated, Settings)
    
    def test_update_with_invalid_field(self):
        """Test update with invalid field name raises error."""
        settings = Settings()
        
        # This should raise TypeError due to unexpected keyword argument
        with pytest.raises(TypeError):
            settings.update(nonexistent_field=123)


class TestGlobalConfigManagement:
    """Test suite for global configuration management."""
    
    def setup_method(self):
        """Reset global config before each test."""
        import src.adaptive_syntax_filter.config.settings as settings_module
        settings_module._GLOBAL_CONFIG = None
    
    def test_get_config_default_preset(self):
        """Test get_config() with default preset when no files exist."""
        # Mock all default paths to not exist
        with patch("pathlib.Path.exists", return_value=False):
            config = get_config()
            
            assert isinstance(config, Settings)
            assert config.alphabet_size > 0
    
    def test_get_config_with_preset(self):
        """Test get_config() with specific preset."""
        config = get_config(preset="canary")
        
        assert isinstance(config, Settings)
        # Should use canary preset
        assert config.alphabet_size > 0
    
    def test_get_config_caching(self):
        """Test that get_config() caches the result."""
        config1 = get_config(preset="minimal")
        config2 = get_config(preset="canary")  # Should return cached, ignore preset
        
        # Should be the same object (cached)
        assert config1 is config2
    
    def test_get_config_reload(self):
        """Test get_config() with reload=True."""
        config1 = get_config(preset="minimal")
        config2 = get_config(preset="canary", reload=True)  # Should reload
        
        # Should be different objects (reloaded)
        assert config1 is not config2
    
    def test_set_config(self):
        """Test set_config() function."""
        custom_settings = Settings(alphabet_size=50, verbose=True)
        set_config(custom_settings)
        
        # Should be retrievable via get_config
        retrieved = get_config()
        assert retrieved is custom_settings
        assert retrieved.alphabet_size == 50
        assert retrieved.verbose is True


class TestIntegrationScenarios:
    """Integration tests for Settings functionality."""
    
    def test_preset_to_toml_to_settings_pipeline(self):
        """Test full pipeline: preset -> TOML -> Settings."""
        # Start with preset
        original = Settings.from_preset("bengalese_finch")
        
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to TOML
            original.to_toml(temp_path)
            
            # Load from TOML
            loaded = Settings.from_toml(temp_path)
            
            # Update some values
            updated = loaded.update(verbose=True, random_seed=42)
            
            # All should be valid Settings objects
            assert isinstance(original, Settings)
            assert isinstance(loaded, Settings)
            assert isinstance(updated, Settings)
            
            # Key values should be preserved through pipeline
            assert loaded.alphabet_size == original.alphabet_size
            assert loaded.n_songs == original.n_songs
            assert loaded.evolution_model == original.evolution_model
            
            # Updates should be applied
            assert updated.verbose is True
            assert updated.random_seed == 42
            assert updated.alphabet_size == original.alphabet_size  # Preserved
            
        finally:
            Path(temp_path).unlink()
    
    def test_settings_with_path_objects(self):
        """Test Settings methods work with Path objects."""
        settings = Settings()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / "test_config.toml"
            
            # Should work with Path objects
            settings.to_toml(toml_path)
            assert toml_path.exists()
            
            loaded = Settings.from_toml(toml_path)
            assert isinstance(loaded, Settings) 