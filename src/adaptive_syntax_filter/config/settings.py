"""Main configuration settings with TOML loading support."""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import os

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python
    except ImportError:
        tomllib = None

try:
    import tomli_w
except ImportError:
    tomli_w = None

from .defaults import BENGALESE_FINCH_CONFIG, CANARY_CONFIG, RESEARCH_CONFIGS, validate_config

@dataclass
class Settings:
    """Main configuration settings for Adaptive Syntax Filter.
    
    Can be loaded from TOML files for user customization while providing
    sensible defaults for different research scenarios.
    """
    
    # Data scale parameters
    alphabet_size: int = 15
    n_songs: int = 300
    song_length_range: Tuple[int, int] = (20, 80)
    
    # Model parameters
    markov_order: int = 2
    evolution_model: str = "sigmoid"
    
    # Algorithm parameters
    max_em_iterations: int = 50
    convergence_threshold: float = 1e-4
    regularization_lambda: float = 1e-3
    
    # Processing parameters
    batch_size: int = 50
    memory_limit_gb: float = 1.0
    n_parallel: int = 1
    
    # Visualization parameters
    figure_dpi: int = 300
    export_formats: List[str] = field(default_factory=lambda: ["pdf", "png"])
    output_dir: str = "output"
    
    # Reproducibility
    random_seed: Optional[int] = None
    
    # Advanced settings
    numerical_precision: str = "float64"
    cache_enabled: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        """Validate settings after initialization."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        from .defaults import DefaultConfig
        temp_config = DefaultConfig(
            alphabet_size=self.alphabet_size,
            n_songs=self.n_songs,
            song_length_range=self.song_length_range,
            markov_order=self.markov_order,
            evolution_model=self.evolution_model,
            max_em_iterations=self.max_em_iterations,
            convergence_threshold=self.convergence_threshold,
            regularization_lambda=self.regularization_lambda,
            batch_size=self.batch_size,
            memory_limit_gb=self.memory_limit_gb,
            figure_dpi=self.figure_dpi,
            export_formats=self.export_formats
        )
        
        warnings = validate_config(temp_config)
        if warnings and self.verbose:
            for warning in warnings:
                print(f"Configuration warning: {warning}")
    
    @classmethod
    def from_preset(cls, preset: str) -> 'Settings':
        """Create settings from a preset configuration.
        
        Parameters
        ----------
        preset : str
            Preset name ('bengalese_finch', 'canary', 'minimal')
            
        Returns
        -------
        Settings
            Settings object with preset values
        """
        if preset not in RESEARCH_CONFIGS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(RESEARCH_CONFIGS.keys())}")
        
        config = RESEARCH_CONFIGS[preset]
        return cls(
            alphabet_size=config.alphabet_size,
            n_songs=config.n_songs,
            song_length_range=config.song_length_range,
            markov_order=config.markov_order,
            evolution_model=config.evolution_model,
            max_em_iterations=config.max_em_iterations,
            convergence_threshold=config.convergence_threshold,
            regularization_lambda=config.regularization_lambda,
            batch_size=config.batch_size,
            memory_limit_gb=config.memory_limit_gb,
            figure_dpi=config.figure_dpi,
            export_formats=config.export_formats.copy()
        )
    
    @classmethod
    def from_toml(cls, toml_path: Union[str, Path]) -> 'Settings':
        """Load settings from TOML file.
        
        Parameters
        ----------
        toml_path : Union[str, Path]
            Path to TOML configuration file
            
        Returns
        -------
        Settings
            Settings object with values from TOML file
            
        Raises
        ------
        ImportError
            If tomllib is not available
        FileNotFoundError
            If TOML file doesn't exist
        """
        if tomllib is None:
            raise ImportError("tomllib not available. Install tomli for Python < 3.11")
        
        toml_path = Path(toml_path)
        if not toml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {toml_path}")
        
        with open(toml_path, 'rb') as f:
            config_data = tomllib.load(f)
        
        # Extract relevant sections and flatten
        settings_data = {}
        
        # Handle nested configuration structure
        if 'data' in config_data:
            settings_data.update(config_data['data'])
        if 'model' in config_data:
            settings_data.update(config_data['model'])
        if 'algorithm' in config_data:
            settings_data.update(config_data['algorithm'])
        if 'processing' in config_data:
            settings_data.update(config_data['processing'])
        if 'visualization' in config_data:
            settings_data.update(config_data['visualization'])
        if 'advanced' in config_data:
            settings_data.update(config_data['advanced'])
        
        # Also handle flat structure
        for key, value in config_data.items():
            if not isinstance(value, dict):
                settings_data[key] = value
        
        return cls(**settings_data)
    
    def to_toml(self, toml_path: Union[str, Path]) -> None:
        """Save settings to TOML file.
        
        Parameters
        ----------
        toml_path : Union[str, Path]
            Path where to save TOML configuration file
            
        Raises
        ------
        ImportError
            If tomli_w is not available
        """
        if tomli_w is None:
            raise ImportError("tomli_w not available. Install tomli-w for TOML writing")
        
        # Organize settings into logical sections
        config_data = {
            'data': {
                'alphabet_size': self.alphabet_size,
                'n_songs': self.n_songs,
                'song_length_range': list(self.song_length_range)
            },
            'model': {
                'markov_order': self.markov_order,
                'evolution_model': self.evolution_model
            },
            'algorithm': {
                'max_em_iterations': self.max_em_iterations,
                'convergence_threshold': self.convergence_threshold,
                'regularization_lambda': self.regularization_lambda
            },
            'processing': {
                'batch_size': self.batch_size,
                'memory_limit_gb': self.memory_limit_gb,
                'n_parallel': self.n_parallel
            },
            'visualization': {
                'figure_dpi': self.figure_dpi,
                'export_formats': self.export_formats,
                'output_dir': self.output_dir
            },
            'advanced': {
                'random_seed': self.random_seed,
                'numerical_precision': self.numerical_precision,
                'cache_enabled': self.cache_enabled,
                'verbose': self.verbose
            }
        }
        
        toml_path = Path(toml_path)
        with open(toml_path, 'wb') as f:
            tomli_w.dump(config_data, f)
    
    def update(self, **kwargs) -> 'Settings':
        """Create new Settings with updated values.
        
        Parameters
        ----------
        **kwargs
            Settings fields to update
            
        Returns
        -------
        Settings
            New Settings object with updated values
        """
        current_dict = asdict(self)
        current_dict.update(kwargs)
        return Settings(**current_dict)


# Global configuration instance
_GLOBAL_CONFIG: Optional[Settings] = None

def get_config(config_path: Optional[Union[str, Path]] = None, 
               preset: Optional[str] = None,
               reload: bool = False) -> Settings:
    """Get global configuration settings.
    
    Parameters
    ----------
    config_path : Optional[Union[str, Path]]
        Path to TOML configuration file. If None, looks for default locations.
    preset : Optional[str]
        Preset configuration name ('bengalese_finch', 'canary', 'minimal').
        Ignored if config_path is provided.
    reload : bool
        Force reload configuration even if already loaded
        
    Returns
    -------
    Settings
        Global configuration settings
    """
    global _GLOBAL_CONFIG
    
    if _GLOBAL_CONFIG is not None and not reload:
        return _GLOBAL_CONFIG
    
    # Try to load from TOML file
    if config_path is not None:
        _GLOBAL_CONFIG = Settings.from_toml(config_path)
    else:
        # Look for default configuration files
        default_paths = [
            'config.toml',
            'adaptive_syntax_filter.toml',
            Path.home() / '.adaptive_syntax_filter.toml',
            Path.cwd() / 'config' / 'config.toml'
        ]
        
        config_loaded = False
        for path in default_paths:
            if Path(path).exists():
                try:
                    _GLOBAL_CONFIG = Settings.from_toml(path)
                    config_loaded = True
                    break
                except Exception as e:
                    if preset is None:  # Only warn if not using preset fallback
                        print(f"Warning: Could not load config from {path}: {e}")
                    continue
        
        # Fall back to preset or defaults
        if not config_loaded:
            if preset is not None:
                _GLOBAL_CONFIG = Settings.from_preset(preset)
            else:
                # Use bengalese finch as default
                _GLOBAL_CONFIG = Settings.from_preset('bengalese_finch')
    
    # Set random seed if specified
    if _GLOBAL_CONFIG.random_seed is not None:
        from .random_state import set_global_seed
        set_global_seed(_GLOBAL_CONFIG.random_seed)
    
    return _GLOBAL_CONFIG

def set_config(settings: Settings) -> None:
    """Set global configuration settings.
    
    Parameters
    ----------
    settings : Settings
        Settings object to use as global configuration
    """
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = settings
    
    # Set random seed if specified
    if settings.random_seed is not None:
        from .random_state import set_global_seed
        set_global_seed(settings.random_seed) 
