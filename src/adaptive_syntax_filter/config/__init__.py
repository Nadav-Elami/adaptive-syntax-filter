"""Configuration management for Adaptive Syntax Filter.

Provides global configuration and random seed management for reproducible research.
"""

from .settings import get_config, Settings
from .random_state import set_global_seed, get_random_state
from .defaults import BENGALESE_FINCH_CONFIG, CANARY_CONFIG, RESEARCH_CONFIGS, EVOLUTION_MODELS, DefaultConfig

__all__ = [
    'get_config',
    'set_global_seed', 
    'get_random_state',
    'Settings',
    'BENGALESE_FINCH_CONFIG',
    'CANARY_CONFIG',
    'RESEARCH_CONFIGS',
    'EVOLUTION_MODELS',
    'DefaultConfig'
] 
