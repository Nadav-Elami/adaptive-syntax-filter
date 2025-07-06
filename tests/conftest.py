"""
Pytest configuration and shared fixtures for the Adaptive Syntax Filter test suite.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from adaptive_syntax_filter.config import set_global_seed


@pytest.fixture(scope="session")
def global_test_seed():
    """Set global random seed for all tests to ensure reproducibility."""
    seed = 42
    set_global_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def small_alphabet():
    """Small alphabet for fast testing."""
    return ['<', 'A', 'B', '>']


@pytest.fixture  
def medium_alphabet():
    """Medium alphabet for standard testing."""
    return ['<', 'A', 'B', 'C', 'D', 'E', '>']


@pytest.fixture
def large_alphabet():
    """Large alphabet for stress testing."""
    return ['<'] + [f'S{i:02d}' for i in range(15)] + ['>']


@pytest.fixture
def test_sequences_small():
    """Small test sequences for quick testing."""
    return [
        ['<', 'A', 'B', '>'],
        ['<', 'B', 'A', '>'],
        ['<', 'A', 'A', 'B', '>'],
    ]


@pytest.fixture
def test_sequences_medium():
    """Medium test sequences for standard testing."""
    sequences = []
    np.random.seed(42)
    
    for _ in range(20):
        length = np.random.randint(3, 8)
        seq = ['<']
        for _ in range(length - 2):
            seq.append(np.random.choice(['A', 'B', 'C', 'D']))
        seq.append('>')
        sequences.append(seq)
    
    return sequences


@pytest.fixture
def test_logits_small():
    """Small test logit matrices."""
    np.random.seed(42)
    return np.random.randn(16, 10)  # 4^2 states, 10 sequences


@pytest.fixture
def test_probabilities_small():
    """Small test probability matrices."""
    np.random.seed(42)
    probs = np.random.rand(16, 10)
    # Normalize each block
    n_symbols = 4
    for i in range(0, 16, n_symbols):
        block_sum = probs[i:i+n_symbols, :].sum(axis=0)
        probs[i:i+n_symbols, :] = probs[i:i+n_symbols, :] / block_sum
    return probs


@pytest.fixture
def synthetic_data_config():
    """Configuration for synthetic data generation."""
    return {
        'alphabet_size': 5,
        'markov_order': 2,
        'n_sequences': 50,
        'sequence_length_range': (4, 10),
        'evolution_model': 'linear',
        'noise_level': 0.1
    }


@pytest.fixture
def performance_test_config():
    """Configuration for performance testing."""
    return {
        'memory_limit_mb': 500,  # 500MB limit for tests
        'time_limit_seconds': 30,  # 30 second limit for tests
        'large_dataset_size': 1000,  # 1000 sequences for performance tests
        'batch_size': 100,  # 100 sequences for batch performance tests
    }


class TestDataGenerator:
    """Helper class for generating test data."""
    
    @staticmethod
    def create_simple_logits(n_symbols: int, n_sequences: int, 
                           markov_order: int = 1, seed: int = 42) -> np.ndarray:
        """Create simple test logit data."""
        np.random.seed(seed)
        state_space_size = (n_symbols ** markov_order) * n_symbols
        logits = np.random.randn(state_space_size, n_sequences)
        
        # Apply constraints (set forbidden transitions to -inf)
        for i in range(0, state_space_size, n_symbols):
            logits[i, :] = -np.inf
            
        return logits
    
    @staticmethod
    def create_test_sequences(alphabet: list, n_sequences: int, 
                            length_range: tuple = (3, 8), seed: int = 42) -> list:
        """Create test sequences."""
        np.random.seed(seed)
        sequences = []
        
        content_symbols = [s for s in alphabet if s not in ['<', '>']]
        
        for _ in range(n_sequences):
            length = np.random.randint(*length_range)
            seq = ['<']
            for _ in range(length - 2):
                seq.append(np.random.choice(content_symbols))
            seq.append('>')
            sequences.append(seq)
            
        return sequences


@pytest.fixture
def test_data_generator():
    """Test data generator fixture."""
    return TestDataGenerator


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "visual: marks tests that generate visual output"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow and integration tests."""
    for item in items:
        # Mark slow tests
        if "performance" in item.nodeid or "large" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.integration) 
