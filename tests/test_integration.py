"""
Integration tests for the complete Adaptive Syntax Filter pipeline.
"""

import pytest
import numpy as np
from adaptive_syntax_filter.config import set_global_seed
from adaptive_syntax_filter.data import SequenceGenerator, GenerationConfig, generate_dataset
from adaptive_syntax_filter.viz import LogitEvolutionDashboard


class TestBasicIntegration:
    """Basic integration tests."""
    
    @pytest.mark.integration
    def test_data_to_visualization_pipeline(self, small_alphabet, global_test_seed):
        """Test basic data generation to visualization pipeline."""
        
        # Generate synthetic data
        config = GenerationConfig(
            alphabet=small_alphabet,
            order=1,
            n_sequences=10,
            seed=global_test_seed
        )
        
        sequences, _ = generate_dataset(config)
        
        # Create simple logit data for visualization test
        state_dim = len(small_alphabet) ** 2
        logits_test = np.random.randn(state_dim, 10)
        
        # Test visualization
        dashboard = LogitEvolutionDashboard(alphabet=small_alphabet, markov_order=1)
        fig = dashboard.plot_logit_evolution_static(logits_test)
        
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        print("✅ Basic integration test passed")
    
    @pytest.mark.integration
    def test_reproducibility(self, small_alphabet):
        """Test pipeline reproducibility."""
        
        def run_pipeline(seed):
            set_global_seed(seed)
            config = GenerationConfig(
                alphabet=small_alphabet,
                order=1,
                n_sequences=5,
                seed=seed
            )
            sequences, _ = generate_dataset(config)
            return sequences
        
        # Run twice with same seed
        seq1 = run_pipeline(42)
        seq2 = run_pipeline(42)
        
        assert seq1 == seq2
        print("✅ Reproducibility test passed") 
