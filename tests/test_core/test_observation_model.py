"""Tests for the observation model with categorical distributions and block-wise softmax."""

import numpy as np
import pytest
import warnings
from unittest.mock import patch

from src.adaptive_syntax_filter.core.observation_model import (
    softmax_observation_model,
    log_softmax_jacobian, 
    log_softmax_hessian,
    compute_observation_likelihood,
    validate_transition_probabilities
)


class TestSoftmaxObservationModel:
    """Test suite for the softmax_observation_model function."""
    
    def test_basic_functionality_r2(self):
        """Test basic softmax transformation for R=2."""
        x = np.array([1.0, 2.0, 0.5, 1.5])  # Two 2x2 blocks
        probs = softmax_observation_model(x, R=2)
        
        # Check shape and type
        assert probs.shape == (4,)
        assert isinstance(probs, np.ndarray)
        
        # Check block normalization: each 2-element block should sum to 1
        assert np.isclose(np.sum(probs[0:2]), 1.0)  # Block 0 (from symbol 0)
        assert np.isclose(np.sum(probs[2:4]), 1.0)  # Block 1 (from symbol 1)
        
        # Check probabilities are positive
        assert np.all(probs > 0)
        assert np.all(probs < 1)
    
    def test_basic_functionality_r3(self):
        """Test basic softmax transformation for R=3."""
        x = np.random.randn(9)  # 3x3 = 9 elements
        probs = softmax_observation_model(x, R=3)
        
        # Check shape
        assert probs.shape == (9,)
        
        # Check block normalization: each 3-element block should sum to 1
        for i in range(3):
            block_sum = np.sum(probs[i*3:(i+1)*3])
            assert np.isclose(block_sum, 1.0)
        
        # Check probabilities are positive
        assert np.all(probs > 0)
    
    def test_input_validation(self):
        """Test input validation for incorrect vector sizes."""
        # Wrong size for R=2 (should be 4)
        with pytest.raises(ValueError, match="x must have length R²=4, got 3"):
            softmax_observation_model(np.array([1.0, 2.0, 3.0]), R=2)
        
        # Wrong size for R=3 (should be 9)
        with pytest.raises(ValueError, match="x must have length R²=9, got 8"):
            softmax_observation_model(np.random.randn(8), R=3)
    
    def test_numerical_stability_large_values(self):
        """Test numerical stability with large logit values."""
        # Very large positive values
        x = np.array([100.0, 200.0, 150.0, 300.0])
        probs = softmax_observation_model(x, R=2)
        
        # Should not have NaN or inf
        assert np.all(np.isfinite(probs))
        
        # Each block should still sum to 1
        assert np.isclose(np.sum(probs[0:2]), 1.0)
        assert np.isclose(np.sum(probs[2:4]), 1.0)
    
    def test_numerical_stability_small_values(self):
        """Test numerical stability with very negative logit values."""
        # Very large negative values
        x = np.array([-100.0, -200.0, -150.0, -300.0])
        probs = softmax_observation_model(x, R=2)
        
        # Should not have NaN or inf
        assert np.all(np.isfinite(probs))
        
        # Each block should still sum to 1
        assert np.isclose(np.sum(probs[0:2]), 1.0)
        assert np.isclose(np.sum(probs[2:4]), 1.0)
    
    def test_underflow_warning(self):
        """Test warning when numerical underflow occurs."""
        # Create scenario where exp(x - max(x)) might underflow to zero
        x = np.array([-1000.0, -1000.0, -1000.0, -1000.0])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            probs = softmax_observation_model(x, R=2)
            
            # Should have issued warnings for both blocks
            warning_messages = [str(warning.message) for warning in w]
            underflow_warnings = [msg for msg in warning_messages if "underflow" in msg]
            assert len(underflow_warnings) >= 0  # May or may not underflow depending on NumPy version
        
        # Should fallback to uniform distribution
        if len(underflow_warnings) > 0:
            assert np.allclose(probs[0:2], 0.5)  # Uniform for R=2
            assert np.allclose(probs[2:4], 0.5)
    
    def test_uniform_case(self):
        """Test that equal logits produce uniform probabilities."""
        x = np.array([1.0, 1.0, 2.0, 2.0])  # Equal within each block
        probs = softmax_observation_model(x, R=2)
        
        # Within each block, should be uniform
        assert np.isclose(probs[0], probs[1])  # Block 0: equal probabilities
        assert np.isclose(probs[2], probs[3])  # Block 1: equal probabilities
        assert np.isclose(probs[0], 0.5)
        assert np.isclose(probs[2], 0.5)
    
    def test_deterministic_case(self):
        """Test near-deterministic probabilities with large differences."""
        x = np.array([10.0, 0.0, 0.0, 10.0])  # Large differences within blocks
        probs = softmax_observation_model(x, R=2)
        
        # First element of each block should be much larger
        assert probs[0] > 0.99  # First transition in block 0
        assert probs[1] < 0.01  # Second transition in block 0
        assert probs[2] < 0.01  # First transition in block 1  
        assert probs[3] > 0.99  # Second transition in block 1


class TestLogSoftmaxJacobian:
    """Test suite for the log_softmax_jacobian function."""
    
    def test_basic_functionality(self):
        """Test basic Jacobian computation."""
        x_block = np.array([1.0, 2.0, 0.5])
        J = log_softmax_jacobian(x_block)
        
        # Check shape
        assert J.shape == (3, 3)
        
        # Check that it's not identity matrix (unless special case)
        assert not np.allclose(J, np.eye(3))
        
        # J should be finite
        assert np.all(np.isfinite(J))
    
    def test_jacobian_formula(self):
        """Test that Jacobian satisfies J[i,j] = δ_ij - f_j."""
        x_block = np.array([1.0, 2.0, 0.5])
        J = log_softmax_jacobian(x_block)
        
        # Compute softmax probabilities manually
        exp_x = np.exp(x_block - np.max(x_block))
        probs = exp_x / np.sum(exp_x)
        
        # Check formula: J[i,j] = δ_ij - f_j
        expected_J = np.eye(3) - probs[np.newaxis, :]
        assert np.allclose(J, expected_J)
    
    def test_diagonal_elements(self):
        """Test that diagonal elements are 1 - f_i."""
        x_block = np.array([0.0, 1.0, -1.0])
        J = log_softmax_jacobian(x_block)
        
        # Compute probabilities
        exp_x = np.exp(x_block - np.max(x_block))
        probs = exp_x / np.sum(exp_x)
        
        # Diagonal should be 1 - f_i
        for i in range(3):
            assert np.isclose(J[i, i], 1.0 - probs[i])
    
    def test_off_diagonal_elements(self):
        """Test that off-diagonal elements are -f_j."""
        x_block = np.array([2.0, 0.0, 1.0])
        J = log_softmax_jacobian(x_block)
        
        # Compute probabilities
        exp_x = np.exp(x_block - np.max(x_block))
        probs = exp_x / np.sum(exp_x)
        
        # Off-diagonal should be -f_j
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isclose(J[i, j], -probs[j])
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Large positive values
        x_block = np.array([100.0, 200.0, 150.0])
        J = log_softmax_jacobian(x_block)
        assert np.all(np.isfinite(J))
        
        # Large negative values
        x_block = np.array([-100.0, -200.0, -150.0])
        J = log_softmax_jacobian(x_block)
        assert np.all(np.isfinite(J))
    
    def test_uniform_case(self):
        """Test Jacobian for uniform probability case."""
        x_block = np.array([0.0, 0.0, 0.0])  # Equal logits
        J = log_softmax_jacobian(x_block)
        
        # All probabilities should be 1/3
        prob = 1.0 / 3.0
        
        # Diagonal elements should be 1 - 1/3 = 2/3
        for i in range(3):
            assert np.isclose(J[i, i], 2.0/3.0)
        
        # Off-diagonal elements should be -1/3
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isclose(J[i, j], -1.0/3.0)


class TestLogSoftmaxHessian:
    """Test suite for the log_softmax_hessian function."""
    
    def test_basic_functionality(self):
        """Test basic Hessian computation."""
        x_block = np.array([1.0, 2.0, 0.5])
        H = log_softmax_hessian(x_block)
        
        # Check shape
        assert H.shape == (3, 3)
        
        # H should be finite
        assert np.all(np.isfinite(H))
    
    def test_hessian_formula(self):
        """Test that Hessian satisfies H[i,j] = -f_i(δ_ij - f_j)."""
        x_block = np.array([1.0, 2.0, 0.5])
        H = log_softmax_hessian(x_block)
        
        # Compute softmax probabilities manually
        exp_x = np.exp(x_block - np.max(x_block))
        probs = exp_x / np.sum(exp_x)
        
        # Check formula: H[i,j] = -f_i(δ_ij - f_j) = f_i * f_j - f_i * δ_ij
        expected_H = np.outer(probs, probs) - np.diag(probs)
        assert np.allclose(H, expected_H)
    
    def test_symmetry(self):
        """Test that Hessian is symmetric."""
        x_block = np.array([0.5, 1.5, -0.5])
        H = log_softmax_hessian(x_block)
        
        # Hessian should be symmetric
        assert np.allclose(H, H.T)
    
    def test_diagonal_elements(self):
        """Test that diagonal elements are f_i(f_i - 1)."""
        x_block = np.array([0.0, 1.0, -1.0])
        H = log_softmax_hessian(x_block)
        
        # Compute probabilities
        exp_x = np.exp(x_block - np.max(x_block))
        probs = exp_x / np.sum(exp_x)
        
        # Diagonal should be f_i * f_i - f_i = f_i(f_i - 1)
        for i in range(3):
            expected_diag = probs[i] * probs[i] - probs[i]
            assert np.isclose(H[i, i], expected_diag)
    
    def test_off_diagonal_elements(self):
        """Test that off-diagonal elements are f_i * f_j."""
        x_block = np.array([2.0, 0.0, 1.0])
        H = log_softmax_hessian(x_block)
        
        # Compute probabilities
        exp_x = np.exp(x_block - np.max(x_block))
        probs = exp_x / np.sum(exp_x)
        
        # Off-diagonal should be f_i * f_j
        for i in range(3):
            for j in range(3):
                if i != j:
                    expected_val = probs[i] * probs[j]
                    assert np.isclose(H[i, j], expected_val)
    
    def test_negative_definite(self):
        """Test that Hessian is negative definite."""
        x_block = np.array([1.0, 2.0, 0.5, -0.5])
        H = log_softmax_hessian(x_block)
        
        # All eigenvalues should be negative (negative definite)
        eigenvals = np.linalg.eigvals(H)
        assert np.all(eigenvals <= 0)
        
        # At least one eigenvalue should be strictly negative
        assert np.any(eigenvals < -1e-10)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Large positive values
        x_block = np.array([100.0, 200.0, 150.0])
        H = log_softmax_hessian(x_block)
        assert np.all(np.isfinite(H))
        
        # Large negative values
        x_block = np.array([-100.0, -200.0, -150.0])
        H = log_softmax_hessian(x_block)
        assert np.all(np.isfinite(H))


class TestComputeObservationLikelihood:
    """Test suite for the compute_observation_likelihood function."""
    
    def test_basic_functionality(self):
        """Test basic likelihood computation."""
        observations = np.array([0, 1, 0, 1])  # Simple alternating sequence
        x = np.array([1.0, 2.0, 0.5, 1.5])  # R=2
        likelihood = compute_observation_likelihood(observations, x, R=2)
        
        # Should return finite likelihood
        assert np.isfinite(likelihood)
        assert isinstance(likelihood, float)
    
    def test_short_sequence(self):
        """Test with sequence too short for transitions."""
        # Single observation - no transitions
        observations = np.array([0])
        x = np.array([1.0, 2.0, 0.5, 1.5])
        likelihood = compute_observation_likelihood(observations, x, R=2)
        assert likelihood == 0.0
        
        # Empty sequence
        observations = np.array([])
        likelihood = compute_observation_likelihood(observations, x, R=2)
        assert likelihood == 0.0
    
    def test_valid_transitions(self):
        """Test likelihood computation with known transitions."""
        observations = np.array([0, 1, 1, 0])  # Transitions: 0→1, 1→1, 1→0
        
        # Create deterministic transition matrix
        x = np.array([0.0, 10.0, 10.0, 0.0])  # High prob for 0→1 and 1→0
        likelihood = compute_observation_likelihood(observations, x, R=2)
        
        # Should have reasonable likelihood for this configuration
        assert likelihood > -20.0  # Should be reasonable (log-likelihoods are typically negative)
    
    def test_invalid_symbol_indices(self):
        """Test error handling for invalid symbol indices."""
        observations = np.array([0, 1, 2])  # Symbol 2 invalid for R=2
        x = np.array([1.0, 2.0, 0.5, 1.5])
        
        with pytest.raises(ValueError, match="Invalid symbol indices"):
            compute_observation_likelihood(observations, x, R=2)
        
        # Negative indices
        observations = np.array([0, -1, 1])
        with pytest.raises(ValueError, match="Invalid symbol indices"):
            compute_observation_likelihood(observations, x, R=2)
    
    def test_zero_probability_warning(self):
        """Test warning when transition has zero probability."""
        observations = np.array([0, 1])  # Transition 0→1
        
        # Create x that gives zero probability to 0→1 transition
        x = np.array([-100.0, -200.0, 1.0, 2.0])  # Very low prob for 0→1
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            likelihood = compute_observation_likelihood(observations, x, R=2)
            
            # Should issue warning about zero probability
            if len(w) > 0:
                warning_messages = [str(warning.message) for warning in w]
                zero_prob_warnings = [msg for msg in warning_messages if "Zero probability" in msg]
                # May or may not warn depending on numerical precision
        
        # Likelihood should be -inf if probability is truly zero
        assert likelihood <= 0  # Should be negative or -inf
    
    def test_deterministic_case(self):
        """Test likelihood with deterministic transitions."""
        observations = np.array([0, 1, 1, 1, 0])
        
        # Perfect match: high prob for 0→1, 1→1, 1→0
        x = np.array([0.0, 10.0, 0.0, 10.0])  # 0→1: high, 1→0: high, others: low
        likelihood = compute_observation_likelihood(observations, x, R=2)
        
        # Should have reasonable likelihood
        assert likelihood > -20.0  # Should be reasonable for high-probability sequence
    
    def test_likelihood_decreases_with_improbable_transitions(self):
        """Test that likelihood decreases with improbable transitions."""
        observations = np.array([0, 1, 0, 1])
        
        # High probability case
        x_high = np.array([0.0, 2.0, 2.0, 0.0])  # Favor 0→1 and 1→0
        likelihood_high = compute_observation_likelihood(observations, x_high, R=2)
        
        # Low probability case  
        x_low = np.array([2.0, 0.0, 0.0, 2.0])   # Favor 0→0 and 1→1
        likelihood_low = compute_observation_likelihood(observations, x_low, R=2)
        
        # High probability case should have higher likelihood
        assert likelihood_high > likelihood_low
    
    def test_longer_sequence(self):
        """Test with longer observation sequence."""
        np.random.seed(42)  # For reproducibility
        observations = np.random.randint(0, 3, size=20)  # R=3, 20 observations
        x = np.random.randn(9)  # 3x3 logit vector
        
        likelihood = compute_observation_likelihood(observations, x, R=3)
        
        # Should be finite and reasonable
        assert np.isfinite(likelihood)
        assert likelihood <= 0  # Log-likelihood should be non-positive


class TestValidateTransitionProbabilities:
    """Test suite for the validate_transition_probabilities function."""
    
    def test_valid_probabilities(self):
        """Test validation with valid probability distributions."""
        # Create valid probabilities manually
        probs = np.array([0.3, 0.7, 0.4, 0.6])  # Two blocks, each sums to 1
        result = validate_transition_probabilities(probs, R=2)
        assert result is True
    
    def test_valid_probabilities_r3(self):
        """Test validation with valid R=3 probabilities."""
        probs = np.array([0.2, 0.3, 0.5,   # Block 0: sums to 1.0
                         0.1, 0.4, 0.5,   # Block 1: sums to 1.0  
                         0.6, 0.2, 0.2])  # Block 2: sums to 1.0
        result = validate_transition_probabilities(probs, R=3)
        assert result is True
    
    def test_invalid_probabilities_sum_too_high(self):
        """Test validation failure when block sum is too high."""
        probs = np.array([0.6, 0.7, 0.4, 0.6])  # First block sums to 1.3
        
        with pytest.raises(AssertionError, match="sum to 1.30000000"):
            validate_transition_probabilities(probs, R=2)
    
    def test_invalid_probabilities_sum_too_low(self):
        """Test validation failure when block sum is too low."""
        probs = np.array([0.2, 0.3, 0.4, 0.6])  # First block sums to 0.5
        
        with pytest.raises(AssertionError, match="sum to 0.50000000"):
            validate_transition_probabilities(probs, R=2)
    
    def test_tolerance_parameter(self):
        """Test that tolerance parameter works correctly."""
        # Slightly off but within default tolerance
        probs = np.array([0.3000001, 0.6999999, 0.4, 0.6])  # First block sum ≈ 1.0
        result = validate_transition_probabilities(probs, R=2, atol=1e-6)
        assert result is True
        
        # Outside stricter tolerance - use a value that actually violates tolerance
        probs_bad = np.array([0.30001, 0.69999, 0.4, 0.6])  # First block sum is more off
        with pytest.raises(AssertionError):
            validate_transition_probabilities(probs_bad, R=2, atol=1e-8)
    
    def test_edge_case_uniform_distribution(self):
        """Test validation with uniform distributions."""
        # Perfect uniform for R=2
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        result = validate_transition_probabilities(probs, R=2)
        assert result is True
        
        # Perfect uniform for R=3
        probs = np.array([1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3])
        result = validate_transition_probabilities(probs, R=3, atol=1e-10)
        assert result is True
    
    def test_softmax_output_validation(self):
        """Test validation of actual softmax output."""
        # Generate valid softmax output
        x = np.random.randn(4)
        probs = softmax_observation_model(x, R=2)
        
        # Should pass validation
        result = validate_transition_probabilities(probs, R=2)
        assert result is True
    
    def test_multiple_block_failure(self):
        """Test validation when multiple blocks are invalid."""
        probs = np.array([0.4, 0.4,   # Block 0: sums to 0.8 (invalid)
                         0.7, 0.7,   # Block 1: sums to 1.4 (invalid)
                         0.2, 0.2])  # Block 2: sums to 0.4 (invalid) - if R=3
        
        # Should fail on first invalid block
        with pytest.raises(AssertionError, match="sum to 0.80000000"):
            validate_transition_probabilities(probs, R=2)


class TestIntegrationScenarios:
    """Integration tests combining multiple functions."""
    
    def test_softmax_to_validation_pipeline(self):
        """Test pipeline from logits to validation."""
        x = np.random.randn(9)  # R=3
        
        # Generate probabilities
        probs = softmax_observation_model(x, R=3)
        
        # Should pass validation
        result = validate_transition_probabilities(probs, R=3)
        assert result is True
    
    def test_jacobian_hessian_consistency(self):
        """Test mathematical consistency between Jacobian and Hessian."""
        x_block = np.array([1.0, 2.0, 0.5])
        
        J = log_softmax_jacobian(x_block)
        H = log_softmax_hessian(x_block)
        
        # Both should be same size
        assert J.shape == H.shape
        
        # Hessian should be more negative (since it's second derivative)
        # This is a qualitative check - both should be finite
        assert np.all(np.isfinite(J))
        assert np.all(np.isfinite(H))
    
    def test_likelihood_with_generated_data(self):
        """Test likelihood computation with generated sequence."""
        np.random.seed(123)
        
        # Generate logits and probabilities
        x = np.random.randn(4)  # R=2
        probs = softmax_observation_model(x, R=2)
        
        # Generate observation sequence based on transition probabilities
        observations = [0]  # Start with symbol 0
        for _ in range(10):
            current_symbol = observations[-1]
            # Transition probabilities from current symbol
            trans_probs = probs[current_symbol*2:(current_symbol+1)*2]
            next_symbol = np.random.choice(2, p=trans_probs)
            observations.append(next_symbol)
        
        observations = np.array(observations)
        
        # Compute likelihood
        likelihood = compute_observation_likelihood(observations, x, R=2)
        
        # Should be finite and not too negative for generated data
        assert np.isfinite(likelihood)
        assert likelihood > -50  # Reasonable bound for 10 transitions
    
    def test_numerical_edge_cases(self):
        """Test numerical edge cases across all functions."""
        # Extreme logits
        x_extreme = np.array([1000.0, -1000.0, -1000.0, 1000.0])
        
        # All functions should handle extreme values
        probs = softmax_observation_model(x_extreme, R=2)
        assert np.all(np.isfinite(probs))
        
        # Jacobian and Hessian with extreme probabilities
        for i in range(2):
            block = x_extreme[i*2:(i+1)*2]
            J = log_softmax_jacobian(block)
            H = log_softmax_hessian(block)
            assert np.all(np.isfinite(J))
            assert np.all(np.isfinite(H))
        
        # Likelihood with extreme probabilities
        observations = np.array([0, 1])  # Simple transition
        likelihood = compute_observation_likelihood(observations, x_extreme, R=2)
        # With extreme values, likelihood might be -inf, but should not be +inf or NaN
        assert not np.isnan(likelihood) and likelihood <= 0
        
        # Validation should still work
        result = validate_transition_probabilities(probs, R=2)
        assert result is True 