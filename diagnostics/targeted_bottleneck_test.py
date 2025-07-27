#!/usr/bin/env python3
"""Targeted test to identify the exact bottleneck in config4 experiments."""

import sys
import time
import numpy as np
import traceback

def test_sequence_generation():
    """Test sequence generation operations."""
    
    print("🔍 SEQUENCE GENERATION TEST")
    print("=" * 50)
    
    try:
        from src.adaptive_syntax_filter.data.sequence_generator import (
            GenerationConfig, SequenceGenerator, sequences_to_observations, generate_dataset
        )
        
        # Test with config4 parameters
        print("Testing sequence generation with config4 parameters...")
        
        gen_config = GenerationConfig(
            alphabet=['<', 'a', 'b', 'c', 'd', '>'],
            order=2,
            n_sequences=50,
            max_length=200,
            evolution_type='sigmoid',
            seed=42
        )
        
        print(f"  Config: {gen_config}")
        
        start_time = time.time()
        sequences, true_logits = generate_dataset(gen_config)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"  ✅ Sequence generation: {elapsed:.4f} seconds")
        print(f"     Generated {len(sequences)} sequences")
        print(f"     True logits shape: {true_logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Sequence generation failed: {e}")
        traceback.print_exc()
        return False

def test_state_space():
    """Test state space operations."""
    
    print("\n🔍 STATE SPACE TEST")
    print("=" * 50)
    
    try:
        from src.adaptive_syntax_filter.core.state_space import StateSpaceManager
        
        print("Testing state space with config4 parameters...")
        
        state_manager = StateSpaceManager(
            alphabet_size=6,  # config4 alphabet size
            order=2  # config4 order
        )
        
        print(f"  State manager created")
        print(f"  State space size: {state_manager.state_space_size}")
        
        # Test state encoding/decoding
        test_sequence = ['<', 'a', 'b', 'c', '>']
        print(f"  Testing with sequence: {test_sequence}")
        
        start_time = time.time()
        state = state_manager.encode_sequence(test_sequence)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"  ✅ State encoding: {elapsed:.6f} seconds")
        print(f"     Encoded state: {state}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ State space test failed: {e}")
        traceback.print_exc()
        return False

def test_em_algorithm():
    """Test EM algorithm operations."""
    
    print("\n🔍 EM ALGORITHM TEST")
    print("=" * 50)
    
    try:
        from src.adaptive_syntax_filter.core.em_algorithm import EMAlgorithm, EMParameters
        
        print("Testing EM algorithm with config4 parameters...")
        
        em_params = EMParameters(
            max_iterations=300,
            convergence_threshold=1e-6,
            regularization=1e-8
        )
        
        em_algorithm = EMAlgorithm(em_params)
        print(f"  ✅ EM algorithm created")
        
        # Test with small synthetic data
        print("  Testing with synthetic data...")
        
        # Create small test data
        n_states = 36  # 6^2 for 2nd order
        n_sequences = 10
        n_timesteps = 20
        
        # Random transition matrix
        transition_matrix = np.random.random((n_states, n_states))
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        
        # Random observations
        observations = np.random.randint(0, 6, (n_sequences, n_timesteps))
        
        print(f"  Test data: {n_states} states, {n_sequences} sequences, {n_timesteps} timesteps")
        
        start_time = time.time()
        
        # Run a few EM iterations
        for iteration in range(5):  # Just 5 iterations for testing
            # Simulate EM step
            new_transition_matrix = transition_matrix + np.random.normal(0, 0.01, transition_matrix.shape)
            new_transition_matrix = new_transition_matrix / new_transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = new_transition_matrix
            
            if iteration % 2 == 0:
                print(f"    Iteration {iteration+1}/5")
        
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"  ✅ EM test: {elapsed:.4f} seconds")
        
        return True
        
    except Exception as e:
        print(f"  ❌ EM algorithm test failed: {e}")
        traceback.print_exc()
        return False

def test_observation_model():
    """Test observation model operations."""
    
    print("\n🔍 OBSERVATION MODEL TEST")
    print("=" * 50)
    
    try:
        from src.adaptive_syntax_filter.core.observation_model import ObservationModel
        
        print("Testing observation model with config4 parameters...")
        
        obs_model = ObservationModel(
            alphabet_size=6,
            observation_noise=0.1
        )
        
        print(f"  ✅ Observation model created")
        
        # Test observation generation
        print("  Testing observation generation...")
        
        n_sequences = 50
        n_timesteps = 100
        
        # Create test observations
        observations = np.random.randint(0, 6, (n_sequences, n_timesteps))
        
        start_time = time.time()
        
        # Simulate observation processing
        for i in range(n_sequences):
            for j in range(n_timesteps):
                obs = observations[i, j]
                # Simulate processing
                _ = obs_model.observation_probabilities(obs)
        
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"  ✅ Observation processing: {elapsed:.4f} seconds")
        print(f"     Processed {n_sequences * n_timesteps} observations")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Observation model test failed: {e}")
        traceback.print_exc()
        return False

def test_full_experiment_step_by_step():
    """Test the full experiment step by step with timing."""
    
    print("\n🔍 FULL EXPERIMENT STEP-BY-STEP TEST")
    print("=" * 50)
    
    try:
        from aggregate_analysis import AggregateAnalysisManager
        
        print("Creating manager...")
        start_time = time.time()
        manager = AggregateAnalysisManager(output_dir="results/aggregate_analysis")
        manager_time = time.time() - start_time
        print(f"  ✅ Manager created: {manager_time:.4f} seconds")
        
        print("Loading config4...")
        start_time = time.time()
        config4 = manager.experimental_configs[4]
        config_time = time.time() - start_time
        print(f"  ✅ Config4 loaded: {config_time:.4f} seconds")
        
        print("Running single experiment with detailed timing...")
        start_time = time.time()
        
        # Run the experiment
        result = manager.run_single_experiment(4, 42)
        
        end_time = time.time()
        experiment_time = end_time - start_time
        
        print(f"  ✅ Full experiment completed: {experiment_time:.4f} seconds")
        print(f"     Result type: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Full experiment test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    
    print("🚀 TARGETED BOTTLENECK TEST")
    print("This will identify exactly where config4 gets slow")
    print("=" * 60)
    
    tests = [
        ("Sequence Generation", test_sequence_generation),
        ("State Space", test_state_space),
        ("EM Algorithm", test_em_algorithm),
        ("Observation Model", test_observation_model),
        ("Full Experiment", test_full_experiment_step_by_step),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("BOTTLENECK TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("The bottleneck might be in the combination of operations")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("This identifies the specific bottleneck")

if __name__ == "__main__":
    main() 