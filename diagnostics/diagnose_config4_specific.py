#!/usr/bin/env python3
"""Focused diagnostic to identify exactly where config4 gets stuck."""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config4_step_by_step():
    """Test config4 loading and execution step by step."""
    
    logger.info("🔍 STEP-BY-STEP CONFIG4 DIAGNOSTIC")
    logger.info("=" * 50)
    
    # Step 1: Import dependencies
    logger.info("Step 1: Testing imports...")
    try:
        from aggregate_analysis import AggregateAnalysisManager
        logger.info("✅ AggregateAnalysisManager imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import AggregateAnalysisManager: {e}")
        return False
    
    # Step 2: Create manager
    logger.info("Step 2: Creating manager...")
    try:
        manager = AggregateAnalysisManager(output_dir="results/aggregate_analysis")
        logger.info("✅ Manager created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create manager: {e}")
        return False
    
    # Step 3: Load config4
    logger.info("Step 3: Loading config4...")
    try:
        config4 = manager.experimental_configs[4]
        logger.info("✅ Config4 loaded successfully")
        logger.info(f"   Config type: {type(config4)}")
        logger.info(f"   Config keys: {list(config4.keys()) if hasattr(config4, 'keys') else 'N/A'}")
    except Exception as e:
        logger.error(f"❌ Failed to load config4: {e}")
        return False
    
    # Step 4: Test sequence generation
    logger.info("Step 4: Testing sequence generation...")
    try:
        from src.adaptive_syntax_filter.data.sequence_generator import (
            GenerationConfig, SequenceGenerator, sequences_to_observations, generate_dataset
        )
        
        # Create a minimal generation config
        gen_config = GenerationConfig(
            n_sequences=10,  # Small number for testing
            sequence_length=config4.get('sequence_length', 6),
            alphabet_size=config4.get('alphabet_size', 5),
            random_seed=42
        )
        
        logger.info("✅ Sequence generation imports successful")
        logger.info(f"   Testing with {gen_config.n_sequences} sequences of length {gen_config.sequence_length}")
        
        # Test actual generation
        generator = SequenceGenerator(gen_config)
        sequences = generator.generate_sequences()
        logger.info(f"✅ Generated {len(sequences)} sequences successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed in sequence generation: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False
    
    # Step 5: Test observation model
    logger.info("Step 5: Testing observation model...")
    try:
        from src.adaptive_syntax_filter.core.observation_model import ObservationModel
        
        # Create observation model
        obs_model = ObservationModel(
            alphabet_size=config4.get('alphabet_size', 5),
            observation_noise=config4.get('observation_noise', 0.1)
        )
        logger.info("✅ Observation model created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed in observation model: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False
    
    # Step 6: Test state space
    logger.info("Step 6: Testing state space...")
    try:
        from src.adaptive_syntax_filter.core.state_space import StateSpaceManager
        
        state_manager = StateSpaceManager(
            alphabet_size=config4.get('alphabet_size', 5),
            order=config4.get('order', 2)
        )
        logger.info("✅ State space manager created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed in state space: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False
    
    # Step 7: Test EM algorithm
    logger.info("Step 7: Testing EM algorithm...")
    try:
        from src.adaptive_syntax_filter.core.em_algorithm import EMAlgorithm, EMParameters
        
        em_params = EMParameters(
            max_iterations=config4.get('max_iterations', 100),
            convergence_threshold=config4.get('convergence_threshold', 1e-6),
            regularization=config4.get('regularization', 1e-8)
        )
        
        em_algorithm = EMAlgorithm(em_params)
        logger.info("✅ EM algorithm created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed in EM algorithm: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False
    
    # Step 8: Test single experiment (with timeout)
    logger.info("Step 8: Testing single experiment (with 30 second timeout)...")
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Experiment timed out after 30 seconds")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        start_time = time.time()
        result = manager.run_single_experiment(4, 42)  # config4, seed 42
        end_time = time.time()
        
        # Cancel timeout
        signal.alarm(0)
        
        logger.info(f"✅ Single experiment completed successfully!")
        logger.info(f"   Runtime: {end_time - start_time:.2f} seconds")
        logger.info(f"   Result type: {type(result)}")
        
    except TimeoutError:
        logger.error("❌ Single experiment timed out after 30 seconds")
        logger.error("   This indicates config4 has a performance issue")
        return False
    except Exception as e:
        logger.error(f"❌ Failed in single experiment: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False
    
    logger.info("🎉 ALL TESTS PASSED!")
    logger.info("Config4 appears to work correctly in isolation")
    return True

def test_config3_comparison():
    """Compare config3 and config4 to identify differences."""
    
    logger.info("\n🔍 CONFIG3 vs CONFIG4 COMPARISON")
    logger.info("=" * 50)
    
    try:
        from aggregate_analysis import AggregateAnalysisManager
        
        manager = AggregateAnalysisManager(output_dir="results/aggregate_analysis")
        
        config3 = manager.experimental_configs[3]
        config4 = manager.experimental_configs[4]
        
        logger.info("Config3 parameters:")
        for key, value in config3.items():
            logger.info(f"   {key}: {value}")
        
        logger.info("\nConfig4 parameters:")
        for key, value in config4.items():
            logger.info(f"   {key}: {value}")
        
        # Find differences
        logger.info("\nKey differences:")
        for key in set(config3.keys()) | set(config4.keys()):
            if key not in config3:
                logger.info(f"   {key}: Only in config4 ({config4[key]})")
            elif key not in config4:
                logger.info(f"   {key}: Only in config3 ({config3[key]})")
            elif config3[key] != config4[key]:
                logger.info(f"   {key}: config3={config3[key]} vs config4={config4[key]}")
        
    except Exception as e:
        logger.error(f"❌ Failed in config comparison: {e}")

def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(description="Focused config4 diagnostic")
    parser.add_argument("--compare-configs", action="store_true", help="Compare config3 and config4")
    
    args = parser.parse_args()
    
    if args.compare_configs:
        test_config3_comparison()
    else:
        success = test_config4_step_by_step()
        
        if success:
            logger.info("\n✅ Config4 works in isolation")
            logger.info("The issue might be in multiprocessing or resource contention")
        else:
            logger.info("\n❌ Config4 has issues even in isolation")
            logger.info("This explains why it gets stuck in multiprocessing")

if __name__ == "__main__":
    main() 