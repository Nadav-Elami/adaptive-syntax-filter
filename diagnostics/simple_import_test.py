#!/usr/bin/env python3
"""Simple import test to identify what's causing the hang."""

import sys
import time

def test_imports_one_by_one():
    """Test imports one by one to find the problematic one."""
    
    print("🔍 TESTING IMPORTS ONE BY ONE")
    print("=" * 40)
    
    # Test 1: Basic imports
    print("Test 1: Basic Python imports...")
    try:
        import numpy as np
        print("✅ numpy imported")
    except Exception as e:
        print(f"❌ numpy failed: {e}")
        return
    
    try:
        import pandas as pd
        print("✅ pandas imported")
    except Exception as e:
        print(f"❌ pandas failed: {e}")
        return
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported")
    except Exception as e:
        print(f"❌ matplotlib failed: {e}")
        return
    
    # Test 2: Local imports
    print("\nTest 2: Local module imports...")
    
    try:
        print("  Testing config_cli...")
        from config_cli import _read_config as load_config
        print("✅ config_cli imported")
    except Exception as e:
        print(f"❌ config_cli failed: {e}")
        return
    
    try:
        print("  Testing research_pipeline...")
        from research_pipeline import run_pipeline
        print("✅ research_pipeline imported")
    except Exception as e:
        print(f"❌ research_pipeline failed: {e}")
        return
    
    # Test 3: Core module imports
    print("\nTest 3: Core module imports...")
    
    try:
        print("  Testing sequence_generator...")
        from src.adaptive_syntax_filter.data.sequence_generator import (
            GenerationConfig, SequenceGenerator, sequences_to_observations, generate_dataset
        )
        print("✅ sequence_generator imported")
    except Exception as e:
        print(f"❌ sequence_generator failed: {e}")
        return
    
    try:
        print("  Testing state_space...")
        from src.adaptive_syntax_filter.core.state_space import StateSpaceManager
        print("✅ state_space imported")
    except Exception as e:
        print(f"❌ state_space failed: {e}")
        return
    
    try:
        print("  Testing em_algorithm...")
        from src.adaptive_syntax_filter.core.em_algorithm import EMAlgorithm, EMParameters, EMResults
        print("✅ em_algorithm imported")
    except Exception as e:
        print(f"❌ em_algorithm failed: {e}")
        return
    
    try:
        print("  Testing temporal_evolution...")
        from src.adaptive_syntax_filter.data.temporal_evolution import compute_evolution_trajectory
        print("✅ temporal_evolution imported")
    except Exception as e:
        print(f"❌ temporal_evolution failed: {e}")
        return
    
    # Test 4: Aggregate analysis import
    print("\nTest 4: Aggregate analysis import...")
    
    try:
        print("  Testing aggregate_analysis...")
        from aggregate_analysis import AggregateAnalysisManager
        print("✅ aggregate_analysis imported")
    except Exception as e:
        print(f"❌ aggregate_analysis failed: {e}")
        return
    
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    print("The issue might be in the actual execution, not imports")

def test_config4_simple():
    """Test config4 with minimal setup."""
    
    print("\n🔍 TESTING CONFIG4 SIMPLE")
    print("=" * 40)
    
    try:
        print("Creating manager...")
        from aggregate_analysis import AggregateAnalysisManager
        
        manager = AggregateAnalysisManager(output_dir="results/aggregate_analysis")
        print("✅ Manager created")
        
        print("Loading config4...")
        config4 = manager.experimental_configs[4]
        print("✅ Config4 loaded")
        print(f"   Config type: {type(config4)}")
        print(f"   Config ID: {config4.config_id}")
        print(f"   Config name: {config4.config_name}")
        print(f"   Description: {config4.description}")
        print(f"   Alphabet: {config4.alphabet}")
        print(f"   Order: {config4.order}")
        print(f"   Evolution type: {config4.evolution_type}")
        print(f"   N sequences: {config4.n_sequences}")
        print(f"   Max length: {config4.max_length}")
        print(f"   Max EM iterations: {config4.max_em_iterations}")
        
        print("Testing single experiment...")
        start_time = time.time()
        result = manager.run_single_experiment(4, 42)
        end_time = time.time()
        
        print(f"✅ Single experiment completed in {end_time - start_time:.2f} seconds")
        print(f"   Result type: {type(result)}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports_one_by_one()
    test_config4_simple() 