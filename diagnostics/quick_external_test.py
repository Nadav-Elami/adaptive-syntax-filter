#!/usr/bin/env python3
"""Quick test to identify why external machine can't complete even one seed."""

import sys
import time
import traceback

def quick_diagnostic():
    """Quick diagnostic focusing on most likely issues."""
    
    print("🔍 QUICK EXTERNAL MACHINE DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Basic imports and memory
    print("Test 1: Basic imports and memory...")
    try:
        import numpy as np
        import pandas as pd
        print("✅ Basic imports work")
        
        # Test memory allocation
        test_array = np.zeros((1000, 1000), dtype=np.float64)
        print("✅ Memory allocation works")
        del test_array
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False
    
    # Test 2: Aggregate analysis import
    print("\nTest 2: Aggregate analysis import...")
    try:
        from aggregate_analysis import AggregateAnalysisManager
        print("✅ Aggregate analysis import works")
    except Exception as e:
        print(f"❌ Aggregate analysis import failed: {e}")
        return False
    
    # Test 3: Manager creation
    print("\nTest 3: Manager creation...")
    try:
        manager = AggregateAnalysisManager(output_dir="results/aggregate_analysis")
        print("✅ Manager creation works")
    except Exception as e:
        print(f"❌ Manager creation failed: {e}")
        return False
    
    # Test 4: Config4 loading
    print("\nTest 4: Config4 loading...")
    try:
        config4 = manager.experimental_configs[4]
        print("✅ Config4 loading works")
        print(f"   Config: {config4.config_name}")
        print(f"   Order: {config4.order}")
        print(f"   Evolution: {config4.evolution_type}")
    except Exception as e:
        print(f"❌ Config4 loading failed: {e}")
        return False
    
    # Test 5: Single experiment with timeout
    print("\nTest 5: Single experiment (2-minute timeout)...")
    try:
        import threading
        
        result = None
        error = None
        
        def run_experiment():
            nonlocal result, error
            try:
                result = manager.run_single_experiment(4, 42)
            except Exception as e:
                error = e
        
        # Start experiment in separate thread
        thread = threading.Thread(target=run_experiment)
        thread.daemon = True
        thread.start()
        
        # Wait with timeout
        timeout_seconds = 120  # 2 minutes
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            print(f"❌ Experiment timed out after {timeout_seconds} seconds")
            print("   This explains why external machine can't complete seeds")
            return False
        elif error:
            print(f"❌ Experiment failed with error: {error}")
            traceback.print_exc()
            return False
        else:
            print(f"✅ Single experiment completed successfully!")
            print(f"   Result type: {type(result)}")
            return True
            
    except Exception as e:
        print(f"❌ Experiment test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_diagnostic()
    
    if success:
        print("\n🎉 All tests passed!")
        print("The external machine should be able to complete experiments")
    else:
        print("\n❌ Tests failed!")
        print("This explains why the external machine can't complete seeds") 