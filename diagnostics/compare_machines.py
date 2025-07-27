#!/usr/bin/env python3
"""Compare local vs external machine configurations to identify performance differences."""

import sys
import platform
import numpy as np
import scipy
import multiprocessing as mp
from pathlib import Path

def compare_machine_configs():
    """Compare machine configurations that could affect performance."""
    
    print("🔍 MACHINE CONFIGURATION COMPARISON")
    print("=" * 50)
    
    # System info
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Processor: {platform.processor()}")
    
    # Package versions
    print(f"\nPackage versions:")
    print(f"  NumPy: {np.__version__}")
    print(f"  SciPy: {scipy.__version__}")
    
    # NumPy configuration
    print(f"\nNumPy configuration:")
    print(f"  BLAS info: {np.__config__.get_info('blas_opt_info')}")
    print(f"  LAPACK info: {np.__config__.get_info('lapack_opt_info')}")
    print(f"  OpenBLAS info: {np.__config__.get_info('openblas_info')}")
    
    # CPU features
    print(f"\nCPU features:")
    try:
        cpu_features = []
        if hasattr(np, 'show_config'):
            np.show_config()
        else:
            print("  (CPU feature detection not available)")
    except:
        print("  (CPU feature detection failed)")
    
    # Multiprocessing info
    print(f"\nMultiprocessing:")
    print(f"  Start method: {mp.get_start_method()}")
    print(f"  Available methods: {mp.get_all_start_methods()}")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\nMemory:")
        print(f"  Total: {memory.total / (1024**3):.1f} GB")
        print(f"  Available: {memory.available / (1024**3):.1f} GB")
        print(f"  Usage: {memory.percent}%")
    except ImportError:
        print(f"\nMemory: psutil not available")
    
    return True

def test_numpy_performance():
    """Test NumPy performance characteristics."""
    
    print(f"\n🔍 NUMPY PERFORMANCE TEST")
    print("=" * 50)
    
    # Test matrix multiplication performance
    print("Testing matrix multiplication...")
    
    sizes = [100, 500, 1000]
    for size in sizes:
        print(f"  {size}x{size} matrix multiplication:")
        
        # Create test matrices
        a = np.random.random((size, size))
        b = np.random.random((size, size))
        
        # Time multiplication
        import time
        start_time = time.time()
        c = np.dot(a, b)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"    Time: {elapsed:.4f} seconds")
        
        # Verify result
        expected_shape = (size, size)
        if c.shape == expected_shape:
            print(f"    ✅ Result shape correct: {c.shape}")
        else:
            print(f"    ❌ Result shape wrong: {c.shape} (expected {expected_shape})")
    
    # Test eigenvalue computation
    print("\nTesting eigenvalue computation...")
    for size in [50, 100]:
        print(f"  {size}x{size} eigenvalue computation:")
        
        # Create symmetric matrix
        a = np.random.random((size, size))
        a = (a + a.T) / 2  # Make symmetric
        
        start_time = time.time()
        eigenvals = np.linalg.eigvals(a)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"    Time: {elapsed:.4f} seconds")
        print(f"    ✅ Eigenvalues computed: {len(eigenvals)} values")
    
    return True

def test_scipy_performance():
    """Test SciPy performance characteristics."""
    
    print(f"\n🔍 SCIPY PERFORMANCE TEST")
    print("=" * 50)
    
    try:
        from scipy import linalg
        
        print("Testing SciPy linear algebra...")
        
        sizes = [50, 100]
        for size in sizes:
            print(f"  {size}x{size} SciPy eigendecomposition:")
            
            # Create test matrix
            a = np.random.random((size, size))
            a = (a + a.T) / 2  # Make symmetric
            
            start_time = time.time()
            eigenvals, eigenvecs = linalg.eigh(a)
            end_time = time.time()
            
            elapsed = end_time - start_time
            print(f"    Time: {elapsed:.4f} seconds")
            print(f"    ✅ Eigenvalues: {len(eigenvals)}")
            
    except Exception as e:
        print(f"❌ SciPy test failed: {e}")
        return False
    
    return True

def test_specific_operations():
    """Test operations that might be used in the experiment."""
    
    print(f"\n🔍 SPECIFIC OPERATIONS TEST")
    print("=" * 50)
    
    # Test softmax computation (used in the experiment)
    print("Testing softmax computation...")
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    sizes = [100, 1000, 10000]
    for size in sizes:
        print(f"  Softmax for {size} elements:")
        
        # Create test data
        x = np.random.random(size)
        
        start_time = time.time()
        result = softmax(x)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"    Time: {elapsed:.6f} seconds")
        print(f"    ✅ Result sum: {np.sum(result):.6f} (should be ~1.0)")
    
    # Test log-likelihood computation
    print("\nTesting log-likelihood computation...")
    
    def compute_log_likelihood(probabilities, observations):
        return np.sum(np.log(probabilities + 1e-10))
    
    sizes = [100, 1000]
    for size in sizes:
        print(f"  Log-likelihood for {size} observations:")
        
        # Create test data
        probs = np.random.random(size)
        obs = np.random.randint(0, 2, size)
        
        start_time = time.time()
        ll = compute_log_likelihood(probs, obs)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"    Time: {elapsed:.6f} seconds")
        print(f"    ✅ Log-likelihood: {ll:.6f}")
    
    return True

def main():
    """Main comparison function."""
    
    print("🚀 MACHINE COMPARISON DIAGNOSTIC")
    print("Run this on both local and external machines to compare")
    print("=" * 60)
    
    tests = [
        ("Machine Configurations", compare_machine_configs),
        ("NumPy Performance", test_numpy_performance),
        ("SciPy Performance", test_scipy_performance),
        ("Specific Operations", test_specific_operations),
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
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n💡 Next steps:")
    print(f"1. Run this script on your LOCAL machine")
    print(f"2. Compare the outputs between machines")
    print(f"3. Look for differences in:")
    print(f"   - NumPy/SciPy versions")
    print(f"   - BLAS/LAPACK configurations")
    print(f"   - Performance test timings")
    print(f"   - CPU features")

if __name__ == "__main__":
    main() 