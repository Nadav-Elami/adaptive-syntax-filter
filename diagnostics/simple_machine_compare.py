#!/usr/bin/env python3
"""Simple machine comparison to identify performance differences."""

import sys
import platform
import numpy as np
import scipy
import time
import multiprocessing as mp

def compare_machines():
    """Compare key machine characteristics."""
    
    print("🔍 MACHINE COMPARISON")
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
    try:
        print(f"  BLAS info: {np.__config__.get_info('blas_opt_info')}")
    except:
        print(f"  BLAS info: Not available")
    
    try:
        print(f"  LAPACK info: {np.__config__.get_info('lapack_opt_info')}")
    except:
        print(f"  LAPACK info: Not available")
    
    # Multiprocessing info
    print(f"\nMultiprocessing:")
    print(f"  Start method: {mp.get_start_method()}")
    print(f"  Available methods: {mp.get_all_start_methods()}")
    
    # Performance tests
    print(f"\nPerformance tests:")
    
    # Matrix multiplication
    print(f"  Matrix multiplication (1000x1000):")
    a = np.random.random((1000, 1000))
    b = np.random.random((1000, 1000))
    
    start_time = time.time()
    c = np.dot(a, b)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"    Time: {elapsed:.4f} seconds")
    
    # Eigenvalue computation
    print(f"  Eigenvalue computation (100x100):")
    a = np.random.random((100, 100))
    a = (a + a.T) / 2  # Make symmetric
    
    start_time = time.time()
    eigenvals = np.linalg.eigvals(a)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"    Time: {elapsed:.4f} seconds")
    
    # Softmax computation
    print(f"  Softmax computation (10000 elements):")
    x = np.random.random(10000)
    
    start_time = time.time()
    exp_x = np.exp(x - np.max(x))
    result = exp_x / np.sum(exp_x)
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"    Time: {elapsed:.6f} seconds")
    
    print(f"\n✅ Machine comparison completed")
    print(f"💡 Compare these results with your LOCAL machine")

if __name__ == "__main__":
    compare_machines() 