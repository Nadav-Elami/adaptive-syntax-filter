#!/usr/bin/env python3
"""Diagnostic to identify differences between local and external machine behavior."""

import argparse
import logging
import sys
import time
import traceback
import os
import platform
import psutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_system_differences():
    """Check for system differences that could cause issues."""
    
    logger.info("🔍 SYSTEM DIFFERENCES DIAGNOSTIC")
    logger.info("=" * 50)
    
    # System info
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    logger.info(f"Memory usage: {memory.percent}%")
    
    # CPU info
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
    
    # Disk info
    disk = psutil.disk_usage('.')
    logger.info(f"Disk free: {disk.free / (1024**3):.1f} GB")
    logger.info(f"Disk usage: {disk.percent}%")
    
    # Environment variables
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    return True

def test_config4_with_timeout():
    """Test config4 with strict timeout to see where it gets stuck."""
    
    logger.info("\n🔍 CONFIG4 TIMEOUT TEST")
    logger.info("=" * 50)
    
    try:
        # Import with timeout
        logger.info("Step 1: Importing modules...")
        start_time = time.time()
        
        from aggregate_analysis import AggregateAnalysisManager
        import_time = time.time() - start_time
        logger.info(f"✅ Import completed in {import_time:.2f} seconds")
        
        # Create manager with timeout
        logger.info("Step 2: Creating manager...")
        start_time = time.time()
        
        manager = AggregateAnalysisManager(output_dir="results/aggregate_analysis")
        manager_time = time.time() - start_time
        logger.info(f"✅ Manager created in {manager_time:.2f} seconds")
        
        # Load config4 with timeout
        logger.info("Step 3: Loading config4...")
        start_time = time.time()
        
        config4 = manager.experimental_configs[4]
        config_time = time.time() - start_time
        logger.info(f"✅ Config4 loaded in {config_time:.2f} seconds")
        
        # Test single experiment with 5-minute timeout
        logger.info("Step 4: Testing single experiment (5-minute timeout)...")
        start_time = time.time()
        
        # Set a timeout using threading
        import threading
        import signal
        
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
        timeout_seconds = 300  # 5 minutes
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            logger.error(f"❌ Experiment timed out after {timeout_seconds} seconds")
            logger.error("   This indicates the external machine has a fundamental issue")
            return False
        elif error:
            logger.error(f"❌ Experiment failed with error: {error}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return False
        else:
            experiment_time = time.time() - start_time
            logger.info(f"✅ Single experiment completed in {experiment_time:.2f} seconds")
            logger.info(f"   Result type: {type(result)}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed in timeout test: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False

def test_memory_usage():
    """Test if memory usage is causing issues."""
    
    logger.info("\n🔍 MEMORY USAGE TEST")
    logger.info("=" * 50)
    
    try:
        import numpy as np
        
        # Test large array allocation
        logger.info("Testing large array allocation...")
        
        # Try to allocate 1GB of memory
        try:
            large_array = np.zeros((1024, 1024, 256), dtype=np.float64)  # ~2GB
            logger.info("✅ Successfully allocated ~2GB array")
            del large_array
        except MemoryError:
            logger.error("❌ Failed to allocate 2GB array - memory issue")
            return False
        
        # Test multiple smaller allocations
        logger.info("Testing multiple smaller allocations...")
        arrays = []
        for i in range(10):
            try:
                arr = np.zeros((512, 512, 128), dtype=np.float64)  # ~256MB each
                arrays.append(arr)
                logger.info(f"✅ Allocated array {i+1}/10 (~256MB)")
            except MemoryError:
                logger.error(f"❌ Failed to allocate array {i+1}/10")
                return False
        
        # Clean up
        del arrays
        logger.info("✅ Memory test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory test failed: {e}")
        return False

def test_multiprocessing_setup():
    """Test multiprocessing setup on external machine."""
    
    logger.info("\n🔍 MULTIPROCESSING SETUP TEST")
    logger.info("=" * 50)
    
    try:
        import multiprocessing as mp
        
        # Check multiprocessing start method
        start_method = mp.get_start_method()
        logger.info(f"Multiprocessing start method: {start_method}")
        
        # Test simple multiprocessing
        def simple_worker(x):
            return x * 2
        
        logger.info("Testing simple multiprocessing...")
        with mp.Pool(processes=2) as pool:
            results = pool.map(simple_worker, [1, 2, 3, 4])
            logger.info(f"✅ Simple multiprocessing works: {results}")
        
        # Test with timeout
        logger.info("Testing multiprocessing with timeout...")
        with mp.Pool(processes=2) as pool:
            try:
                results = pool.map_async(simple_worker, [1, 2, 3, 4])
                results.get(timeout=30)  # 30 second timeout
                logger.info("✅ Multiprocessing with timeout works")
            except Exception as e:
                logger.error(f"❌ Multiprocessing timeout test failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Multiprocessing test failed: {e}")
        return False

def test_file_system():
    """Test file system operations."""
    
    logger.info("\n🔍 FILE SYSTEM TEST")
    logger.info("=" * 50)
    
    try:
        # Test directory creation
        test_dir = Path("test_external_diagnostic")
        test_dir.mkdir(exist_ok=True)
        logger.info("✅ Directory creation works")
        
        # Test file writing
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")
        logger.info("✅ File writing works")
        
        # Test file reading
        content = test_file.read_text()
        logger.info("✅ File reading works")
        
        # Test large file operations
        logger.info("Testing large file operations...")
        large_content = "x" * (1024 * 1024)  # 1MB
        large_file = test_dir / "large_test.txt"
        large_file.write_text(large_content)
        logger.info("✅ Large file writing works")
        
        # Clean up
        test_file.unlink()
        large_file.unlink()
        test_dir.rmdir()
        logger.info("✅ File system test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ File system test failed: {e}")
        return False

def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(description="External vs Local machine diagnostic")
    parser.add_argument("--skip-experiment", action="store_true", 
                       help="Skip the long experiment test")
    
    args = parser.parse_args()
    
    logger.info("🚀 EXTERNAL VS LOCAL MACHINE DIAGNOSTIC")
    logger.info("This will help identify why external machine can't complete even one seed")
    
    # Run all tests
    tests = [
        ("System Differences", check_system_differences),
        ("Memory Usage", test_memory_usage),
        ("Multiprocessing Setup", test_multiprocessing_setup),
        ("File System", test_file_system),
    ]
    
    if not args.skip_experiment:
        tests.append(("Config4 Timeout Test", test_config4_with_timeout))
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    logger.info(f"Tests passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    if passed == total:
        logger.info("\n🎉 All tests passed!")
        logger.info("The issue might be in the specific experiment logic")
    else:
        logger.info(f"\n⚠️ {total - passed} tests failed")
        logger.info("This explains why the external machine can't complete experiments")

if __name__ == "__main__":
    main() 