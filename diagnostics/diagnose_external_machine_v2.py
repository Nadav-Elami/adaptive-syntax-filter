#!/usr/bin/env python3
"""Enhanced diagnostic script to find the real issue on external machine.

This version tests more specific scenarios to identify what's actually causing
the machine to get stuck when it worked fine for config3 before.
"""

import logging
import sys
import time
import multiprocessing as mp
from pathlib import Path
import psutil
import os
import subprocess
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diagnose_external_machine_v2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_config4_specific():
    """Test if config4 specifically has issues that config3 didn't."""
    logger.info("=== CONFIG4 SPECIFIC TEST ===")
    
    try:
        from aggregate_analysis import AggregateAnalysisManager
        manager = AggregateAnalysisManager()
        
        # Test loading config4
        logger.info("Testing config4 loading...")
        manager._load_single_config(4)
        logger.info("✅ Config4 loads successfully")
        
        # Test a single experiment with config4
        logger.info("Testing single config4 experiment...")
        result = manager.run_single_experiment(4, 0)
        logger.info(f"✅ Single config4 experiment completed in {result.runtime_seconds:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config4 test failed: {e}")
        return False

def test_config4_vs_config3():
    """Compare config4 and config3 to see if there are differences."""
    logger.info("=== CONFIG4 vs CONFIG3 COMPARISON ===")
    
    try:
        from aggregate_analysis import AggregateAnalysisManager
        manager = AggregateAnalysisManager()
        
        # Load both configs
        manager._load_single_config(3)
        config3 = manager.experimental_configs[3]
        logger.info(f"Config3: {config3.description}")
        
        manager._load_single_config(4)
        config4 = manager.experimental_configs[4]
        logger.info(f"Config4: {config4.description}")
        
        # Compare key parameters
        logger.info("Comparing parameters:")
        logger.info(f"  Config3 alphabet: {config3.alphabet}")
        logger.info(f"  Config4 alphabet: {config4.alphabet}")
        logger.info(f"  Config3 order: {config3.order}")
        logger.info(f"  Config4 order: {config4.order}")
        logger.info(f"  Config3 evolution: {config3.evolution_type}")
        logger.info(f"  Config4 evolution: {config4.evolution_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config comparison failed: {e}")
        return False

def test_multiprocessing_with_timeout():
    """Test multiprocessing with a timeout to see if it gets stuck."""
    logger.info("=== MULTIPROCESSING TIMEOUT TEST ===")
    
    def simple_worker(x):
        """Simple worker that just returns the input."""
        time.sleep(0.1)
        return x * 2
    
    try:
        logger.info("Testing multiprocessing with 2 workers and 10 second timeout...")
        
        # Set up timeout
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Multiprocessing test timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        with mp.Pool(2) as pool:
            results = pool.map(simple_worker, range(10))
        
        signal.alarm(0)  # Cancel timeout
        logger.info(f"✅ Multiprocessing test completed: {results}")
        return True
        
    except TimeoutError:
        logger.error("❌ Multiprocessing test TIMED OUT - this is the issue!")
        return False
    except Exception as e:
        logger.error(f"❌ Multiprocessing test failed: {e}")
        return False

def test_real_experiment_with_timeout():
    """Test a real experiment with timeout to see if it gets stuck."""
    logger.info("=== REAL EXPERIMENT TIMEOUT TEST ===")
    
    def experiment_worker(args):
        """Worker that runs a real experiment."""
        config_id, seed = args
        try:
            from aggregate_analysis import AggregateAnalysisManager
            manager = AggregateAnalysisManager()
            manager._load_single_config(config_id)
            result = manager.run_single_experiment(config_id, seed)
            return f"SUCCESS: config_id={config_id}, seed={seed}"
        except Exception as e:
            return f"FAILED: config_id={config_id}, seed={seed} - {e}"
    
    try:
        logger.info("Testing real experiment with 30 second timeout...")
        
        # Set up timeout
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Real experiment test timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        with mp.Pool(2) as pool:
            args_list = [(4, 0), (4, 1)]
            results = pool.map(experiment_worker, args_list)
        
        signal.alarm(0)  # Cancel timeout
        for result in results:
            logger.info(f"   {result}")
        logger.info("✅ Real experiment test completed")
        return True
        
    except TimeoutError:
        logger.error("❌ Real experiment test TIMED OUT - this confirms the issue!")
        return False
    except Exception as e:
        logger.error(f"❌ Real experiment test failed: {e}")
        return False

def check_system_load():
    """Check if the system is under heavy load."""
    logger.info("=== SYSTEM LOAD CHECK ===")
    
    # CPU load
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU Usage: {cpu_percent:.1f}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    logger.info(f"Memory Usage: {memory.percent:.1f}%")
    
    # Number of processes
    process_count = len(psutil.pids())
    logger.info(f"Total Processes: {process_count}")
    
    # Check for stuck Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    logger.info(f"Python Processes: {len(python_processes)}")
    for proc in python_processes:
        logger.info(f"   PID {proc['pid']}: {' '.join(proc['cmdline'][:3])}")
    
    # Check if system is overloaded
    if cpu_percent > 80:
        logger.warning("⚠️ High CPU usage - system may be overloaded")
        return False
    elif memory.percent > 90:
        logger.warning("⚠️ High memory usage - system may be overloaded")
        return False
    else:
        logger.info("✅ System load looks normal")
        return True

def test_file_system():
    """Test if there are file system issues."""
    logger.info("=== FILE SYSTEM TEST ===")
    
    try:
        # Test directory creation
        test_dir = Path("test_multiprocessing")
        test_dir.mkdir(exist_ok=True)
        logger.info("✅ Directory creation works")
        
        # Test file writing
        test_file = test_dir / "test.txt"
        test_file.write_text("test")
        logger.info("✅ File writing works")
        
        # Test file reading
        content = test_file.read_text()
        logger.info("✅ File reading works")
        
        # Cleanup
        test_file.unlink()
        test_dir.rmdir()
        logger.info("✅ File cleanup works")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File system test failed: {e}")
        return False

def check_python_environment():
    """Check Python environment for issues."""
    logger.info("=== PYTHON ENVIRONMENT CHECK ===")
    
    # Python version
    logger.info(f"Python Version: {sys.version}")
    
    # Multiprocessing start method
    start_method = mp.get_start_method()
    logger.info(f"Multiprocessing Start Method: {start_method}")
    
    # Available start methods
    available_methods = mp.get_all_start_methods()
    logger.info(f"Available Start Methods: {available_methods}")
    
    # Check for any stuck processes
    logger.info("Checking for stuck processes...")
    stuck_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
        try:
            if 'python' in proc.info['name'].lower() and 'aggregate' in ' '.join(proc.info['cmdline']):
                stuck_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if stuck_processes:
        logger.warning(f"⚠️ Found {len(stuck_processes)} potentially stuck processes:")
        for proc in stuck_processes:
            logger.warning(f"   PID {proc['pid']}: {proc['status']} - {' '.join(proc['cmdline'][:3])}")
    else:
        logger.info("✅ No stuck processes found")
    
    return len(stuck_processes) == 0

def recommend_next_steps(config4_test, config_comparison, mp_test, real_test, system_ok, fs_test, env_ok):
    """Recommend next steps based on diagnostic results."""
    logger.info("=== DIAGNOSTIC SUMMARY ===")
    
    issues = []
    solutions = []
    
    if not config4_test:
        issues.append("Config4 has specific issues")
        solutions.append("Check config4 configuration file")
    
    if not mp_test:
        issues.append("Basic multiprocessing times out")
        solutions.append("System may be overloaded or has resource limits")
    
    if not real_test:
        issues.append("Real experiments time out")
        solutions.append("The actual experiment code is getting stuck")
    
    if not system_ok:
        issues.append("System is under heavy load")
        solutions.append("Wait for system to be less busy or reduce worker count")
    
    if not fs_test:
        issues.append("File system issues")
        solutions.append("Check disk space and permissions")
    
    if not env_ok:
        issues.append("Stuck processes detected")
        solutions.append("Kill stuck processes before running analysis")
    
    if issues:
        logger.warning("ISSUES DETECTED:")
        for issue in issues:
            logger.warning(f"   - {issue}")
        
        logger.info("RECOMMENDED SOLUTIONS:")
        for solution in solutions:
            logger.info(f"   - {solution}")
        
        logger.info("IMMEDIATE ACTIONS:")
        logger.info("   1. Kill any stuck processes: pkill -f aggregate_analysis")
        logger.info("   2. Wait for system to be less busy")
        logger.info("   3. Try with very few workers: --parallel 2")
        logger.info("   4. Check if config4 has different requirements than config3")
        
    else:
        logger.info("✅ No obvious issues detected")
        logger.info("💡 Try running with --parallel 2 to see if it's a worker count issue")

def main():
    """Run enhanced diagnostics."""
    logger.info("STARTING ENHANCED EXTERNAL MACHINE DIAGNOSTIC V2")
    logger.info("This will help identify why config4 gets stuck when config3 worked")
    
    # Run all diagnostics
    config4_test = test_config4_specific()
    config_comparison = test_config4_vs_config3()
    mp_test = test_multiprocessing_with_timeout()
    real_test = test_real_experiment_with_timeout()
    system_ok = check_system_load()
    fs_test = test_file_system()
    env_ok = check_python_environment()
    
    # Provide recommendations
    recommend_next_steps(config4_test, config_comparison, mp_test, real_test, system_ok, fs_test, env_ok)
    
    logger.info("ENHANCED DIAGNOSTIC COMPLETE")
    logger.info("Check the log file for detailed results")

if __name__ == "__main__":
    main() 