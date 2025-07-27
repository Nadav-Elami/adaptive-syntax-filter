#!/usr/bin/env python3
"""Diagnostic script to troubleshoot external machine multiprocessing issues.

This script helps identify why the external machine gets stuck on the first seed
when the same script worked fine for config3.
"""

import logging
import sys
import time
import multiprocessing as mp
from pathlib import Path
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diagnose_external_machine.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_system_resources():
    """Check system resources and identify potential bottlenecks."""
    logger.info("=== SYSTEM RESOURCE DIAGNOSTIC ===")
    
    # CPU info
    cpu_count = mp.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU Cores: {cpu_count}")
    logger.info(f"CPU Usage: {cpu_percent:.1f}%")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    logger.info(f"Memory Usage: {memory.percent:.1f}%")
    
    # Disk info
    disk = psutil.disk_usage('/')
    logger.info(f"Disk Free: {disk.free / (1024**3):.1f} GB")
    logger.info(f"Disk Usage: {disk.percent:.1f}%")
    
    # Process info
    current_process = psutil.Process()
    logger.info(f"Current Process Memory: {current_process.memory_info().rss / (1024**2):.1f} MB")
    
    return {
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'memory_available_gb': memory.available / (1024**3),
        'memory_percent': memory.percent,
        'disk_free_gb': disk.free / (1024**3)
    }

def test_basic_multiprocessing():
    """Test basic multiprocessing functionality."""
    logger.info("=== BASIC MULTIPROCESSING TEST ===")
    
    def simple_worker(x):
        """Simple worker function."""
        time.sleep(0.1)  # Simulate some work
        return x * 2
    
    try:
        logger.info("Testing basic multiprocessing with 4 workers...")
        with mp.Pool(4) as pool:
            results = pool.map(simple_worker, range(10))
        logger.info(f"Basic multiprocessing test PASSED: {results}")
        return True
    except Exception as e:
        logger.error(f"Basic multiprocessing test FAILED: {e}")
        return False

def test_config_loading():
    """Test if config loading works in multiprocessing."""
    logger.info("=== CONFIG LOADING TEST ===")
    
    def config_worker(config_id):
        """Worker that loads a config."""
        try:
            from aggregate_analysis import AggregateAnalysisManager
            manager = AggregateAnalysisManager()
            manager._load_single_config(config_id)
            return f"Config {config_id} loaded successfully"
        except Exception as e:
            return f"Config {config_id} failed: {e}"
    
    try:
        logger.info("Testing config loading in multiprocessing...")
        with mp.Pool(2) as pool:
            results = pool.map(config_worker, [1, 2])
        for result in results:
            logger.info(f"   {result}")
        return True
    except Exception as e:
        logger.error(f"Config loading test FAILED: {e}")
        return False

def test_single_experiment():
    """Test running a single experiment in multiprocessing."""
    logger.info("=== SINGLE EXPERIMENT TEST ===")
    
    def experiment_worker(args):
        """Worker that runs a single experiment."""
        config_id, seed = args
        try:
            from aggregate_analysis import AggregateAnalysisManager
            manager = AggregateAnalysisManager()
            manager._load_single_config(config_id)
            result = manager.run_single_experiment(config_id, seed)
            return f"Experiment config_id={config_id}, seed={seed} completed"
        except Exception as e:
            return f"Experiment config_id={config_id}, seed={seed} failed: {e}"
    
    try:
        logger.info("Testing single experiment in multiprocessing...")
        with mp.Pool(2) as pool:
            args_list = [(4, 0), (4, 1)]  # Test config 4 with seeds 0, 1
            results = pool.map(experiment_worker, args_list)
        for result in results:
            logger.info(f"   {result}")
        return True
    except Exception as e:
        logger.error(f"Single experiment test FAILED: {e}")
        return False

def test_worker_initialization():
    """Test the worker initialization that might be causing issues."""
    logger.info("=== WORKER INITIALIZATION TEST ===")
    
    # Global variables for worker initialization
    _CONFIG_CACHE = {}
    _DIRECTORIES_CREATED = False
    
    def _initialize_worker_process(config_id: int, output_dir: str):
        """Test the worker initialization function."""
        global _CONFIG_CACHE, _DIRECTORIES_CREATED
        
        try:
            logger.info(f"Worker initializing for config_id={config_id}")
            
            # Load and cache config once per worker
            if config_id not in _CONFIG_CACHE:
                from aggregate_analysis import AggregateAnalysisManager
                manager = AggregateAnalysisManager(output_dir=output_dir)
                manager._load_single_config(config_id)
                _CONFIG_CACHE[config_id] = {
                    'config': manager.experimental_configs[config_id],
                    'output_dir': manager.output_dir
                }
                logger.info(f"Config {config_id} cached successfully")
            
            # Ensure directories exist (thread-safe)
            if not _DIRECTORIES_CREATED:
                output_path = Path(output_dir)
                (output_path / f"config_{config_id}_data").mkdir(parents=True, exist_ok=True)
                _DIRECTORIES_CREATED = True
                logger.info(f"Directories created for config {config_id}")
                
        except Exception as e:
            logger.error(f"Worker initialization failed: {e}")
            raise
    
    def _test_experiment_wrapper(args):
        """Test wrapper function."""
        config_id, seed, output_dir = args
        
        try:
            logger.info(f"Worker processing config_id={config_id}, seed={seed}")
            
            # Use cached config instead of creating new manager
            if config_id not in _CONFIG_CACHE:
                _initialize_worker_process(config_id, output_dir)
            
            # Import here to avoid pickle issues
            from aggregate_analysis import AggregateAnalysisManager
            
            # Create minimal manager instance
            manager = AggregateAnalysisManager(output_dir=output_dir)
            manager.experimental_configs[config_id] = _CONFIG_CACHE[config_id]['config']
            
            # Run a minimal experiment (just return success for testing)
            logger.info(f"Experiment config_id={config_id}, seed={seed} would run here")
            return f"SUCCESS: config_id={config_id}, seed={seed}"
            
        except Exception as e:
            logger.error(f"Failed experiment config_id={config_id}, seed={seed}: {e}")
            return f"FAILED: config_id={config_id}, seed={seed} - {e}"
    
    try:
        logger.info("Testing worker initialization with 2 workers...")
        output_dir = "results/aggregate_analysis"
        
        with mp.Pool(processes=2, 
                     initializer=_initialize_worker_process,
                     initargs=(4, output_dir)) as pool:
            
            args_list = [(4, 0, output_dir), (4, 1, output_dir)]
            results = pool.map(_test_experiment_wrapper, args_list)
            
        for result in results:
            logger.info(f"   {result}")
        return True
        
    except Exception as e:
        logger.error(f"Worker initialization test FAILED: {e}")
        return False

def check_environment_differences():
    """Check for environment differences that might cause issues."""
    logger.info("=== ENVIRONMENT DIFFERENCES ===")
    
    # Python version
    logger.info(f"Python Version: {sys.version}")
    
    # Multiprocessing start method
    start_method = mp.get_start_method()
    logger.info(f"Multiprocessing Start Method: {start_method}")
    
    # Available start methods
    available_methods = mp.get_all_start_methods()
    logger.info(f"Available Start Methods: {available_methods}")
    
    # Environment variables
    relevant_vars = ['PYTHONPATH', 'PATH', 'HOME', 'USER']
    for var in relevant_vars:
        value = os.environ.get(var, 'NOT_SET')
        logger.info(f"Environment {var}: {value}")
    
    # Working directory
    logger.info(f"Working Directory: {os.getcwd()}")
    
    return {
        'start_method': start_method,
        'available_methods': available_methods
    }

def recommend_solutions(resources, mp_test, config_test, exp_test, init_test, env_info):
    """Recommend solutions based on diagnostic results."""
    logger.info("=== RECOMMENDATIONS ===")
    
    issues = []
    solutions = []
    
    # Check resource constraints
    if resources['memory_available_gb'] < 10:
        issues.append("Low available memory")
        solutions.append("Reduce worker count to 4-8 workers")
    
    if resources['cpu_percent'] > 80:
        issues.append("High CPU usage")
        solutions.append("Wait for system to be less busy")
    
    # Check multiprocessing issues
    if not mp_test:
        issues.append("Basic multiprocessing failed")
        solutions.append("Check Python installation and multiprocessing support")
    
    if not config_test:
        issues.append("Config loading in multiprocessing failed")
        solutions.append("Check file permissions and config file accessibility")
    
    if not exp_test:
        issues.append("Single experiment in multiprocessing failed")
        solutions.append("Check dependencies and import issues")
    
    if not init_test:
        issues.append("Worker initialization failed")
        solutions.append("Use simpler worker initialization or reduce worker count")
    
    # Check start method
    if env_info['start_method'] == 'fork' and 'linux' not in sys.platform.lower():
        issues.append("Fork start method on non-Linux platform")
        solutions.append("Try setting start method to 'spawn': mp.set_start_method('spawn')")
    
    # Provide specific recommendations
    if issues:
        logger.warning("ISSUES DETECTED:")
        for issue in issues:
            logger.warning(f"   - {issue}")
        
        logger.info("RECOMMENDED SOLUTIONS:")
        for solution in solutions:
            logger.info(f"   - {solution}")
        
        logger.info("IMMEDIATE ACTIONS:")
        logger.info("   1. Try with fewer workers: --parallel 4")
        logger.info("   2. Use the deadlock-free version: aggregate_analysis_deadlock_free.py")
        logger.info("   3. Check if any processes are stuck and kill them")
        
    else:
        logger.info("No obvious issues detected. Try the deadlock-free version.")

def main():
    """Run all diagnostics."""
    logger.info("STARTING EXTERNAL MACHINE DIAGNOSTIC")
    logger.info("This will help identify why the external machine gets stuck")
    
    # Run all diagnostics
    resources = check_system_resources()
    mp_test = test_basic_multiprocessing()
    config_test = test_config_loading()
    exp_test = test_single_experiment()
    init_test = test_worker_initialization()
    env_info = check_environment_differences()
    
    # Provide recommendations
    recommend_solutions(resources, mp_test, config_test, exp_test, init_test, env_info)
    
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("Check the log file for detailed results")

if __name__ == "__main__":
    main() 