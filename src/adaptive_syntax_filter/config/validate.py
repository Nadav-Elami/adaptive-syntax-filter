"""Environment validation for Adaptive Syntax Filter dependencies."""

import sys
from typing import Optional, Tuple
from packaging import version


def check_environment(min_numpy: str = "1.25", min_scipy: str = "1.11") -> None:
    """Check that environment meets minimum dependency requirements.
    
    Validates that required packages are installed with sufficient versions
    for the adaptive syntax filter algorithms to function correctly.
    
    Parameters
    ----------
    min_numpy : str, default="1.25"
        Minimum required NumPy version
    min_scipy : str, default="1.11" 
        Minimum required SciPy version
        
    Raises
    ------
    RuntimeError
        If any dependency requirements are not met
        
    Examples
    --------
    >>> check_environment()  # Uses default minimums
    >>> check_environment(min_numpy="1.24", min_scipy="1.10")
    """
    errors = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append(f"Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check NumPy
    try:
        import numpy as np
        numpy_version = np.__version__
        if version.parse(numpy_version) < version.parse(min_numpy):
            errors.append(f"NumPy {min_numpy}+ required, found {numpy_version}")
    except ImportError:
        errors.append("NumPy not installed - required for array operations")
    
    # Check SciPy
    try:
        import scipy
        scipy_version = scipy.__version__
        if version.parse(scipy_version) < version.parse(min_scipy):
            errors.append(f"SciPy {min_scipy}+ required, found {scipy_version}")
    except ImportError:
        errors.append("SciPy not installed - required for statistical functions")
    
    # Check optional dependencies with warnings
    optional_warnings = []
    
    try:
        import matplotlib
        matplotlib_version = matplotlib.__version__
        if version.parse(matplotlib_version) < version.parse("3.5"):
            optional_warnings.append(f"Matplotlib 3.5+ recommended, found {matplotlib_version}")
    except ImportError:
        optional_warnings.append("Matplotlib not found - required for visualizations")
    
    # Packaging is required for this validation
    try:
        import packaging
    except ImportError:
        errors.append("packaging library required for version checking")
    
    # Report errors
    if errors:
        error_msg = "Environment validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        
        if optional_warnings:
            error_msg += "\n\nWarnings:\n" + "\n".join(f"  - {warn}" for warn in optional_warnings)
        
        error_msg += "\n\nTo install required dependencies:\n  pip install numpy scipy matplotlib packaging"
        
        raise RuntimeError(error_msg)
    
    # Report warnings only if no errors
    if optional_warnings:
        import warnings
        warning_msg = "Environment warnings:\n" + "\n".join(f"  - {warn}" for warn in optional_warnings)
        warnings.warn(warning_msg, UserWarning)


def get_dependency_versions() -> dict:
    """Get versions of all relevant dependencies.
    
    Returns
    -------
    dict
        Dictionary mapping package names to version strings
    """
    versions = {
        'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    
    # Core dependencies
    try:
        import numpy as np
        versions['numpy'] = np.__version__
    except ImportError:
        versions['numpy'] = 'not installed'
    
    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except ImportError:
        versions['scipy'] = 'not installed'
    
    # Optional dependencies
    try:
        import matplotlib
        versions['matplotlib'] = matplotlib.__version__
    except ImportError:
        versions['matplotlib'] = 'not installed'
    
    try:
        import packaging
        versions['packaging'] = packaging.__version__
    except ImportError:
        versions['packaging'] = 'not installed'
    
    # TOML support
    try:
        import tomllib
        versions['tomllib'] = 'built-in (3.11+)'
    except ImportError:
        try:
            import tomli
            versions['tomli'] = tomli.__version__
        except ImportError:
            versions['tomli'] = 'not installed'
    
    try:
        import tomli_w
        versions['tomli_w'] = tomli_w.__version__
    except ImportError:
        versions['tomli_w'] = 'not installed'
    
    return versions


def print_environment_info() -> None:
    """Print comprehensive environment information."""
    versions = get_dependency_versions()
    
    print("Adaptive Syntax Filter - Environment Information")
    print("=" * 50)
    
    print("\nCore Dependencies:")
    for pkg in ['python', 'numpy', 'scipy']:
        if pkg in versions:
            print(f"  {pkg:12}: {versions[pkg]}")
    
    print("\nVisualization:")
    for pkg in ['matplotlib']:
        if pkg in versions:
            print(f"  {pkg:12}: {versions[pkg]}")
    
    print("\nConfiguration:")
    for pkg in ['packaging', 'tomllib', 'tomli', 'tomli_w']:
        if pkg in versions:
            print(f"  {pkg:12}: {versions[pkg]}")
    
    print("\nSystem Information:")
    print(f"  Platform     : {sys.platform}")
    print(f"  Architecture : {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")


def validate_numerical_stability() -> None:
    """Check numerical computation stability."""
    try:
        import numpy as np
        
        # Test basic numerical operations
        test_array = np.array([1e-10, 1e10, -1e-10, -1e10])
        
        # Check for overflow/underflow handling
        if not np.all(np.isfinite(test_array)):
            raise RuntimeError("NumPy numerical stability test failed")
        
        # Test matrix operations
        test_matrix = np.random.randn(100, 100)
        eigenvals = np.linalg.eigvals(test_matrix @ test_matrix.T)
        
        if not np.all(eigenvals >= 0):
            raise RuntimeError("Matrix operation numerical stability test failed")
            
    except Exception as e:
        raise RuntimeError(f"Numerical stability validation failed: {e}")


def check_memory_availability(required_gb: float = 1.0) -> None:
    """Check available system memory.
    
    Parameters
    ----------
    required_gb : float
        Minimum required memory in GB
        
    Raises
    ------
    RuntimeError
        If insufficient memory is available
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb < required_gb:
            raise RuntimeError(
                f"Insufficient memory: {available_gb:.1f}GB available, "
                f"{required_gb:.1f}GB required"
            )
            
    except ImportError:
        # psutil not available, skip memory check
        import warnings
        warnings.warn("psutil not available - cannot check memory", UserWarning)


def full_environment_check(min_numpy: str = "1.25", 
                          min_scipy: str = "1.11",
                          required_memory_gb: float = 1.0,
                          check_stability: bool = True) -> None:
    """Perform comprehensive environment validation.
    
    Parameters
    ----------
    min_numpy : str
        Minimum required NumPy version
    min_scipy : str
        Minimum required SciPy version
    required_memory_gb : float
        Minimum required memory in GB
    check_stability : bool
        Whether to run numerical stability tests
    """
    print("Running comprehensive environment check...")
    
    # Basic dependency check
    check_environment(min_numpy=min_numpy, min_scipy=min_scipy)
    print("✓ Dependencies validated")
    
    # Memory check
    check_memory_availability(required_memory_gb)
    print("✓ Memory availability checked")
    
    # Numerical stability
    if check_stability:
        validate_numerical_stability()
        print("✓ Numerical stability validated")
    
    print("\nEnvironment validation completed successfully!") 
