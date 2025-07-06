"""Tests for environment validation functionality.

Tests the environment validation system that checks dependencies,
versions, and system requirements for the Adaptive Syntax Filter.
"""

import pytest
import sys
import warnings
from unittest.mock import patch, MagicMock
from io import StringIO

from src.adaptive_syntax_filter.config.validate import (
    check_environment,
    get_dependency_versions,
    print_environment_info,
    validate_numerical_stability,
    check_memory_availability,
    full_environment_check
)


class TestEnvironmentChecking:
    """Test suite for environment validation functions."""
    
    def test_check_environment_success(self):
        """Test successful environment validation."""
        # Should not raise any exceptions with current environment
        check_environment()
        
        # Test with custom version requirements
        check_environment(min_numpy="1.20", min_scipy="1.7")
    
    @patch('src.adaptive_syntax_filter.config.validate.version')
    def test_check_environment_numpy_version_failure(self, mock_version):
        """Test NumPy version validation failure."""
        # Mock version.parse to simulate old NumPy version
        mock_version.parse.side_effect = lambda v: (
            MagicMock(__lt__=lambda self, other: v == "1.20.0")
            if v == "1.20.0" else 
            MagicMock(__lt__=lambda self, other: False)
        )
        
        # Mock numpy to have old version
        with patch.dict('sys.modules', {'numpy': MagicMock(__version__="1.20.0")}):
            with pytest.raises(RuntimeError) as exc_info:
                check_environment(min_numpy="1.25")
            
            error_msg = str(exc_info.value)
            assert "NumPy 1.25+ required" in error_msg
            assert "found 1.20.0" in error_msg
    
    @patch('src.adaptive_syntax_filter.config.validate.version')
    def test_check_environment_scipy_version_failure(self, mock_version):
        """Test SciPy version validation failure."""
        # Mock version parsing for old SciPy
        mock_version.parse.side_effect = lambda v: (
            MagicMock(__lt__=lambda self, other: v == "1.8.0")
            if v == "1.8.0" else 
            MagicMock(__lt__=lambda self, other: False)
        )
        
        # Mock scipy to have old version
        with patch.dict('sys.modules', {'scipy': MagicMock(__version__="1.8.0")}):
            with pytest.raises(RuntimeError) as exc_info:
                check_environment(min_scipy="1.11")
            
            error_msg = str(exc_info.value)
            assert "SciPy 1.11+ required" in error_msg
            assert "found 1.8.0" in error_msg
    
    @patch('src.adaptive_syntax_filter.config.validate.version')
    def test_check_environment_matplotlib_warnings(self, mock_version):
        """Test matplotlib version warnings."""
        # Mock version parsing for old matplotlib
        mock_version.parse.side_effect = lambda v: (
            MagicMock(__lt__=lambda self, other: v == "3.4.0")
            if v == "3.4.0" else 
            MagicMock(__lt__=lambda self, other: False)
        )
        
        with patch.dict('sys.modules', {'matplotlib': MagicMock(__version__="3.4.0")}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_environment()
                
                # Should have warning about matplotlib version
                assert len(w) > 0
                warning_msg = str(w[0].message)
                assert "Matplotlib 3.5+ recommended" in warning_msg
                assert "found 3.4.0" in warning_msg
    
    def test_check_environment_matplotlib_missing_warning(self):
        """Test warning when matplotlib is missing."""
        matplotlib_backup = sys.modules.get('matplotlib')
        if 'matplotlib' in sys.modules:
            del sys.modules['matplotlib']
        
        try:
            with patch.dict('sys.modules', {'matplotlib': None}, clear=False):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    check_environment()
                    
                    # Should have warning about missing matplotlib
                    assert len(w) > 0
                    warning_msg = str(w[0].message)
                    assert "Matplotlib not found" in warning_msg
        finally:
            if matplotlib_backup:
                sys.modules['matplotlib'] = matplotlib_backup


class TestDependencyVersions:
    """Test suite for dependency version reporting."""
    
    def test_get_dependency_versions_success(self):
        """Test successful dependency version gathering."""
        versions = get_dependency_versions()
        
        # Check that essential keys are present
        assert 'python' in versions
        assert 'numpy' in versions
        assert 'scipy' in versions
        assert 'matplotlib' in versions
        assert 'packaging' in versions
        
        # Python version should be parseable
        python_version = versions['python']
        assert isinstance(python_version, str)
        assert len(python_version.split('.')) >= 2
    
    def test_get_dependency_versions_with_missing_packages(self):
        """Test version gathering with missing optional packages."""
        # Mock missing optional packages
        with patch.dict('sys.modules', {'tomllib': None, 'tomli': None, 'tomli_w': None}, clear=False):
            versions = get_dependency_versions()
            
            # Should still work and report missing packages
            assert 'tomli' in versions
            assert versions['tomli'] == 'not installed'
            assert 'tomli_w' in versions 
            assert versions['tomli_w'] == 'not installed'
    
    def test_get_dependency_versions_tomllib_builtin(self):
        """Test detection of built-in tomllib (Python 3.11+)."""
        # Mock tomllib as available (simulating Python 3.11+)
        mock_tomllib = MagicMock()
        with patch.dict('sys.modules', {'tomllib': mock_tomllib}):
            versions = get_dependency_versions()
            
            assert 'tomllib' in versions
            assert 'built-in' in versions['tomllib']


class TestEnvironmentReporting:
    """Test suite for environment information reporting."""
    
    def test_print_environment_info(self, capsys):
        """Test environment information printing."""
        print_environment_info()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that key sections are present
        assert "Adaptive Syntax Filter - Environment Information" in output
        assert "Core Dependencies:" in output
        assert "Visualization:" in output
        assert "Configuration:" in output
        assert "System Information:" in output
        
        # Check that key packages are listed
        assert "python" in output
        assert "numpy" in output
        assert "scipy" in output
        assert "matplotlib" in output
        
        # Check system info
        assert "Platform" in output
        assert "Architecture" in output


class TestNumericalStability:
    """Test suite for numerical stability validation."""
    
    def test_validate_numerical_stability_success(self):
        """Test successful numerical stability validation."""
        # Should not raise any exceptions with healthy NumPy
        validate_numerical_stability()
    
    @patch('numpy.isfinite')
    def test_validate_numerical_stability_array_failure(self, mock_isfinite):
        """Test numerical stability failure with array operations."""
        mock_isfinite.return_value = False
        
        with pytest.raises(RuntimeError) as exc_info:
            validate_numerical_stability()
        
        error_msg = str(exc_info.value)
        assert "NumPy numerical stability test failed" in error_msg
    
    @patch('numpy.linalg.eigvals')
    def test_validate_numerical_stability_matrix_failure(self, mock_eigvals):
        """Test numerical stability failure with matrix operations."""
        # Mock negative eigenvalues (impossible for A^T A)
        import numpy as np
        mock_eigvals.return_value = np.array([-1.0, 0.5, 1.0])
        
        with pytest.raises(RuntimeError) as exc_info:
            validate_numerical_stability()
        
        error_msg = str(exc_info.value)
        assert "Matrix operation numerical stability test failed" in error_msg
    
    @patch('numpy.random.randn')
    def test_validate_numerical_stability_exception_handling(self, mock_randn):
        """Test handling of unexpected exceptions in stability validation."""
        mock_randn.side_effect = Exception("Mock numpy error")
        
        with pytest.raises(RuntimeError) as exc_info:
            validate_numerical_stability()
        
        error_msg = str(exc_info.value)
        assert "Numerical stability validation failed" in error_msg
        assert "Mock numpy error" in error_msg


class TestMemoryValidation:
    """Test suite for memory availability checking."""
    
    @patch('psutil.virtual_memory')
    def test_check_memory_availability_success(self, mock_virtual_memory):
        """Test successful memory availability check."""
        # Mock sufficient memory (2GB available)
        mock_memory = MagicMock()
        mock_memory.available = 2 * 1024**3  # 2GB in bytes
        mock_virtual_memory.return_value = mock_memory
        
        # Should not raise with 1GB requirement
        check_memory_availability(required_gb=1.0)
    
    @patch('psutil.virtual_memory')
    def test_check_memory_availability_insufficient(self, mock_virtual_memory):
        """Test insufficient memory handling."""
        # Mock insufficient memory (0.5GB available)
        mock_memory = MagicMock()
        mock_memory.available = 0.5 * 1024**3  # 0.5GB in bytes
        mock_virtual_memory.return_value = mock_memory
        
        with pytest.raises(RuntimeError) as exc_info:
            check_memory_availability(required_gb=1.0)
        
        error_msg = str(exc_info.value)
        assert "Insufficient memory" in error_msg
        assert "0.5GB available" in error_msg
        assert "1.0GB required" in error_msg
    
    def test_check_memory_availability_psutil_missing(self):
        """Test handling when psutil is not available."""
        # Temporarily remove psutil from modules
        psutil_backup = sys.modules.get('psutil')
        if 'psutil' in sys.modules:
            del sys.modules['psutil']
        
        try:
            with patch.dict('sys.modules', {'psutil': None}, clear=False):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    check_memory_availability(required_gb=1.0)
                    
                    # Should have warning about missing psutil
                    assert len(w) > 0
                    warning_msg = str(w[0].message)
                    assert "psutil not available" in warning_msg
                    assert "cannot check memory" in warning_msg
        finally:
            if psutil_backup:
                sys.modules['psutil'] = psutil_backup


class TestFullEnvironmentCheck:
    """Test suite for comprehensive environment validation."""
    
    @patch('src.adaptive_syntax_filter.config.validate.check_environment')
    @patch('src.adaptive_syntax_filter.config.validate.check_memory_availability')
    @patch('src.adaptive_syntax_filter.config.validate.validate_numerical_stability')
    def test_full_environment_check_success(self, mock_stability, mock_memory, mock_environment, capsys):
        """Test successful comprehensive environment check."""
        full_environment_check(
            min_numpy="1.20",
            min_scipy="1.7", 
            required_memory_gb=0.5,
            check_stability=True
        )
        
        # Verify all checks were called
        mock_environment.assert_called_once_with(min_numpy="1.20", min_scipy="1.7")
        mock_memory.assert_called_once_with(0.5)
        mock_stability.assert_called_once()
        
        # Check output messages
        captured = capsys.readouterr()
        output = captured.out
        assert "Running comprehensive environment check" in output
        assert "Dependencies validated" in output
        assert "Memory availability checked" in output
        assert "Numerical stability validated" in output
        assert "Environment validation completed successfully" in output
    
    @patch('src.adaptive_syntax_filter.config.validate.check_environment')
    @patch('src.adaptive_syntax_filter.config.validate.check_memory_availability')
    @patch('src.adaptive_syntax_filter.config.validate.validate_numerical_stability')
    def test_full_environment_check_no_stability(self, mock_stability, mock_memory, mock_environment, capsys):
        """Test comprehensive check with stability testing disabled."""
        full_environment_check(
            min_numpy="1.20",
            min_scipy="1.7",
            required_memory_gb=0.5,
            check_stability=False
        )
        
        # Verify stability check was not called
        mock_stability.assert_not_called()
        
        # Check output doesn't mention stability
        captured = capsys.readouterr()
        output = captured.out
        assert "Numerical stability validated" not in output
    
    @patch('src.adaptive_syntax_filter.config.validate.check_environment')
    def test_full_environment_check_failure_propagation(self, mock_environment):
        """Test that failures in individual checks are propagated."""
        mock_environment.side_effect = RuntimeError("Dependency check failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            full_environment_check()
        
        assert "Dependency check failed" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_version_checking_edge_cases(self):
        """Test edge cases in version checking."""
        # Test with unusual version strings
        versions = get_dependency_versions()
        
        # Python version should be properly formatted
        python_version = versions['python']
        parts = python_version.split('.')
        assert len(parts) >= 2
        for part in parts:
            assert part.isdigit()
    
    def test_basic_functionality_coverage(self):
        """Test basic functionality to ensure coverage."""
        # Test successful environment check
        check_environment()
        
        # Test version gathering
        versions = get_dependency_versions()
        assert isinstance(versions, dict)
        assert 'python' in versions
        
        # Test print functionality  
        print_environment_info()
        
        # Test numerical stability
        validate_numerical_stability()
        
        # Test memory check (will warn if psutil missing, but won't fail)
        check_memory_availability(required_gb=0.1)
        
        # Test full check
        full_environment_check(min_numpy="1.20", min_scipy="1.7", required_memory_gb=0.1) 