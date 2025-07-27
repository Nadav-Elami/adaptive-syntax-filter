# Diagnostic Scripts

This directory contains diagnostic scripts used during development and troubleshooting of the Adaptive Syntax Filter project.

## Files

### Machine Diagnostics
- `diagnose_external_machine.py` - Original diagnostic script for external machine issues
- `diagnose_external_machine_v2.py` - Updated diagnostic script for external machine issues
- `diagnose_config4_specific.py` - Diagnostic script for config4-specific issues

### Comparison Scripts
- `compare_machines.py` - Diagnostic script for machine comparison
- `simple_machine_compare.py` - Simple machine comparison script
- `external_vs_local_diagnostic.py` - Diagnostic script comparing external vs local machines

### Testing Scripts
- `simple_import_test.py` - Diagnostic script for import testing
- `quick_external_test.py` - Diagnostic script for quick external testing
- `targeted_bottleneck_test.py` - Diagnostic script for bottleneck testing

### Utility Scripts
- `simple_cleanup.py` - Simple cleanup utility

## Purpose

These scripts were created during the development process to:
- Diagnose issues with external machine setup
- Compare performance between different machines
- Test import functionality and dependencies
- Identify bottlenecks in the analysis pipeline
- Troubleshoot specific configuration issues

## Status

These scripts are **archived for reference only**. They were used during the development phase and are no longer needed for normal operation. The issues they were designed to solve have been resolved.

## Usage (Historical)

```bash
# Diagnose external machine issues
python diagnose_external_machine_v2.py

# Compare machine performance
python compare_machines.py

# Test imports
python simple_import_test.py

# Quick external test
python quick_external_test.py
```

## Notes

- These scripts may contain hardcoded paths specific to the development environment
- They are not maintained and may not work with current versions of dependencies
- They are kept for historical reference and potential future troubleshooting 