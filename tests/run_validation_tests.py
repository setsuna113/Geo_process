#!/usr/bin/env python
"""Run all validation framework tests."""

import sys
import subprocess
from pathlib import Path

# Test modules to run
TEST_MODULES = [
    # Unit tests
    "tests/domain/validators/test_coordinate_integrity.py",
    "tests/domain/validators/test_validation_error_scenarios.py",
    
    # Integration tests
    "tests/processors/data_preparation/test_coordinate_merger_validation.py",
    "tests/processors/data_preparation/test_resampling_processor_validation.py",
    "tests/pipelines/test_orchestrator_validation.py"
]


def run_tests():
    """Run all validation tests."""
    print("=" * 80)
    print("Running Validation Framework Tests")
    print("=" * 80)
    
    failed_modules = []
    
    for test_module in TEST_MODULES:
        print(f"\n{'='*60}")
        print(f"Running: {test_module}")
        print(f"{'='*60}")
        
        # Run pytest on the module
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_module, "-v", "--tb=short"],
            capture_output=False
        )
        
        if result.returncode != 0:
            failed_modules.append(test_module)
            print(f"\n❌ FAILED: {test_module}")
        else:
            print(f"\n✅ PASSED: {test_module}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION TEST SUMMARY")
    print("=" * 80)
    print(f"Total test modules: {len(TEST_MODULES)}")
    print(f"Passed: {len(TEST_MODULES) - len(failed_modules)}")
    print(f"Failed: {len(failed_modules)}")
    
    if failed_modules:
        print("\nFailed modules:")
        for module in failed_modules:
            print(f"  - {module}")
        return 1
    else:
        print("\n✅ All validation tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(run_tests())