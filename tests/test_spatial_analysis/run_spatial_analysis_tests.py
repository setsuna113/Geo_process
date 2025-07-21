#!/usr/bin/env python3
"""Test runner for spatial_analysis module tests."""

import sys
import subprocess
from pathlib import Path

def run_test_file(test_file: str) -> tuple[bool, str]:
    """Run a single test file and return success status and output."""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, '-v',
            '--tb=short', '--no-header'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, f"Error running {test_file}: {str(e)}"

def main():
    """Run all spatial_analysis tests."""
    test_files = [
        'tests/test_spatial_analysis/test_base_analyzer.py',
        'tests/test_spatial_analysis/test_gwpca_analyzer.py',
        'tests/test_spatial_analysis/test_maxp_analyzer.py',
        'tests/test_spatial_analysis/test_som_analyzer.py',
        'tests/test_spatial_analysis/test_integration.py',
    ]
    
    print("🧪 Running Spatial Analysis Module Tests")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        test_name = Path(test_file).stem
        print(f"\n📋 Running {test_name}...")
        
        success, output = run_test_file(test_file)
        
        if success:
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
            print("Error output:")
            print(output)
            failed += 1
    
    print(f"\n🎯 Test Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All spatial_analysis module tests PASSED!")
        return 0
    else:
        print(f"💥 {failed} test file(s) FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
