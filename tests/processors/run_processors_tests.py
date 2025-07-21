#!/usr/bin/env python3

"""
Test runner for processors module.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run all processor tests."""
    print("=" * 60)
    print("RUNNING PROCESSORS MODULE TESTS")
    print("=" * 60)
    print()
    
    # Get the tests directory
    tests_dir = Path(__file__).parent / "data_preparation"
    
    # Find all test files
    test_files = sorted(tests_dir.glob("test_*.py"))
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_file in test_files:
        print(f"🧪 Running {test_file.name}...")
        print("-" * 40)
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(test_file), 
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd="/home/jason/geo")
            
            if result.returncode == 0:
                print(f"✅ {test_file.name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_file.name} FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                failed += 1
                failed_tests.append(test_file.name)
        except Exception as e:
            print(f"❌ Error running {test_file.name}: {e}")
            failed += 1
            failed_tests.append(test_file.name)
        
        print()
    
    # Summary
    print("=" * 60)
    print("PROCESSORS MODULE TESTS SUMMARY")
    print("=" * 60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 All processors module tests PASSED!")
    else:
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
