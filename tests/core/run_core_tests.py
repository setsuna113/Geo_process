#!/usr/bin/env python3
"""
Test runner for core module tests.
Executes all tests in the tests/core/ directory.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_core_tests():
    """Run all core module tests."""
    core_dir = Path(__file__).parent
    test_files = [
        "test_enhanced_base_integration.py",
        "test_registry.py"
    ]
    
    print("=" * 60)
    print("RUNNING CORE MODULE TESTS")
    print("=" * 60)
    
    total_tests = len(test_files)
    passed_tests = 0
    failed_tests = []
    
    for test_file in test_files:
        test_path = core_dir / test_file
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue
            
        print(f"\n🧪 Running {test_file}...")
        print("-" * 40)
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(test_path), 
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=core_dir.parent.parent)
            
            if result.returncode == 0:
                print(f"✅ {test_file} PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_file} FAILED")
                failed_tests.append(test_file)
                print("Error output:")
                print(result.stdout)
                print(result.stderr)
                
        except Exception as e:
            print(f"💥 Error running {test_file}: {e}")
            failed_tests.append(test_file)
    
    # Summary
    print("\n" + "=" * 60)
    print("CORE MODULE TESTS SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")
        return False
    else:
        print("🎉 All core module tests PASSED!")
        return True

if __name__ == "__main__":
    success = run_core_tests()
    sys.exit(0 if success else 1)
