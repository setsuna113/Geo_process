#!/usr/bin/env python3
"""
Master test runner for all test modules.
Executes test runners for each module in sequence.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_all_tests():
    """Run all test modules."""
    tests_dir = Path(__file__).parent
    
    # Test modules to run (excluding integrated_test as requested)
    test_modules = [
        ("config", "tests/config/run_config_tests.py"),
        ("base", "tests/base/run_base_tests.py"),
        ("core", "tests/core/run_core_tests.py"),
        ("database", "tests/database/run_database_tests.py"),
        ("grid_systems", "tests/grid_systems/run_grid_systems_tests.py"),
        ("raster_data", "tests/raster_data/run_raster_data_tests.py")
    ]
    
    print("=" * 80)
    print("RUNNING ALL TEST MODULES")
    print("=" * 80)
    
    total_modules = len(test_modules)
    passed_modules = 0
    failed_modules = []
    
    for module_name, runner_script in test_modules:
        runner_path = tests_dir.parent / runner_script
        
        if not runner_path.exists():
            print(f"⚠️  Test runner not found: {runner_script}")
            continue
            
        print(f"\n🚀 Running {module_name} module tests...")
        print("=" * 60)
        
        try:
            result = subprocess.run([
                sys.executable, str(runner_path)
            ], cwd=tests_dir.parent)
            
            if result.returncode == 0:
                print(f"✅ {module_name} module tests PASSED")
                passed_modules += 1
            else:
                print(f"❌ {module_name} module tests FAILED")
                failed_modules.append(module_name)
                
        except Exception as e:
            print(f"💥 Error running {module_name} tests: {e}")
            failed_modules.append(module_name)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("OVERALL TEST SUMMARY")
    print("=" * 80)
    print(f"Total modules: {total_modules}")
    print(f"Passed: {passed_modules}")
    print(f"Failed: {len(failed_modules)}")
    
    if failed_modules:
        print(f"Failed modules: {', '.join(failed_modules)}")
        print("\n💡 You can run individual module tests using:")
        for module in failed_modules:
            print(f"   python tests/{module}/run_{module}_tests.py")
        return False
    else:
        print("🎉 ALL TEST MODULES PASSED!")
        return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
