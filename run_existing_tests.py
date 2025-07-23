#!/usr/bin/env python3
"""Run selected existing tests to verify my updates don't break functionality."""

import subprocess
import sys
from pathlib import Path

def run_test(test_path, description):
    """Run a single test and report results."""
    print(f"\n🔍 {description}")
    print("-" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_path, '-v'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ PASSED: {description}")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"💥 ERROR: {description} - {e}")
        return False

def main():
    print("🧪 Testing Existing Functionality After My Updates")
    print("=" * 60)
    
    tests_to_run = [
        # Config tests
        ("tests/config/test_config.py::TestConfigurationSystem::test_configuration_import", 
         "Config system import test"),
        ("tests/config/test_config.py::TestConfigurationSystem::test_output_formats_section", 
         "Config output formats test"),
        
        # Base classes (shouldn't be affected)
        ("tests/base/test_base_classes.py::TestBaseDataset::test_initialization", 
         "Base dataset initialization"),
         
        # Core registry (shouldn't be affected)
        ("tests/core/test_registry.py::TestComponentRegistry::test_registry_initialization", 
         "Component registry test"),
    ]
    
    passed = 0
    total = len(tests_to_run)
    
    for test_path, description in tests_to_run:
        if run_test(test_path, description):
            passed += 1
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All existing tests still pass! My updates are compatible.")
    else:
        print(f"\n⚠️ Some tests failed. Need to investigate.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)