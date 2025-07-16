#!/usr/bin/env python3
"""Run comprehensive database tests."""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all database tests with different configurations."""
    
    test_commands = [
        # Quick tests (exclude slow ones)
        ["python", "-m", "pytest", "tests/database/", "-v", "-m", "not slow"],
        
        # Performance tests
        ["python", "-m", "pytest", "tests/database/test_performance.py", "-v", "-s"],
        
        # Integration tests
        ["python", "-m", "pytest", "tests/database/test_integration.py", "-v"],
        
        # All tests (if previous pass)
        ["python", "-m", "pytest", "tests/database/", "-v", "--tb=short"]
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n{'='*60}")
        print(f"Running test suite {i}/{len(test_commands)}: {' '.join(cmd[3:])}")
        print(f"{'='*60}")
        
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            print(f"\n❌ Test suite {i} failed!")
            return False
        else:
            print(f"\n✅ Test suite {i} passed!")
    
    print(f"\n🎉 All database test suites passed!")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)