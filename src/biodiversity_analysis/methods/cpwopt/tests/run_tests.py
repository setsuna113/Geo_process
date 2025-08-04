#!/usr/bin/env python
"""Run all CP-WOPT tests."""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

def run_all_tests():
    """Discover and run all tests."""
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover all test files
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


def run_specific_test(test_module):
    """Run a specific test module."""
    try:
        # Import the test module
        module = __import__(f'test_{test_module}', fromlist=[''])
        
        # Load tests from module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
    
    except ImportError as e:
        print(f"Error: Could not import test module 'test_{test_module}': {e}")
        return 1


def list_test_modules():
    """List available test modules."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = [f[5:-3] for f in os.listdir(test_dir) 
                  if f.startswith('test_') and f.endswith('.py')]
    
    print("Available test modules:")
    for module in sorted(test_files):
        print(f"  - {module}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            list_test_modules()
            sys.exit(0)
        else:
            # Run specific test module
            exit_code = run_specific_test(sys.argv[1])
    else:
        # Run all tests
        print("Running all CP-WOPT tests...")
        print("=" * 60)
        exit_code = run_all_tests()
    
    sys.exit(exit_code)