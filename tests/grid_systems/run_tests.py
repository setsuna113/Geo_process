"""Run all grid system tests with proper ordering."""

import pytest
import sys
from pathlib import Path

def run_grid_system_tests():
    """Run grid system tests in proper order."""
    
    # Add src to path
    src_path = Path(__file__).parent.parent.parent / 'src'
    sys.path.insert(0, str(src_path))
    
    # Test order
    test_sequence = [
        # Unit tests first
        "test_bounds_manager.py::TestBoundsDefinition",
        "test_bounds_manager.py::TestBoundsManager",
        "test_cubic_grid.py::TestCubicGrid",
        "test_hexagonal_grid.py::TestHexagonalGrid", 
        "test_grid_factory.py::TestGridSpecification",
        "test_grid_factory.py::TestGridFactory",
        
        # Integration tests
        "test_integration.py::TestGridSystemIntegration",
        
        # System tests
        "test_system.py::TestGridSystemE2E",
        
        # Performance tests (optional)
        "test_system.py::TestGridSystemPerformance -m slow"
    ]
    
    # Run tests in sequence
    failed = False
    for test in test_sequence:
        print(f"\n{'='*60}")
        print(f"Running: {test}")
        print('='*60)
        
        result = pytest.main(["-xvs", test])
        
        if result != 0:
            print(f"\nFAILED: {test}")
            failed = True
            break
    
    if not failed:
        print("\n" + "="*60)
        print("ALL GRID SYSTEM TESTS PASSED!")
        print("="*60)
        
        # Run coverage report
        pytest.main(["--cov=src.grid_systems", "--cov-report=term-missing"])
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(run_grid_system_tests())