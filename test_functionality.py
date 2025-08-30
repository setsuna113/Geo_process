#!/usr/bin/env python3
"""Test that core functionality still works after migration."""

import sys
import numpy as np
from pathlib import Path

def test_abstractions():
    """Test abstractions work correctly."""
    print("\n=== Testing Abstractions ===")
    
    # Test logging
    from src.abstractions import get_logger
    logger = get_logger("test")
    print("✓ Logger creation works")
    
    # Test validation
    from src.abstractions import validate_parameters
    valid, errors = validate_parameters(
        {"param1": 1, "param2": "test"},
        ["param1", "param2"]
    )
    assert valid == True
    assert errors == []
    print("✓ Parameter validation works")
    
    # Test I/O
    from src.abstractions import save_json, load_json
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        test_path = Path(f.name)
    
    test_data = {"key": "value", "number": 42}
    save_json(test_data, test_path)
    loaded = load_json(test_path)
    assert loaded == test_data
    test_path.unlink()
    print("✓ JSON I/O works")
    
    return True

def test_analyzers():
    """Test analyzer interfaces work."""
    print("\n=== Testing Analyzers ===")
    
    # Test SOM analyzer
    from src.biodiversity_analysis.methods.som.analyzer_adapter import SOMAnalyzer
    som = SOMAnalyzer(config=None)
    params = som.get_default_parameters()
    assert isinstance(params, dict)
    print("✓ SOM analyzer instantiation works")
    
    # Test KMeans analyzer
    from src.biodiversity_analysis.methods.kmeans_hierarchical.analyzer_adapter import KMeansHierarchicalAnalyzerAdapter
    kmeans = KMeansHierarchicalAnalyzerAdapter(config=None)
    params = kmeans.get_default_parameters()
    assert isinstance(params, dict)
    print("✓ KMeans analyzer instantiation works")
    
    # Test CPWOPT analyzer
    from src.biodiversity_analysis.methods.cpwopt.analyzer_adapter import CPWOPTAnalyzer
    # CPWOPT requires pipeline_config, pass None for testing
    cpwopt = CPWOPTAnalyzer(pipeline_config=None)
    params = cpwopt.get_default_parameters()
    assert isinstance(params, dict)
    print("✓ CPWOPT analyzer instantiation works")
    
    return True

def test_config():
    """Test config modules work."""
    print("\n=== Testing Config ===")
    
    # Test config imports
    from src.config import config, Config
    assert config is not None
    print("✓ Config module imports correctly")
    
    return True

def main():
    """Run all functionality tests."""
    print("Testing functionality after migration...")
    print("=" * 60)
    
    all_passed = True
    
    try:
        test_abstractions()
    except Exception as e:
        print(f"✗ Abstractions test failed: {e}")
        all_passed = False
    
    try:
        test_analyzers()
    except Exception as e:
        print(f"✗ Analyzers test failed: {e}")
        all_passed = False
    
    try:
        test_config()
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All functionality tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())