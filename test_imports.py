#!/usr/bin/env python3
"""Test that all modules import correctly after migration."""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test critical imports."""
    results = []
    
    # Test core abstractions
    try:
        from src.abstractions import (
            get_logger, save_json, load_json,
            validate_parameters, IAnalyzer
        )
        results.append(("Core abstractions", True, None))
    except Exception as e:
        results.append(("Core abstractions", False, str(e)))
    
    # Test biodiversity modules
    modules_to_test = [
        ("CPWOPT", "src.biodiversity_analysis.methods.cpwopt.analyzer_adapter"),
        ("SOM", "src.biodiversity_analysis.methods.som.analyzer"),
        ("KMeans", "src.biodiversity_analysis.methods.kmeans_hierarchical.hierarchical_analyzer"),
        ("Config modules", "src.config"),
        ("Biodiversity shared", "src.biodiversity_analysis.shared"),
        ("Raster merge", "src.raster_merge"),
        ("Statistical modules", "src.biodiversity_analysis.statistical"),
        ("ML methods", "src.machine_learning.methods"),
    ]
    
    for name, module_path in modules_to_test:
        try:
            __import__(module_path)
            results.append((name, True, None))
        except Exception as e:
            error_msg = str(e)
            # Get more detailed error info
            if hasattr(e, '__traceback__'):
                tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
                for line in tb_lines:
                    if '.py", line' in line:
                        error_msg = line.strip()
                        break
            results.append((name, False, error_msg))
    
    # Print results
    print("\nImport Test Results:")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, success, error in results:
        if success:
            print(f"✓ {name} imports correctly")
            passed += 1
        else:
            print(f"✗ {name} failed: {error}")
            failed += 1
    
    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)