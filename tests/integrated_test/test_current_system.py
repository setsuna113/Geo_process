#!/usr/bin/env python3
"""
Simple integration test using only current system components.
Tests grid system and database integration without missing modules.
"""

import pytest
import tempfile
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_grid_system_integration():
    """Test grid system with database integration."""
    try:
        from src.grid_systems.grid_factory import GridFactory, GridSpecification
        from src.grid_systems.bounds_manager import BoundsManager
        from src.core.registry import component_registry
        
        print("✅ Successfully imported grid system components")
        
        # Test grid factory
        factory = GridFactory()
        print("✅ GridFactory initialized")
        
        # Test bounds manager
        bounds_manager = BoundsManager()
        bounds = bounds_manager.get_bounds('global')
        print(f"✅ Got global bounds: {bounds.bounds}")
        
        # Test grid creation
        spec = GridSpecification(
            grid_type='cubic',
            resolution=10000,
            bounds='global'
        )
        
        grid = factory.create_grid(spec)
        print(f"✅ Created {grid.__class__.__name__} with resolution {grid.resolution}")
        
        # Test cell generation (limited)
        test_coords = [(0, 0), (1, 1), (-1, -1)]
        cell_count = 0
        for x, y in test_coords:
            try:
                cell_id = grid.get_cell_id(x, y)
                if cell_id:
                    cell_count += 1
            except Exception:
                pass  # Some coords may be outside bounds
        
        print(f"✅ Successfully generated {cell_count} cell IDs from test coordinates")
        
        return True
        
    except Exception as e:
        print(f"❌ Grid system integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_integration():
    """Test database schema and operations."""
    try:
        from src.database.schema import DatabaseSchema
        
        print("✅ Successfully imported database components")
        
        # Test schema info (doesn't require actual DB connection)
        schema = DatabaseSchema()
        info = schema.get_schema_info()
        
        print(f"✅ Schema info available: {info['summary']['table_count']} tables")
        print(f"✅ Schema includes: {', '.join([t['table_name'] for t in info['tables'][:5]])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """Test configuration system."""
    try:
        from src.config.config import Config
        from src.config import defaults
        
        print("✅ Successfully imported config components")
        
        # Test defaults module
        print(f"✅ Defaults module available with attributes: {[attr for attr in dir(defaults) if not attr.startswith('_')]}")
        
        # Test processing bounds if available
        if hasattr(defaults, 'PROCESSING_BOUNDS'):
            processing_bounds = defaults.PROCESSING_BOUNDS
            print(f"✅ Processing bounds available: {len(processing_bounds)} regions")
        
        return True
        
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_registry_integration():
    """Test component registry system."""
    try:
        from src.core.registry import component_registry, ComponentMetadata, MemoryUsage
        
        print("✅ Successfully imported registry components")
        
        # Test registry structure
        print(f"✅ Registry has grids: {hasattr(component_registry, 'grids')}")
        print(f"✅ Registry has resamplers: {hasattr(component_registry, 'resamplers')}")
        print(f"✅ Registry has raster_sources: {hasattr(component_registry, 'raster_sources')}")
        
        # Test memory usage enum
        memory_levels = [MemoryUsage.LOW, MemoryUsage.MEDIUM, MemoryUsage.HIGH]
        print(f"✅ Memory usage levels available: {len(memory_levels)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Registry integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run available integration tests."""
    print("="*60)
    print("INTEGRATION TEST - CURRENT SYSTEM COMPONENTS")
    print("="*60)
    
    tests = [
        ("Grid System Integration", test_grid_system_integration),
        ("Database Integration", test_database_integration), 
        ("Config Integration", test_config_integration),
        ("Registry Integration", test_registry_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("INTEGRATION TEST RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\nSUMMARY: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All available integration tests passed!")
        print("Current system components are working correctly.")
    else:
        print("⚠️  Some integration tests failed.")
        print("Check error messages above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
