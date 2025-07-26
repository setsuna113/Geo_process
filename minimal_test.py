#!/usr/bin/env python3
"""Minimal test of Phase 1 lightweight metadata extraction."""

import sys
from pathlib import Path
import os

# Direct imports without the problematic global db connection
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set environment
os.environ['FORCE_TEST_MODE'] = 'true'
os.environ['DB_NAME'] = 'geoprocess_db'

def test_gdal_timeout():
    """Test the GDAL timeout mechanism."""
    print("=== Testing GDAL Timeout Mechanism ===")
    
    from src.domain.raster.loaders.lightweight_metadata import gdal_timeout, TimeoutError
    import time
    
    print("1. Testing timeout context manager...")
    
    try:
        with gdal_timeout(2):  # 2 second timeout
            print("   Starting operation that should timeout...")
            time.sleep(3)  # Sleep longer than timeout
            print("   This should not print!")
    except TimeoutError as e:
        print(f"✅ Timeout caught correctly: {e}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("❌ Timeout did not work!")
    return False

def test_lightweight_extractor_creation():
    """Test creating the lightweight extractor without database."""
    print("\n=== Testing Lightweight Extractor Creation ===")
    
    from src.domain.raster.loaders.lightweight_metadata import LightweightMetadataExtractor
    
    try:
        print("1. Creating extractor with mock database...")
        
        class MockDB:
            def get_cursor(self):
                return MockCursor()
        
        class MockCursor:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def execute(self, *args):
                pass
            def fetchone(self):
                return {'id': 'test123'}
        
        extractor = LightweightMetadataExtractor(
            db_connection=MockDB(),
            timeout_seconds=30,
            progress_callback=lambda msg, pct: print(f"   Progress: {msg} ({pct:.1f}%)")
        )
        
        print("✅ Lightweight extractor created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create extractor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_raster_file_access():
    """Test that we can access the test raster files."""
    print("\n=== Testing Raster File Access ===")
    
    raster_files = [
        Path(__file__).parent / 'data' / 'richness_maps' / 'daru-plants-richness.tif',
        Path(__file__).parent / 'data' / 'richness_maps' / 'iucn-terrestrial-richness.tif'
    ]
    
    all_found = True
    for raster_file in raster_files:
        if raster_file.exists():
            size_mb = raster_file.stat().st_size / (1024*1024)
            print(f"✅ Found: {raster_file.name} ({size_mb:.1f} MB)")
        else:
            print(f"❌ Missing: {raster_file}")
            all_found = False
    
    return all_found

def main():
    """Run minimal tests."""
    print("🧪 Minimal Test of Phase 1 Components")
    print("=" * 50)
    
    test1 = test_gdal_timeout()
    test2 = test_lightweight_extractor_creation() 
    test3 = test_raster_file_access()
    
    print("\n" + "=" * 50)
    print("📋 MINIMAL TEST SUMMARY:")
    print(f"✅ GDAL Timeout:        {'PASSED' if test1 else 'FAILED'}")
    print(f"✅ Extractor Creation:  {'PASSED' if test2 else 'FAILED'}")
    print(f"✅ Raster File Access:  {'PASSED' if test3 else 'FAILED'}")
    
    if test1 and test2 and test3:
        print("\n🎉 MINIMAL TESTS PASSED!")
        print("   Core Phase 1 components are working.")
        print("   The hang issue should be fixed by:")
        print("   - GDAL timeout protection")
        print("   - Lightweight metadata extraction")
        print("   - Progress callback system")
        return True
    else:
        print("\n❌ SOME TESTS FAILED.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)