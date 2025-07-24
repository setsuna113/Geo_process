#!/usr/bin/env python3
"""Test Phase 1 fixes for the 7-hour hang issue."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Set environment to use geoprocess_db
os.environ['FORCE_TEST_MODE'] = 'true'
os.environ['DB_NAME'] = 'geoprocess_db'

def test_lightweight_metadata():
    """Test the lightweight metadata extraction."""
    print("=== Testing Lightweight Metadata Extraction ===")
    
    # Test the lightweight metadata extractor directly
    from src.raster_data.loaders.lightweight_metadata import LightweightMetadataExtractor
    from src.database.connection import DatabaseManager
    
    try:
        print("1. Creating database connection...")
        db = DatabaseManager()
        print("✅ Database connection successful")
        
        print("2. Testing lightweight metadata extractor...")
        extractor = LightweightMetadataExtractor(
            db_connection=db,
            timeout_seconds=30,
            progress_callback=lambda msg, pct: print(f"   Progress: {msg} ({pct:.1f}%)")
        )
        print("✅ Lightweight extractor created")
        
        # Test with a real raster file
        raster_file = project_root / 'data' / 'richness_maps' / 'daru-plants-richness.tif'
        if raster_file.exists():
            print(f"3. Testing metadata extraction from: {raster_file.name}")
            print(f"   File size: {raster_file.stat().st_size / (1024*1024):.1f} MB")
            
            metadata = extractor.extract_essential_metadata(raster_file)
            
            print("✅ Metadata extraction completed!")
            print("   Key metadata:")
            print(f"   - Extraction mode: {metadata.get('extraction_mode')}")
            print(f"   - File size: {metadata['file_info']['size_mb']:.1f} MB")
            if metadata['spatial_info'].get('bounds'):
                bounds = metadata['spatial_info']['bounds']
                print(f"   - Bounds: [{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]")
            print(f"   - Resolution: {metadata['spatial_info'].get('resolution_degrees', 'unknown')}")
            
            return True
        else:
            print(f"❌ Test raster file not found: {raster_file}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_raster_catalog():
    """Test the updated raster catalog with lightweight registration."""
    print("\n=== Testing Raster Catalog Lightweight Registration ===")
    
    try:
        from src.config.config import Config
        from src.database.connection import DatabaseManager
        from src.raster_data.catalog import RasterCatalog
        
        print("1. Loading configuration...")
        config = Config()
        print("✅ Configuration loaded")
        
        print("2. Creating database and catalog...")
        db = DatabaseManager()
        catalog = RasterCatalog(db, config)
        print("✅ Catalog created")
        
        # Test lightweight registration
        raster_file = project_root / 'data' / 'richness_maps' / 'daru-plants-richness.tif'
        if raster_file.exists():
            print(f"3. Testing lightweight registration of: {raster_file.name}")
            
            def progress_cb(msg, pct):
                print(f"   Progress: {msg} ({pct:.1f}%)")
            
            entry = catalog.add_raster_lightweight(
                raster_file,
                dataset_type='richness_data',
                validate=False,
                progress_callback=progress_cb
            )
            
            print("✅ Lightweight registration completed!")
            print(f"   - Entry ID: {entry.id}")
            print(f"   - Name: {entry.name}")
            print(f"   - Resolution: {entry.resolution_degrees}")
            print(f"   - File size: {entry.file_size_mb:.1f} MB")
            print(f"   - Bounds: {entry.bounds}")
            
            return True
        else:
            print(f"❌ Test raster file not found: {raster_file}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Phase 1 tests."""
    print("🧪 Testing Phase 1 Fixes for 7-Hour Hang Issue")
    print("=" * 60)
    
    # Test 1: Lightweight metadata extraction
    test1_passed = test_lightweight_metadata()
    
    # Test 2: Raster catalog lightweight registration  
    test2_passed = test_raster_catalog()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print(f"✅ Lightweight Metadata Extraction: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Raster Catalog Registration:     {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Phase 1 fixes are working correctly.")
        print("   The 7-hour hang issue should be resolved.")
        print("   The system now uses lightweight metadata extraction with timeouts.")
        return True
    else:
        print("\n❌ SOME TESTS FAILED. Phase 1 fixes need more work.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)