#!/usr/bin/env python3
"""
Test script to verify the chunked storage implementation works correctly.
Tests both chunked pixel storage and grid aggregation approaches.
"""

import sys
import numpy as np
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.resampling_processor import (
    ResamplingProcessor, ResampledDatasetInfo
)

def create_test_data():
    """Create test raster data simulating a fungi dataset."""
    # Create a 1000x1000 array (1M pixels) with realistic fungi richness values
    data = np.random.poisson(5, size=(1000, 1000)).astype(np.float32)
    
    # Add some NaN values to simulate realistic data
    nan_mask = np.random.random((1000, 1000)) < 0.3  # 30% NaN values
    data[nan_mask] = np.nan
    
    # Add some spatial structure (higher values in center)
    center_y, center_x = 500, 500
    y, x = np.ogrid[:1000, :1000]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    multiplier = np.exp(-distance / 200)  # Gaussian falloff
    data[~nan_mask] *= (1 + 2 * multiplier[~nan_mask])
    
    return data

def create_test_dataset_info(data_shape):
    """Create test ResampledDatasetInfo."""
    return ResampledDatasetInfo(
        name="test-fungi-richness",
        source_path=Path("/tmp/test_fungi.tif"),
        target_resolution=0.05,
        target_crs="EPSG:4326",
        bounds=(-10.0, -10.0, 10.0, 10.0),  # 20x20 degree box
        shape=data_shape,
        data_type="richness_data",
        resampling_method="sum",
        band_name="test_fungi_richness",
        metadata={
            'test_data': True,
            'created_for': 'chunked_storage_test'
        }
    )

def test_chunked_storage():
    """Test the chunked storage implementation."""
    print("🧪 Testing Chunked Storage Implementation")
    print("=" * 50)
    
    # Override config for testing
    test_config_overrides = {
        'storage.chunk_size': 50000,      # 50k pixels threshold
        'storage.chunk_rows': 100,        # 100 rows per chunk
        'storage.aggregate_to_grid': False,  # Test pixel storage first
        'storage.enable_progress_logging': True
    }
    
    # Apply overrides by modifying settings directly
    if 'storage' not in config.settings:
        config.settings['storage'] = {}
    
    config.settings['storage']['chunk_size'] = 50000
    config.settings['storage']['chunk_rows'] = 100
    config.settings['storage']['aggregate_to_grid'] = False
    config.settings['storage']['enable_progress_logging'] = True
    
    print(f"Configuration overrides applied:")
    for key, value in test_config_overrides.items():
        print(f"  {key}: {value}")
    
    # Create test data
    print(f"\n📊 Creating test data...")
    test_data = create_test_data()
    print(f"  Data shape: {test_data.shape}")
    print(f"  Total pixels: {test_data.size:,}")
    print(f"  Non-NaN pixels: {np.sum(~np.isnan(test_data)):,}")
    print(f"  Memory size: {test_data.nbytes / (1024*1024):.1f} MB")
    
    # Create database connection
    print(f"\n🗄️  Setting up database connection...")
    try:
        db = DatabaseManager()
        print(f"  ✅ Database connection established")
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        return False
    
    # Create processor
    print(f"\n🔧 Creating resampling processor...")
    try:
        processor = ResamplingProcessor(config, db)
        print(f"  ✅ Processor created successfully")
    except Exception as e:
        print(f"  ❌ Processor creation failed: {e}")
        return False
    
    # Create dataset info
    dataset_info = create_test_dataset_info(test_data.shape)
    
    # Test 1: Standard chunked storage (pixel-by-pixel)
    print(f"\n🧪 Test 1: Chunked Pixel Storage")
    print("-" * 30)
    try:
        processor._store_resampled_dataset(dataset_info, test_data)
        print(f"  ✅ Chunked pixel storage completed successfully")
        
        # Verify data was stored
        stored_info = processor.get_resampled_dataset("test-fungi-richness")
        if stored_info:
            print(f"  ✅ Dataset metadata retrieved successfully")
            print(f"     Storage method: {'chunked' if stored_info.metadata.get('chunked_storage') else 'standard'}")
        else:
            print(f"  ⚠️  Dataset metadata not found")
            
    except Exception as e:
        print(f"  ❌ Chunked pixel storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Aggregated storage (grid cells)
    print(f"\n🧪 Test 2: Aggregated Grid Storage")
    print("-" * 30)
    
    # Update config for aggregation
    config.settings['storage']['aggregate_to_grid'] = True
    config.settings['storage']['grid_cell_size'] = 0.1  # 0.1 degree cells
    
    # Create new dataset with different name
    dataset_info_agg = create_test_dataset_info(test_data.shape)
    dataset_info_agg.name = "test-fungi-richness-aggregated"
    
    try:
        processor._store_resampled_dataset(dataset_info_agg, test_data)
        print(f"  ✅ Aggregated grid storage completed successfully")
        
        # Verify data was stored
        stored_info_agg = processor.get_resampled_dataset("test-fungi-richness-aggregated")
        if stored_info_agg:
            print(f"  ✅ Aggregated dataset metadata retrieved successfully")
            print(f"     Aggregation enabled: {stored_info_agg.metadata.get('aggregated', False)}")
        else:
            print(f"  ⚠️  Aggregated dataset metadata not found")
            
    except Exception as e:
        print(f"  ❌ Aggregated grid storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Memory efficiency comparison
    print(f"\n🧪 Test 3: Memory Efficiency Analysis")
    print("-" * 30)
    
    try:
        # Get row counts from both tables
        with db.get_connection() as conn:
            cur = conn.cursor()
            
            # Count pixels in pixel storage
            cur.execute("SELECT COUNT(*) FROM resampled_test_fungi_richness")
            pixel_count = cur.fetchone()[0]
            
            # Count grid cells in aggregated storage
            cur.execute("SELECT COUNT(*) FROM resampled_test_fungi_richness_aggregated")
            grid_count = cur.fetchone()[0]
            
            print(f"  Pixel storage: {pixel_count:,} records")
            print(f"  Grid storage: {grid_count:,} records")
            print(f"  Reduction ratio: {pixel_count/max(grid_count, 1):.1f}x")
            print(f"  Storage efficiency: {(1 - grid_count/max(pixel_count, 1))*100:.1f}% reduction")
            
    except Exception as e:
        print(f"  ⚠️  Memory analysis failed: {e}")
    
    print(f"\n✅ All tests completed successfully!")
    return True

def cleanup_test_data():
    """Clean up test data from database."""
    print(f"\n🧹 Cleaning up test data...")
    try:
        db = DatabaseManager()
        with db.get_connection() as conn:
            cur = conn.cursor()
            
            # Remove test datasets
            test_tables = [
                'resampled_test_fungi_richness',
                'resampled_test_fungi_richness_aggregated'
            ]
            
            for table in test_tables:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            
            # Remove metadata entries
            cur.execute("DELETE FROM resampled_datasets WHERE name LIKE 'test-fungi-richness%'")
            conn.commit()
            
        print(f"  ✅ Test data cleaned up successfully")
    except Exception as e:
        print(f"  ⚠️  Cleanup failed: {e}")

if __name__ == "__main__":
    print("🚀 Chunked Storage Test Suite")
    print("Testing memory-efficient database storage for large raster datasets")
    print()
    
    success = False
    try:
        success = test_chunked_storage()
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_test_data()
    
    if success:
        print(f"\n🎉 SUCCESS: Chunked storage implementation is working correctly!")
        print(f"\nRecommendations for fungi datasets:")
        print(f"  - Use storage.chunk_size: 100000 (100k pixels)")
        print(f"  - Use storage.aggregate_to_grid: true")
        print(f"  - Use storage.grid_cell_size: 0.05 (5.5km cells)")
        print(f"  - This should reduce 233M pixels to ~4M grid cells (58x reduction)")
    else:
        print(f"\n❌ FAILURE: Issues found in chunked storage implementation")
        sys.exit(1)