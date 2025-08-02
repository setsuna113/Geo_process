#!/usr/bin/env python3
"""
Memory profiling test to identify specific memory issues in chunked storage.
"""

import sys
import numpy as np
import psutil
import time
import gc
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.resampling_processor import (
    ResamplingProcessor, ResampledDatasetInfo
)

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / (1024 * 1024),
        'vms_mb': mem_info.vms / (1024 * 1024),
        'system_percent': psutil.virtual_memory().percent
    }

def log_memory(stage, start_mem=None):
    """Log memory usage at different stages."""
    current_mem = get_memory_usage()
    
    if start_mem:
        rss_diff = current_mem['rss_mb'] - start_mem['rss_mb']
        vms_diff = current_mem['vms_mb'] - start_mem['vms_mb']
        sys_diff = current_mem['system_percent'] - start_mem['system_percent']
        
        print(f"📊 {stage}:")
        print(f"   RSS: {current_mem['rss_mb']:.1f}MB ({rss_diff:+.1f}MB)")
        print(f"   VMS: {current_mem['vms_mb']:.1f}MB ({vms_diff:+.1f}MB)")
        print(f"   System: {current_mem['system_percent']:.1f}% ({sys_diff:+.1f}%)")
    else:
        print(f"📊 {stage}:")
        print(f"   RSS: {current_mem['rss_mb']:.1f}MB")
        print(f"   VMS: {current_mem['vms_mb']:.1f}MB") 
        print(f"   System: {current_mem['system_percent']:.1f}%")
    
    return current_mem

def create_test_data():
    """Create smaller test data for memory profiling."""
    # Smaller dataset for focused memory analysis
    data = np.random.poisson(3, size=(500, 500)).astype(np.float32)
    nan_mask = np.random.random((500, 500)) < 0.4  # 40% NaN
    data[nan_mask] = np.nan
    return data

def create_test_dataset_info(data_shape):
    """Create test ResampledDatasetInfo."""
    return ResampledDatasetInfo(
        name="memory-test-fungi",
        source_path=Path("/tmp/memory_test.tif"),
        target_resolution=0.05,
        target_crs="EPSG:4326",
        bounds=(-5.0, -5.0, 5.0, 5.0),
        shape=data_shape,
        data_type="richness_data",
        resampling_method="sum",
        band_name="memory_test_fungi",
        metadata={'test_data': True}
    )

def profile_chunked_storage():
    """Profile chunked storage with detailed memory monitoring."""
    print("🔬 Memory Profiling: Chunked Storage Implementation")
    print("=" * 60)
    
    # Baseline memory
    baseline = log_memory("Baseline")
    
    # Configure for chunked storage
    if 'storage' not in config.settings:
        config.settings['storage'] = {}
    
    config.settings['storage']['chunk_size'] = 10000      # Low threshold
    config.settings['storage']['chunk_rows'] = 50         # Small chunks
    config.settings['storage']['aggregate_to_grid'] = False
    config.settings['storage']['enable_progress_logging'] = False  # Reduce noise
    
    # Create test data
    print("\n🔧 Creating test data...")
    start_mem = get_memory_usage()
    test_data = create_test_data()
    data_mem = log_memory("After creating test data", start_mem)
    
    print(f"   Data shape: {test_data.shape}")
    print(f"   Data size: {test_data.nbytes / (1024*1024):.1f}MB")
    print(f"   Non-NaN values: {np.sum(~np.isnan(test_data)):,}")
    
    # Initialize components
    print("\n🔗 Initializing database and processor...")
    start_mem = get_memory_usage()
    db = DatabaseManager()
    processor = ResamplingProcessor(config, db)
    init_mem = log_memory("After initialization", start_mem)
    
    # Create dataset info
    dataset_info = create_test_dataset_info(test_data.shape)
    
    # Test chunked storage with detailed monitoring
    print("\n💾 Testing chunked storage (pixel-by-pixel)...")
    
    # Force garbage collection before test
    gc.collect()
    storage_start_mem = get_memory_usage()
    
    try:
        # Monitor during storage
        print("   Starting storage operation...")
        start_time = time.time()
        
        processor._store_resampled_dataset(dataset_info, test_data)
        
        end_time = time.time()
        storage_end_mem = log_memory("After chunked storage", storage_start_mem)
        
        print(f"   ✅ Storage completed in {end_time - start_time:.2f}s")
        
        # Test memory cleanup
        print("\n🧹 Testing memory cleanup...")
        cleanup_start_mem = get_memory_usage()
        
        # Delete large objects
        del test_data
        gc.collect()
        
        cleanup_end_mem = log_memory("After cleanup", cleanup_start_mem)
        
        # Test aggregated storage for comparison
        print("\n🌐 Testing aggregated storage...")
        
        # Create new test data
        test_data_2 = create_test_data()
        dataset_info_2 = create_test_dataset_info(test_data_2.shape)
        dataset_info_2.name = "memory-test-fungi-agg"
        
        # Configure for aggregation
        config.settings['storage']['aggregate_to_grid'] = True
        config.settings['storage']['grid_cell_size'] = 0.1
        
        agg_start_mem = get_memory_usage()
        
        processor._store_resampled_dataset(dataset_info_2, test_data_2)
        
        agg_end_mem = log_memory("After aggregated storage", agg_start_mem)
        
        # Compare storage efficiency
        print("\n📈 Storage Efficiency Analysis:")
        
        with db.get_connection() as conn:
            cur = conn.cursor()
            
            # Count records in both tables
            cur.execute("SELECT COUNT(*) FROM resampled_memory_test_fungi")
            pixel_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM resampled_memory_test_fungi_agg")
            grid_count = cur.fetchone()[0]
            
            print(f"   Pixel storage: {pixel_count:,} records") 
            print(f"   Grid storage: {grid_count:,} records")
            if grid_count > 0:
                print(f"   Reduction ratio: {pixel_count/grid_count:.1f}x")
                print(f"   Storage efficiency: {(1-grid_count/pixel_count)*100:.1f}% reduction")
        
        return True
        
    except Exception as e:
        error_mem = log_memory("After error", storage_start_mem)
        print(f"   ❌ Storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_data():
    """Clean up test data."""
    print("\n🧹 Cleaning up test data...")
    try:
        db = DatabaseManager()
        with db.get_connection() as conn:
            cur = conn.cursor()
            
            # Drop test tables
            test_tables = [
                'resampled_memory_test_fungi',
                'resampled_memory_test_fungi_agg'
            ]
            
            for table in test_tables:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            
            # Remove metadata
            cur.execute("DELETE FROM resampled_datasets WHERE name LIKE 'memory-test-fungi%'")
            conn.commit()
            
        print("   ✅ Cleanup completed")
    except Exception as e:
        print(f"   ⚠️  Cleanup failed: {e}")

if __name__ == "__main__":
    print("🔬 Memory Profiling Test Suite")
    print("Investigating memory usage patterns in chunked storage")
    print()
    
    success = False
    try:
        success = profile_chunked_storage()
    except KeyboardInterrupt:
        print(f"\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_test_data()
        
        # Final memory check
        print(f"\n📊 Final Memory State:")
        log_memory("Final")
    
    if success:
        print(f"\n✅ Memory profiling completed successfully")
    else:
        print(f"\n❌ Memory profiling encountered issues")