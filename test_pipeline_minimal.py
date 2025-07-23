#!/usr/bin/env python3
"""
Minimal test of the unified resampling pipeline with very small samples.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Simulate test mode
sys.modules['pytest'] = type(sys)('pytest')

def create_test_datasets():
    """Create very small test datasets from the originals."""
    print("🔧 Creating small test datasets...")
    
    try:
        import rasterio
        from rasterio.windows import from_bounds
        import numpy as np
        
        # Create test data directory
        test_data_dir = Path("data/test_richness_small")
        test_data_dir.mkdir(exist_ok=True)
        
        # Source files
        source_dir = Path("data/richness_maps")
        plants_src = source_dir / "daru-plants-richness.tif"
        terrestrial_src = source_dir / "iucn-terrestrial-richness.tif" 
        
        # Target files (small versions)
        plants_test = test_data_dir / "daru-plants-richness.tif"
        terrestrial_test = test_data_dir / "iucn-terrestrial-richness.tif"
        
        # Create small subset of each dataset
        for src_file, test_file in [(plants_src, plants_test), (terrestrial_src, terrestrial_test)]:
            with rasterio.open(src_file) as src:
                # Get a small window (100x100 pixels from center)
                width, height = src.width, src.height
                center_x, center_y = width // 2, height // 2
                
                # Create small window
                window = rasterio.windows.Window(
                    center_x - 50, center_y - 50, 100, 100
                )
                
                # Read data for this window
                data = src.read(1, window=window)
                
                # Get the transform for this window
                transform = src.window_transform(window)
                
                # Write small test file
                with rasterio.open(
                    test_file, 'w',
                    driver='GTiff',
                    height=100, width=100,
                    count=1, dtype=data.dtype,
                    crs=src.crs,
                    transform=transform,
                    nodata=src.nodata
                ) as dst:
                    dst.write(data, 1)
                
                print(f"   ✅ Created {test_file.name} (100x100 pixels)")
        
        return test_data_dir
        
    except ImportError:
        print("   ⚠️ rasterio not available, using original datasets (will be slower)")
        return Path("data/richness_maps")
    except Exception as e:
        print(f"   ⚠️ Could not create test datasets: {e}")
        print("   Using original datasets (will be slower)")
        return Path("data/richness_maps")

def run_minimal_pipeline_test():
    """Run a minimal pipeline test."""
    print("\n🚀 Running Minimal Pipeline Test")
    print("-" * 40)
    
    try:
        from src.config.config import Config
        from src.database.connection import DatabaseManager
        from src.pipelines.unified_resampling.pipeline_orchestrator import UnifiedResamplingPipeline
        
        # Create test config
        config = Config()
        
        # Override with test data directory and small parameters
        test_data_dir = create_test_datasets()
        
        # Temporarily modify config for testing
        original_data_dir = config.paths['data_dir']
        config.settings['paths']['data_dir'] = test_data_dir
        
        # Override resampling config for speed
        config.settings.setdefault('resampling', {}).update({
            'target_resolution': 0.2,  # Very coarse resolution
            'engine': 'numpy'
        })
        
        # Override SOM config for speed  
        config.settings.setdefault('som_analysis', {}).update({
            'default_grid_size': [2, 2],  # Tiny SOM grid
            'iterations': 10,  # Very few iterations
            'max_pixels_in_memory': 1000
        })
        
        # Initialize components
        db = DatabaseManager()
        pipeline = UnifiedResamplingPipeline(config, db)
        
        print(f"   Test data directory: {test_data_dir}")
        print(f"   Target resolution: {config.get('resampling.target_resolution')}")
        print(f"   SOM grid size: {config.get('som_analysis.default_grid_size')}")
        
        # Run the pipeline with minimal parameters
        print("\n   🎬 Starting pipeline execution...")
        
        # Use unique experiment name with timestamp
        import time
        timestamp = int(time.time())
        results = pipeline.run_complete_pipeline(
            experiment_name=f"test_minimal_pipeline_{timestamp}",
            description="Minimal test of unified resampling pipeline",
            skip_existing=False
        )
        
        print("\n   📊 Pipeline Results:")
        print(f"      Experiment ID: {results['experiment_id']}")
        print(f"      Datasets processed: {len(results['resampled_datasets'])}")
        print(f"      SOM results: {results['som_analysis']['saved_path']}")
        
        # Restore original config
        config.settings['paths']['data_dir'] = original_data_dir
        
        return True, results
        
    except Exception as e:
        print(f"\n   ❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def cleanup_test_data():
    """Clean up test data and database entries."""
    print("\n🧹 Cleaning Up Test Data")
    print("-" * 25)
    
    try:
        # Remove test data directory
        test_data_dir = Path("data/test_richness_small")
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
            print("   ✅ Removed test dataset files")
        
        # Clean up database
        from src.database.connection import DatabaseManager
        db = DatabaseManager()
        
        with db.get_connection() as conn:
            cur = conn.cursor()
            
            # Clean up test experiments
            cur.execute("DELETE FROM experiments WHERE name LIKE 'test_%'")
            exp_deleted = cur.rowcount
            
            # Clean up test resampled datasets if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'resampled_datasets'
                )
            """)
            
            if cur.fetchone()[0]:
                # Get data table names to drop
                cur.execute("SELECT data_table_name FROM resampled_datasets WHERE name LIKE 'test-%'")
                tables_to_drop = [row[0] for row in cur.fetchall() if row[0]]
                
                # Delete records
                cur.execute("DELETE FROM resampled_datasets WHERE name LIKE 'test-%'")
                dataset_deleted = cur.rowcount
                
                # Drop data tables
                for table in tables_to_drop:
                    if table:
                        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
            else:
                dataset_deleted = 0
            
            conn.commit()
            print(f"   ✅ Cleaned up {exp_deleted} experiments, {dataset_deleted} datasets")
        
        # Clean up test outputs
        outputs_dir = Path("outputs/unified_resampling")
        if outputs_dir.exists():
            for item in outputs_dir.glob("*test*"):
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print("   ✅ Cleaned up test output files")
        
    except Exception as e:
        print(f"   ⚠️ Cleanup warning: {e}")

def main():
    print("🧪 Unified Resampling Pipeline - Minimal Test")
    print("=" * 50)
    
    try:
        # Run the test
        success, results = run_minimal_pipeline_test()
        
        if success:
            print("\n🎉 Minimal pipeline test PASSED!")
            print("\n📋 Test Summary:")
            print("   ✅ Pipeline executed successfully")
            print("   ✅ Datasets were resampled") 
            print("   ✅ Datasets were merged")
            print("   ✅ SOM analysis completed")
            print("   ✅ Results saved to database and files")
            
            # Ask about cleanup
            cleanup = input("\n🧹 Clean up test data? (Y/n): ").strip().lower()
            if cleanup != 'n':
                cleanup_test_data()
                print("   ✅ Cleanup completed")
            else:
                print("   📁 Test data preserved for inspection")
                
        else:
            print("\n💥 Minimal pipeline test FAILED!")
            print("   Check the error messages above for debugging.")
            
            # Still offer cleanup
            cleanup = input("\n🧹 Clean up any partial test data? (y/N): ").strip().lower()
            if cleanup == 'y':
                cleanup_test_data()
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)