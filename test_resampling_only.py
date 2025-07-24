#!/usr/bin/env python3
"""Test just the resampling processor to identify where the hang occurs."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Set environment for test mode
os.environ['FORCE_TEST_MODE'] = 'true'
os.environ['DB_NAME'] = 'geoprocess_db'

def test_resampling_only():
    """Test only the resampling processor without the full pipeline."""
    print("=== Testing Resampling Processor Only ===")
    
    try:
        from src.config.config import Config
        from src.database.connection import DatabaseManager
        from src.processors.data_preparation.resampling_processor import ResamplingProcessor
        
        print("1. Loading configuration...")
        config = Config()
        
        # Override config for testing
        config.settings['resampling']['target_resolution'] = 0.2  # Even larger for speed
        config.settings['datasets']['target_datasets'] = [
            {
                'name': 'plants-richness-test',
                'path': str(project_root / 'data' / 'richness_maps' / 'daru-plants-richness.tif'),
                'data_type': 'richness_data',
                'band_name': 'plants_richness',
                'enabled': True
            }
        ]
        print("✅ Configuration loaded")
        
        print("2. Creating database connection...")
        db = DatabaseManager()
        print("✅ Database connected")
        
        print("3. Creating resampling processor...")
        processor = ResamplingProcessor(config, db)
        print("✅ Processor created")
        
        print("4. Testing single dataset resampling...")
        dataset_config = config.settings['datasets']['target_datasets'][0]
        
        def progress_callback(message, percent):
            print(f"   Progress: {message} ({percent:.1f}%)")
        
        print(f"   Processing: {dataset_config['name']}")
        print(f"   File: {dataset_config['path']}")
        
        # This is where it likely hangs
        result = processor.resample_dataset(dataset_config, progress_callback)
        
        print("✅ Resampling completed!")
        print(f"   Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_resampling_only()
    sys.exit(0 if success else 1)