#!/usr/bin/env python3
"""Test script to verify pipeline fixes"""

import sys
sys.path.insert(0, '.')

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.resampling_processor import ResamplingProcessor

def test_passthrough_storage():
    """Test that passthrough data gets stored correctly"""
    print("Testing passthrough data storage...")
    
    config = Config()
    db = DatabaseManager()
    
    # Get dataset configs
    datasets = config.get('datasets.target_datasets', [])
    enabled_datasets = [d for d in datasets if d.get('enabled', True)]
    
    print(f"Found {len(enabled_datasets)} enabled datasets")
    
    # Create resampling processor
    processor = ResamplingProcessor(config, db)
    
    # Only test with terrestrial-richness (smaller file)
    test_dataset = [d for d in enabled_datasets if d['name'] == 'terrestrial-richness']
    if not test_dataset:
        print("ERROR: terrestrial-richness dataset not found")
        return
        
    for dataset in test_dataset:
        print(f"\nProcessing {dataset['name']}...")
        
        # Check if dataset exists
        existing = processor.get_resampled_dataset(dataset['name'])
        if existing:
            print(f"  - Found existing dataset (will be cleaned if table missing)")
        
        # Process dataset
        try:
            result = processor.resample_dataset(dataset)
            print(f"  - Processing complete: {result.resampling_method}")
            
            # Verify data was stored
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    table_name = f"passthrough_{dataset['name'].replace('-', '_')}"
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cur.fetchone()[0]
                    print(f"  - Stored {count:,} values in {table_name}")
                    
        except Exception as e:
            print(f"  - ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_passthrough_storage()