#!/usr/bin/env python
"""Test fungi dataset processing with name fix."""

from src.processors.data_preparation.resampling_processor import ResamplingProcessor
from src.database.connection import DatabaseManager
from src.config import config
from pathlib import Path

print("Testing fungi dataset processing...")

# Setup
db = DatabaseManager()
processor = ResamplingProcessor(config, db)

# Test configs for both fungi datasets
fungi_configs = [
    {
        'name': 'am-fungi-richness',
        'path': '/maps/spun/AM_fungi/AM_Fungi_Richness_Predicted.tif', 
        'data_type': 'richness_data',
        'band_name': 'richness',
        'resampling_method': 'sum',
        'resolved_path': '/maps/spun/AM_fungi/AM_Fungi_Richness_Predicted.tif'
    },
    {
        'name': 'ecm-fungi-richness',
        'path': '/maps/spun/EcM_fungi/EcM_Fungi_Richness_Predicted.tif',
        'data_type': 'richness_data', 
        'band_name': 'richness',
        'resampling_method': 'sum',
        'resolved_path': '/maps/spun/EcM_fungi/EcM_Fungi_Richness_Predicted.tif'
    }
]

# Test each dataset
for config_data in fungi_configs:
    print(f"\n{'='*60}")
    print(f"Testing: {config_data['name']}")
    print(f"Path: {config_data['path']}")
    
    try:
        # First check if it exists in catalog
        from src.domain.raster.catalog import RasterCatalog
        catalog = RasterCatalog(db)
        existing = catalog.get_raster(config_data['name'])
        if existing:
            print(f"✓ Found in catalog with correct name")
            print(f"  Type: {type(existing)}")
            print(f"  Resolution: {existing.resolution_degrees}")
        else:
            print("✗ Not found in catalog (will be registered)")
        
        # Try resampling
        print("\nAttempting resampling...")
        result = processor.resample_dataset(config_data)
        
        print(f"✓ SUCCESS!")
        print(f"  Result type: {type(result)}")
        print(f"  Name: {result.name}")
        print(f"  Resolution: {result.target_resolution}")
        print(f"  Shape: {result.shape}")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        print("\nStack trace:")
        traceback.print_exc()
        
        # Additional debug info
        print("\nDEBUG: Checking what's in the database...")
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT name, file_path 
                FROM raster_sources 
                WHERE name LIKE '%fungi%' OR file_path LIKE '%fungi%'
                ORDER BY created_at DESC
                LIMIT 10
            """)
            print("Recent fungi entries:")
            for row in cursor.fetchall():
                print(f"  - {row['name']} -> {row['file_path']}")

print("\n" + "="*60)
print("Test complete!")