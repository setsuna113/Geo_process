#!/usr/bin/env python3
"""Test why existing datasets are not found"""

import sys
sys.path.insert(0, '.')

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.resampling_processor import ResamplingProcessor

config = Config()
db = DatabaseManager()
processor = ResamplingProcessor(config, db)

print("Testing get_resampled_dataset...")

# Test for plants-richness
dataset_name = "plants-richness"
print(f"\nLooking for: {dataset_name}")
result = processor.get_resampled_dataset(dataset_name)
if result:
    print(f"  Found: {result.name}")
    print(f"  Resolution: {result.target_resolution}")
    print(f"  Table: {result.metadata.get('data_table_name', 'N/A')}")
else:
    print("  Not found!")

# Check database directly
print("\nChecking database directly...")
with db.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT name, resampling_method, data_table_name FROM resampled_datasets")
        rows = cur.fetchall()
        for row in rows:
            print(f"  DB: {row}")