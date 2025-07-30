#!/usr/bin/env python3
"""Test BaseProcessor initialization fix."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.resampling_processor import ResamplingProcessor

print("Testing ResamplingProcessor initialization...")

try:
    # Create processor
    db = DatabaseManager()
    processor = ResamplingProcessor(config, db)
    
    print("✓ ResamplingProcessor initialized successfully!")
    
    # Check attributes
    print(f"  Config: {processor.config is not None}")
    print(f"  Database: {processor.db is not None}")
    print(f"  Catalog: {processor.catalog is not None}")
    print(f"  Cache manager: {processor.cache_manager is not None}")
    print(f"  Memory manager: {processor.memory_manager is not None}")
    
    # Check methods
    print(f"  Has resample_dataset_memory_aware: {hasattr(processor, 'resample_dataset_memory_aware')}")
    
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")