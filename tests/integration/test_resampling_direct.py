#!/usr/bin/env python3
"""Direct test of resampling processor memory-aware functionality."""

import sys
from pathlib import Path
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.resampling_processor import ResamplingProcessor

# Monitor memory usage
def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

print("Direct test of ResamplingProcessor memory-aware method...")
print(f"Memory-aware processing enabled: {config.get('resampling.enable_memory_aware_processing', False)}")

# Create processor
db = DatabaseManager()
processor = ResamplingProcessor(config, db)

# Test dataset config
test_dataset = {
    'name': 'plants-richness',
    'path': '/maps/mwd24/richness/daru-plants-richness.tif',
    'resolved_path': '/maps/mwd24/richness/daru-plants-richness.tif'
}

print(f"\nTesting with dataset: {test_dataset['name']}")
print(f"Path: {test_dataset['path']}")

# Track memory before
initial_memory = get_memory_mb()
print(f"\nInitial memory usage: {initial_memory:.1f} MB")

try:
    # Test legacy method (should show deprecation warning)
    print("\n--- Testing legacy resample_dataset method ---")
    result_legacy = processor.resample_dataset(test_dataset)
    legacy_memory = get_memory_mb()
    print(f"Legacy method memory after: {legacy_memory:.1f} MB (+{legacy_memory - initial_memory:.1f} MB)")
    print(f"Result: {result_legacy.dataset_name}, shape: {result_legacy.shape}")
    
    # Test memory-aware method
    print("\n--- Testing memory-aware resample_dataset_memory_aware method ---")
    result_memory_aware = processor.resample_dataset_memory_aware(test_dataset)
    memory_aware_memory = get_memory_mb()
    print(f"Memory-aware method memory after: {memory_aware_memory:.1f} MB (+{memory_aware_memory - initial_memory:.1f} MB)")
    print(f"Result: {result_memory_aware.dataset_name}, shape: {result_memory_aware.shape}")
    
    # Compare results
    print(f"\nBoth methods produced same shape: {result_legacy.shape == result_memory_aware.shape}")
    print(f"Memory difference: {memory_aware_memory - legacy_memory:.1f} MB")
    
except Exception as e:
    print(f"\nError during testing: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")