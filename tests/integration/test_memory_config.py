#!/usr/bin/env python3
"""Test memory-aware configuration and basic functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import config

print("Memory-Aware Processing Configuration:")
print("=" * 50)

# Check resampling configuration
resampling_config = config.get('resampling', {})
print("\nResampling settings:")
for key, value in resampling_config.items():
    print(f"  {key}: {value}")

# Check monitoring configuration
monitoring_config = config.get('monitoring', {})
print("\nMonitoring settings:")
for key, value in monitoring_config.items():
    print(f"  {key}: {value}")

# Check merge configuration
merge_config = config.get('merge', {})
print("\nMerge settings:")
for key, value in merge_config.items():
    print(f"  {key}: {value}")

# Check export configuration
export_config = config.get('export', {})
print("\nExport settings:")
for key, value in export_config.items():
    print(f"  {key}: {value}")

# Check if WindowedStorageManager is available
print("\n" + "=" * 50)
print("Testing WindowedStorageManager import...")
try:
    from src.processors.data_preparation.windowed_storage import WindowedStorageManager
    print("✓ WindowedStorageManager imported successfully")
    
    # Check its methods
    methods = [m for m in dir(WindowedStorageManager) if not m.startswith('_')]
    print(f"  Available methods: {', '.join(methods[:5])}...")
    
except ImportError as e:
    print(f"✗ Failed to import WindowedStorageManager: {e}")

# Check if memory-aware methods exist
print("\nChecking ResamplingProcessor methods...")
try:
    from src.processors.data_preparation.resampling_processor import ResamplingProcessor
    
    # Check if memory-aware method exists
    if hasattr(ResamplingProcessor, 'resample_dataset_memory_aware'):
        print("✓ resample_dataset_memory_aware method exists")
    else:
        print("✗ resample_dataset_memory_aware method NOT found")
        
    if hasattr(ResamplingProcessor, '_handle_passthrough_memory_aware'):
        print("✓ _handle_passthrough_memory_aware method exists")
    else:
        print("✗ _handle_passthrough_memory_aware method NOT found")
        
    if hasattr(ResamplingProcessor, '_handle_resampling_memory_aware'):
        print("✓ _handle_resampling_memory_aware method exists")
    else:
        print("✗ _handle_resampling_memory_aware method NOT found")
        
except Exception as e:
    print(f"✗ Error checking ResamplingProcessor: {e}")

print("\nConfiguration test complete!")