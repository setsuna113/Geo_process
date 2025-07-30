#!/usr/bin/env python3
"""Test memory-aware resampling functionality."""

import sys
from pathlib import Path
import psutil
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineContext
from src.pipelines.stages.load_stage import DataLoadStage
from src.pipelines.stages.resample_stage import ResampleStage

# Monitor memory usage
def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# Create context
db = DatabaseManager()
context = PipelineContext(
    config=config,
    db=db,
    output_dir=Path("output/test_memory_aware"),
    checkpoint_dir=Path("output/test_memory_aware/checkpoints"),
    experiment_id="test_memory_aware"
)

print("Testing Memory-Aware Resampling...")
print(f"Memory-aware processing enabled: {config.get('resampling.enable_memory_aware_processing', False)}")
print(f"Window size: {config.get('resampling.window_size', 2048)}")
print(f"Window overlap: {config.get('resampling.window_overlap', 128)}")

# First, load datasets
load_stage = DataLoadStage()
load_result = load_stage.execute(context)
print(f"\nLoaded {len(context.get('loaded_datasets', []))} datasets")

# Track memory before resampling
initial_memory = get_memory_mb()
print(f"\nInitial memory usage: {initial_memory:.1f} MB")

# Test resampling stage
resample_stage = ResampleStage()
print(f"\nTesting ResampleStage...")
print(f"Stage name: {resample_stage.name}")
print(f"Dependencies: {resample_stage.dependencies}")
print(f"Memory requirements: {resample_stage.memory_requirements} GB")

# Execute resampling
try:
    start_time = time.time()
    
    # Monitor memory during execution
    print("\nExecuting resampling...")
    result = resample_stage.execute(context)
    
    elapsed_time = time.time() - start_time
    peak_memory = get_memory_mb()
    
    print(f"\nExecution result:")
    print(f"  Success: {result.success}")
    print(f"  Time elapsed: {elapsed_time:.1f} seconds")
    print(f"  Peak memory usage: {peak_memory:.1f} MB")
    print(f"  Memory increase: {peak_memory - initial_memory:.1f} MB")
    
    print(f"\nMetrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Check resampled datasets
    resampled_datasets = context.get('resampled_datasets', [])
    print(f"\nResampled datasets: {len(resampled_datasets)}")
    for ds in resampled_datasets:
        print(f"  - {ds.dataset_name}")
        print(f"    Shape: {ds.shape}")
        print(f"    Target resolution: {ds.target_resolution:.6f}°")
        print(f"    Passthrough: {ds.metadata.get('passthrough', False)}")
        
except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()

# Final memory check
final_memory = get_memory_mb()
print(f"\nFinal memory usage: {final_memory:.1f} MB")
print(f"Total memory change: {final_memory - initial_memory:.1f} MB")

print("\nTest complete!")