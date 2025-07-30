#!/usr/bin/env python3
"""Simple pipeline test to verify basic functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineContext
from src.pipelines.stages.load_stage import DataLoadStage

# Create context
db = DatabaseManager()
context = PipelineContext(
    config=config,
    db=db,
    output_dir=Path("output/test_pipeline"),
    checkpoint_dir=Path("output/test_pipeline/checkpoints"),
    experiment_id="test_simple"
)

# Test just the load stage
load_stage = DataLoadStage()

print("Testing DataLoadStage...")
print(f"Stage name: {load_stage.name}")
print(f"Dependencies: {load_stage.dependencies}")
print(f"Memory requirements: {load_stage.memory_requirements} GB")

# Validate
valid, errors = load_stage.validate()
print(f"Validation: {'PASS' if valid else 'FAIL'}")
if errors:
    for error in errors:
        print(f"  - {error}")

# Execute
try:
    result = load_stage.execute(context)
    print(f"\nExecution result:")
    print(f"  Success: {result.success}")
    print(f"  Metrics: {result.metrics}")
    if result.warnings:
        print(f"  Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")
    
    # Check what was loaded
    loaded_datasets = context.get('loaded_datasets', [])
    print(f"\nLoaded datasets: {len(loaded_datasets)}")
    for ds in loaded_datasets:
        print(f"  - {ds['name']}: {ds['path']}")
        print(f"    Needs resampling: {ds['needs_resampling']}")
        if ds['source_resolution']:
            print(f"    Resolution: {ds['source_resolution']:.6f}°")
        
except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")