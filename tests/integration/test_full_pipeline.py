#!/usr/bin/env python3
"""Test full pipeline integration with all stages."""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineOrchestrator

print("Full Pipeline Integration Test")
print("=" * 50)

# Create orchestrator
db = DatabaseManager()
orchestrator = PipelineOrchestrator(config, db)

# Configure test run
experiment_id = f"integration_test_{int(time.time())}"
output_dir = Path(f"output/integration_test/{experiment_id}")
checkpoint_dir = output_dir / "checkpoints"

print(f"\nExperiment ID: {experiment_id}")
print(f"Output directory: {output_dir}")
print(f"Memory-aware processing: {config.get('resampling.enable_memory_aware_processing', False)}")
print(f"Monitoring enabled: {config.get('monitoring.enable_database_logging', False)}")
print(f"Export formats: {config.get('export.formats', ['csv'])}")

# Test stage validation
print("\nValidating stages...")
stages = ['data_load', 'resample', 'merge', 'export']
print("  (Stage validation will be done during pipeline execution)")

# Run pipeline
print("\nRunning pipeline...")
try:
    start_time = time.time()
    
    # Execute pipeline
    result = orchestrator.run_pipeline(
        experiment_name=experiment_id,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        resume_from_checkpoint=False
    )
    success = result.get('success', False)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nPipeline completed in {elapsed_time:.1f} seconds")
    print(f"Success: {success}")
    
    # Check outputs
    if output_dir.exists():
        print(f"\nOutput files:")
        for file in output_dir.rglob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024**2)
                print(f"  - {file.relative_to(output_dir)} ({size_mb:.1f} MB)")
    
    # Check monitoring data
    if config.get('monitoring.enable_database_logging', False):
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Check logs
                cur.execute("""
                    SELECT COUNT(*) FROM pipeline_logs 
                    WHERE experiment_id = %s
                """, (experiment_id,))
                log_count = cur.fetchone()[0]
                
                # Check progress
                cur.execute("""
                    SELECT COUNT(*) FROM pipeline_progress 
                    WHERE experiment_id = %s
                """, (experiment_id,))
                progress_count = cur.fetchone()[0]
                
                print(f"\nMonitoring data:")
                print(f"  Logs written: {log_count}")
                print(f"  Progress updates: {progress_count}")
    
except Exception as e:
    print(f"\nPipeline failed: {e}")
    import traceback
    traceback.print_exc()
    success = False

# Summary
print(f"\n{'='*50}")
print(f"Integration test {'PASSED' if success else 'FAILED'}")
print(f"{'='*50}")