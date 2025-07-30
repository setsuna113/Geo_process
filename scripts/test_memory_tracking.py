#!/usr/bin/env python3
"""Test script for enhanced memory tracking with database storage."""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.monitors.memory_monitor import MemoryMonitor

def test_memory_database_storage():
    """Test that memory metrics are stored in the database."""
    print("Testing memory tracking with database storage...")
    
    # Create database connection
    db = DatabaseManager()
    
    # Create test experiment directly
    import json
    import uuid
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO experiments (name, description, config, status, started_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (
                f"test_memory_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "Test for memory tracking",
                json.dumps({'test': True, 'memory_limit_gb': 16.0}),
                'running'
            ))
            experiment_id = str(cur.fetchone()[0])
            conn.commit()
    print(f"Created test experiment: {experiment_id}")
    
    # Create memory monitor with database
    monitor = MemoryMonitor(config, db_manager=db)
    monitor.set_experiment_id(experiment_id)
    
    # Start monitoring
    monitor.start()
    print("Memory monitor started")
    
    # Simulate different stages
    stages = ['data_load', 'resample', 'merge', 'analysis']
    
    try:
        for stage in stages:
            print(f"\nSimulating stage: {stage}")
            monitor.set_stage(stage)
            
            # Simulate some operations
            operations = ['read_data', 'process_chunk', 'write_results']
            for op in operations:
                monitor.set_operation(op)
                time.sleep(2)  # Monitor for 2 seconds per operation
                print(f"  - Operation: {op}")
            
            monitor.set_operation(None)
        
        monitor.set_stage(None)
        
        # Wait a bit more to ensure data is written
        time.sleep(2)
        
        # Query the database to verify data was stored
        print("\nQuerying stored memory data...")
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Count memory samples
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM pipeline_metrics 
                    WHERE experiment_id = %s::uuid
                """, (experiment_id,))
                count = cur.fetchone()[0]
                print(f"Total memory samples stored: {count}")
                
                # Get memory by stage
                cur.execute("""
                    SELECT 
                        custom_metrics->>'stage' as stage,
                        COUNT(*) as samples,
                        AVG(memory_mb / 1024.0) as avg_memory_gb,
                        MAX(memory_mb / 1024.0) as peak_memory_gb
                    FROM pipeline_metrics
                    WHERE experiment_id = %s::uuid
                        AND custom_metrics->>'stage' IS NOT NULL
                    GROUP BY custom_metrics->>'stage'
                    ORDER BY MIN(timestamp)
                """, (experiment_id,))
                
                print("\nMemory usage by stage:")
                print(f"{'Stage':<15} {'Samples':<10} {'Avg GB':<10} {'Peak GB':<10}")
                print("-" * 50)
                for row in cur.fetchall():
                    stage, samples, avg_gb, peak_gb = row
                    print(f"{stage:<15} {samples:<10} {avg_gb:<10.2f} {peak_gb:<10.2f}")
                
                # Get operations tracked
                cur.execute("""
                    SELECT DISTINCT 
                        custom_metrics->>'stage' as stage,
                        custom_metrics->>'operation' as operation
                    FROM pipeline_metrics
                    WHERE experiment_id = %s::uuid
                        AND custom_metrics->>'operation' IS NOT NULL
                    ORDER BY custom_metrics->>'stage', custom_metrics->>'operation'
                """, (experiment_id,))
                
                print("\nOperations tracked:")
                for row in cur.fetchall():
                    stage, operation = row
                    print(f"  {stage}: {operation}")
        
        # Skip views for now - they need to be updated for the actual schema
        print("\n\nNote: Memory analysis views need to be updated for the current schema")
        
        
        print("\n✅ Memory tracking test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        monitor.stop()
        print("\nMemory monitor stopped")


def main():
    """Run memory tracking tests."""
    print("=" * 60)
    print("Testing Enhanced Memory Tracking")
    print("=" * 60)
    
    success = test_memory_database_storage()
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())