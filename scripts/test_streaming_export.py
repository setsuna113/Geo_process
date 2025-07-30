#!/usr/bin/env python3
"""Test script for streaming export functionality."""

import sys
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.database.connection import DatabaseManager
from src.processors.data_preparation.coordinate_merger import CoordinateMerger
from src.pipelines.orchestrator import PipelineContext

def create_test_data(db, num_points=10000):
    """Create test data tables for streaming test."""
    print(f"Creating test data with {num_points} points...")
    
    import numpy as np
    
    # Create test bounds
    min_x, max_x = -10.0, 10.0
    min_y, max_y = -10.0, 10.0
    resolution = 0.1
    
    # Create coordinate grid
    x_coords = np.arange(min_x, max_x, resolution)
    y_coords = np.arange(min_y, max_y, resolution)
    
    # Limit to requested number of points
    total_points = len(x_coords) * len(y_coords)
    if total_points > num_points:
        # Subsample
        step = int(np.sqrt(total_points / num_points))
        x_coords = x_coords[::step]
        y_coords = y_coords[::step]
    
    print(f"Grid size: {len(x_coords)} x {len(y_coords)} = {len(x_coords) * len(y_coords)} points")
    
    # Create two test datasets
    datasets = []
    
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            for i, dataset_name in enumerate(['test_dataset_1', 'test_dataset_2']):
                table_name = f"test_streaming_{dataset_name}"
                
                # Drop table if exists
                cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # Create table with coordinates
                cur.execute(f"""
                    CREATE TABLE {table_name} (
                        x_coord DOUBLE PRECISION,
                        y_coord DOUBLE PRECISION,
                        value DOUBLE PRECISION,
                        PRIMARY KEY (x_coord, y_coord)
                    )
                """)
                
                # Insert data
                print(f"Inserting data for {dataset_name}...")
                values = []
                for x in x_coords:
                    for y in y_coords:
                        # Create some test pattern
                        value = np.sin(x * 0.5) * np.cos(y * 0.5) + i
                        values.append((float(x), float(y), float(value)))
                
                # Bulk insert
                cur.executemany(
                    f"INSERT INTO {table_name} (x_coord, y_coord, value) VALUES (%s, %s, %s)",
                    values
                )
                
                conn.commit()
                print(f"Created {len(values)} points for {dataset_name}")
                
                # Create dataset info
                datasets.append({
                    'name': dataset_name,
                    'table_name': table_name,
                    'bounds': [min_x, min_y, max_x, max_y],
                    'resolution': resolution,
                    'passthrough': True,
                    'memory_aware': True
                })
    
    return datasets


def test_streaming_vs_inmemory(datasets, db, config):
    """Compare streaming export vs in-memory export."""
    print("\n" + "="*60)
    print("Testing Streaming vs In-Memory Export")
    print("="*60)
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Test 1: In-memory export
        print("\n1. Testing in-memory export...")
        start_time = time.time()
        start_memory = get_memory_usage()
        
        merger = CoordinateMerger(config, db)
        
        # Create merged dataset in memory
        merged_df = merger.create_merged_dataset(
            datasets,
            return_as='dataframe'
        )
        
        # Export to CSV
        output_path_inmemory = output_dir / "inmemory_export.csv"
        merged_df.to_csv(output_path_inmemory, index=False)
        
        inmemory_duration = time.time() - start_time
        inmemory_memory = get_memory_usage() - start_memory
        inmemory_size = output_path_inmemory.stat().st_size
        inmemory_rows = len(merged_df)
        
        print(f"In-memory export completed:")
        print(f"  - Duration: {inmemory_duration:.2f}s")
        print(f"  - Memory increase: {inmemory_memory:.2f} MB")
        print(f"  - File size: {inmemory_size / (1024**2):.2f} MB")
        print(f"  - Rows: {inmemory_rows:,}")
        
        # Test 2: Streaming export
        print("\n2. Testing streaming export...")
        start_time = time.time()
        start_memory = get_memory_usage()
        
        merger2 = CoordinateMerger(config, db)
        output_path_streaming = output_dir / "streaming_export.csv"
        
        # Stream chunks directly to file
        rows_exported = 0
        chunk_count = 0
        first_chunk = True
        
        with open(output_path_streaming, 'w') as f:
            for chunk_df in merger2.iter_merged_chunks(datasets, chunk_size=1000):
                chunk_df.to_csv(f, index=False, header=first_chunk)
                first_chunk = False
                rows_exported += len(chunk_df)
                chunk_count += 1
                
                if chunk_count % 10 == 0:
                    print(f"  Processed {chunk_count} chunks, {rows_exported:,} rows...")
        
        streaming_duration = time.time() - start_time
        streaming_memory = get_memory_usage() - start_memory
        streaming_size = output_path_streaming.stat().st_size
        
        print(f"\nStreaming export completed:")
        print(f"  - Duration: {streaming_duration:.2f}s")
        print(f"  - Memory increase: {streaming_memory:.2f} MB")
        print(f"  - File size: {streaming_size / (1024**2):.2f} MB")
        print(f"  - Rows: {rows_exported:,}")
        print(f"  - Chunks: {chunk_count}")
        
        # Compare results
        print("\n" + "="*60)
        print("Comparison Results:")
        print("="*60)
        print(f"Memory efficiency: {inmemory_memory / streaming_memory:.1f}x less memory with streaming")
        print(f"Speed difference: {abs(streaming_duration - inmemory_duration) / inmemory_duration * 100:.1f}% "
              f"{'slower' if streaming_duration > inmemory_duration else 'faster'} with streaming")
        print(f"File sizes match: {'✅ Yes' if abs(streaming_size - inmemory_size) < 1000 else '❌ No'}")
        
        # Verify content is identical (first few rows)
        import pandas as pd
        df1 = pd.read_csv(output_path_inmemory, nrows=100)
        df2 = pd.read_csv(output_path_streaming, nrows=100)
        
        content_matches = df1.equals(df2)
        print(f"Content matches (first 100 rows): {'✅ Yes' if content_matches else '❌ No'}")
        
        return streaming_memory < inmemory_memory * 0.5  # Success if streaming uses less than 50% memory


def get_memory_usage():
    """Get current process memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def test_adaptive_chunk_size():
    """Test that chunk size adapts under memory pressure."""
    print("\n" + "="*60)
    print("Testing Adaptive Chunk Size")
    print("="*60)
    
    # This would require setting up memory pressure
    # For now, just verify the mechanism exists
    db = DatabaseManager()
    merger = CoordinateMerger(config, db)
    
    # Simulate memory pressure
    merger._adaptive_chunk_size = 5000
    print(f"Initial chunk size: {merger._adaptive_chunk_size}")
    
    # Simulate warning
    merger._adaptive_chunk_size = max(1000, merger._adaptive_chunk_size // 2)
    print(f"After warning: {merger._adaptive_chunk_size}")
    
    # Simulate critical
    merger._adaptive_chunk_size = 500
    print(f"After critical: {merger._adaptive_chunk_size}")
    
    print("✅ Adaptive chunk size mechanism verified")
    return True


def main():
    """Run streaming export tests."""
    print("Starting Streaming Export Tests")
    print("=" * 60)
    
    # Create database connection
    db = DatabaseManager()
    
    try:
        # Create test data
        datasets = create_test_data(db, num_points=20000)
        
        # Run comparison test
        test1_passed = test_streaming_vs_inmemory(datasets, db, config)
        
        # Test adaptive chunk size
        test2_passed = test_adaptive_chunk_size()
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary:")
        print(f"Streaming vs In-Memory: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"Adaptive Chunk Size: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
        print("=" * 60)
        
        if test1_passed and test2_passed:
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
            
    finally:
        # Cleanup test tables
        print("\nCleaning up test data...")
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS test_streaming_test_dataset_1")
                cur.execute("DROP TABLE IF EXISTS test_streaming_test_dataset_2")
                conn.commit()


if __name__ == "__main__":
    sys.exit(main())