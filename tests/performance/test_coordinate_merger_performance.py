"""Performance tests for CoordinateMerger chunked processing."""

import unittest
import time
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import psutil

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


class TestCoordinateMergerPerformance(unittest.TestCase):
    """Performance tests for CoordinateMerger chunked operations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.output_dir = Path(cls.temp_dir) / "output"
        cls.output_dir.mkdir(exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)
        
    def create_mock_dataframe(self, num_points, dataset_name, bounds=None):
        """Create a mock dataframe simulating coordinate data."""
        if bounds is None:
            bounds = (-180, -90, 180, 90)
            
        min_x, min_y, max_x, max_y = bounds
        
        # Create random coordinates within bounds
        np.random.seed(42)  # For reproducibility
        x_coords = np.random.uniform(min_x, max_x, num_points)
        y_coords = np.random.uniform(min_y, max_y, num_points)
        values = np.random.rand(num_points)
        
        return pd.DataFrame({
            'x': x_coords,
            'y': y_coords,
            dataset_name: values
        })
        
    def test_chunk_calculation_performance(self):
        """Test performance of chunk boundary calculations."""
        overall_bounds = (-180, -90, 180, 90)
        min_resolution = 0.1  # 0.1 degree resolution
        
        chunk_sizes = [100, 500, 1000, 5000]
        calculation_times = []
        
        for chunk_size in chunk_sizes:
            start_time = time.time()
            
            # Calculate chunk grid
            min_x, min_y, max_x, max_y = overall_bounds
            width = max_x - min_x
            height = max_y - min_y
            
            chunks_x = max(1, int(width / (chunk_size * min_resolution)))
            chunks_y = max(1, int(height / (chunk_size * min_resolution)))
            
            chunk_width = width / chunks_x
            chunk_height = height / chunks_y
            
            # Generate all chunk bounds
            chunk_bounds = []
            for i in range(chunks_x):
                for j in range(chunks_y):
                    bounds = (
                        min_x + i * chunk_width,
                        min_y + j * chunk_height,
                        min_x + (i + 1) * chunk_width,
                        min_y + (j + 1) * chunk_height
                    )
                    chunk_bounds.append(bounds)
                    
            calc_time = time.time() - start_time
            calculation_times.append((chunk_size, len(chunk_bounds), calc_time))
            
            # Should be very fast
            self.assertLess(calc_time, 0.1)
            
        print("\nChunk Calculation Performance:")
        for chunk_size, num_chunks, calc_time in calculation_times:
            print(f"  Chunk size {chunk_size}: {num_chunks} chunks calculated in {calc_time:.6f}s")
            
    def test_merge_performance_scaling(self):
        """Test merge performance with increasing data sizes."""
        dataset_sizes = [1000, 10000, 50000]  # Number of points per dataset
        merge_times = []
        
        for size in dataset_sizes:
            # Create test dataframes
            df1 = self.create_mock_dataframe(size, 'dataset1')
            df2 = self.create_mock_dataframe(size, 'dataset2')
            
            # Round coordinates for merging
            for df in [df1, df2]:
                df['x'] = df['x'].round(6)
                df['y'] = df['y'].round(6)
                
            start_time = time.time()
            
            # Perform merge
            merged = df1.merge(df2, on=['x', 'y'], how='outer')
            
            merge_time = time.time() - start_time
            merge_times.append((size, len(merged), merge_time))
            
            # Verify merge worked
            self.assertGreater(len(merged), 0)
            self.assertIn('dataset1', merged.columns)
            self.assertIn('dataset2', merged.columns)
            
        print("\nCoordinate Merge Performance:")
        for size, result_size, merge_time in merge_times:
            print(f"  {size:,} points per dataset: {result_size:,} merged points in {merge_time:.3f}s")
            
    def test_bounded_query_performance(self):
        """Test performance of spatially bounded queries."""
        # Create a large dataset
        large_df = self.create_mock_dataframe(100000, 'large_dataset')
        
        # Test different query bounds
        query_bounds_list = [
            (-10, -10, 10, 10),      # Small central area
            (-90, -45, 90, 45),      # Half the world
            (-180, -90, 0, 0),       # Quarter of the world
            (-180, -90, 180, 90)     # Full world
        ]
        
        query_times = []
        
        for bounds in query_bounds_list:
            min_x, min_y, max_x, max_y = bounds
            
            start_time = time.time()
            
            # Perform spatial filter
            filtered = large_df[
                (large_df['x'] >= min_x) & 
                (large_df['x'] <= max_x) &
                (large_df['y'] >= min_y) & 
                (large_df['y'] <= max_y)
            ]
            
            query_time = time.time() - start_time
            query_times.append((bounds, len(filtered), query_time))
            
        print("\nSpatial Query Performance (100K points):")
        for bounds, result_count, query_time in query_times:
            area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            print(f"  Area {area:,} sq degrees: {result_count:,} points in {query_time:.6f}s")
            
    def test_memory_efficiency_chunked_processing(self):
        """Test memory efficiency of chunked processing simulation."""
        # Simulate processing large datasets in chunks
        total_points = 1000000  # 1M points total
        chunk_sizes = [10000, 50000, 100000, 500000]
        
        process = psutil.Process()
        memory_usage = []
        
        for chunk_size in chunk_sizes:
            # Reset by forcing garbage collection
            import gc
            gc.collect()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = initial_memory
            
            # Simulate chunked processing
            num_chunks = (total_points + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                # Create chunk data
                actual_size = min(chunk_size, total_points - i * chunk_size)
                chunk_df = self.create_mock_dataframe(actual_size, f'chunk_{i}')
                
                # Simulate some processing
                _ = chunk_df.describe()
                _ = chunk_df['x'].mean()
                
                # Track peak memory
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                # Clear chunk data
                del chunk_df
                
            memory_increase = peak_memory - initial_memory
            memory_usage.append((chunk_size, num_chunks, memory_increase))
            
        print("\nMemory Efficiency (Processing 1M points):")
        for chunk_size, num_chunks, mem_increase in memory_usage:
            print(f"  Chunk size {chunk_size:,}: {num_chunks} chunks, "
                  f"peak memory increase: {mem_increase:.1f} MB")
            
    def test_parquet_write_performance(self):
        """Test performance of writing merged data to parquet."""
        sizes = [10000, 50000, 100000]
        write_times = []
        
        for size in sizes:
            # Create merged dataframe with multiple columns
            df = pd.DataFrame({
                'x': np.random.uniform(-180, 180, size),
                'y': np.random.uniform(-90, 90, size),
                'dataset1': np.random.rand(size),
                'dataset2': np.random.rand(size),
                'dataset3': np.random.rand(size),
                'dataset4': np.random.rand(size),
                'dataset5': np.random.rand(size)
            })
            
            output_path = self.output_dir / f"test_{size}.parquet"
            
            start_time = time.time()
            df.to_parquet(output_path, compression='snappy')
            write_time = time.time() - start_time
            
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            write_times.append((size, write_time, file_size_mb))
            
            # Clean up
            output_path.unlink()
            
        print("\nParquet Write Performance:")
        for size, write_time, file_size in write_times:
            throughput = file_size / write_time  # MB/s
            print(f"  {size:,} rows: {write_time:.3f}s, "
                  f"{file_size:.1f} MB, {throughput:.1f} MB/s")


if __name__ == '__main__':
    unittest.main()