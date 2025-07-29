"""Performance tests for validation framework."""

import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import Mock

from src.domain.validators.coordinate_integrity import (
    BoundsConsistencyValidator, CoordinateTransformValidator, ParquetValueValidator
)
from src.abstractions.interfaces.validator import CompositeValidator


class TestValidationPerformance:
    """Test validation framework performance."""
    
    def test_bounds_validator_performance(self):
        """Test bounds validation performance."""
        validator = BoundsConsistencyValidator()
        
        # Time multiple validations
        iterations = 1000
        data = {
            'bounds': (-180.0, -90.0, 180.0, 90.0),
            'crs': 'EPSG:4326'
        }
        
        start = time.time()
        for _ in range(iterations):
            result = validator.validate(data)
        duration = time.time() - start
        
        # Should be very fast - less than 1ms per validation
        avg_time_ms = (duration / iterations) * 1000
        assert avg_time_ms < 1.0, f"Bounds validation too slow: {avg_time_ms:.2f}ms per validation"
        
        print(f"Bounds validation: {avg_time_ms:.3f}ms average")
    
    def test_value_validator_performance_small_dataset(self):
        """Test value validation performance on small datasets."""
        validator = ParquetValueValidator()
        
        # Small dataset (1000 rows)
        df = pd.DataFrame({
            'x': np.random.uniform(-180, 180, 1000),
            'y': np.random.uniform(-90, 90, 1000),
            'value': np.random.normal(0, 1, 1000)
        })
        
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            result = validator.validate(df)
        duration = time.time() - start
        
        # Should complete quickly - less than 10ms per validation
        avg_time_ms = (duration / iterations) * 1000
        assert avg_time_ms < 10.0, f"Small dataset validation too slow: {avg_time_ms:.2f}ms"
        
        print(f"Small dataset (1K rows): {avg_time_ms:.3f}ms average")
    
    def test_value_validator_performance_large_dataset(self):
        """Test value validation performance on large datasets."""
        validator = ParquetValueValidator()
        
        # Large dataset (1 million rows)
        size = 1_000_000
        df = pd.DataFrame({
            'x': np.random.uniform(-180, 180, size),
            'y': np.random.uniform(-90, 90, size),
            'temperature': np.random.normal(20, 5, size),
            'precipitation': np.random.exponential(50, size)
        })
        
        # Add some nulls
        null_indices = np.random.choice(size, size=int(size * 0.05), replace=False)
        df.loc[null_indices, 'temperature'] = np.nan
        
        # Time single validation
        start = time.time()
        result = validator.validate(df)
        duration = time.time() - start
        
        # Should complete within reasonable time - less than 2 seconds
        assert duration < 2.0, f"Large dataset validation too slow: {duration:.2f}s"
        
        print(f"Large dataset (1M rows): {duration:.3f}s")
        
        # Check result is valid
        assert result is not None
        assert hasattr(result, 'is_valid')
    
    def test_composite_validator_performance(self):
        """Test composite validator performance."""
        # Create composite validator
        composite = CompositeValidator([
            BoundsConsistencyValidator(),
            ParquetValueValidator()
        ])
        
        # Test data
        df = pd.DataFrame({
            'x': np.random.uniform(-10, 10, 10000),
            'y': np.random.uniform(-10, 10, 10000),
            'value': np.random.normal(100, 20, 10000)
        })
        
        bounds_data = {
            'bounds': (-10.0, -10.0, 10.0, 10.0),
            'crs': 'EPSG:4326',
            'dataframe': df
        }
        
        iterations = 50
        start = time.time()
        for _ in range(iterations):
            result = composite.validate(bounds_data)
        duration = time.time() - start
        
        avg_time_ms = (duration / iterations) * 1000
        assert avg_time_ms < 20.0, f"Composite validation too slow: {avg_time_ms:.2f}ms"
        
        print(f"Composite validation (10K rows): {avg_time_ms:.3f}ms average")
    
    def test_validation_memory_usage(self):
        """Test validation doesn't consume excessive memory."""
        import psutil
        import gc
        
        validator = ParquetValueValidator()
        
        # Get initial memory
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and validate large dataset
        size = 2_000_000
        df = pd.DataFrame({
            'x': np.random.uniform(-180, 180, size),
            'y': np.random.uniform(-90, 90, size),
            'value': np.random.normal(0, 1, size)
        })
        
        # Run validation multiple times
        for _ in range(5):
            result = validator.validate(df)
        
        # Check memory after validation
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f}MB increase"
        
        print(f"Memory usage: {memory_increase:.1f}MB increase for 2M row validation")
    
    def test_validation_scalability(self):
        """Test validation scales linearly with data size."""
        validator = ParquetValueValidator()
        
        sizes = [10000, 50000, 100000, 500000]
        times = []
        
        for size in sizes:
            df = pd.DataFrame({
                'x': np.random.uniform(-180, 180, size),
                'y': np.random.uniform(-90, 90, size),
                'value': np.random.normal(0, 1, size)
            })
            
            # Time validation
            start = time.time()
            result = validator.validate(df)
            duration = time.time() - start
            times.append(duration)
            
            print(f"Size {size:,}: {duration:.3f}s")
        
        # Check that time increases roughly linearly
        # Calculate ratios between consecutive sizes
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Time ratio should be within 2x of size ratio (allowing for overhead)
            assert time_ratio < size_ratio * 2, \
                f"Non-linear scaling: {size_ratio}x size increase caused {time_ratio}x time increase"
    
    def test_concurrent_validation_performance(self):
        """Test performance with concurrent validations."""
        import concurrent.futures
        
        validator = ParquetValueValidator()
        
        # Create test data
        df = pd.DataFrame({
            'x': np.random.uniform(-180, 180, 50000),
            'y': np.random.uniform(-90, 90, 50000),
            'value': np.random.normal(0, 1, 50000)
        })
        
        # Sequential validation
        sequential_start = time.time()
        for _ in range(10):
            result = validator.validate(df)
        sequential_time = time.time() - sequential_start
        
        # Concurrent validation
        concurrent_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(validator.validate, df) for _ in range(10)]
            results = [f.result() for f in futures]
        concurrent_time = time.time() - concurrent_start
        
        # Concurrent should be faster (at least 1.5x)
        speedup = sequential_time / concurrent_time
        assert speedup > 1.5, f"Insufficient concurrency speedup: {speedup:.2f}x"
        
        print(f"Concurrency speedup: {speedup:.2f}x")


if __name__ == "__main__":
    # Run with: python -m pytest tests/domain/validators/test_validation_performance.py -v -s
    pytest.main([__file__, "-v", "-s"])