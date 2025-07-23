# tests/test_resampling/test_performance.py
"""Performance tests for resampling operations."""

import pytest
import numpy as np
import time
from contextlib import contextmanager

from src.resampling import GDALResampler, NumpyResampler
from src.resampling.engines.base_resampler import ResamplingConfig


@contextmanager
def time_operation(name: str):
    """Time an operation."""
    start = time.time()
    yield
    end = time.time()
    print(f"\n{name}: {end - start:.2f} seconds")


class TestResamplingPerformance:
    """Performance benchmarks for resampling."""
    
    @pytest.mark.benchmark
    def test_gdal_vs_numpy_performance(self):
        """Compare GDAL and NumPy resampler performance."""
        # Create test data
        sizes = [(10, 10), (20, 20), (50, 50)]
        
        for size in sizes:
            data = np.random.rand(*size).astype(np.float32)
            bounds = (0, 0, size[1], size[0])
            
            config = ResamplingConfig(
                source_resolution=1.0,
                target_resolution=2.0,
                method='bilinear'
            )
            
            # Test GDAL
            gdal_resampler = GDALResampler(config)
            with time_operation(f"GDAL {size}"):
                gdal_result = gdal_resampler.resample(data, bounds)
            
            # Test NumPy
            numpy_resampler = NumpyResampler(config)
            with time_operation(f"NumPy {size}"):
                numpy_result = numpy_resampler.resample(data, bounds)
            
            # Verify similar results
            assert gdal_result.data.shape == numpy_result.data.shape
    
    @pytest.mark.benchmark
    def test_different_methods_performance(self):
        """Benchmark different resampling methods."""
        data = np.random.rand(50, 50).astype(np.float32)
        bounds = (0, 0, 50, 50)
        
        methods = ['nearest', 'bilinear', 'mean', 'area_weighted']
        
        for method in methods:
            config = ResamplingConfig(
                source_resolution=1.0,
                target_resolution=5.0,
                method=method
            )
            
            resampler = NumpyResampler(config)
            
            with time_operation(f"Method: {method}"):
                result = resampler.resample(data, bounds)
                assert result.data is not None