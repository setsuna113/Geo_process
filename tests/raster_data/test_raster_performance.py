# tests/raster_data/test_performance.py
import pytest
import time
import numpy as np
from pathlib import Path

from src.raster_data.loaders.geotiff_loader import GeoTIFFLoader

class TestRasterPerformance:
    """Performance tests for raster operations."""
    
    @pytest.mark.slow
    def test_large_raster_loading_speed(self, real_config, raster_helper, test_data_dir):
        """Test loading speed for large rasters."""
        # Create test raster (reduced size for faster testing)
        large_raster = test_data_dir / "perf_test.tif"
        raster_helper.create_test_raster(
            large_raster,
            width=500,  # Reduced from 5000
            height=500,  # Reduced from 5000
            pattern="gradient"
        )
        
        loader = GeoTIFFLoader(real_config)
        
        # Time metadata extraction
        start = time.time()
        metadata = loader.extract_metadata(large_raster)
        metadata_time = time.time() - start
        
        # Should be fast (no data loading)
        assert metadata_time < 0.5  # Less than 500ms
        
        # Time tile iteration
        loader.tile_size = 100  # Reduced tile size
        start = time.time()
        tile_count = 0
        
        for window, data in loader.iter_tiles(large_raster):
            tile_count += 1
            # Simulate light processing
            _ = np.mean(data)
        
        tile_time = time.time() - start
        tiles_per_second = tile_count / tile_time
        
        assert tile_count == 25  # 500/100 = 5x5 = 25 tiles
        assert tiles_per_second > 5  # Should process >5 tiles/second
        
    @pytest.mark.slow
    def test_point_sampling_performance(self, real_config, large_raster):
        """Test point sampling performance."""
        loader = GeoTIFFLoader(real_config)
        metadata = loader.extract_metadata(large_raster)
        
        # Generate random points
        n_points = 10000
        x_coords = np.random.uniform(metadata.bounds[0], metadata.bounds[2], n_points)
        y_coords = np.random.uniform(metadata.bounds[1], metadata.bounds[3], n_points)
        
        with loader.open_lazy(large_raster) as reader:
            start = time.time()
            
            values = []
            for x, y in zip(x_coords, y_coords):
                value = reader.read_point(x, y)
                values.append(value)
            
            elapsed = time.time() - start
        
        points_per_second = n_points / elapsed
        
        # Should be very fast for point sampling
        assert points_per_second > 1000  # >1000 points/second
        assert len(values) == n_points