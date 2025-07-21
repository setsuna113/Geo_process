# tests/raster_data/test_geotiff_loader.py
import pytest
import numpy as np
from pathlib import Path

from src.raster_data.loaders.geotiff_loader import GeoTIFFLoader
from src.raster_data.loaders.base_loader import RasterWindow

class TestGeoTIFFLoader:
    """Test GeoTIFF loader functionality."""
    
    def test_can_handle(self, real_config):
        loader = GeoTIFFLoader(real_config)
        
        # Should handle various GeoTIFF extensions
        assert loader.can_handle(Path("test.tif"))
        assert loader.can_handle(Path("test.tiff"))
        assert loader.can_handle(Path("test.TIF"))
        assert loader.can_handle(Path("test.gtiff"))
        
        # Should not handle other formats
        assert not loader.can_handle(Path("test.nc"))
        assert not loader.can_handle(Path("test.jpg"))
        
    def test_extract_metadata(self, real_config, sample_raster):
        loader = GeoTIFFLoader(real_config)
        metadata = loader.extract_metadata(sample_raster)
        
        assert metadata.width == 200
        assert metadata.height == 200
        assert metadata.band_count == 1
        assert metadata.data_type in ["Int32", "Float32"]
        assert abs(metadata.resolution_degrees - 0.1) < 0.01
        assert metadata.crs is not None
        
    def test_metadata_caching(self, real_config, sample_raster):
        loader = GeoTIFFLoader(real_config)
        
        # First call
        metadata1 = loader.extract_metadata(sample_raster)
        
        # Second call should use cache
        metadata2 = loader.extract_metadata(sample_raster)
        
        assert metadata1 == metadata2
        
    def test_read_window(self, real_config, sample_raster):
        loader = GeoTIFFLoader(real_config)
        
        with loader.open_lazy(sample_raster) as reader:
            # Read small window (x, y, width, height)
            window = RasterWindow(10, 10, 20, 20)
            data = reader.read_window(window)
            
            assert data.shape == (20, 20)
            assert data.dtype in [np.int32, np.float32]
            
            # Check data has expected pattern (hotspots)
            assert np.max(data) > np.min(data)  # Not uniform
            
    def test_read_point(self, real_config, sample_raster):
        loader = GeoTIFFLoader(real_config)
        metadata = loader.extract_metadata(sample_raster)
        
        with loader.open_lazy(sample_raster) as reader:
            # Read center point
            center_x = (metadata.bounds[0] + metadata.bounds[2]) / 2
            center_y = (metadata.bounds[1] + metadata.bounds[3]) / 2
            
            value = reader.read_point(center_x, center_y)
            assert value is not None
            assert isinstance(value, float)
            
            # Read outside bounds
            value = reader.read_point(metadata.bounds[0] - 1, center_y)
            assert value is None
            
    def test_iter_tiles(self, real_config, sample_raster):
        loader = GeoTIFFLoader(real_config)
        loader.tile_size = 50  # 4x4 tiles for 200x200 raster
        
        tiles = list(loader.iter_tiles(sample_raster))
        
        assert len(tiles) == 16  # 4x4
        
        # Check tile properties
        for window, data in tiles:
            assert window.width <= 50
            assert window.height <= 50
            assert data.shape == (window.height, window.width)
            
    def test_load_window_geographic(self, real_config, sample_raster):
        loader = GeoTIFFLoader(real_config)
        metadata = loader.extract_metadata(sample_raster)
        
        # Load center quarter of raster
        west = metadata.bounds[0] + (metadata.bounds[2] - metadata.bounds[0]) * 0.25
        east = metadata.bounds[0] + (metadata.bounds[2] - metadata.bounds[0]) * 0.75
        south = metadata.bounds[1] + (metadata.bounds[3] - metadata.bounds[1]) * 0.25
        north = metadata.bounds[1] + (metadata.bounds[3] - metadata.bounds[1]) * 0.75
        
        data = loader.load_window(sample_raster, (west, south, east, north))
        
        assert data.shape == (100, 100)  # Half width and height
        
    def test_memory_efficiency(self, real_config, large_raster):
        """Test that lazy loading doesn't load entire raster."""
        import psutil
        import gc
        
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        loader = GeoTIFFLoader(real_config)
        
        with loader.open_lazy(large_raster) as reader:
            # Read small window from large raster
            window = RasterWindow(50, 50, 10, 10)  # Adjusted for 100x100 raster
            data = reader.read_window(window)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            # Should use much less memory than full raster
            # 1000x1000 int32 = ~4MB, but we only read 10x10
            assert memory_increase < 10  # Less than 10MB increase
            assert data.shape == (10, 10)