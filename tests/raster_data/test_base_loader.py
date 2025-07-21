# tests/raster_data/test_base_loader.py
import pytest
import numpy as np
from pathlib import Path

from src.raster_data.loaders.base_loader import (
    RasterWindow, RasterMetadata, BaseRasterLoader
)

class ConcreteLoader(BaseRasterLoader):
    """Concrete implementation for testing."""
    
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix == '.test'
    
    def extract_metadata(self, file_path: Path) -> RasterMetadata:
        return RasterMetadata(
            width=100,
            height=100,
            bounds=(-10, 40, 10, 60),
            crs="EPSG:4326",
            pixel_size=(0.2, -0.2),
            data_type="Int32",
            nodata_value=0,
            band_count=1
        )
    
    def _open_dataset(self, file_path: Path):
        return {"path": file_path, "open": True}
    
    def _read_window(self, dataset, window: RasterWindow, band: int = 1) -> np.ndarray:
        return np.ones((window.height, window.width), dtype=np.int32) * 42
    
    def _close_dataset(self, dataset):
        dataset["open"] = False

class TestRasterWindow:
    """Test RasterWindow dataclass."""
    
    def test_window_creation(self):
        window = RasterWindow(10, 20, 30, 40)
        assert window.col_off == 10
        assert window.row_off == 20
        assert window.width == 30
        assert window.height == 40
    
    def test_window_slice(self):
        window = RasterWindow(10, 20, 30, 40)
        row_slice, col_slice = window.slice
        assert row_slice == slice(20, 60)
        assert col_slice == slice(10, 40)

class TestRasterMetadata:
    """Test RasterMetadata dataclass."""
    
    def test_metadata_creation(self):
        metadata = RasterMetadata(
            width=1000,
            height=800,
            bounds=(-180, -90, 180, 90),
            crs="EPSG:4326",
            pixel_size=(0.1, -0.1),
            data_type="Float32",
            nodata_value=-9999,
            band_count=3
        )
        
        assert metadata.width == 1000
        assert metadata.resolution_degrees == 0.1
        
class TestBaseRasterLoader:
    """Test BaseRasterLoader functionality."""
    
    def test_bounds_to_window(self, real_config):
        loader = ConcreteLoader(real_config)
        metadata = loader.extract_metadata(Path("test.test"))
        
        # Test exact bounds
        window = loader._bounds_to_window((-10, 40, 10, 60), metadata)
        assert window.col_off == 0
        assert window.row_off == 0
        assert window.width == 100
        assert window.height == 100
        
        # Test partial bounds
        window = loader._bounds_to_window((-5, 45, 5, 55), metadata)
        assert window.col_off == 25
        assert window.row_off == 25
        assert window.width == 50
        assert window.height == 50
        
    def test_generate_tile_windows(self, real_config):
        loader = ConcreteLoader(real_config)
        loader.tile_size = 40  # Override for test
        metadata = loader.extract_metadata(Path("test.test"))
        
        windows = list(loader._generate_tile_windows(metadata))
        
        # Should have 3x3 tiles (100/40 rounds up)
        assert len(windows) == 9
        
        # Check first and last windows
        assert windows[0].col_off == 0
        assert windows[0].row_off == 0
        assert windows[0].width == 40
        assert windows[0].height == 40
        
        # Last window should be partial
        assert windows[-1].width == 20  # 100 - 80
        assert windows[-1].height == 20
        
    def test_lazy_loading_context(self, real_config, tmp_path):
        loader = ConcreteLoader(real_config)
        test_file = tmp_path / "test.test"
        test_file.touch()
        
        with loader.open_lazy(test_file) as reader:
            assert reader is not None
            data = reader.read_window(RasterWindow(0, 0, 10, 10))
            assert data.shape == (10, 10)
            assert np.all(data == 42)
            
    def test_memory_estimation(self, real_config):
        loader = ConcreteLoader(real_config)
        
        # Test different window sizes and data types
        window = RasterWindow(0, 0, 1000, 1000)
        
        # Int32 = 4 bytes per pixel, 1000x1000 = 1M pixels
        # 4M bytes / (1024*1024) = 3.814697 MB
        mem_mb = loader.estimate_memory_usage(window, np.dtype(np.int32))
        assert abs(mem_mb - 3.814697) < 0.01  # ~3.81MB
        
        # Float64 = 8 bytes per pixel
        # 8M bytes / (1024*1024) = 7.629395 MB  
        mem_mb = loader.estimate_memory_usage(window, np.dtype(np.float64))
        assert abs(mem_mb - 7.629395) < 0.01  # ~7.63MB