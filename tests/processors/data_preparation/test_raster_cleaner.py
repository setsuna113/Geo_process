# tests/processors/data_preparation/test_raster_cleaner.py
import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from src.processors.data_preparation.raster_cleaner import RasterCleaner, CleaningStats
from src.raster_data.loaders.base_loader import RasterWindow, RasterMetadata

class TestCleaningStats:
    """Test CleaningStats dataclass."""
    
    def test_cleaning_ratio_calculation(self):
        stats = CleaningStats(
            total_pixels=1000,
            nodata_pixels=100,
            outliers_removed=50,
            negative_values_fixed=20,
            capped_values=30,
            final_valid_pixels=800,
            value_range=(0, 1000)
        )
        
        # (50 + 20 + 30) / 1000 = 0.1
        assert stats.cleaning_ratio == 0.1
    
    def test_cleaning_ratio_zero_pixels(self):
        stats = CleaningStats(
            total_pixels=0,
            nodata_pixels=0,
            outliers_removed=0,
            negative_values_fixed=0,
            capped_values=0,
            final_valid_pixels=0,
            value_range=(0, 0)
        )
        
        assert stats.cleaning_ratio == 0.0

class TestRasterCleaner:
    """Test RasterCleaner with mocked dependencies."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.raster_processing.tile_size = 100
        config.raster_processing.get.return_value = 100  # For tile_size lookup
        config.get.return_value = {'log_operations': True}
        return config
    
    @pytest.fixture
    def mock_db(self):
        db = Mock()
        mock_conn = MagicMock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        db.get_connection.return_value = mock_conn
        return db
    
    @pytest.fixture
    def cleaner(self, mock_config, mock_db):
        with patch('src.processors.data_preparation.raster_cleaner.GeoTIFFLoader'):
            return RasterCleaner(mock_config, mock_db)
    
    def test_initialization(self, cleaner):
        assert cleaner.tile_size == 100
        assert cleaner.log_operations == True
        assert 'plants' in cleaner.value_constraints
        assert 'vertebrates' in cleaner.value_constraints
    
    def test_clean_tile_remove_negatives(self, cleaner):
        # Create test tile with negative values
        tile_data = np.array([[-10, 5, 100], [0, -5, 200], [50, 75, -1]])
        window = RasterWindow(0, 0, 3, 3)
        constraints = {'min': 0, 'max': 1000, 'outlier_std': 3}
        
        cleaned, stats, log = cleaner._clean_tile(
            tile_data, None, constraints, window
        )
        
        # Check negatives were fixed
        assert np.all(cleaned >= 0)
        assert stats.negative_values_fixed == 3
        assert len(log) > 0
        assert log[0]['operation'] == 'fix_negative'
    
    def test_clean_tile_remove_outliers(self, cleaner):
        # Create test tile with outliers
        tile_data = np.array([[10, 20, 30], [15, 25, 35], [20, 30, 1000]])
        window = RasterWindow(0, 0, 3, 3)
        constraints = {'min': 0, 'max': 2000, 'outlier_std': 2}
        
        cleaned, stats, log = cleaner._clean_tile(
            tile_data, None, constraints, window
        )
        
        # Check outlier was capped
        assert np.max(cleaned) < 1000
        assert stats.outliers_removed > 0
    
    def test_clean_tile_apply_constraints(self, cleaner):
        # Create test tile exceeding max constraint
        tile_data = np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]])
        window = RasterWindow(0, 0, 3, 3)
        constraints = {'min': 0, 'max': 500, 'outlier_std': 0}  # No outlier removal
        
        cleaned, stats, log = cleaner._clean_tile(
            tile_data, None, constraints, window
        )
        
        # Check values were capped at max
        assert np.max(cleaned) == 500
        assert stats.capped_values == 4  # Values 600, 700, 800, 900
    
    def test_clean_tile_handle_nodata(self, cleaner):
        # Create test tile with nodata
        tile_data = np.array([[10, -9999, 30], [40, 50, -9999], [70, 80, 90]])
        window = RasterWindow(0, 0, 3, 3)
        constraints = {'min': 0, 'max': 1000, 'outlier_std': 3}
        
        cleaned, stats, log = cleaner._clean_tile(
            tile_data, -9999, constraints, window
        )
        
        # Check nodata values preserved
        assert np.sum(cleaned == -9999) == 2
        assert stats.nodata_pixels == 2
        assert stats.final_valid_pixels == 7
    
    @patch('src.processors.data_preparation.raster_cleaner.logger')
    def test_clean_raster_full_process(self, mock_logger, cleaner):
        # Mock loader methods
        mock_metadata = RasterMetadata(
            width=100, height=100,
            bounds=(-10, 40, 10, 60),
            crs="EPSG:4326",
            pixel_size=(0.2, -0.2),
            data_type="Int32",
            nodata_value=-9999,
            band_count=1
        )
        
        cleaner.loader.extract_metadata.return_value = mock_metadata
        
        # Mock tile iteration
        test_tile = np.random.randint(-50, 500, size=(10, 10))
        test_windows = [RasterWindow(0, 0, 10, 10)]
        cleaner.loader.iter_tiles.return_value = [(test_windows[0], test_tile)]
        
        # Mock lazy loading context
        mock_reader = Mock()
        cleaner.loader.open_lazy.return_value.__enter__ = Mock(return_value=mock_reader)
        cleaner.loader.open_lazy.return_value.__exit__ = Mock(return_value=None)
        
        # Run cleaning
        result = cleaner.clean_raster(
            Path("test.tif"),
            dataset_type="plants",
            output_path=None,
            in_place=False
        )
        
        assert 'statistics' in result
        assert 'metadata' in result
        assert result['metadata']['dataset_type'] == 'plants'
        assert mock_logger.info.called
    
    def test_update_stats(self, cleaner):
        total_stats = CleaningStats(
            total_pixels=100,
            nodata_pixels=10,
            outliers_removed=5,
            negative_values_fixed=3,
            capped_values=2,
            final_valid_pixels=80,
            value_range=(0, 100)
        )
        
        tile_stats = CleaningStats(
            total_pixels=50,
            nodata_pixels=5,
            outliers_removed=2,
            negative_values_fixed=1,
            capped_values=1,
            final_valid_pixels=40,
            value_range=(10, 150)
        )
        
        cleaner._update_stats(total_stats, tile_stats)
        
        assert total_stats.total_pixels == 150
        assert total_stats.nodata_pixels == 15
        assert total_stats.outliers_removed == 7
        assert total_stats.value_range == (0, 150)