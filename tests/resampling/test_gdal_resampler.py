# tests/test_resampling/test_gdal_resampler.py
"""Unit tests for GDAL resampler."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.resampling.engines.gdal_resampler import GDALResampler
from src.resampling.engines.base_resampler import ResamplingConfig


class TestGDALResampler:
    """Test GDAL resampler implementation."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ResamplingConfig(
            source_resolution=0.01,
            target_resolution=0.1,
            method='bilinear'
        )
    
    @pytest.fixture
    def resampler(self, config):
        """Create resampler instance."""
        return GDALResampler(config)
    
    def test_register_strategies(self, resampler):
        """Test strategy registration."""
        assert 'area_weighted' in resampler.strategies
        assert 'sum' in resampler.strategies
        assert 'majority' in resampler.strategies
        assert 'bilinear' in resampler.strategies
        assert 'nearest' in resampler.strategies
    
    @patch('src.resampling.engines.gdal_resampler.gdal')
    def test_resample_gdal_method(self, mock_gdal, resampler):
        """Test resampling with GDAL built-in method."""
        # Mock GDAL objects
        mock_driver = Mock()
        mock_src_ds = Mock()
        mock_tgt_ds = Mock()
        mock_band = Mock()
        
        mock_gdal.GetDriverByName.return_value = mock_driver
        mock_driver.Create.side_effect = [mock_src_ds, mock_tgt_ds]
        mock_src_ds.GetRasterBand.return_value = mock_band
        mock_tgt_ds.GetRasterBand.return_value = mock_band
        mock_band.ReadAsArray.return_value = np.ones((10, 10))
        
        # Test data
        source_data = np.random.rand(100, 100)
        source_bounds = (0, 0, 1, 1)
        
        # Perform resampling
        result = resampler.resample(source_data, source_bounds)
        
        # Verify GDAL calls
        mock_gdal.GetDriverByName.assert_called_with('MEM')
        mock_gdal.ReprojectImage.assert_called_once()
        
        assert result.data.shape == (10, 10)
        assert result.method == 'bilinear'
    
    def test_resample_custom_strategy(self):
        """Test resampling with custom strategy."""
        config = ResamplingConfig(
            source_resolution=0.01,
            target_resolution=0.02,
            method='area_weighted'
        )
        resampler = GDALResampler(config)
        
        # Mock the strategy
        mock_strategy = Mock()
        mock_strategy.resample.return_value = np.ones((50, 50))
        resampler.strategies['area_weighted'] = mock_strategy
        
        # Test data
        source_data = np.random.rand(100, 100)
        source_bounds = (0, 0, 1, 1)
        
        # Perform resampling
        result = resampler.resample(source_data, source_bounds)
        
        # Verify strategy was called
        mock_strategy.resample.assert_called_once()
        assert result.data.shape == (50, 50)
    
    def test_build_pixel_mapping(self, resampler):
        """Test pixel mapping calculation."""
        mapping = resampler._build_pixel_mapping(
            source_shape=(100, 100),
            source_bounds=(0, 0, 1, 1),
            target_shape=(10, 10),
            target_bounds=(0, 0, 1, 1)
        )
        
        assert mapping.shape[1] == 2  # target_idx, source_idx
        assert np.all(mapping[:, 0] >= 0)
        assert np.all(mapping[:, 0] < 100)  # 10x10 = 100 target pixels
    
    @patch('src.resampling.engines.gdal_resampler.gdal')
    def test_calculate_gdal_coverage(self, mock_gdal, resampler):
        """Test coverage calculation."""
        # Mock GDAL objects
        mock_src_ds = Mock()
        mock_tgt_ds = Mock()
        mock_band = Mock()
        
        mock_src_ds.GetRasterBand.return_value = mock_band
        mock_src_ds.RasterXSize = 100
        mock_src_ds.RasterYSize = 100
        mock_tgt_ds.RasterXSize = 10
        mock_tgt_ds.RasterYSize = 10
        
        # Create validity mask with some nodata
        validity_data = np.ones((100, 100))
        validity_data[:10, :10] = 0  # 10% nodata
        mock_band.ReadAsArray.return_value = validity_data
        
        # Mock coverage calculation
        mock_driver = Mock()
        mock_gdal.GetDriverByName.return_value = mock_driver
        mock_coverage_ds = Mock()
        mock_coverage_band = Mock()
        mock_driver.Create.side_effect = [Mock(), mock_coverage_ds]
        mock_coverage_ds.GetRasterBand.return_value = mock_coverage_band
        mock_coverage_band.ReadAsArray.return_value = np.ones((10, 10)) * 0.9
        
        coverage = resampler._calculate_gdal_coverage(mock_src_ds, mock_tgt_ds)
        
        assert coverage.shape == (10, 10)
        assert np.all(coverage <= 100)
        assert np.all(coverage >= 0)