# tests/test_resampling/test_base_resampler.py
"""Unit tests for base resampler functionality."""

import pytest
import numpy as np

from src.resampling.engines.base_resampler import (
    BaseResampler, ResamplingConfig, ResamplingResult
)


class TestResamplingConfig:
    """Test ResamplingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ResamplingConfig(
            source_resolution=0.01,
            target_resolution=0.1,
            method='bilinear'
        )
        
        assert config.source_resolution == 0.01
        assert config.target_resolution == 0.1
        assert config.method == 'bilinear'
        assert config.source_crs == 'EPSG:4326'
        assert config.cache_results is True
        assert config.chunk_size == 1000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ResamplingConfig(
            source_resolution=0.008983,
            target_resolution=0.08983,
            method='area_weighted',
            bounds=(-180, -90, 180, 90),
            preserve_sum=True,
            nodata_value=-9999,
            dtype=np.dtype(np.int32)
        )
        
        assert config.preserve_sum is True
        assert config.nodata_value == -9999
        assert config.dtype == np.dtype(np.int32)


class TestResamplingResult:
    """Test ResamplingResult dataclass."""
    
    def test_to_xarray(self):
        """Test conversion to xarray DataArray."""
        data = np.random.rand(100, 100)
        result = ResamplingResult(
            data=data,
            bounds=(-180, -90, 180, 90),
            resolution=0.1,
            crs='EPSG:4326',
            method='bilinear',
            metadata={'test': 'value'}
        )
        
        da = result.to_xarray()
        
        assert da.shape == (100, 100)
        assert 'lat' in da.dims
        assert 'lon' in da.dims
        assert da.attrs['crs'] == 'EPSG:4326'
        assert da.attrs['resolution'] == 0.1
        assert da.attrs['test'] == 'value'
    
    def test_coverage_map(self):
        """Test result with coverage map."""
        data = np.ones((50, 50))
        coverage = np.random.rand(50, 50) * 100
        
        result = ResamplingResult(
            data=data,
            bounds=(0, 0, 50, 50),
            resolution=1.0,
            crs='EPSG:4326',
            method='area_weighted',
            coverage_map=coverage
        )
        
        assert result.coverage_map is not None
        assert result.coverage_map.shape == data.shape
        assert np.all(result.coverage_map >= 0)
        assert np.all(result.coverage_map <= 100)


class ConcreteResampler(BaseResampler):
    """Concrete implementation for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.is_upsampling = False
        self.scale_factor = 1.0
        self._register_strategies()
    

    def _register_strategies(self):
        self.strategies = {'test': None}
    
    def resample(self, source_data, source_bounds=None, 
                 target_bounds=None, progress_callback=None):
        return ResamplingResult(
            data=np.ones((10, 10)),
            bounds=(0, 0, 10, 10),
            resolution=1.0,
            crs='EPSG:4326',
            method='test'
        )


class TestBaseResampler:
    """Test BaseResampler abstract class."""
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = ResamplingConfig(
            source_resolution=0.01,
            target_resolution=0.1,
            method='test'
        )
        resampler = ConcreteResampler(config)
        resampler.validate_config()
        
        # Should not raise
        resampler.validate_config()
        
        assert resampler.is_upsampling is False
        assert abs(resampler.scale_factor - 0.1) < 1e-10
    
    def test_validate_config_invalid_resolution(self):
        """Test validation with invalid resolution."""
        config = ResamplingConfig(
            source_resolution=-0.01,
            target_resolution=0.1,
            method='test'
        )
        resampler = ConcreteResampler(config)
        
        with pytest.raises(ValueError, match="Source resolution must be positive"):
            resampler.validate_config()
    
    def test_validate_config_unknown_method(self):
        """Test validation with unknown method."""
        config = ResamplingConfig(
            source_resolution=0.01,
            target_resolution=0.1,
            method='unknown'
        )
        resampler = ConcreteResampler(config)
        
        with pytest.raises(ValueError, match="Unknown resampling method"):
            resampler.validate_config()
    
    def test_calculate_output_shape(self):
        """Test output shape calculation."""
        config = ResamplingConfig(
            source_resolution=0.01,
            target_resolution=0.1,
            method='test'
        )
        resampler = ConcreteResampler(config)
        resampler.validate_config()
        
        # Test exact division
        shape = resampler.calculate_output_shape((0, 0, 10, 10))
        assert shape == (100, 100)
        
        # Test with rounding
        shape = resampler.calculate_output_shape((0, 0, 10.5, 10.5))
        assert shape == (105, 105)
    
    def test_calculate_coverage(self):
        """Test coverage calculation."""
        config = ResamplingConfig(
            source_resolution=0.01,
            target_resolution=0.02,
            method='test'
        )
        resampler = ConcreteResampler(config)
        resampler.validate_config()
        
        # Create mock mapping
        mapping = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],  # 4 source pixels -> target pixel 0
            [1, 4],
            [1, 5]   # 2 source pixels -> target pixel 1
        ])
        
        coverage = resampler.calculate_coverage(
            source_shape=(10, 10),
            target_shape=(2, 2),
            mapping=mapping
        )
        
        assert coverage.shape == (2, 2)
        assert coverage.flat[0] == 100.0  # 4/4 = 100%
        assert coverage.flat[1] == 50.0   # 2/4 = 50%
    
    def test_handle_dtype_conversion(self):
        """Test data type conversion with overflow handling."""
        config = ResamplingConfig(
            source_resolution=0.01,
            target_resolution=0.1,
            method='test',
            dtype=np.dtype(np.uint8)
        )
        resampler = ConcreteResampler(config)
        resampler.validate_config()
        
        # Test clipping
        data = np.array([-10, 0, 128, 300])
        converted = resampler.handle_dtype_conversion(data)
        
        assert converted.dtype == np.uint8
        assert converted[0] == 0      # Clipped from -10
        assert converted[1] == 0
        assert converted[2] == 128
        assert converted[3] == 255    # Clipped from 300