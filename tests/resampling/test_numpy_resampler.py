# tests/test_resampling/test_numpy_resampler.py
"""Unit tests for NumPy resampler."""

import pytest
import numpy as np

from src.resampling.engines.numpy_resampler import NumpyResampler
from src.resampling.engines.base_resampler import ResamplingConfig


class TestNumpyResampler:
    """Test NumPy resampler implementation."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ResamplingConfig(
            source_resolution=0.01,
            target_resolution=0.02,
            method='mean'
        )
    
    @pytest.fixture
    def resampler(self, config):
        """Create resampler instance."""
        return NumpyResampler(config)
    
    def test_nearest_neighbor(self, resampler):
        """Test nearest neighbor resampling."""
        source = np.array([[1, 2], [3, 4]])
        result = resampler._nearest_neighbor(source, (4, 4))
        
        assert result.shape == (4, 4)
        assert result[0, 0] == 1
        assert result[0, 2] == 2
        assert result[2, 0] == 3
        assert result[2, 2] == 4
    
    def test_bilinear(self, resampler):
        """Test bilinear interpolation."""
        source = np.array([[0, 10], [20, 30]], dtype=float)
        result = resampler._bilinear(source, (4, 4))
        
        assert result.shape == (4, 4)
        # Check interpolated values
        assert 0 < result[1, 1] < 30  # Should be between corners
    
    def test_mean_aggregate_downsampling(self, resampler):
        """Test mean aggregation for downsampling."""
        # Create 4x4 source that can be evenly divided into 2x2
        source = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ], dtype=float)
        
        result = resampler._mean_aggregate(source, (2, 2))
        
        assert result.shape == (2, 2)
        assert result[0, 0] == 1.0  # Mean of top-left block
        assert result[0, 1] == 2.0  # Mean of top-right block
        assert result[1, 0] == 3.0  # Mean of bottom-left block
        assert result[1, 1] == 4.0  # Mean of bottom-right block
    
    def test_mean_aggregate_with_nodata(self, resampler):
        """Test mean aggregation with nodata values."""
        resampler.config.nodata_value = -9999
        
        source = np.array([
            [1, 2, -9999, 4],
            [5, 6, -9999, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=float)
        
        result = resampler._mean_aggregate(source, (2, 2))
        
        assert result.shape == (2, 2)
        assert result[0, 0] == np.mean([1, 2, 5, 6])
        assert result[0, 1] == np.mean([4, 8])  # Excluding nodata
    
    def test_max_aggregate(self, resampler):
        """Test maximum aggregation."""
        source = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ])
        
        result = resampler._max_aggregate(source, (2, 2))
        
        assert result.shape == (2, 2)
        assert result[0, 0] == 6   # Max of top-left block
        assert result[0, 1] == 8   # Max of top-right block
        assert result[1, 0] == 14  # Max of bottom-left block
        assert result[1, 1] == 16  # Max of bottom-right block
    
    def test_min_aggregate(self, resampler):
        """Test minimum aggregation."""
        source = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ])
        
        result = resampler._min_aggregate(source, (2, 2))
        
        assert result.shape == (2, 2)
        assert result[0, 0] == 1   # Min of top-left block
        assert result[0, 1] == 3   # Min of top-right block
        assert result[1, 0] == 9   # Min of bottom-left block
        assert result[1, 1] == 11  # Min of bottom-right block
    
    def test_build_pixel_mapping_with_weights(self, resampler):
        """Test pixel mapping with area weights."""
        mapping = resampler._build_pixel_mapping(
            source_shape=(10, 10),
            source_bounds=(0, 0, 1, 1),
            target_shape=(5, 5),
            target_bounds=(0, 0, 1, 1)
        )
        
        assert mapping.shape[1] == 3  # target_idx, source_idx, weight
        assert np.all(mapping[:, 2] > 0)  # All weights positive
        assert np.all(mapping[:, 2] <= 1)  # All weights <= 1
    
    def test_calculate_coverage_map(self, resampler):
        """Test coverage map calculation."""
        # Source with some nodata
        source = np.ones((10, 10))
        source[:2, :2] = -9999
        resampler.config.nodata_value = -9999
        
        coverage = resampler._calculate_coverage_map(
            source,
            source_bounds=(0, 0, 1, 1),
            target_shape=(5, 5),
            target_bounds=(0, 0, 1, 1)
        )
        
        assert coverage.shape == (5, 5)
        assert coverage[0, 0] < 100  # Top-left has nodata
        assert coverage[4, 4] == 100  # Bottom-right fully covered