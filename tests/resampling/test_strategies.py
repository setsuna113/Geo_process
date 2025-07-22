# tests/test_resampling/test_strategies.py
"""Unit tests for resampling strategies."""

import numpy as np
from unittest.mock import Mock

from src.resampling.strategies.area_weighted import AreaWeightedStrategy
from src.resampling.strategies.sum_aggregation import SumAggregationStrategy
from src.resampling.strategies.majority_vote import MajorityVoteStrategy


class TestAreaWeightedStrategy:
    """Test area-weighted resampling strategy."""
    
    def test_resample_basic(self):
        """Test basic area-weighted resampling."""
        strategy = AreaWeightedStrategy()
        
        # Simple 2x2 source
        source = np.array([[1, 2], [3, 4]], dtype=float)
        
        # Mapping: all source pixels contribute to target pixel 0
        mapping = np.array([
            [0, 0, 0.25],
            [0, 1, 0.25],
            [0, 2, 0.25],
            [0, 3, 0.25]
        ])
        
        config = Mock()
        config.nodata_value = None
        
        result = strategy.resample(source, (1, 1), mapping, config)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == 2.5  # Weighted average
    
    def test_resample_with_nodata(self):
        """Test resampling with nodata values."""
        strategy = AreaWeightedStrategy()
        
        source = np.array([[1, -999], [3, 4]], dtype=float)
        mapping = np.array([
            [0, 0, 0.25],
            [0, 1, 0.25],
            [0, 2, 0.25],
            [0, 3, 0.25]
        ])
        
        config = Mock()
        config.nodata_value = -999
        
        result = strategy.resample(source, (1, 1), mapping, config)
        
        # Should exclude nodata from average
        expected = (1 * 0.25 + 3 * 0.25 + 4 * 0.25) / 0.75
        assert np.isclose(result[0, 0], expected)
    
    def test_resample_with_progress(self):
        """Test resampling with progress callback."""
        strategy = AreaWeightedStrategy()
        
        source = np.random.rand(100)
        mapping = np.array([[i, i, 1.0] for i in range(100)])
        
        config = Mock()
        config.nodata_value = None
        
        progress_calls = []
        def progress_callback(p):
            progress_calls.append(p)
        
        result = strategy.resample(source, (10, 10), mapping, config, progress_callback)
        
        assert result is not None
        assert len(progress_calls) > 0
        assert progress_calls[-1] == 100


class TestSumAggregationStrategy:
    """Test sum aggregation strategy."""
    
    def test_downsample_sum(self):
        """Test downsampling with sum aggregation."""
        strategy = SumAggregationStrategy()
        
        # 2x2 source with counts
        source = np.array([[1, 2], [3, 4]], dtype=float)
        
        # All source pixels map to single target
        mapping = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3]
        ])
        
        config = Mock()
        config.nodata_value = None
        config.source_resolution = 0.01
        config.target_resolution = 0.02
        
        result = strategy.resample(source, (1, 1), mapping, config)
        
        assert result[0, 0] == 10  # Sum of all values
    
    def test_upsample_distribute(self):
        """Test upsampling with count distribution."""
        strategy = SumAggregationStrategy()
        
        # Single source pixel with count
        source = np.array([[100]], dtype=float)
        
        # Source pixel 0 maps to 4 target pixels
        mapping = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0]
        ])
        
        config = Mock()
        config.nodata_value = None
        config.source_resolution = 0.02
        config.target_resolution = 0.01
        
        result = strategy._upsample_counts(source, (2, 2), mapping, config, None)
        
        assert result.sum() == 100  # Total preserved
        assert np.all(result == 25)  # Evenly distributed


class TestMajorityVoteStrategy:
    """Test majority vote strategy."""
    
    def test_simple_majority(self):
        """Test simple majority voting."""
        strategy = MajorityVoteStrategy()
        
        # Categorical data
        source = np.array([[1, 1], [2, 1]], dtype=int)
        
        # All map to single target
        mapping = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3]
        ])
        
        config = Mock()
        config.nodata_value = None
        
        result = strategy.resample(source, (1, 1), mapping, config)
        
        assert result[0, 0] == 1  # Majority value
    
    def test_weighted_majority(self):
        """Test weighted majority voting."""
        strategy = MajorityVoteStrategy()
        
        source = np.array([[1, 2], [2, 3]], dtype=int)
        
        # With weights
        mapping = np.array([
            [0, 0, 0.1],   # class 1, weight 0.1
            [0, 1, 0.3],   # class 2, weight 0.3  
            [0, 2, 0.4],   # class 2, weight 0.4
            [0, 3, 0.2]    # class 3, weight 0.2
        ])
        
        config = Mock()
        config.nodata_value = None
        
        result = strategy.resample(source, (1, 1), mapping, config)
        
        assert result[0, 0] == 2  # Class 2 has total weight 0.7