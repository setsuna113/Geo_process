#!/usr/bin/env python3
"""Test average aggregation strategy for SDM predictions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.domain.resampling.strategies.average_aggregation import AverageAggregationStrategy


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self, source_res=0.01, target_res=0.02, nodata=-9999):
        self.source_resolution = source_res
        self.target_resolution = target_res  # Larger value = coarser = downsampling
        self.nodata_value = nodata


def test_average_downsampling():
    """Test average aggregation for downsampling (most common case)."""
    strategy = AverageAggregationStrategy()
    
    # Create test data: 4x4 grid with values 0-15
    # [[0,  1,  2,  3],
    #  [4,  5,  6,  7],
    #  [8,  9, 10, 11],
    #  [12,13, 14, 15]]
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    target_shape = (2, 2)  # Downsample by 2x
    
    # Create mapping: each target pixel gets 4 source pixels
    # Mapping format: [target_idx, source_idx, weight]
    # source_idx is flattened index: row*width + col
    mapping = []
    
    # Build mapping for 2x2 blocks
    for tgt_row in range(2):
        for tgt_col in range(2):
            tgt_idx = tgt_row * 2 + tgt_col
            # Each target pixel covers a 2x2 block in source
            for src_row in range(tgt_row*2, tgt_row*2+2):
                for src_col in range(tgt_col*2, tgt_col*2+2):
                    src_idx = src_row * 4 + src_col
                    mapping.append([tgt_idx, src_idx, 1.0])
    
    mapping = np.array(mapping)
    
    config = MockConfig()
    result = strategy.resample(source, target_shape, mapping, config)
    
    # Expected averages:
    # (0+1+4+5)/4 = 2.5, (2+3+6+7)/4 = 4.5
    # (8+9+12+13)/4 = 10.5, (10+11+14+15)/4 = 12.5
    expected = np.array([[2.5, 4.5], [10.5, 12.5]])
    
    np.testing.assert_array_almost_equal(result, expected)
    print("✓ Average downsampling test passed")


def test_average_with_nodata():
    """Test average aggregation with nodata values."""
    strategy = AverageAggregationStrategy()
    
    # Create test data with nodata
    source = np.array([[1.0, 2.0, -9999, 4.0],
                       [5.0, -9999, 7.0, 8.0],
                       [9.0, 10.0, 11.0, -9999],
                       [-9999, 14.0, 15.0, 16.0]], dtype=np.float32)
    
    target_shape = (2, 2)
    
    # Same mapping as before
    mapping = np.array([
        [0, 0, 1.0], [0, 1, 1.0], [0, 4, 1.0], [0, 5, 1.0],
        [1, 2, 1.0], [1, 3, 1.0], [1, 6, 1.0], [1, 7, 1.0],
        [2, 8, 1.0], [2, 9, 1.0], [2, 12, 1.0], [2, 13, 1.0],
        [3, 10, 1.0], [3, 11, 1.0], [3, 14, 1.0], [3, 15, 1.0]
    ])
    
    config = MockConfig(nodata=-9999)
    result = strategy.resample(source, target_shape, mapping, config)
    
    # Expected averages (excluding nodata):
    # Target 0: source indices [0,1,4,5] -> values [1,2,-9999,5] -> avg of [1,2,5] = 8/3
    # Target 1: source indices [2,3,6,7] -> values [-9999,4,7,8] -> avg of [4,7,8] = 19/3  
    # Target 2: source indices [8,9,12,13] -> values [9,10,-9999,14] -> avg of [9,10,14] = 33/3 = 11
    # Target 3: source indices [10,11,14,15] -> values [11,-9999,15,16] -> avg of [11,15,16] = 42/3 = 14
    expected = np.array([[8/3, 19/3], [11.0, 14.0]])
    
    np.testing.assert_array_almost_equal(result, expected, decimal=2)
    print("✓ Average with nodata test passed")


def test_area_weighted_average():
    """Test area-weighted averaging."""
    strategy = AverageAggregationStrategy()
    
    # Simple 2x2 -> 1x1 case with unequal area weights
    source = np.array([[10.0, 20.0],
                       [30.0, 40.0]], dtype=np.float32)
    
    target_shape = (1, 1)
    
    # Mapping with area weights (third column)
    # Let's say pixel 0 covers 40%, pixel 1 covers 20%, pixel 2 covers 30%, pixel 3 covers 10%
    mapping = np.array([
        [0, 0, 0.4],  # 40% from source pixel 0 (value=10)
        [0, 1, 0.2],  # 20% from source pixel 1 (value=20)
        [0, 2, 0.3],  # 30% from source pixel 2 (value=30)
        [0, 3, 0.1]   # 10% from source pixel 3 (value=40)
    ])
    
    config = MockConfig()
    result = strategy.resample(source, target_shape, mapping, config)
    
    # Expected: 10*0.4 + 20*0.2 + 30*0.3 + 40*0.1 = 4 + 4 + 9 + 4 = 21
    expected = np.array([[21.0]])
    
    np.testing.assert_array_almost_equal(result, expected)
    print("✓ Area-weighted average test passed")


if __name__ == "__main__":
    test_average_downsampling()
    test_average_with_nodata()
    test_area_weighted_average()
    print("\nAll tests passed! ✨")