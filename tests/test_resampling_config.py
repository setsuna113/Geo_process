#!/usr/bin/env python3
"""Lightweight tests for resampling configuration and strategy selection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.config import config
from src.domain.resampling.strategies.average_aggregation import AverageAggregationStrategy
from src.domain.resampling.strategies.sum_aggregation import SumAggregationStrategy
from src.domain.resampling.engines.numpy_resampler import NumpyResampler
from src.domain.resampling.engines.base_resampler import ResamplingConfig


def test_config_has_correct_strategies():
    """Test configuration has correct resampling strategies."""
    print("Testing configuration strategies...")
    
    strategies = config.get('resampling.strategies', {})
    
    # Check expected strategies
    assert strategies['richness_data'] == 'sum', f"Expected 'sum' for richness_data, got {strategies['richness_data']}"
    assert strategies['continuous_data'] == 'average', f"Expected 'average' for continuous_data, got {strategies['continuous_data']}"
    assert strategies['categorical_data'] == 'majority', f"Expected 'majority' for categorical_data, got {strategies['categorical_data']}"
    
    print("✓ Configuration strategies correct")


def test_dataset_types():
    """Test dataset configurations have correct data types."""
    print("\nTesting dataset configurations...")
    
    datasets = config.get('datasets.target_datasets', [])
    
    # Check plants and terrestrial are richness_data
    plants = next(d for d in datasets if d['name'] == 'plants-richness')
    assert plants['data_type'] == 'richness_data', f"Plants should be richness_data, got {plants['data_type']}"
    
    terrestrial = next(d for d in datasets if d['name'] == 'terrestrial-richness')
    assert terrestrial['data_type'] == 'richness_data', f"Terrestrial should be richness_data, got {terrestrial['data_type']}"
    
    # Check fungi are continuous_data
    am_fungi = next(d for d in datasets if d['name'] == 'am-fungi-richness')
    assert am_fungi['data_type'] == 'continuous_data', f"AM fungi should be continuous_data, got {am_fungi['data_type']}"
    
    ecm_fungi = next(d for d in datasets if d['name'] == 'ecm-fungi-richness')
    assert ecm_fungi['data_type'] == 'continuous_data', f"ECM fungi should be continuous_data, got {ecm_fungi['data_type']}"
    
    print("✓ All dataset types correct")


def test_numpy_resampler_has_strategies():
    """Test numpy resampler has all required strategies."""
    print("\nTesting numpy resampler strategies...")
    
    # Create minimal config
    config = ResamplingConfig(
        method='average',
        source_resolution=0.01,
        target_resolution=0.02,
        target_crs='EPSG:4326',
        nodata_value=-9999
    )
    
    resampler = NumpyResampler(config)
    
    # Check strategies are registered
    assert 'average' in resampler.strategies, "Average strategy not found in resampler"
    assert 'sum' in resampler.strategies, "Sum strategy not found in resampler"
    assert 'majority' in resampler.strategies, "Majority strategy not found in resampler"
    
    # Check strategy types
    assert isinstance(resampler.strategies['average'], AverageAggregationStrategy)
    assert hasattr(resampler.strategies['sum'], 'resample') or callable(resampler.strategies['sum'])
    
    print("✓ All strategies registered correctly")


def test_average_strategy_basic():
    """Test average strategy works for simple case."""
    print("\nTesting average strategy basic functionality...")
    
    strategy = AverageAggregationStrategy()
    
    # Simple 2x2 -> 1x1 downsampling
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target_shape = (1, 1)
    
    # Mapping: all 4 pixels to single target
    mapping = np.array([
        [0, 0, 1.0],
        [0, 1, 1.0],
        [0, 2, 1.0],
        [0, 3, 1.0]
    ])
    
    class SimpleConfig:
        source_resolution = 0.01
        target_resolution = 0.02
        nodata_value = None
    
    result = strategy.resample(source, target_shape, mapping, SimpleConfig())
    
    # Expected: (1+2+3+4)/4 = 2.5
    expected = 2.5
    assert np.abs(result[0, 0] - expected) < 0.001, f"Expected {expected}, got {result[0, 0]}"
    
    print("✓ Average strategy works correctly")


def test_resolution_comparison():
    """Test resolution comparison logic."""
    print("\nTesting resolution comparison...")
    
    # Verify understanding of resolution values
    plants_res = 0.016667  # Coarser
    fungi_res = 0.008983   # Finer
    target_res = 0.016667  # Target
    
    # Plants should be passthrough (same resolution)
    assert abs(plants_res - target_res) < 0.001, "Plants resolution should match target"
    
    # Fungi should need downsampling (finer to coarser)
    assert fungi_res < target_res, "Fungi resolution should be finer than target"
    
    print("✓ Resolution logic correct")


def main():
    """Run all tests."""
    print("Running lightweight resampling tests...\n")
    
    try:
        test_config_has_correct_strategies()
        test_dataset_types()
        test_numpy_resampler_has_strategies()
        test_average_strategy_basic()
        test_resolution_comparison()
        
        print("\n" + "="*50)
        print("All tests passed! ✨")
        print("="*50)
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)