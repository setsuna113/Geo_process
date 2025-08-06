#!/usr/bin/env python
"""
Benchmark script to compare MESF performance: original vs optimized.
"""

import numpy as np
import time
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/yl998/dev/geo')

from src.machine_learning.spatial import (
    compute_mesf, 
    compute_mesf_optimized
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_coordinates(n_samples):
    """Generate random lat/lon coordinates for testing."""
    np.random.seed(42)
    lon = np.random.uniform(-180, 180, n_samples)
    lat = np.random.uniform(-60, 60, n_samples)
    return np.column_stack([lon, lat])


def benchmark_mesf(coords, config, use_optimized=False):
    """Benchmark MESF computation."""
    func = compute_mesf_optimized if use_optimized else compute_mesf
    name = "Optimized" if use_optimized else "Original"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking {name} MESF with {len(coords):,} samples")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = func(coords, config)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ {name} completed in {elapsed:.2f} seconds")
        logger.info(f"   Generated {result.shape[1]} eigenvectors")
        return elapsed, result
    except Exception as e:
        logger.error(f"❌ {name} failed: {e}")
        return None, None


def main():
    """Run performance comparison."""
    
    # Configuration matching ml_config.yml
    config = {
        'mesf': {
            'variance_threshold': 0.75,
            'max_eigenvectors': 10,
            'min_eigenvectors': 5,
            'distance_threshold': 100  # km
        },
        'nystrom': {
            'enable_for_large_data': True,
            'size_threshold': 100000,
            'n_landmark': 5000,
            'n_sample': 25000
        }
    }
    
    # Test with different dataset sizes
    test_sizes = [100, 500, 1000, 5000]
    
    results = []
    
    for n_samples in test_sizes:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Testing with {n_samples:,} samples")
        logger.info(f"{'#'*70}")
        
        # Generate test data
        coords = generate_test_coordinates(n_samples)
        
        # Test original implementation
        time_original, result_original = benchmark_mesf(coords, config, use_optimized=False)
        
        # Test optimized implementation
        time_optimized, result_optimized = benchmark_mesf(coords, config, use_optimized=True)
        
        # Compare results
        if time_original is not None and time_optimized is not None:
            speedup = time_original / time_optimized
            logger.info(f"\n📊 Performance Summary for {n_samples:,} samples:")
            logger.info(f"   Original: {time_original:.3f}s")
            logger.info(f"   Optimized: {time_optimized:.3f}s")
            logger.info(f"   Speedup: {speedup:.2f}x")
            
            # Check if results are similar
            if result_original is not None and result_optimized is not None:
                if result_original.shape == result_optimized.shape:
                    if result_original.shape[1] > 0:  # Only compare if we have eigenvectors
                        # Eigenvectors can have different signs, so check absolute values
                        max_diff = np.max(np.abs(np.abs(result_original.values) - np.abs(result_optimized.values)))
                        logger.info(f"   Max difference in eigenvectors: {max_diff:.6f}")
                        if max_diff < 0.01:
                            logger.info(f"   ✅ Results are numerically consistent")
                        else:
                            logger.warning(f"   ⚠️ Results differ significantly")
                    else:
                        logger.info(f"   ✅ Both returned 0 eigenvectors (sparse data)")
                else:
                    logger.warning(f"   ⚠️ Different number of eigenvectors: {result_original.shape[1]} vs {result_optimized.shape[1]}")
            
            results.append({
                'n_samples': n_samples,
                'time_original': time_original,
                'time_optimized': time_optimized,
                'speedup': speedup
            })
    
    # Print overall summary
    if results:
        logger.info(f"\n{'='*70}")
        logger.info("OVERALL PERFORMANCE SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"{'Samples':<10} {'Original':<12} {'Optimized':<12} {'Speedup':<10}")
        logger.info(f"{'-'*46}")
        for r in results:
            logger.info(f"{r['n_samples']:<10} {r['time_original']:<12.3f} {r['time_optimized']:<12.3f} {r['speedup']:<10.2f}x")
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        logger.info(f"\nAverage speedup: {avg_speedup:.2f}x")
    
    # Test with a large dataset if memory allows
    logger.info(f"\n{'#'*70}")
    logger.info("# Testing with larger dataset (Nystrom approximation)")
    logger.info(f"{'#'*70}")
    
    # This will trigger Nystrom approximation
    large_n = 150000
    logger.info(f"\nGenerating {large_n:,} samples (will use Nystrom)...")
    coords_large = generate_test_coordinates(large_n)
    
    # Only test optimized version for large dataset
    logger.info("\nTesting optimized Nystrom implementation...")
    time_nystrom, result_nystrom = benchmark_mesf(coords_large, config, use_optimized=True)
    
    if time_nystrom is not None:
        logger.info(f"\n✅ Nystrom approximation completed in {time_nystrom:.2f}s for {large_n:,} samples")
        logger.info(f"   Generated {result_nystrom.shape[1]} eigenvectors")


if __name__ == "__main__":
    main()