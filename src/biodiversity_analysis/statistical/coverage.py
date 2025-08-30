from src.abstractions import get_logger
"""
Coverage analysis - Linus style.

"Simple. Clean. No bullshit."
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import logging
import rasterio

from src.biodiversity_analysis.statistical.data_utils import save_results

logger = get_logger(__name__)


def analyze_coverage(
    data: Dict[str, Any],
    output_dir: Path,
    **kwargs  # Ignore extra config
) -> Dict[str, Any]:
    """
    Analyze data coverage from multiband raster.
    
    LINUS: "No special cases. Everything is multiband now."
    
    Args:
        data: Dict with 'multiband' (rasterio DatasetReader) and 'band_names' (list)
        output_dir: Output directory
        
    Returns:
        Dict[str, Any] with coverage statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract multiband raster and band names - NO SPECIAL CASES
    multiband = data['multiband']
    band_names = data['band_names']
    
    results = {'datasets': {}}
    
    # Process each band
    for band_idx, band_name in enumerate(band_names, 1):
        logger.info(f"Processing band {band_idx}: {band_name}")
        stats = _compute_coverage_stats_band(multiband, band_idx, band_name)
        results['datasets'][band_name] = stats
    
    # Summary statistics
    n_datasets = len(results['datasets'])
    if n_datasets > 0:
        coverages = [d['coverage_percent'] for d in results['datasets'].values()]
        results['summary'] = {
            'n_datasets': n_datasets,
            'mean_coverage': float(np.mean(coverages)),
            'min_coverage': float(np.min(coverages)),
            'max_coverage': float(np.max(coverages))
        }
    else:
        results['summary'] = {
            'n_datasets': 0,
            'mean_coverage': 0,
            'min_coverage': 0,
            'max_coverage': 0
        }
    
    # Save results
    save_results(results, output_dir / 'coverage_results.json')
    
    return Dict[str, Any](
        success=True,
        method='coverage',
        output_dir=output_dir,
        metrics=results
    )


def _compute_coverage_stats_band(src: rasterio.DatasetReader, band_idx: int, band_name: str) -> Dict[str, Any]:
    """Compute coverage statistics for a specific band.
    
    LINUS: "One function, one purpose. Process one band."
    
    Args:
        src: Multiband raster dataset
        band_idx: Band index (1-based)
        band_name: Name of the band
        
    Returns:
        Statistics for this band
    """
    stats = {
        'sum': 0,
        'sum_sq': 0,
        'valid_count': 0,
        'total_count': 0,
        'min': float('inf'),
        'max': float('-inf')
    }
    
    # Process in blocks - memory efficient
    for ji, window in src.block_windows(band_idx):
        data = src.read(band_idx, window=window, masked=True)
        
        # Get valid (non-masked) data
        valid = data.compressed()
        
        if len(valid) > 0:
            stats['sum'] += valid.sum()
            stats['sum_sq'] += (valid ** 2).sum()
            stats['valid_count'] += len(valid)
            stats['min'] = min(stats['min'], float(valid.min()))
            stats['max'] = max(stats['max'], float(valid.max()))
        
        stats['total_count'] += data.size
    
    # Compute final statistics
    if stats['valid_count'] > 0:
        mean = stats['sum'] / stats['valid_count']
        variance = (stats['sum_sq'] / stats['valid_count']) - mean ** 2
        std = np.sqrt(variance) if variance > 0 else 0
        
        return {
            'band_name': band_name,
            'band_index': band_idx,
            'shape': (src.height, src.width),
            'total_pixels': int(stats['total_count']),
            'valid_pixels': int(stats['valid_count']),
            'coverage_percent': float(100 * stats['valid_count'] / stats['total_count']),
            'min': float(stats['min']),
            'max': float(stats['max']),
            'mean': float(mean),
            'std': float(std),
            'bounds': src.bounds._asdict() if hasattr(src.bounds, '_asdict') else dict(src.bounds),
            'crs': str(src.crs) if src.crs else None
        }
    else:
        return {
            'band_name': band_name,
            'band_index': band_idx,
            'shape': (src.height, src.width),
            'total_pixels': int(stats['total_count']),
            'valid_pixels': 0,
            'coverage_percent': 0.0,
            'error': 'No valid data'
        }