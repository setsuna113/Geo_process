from src.abstractions import get_logger
"""
Coverage analysis - Linus style.

"Simple. Clean. No bullshit."
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import logging

from src.biodiversity_analysis.statistical.data_utils import save_results
from src.biodiversity_analysis.statistical.result_types import MethodResult

logger = get_logger(__name__)


def analyze_coverage(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    **kwargs  # Ignore extra config
) -> MethodResult:
    """
    Analyze data coverage - SIMPLE version.
    
    Data already loaded by orchestrator. Just analyze it.
    
    Args:
        data: Dict of arrays (already loaded)
        output_dir: Output directory
        
    Returns:
        MethodResult with coverage statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'datasets': {}}
    raster_loader = RasterLoader()
    
    for name, array in data.items():
        if array is None:
            continue
            
        # Simple statistics
        valid_mask = ~np.isnan(array)
        valid_pixels = valid_mask.sum()
        total_pixels = array.size
        
        stats = {
            'shape': array.shape,
            'total_pixels': int(total_pixels),
            'valid_pixels': int(valid_pixels),
            'coverage_percent': float(100 * valid_pixels / total_pixels),
            'min': float(np.nanmin(array)) if valid_pixels > 0 else None,
            'max': float(np.nanmax(array)) if valid_pixels > 0 else None,
            'mean': float(np.nanmean(array)) if valid_pixels > 0 else None,
            'std': float(np.nanstd(array)) if valid_pixels > 0 else None
        }
        
        results['datasets'][name] = stats
        
        # Record: {name}: {valid_pixels:,}/{total_pixels:,} pixels ({coverage_percent}%)
    
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
    
    return MethodResult(
        success=True,
        method='coverage',
        output_dir=output_dir,
        metrics=results
    )