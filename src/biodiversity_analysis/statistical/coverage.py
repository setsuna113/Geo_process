"""
Coverage analysis - Linus style.

262行 -> 80行
"Complexity is the root of all evil"
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

from src.biodiversity_analysis.statistical.data_utils import save_results
from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader


def analyze_coverage(
    data_source: Dict[str, str],
    output_dir: Union[str, Path],
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze data coverage across biodiversity datasets.
    
    Args:
        data_source: Dict of dataset names to raster paths
        output_dir: Output directory
        
    Returns:
        Coverage statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load rasters
    raster_loader = RasterLoader()
    results = {'datasets': {}}
    
    for name, path in data_source.items():
        if path is None:
            continue
            
        # Load and compute stats
        data = raster_loader.load_single(path)
        valid_mask = ~np.isnan(data)
        valid_count = valid_mask.sum()
        total_count = data.size
        
        # Basic statistics
        stats = {
            'path': str(path),
            'shape': data.shape,
            'total_pixels': int(total_count),
            'valid_pixels': int(valid_count),
            'coverage_percent': float(100 * valid_count / total_count),
            'missing_percent': float(100 * (total_count - valid_count) / total_count)
        }
        
        # Value statistics for valid data
        if valid_count > 0:
            valid_data = data[valid_mask]
            stats.update({
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'median': float(np.median(valid_data)),
                'q25': float(np.percentile(valid_data, 25)),
                'q75': float(np.percentile(valid_data, 75))
            })
        
        results['datasets'][name] = stats
    
    # Summary
    coverages = [d['coverage_percent'] for d in results['datasets'].values()]
    results['summary'] = {
        'n_datasets': len(results['datasets']),
        'mean_coverage': float(np.mean(coverages)) if coverages else 0,
        'min_coverage': float(np.min(coverages)) if coverages else 0,
        'max_coverage': float(np.max(coverages)) if coverages else 0
    }
    
    # Save
    save_results(results, output_dir / 'coverage_results.json')
    
    # Simple text report
    with open(output_dir / 'coverage_report.txt', 'w') as f:
        f.write("COVERAGE ANALYSIS\n")
        f.write("=================\n\n")
        for name, stats in results['datasets'].items():
            f.write(f"{name}:\n")
            f.write(f"  Coverage: {stats['coverage_percent']:.1f}%\n")
            if 'mean' in stats:
                f.write(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]\n")
                f.write(f"  Mean: {stats['mean']:.1f} ± {stats['std']:.1f}\n")
            f.write("\n")
    
    return results