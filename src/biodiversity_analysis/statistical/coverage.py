"""
Coverage analysis with memory efficiency - Linus style.

"Don't load what you don't need."
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import rasterio

from src.biodiversity_analysis.statistical.data_utils import save_results


def analyze_coverage(
    data_source: Dict[str, str],
    output_dir: Union[str, Path],
    chunk_size: int = 1000,  # Read 1000 rows at a time
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze data coverage with minimal memory usage.
    
    Args:
        data_source: Dict of dataset names to raster paths
        output_dir: Output directory
        chunk_size: Rows to read at once
        
    Returns:
        Coverage statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'datasets': {}}
    
    for name, path in data_source.items():
        if path is None or not isinstance(path, str):
            continue
        
        # Use rasterio for efficient chunked reading
        with rasterio.open(path) as src:
            # Get metadata without loading data
            stats = {
                'path': str(path),
                'shape': src.shape,
                'total_pixels': int(src.height * src.width),
            }
            
            # Accumulate statistics in chunks
            valid_count = 0
            value_sum = 0.0
            value_sum2 = 0.0  # For std calculation
            value_min = float('inf')
            value_max = float('-inf')
            all_values = []  # Only for percentiles, sample if too big
            
            # Process in chunks
            for row_start in range(0, src.height, chunk_size):
                row_end = min(row_start + chunk_size, src.height)
                
                # Read chunk
                chunk = src.read(1, window=((row_start, row_end), (0, src.width)))
                
                # Handle nodata
                if src.nodata is not None:
                    valid_mask = (chunk != src.nodata) & ~np.isnan(chunk)
                else:
                    valid_mask = ~np.isnan(chunk)
                
                valid_chunk = chunk[valid_mask]
                
                if len(valid_chunk) > 0:
                    # Update counters
                    valid_count += len(valid_chunk)
                    value_sum += valid_chunk.sum()
                    value_sum2 += (valid_chunk ** 2).sum()
                    value_min = min(value_min, valid_chunk.min())
                    value_max = max(value_max, valid_chunk.max())
                    
                    # Sample for percentiles (don't keep everything)
                    if len(all_values) < 100000:  # Keep max 100k samples
                        all_values.extend(valid_chunk.tolist())
                    elif np.random.random() < 0.1:  # Random 10% after that
                        sample_size = min(1000, len(valid_chunk))
                        indices = np.random.choice(len(valid_chunk), sample_size, replace=False)
                        all_values.extend(valid_chunk[indices].tolist())
            
            # Calculate final statistics
            stats['valid_pixels'] = int(valid_count)
            stats['coverage_percent'] = float(100 * valid_count / stats['total_pixels'])
            stats['missing_percent'] = float(100 - stats['coverage_percent'])
            
            if valid_count > 0:
                mean = value_sum / valid_count
                variance = (value_sum2 / valid_count) - mean ** 2
                std = np.sqrt(max(0, variance))  # Avoid numerical issues
                
                stats.update({
                    'min': float(value_min),
                    'max': float(value_max),
                    'mean': float(mean),
                    'std': float(std),
                })
                
                # Percentiles from sample
                if all_values:
                    all_values = np.array(all_values[:100000])  # Cap at 100k
                    stats.update({
                        'median': float(np.median(all_values)),
                        'q25': float(np.percentile(all_values, 25)),
                        'q75': float(np.percentile(all_values, 75))
                    })
            
            results['datasets'][name] = stats
    
    # Summary
    if results['datasets']:
        coverages = [d['coverage_percent'] for d in results['datasets'].values()]
        results['summary'] = {
            'n_datasets': len(results['datasets']),
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