"""
Overlap analysis with chunked processing - Linus style.

"Don't load what you don't need."
Actually efficient, not pretend efficient.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Tuple
import rasterio
from scipy import stats

from src.biodiversity_analysis.statistical.data_utils import save_results


def analyze_overlap(
    data_source: Dict[str, str],
    output_dir: Union[str, Path],
    hotspot_percentile: float = 90,
    chunk_size: int = 1000,
    plot: bool = True
) -> Dict[str, Any]:
    """
    Analyze spatial overlap WITHOUT loading everything to memory.
    
    Args:
        data_source: Dict of dataset names to raster paths
        output_dir: Output directory
        hotspot_percentile: Threshold for hotspots
        chunk_size: Rows to process at once
        plot: Generate visualizations
        
    Returns:
        Overlap statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(data_source) < 2:
        raise ValueError("Need at least 2 datasets")
    
    # Open rasters without loading
    rasters = {}
    shapes = []
    for name, path in data_source.items():
        if path and isinstance(path, str):
            src = rasterio.open(path)
            rasters[name] = src
            shapes.append(src.shape)
    
    if not rasters:
        return {'error': 'No valid rasters'}
    
    # Verify same shape
    if len(set(shapes)) > 1:
        raise ValueError(f"Rasters have different shapes: {shapes}")
    
    height, width = shapes[0]
    total_pixels = height * width
    
    # 1. Coverage analysis - chunked
    coverage = {}
    valid_counts = {name: 0 for name in rasters}
    value_sums = {name: 0.0 for name in rasters}
    value_maxs = {name: float('-inf') for name in rasters}
    
    # Track overlap counts
    all_valid_count = 0
    pairwise_valid = {}
    for i, name1 in enumerate(rasters.keys()):
        for name2 in list(rasters.keys())[i+1:]:
            pairwise_valid[f"{name1}_vs_{name2}"] = 0
    
    # Process in chunks
    for row_start in range(0, height, chunk_size):
        row_end = min(row_start + chunk_size, height)
        window = ((row_start, row_end), (0, width))
        
        # Read chunk from each raster
        chunks = {}
        masks = {}
        for name, src in rasters.items():
            chunk = src.read(1, window=window)
            chunks[name] = chunk
            masks[name] = ~np.isnan(chunk) & (chunk != src.nodata if src.nodata else True)
            
            # Update stats
            valid = masks[name]
            if valid.sum() > 0:
                valid_counts[name] += valid.sum()
                value_sums[name] += chunk[valid].sum()
                value_maxs[name] = max(value_maxs[name], chunk[valid].max())
        
        # Check all valid overlap
        all_valid = np.ones_like(next(iter(masks.values())), dtype=bool)
        for mask in masks.values():
            all_valid &= mask
        all_valid_count += all_valid.sum()
        
        # Pairwise overlaps
        names = list(rasters.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                overlap = masks[name1] & masks[name2]
                pairwise_valid[f"{name1}_vs_{name2}"] += overlap.sum()
    
    # Finalize coverage stats
    for name in rasters:
        coverage[name] = {
            'valid': int(valid_counts[name]),
            'percent': float(100 * valid_counts[name] / total_pixels),
            'mean': float(value_sums[name] / valid_counts[name]) if valid_counts[name] > 0 else 0,
            'max': float(value_maxs[name]) if value_maxs[name] > float('-inf') else 0
        }
    
    coverage['all_overlap'] = {
        'pixels': int(all_valid_count),
        'percent': float(100 * all_valid_count / total_pixels)
    }
    
    # 2. Pairwise statistics
    pairwise = {}
    names = list(rasters.keys())
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            key = f"{name1}_vs_{name2}"
            intersection = pairwise_valid[key]
            union = valid_counts[name1] + valid_counts[name2] - intersection
            
            pairwise[key] = {
                'jaccard': float(intersection / union) if union > 0 else 0,
                'overlap_pixels': int(intersection),
                'overlap_percent': float(100 * intersection / total_pixels)
            }
    
    # 3. Correlation (sample-based to avoid memory issues)
    _add_correlations_sampled(rasters, pairwise, sample_size=100000)
    
    # 4. Hotspot analysis (simplified)
    hotspots = _analyze_hotspots_chunked(rasters, hotspot_percentile, chunk_size)
    
    # Close rasters
    for src in rasters.values():
        src.close()
    
    # Results
    results = {
        'n_datasets': len(data_source),
        'datasets': list(data_source.keys()),
        'coverage': coverage,
        'pairwise': pairwise,
        'hotspots': hotspots
    }
    
    # Save
    save_results(results, output_dir / 'overlap_results.json')
    _write_report(results, output_dir / 'overlap_report.txt')
    
    return results


def _add_correlations_sampled(rasters: Dict, pairwise: Dict, sample_size: int):
    """Add correlation using spatial sampling."""
    # Calculate sampling step
    height = next(iter(rasters.values())).height
    width = next(iter(rasters.values())).width
    total_pixels = height * width
    
    step = max(1, int(np.sqrt(total_pixels / sample_size)))
    
    # Sample data
    samples = {name: [] for name in rasters}
    for row in range(0, height, step):
        for col in range(0, width, step):
            window = ((row, row+1), (col, col+1))
            for name, src in rasters.items():
                val = src.read(1, window=window)[0, 0]
                samples[name].append(val)
    
    # Calculate correlations
    samples = {name: np.array(vals) for name, vals in samples.items()}
    names = list(rasters.keys())
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            key = f"{name1}_vs_{name2}"
            
            # Remove NaN pairs
            valid = ~(np.isnan(samples[name1]) | np.isnan(samples[name2]))
            if valid.sum() > 100:
                r, p = stats.pearsonr(samples[name1][valid], samples[name2][valid])
                pairwise[key]['correlation'] = float(r)
                pairwise[key]['p_value'] = float(p)


def _analyze_hotspots_chunked(rasters: Dict, percentile: float, chunk_size: int) -> Dict:
    """Analyze hotspots with chunked processing."""
    # First pass: get thresholds
    thresholds = {}
    for name, src in rasters.items():
        values = []
        for row in range(0, src.height, chunk_size):
            window = ((row, min(row + chunk_size, src.height)), (0, src.width))
            chunk = src.read(1, window=window)
            valid = ~np.isnan(chunk) & (chunk != src.nodata if src.nodata else True)
            if valid.sum() > 0:
                # Sample for percentile calculation
                if len(values) < 100000:
                    values.extend(chunk[valid].tolist())
        
        if values:
            thresholds[name] = np.percentile(values, percentile)
    
    # Second pass: count hotspots
    hotspots = {}
    for name, threshold in thresholds.items():
        hotspot_count = 0
        src = rasters[name]
        
        for row in range(0, src.height, chunk_size):
            window = ((row, min(row + chunk_size, src.height)), (0, src.width))
            chunk = src.read(1, window=window)
            hotspot_count += (chunk >= threshold).sum()
        
        hotspots[name] = {
            'threshold': float(threshold),
            'pixels': int(hotspot_count),
            'percent': float(100 * hotspot_count / (src.height * src.width))
        }
    
    return hotspots


def _write_report(results: Dict, path: Path):
    """Write text report."""
    with open(path, 'w') as f:
        f.write("OVERLAP ANALYSIS (Memory Efficient)\n")
        f.write("====================================\n\n")
        f.write(f"Datasets: {', '.join(results['datasets'])}\n\n")
        
        f.write("COVERAGE:\n")
        for name, cov in results['coverage'].items():
            if name != 'all_overlap':
                f.write(f"  {name}: {cov['percent']:.1f}%\n")
        f.write(f"  All overlap: {results['coverage']['all_overlap']['percent']:.1f}%\n\n")
        
        f.write("PAIRWISE OVERLAP:\n")
        for key, vals in results['pairwise'].items():
            f.write(f"  {key}: Jaccard={vals['jaccard']:.3f}")
            if 'correlation' in vals:
                f.write(f", r={vals['correlation']:.3f}")
            f.write("\n")