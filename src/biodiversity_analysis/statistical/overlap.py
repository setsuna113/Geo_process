from src.abstractions import get_logger
"""
Overlap analysis with memory-efficient chunked processing - Linus style.

"Don't load what you don't need."
ONE implementation, handles both in-memory and chunked.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any
import rasterio
from scipy import stats

from src.biodiversity_analysis.statistical.data_utils import save_results


def analyze_overlap(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze spatial overlap - in-memory arrays ONLY.
    
    Data loading is orchestrator's job, not ours.
    
    Args:
        data: Dict[str, ndarray] - spatial arrays
        output_dir: Output directory
        **kwargs: Config including:
            - hotspot_percentile: Threshold for hotspots (default: 90)
            - plot: Generate visualizations (default: True)
        
    Returns:
        Dict[str, Any] with overlap statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract config with defaults
    hotspot_percentile = kwargs.get('hotspot_percentile', 90)
    plot = kwargs.get('plot', True)
    chunk_size = kwargs.get('chunk_size', 1000)  # Default chunk size for file processing
    
    try:
        if len(data) < 2:
            raise ValueError("Need at least 2 datasets")
        
        # Determine data type from first item
        first_value = next(iter(data.values()))
        
        if isinstance(first_value, np.ndarray):
            # In-memory processing
            results = _analyze_arrays(data, hotspot_percentile)
        elif isinstance(first_value, str):
            # File-based chunked processing
            results = _analyze_chunked(data, hotspot_percentile, chunk_size)
        else:
            raise TypeError(f"Unsupported data type: {type(first_value)}")
        
        # Generate RGB overlap map ONLY - no other visualizations needed
        if plot:
            # Create the RGB overlap map (aboveground vs belowground)
            rgb_output = output_dir / 'rgb_biodiversity_overlap.png'
            # create_rgb_overlap_map(data, rgb_output)
        
        # Save results
        save_results(results, output_dir / 'overlap_results.json')
        
        # Generate text report
        _write_report(results, output_dir / 'overlap_report.txt')
        
        return Dict[str, Any](
            success=True,
            method='overlap',
            output_dir=output_dir,
            metrics=results
        )
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        import logging
        logger = get_logger(__name__)
        logger.error(f"Overlap analysis failed: {error_msg}")
        return Dict[str, Any](
            success=False,
            method='overlap',
            error=str(e)
        )


def _analyze_arrays(data: Dict[str, np.ndarray], hotspot_percentile: float) -> Dict[str, Any]:
    """Analyze overlap for in-memory arrays."""
    
    # Get valid masks
    masks = {name: ~np.isnan(arr) for name, arr in data.items()}
    
    # 1. Coverage stats
    coverage = {}
    total_pixels = next(iter(data.values())).size
    
    for name, mask in masks.items():
        valid_pixels = mask.sum()
        coverage[name] = {
            'valid': int(valid_pixels),
            'percent': float(100 * valid_pixels / total_pixels),
            'mean': float(np.nanmean(data[name])),
            'max': float(np.nanmax(data[name]))
        }
    
    # All overlap
    all_valid = np.ones_like(next(iter(masks.values())), dtype=bool)
    for mask in masks.values():
        all_valid &= mask
    coverage['all_overlap'] = {
        'pixels': int(all_valid.sum()),
        'percent': float(100 * all_valid.sum() / total_pixels)
    }
    
    # 2. Pairwise overlap
    pairwise = {}
    names = list(data.keys())
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            mask1, mask2 = masks[name1], masks[name2]
            intersection = (mask1 & mask2).sum()
            union = (mask1 | mask2).sum()
            
            key = f"{name1}_vs_{name2}"
            pairwise[key] = {
                'jaccard': float(intersection / union) if union > 0 else 0,
                'overlap_pixels': int(intersection),
                'overlap_percent': float(100 * intersection / total_pixels)
            }
            
            # Correlation where both valid
            if intersection > 100:
                common = mask1 & mask2
                vals1 = data[name1][common]
                vals2 = data[name2][common]
                r, p = stats.pearsonr(vals1, vals2)
                pairwise[key]['correlation'] = float(r)
                pairwise[key]['p_value'] = float(p)
    
    # 3. Hotspots
    hotspots = _compute_hotspots(data, masks, hotspot_percentile)
    
    return {
        'n_datasets': len(data),
        'datasets': list(data.keys()),
        'coverage': coverage,
        'pairwise': pairwise,
        'hotspots': hotspots
    }


def _analyze_chunked(
    data_paths: Dict[str, str], 
    hotspot_percentile: float,
    chunk_size: int
) -> Dict[str, Any]:
    """Analyze overlap using chunked file reading."""
    
    # Open rasters without loading
    rasters = {}
    shapes = []
    
    for name, path in data_paths.items():
        if path and isinstance(path, str) and Path(path).exists():
            src = rasterio.open(path)
            rasters[name] = src
            shapes.append(src.shape)
    
    if not rasters:
        raise ValueError("No valid raster files found")
    
    # Verify same shape
    if len(set(shapes)) > 1:
        # Close files before raising
        for src in rasters.values():
            src.close()
        raise ValueError(f"Rasters have different shapes: {shapes}")
    
    height, width = shapes[0]
    total_pixels = height * width
    
    try:
        # Initialize statistics
        stats_tracker = _ChunkedStatsTracker(rasters.keys(), total_pixels)
        
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
            
            # Update statistics
            stats_tracker.update(chunks, masks)
        
        # Compute final statistics
        results = stats_tracker.finalize(hotspot_percentile)
        
    finally:
        # Always close rasters
        for src in rasters.values():
            src.close()
    
    return results


class _ChunkedStatsTracker:
    """Track statistics during chunked processing."""
    
    def __init__(self, names, total_pixels):
        self.names = list(names)
        self.total_pixels = total_pixels
        
        # Coverage tracking
        self.valid_counts = {name: 0 for name in names}
        self.value_sums = {name: 0.0 for name in names}
        self.value_maxs = {name: float('-inf') for name in names}
        self.all_valid_count = 0
        
        # Pairwise tracking
        self.pairwise_valid = {}
        self.pairwise_sums1 = {}
        self.pairwise_sums2 = {}
        self.pairwise_sums12 = {}
        
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                key = f"{name1}_vs_{name2}"
                self.pairwise_valid[key] = 0
                self.pairwise_sums1[key] = 0.0
                self.pairwise_sums2[key] = 0.0
                self.pairwise_sums12[key] = 0.0
        
        # Hotspot tracking - store all values for percentile calculation
        self.all_values = {name: [] for name in names}
    
    def update(self, chunks, masks):
        """Update statistics with new chunk."""
        
        # Coverage stats
        for name in self.names:
            valid = masks[name]
            if valid.sum() > 0:
                self.valid_counts[name] += valid.sum()
                self.value_sums[name] += chunks[name][valid].sum()
                self.value_maxs[name] = max(self.value_maxs[name], chunks[name][valid].max())
                # Store values for hotspot calculation
                self.all_values[name].extend(chunks[name][valid].tolist())
        
        # All valid overlap
        all_valid = np.ones_like(next(iter(masks.values())), dtype=bool)
        for mask in masks.values():
            all_valid &= mask
        self.all_valid_count += all_valid.sum()
        
        # Pairwise overlap
        for i, name1 in enumerate(self.names):
            for name2 in self.names[i+1:]:
                key = f"{name1}_vs_{name2}"
                common = masks[name1] & masks[name2]
                if common.sum() > 0:
                    self.pairwise_valid[key] += common.sum()
                    vals1 = chunks[name1][common]
                    vals2 = chunks[name2][common]
                    self.pairwise_sums1[key] += vals1.sum()
                    self.pairwise_sums2[key] += vals2.sum()
                    self.pairwise_sums12[key] += (vals1 * vals2).sum()
    
    def finalize(self, hotspot_percentile):
        """Compute final statistics."""
        
        # Coverage
        coverage = {}
        for name in self.names:
            valid_pixels = self.valid_counts[name]
            coverage[name] = {
                'valid': int(valid_pixels),
                'percent': float(100 * valid_pixels / self.total_pixels),
                'mean': float(self.value_sums[name] / valid_pixels) if valid_pixels > 0 else 0,
                'max': float(self.value_maxs[name]) if self.value_maxs[name] != float('-inf') else 0
            }
        
        coverage['all_overlap'] = {
            'pixels': int(self.all_valid_count),
            'percent': float(100 * self.all_valid_count / self.total_pixels)
        }
        
        # Pairwise
        pairwise = {}
        for i, name1 in enumerate(self.names):
            for name2 in self.names[i+1:]:
                key = f"{name1}_vs_{name2}"
                intersection = self.pairwise_valid[key]
                
                # Jaccard index (standard formula: |A∩B| / |A∪B|)
                union = self.valid_counts[name1] + self.valid_counts[name2] - intersection
                
                pairwise[key] = {
                    'jaccard': float(intersection / union) if union > 0 else 0,
                    'overlap_pixels': int(intersection),
                    'overlap_percent': float(100 * intersection / self.total_pixels)
                }
                
                # Correlation
                if intersection > 100:
                    mean1 = self.pairwise_sums1[key] / intersection
                    mean2 = self.pairwise_sums2[key] / intersection
                    # Simplified correlation (not exact but good enough)
                    cov = (self.pairwise_sums12[key] / intersection) - (mean1 * mean2)
                    # We'd need variance tracking for proper correlation
                    # For now, just mark as computed
                    pairwise[key]['correlation_available'] = False
        
        # Hotspots
        hotspots = {}
        for name in self.names:
            if self.all_values[name]:
                values = np.array(self.all_values[name])
                threshold = np.percentile(values, hotspot_percentile)
                hotspot_count = (values >= threshold).sum()
                hotspots[name] = {
                    'threshold': float(threshold),
                    'pixels': int(hotspot_count),
                    'percent': float(100 * hotspot_count / len(values))
                }
        
        return {
            'n_datasets': len(self.names),
            'datasets': self.names,
            'coverage': coverage,
            'pairwise': pairwise,
            'hotspots': hotspots
        }


def _compute_hotspots(data: Dict[str, np.ndarray], masks: Dict, percentile: float) -> Dict:
    """Compute hotspot statistics."""
    hotspots = {}
    names = list(data.keys())
    
    # Individual hotspots
    for name, arr in data.items():
        valid = masks[name]
        if valid.sum() > 0:
            threshold = np.nanpercentile(arr, percentile)
            hotspot_mask = arr >= threshold
            hotspots[name] = {
                'threshold': float(threshold),
                'pixels': int(hotspot_mask.sum()),
                'percent': float(100 * hotspot_mask.sum() / valid.sum())
            }
    
    # Hotspot overlap
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            if name1 in hotspots and name2 in hotspots:
                hot1 = data[name1] >= hotspots[name1]['threshold']
                hot2 = data[name2] >= hotspots[name2]['threshold']
                overlap = (hot1 & hot2).sum()
                key = f"{name1}_vs_{name2}_hotspots"
                hotspots[key] = {
                    'overlap_pixels': int(overlap),
                    'percent_of_hot1': float(100 * overlap / hot1.sum()) if hot1.sum() > 0 else 0
                }
    
    return hotspots


# REMOVED: Old visualization function - replaced with RGB overlap map
# def _generate_plots(...) - see overlap_rgb.py for new implementation


def _write_report(results: Dict, filepath: Path):
    """Write text report of overlap analysis."""
    with open(filepath, 'w') as f:
        f.write("OVERLAP ANALYSIS\n")
        f.write("================\n\n")
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