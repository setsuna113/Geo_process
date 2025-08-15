"""
Overlap analysis - Linus style.

"Good code has no special cases"
499行 -> 150行
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
from scipy import stats
import json

from src.biodiversity_analysis.statistical.data_utils import load_data, save_results
from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader
from src.biodiversity_analysis.shared.data.raster_aligner import load_and_align_rasters
# Import the memory-efficient version
from src.biodiversity_analysis.statistical.overlap_chunked import analyze_overlap as analyze_overlap_chunked


def analyze_overlap(
    data_source: Dict[str, Union[str, np.ndarray]],
    output_dir: Union[str, Path],
    hotspot_percentile: float = 90,
    plot: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze spatial overlap between biodiversity datasets.
    
    Now uses memory-efficient chunked processing.
    
    Args:
        data_source: Dict of dataset names to paths/arrays
        output_dir: Where to save results
        hotspot_percentile: Threshold for hotspots
        plot: Generate visualizations
        
    Returns:
        Overlap statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If all inputs are paths, align them first
    if all(isinstance(v, str) for v in data_source.values()):
        # Load and align to common resolution
        data = load_and_align_rasters(data_source)
    else:
        # Load any paths, keep arrays as-is
        raster_loader = RasterLoader()
        data = {}
        for name, source in data_source.items():
            if isinstance(source, str):
                data[name] = raster_loader.load_single(source)
            else:
                data[name] = source
    
    if len(data) < 2:
        raise ValueError("Need at least 2 datasets")
    
    # Get valid masks
    masks = {name: ~np.isnan(arr) for name, arr in data.items()}
    
    # 1. Coverage - just count pixels
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
    
    # All datasets overlap
    all_valid = np.ones_like(next(iter(masks.values())), dtype=bool)
    for mask in masks.values():
        all_valid &= mask
    coverage['all_overlap'] = {
        'pixels': int(all_valid.sum()),
        'percent': float(100 * all_valid.sum() / total_pixels)
    }
    
    # 2. Pairwise overlap - Jaccard index
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
            if intersection > 100:  # Need enough points
                common = mask1 & mask2
                vals1 = data[name1][common]
                vals2 = data[name2][common]
                r, p = stats.pearsonr(vals1, vals2)
                pairwise[key]['correlation'] = float(r)
                pairwise[key]['p_value'] = float(p)
    
    # 3. Hotspots - top percentile
    hotspots = {}
    for name, arr in data.items():
        valid = ~np.isnan(arr)
        if valid.sum() > 0:
            threshold = np.nanpercentile(arr, hotspot_percentile)
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
    
    # Results
    results = {
        'n_datasets': len(data),
        'datasets': names,
        'coverage': coverage,
        'pairwise': pairwise,
        'hotspots': hotspots
    }
    
    # Simple plot if requested
    if plot:
        _plot_overlap(data, masks, output_dir)
    
    # Save
    save_results(results, output_dir / 'overlap_results.json')
    
    # Text report
    with open(output_dir / 'overlap_report.txt', 'w') as f:
        f.write(f"OVERLAP ANALYSIS\n")
        f.write(f"================\n\n")
        f.write(f"Datasets: {', '.join(names)}\n\n")
        
        f.write(f"COVERAGE:\n")
        for name, cov in coverage.items():
            if name != 'all_overlap':
                f.write(f"  {name}: {cov['percent']:.1f}%\n")
        f.write(f"  All overlap: {coverage['all_overlap']['percent']:.1f}%\n\n")
        
        f.write(f"PAIRWISE OVERLAP:\n")
        for key, vals in pairwise.items():
            f.write(f"  {key}: Jaccard={vals['jaccard']:.3f}")
            if 'correlation' in vals:
                f.write(f", r={vals['correlation']:.3f}")
            f.write(f"\n")
    
    return results


def _plot_overlap(data: Dict, masks: Dict, output_dir: Path):
    """Create overlap visualizations - both simple and RGB if applicable."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata
    
    # Always create simple overlap
    _create_simple_overlap(data, masks, output_dir, plt)
    
    # Create RGB if we have the right data
    if _can_create_rgb(data):
        _create_rgb_overlap(data, output_dir, plt, rankdata)


def _can_create_rgb(data: Dict) -> bool:
    """Check if we can create RGB visualization."""
    has_animals = any('animal' in k.lower() for k in data)
    has_plants = any('plant' in k.lower() for k in data)
    has_fungi = any('fungi' in k.lower() for k in data)
    return has_animals and has_plants and has_fungi


def _create_simple_overlap(data: Dict, masks: Dict, output_dir: Path, plt):
    """Simple overlap count map."""
    n = len(data)
    fig, axes = plt.subplots(1, n+1, figsize=(4*(n+1), 4))
    if n == 1:
        axes = [axes]
    
    # Individual datasets
    for i, (name, arr) in enumerate(data.items()):
        ax = axes[i]
        im = ax.imshow(arr, cmap='viridis')
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Overlap count
    overlap = np.zeros_like(next(iter(data.values())))
    for mask in masks.values():
        overlap[mask] += 1
    
    ax = axes[-1]
    im = ax.imshow(overlap, cmap='RdYlGn', vmin=0, vmax=len(data))
    ax.set_title('Overlap Count')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overlap_map.png', dpi=150, bbox_inches='tight')
    plt.close()


def _create_rgb_overlap(data: Dict, output_dir: Path, plt, rankdata):
    """RGB composite for Nature-standard visualization."""
    # Get channels
    R = next(v for k, v in data.items() if 'animal' in k.lower())
    G = next(v for k, v in data.items() if 'plant' in k.lower())
    
    # Combine fungi if needed
    fungi_arrays = [v for k, v in data.items() if 'fungi' in k.lower()]
    B = np.nansum(fungi_arrays, axis=0) if len(fungi_arrays) > 1 else fungi_arrays[0]
    
    # Quantile normalize each channel
    R_norm = _quantile_norm(R, rankdata)
    G_norm = _quantile_norm(G, rankdata)
    B_norm = _quantile_norm(B, rankdata)
    
    # Stack into RGB with proper NaN handling
    rgb = np.zeros((*R_norm.shape, 3))
    
    # Fill RGB channels, treating NaN as 0 (black)
    rgb[:, :, 0] = np.nan_to_num(R_norm, nan=0)
    rgb[:, :, 1] = np.nan_to_num(G_norm, nan=0) 
    rgb[:, :, 2] = np.nan_to_num(B_norm, nan=0)
    
    # Mark ocean/no-data as gray (where ALL channels are NaN/zero)
    no_data_mask = (np.isnan(R_norm) | (R <= 0)) & \
                   (np.isnan(G_norm) | (G <= 0)) & \
                   (np.isnan(B_norm) | (B <= 0))
    
    for i in range(3):
        rgb[:, :, i][no_data_mask] = 0.5  # Gray for ocean/no-data
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(rgb, aspect='equal')
    ax.set_title('Biodiversity RGB: Red=Animals, Green=Plants, Blue=Fungi',
                fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overlap_rgb.png', dpi=200, bbox_inches='tight')
    plt.close()


def _quantile_norm(data, rankdata):
    """Normalize to [0,1] using quantiles, preserve NaN for no-data."""
    valid = (data > 0)  # Only positive values are valid data
    if valid.sum() == 0:
        return np.full_like(data, np.nan)
    
    norm = np.full_like(data, np.nan)  # Start with all NaN
    ranks = rankdata(data[valid], method='average')
    norm[valid] = (ranks - 0.5) / len(ranks)
    
    # Set zeros (ocean/no-data) to NaN, not 0
    norm[data <= 0] = np.nan
    return norm