"""
Global map visualization - Linus style.

284行 -> 90行
"Bad programmers worry about the code. Good programmers worry about data structures."
"""

import numpy as np
from pathlib import Path
from typing import Dict, Union, Any
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure
from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader


def analyze_global_maps(
    data_source: Dict[str, str],
    output_dir: Union[str, Path],
    **kwargs
) -> Dict[str, Any]:
    """
    Create global map visualizations.
    
    Args:
        data_source: Dict of dataset names to raster paths
        output_dir: Output directory
        
    Returns:
        Map statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load rasters
    raster_loader = RasterLoader()
    data = {}
    stats = {}
    
    for name, path in data_source.items():
        if path is None:
            continue
        
        # Load data
        arr = raster_loader.load_single(path)
        data[name] = arr
        
        # Compute stats
        valid = ~np.isnan(arr)
        if valid.sum() > 0:
            stats[name] = {
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
                'mean': float(np.nanmean(arr)),
                'coverage': float(100 * valid.sum() / arr.size)
            }
    
    if not data:
        return {'error': 'No data loaded'}
    
    # Create global map grid
    n = len(data)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    
    fig = plt.figure(figsize=(8*cols, 4*rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx, (name, arr) in enumerate(data.items()):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        
        # Plot map
        im = ax.imshow(arr, cmap='viridis', aspect='equal', 
                      extent=[-180, 180, -90, 90])
        
        # Simple formatting
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if stats[name]['max'] > 1000:
            cbar.formatter.set_powerlimits((0, 0))
        
    plt.suptitle('Global Biodiversity Overview', fontsize=14, y=1.02)
    save_figure(fig, output_dir / 'global_map_overview.png')
    plt.close(fig)
    
    # Save results
    results = {
        'datasets': list(data.keys()),
        'statistics': stats
    }
    save_results(results, output_dir / 'global_map_results.json')
    
    return results