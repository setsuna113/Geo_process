"""
Latitudinal gradient analysis - Linus style.

287行 -> 100行
"If you need more than 3 levels of indentation, you're screwed"
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import matplotlib.pyplot as plt

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure
from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader


def analyze_latitudinal(
    data_source: Dict[str, str],
    output_dir: Union[str, Path],
    lat_bins: int = 36,  # 5-degree bands
    plot: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze latitudinal gradients in biodiversity.
    
    Args:
        data_source: Dict of dataset names to raster paths
        output_dir: Output directory
        lat_bins: Number of latitude bins
        plot: Generate plots
        
    Returns:
        Latitudinal gradient statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load rasters
    raster_loader = RasterLoader()
    results = {'datasets': {}, 'lat_bins': lat_bins}
    
    # Assume global extent: -90 to 90 latitude
    lat_edges = np.linspace(-90, 90, lat_bins + 1)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    results['latitude_centers'] = lat_centers.tolist()
    
    for name, path in data_source.items():
        if path is None:
            continue
        
        # Load data
        data = raster_loader.load_single(path)
        height, width = data.shape
        
        # Create latitude array (assuming global extent)
        lats = np.linspace(90, -90, height)
        lat_array = np.repeat(lats[:, np.newaxis], width, axis=1)
        
        # Bin by latitude
        valid_mask = ~np.isnan(data)
        lat_means = []
        lat_stds = []
        lat_counts = []
        
        for i in range(lat_bins):
            lat_min, lat_max = lat_edges[i], lat_edges[i+1]
            bin_mask = (lat_array >= lat_min) & (lat_array < lat_max) & valid_mask
            
            if bin_mask.sum() > 0:
                lat_means.append(float(np.mean(data[bin_mask])))
                lat_stds.append(float(np.std(data[bin_mask])))
                lat_counts.append(int(bin_mask.sum()))
            else:
                lat_means.append(np.nan)
                lat_stds.append(np.nan)
                lat_counts.append(0)
        
        # Store results
        results['datasets'][name] = {
            'means': lat_means,
            'stds': lat_stds,
            'counts': lat_counts,
            'peak_latitude': float(lat_centers[np.nanargmax(lat_means)]) if any(~np.isnan(lat_means)) else np.nan,
            'tropical_mean': float(np.nanmean([m for m, lat in zip(lat_means, lat_centers) 
                                              if -23.5 <= lat <= 23.5])),
            'temperate_mean': float(np.nanmean([m for m, lat in zip(lat_means, lat_centers) 
                                               if 23.5 < abs(lat) <= 66.5]))
        }
    
    # Plot if requested
    if plot and results['datasets']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, data in results['datasets'].items():
            means = np.array(data['means'])
            valid = ~np.isnan(means)
            ax.plot(lat_centers[valid], means[valid], 'o-', label=name, linewidth=2)
        
        ax.set_xlabel('Latitude (degrees)')
        ax.set_ylabel('Biodiversity Richness')
        ax.set_title('Latitudinal Gradient')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        save_figure(fig, output_dir / 'latitudinal_gradient.png')
        plt.close(fig)
    
    # Save
    save_results(results, output_dir / 'latitudinal_results.json')
    
    return results