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
        
        # Bin by latitude WITH area weighting
        valid_mask = ~np.isnan(data)
        lat_means = []
        lat_stds = []
        lat_counts = []
        lat_weighted_means = []  # Area-weighted means
        
        # Calculate pixel areas (proportional to cos(latitude))
        # This accounts for projection distortion
        lat_radians = np.deg2rad(lats)
        pixel_areas = np.cos(lat_radians)
        pixel_areas = np.repeat(pixel_areas[:, np.newaxis], width, axis=1)
        
        for i in range(lat_bins):
            lat_min, lat_max = lat_edges[i], lat_edges[i+1]
            bin_mask = (lat_array >= lat_min) & (lat_array < lat_max) & valid_mask
            
            if bin_mask.sum() > 0:
                # Unweighted stats (for comparison)
                lat_means.append(float(np.mean(data[bin_mask])))
                lat_stds.append(float(np.std(data[bin_mask])))
                lat_counts.append(int(bin_mask.sum()))
                
                # Area-weighted mean (correct for projection)
                weights = pixel_areas[bin_mask]
                values = data[bin_mask]
                weighted_mean = np.average(values, weights=weights)
                lat_weighted_means.append(float(weighted_mean))
            else:
                lat_means.append(np.nan)
                lat_stds.append(np.nan)
                lat_counts.append(0)
                lat_weighted_means.append(np.nan)
        
        # Store results with both weighted and unweighted
        results['datasets'][name] = {
            'means': lat_means,
            'weighted_means': lat_weighted_means,  # Area-corrected means
            'stds': lat_stds,
            'counts': lat_counts,
            'peak_latitude': float(lat_centers[np.nanargmax(lat_weighted_means)]) if any(~np.isnan(lat_weighted_means)) else np.nan,
            'tropical_mean': float(np.nanmean([m for m, lat in zip(lat_weighted_means, lat_centers) 
                                              if -23.5 <= lat <= 23.5])),
            'temperate_mean': float(np.nanmean([m for m, lat in zip(lat_weighted_means, lat_centers) 
                                               if 23.5 < abs(lat) <= 66.5])),
            'projection_note': 'weighted_means corrects for latitude area distortion'
        }
    
    # Plot if requested - SEPARATE SUBPLOTS for each dataset (different scales)
    if plot and results['datasets']:
        # Determine subplot layout based on number of datasets
        n_datasets = len(results['datasets'])
        if n_datasets <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        else:
            cols = int(np.ceil(np.sqrt(n_datasets)))
            rows = int(np.ceil(n_datasets / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
            axes = axes.flatten()
        
        # Colors for common dataset names
        color_map = {
            'plants': 'green', 'animals': 'orange',
            'am_fungi': 'purple', 'ecm_fungi': 'brown',
            'total_fungi': 'darkred'
        }
        
        for idx, (name, data) in enumerate(results['datasets'].items()):
            ax = axes[idx]
            
            # Use area-weighted means for accurate representation
            means = np.array(data['weighted_means'])
            stds = np.array(data.get('stds', [np.nan]*len(means)))
            valid = ~np.isnan(means)
            
            # Choose color
            color = color_map.get(name, f'C{idx}')
            
            # Main line plot
            ax.plot(lat_centers[valid], means[valid], 
                   'o-', color=color, linewidth=2.5, 
                   markersize=5, alpha=0.9)
            
            # Add error bands if available
            if 'stds' in data and any(~np.isnan(stds)):
                ax.fill_between(lat_centers[valid], 
                               (means - stds)[valid], 
                               (means + stds)[valid],
                               color=color, alpha=0.2)
            
            # Formatting
            ax.set_xlabel('Latitude (degrees)', fontsize=11)
            ax.set_ylabel('Richness', fontsize=11)
            ax.set_title(f'{name.replace("_", " ").title()}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Reference lines
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)  # Equator
            ax.axvline(x=23.5, color='gray', linestyle=':', alpha=0.3)  # Tropics
            ax.axvline(x=-23.5, color='gray', linestyle=':', alpha=0.3)
            
            # Mark peak
            if any(valid):
                peak_idx = np.nanargmax(means)
                peak_lat = data.get('peak_latitude', lat_centers[peak_idx])
                peak_val = means[peak_idx]
                ax.plot(peak_lat, peak_val, 'r*', markersize=10, alpha=0.7)
                
                # Add stats box
                stats_text = f'Peak: {peak_lat:.1f}°\n'
                if 'tropical_mean' in data:
                    stats_text += f'Tropical: {data["tropical_mean"]:.0f}\n'
                if 'temperate_mean' in data:
                    stats_text += f'Temperate: {data["temperate_mean"]:.0f}'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=9, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(n_datasets, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Latitudinal Biodiversity Gradients', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, output_dir / 'latitudinal_gradient.png', dpi=150)
        plt.close(fig)
    
    # Save
    save_results(results, output_dir / 'latitudinal_results.json')
    
    return results