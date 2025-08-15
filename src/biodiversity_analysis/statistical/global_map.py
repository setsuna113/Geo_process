"""
Global map visualization - Linus style.

"Show me the data, not your art project."
Clean, simple, informative maps.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import logging

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure

logger = logging.getLogger(__name__)


def analyze_global_maps(
    data: Dict[str, np.ndarray],
    output_dir: Union[str, Path],
    **kwargs
) -> Dict[str, Any]:
    """
    Create global map visualizations.
    
    Data already loaded. Just visualize it.
    
    Args:
        data: Dict of numpy arrays (already loaded)
        output_dir: Output directory
        
    Returns:
        Map statistics and generated plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data:
        return {'error': 'No data provided'}
    
    # Process each dataset
    stats = {}
    
    for name, arr in data.items():
        if arr is None:
            continue
            
        # Calculate statistics
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0:
            stats[name] = {
                'min': float(valid.min()),
                'max': float(valid.max()),
                'mean': float(valid.mean()),
                'std': float(valid.std()),
                'coverage': float(100 * len(valid) / arr.size),
                'shape': arr.shape,
                'valid_pixels': int(len(valid))
            }
            
            # Create visualization
            _create_global_map(arr, name, output_dir, **kwargs)
        else:
            stats[name] = {'error': 'No valid data'}
    
    # Create combined comparison plot if multiple datasets
    if len(stats) > 1:
        _create_comparison_plot(data, output_dir, **kwargs)
    
    # Save statistics
    save_results(stats, output_dir / 'global_map_stats.json')
    
    return stats


def _create_global_map(
    arr: np.ndarray,
    name: str,
    output_dir: Path,
    **kwargs
) -> None:
    """Create a single global map visualization."""
    
    # Downsample for visualization if needed
    max_width = kwargs.get('max_viz_width', 2000)
    max_height = kwargs.get('max_viz_height', 1000)
    
    height, width = arr.shape
    downsample = max(1, max(width // max_width, height // max_height))
    
    if downsample > 1:
        arr_viz = arr[::downsample, ::downsample]
        logger.info(f"{name}: Downsampled {downsample}x for visualization")
    else:
        arr_viz = arr
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Mask invalid values
    masked = np.ma.masked_invalid(arr_viz)
    
    # Get data range (use percentiles to handle outliers)
    valid_data = masked.compressed()
    if len(valid_data) > 0:
        vmin, vmax = np.percentile(valid_data, [2, 98])
    else:
        vmin, vmax = 0, 1
    
    # Better titles
    title_map = {
        'animal_richness': 'Global Terrestrial Animal Species Richness',
        'plant_richness': 'Global Plant Species Richness',
        'am_fungi_richness': 'Global Arbuscular Mycorrhizal (AM) Fungi Richness',
        'ecm_fungi_richness': 'Global Ectomycorrhizal (ECM) Fungi Richness',
        'am_fungi_cv': 'AM Fungi Coefficient of Variation',
        'ecm_fungi_cv': 'ECM Fungi Coefficient of Variation'
    }
    title = title_map.get(name, name.replace('_', ' ').title())
    
    # Choose colormap based on data type
    if 'cv' in name or 'uncertainty' in name:
        cmap = 'RdYlBu_r'  # High CV = red (bad), Low CV = blue (good)
    else:
        cmap = 'viridis'  # Standard for richness
    
    # Plot
    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        interpolation='none'
    )
    
    # Add coastlines effect (simple version)
    ax.contour(~np.isnan(arr_viz), levels=[0.5], colors='black', linewidths=0.5, alpha=0.3)
    
    # Format
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)
    cbar.set_label('Species Richness' if 'cv' not in name else 'Coefficient of Variation', 
                   fontsize=10)
    
    # Add statistics text
    stats_text = f"Mean: {valid_data.mean():.1f}\nRange: {vmin:.1f} - {vmax:.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'global_{name}.png', dpi=150)
    plt.close()


def _create_comparison_plot(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    **kwargs
) -> None:
    """Create comparison plot of all datasets."""
    
    n_datasets = len(data)
    if n_datasets < 2:
        return
    
    # Create subplot grid
    cols = 2
    rows = (n_datasets + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.ravel() if rows > 1 else axes
    
    for idx, (name, arr) in enumerate(data.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Downsample for comparison plot
        height, width = arr.shape
        downsample = max(1, max(width // 1000, height // 500))
        if downsample > 1:
            arr_viz = arr[::downsample, ::downsample]
        else:
            arr_viz = arr
        
        # Mask and plot
        masked = np.ma.masked_invalid(arr_viz)
        valid = masked.compressed()
        
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, [2, 98])
            
            im = ax.imshow(
                masked,
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
                interpolation='none'
            )
            
            # Simple title
            ax.set_title(name.replace('_', ' ').title(), fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Small colorbar
            plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.05)
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Global Biodiversity Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, output_dir / 'global_comparison.png', dpi=150)
    plt.close()


def create_nature_style_map(
    arr: np.ndarray,
    name: str,
    output_dir: Path,
    **kwargs
) -> None:
    """
    Create publication-quality map.
    
    For when you actually need to publish.
    """
    from matplotlib.colors import LinearSegmentedColormap
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    
    # Mask invalid
    masked = np.ma.masked_invalid(arr)
    valid = masked.compressed()
    
    if len(valid) == 0:
        plt.close()
        return
    
    # Get robust limits
    vmin, vmax = np.percentile(valid, [5, 95])
    
    # Custom colormap for publication
    if 'fungi' in name:
        colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', 
                  '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    else:
        colors = ['#ffffe5', '#fff7bc', '#fee391', '#fec44f',
                  '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    # Plot data
    im = ax.imshow(
        masked,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='bilinear'
    )
    
    # Add geographic features
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5)
    
    # Clean title
    title_map = {
        'animal_richness': 'Terrestrial Vertebrate Richness',
        'plant_richness': 'Vascular Plant Richness',
        'am_fungi_richness': 'AM Fungi Richness',
        'ecm_fungi_richness': 'ECM Fungi Richness'
    }
    
    ax.set_title(title_map.get(name, name), fontsize=14, pad=10)
    
    # Add gridlines
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, fraction=0.05, shrink=0.8)
    cbar.set_label('Species richness', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'nature_style_{name}.png', dpi=300)
    plt.close()