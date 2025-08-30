from src.abstractions import get_logger
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
from src.biodiversity_analysis.statistical.smart_visualization import smart_imshow
from src.biodiversity_analysis.statistical.professional_map_visualizer import ProfessionalMapVisualizer

logger = get_logger(__name__)
from src.biodiversity_analysis.statistical.result_types import MethodResult


def analyze_global_maps(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    **kwargs
) -> MethodResult:
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
    logger.info(f"[DEBUG] analyze_global_maps received output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[DEBUG] Created/verified directory: {output_dir}")
    
    if not data:
        return MethodResult(
            success=False,
            method='global_maps',
            error='No data provided'
        )
    
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
    
    return MethodResult(
        success=True,
        method='global_maps',
        output_dir=output_dir,
        metrics=stats
    )


def _create_global_map(
    arr: np.ndarray,
    name: str,
    output_dir: Path,
    **kwargs
) -> None:
    """Create a PROPER global map with correct projection.
    
    NO MORE DISTORTED MAPS! Using professional cartographic tools.
    """
    
    logger.info(f"{name}: Creating map with proper projection (shape={arr.shape})")
    
    # Use the PROFESSIONAL map visualizer with CORRECT projection
    # Default to Robinson projection (standard for global biodiversity)
    projection = kwargs.get('projection', 'robinson')
    
    # Determine colormap
    if 'cv' in name or 'uncertainty' in name:
        colormap = 'RdYlBu_r'  # High CV = red (bad), Low CV = blue (good)
    else:
        colormap = 'viridis'  # Standard for richness
    
    # Create the map with PROPER projection
    output_path = output_dir / f'global_{name}.png'
    logger.debug(f"Creating map at: {output_path}")
    
    # Assuming data is on regular lat/lon grid
    # Most biodiversity data is in WGS84 (EPSG:4326)
    # bounds = (west, east, south, north)
    bounds = (-180, 180, -90, 90)  # Global extent
    
    # Use professional map visualizer
    ProfessionalMapVisualizer.create_global_biodiversity_map(
        data=arr,
        name=name,
        output_path=output_path,
        bounds=bounds,
        projection=projection,
        publication_quality=True,
        add_features=True,  # Add coastlines, borders, etc.
        colormap=colormap,
        title=None  # Auto-generated from name
    )
    
    logger.info(f"Created properly projected map: {output_path}")


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
        
        # Data will be handled by smart_imshow
        arr_viz = arr
        
        # Mask and plot
        masked = np.ma.masked_invalid(arr_viz)
        valid = masked.compressed()
        
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, [2, 98])
            
            # Use smart_imshow for large arrays
            im = smart_imshow(
                ax,
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
    
    # Rasterize for efficient output
    from src.biodiversity_analysis.statistical.optimized_visualization import rasterize_for_pdf
    rasterize_for_pdf(fig, dpi=150)
    
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
    NOW WITH PROPER HANDLING OF 225M PIXELS!
    """
    # Use ProfessionalMapVisualizer - it handles everything properly
    ProfessionalMapVisualizer.create_global_biodiversity_map(
        data=arr,
        name=name,
        output_path=output_dir / f'nature_style_{name}.png',
        projection='robinson',
        publication_quality=True,
        add_features=True,
        title=None  # Will auto-generate proper title
    )
    return
    
    # OLD CODE BELOW - KEPT FOR REFERENCE BUT NOT EXECUTED
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
    
    # Plot data WITH PROPER EXTENT!
    # CRITICAL: Extent is the GEOGRAPHIC bounds, not array dimensions!
    # Our data (from TIF files) covers this geographic range:
    extent = [-180.0, 180.0, -89.83, 83.67]  # [west, east, south, north]
    
    im = ax.imshow(
        masked,
        extent=extent,  # Geographic extent (doesn't change with downsampling!)
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
    
    # Clean title - REAL production data names!
    title_map = {
        'animal_richness': 'Terrestrial Vertebrate Richness',
        'plant_richness': 'Vascular Plant Richness',
        'am_fungi_richness': 'AM Fungi Richness',
        'ecm_fungi_richness': 'ECM Fungi Richness',
        'total_fungi_richness': 'Total Fungi Richness'
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