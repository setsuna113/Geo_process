"""
Global map visualization - Professional cartographic standards.

"Use the right tool for the job" - Linus
Cartopy for projections, rasterio for data handling.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import logging
import rasterio
from rasterio.plot import show as rasterio_show
import rasterio.enums  # Need for Resampling during read
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure
from src.biodiversity_analysis.statistical.result_types import MethodResult

logger = logging.getLogger(__name__)


def analyze_global_maps(
    data: Dict[str, Union[str, rasterio.DatasetReader]],
    output_dir: Path,
    **kwargs
) -> MethodResult:
    """
    Create global map visualizations using professional cartographic tools.
    
    LINUS: ONLY accepts the standard format from loader:
    {
        'multiband': rasterio.DatasetReader,
        'band_names': List[str]
    }
    
    Args:
        data: Standard multiband dict from loader
        output_dir: Output directory
        
    Returns:
        Map statistics and generated plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # STANDARD API: Must have multiband and band_names
    if 'multiband' not in data or 'band_names' not in data:
        raise ValueError(
            "Invalid data format! Expected {'multiband': dataset, 'band_names': [...]}\n"
            "This is a bug in the data loader - it should always provide merged multiband rasters!"
        )
    
    multiband_src = data['multiband']
    band_names = data['band_names']
    
    if not hasattr(multiband_src, 'read'):
        raise TypeError(f"'multiband' must be a rasterio dataset, got {type(multiband_src)}")
    
    # Process each band from the multiband raster
    stats = {}
    
    for i, name in enumerate(band_names[:multiband_src.count], 1):
        # Each band is accessed through the multiband dataset
        src = multiband_src  # Use the multiband dataset
        band_num = i  # Track which band we're processing
            
        # Calculate statistics using rasterio (memory efficient)
        stats[name] = _compute_raster_stats(src, band_num)
        
        # Create individual map (it downsamples internally during read)
        _create_cartopy_map(src, name, output_dir, band_num=band_num, **kwargs)
        
        # REMOVED redundant downsampling - _create_cartopy_map already handles it!
    
    # Save statistics
    save_results(stats, output_dir / 'global_map_stats.json')
    
    return MethodResult(
        success=True,
        method='global_maps',
        output_dir=output_dir,
        metrics=stats
    )


def _compute_raster_stats(src: rasterio.DatasetReader, band_num: int = 1) -> Dict[str, Any]:
    """Compute statistics from raster without loading full array.
    
    LINUS FIX: Don't collect 803M values! Use online statistics.
    """
    # Use online statistics - no collecting all values!
    stats = {}
    
    # Online statistics computation
    count = 0
    sum_val = 0.0
    sum_sq = 0.0
    min_val = float('inf')
    max_val = float('-inf')
    
    # Process blocks but DON'T store all values
    for ji, window in src.block_windows(band_num):
        data = src.read(band_num, window=window, masked=True)
        valid = data.compressed()
        if len(valid) > 0:
            count += len(valid)
            sum_val += valid.sum()
            sum_sq += (valid ** 2).sum()
            min_val = min(min_val, valid.min())
            max_val = max(max_val, valid.max())
    
    if count > 0:
        mean = sum_val / count
        variance = (sum_sq / count) - (mean ** 2)
        std = np.sqrt(max(0, variance))  # Handle numerical issues
        stats = {
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean),
            'std': float(std),
            'coverage': float(100 * count / (src.height * src.width)),
            'shape': (src.height, src.width),
            'valid_pixels': int(count),
            'bounds': src.bounds,
            'crs': str(src.crs) if src.crs else None
        }
    else:
        stats = {'error': 'No valid data'}
    
    return stats


def _create_global_map(
    arr: np.ndarray,
    name: str,
    output_dir: Path,
    **kwargs
) -> None:
    """DEPRECATED - Use _create_cartopy_map instead.
    
    This is kept only for backward compatibility.
    """
    logger.warning(f"Using deprecated _create_global_map for {name}. Switch to _create_cartopy_map!")
    
    # Create temp raster in memory for compatibility
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    
    height, width = arr.shape
    transform = from_bounds(-180, -90, 180, 90, width, height)
    
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=arr.dtype,
            transform=transform,
            crs='EPSG:4326'
        ) as dataset:
            dataset.write(arr, 1)
            _create_cartopy_map(dataset, name, output_dir, **kwargs)


def _create_cartopy_map(
    src: rasterio.DatasetReader,
    name: str,
    output_dir: Path,
    band_num: int = 1,
    **kwargs
) -> None:
    """Create professional map using cartopy with proper projections."""
    
    # Use loader's downsample method - NO DUPLICATION!
    from src.biodiversity_analysis.shared.data.unified_loader import UnifiedLoader
    
    max_pixels = kwargs.get('max_pixels', 2_000_000)
    data = UnifiedLoader.downsample_on_read(
        src=src,
        band_num=band_num,
        max_pixels=max_pixels,
        resampling='nearest'
    )
    
    # Get valid data for color scaling
    valid = data.compressed()
    if len(valid) == 0:
        logger.warning(f"No valid data in {name}")
        return
    
    # Robust percentiles for visualization
    vmin, vmax = np.percentile(valid, [5, 95])
    
    # Create figure with cartopy projection
    # OPTIMIZATION: Use PlateCarree for best performance when possible
    projection = kwargs.get('projection', 'platecarree')  # Changed default for speed
    
    if projection == 'robinson':
        proj = ccrs.Robinson()
    elif projection == 'mollweide':
        proj = ccrs.Mollweide()
    elif projection == 'platecarree':
        proj = ccrs.PlateCarree()
    else:
        proj = ccrs.PlateCarree()  # Default to fastest
    
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=proj)
    
    # Get extent from raster bounds
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    
    # Custom colormap based on data type
    if 'fungi' in name.lower():
        colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', 
                  '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    elif 'plant' in name.lower():
        colors = ['#ffffe5', '#fff7bc', '#fee391', '#fec44f',
                  '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    else:
        colors = 'viridis'
    
    if isinstance(colors, list):
        cmap = LinearSegmentedColormap.from_list('custom', colors)
    else:
        cmap = colors
    
    # Plot data with proper transform
    # OPTIMIZATION: When using PlateCarree projection with PlateCarree data, no transform needed
    if projection == 'platecarree':
        # Direct plotting without transform is much faster
        im = ax.imshow(
            data,
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'  # Faster than bilinear
        )
    else:
        # Need transform for other projections
        im = ax.imshow(
            data,
            extent=extent,
            transform=ccrs.PlateCarree(),  # Source data CRS
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )
    
    # Add geographic features with resolution optimization
    # Use 110m resolution for faster plotting (vs 10m or 50m)
    resolution = '110m'  # Coarsest resolution for best performance
    
    # Add all features - they're mandatory for professional output
    ax.add_feature(cfeature.OCEAN.with_scale(resolution), color='lightblue', alpha=0.2, zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), linewidth=0.5, color='gray', zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale(resolution), linewidth=0.3, color='gray', alpha=0.5, zorder=1)
    ax.add_feature(cfeature.LAND.with_scale(resolution), color='white', alpha=0.3, zorder=0)
    
    # Title mapping for professional output
    title_map = {
        'animals': 'Terrestrial Vertebrate Species Richness',
        'plants': 'Vascular Plant Species Richness',
        'am_fungi': 'Arbuscular Mycorrhizal Fungi Richness',
        'ecm_fungi': 'Ectomycorrhizal Fungi Richness',
        'merged_fungi': 'Total Fungi Species Richness'
    }
    
    title = title_map.get(name, name.replace('_', ' ').title())
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines - mandatory for map readability
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, zorder=0)
    gl.top_labels = False
    gl.right_labels = False
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                       pad=0.05, fraction=0.05, shrink=0.8)
    cbar.set_label('Species Richness', fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    # Add statistics annotation
    stats_text = f'Mean: {valid.mean():.1f}\n'
    stats_text += f'Max: {valid.max():.0f}\n'
    stats_text += f'Coverage: {100 * len(valid) / (src.height * src.width):.1f}%'
    
    ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'{name}_global_map.png'
    save_figure(fig, output_path, dpi=150)
    plt.close()
    
    logger.info(f"Created cartopy map for {name} at {output_path}")


def _create_comparison_plot_fast(
    downsampled_arrays: Dict[str, np.ndarray],
    output_dir: Path
) -> None:
    """Create comparison plot using ALREADY downsampled arrays.
    
    LINUS: This is the RIGHT way - reuse downsampled data!
    No double loading of 803M pixels!
    """
    n_datasets = len(downsampled_arrays)
    if n_datasets < 2:
        return
    
    # Create subplot grid
    cols = 2
    rows = (n_datasets + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = np.array([axes]).ravel()
    else:
        axes = axes.ravel()
    
    for idx, (name, arr) in enumerate(downsampled_arrays.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Convert to float for masking
        arr_viz = arr.astype(np.float32)
        
        # Mask and plot
        masked = np.ma.masked_invalid(arr_viz)
        valid = masked.compressed()
        
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, [2, 98])
            
            # Simple imshow for pre-downsampled data
            im = ax.imshow(
                masked,
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                aspect='auto'
            )
            
            ax.set_title(f'{name.replace("_", " ").title()}', fontsize=12)
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Global Biodiversity Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'biodiversity_comparison.png'
    save_figure(plt.gcf(), output_path, dpi=100)
    plt.close()
    
    logger.info(f"Created comparison plot at {output_path}")


def _create_comparison_plot(
    data: Dict[str, Union[str, rasterio.DatasetReader]],
    output_dir: Path,
    **kwargs
) -> None:
    """Create comparison plot of all datasets.
    
    LINUS FIX: Handle file paths properly - don't assume arrays!
    """
    
    n_datasets = len(data)
    if n_datasets < 2:
        return
    
    # Create subplot grid
    cols = 2
    rows = (n_datasets + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.ravel() if rows > 1 else axes
    
    for idx, (name, data_source) in enumerate(data.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Load the data properly
        if isinstance(data_source, str):
            with rasterio.open(data_source) as src:
                # Downsample for visualization
                factor = max(1, src.height // 1000)
                arr = src.read(
                    1,
                    out_shape=(src.height // factor, src.width // factor)
                )
        else:
            # It's already a rasterio reader
            factor = max(1, data_source.height // 1000)
            arr = data_source.read(
                1,
                out_shape=(data_source.height // factor, data_source.width // factor)
            )
        
        # Convert to float for masking
        arr_viz = arr.astype(np.float32)
        
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


def _create_cartopy_map_from_array(
    arr: np.ndarray,
    name: str,
    output_dir: Path,
    **kwargs
) -> None:
    """Create cartopy map from numpy array with proper downsampling."""
    
    # Downsample if needed
    max_pixels = kwargs.get('max_pixels', 2_000_000)
    if arr.size > max_pixels:
        factor = int(np.sqrt(arr.size / max_pixels))
        arr_viz = arr[::factor, ::factor]
        logger.info(f"Downsampled by {factor}x for visualization")
    else:
        arr_viz = arr
    
    # Mask invalid
    masked = np.ma.masked_invalid(arr_viz)
    valid = masked.compressed()
    
    if len(valid) == 0:
        logger.warning(f"No valid data in {name}")
        return
    
    # Get robust limits
    vmin, vmax = np.percentile(valid, [5, 95])
    
    # Create figure with cartopy projection
    projection = kwargs.get('projection', 'robinson')
    
    if projection == 'robinson':
        proj = ccrs.Robinson()
    elif projection == 'mollweide':
        proj = ccrs.Mollweide()
    elif projection == 'platecarree':
        proj = ccrs.PlateCarree()
    else:
        proj = ccrs.Robinson()
    
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=proj)
    
    # Assume global extent
    extent = [-180, 180, -90, 90]
    
    # Custom colormap based on data type
    if 'fungi' in name.lower():
        colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', 
                  '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    elif 'plant' in name.lower():
        colors = ['#ffffe5', '#fff7bc', '#fee391', '#fec44f',
                  '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506']
    else:
        colors = 'viridis'
    
    if isinstance(colors, list):
        cmap = LinearSegmentedColormap.from_list('custom', colors)
    else:
        cmap = colors
    
    # Plot data
    im = ax.imshow(
        masked,
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='bilinear'
    )
    
    # Add geographic features
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.2, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5, zorder=1)
    ax.add_feature(cfeature.LAND, color='white', alpha=0.3, zorder=0)
    
    # Title mapping
    title_map = {
        'animals': 'Terrestrial Vertebrate Species Richness',
        'plants': 'Vascular Plant Species Richness',
        'am_fungi': 'Arbuscular Mycorrhizal Fungi Richness',
        'ecm_fungi': 'Ectomycorrhizal Fungi Richness',
        'merged_fungi': 'Total Fungi Species Richness'
    }
    
    title = title_map.get(name, name.replace('_', ' ').title())
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, zorder=0)
    gl.top_labels = False
    gl.right_labels = False
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                       pad=0.05, fraction=0.05, shrink=0.8)
    cbar.set_label('Species Richness', fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'nature_style_{name}.png', dpi=300)
    plt.close()


def create_nature_style_map(
    arr: np.ndarray,
    name: str,
    output_dir: Path,
    **kwargs
) -> None:
    """
    Create publication-quality map using cartopy.
    
    For when you actually need to publish.
    Properly handles large arrays with downsampling.
    """
    # Use cartopy directly for publication-quality maps
    _create_cartopy_map_from_array(arr, name, output_dir, **kwargs)
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