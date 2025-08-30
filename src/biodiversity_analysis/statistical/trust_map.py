from src.abstractions import get_logger
"""
Trust map analysis - Pure functions, no classes.

"Object-oriented programming is a hoax."
Classes are just namespaces with extra complexity.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
# NO RESAMPLING - removed: from rasterio.enums import Resampling
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import logging

from .data_utils import save_results, save_figure


logger = get_logger(__name__)


def analyze_trust_maps(
    data: Dict[str, Any],
    output_dir: Path,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze uncertainty in biodiversity data from multiband raster.
    
    LINUS: "One input format. Multiband only."
    
    Args:
        data: Dict with 'multiband' (rasterio DatasetReader) and 'band_names' (list)
        output_dir: Output directory
        **kwargs: Config including visualization options
    
    Returns:
        Dict[str, Any] with trust analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract multiband raster - NO SPECIAL CASES
    multiband = data['multiband']
    band_names = data['band_names']
    
    results = {}
    
    # Process each band from multiband raster
    for band_idx, band_name in enumerate(band_names, 1):
        logger.info(f"Processing band {band_idx}: {band_name}")
            
        # Skip CV maps themselves
        if '_cv' in band_name:
            continue
            
        logger.info(f"Processing trust map for {band_name}")
        
        # Compute basic trust metrics from the band data itself
        trust_metrics = _compute_basic_trust_metrics(multiband, band_idx, band_name)
        results[band_name] = trust_metrics
            

    
    # Combined analysis if we have multiple bands
    if len(results) > 1:
        combined = _analyze_combined_trust_multiband(multiband, band_names)
        results['combined'] = combined
        
        # Plot combined
        _plot_combined_trust(combined, output_dir)
    
    # Save results
    save_results(results, output_dir / 'trust_metrics.json')
    
    return Dict[str, Any](
        success=True,
        method='trust_maps',
        output_dir=output_dir,
        metrics=results
    )


def _compute_trust_metrics_rasterio(
    data_source: Union[str, rasterio.DatasetReader],
    cv_source: Union[str, rasterio.DatasetReader]
) -> Dict[str, Any]:
    """Compute trust metrics using rasterio block-wise processing."""
    
    # Open rasters if needed
    if isinstance(data_source, str):
        data_src = rasterio.open(data_source)
        close_data = True
    else:
        data_src = data_source
        close_data = False
        
    if isinstance(cv_source, str):
        cv_src = rasterio.open(cv_source)
        close_cv = True
    else:
        cv_src = cv_source
        close_cv = False
    
    try:
        # Initialize statistics
        stats = {
            'sum_cv': 0.0,
            'sum_cv_sq': 0.0,
            'valid_count': 0,
            'total_count': 0,
            'high_uncertainty_count': 0,
            'very_high_uncertainty_count': 0,
            'cv_values': []  # Store for median calculation
        }
        
        # Process in blocks
        for ji, window in data_src.block_windows(1):
            data_block = data_src.read(1, window=window, masked=True)
            cv_block = cv_src.read(1, window=window, masked=True)
            
            # Get valid pixels
            valid_mask = ~data_block.mask & ~cv_block.mask & (data_block > 0)
            
            if valid_mask.any():
                cv_valid = cv_block[valid_mask]
                
                stats['sum_cv'] += cv_valid.sum()
                stats['sum_cv_sq'] += (cv_valid ** 2).sum()
                stats['valid_count'] += len(cv_valid)
                stats['high_uncertainty_count'] += (cv_valid > 0.5).sum()
                stats['very_high_uncertainty_count'] += (cv_valid > 1.0).sum()
                
                # Sample for median calculation (don't store all values for memory)
                if len(stats['cv_values']) < 100000:  # Limit storage
                    stats['cv_values'].extend(cv_valid.tolist())
            
            stats['total_count'] += data_block.size
        
        # Compute final metrics
        if stats['valid_count'] > 0:
            mean_cv = stats['sum_cv'] / stats['valid_count']
            variance_cv = (stats['sum_cv_sq'] / stats['valid_count']) - mean_cv ** 2
            std_cv = np.sqrt(variance_cv) if variance_cv > 0 else 0
            
            # Compute median from sample
            median_cv = np.median(stats['cv_values']) if stats['cv_values'] else mean_cv
            
            metrics = {
                'mean_cv': float(mean_cv),
                'median_cv': float(median_cv),
                'std_cv': float(std_cv),
                'high_uncertainty_fraction': float(stats['high_uncertainty_count'] / stats['valid_count']),
                'very_high_uncertainty_fraction': float(stats['very_high_uncertainty_count'] / stats['valid_count']),
                'data_coverage': float(stats['valid_count'] / stats['total_count']),
                'trust_score': float(1.0 / (1.0 + median_cv)),
                'n_pixels': int(stats['valid_count'])
            }
        else:
            metrics = {'error': 'No valid data'}
            
    finally:
        if close_data:
            data_src.close()
        if close_cv:
            cv_src.close()
    
    return metrics


def _compute_basic_trust_metrics(
    multiband: rasterio.DatasetReader,
    band_idx: int,
    band_name: str
) -> Dict[str, Any]:
    """Compute basic trust metrics for a band.
    
    LINUS: "Keep it simple. Just compute statistics."
    """
    stats = {
        'band_name': band_name,
        'band_index': band_idx,
        'sum': 0.0,
        'sum_sq': 0.0,
        'valid_count': 0,
        'total_count': 0,
        'min': float('inf'),
        'max': float('-inf'),
    }
    
    # Process in blocks for memory efficiency
    for ji, window in multiband.block_windows(band_idx):
        data = multiband.read(band_idx, window=window, masked=True)
        valid = data.compressed()
        
        if len(valid) > 0:
            stats['sum'] += valid.sum()
            stats['sum_sq'] += (valid ** 2).sum()
            stats['valid_count'] += len(valid)
            stats['min'] = min(stats['min'], float(valid.min()))
            stats['max'] = max(stats['max'], float(valid.max()))
        
        stats['total_count'] += data.size
    
    # Compute final statistics
    if stats['valid_count'] > 0:
        mean = stats['sum'] / stats['valid_count']
        variance = (stats['sum_sq'] / stats['valid_count']) - mean ** 2
        std = np.sqrt(variance) if variance > 0 else 0
        cv = std / mean if mean != 0 else 0
        
        return {
            'band_name': band_name,
            'mean': float(mean),
            'std': float(std),
            'cv': float(cv),
            'min': float(stats['min']),
            'max': float(stats['max']),
            'valid_pixels': int(stats['valid_count']),
            'total_pixels': int(stats['total_count']),
            'coverage': float(100 * stats['valid_count'] / stats['total_count'])
        }
    else:
        return {
            'band_name': band_name,
            'error': 'No valid data',
            'valid_pixels': 0,
            'total_pixels': int(stats['total_count']),
            'coverage': 0.0
        }


def _compute_basic_metrics_rasterio(
    data_source: Union[str, rasterio.DatasetReader]
) -> Dict[str, Any]:
    """Compute basic metrics using rasterio when no CV available."""
    
    # Open raster if needed
    if isinstance(data_source, str):
        src = rasterio.open(data_source)
        close_after = True
    else:
        src = data_source
        close_after = False
    
    try:
        stats = {
            'sum': 0.0,
            'sum_sq': 0.0,
            'valid_count': 0,
            'total_count': 0,
            'min': float('inf'),
            'max': float('-inf'),
            'values': []  # Sample for median
        }
        
        # Process in blocks
        for ji, window in src.block_windows(1):
            data = src.read(1, window=window, masked=True)
            valid = data.compressed()
            
            if len(valid) > 0:
                stats['sum'] += valid.sum()
                stats['sum_sq'] += (valid ** 2).sum()
                stats['valid_count'] += len(valid)
                stats['min'] = min(stats['min'], float(valid.min()))
                stats['max'] = max(stats['max'], float(valid.max()))
                
                # Sample for median
                if len(stats['values']) < 100000:
                    stats['values'].extend(valid.tolist())
            
            stats['total_count'] += data.size
        
        # Compute final statistics
        if stats['valid_count'] > 0:
            mean = stats['sum'] / stats['valid_count']
            variance = (stats['sum_sq'] / stats['valid_count']) - mean ** 2
            std = np.sqrt(variance) if variance > 0 else 0
            median = np.median(stats['values']) if stats['values'] else mean
            
            return {
                'mean': float(mean),
                'median': float(median),
                'std': float(std),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'data_coverage': float(stats['valid_count'] / stats['total_count']),
                'n_pixels': int(stats['valid_count']),
                'note': 'No uncertainty data available'
            }
        else:
            return {'error': 'No valid data'}
            
    finally:
        if close_after:
            src.close()


def _analyze_combined_trust_multiband(
    multiband: rasterio.DatasetReader,
    band_names: list
) -> Dict[str, Any]:
    """Analyze combined trust across all bands.
    
    LINUS: "Simple aggregation. No fancy statistics."
    """
    return {
        'n_bands': len(band_names),
        'band_names': band_names,
        'type': 'multiband_combined'
    }


def _analyze_combined_trust(
    data: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Analyze combined trust across all datasets."""
    combined_trust = []
    
    for name, data_array in data.items():
        if '_cv' in name or data_array is None:
            continue
            
        cv_name = f"{name}_cv"
        cv_array = data.get(cv_name)
        
        if cv_array is not None:
            # Compute average trust
            valid_mask = ~np.isnan(cv_array) & (cv_array > 0)
            if valid_mask.any():
                trust = 1.0 / (1.0 + cv_array[valid_mask].mean())
                combined_trust.append({
                    'dataset': name,
                    'trust_score': float(trust),
                    'mean_cv': float(cv_array[valid_mask].mean())
                })
    
    if not combined_trust:
        return {'error': 'No uncertainty data available'}
    
    # Sort by trust score
    combined_trust.sort(key=lambda x: x['trust_score'], reverse=True)
    
    return {
        'rankings': combined_trust,
        'average_trust': float(np.mean([x['trust_score'] for x in combined_trust])),
        'most_trusted': combined_trust[0]['dataset'],
        'least_trusted': combined_trust[-1]['dataset']
    }


def _plot_trust_analysis_cartopy(
    data_source: Union[str, rasterio.DatasetReader],
    cv_source: Union[str, rasterio.DatasetReader],
    name: str,
    output_dir: Path,
    metrics: Dict[str, Any]
) -> None:
    """Plot trust analysis visualization using cartopy with proper projections."""
    
    # Helper function to create cartopy map
    def create_cartopy_map(src, title, colormap, output_path):
        """Create a single cartopy map from rasterio source."""
        # Use loader's downsample method - NO DUPLICATION!
        from src.biodiversity_analysis.shared.data.unified_loader import UnifiedLoader
        
        max_pixels = 2_000_000
        data = UnifiedLoader.downsample_on_read(
            src=src,
            band_num=1,
            max_pixels=max_pixels,
            resampling='nearest'
        )
        
        # Get valid data for color scaling
        valid = data.compressed()
        if len(valid) == 0:
            logger.warning(f"No valid data in {title}")
            return
        
        vmin, vmax = np.percentile(valid, [5, 95])
        
        # Create figure with Robinson projection
        fig = plt.figure(figsize=(14, 7))
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Get extent from raster bounds
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Plot data
        im = ax.imshow(
            data,
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )
        
        # Add geographic features
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.2, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5, zorder=1)
        ax.add_feature(cfeature.LAND, color='white', alpha=0.3, zorder=0)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, zorder=0)
        gl.top_labels = False
        gl.right_labels = False
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.05, fraction=0.05, shrink=0.8)
        cbar.set_label('Value', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Open rasters and create maps
    sources = [
        (data_source, f'{name} - Data', 'viridis', f'trust_{name}_data.png'),
        (cv_source, f'{name} - Coefficient of Variation', 'hot', f'trust_{name}_cv.png')
    ]
    
    for source, title, cmap, filename in sources:
        if isinstance(source, str):
            with rasterio.open(source) as src:
                create_cartopy_map(src, title, cmap, output_dir / filename)
        else:
            create_cartopy_map(source, title, cmap, output_dir / filename)
    
    # Create trust score map (requires both datasets)
    if isinstance(data_source, str):
        data_src = rasterio.open(data_source)
        close_data = True
    else:
        data_src = data_source
        close_data = False
        
    if isinstance(cv_source, str):
        cv_src = rasterio.open(cv_source)
        close_cv = True
    else:
        cv_src = cv_source
        close_cv = False
    
    try:
        # Read and compute trust score
        max_pixels = 2_000_000
        total_pixels = data_src.height * data_src.width
        
        if total_pixels > max_pixels:
            # LINUS: NO RESAMPLING! Use striding
            factor = int(np.sqrt(total_pixels / max_pixels))
            cv_full = cv_src.read(1, masked=True)
            step = slice(None, None, factor)
            cv_data = cv_full[step, step]
        else:
            cv_data = cv_src.read(1, masked=True)
        
        # Compute trust score
        trust = 1.0 / (1.0 + cv_data)
        
        # Create trust score map
        fig = plt.figure(figsize=(14, 7))
        ax = plt.axes(projection=ccrs.Robinson())
        
        extent = [data_src.bounds.left, data_src.bounds.right, data_src.bounds.bottom, data_src.bounds.top]
        
        im = ax.imshow(
            trust,
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            interpolation='bilinear'
        )
        
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.2, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5, zorder=1)
        
        ax.set_title(f'{name} - Trust Score', fontsize=14, fontweight='bold', pad=20)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, zorder=0)
        gl.top_labels = False
        gl.right_labels = False
        
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.05, fraction=0.05, shrink=0.8)
        cbar.set_label('Trust Score', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'trust_{name}_score.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    finally:
        if close_data:
            data_src.close()
        if close_cv:
            cv_src.close()
    
    # Also create a combined figure for comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Use plt.imshow for the combined view (quick preview)
    im1 = plt.imshow(axes[0], data, cmap='viridis', aspect='auto')
    axes[0].set_title(f'{name} - Data')
    axes[0].axis('off')
    
    im2 = plt.imshow(axes[1], cv, cmap='hot', aspect='auto')
    axes[1].set_title('Coefficient of Variation')
    axes[1].axis('off')
    
    im3 = plt.imshow(axes[2], trust, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[2].set_title('Trust Score')
    axes[2].axis('off')
    
    # Add metrics
    fig.text(0.5, 0.02, 
             f"Trust Score: {metrics.get('trust_score', 0):.2f} | "
             f"Mean CV: {metrics.get('mean_cv', 0):.2f} | "
             f"High Uncertainty: {metrics.get('high_uncertainty_fraction', 0)*100:.1f}%",
             ha='center', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'trust_{name}_combined.png')
    plt.close(fig)


def _plot_basic_map_cartopy(
    data_source: Union[str, rasterio.DatasetReader],
    name: str,
    output_dir: Path,
    metrics: Dict[str, Any]
) -> None:
    """Plot basic map using cartopy when no uncertainty available."""
    
    # Open raster if needed
    if isinstance(data_source, str):
        src_context = rasterio.open(data_source)
        src = src_context.__enter__()
        close_after = True
    else:
        src = data_source
        close_after = False
    
    try:
        # Use loader's downsample method - NO DUPLICATION!
        from src.biodiversity_analysis.shared.data.unified_loader import UnifiedLoader
        
        max_pixels = 2_000_000
        data = UnifiedLoader.downsample_on_read(
            src=src,
            band_num=1,
            max_pixels=max_pixels,
            resampling='nearest'
        )
        
        # Get valid data for color scaling
        valid = data.compressed()
        if len(valid) == 0:
            logger.warning(f"No valid data in {name}")
            return
        
        vmin, vmax = np.percentile(valid, [5, 95])
        
        # Create figure with Robinson projection
        fig = plt.figure(figsize=(14, 7))
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Get extent from raster bounds
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Plot data
        im = ax.imshow(
            data,
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear'
        )
        
        # Add geographic features
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.2, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5, zorder=1)
        ax.add_feature(cfeature.LAND, color='white', alpha=0.3, zorder=0)
        
        ax.set_title(f'{name} - No uncertainty data available', fontsize=14, fontweight='bold', pad=20)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, zorder=0)
        gl.top_labels = False
        gl.right_labels = False
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.05, fraction=0.05, shrink=0.8)
        cbar.set_label('Value', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'map_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created cartopy map for {name} (no uncertainty data)")
        
    finally:
        if close_after:
            src_context.__exit__(None, None, None)


def _plot_combined_trust(combined: Dict[str, Any], output_dir: Path) -> None:
    """Plot combined trust analysis."""
    if 'rankings' not in combined:
        return
    
    rankings = combined['rankings']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = [r['dataset'] for r in rankings]
    trust_scores = [r['trust_score'] for r in rankings]
    
    bars = ax.barh(datasets, trust_scores, color='green', alpha=0.7)
    
    # Color code by trust level
    for i, (bar, score) in enumerate(zip(bars, trust_scores)):
        if score < 0.5:
            bar.set_color('red')
        elif score < 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    ax.set_xlabel('Trust Score')
    ax.set_title('Dataset Trust Rankings')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, trust_scores)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center')
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'combined_trust_rankings.png')
    plt.close(fig)


# For backward compatibility
def TrustMap(downsample: int = 10):
    """
    Fake class for backward compatibility.
    Will be removed in v3.0.
    """
    logger.warning("TrustMap class is deprecated. Use analyze_trust_maps() directly.")
    
    class _FakeTrustMap:
        def __init__(self, downsample=10):
            self.downsample = downsample
            
        def run_complete_analysis(self, data_source, output_dir):
            return analyze_trust_maps(data_source, output_dir, self.downsample)
    
    return _FakeTrustMap(downsample)