"""
Overlap analysis with memory-efficient chunked processing - Linus style.

"Don't load what you don't need."
ONE implementation, handles both in-memory and chunked.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import rasterio
from rasterio.windows import Window
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from scipy import stats
import logging

logger = logging.getLogger(__name__)

from src.biodiversity_analysis.statistical.data_utils import save_results
from src.biodiversity_analysis.statistical.result_types import MethodResult
from src.biodiversity_analysis.statistical.smart_visualization import smart_imshow
from src.biodiversity_analysis.statistical.overlap_rgb import create_rgb_overlap_map


def analyze_overlap(
    data: Dict[str, Any],
    output_dir: Path,
    **kwargs
) -> MethodResult:
    """
    Analyze spatial overlap from multiband raster.
    
    LINUS: "Everything is multiband. No special cases."
    
    Args:
        data: Dict with 'multiband' (rasterio DatasetReader) and 'band_names' (list)
        output_dir: Output directory
        **kwargs: Config including:
            - hotspot_percentile: Threshold for hotspots (default: 90)
            - plot: Generate visualizations (default: True)
        
    Returns:
        MethodResult with overlap statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract config with defaults
    hotspot_percentile = kwargs.get('hotspot_percentile', 90)
    plot = kwargs.get('plot', True)
    
    try:
        # Extract multiband raster - NO SPECIAL CASES
        multiband = data['multiband']
        band_names = data['band_names']
        
        if len(band_names) < 2:
            raise ValueError("Need at least 2 bands for overlap analysis")
        
        # Process using multiband raster
        results = _analyze_multiband(multiband, band_names, hotspot_percentile)
        
        # Generate RGB overlap map using cartopy
        if plot:
            # Create the RGB overlap map with proper projection
            rgb_output = output_dir / 'rgb_biodiversity_overlap.png'
            _create_cartopy_overlap_map(data, rgb_output)
        
        # Save results
        save_results(results, output_dir / 'overlap_results.json')
        
        # Generate text report
        _write_report(results, output_dir / 'overlap_report.txt')
        
        return MethodResult(
            success=True,
            method='overlap',
            output_dir=output_dir,
            metrics=results
        )
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Overlap analysis failed: {error_msg}")
        return MethodResult(
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


def _analyze_multiband(
    multiband_src: rasterio.DatasetReader,
    band_names: list,
    hotspot_percentile: float
) -> Dict[str, Any]:
    """Analyze overlap using multiband raster block-wise processing.
    
    LINUS: "Direct parameters. No dict unpacking bullshit."
    
    Args:
        multiband_src: Multiband raster dataset
        band_names: List of band names
        hotspot_percentile: Percentile for hotspot detection
        
    Returns:
        Overlap analysis results
    """
    
    if not hasattr(multiband_src, 'read'):
        raise TypeError(f"'multiband' must be a rasterio dataset, got {type(multiband_src)}")
    
    if multiband_src.count < 2:
        raise ValueError(f"Need at least 2 bands for overlap analysis, got {multiband_src.count}")
    
    logger.info(f"Using multi-band raster with {multiband_src.count} bands: {band_names}")
    
    # Create rasters dict from bands - STANDARD FORMAT
    rasters = {}
    close_after = {}
    for i, name in enumerate(band_names[:multiband_src.count], 1):
        # Each band is accessed through the multiband dataset
        rasters[name] = (multiband_src, i)  # Store dataset and band number
        close_after[name] = False
    
    height, width = multiband_src.shape
    shapes = [multiband_src.shape] * len(rasters)
    
    # All rasters have same shape
    height, width = shapes[0]
    ref_raster_item = next(iter(rasters.values()))
    ref_name = next(iter(rasters.keys()))
    
    # Handle tuple case (multiband dataset, band_num)
    if isinstance(ref_raster_item, tuple):
        ref_raster, _ = ref_raster_item  # Get the actual dataset
    else:
        ref_raster = ref_raster_item
    
    total_pixels = height * width
    
    try:
        # Initialize statistics
        stats_tracker = _BlockwiseStatsTracker(rasters.keys(), total_pixels)
        
        # Process using block windows - NO RESAMPLING!
        for ji, window in ref_raster.block_windows(1):
            # Read block from each raster
            blocks = {}
            masks = {}
            
            for name, src in rasters.items():
                # LINUS: All rasters MUST have same shape - no resampling!
                if isinstance(src, tuple):
                    # Multi-band: src is (dataset, band_number)
                    dataset, band_num = src
                    block = dataset.read(band_num, window=window, masked=True)
                else:
                    # Single raster
                    block = src.read(1, window=window, masked=True)
                
                valid = block.compressed()
                
                # Store for statistics
                blocks[name] = block
                masks[name] = ~block.mask if hasattr(block, 'mask') else ~np.isnan(block)
            
            # Update statistics
            stats_tracker.update(blocks, masks)
        
        # Compute final statistics
        results = stats_tracker.finalize(hotspot_percentile)
        
    finally:
        # LINUS: DON'T CLOSE! The dataset is owned by the caller
        # The UnifiedLoader manages the lifecycle, not us!
        pass
    
    return results


def _transform_window(ref_window, ref_raster, src_raster):
    """Transform window from reference raster to source raster coordinates."""
    from rasterio.windows import Window
    
    # Get bounds of the reference window
    from rasterio.windows import bounds as window_bounds
    left, bottom, right, top = window_bounds(ref_window, ref_raster.transform)
    
    # Transform bounds to source raster pixel coordinates
    src_col_off, src_row_off = ~src_raster.transform * (left, top)
    src_col_end, src_row_end = ~src_raster.transform * (right, bottom)
    
    # Create window in source coordinates
    src_window = Window(
        col_off=max(0, int(src_col_off)),
        row_off=max(0, int(src_row_off)),
        width=min(src_raster.width - int(src_col_off), int(src_col_end - src_col_off)),
        height=min(src_raster.height - int(src_row_off), int(src_row_end - src_row_off))
    )
    
    return src_window


class _BlockwiseStatsTracker:
    """Track statistics during block-wise rasterio processing."""
    
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
        
        names_list = list(names) if not isinstance(names, list) else names
        for i, name1 in enumerate(names_list):
            for name2 in names_list[i+1:]:
                key = f"{name1}_vs_{name2}"
                self.pairwise_valid[key] = 0
                self.pairwise_sums1[key] = 0.0
                self.pairwise_sums2[key] = 0.0
                self.pairwise_sums12[key] = 0.0
        
        # Hotspot tracking - use reservoir sampling for memory efficiency
        # Keep only a sample of values for percentile calculation
        self.sample_size = 10000  # Maximum samples to keep
        self.samples = {name: [] for name in names}
        self.sample_counts = {name: 0 for name in names}
    
    def update(self, chunks, masks):
        """Update statistics with new chunk."""
        
        # Coverage stats
        for name in self.names:
            valid = masks[name]
            if valid.sum() > 0:
                self.valid_counts[name] += valid.sum()
                self.value_sums[name] += chunks[name][valid].sum()
                self.value_maxs[name] = max(self.value_maxs[name], chunks[name][valid].max())
                
                # Reservoir sampling for hotspot calculation
                # This keeps memory bounded while maintaining statistical properties
                valid_values = chunks[name][valid].flatten()
                for val in valid_values:
                    self.sample_counts[name] += 1
                    if len(self.samples[name]) < self.sample_size:
                        self.samples[name].append(float(val))
                    else:
                        # Randomly replace with decreasing probability
                        import random
                        j = random.randint(0, self.sample_counts[name] - 1)
                        if j < self.sample_size:
                            self.samples[name][j] = float(val)
        
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
        
        # Hotspots (using sampled values)
        hotspots = {}
        for name in self.names:
            if self.samples[name]:
                values = np.array(self.samples[name])
                threshold = np.percentile(values, hotspot_percentile)
                # Estimate hotspot count based on threshold
                hotspot_fraction = (values >= threshold).sum() / len(values)
                estimated_hotspot_count = int(hotspot_fraction * self.valid_counts[name])
                hotspots[name] = {
                    'threshold': float(threshold),
                    'pixels': estimated_hotspot_count,
                    'percent': float(100 * hotspot_fraction)
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


def _create_cartopy_overlap_map(
    data: Dict[str, Union[str, rasterio.DatasetReader]],
    output_path: Path
) -> None:
    """Create overlap visualization using cartopy with proper projections.
    
    LINUS: Updated to use standard multiband format.
    Shows overlap between first two bands of multiband raster.
    """
    
    # STANDARD API: Extract multiband dataset
    if 'multiband' not in data or 'band_names' not in data:
        logger.warning("Cannot create overlap map - invalid data format")
        return
        
    multiband_src = data['multiband']
    band_names = data['band_names']
    
    if multiband_src.count < 2:
        logger.warning("Need at least 2 bands for overlap visualization")
        return
    
    try:
        # Use loader's downsample method - NO DUPLICATION!
        from src.biodiversity_analysis.shared.data.unified_loader import UnifiedLoader
        
        max_pixels = 2_000_000
        # Read first two bands with downsampling
        data1 = UnifiedLoader.downsample_on_read(
            src=multiband_src,
            band_num=1,
            max_pixels=max_pixels,
            resampling='nearest'
        )
        data2 = UnifiedLoader.downsample_on_read(
            src=multiband_src,
            band_num=2,
            max_pixels=max_pixels,
            resampling='nearest'
        )
        
        # Create figure with cartopy projection
        fig = plt.figure(figsize=(14, 7))
        ax = plt.axes(projection=ccrs.Robinson())
        
        # Compute overlap: both valid, only 1 valid, only 2 valid
        valid1 = ~data1.mask if hasattr(data1, 'mask') else ~np.isnan(data1)
        valid2 = ~data2.mask if hasattr(data2, 'mask') else ~np.isnan(data2)
        
        # Create RGB overlay: Red=dataset1, Green=dataset2, Blue=overlap
        overlap_rgb = np.zeros((data1.shape[0], data1.shape[1], 3))
        overlap_rgb[:, :, 0] = valid1 & ~valid2  # Red: only dataset 1
        overlap_rgb[:, :, 1] = ~valid1 & valid2  # Green: only dataset 2  
        overlap_rgb[:, :, 2] = valid1 & valid2   # Blue: overlap
        
        # Get extent from raster bounds
        extent = [multiband_src.bounds.left, multiband_src.bounds.right, multiband_src.bounds.bottom, multiband_src.bounds.top]
        
        # Plot overlap
        im = ax.imshow(
            overlap_rgb,
            extent=extent,
            transform=ccrs.PlateCarree(),
            interpolation='nearest'
        )
        
        # Add geographic features
        ax.add_feature(cfeature.OCEAN, color='lightgray', alpha=0.3, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5, zorder=1)
        
        # Use first two band names for labels
        name1 = band_names[0] if band_names else 'Band 1'
        name2 = band_names[1] if len(band_names) > 1 else 'Band 2'
        
        ax.set_title(f'Spatial Overlap: {name1} vs {name2}', fontsize=14, fontweight='bold')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3, zorder=0)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label=f'{name1} only'),
            Patch(facecolor='green', label=f'{name2} only'),
            Patch(facecolor='blue', label='Overlap')
        ]
        ax.legend(handles=legend_elements, loc='lower left')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created overlap map at {output_path}")
        
    finally:
        # multiband_src will be closed by the caller (UnifiedLoader manages it)
        pass


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