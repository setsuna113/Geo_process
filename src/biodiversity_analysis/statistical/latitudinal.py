from src.abstractions import get_logger
"""
Latitudinal gradient analysis - Linus style.

"Show me the patterns, not your theories."
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import matplotlib.pyplot as plt
import rasterio
# NO RESAMPLING - removed: from rasterio.enums import Resampling
import logging

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure

logger = get_logger(__name__)


def analyze_latitudinal(
    data: Dict[str, Any],
    output_dir: Path,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze latitudinal gradients from multiband raster.
    
    LINUS: "One format to rule them all. Multiband only."
    
    Args:
        data: Dict with 'multiband' (rasterio DatasetReader) and 'band_names' (list)
        output_dir: Output directory
        **kwargs: Config including:
            - lat_bins: Number of latitude bins (default: 100 for fine resolution)
            - plot: Generate visualizations (default: True)
            - max_pixels: Maximum pixels before downsampling (default: 10M)
        
    Returns:
        Dict[str, Any] with latitudinal gradient statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract config with defaults
    lat_bins = kwargs.get('lat_bins', 100)  # Fine resolution for smooth gradients
    plot = kwargs.get('plot', True)
    smoothing_method = kwargs.get('smoothing_method', None)  # None, 'moving_average', 'loess'
    max_pixels = kwargs.get('max_pixels', 10_000_000)  # 10M pixels max for analysis
    
    # Academic standard: use actual resolution, no fake smoothing
    # Options:
    # - lat_bins=180 for 1-degree resolution (smoothest acceptable)
    # - lat_bins=90 for 2-degree resolution
    # - lat_bins=36 for 5-degree resolution (standard)
    # - lat_bins=18 for 10-degree resolution (coarse)
    
    # Extract multiband raster - NO SPECIAL CASES
    multiband = data['multiband']
    band_names = data['band_names']
    
    results = {'datasets': {}, 'lat_bins': lat_bins}
    
    # Assume global extent: -90 to 90 latitude
    lat_edges = np.linspace(-90, 90, lat_bins + 1)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    results['latitude_centers'] = lat_centers.tolist()
    
    # Process each band from the multiband raster
    for band_idx, band_name in enumerate(band_names, 1):
        logger.info(f"Processing band {band_idx}: {band_name}")
        
        try:
            # Get geographic bounds from raster
            bounds = multiband.bounds
            lat_min, lat_max = bounds.bottom, bounds.top
            height = multiband.height
            width = multiband.width
            
            # Initialize collectors for each latitude bin
            lat_sums = np.zeros(lat_bins)
            lat_sum_sqs = np.zeros(lat_bins)  
            lat_counts = np.zeros(lat_bins, dtype=np.int64)
            
            # Determine if we need to downsample
            total_pixels = height * width
            downsample_factor = 1
            if total_pixels > max_pixels:
                downsample_factor = int(np.sqrt(total_pixels / max_pixels))
                logger.info(f"Will process {band_name} with {downsample_factor}x downsampling")
            
            # Process in windows to avoid loading entire raster
            window_size = 512  # Process in 512x512 chunks
            for ji, window in multiband.block_windows(band_idx):
                # LINUS: NO RESAMPLING! Use striding for memory efficiency
                win_data = multiband.read(band_idx, window=window, masked=True)
                
                if downsample_factor > 1:
                    # Stride the data instead of resampling
                    step = slice(None, None, downsample_factor)
                    win_data = win_data[step, step]
                    
                    # Calculate latitude array for strided window
                    win_top = lat_max - (window.row_off / height) * (lat_max - lat_min)
                    win_bottom = lat_max - ((window.row_off + window.height) / height) * (lat_max - lat_min)
                    win_lats = np.linspace(win_top, win_bottom, win_data.shape[0])
                else:
                    # Use at full resolution
                    
                    # Calculate latitude array for this window
                    win_top = lat_max - (window.row_off / height) * (lat_max - lat_min)
                    win_bottom = lat_max - ((window.row_off + window.height) / height) * (lat_max - lat_min) 
                    win_lats = np.linspace(win_top, win_bottom, window.height)
                
                # Create latitude array for window
                lat_array = np.repeat(win_lats[:, np.newaxis], win_data.shape[1], axis=1)
                
                # Get valid mask
                valid_mask = ~win_data.mask if hasattr(win_data, 'mask') else ~np.isnan(win_data)
                
                # Accumulate statistics for each latitude bin
                for i in range(lat_bins):
                    lat_min_bin = lat_edges[i]
                    lat_max_bin = lat_edges[i + 1]
                    
                    # Select data in this latitude band
                    in_band = (lat_array >= lat_min_bin) & (lat_array < lat_max_bin) & valid_mask
                    if not np.any(in_band):
                        continue
                        
                    bin_data = win_data[in_band]
                    lat_sums[i] += np.sum(bin_data)
                    lat_sum_sqs[i] += np.sum(bin_data ** 2)
                    lat_counts[i] += len(bin_data)
            
            # Calculate final statistics from accumulated values
            lat_means = []
            lat_stds = []
            
            for i in range(lat_bins):
                lat_center = lat_centers[i]
                
                if lat_counts[i] > 0:
                    # Calculate mean and std from accumulated values
                    mean = lat_sums[i] / lat_counts[i]
                    variance = (lat_sum_sqs[i] / lat_counts[i]) - (mean ** 2)
                    std = np.sqrt(max(0, variance))  # Avoid negative variance due to rounding
                    
                    # Area weight = cos(latitude) - not used in stats but could be
                    # area_weight = np.cos(np.radians(lat_center))
                    
                    lat_means.append(float(mean))
                    lat_stds.append(float(std))
                else:
                    lat_means.append(None)
                    lat_stds.append(None)
            
            # Apply POST-PROCESSING smoothing if requested (academically acceptable)
            if smoothing_method and any(m is not None for m in lat_means):
                valid_indices = [i for i, m in enumerate(lat_means) if m is not None]
                
                if smoothing_method == 'moving_average' and len(valid_indices) > 5:
                    # Simple moving average (window=3)
                    from scipy.ndimage import uniform_filter1d
                    valid_means = np.array([lat_means[i] for i in valid_indices])
                    smoothed = uniform_filter1d(valid_means, size=3, mode='reflect')
                    for idx, i in enumerate(valid_indices):
                        lat_means[i] = float(smoothed[idx])
                        
                elif smoothing_method == 'loess' and len(valid_indices) > 10:
                    # LOESS smoothing (locally weighted regression)
                    from scipy.signal import savgol_filter
                    valid_means = np.array([lat_means[i] for i in valid_indices])
                    window_length = min(11, len(valid_indices) if len(valid_indices) % 2 == 1 else len(valid_indices) - 1)
                    if window_length >= 3:
                        smoothed = savgol_filter(valid_means, window_length, 2)
                        for idx, i in enumerate(valid_indices):
                            lat_means[i] = float(smoothed[idx])
            
            # Store results
            total_valid = int(np.sum(lat_counts))
            total_pixels = height * width
            results['datasets'][band_name] = {
                'means': lat_means,
                'stds': lat_stds,
                'counts': [int(c) for c in lat_counts],  # Convert numpy array to list
                'total_valid': total_valid,
                'coverage_percent': float(100 * total_valid / total_pixels) if total_pixels > 0 else 0,
                'smoothing_method': smoothing_method
            }
            
            # Find peak diversity latitude
            valid_means = [(i, m) for i, m in enumerate(lat_means) if m is not None]
            if valid_means:
                peak_idx, peak_val = max(valid_means, key=lambda x: x[1])
                results['datasets'][band_name]['peak_latitude'] = float(lat_centers[peak_idx])
                results['datasets'][band_name]['peak_value'] = float(peak_val)
        
        except Exception as e:
            logger.error(f"Error processing band {band_name}: {e}")
            raise  # Propagate the error instead of storing it
    
    # Generate plots
    if plot and results['datasets']:
        _plot_latitudinal_gradients(results, output_dir)
    
    # Save results
    save_results(results, output_dir / 'latitudinal_gradients.json')
    
    return Dict[str, Any](
        success=True,
        method='latitudinal',
        output_dir=output_dir,
        metrics=results
    )


def _plot_latitudinal_gradients(results: Dict[str, Any], output_dir: Path, generate_combined: bool = False):
    """Create latitudinal gradient plots."""
    
    lat_centers = np.array(results['latitude_centers'])
    n_datasets = len(results['datasets'])
    
    # Generate individual plots for each dataset (better for different ranges)
    for name, data in results['datasets'].items():
        # Create individual figure for this dataset
        fig, (ax_mean, ax_count) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        
        means = data['means']
        counts = data['counts']
        
        # Replace None with NaN for plotting
        means_plot = [m if m is not None else np.nan for m in means]
        
        # Plot mean richness
        # With 100 bins, use continuous line without markers
        ax_mean.plot(lat_centers, means_plot, '-', linewidth=2, color='#2E86AB')
        ax_mean.set_xlabel('Latitude (degrees)', fontsize=12)
        ax_mean.set_ylabel('Mean Species Richness', fontsize=12)
        ax_mean.set_title(f'{name.replace("_", " ").title()} - Latitudinal Gradient', fontsize=14, fontweight='bold')
        ax_mean.grid(True, alpha=0.3)
        ax_mean.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax_mean.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax_mean.set_xlim(-90, 90)
        
        # Add peak latitude line if exists
        if 'peak_latitude' in data:
            ax_mean.axvline(x=data['peak_latitude'], color='r', linestyle='--', 
                          alpha=0.5, label=f"Peak: {data['peak_latitude']:.1f}°")
            ax_mean.legend()
        
        # Plot sample counts
        ax_count.bar(lat_centers, counts, width=lat_centers[1] - lat_centers[0], 
                    alpha=0.7, color='#A0C1D1', edgecolor='none')
        ax_count.set_xlabel('Latitude (degrees)', fontsize=12)
        ax_count.set_ylabel('Sample Count', fontsize=10)
        ax_count.set_title('Data Coverage', fontsize=11)
        ax_count.grid(True, alpha=0.3, axis='y')
        ax_count.set_xlim(-90, 90)
        
        plt.tight_layout()
        
        # Save individual plot with dataset name
        save_figure(fig, output_dir / f'latitudinal_{name}.png', dpi=150)
        plt.close()
    
    # Only create combined plot if requested
    if generate_combined:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, data in results['datasets'].items():
            means = data['means']
            means_plot = [m if m is not None else np.nan for m in means]
            
            # Plot with label
            label = name.replace('_richness', '').replace('_', ' ').title()
            ax.plot(lat_centers, means_plot, 'o-', label=label, linewidth=2, markersize=4)
        
        ax.set_xlabel('Latitude (degrees)', fontsize=12)
        ax.set_ylabel('Mean Species Richness)', fontsize=12)
        ax.set_title('Global Latitudinal Diversity Gradients', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend(loc='best')
        
        # Add latitude band labels
        ax.axvspan(-90, -60, alpha=0.1, color='blue', label='Antarctic')
        ax.axvspan(60, 90, alpha=0.1, color='blue', label='Arctic')
        ax.axvspan(-23.5, 23.5, alpha=0.1, color='red', label='Tropics')
    
        plt.tight_layout()
        save_figure(fig, output_dir / 'latitudinal_combined.png', dpi=150)
        plt.close()
    
    logger.info(f"Saved latitudinal gradient plots to {output_dir}")