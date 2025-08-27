"""
Latitudinal gradient analysis - Linus style.

"Show me the patterns, not your theories."
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import matplotlib.pyplot as plt
import logging

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure

logger = logging.getLogger(__name__)
from src.biodiversity_analysis.statistical.result_types import MethodResult


def analyze_latitudinal(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    **kwargs
) -> MethodResult:
    """
    Analyze latitudinal gradients in biodiversity.
    
    Data already loaded. Just analyze the gradients.
    
    Args:
        data: Dict[str, ndarray] - spatial arrays with coordinates
        output_dir: Output directory
        **kwargs: Config including:
            - lat_bins: Number of latitude bins (default: 36 for 5-degree bands)
            - plot: Generate visualizations (default: True)
        
    Returns:
        MethodResult with latitudinal gradient statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract config with defaults
    lat_bins = kwargs.get('lat_bins', 100)  # Changed to 100 bins for finer resolution
    plot = kwargs.get('plot', True)
    smoothing_method = kwargs.get('smoothing_method', None)  # None, 'moving_average', 'loess'
    
    # Academic standard: use actual resolution, no fake smoothing
    # Options:
    # - lat_bins=180 for 1-degree resolution (smoothest acceptable)
    # - lat_bins=90 for 2-degree resolution
    # - lat_bins=36 for 5-degree resolution (standard)
    # - lat_bins=18 for 10-degree resolution (coarse)
    
    results = {'datasets': {}, 'lat_bins': lat_bins}
    
    # Assume global extent: -90 to 90 latitude
    lat_edges = np.linspace(-90, 90, lat_bins + 1)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    results['latitude_centers'] = lat_centers.tolist()
    
    for name, arr in data.items():
        if arr is None:
            continue
        
        # LINUS: Skip CV columns - they're NOT biodiversity metrics!
        if '_cv' in name or 'coeff_var' in name:
            logger.info(f"Skipping CV column: {name}")
            continue
        
        height, width = arr.shape
        
        # Create latitude array (assuming global extent)
        lats = np.linspace(90, -90, height)
        lat_array = np.repeat(lats[:, np.newaxis], width, axis=1)
        
        # Bin by latitude WITH area weighting
        valid_mask = ~np.isnan(arr)
        lat_means = []
        lat_stds = []
        lat_counts = []
        
        # Calculate area weights (cosine of latitude)
        for i in range(lat_bins):
            lat_min = lat_edges[i]
            lat_max = lat_edges[i + 1]
            lat_center = lat_centers[i]
            
            # Select data in this latitude band
            in_band = (lat_array >= lat_min) & (lat_array < lat_max) & valid_mask
            bin_data = arr[in_band]
            
            if len(bin_data) > 0:
                # Area weight = cos(latitude)
                area_weight = np.cos(np.radians(lat_center))
                
                lat_means.append(float(np.mean(bin_data)))
                lat_stds.append(float(np.std(bin_data)))
                lat_counts.append(int(len(bin_data)))
            else:
                lat_means.append(None)
                lat_stds.append(None)
                lat_counts.append(0)
        
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
        results['datasets'][name] = {
            'means': lat_means,
            'stds': lat_stds,
            'counts': lat_counts,
            'total_valid': int(valid_mask.sum()),
            'coverage_percent': float(100 * valid_mask.sum() / arr.size),
            'smoothing_method': smoothing_method
        }
        
        # Find peak diversity latitude
        valid_means = [(i, m) for i, m in enumerate(lat_means) if m is not None]
        if valid_means:
            peak_idx, peak_val = max(valid_means, key=lambda x: x[1])
            results['datasets'][name]['peak_latitude'] = float(lat_centers[peak_idx])
            results['datasets'][name]['peak_value'] = float(peak_val)
    
    # Generate plots
    if plot and results['datasets']:
        _plot_latitudinal_gradients(results, output_dir)
    
    # Save results
    save_results(results, output_dir / 'latitudinal_gradients.json')
    
    return MethodResult(
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