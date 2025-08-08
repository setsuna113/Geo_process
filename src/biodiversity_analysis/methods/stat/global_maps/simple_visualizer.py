"""
Simple Biodiversity Visualizer
===============================

Simple visualizations for biodiversity data without complex dependencies.
Uses only matplotlib and numpy for maximum compatibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from pathlib import Path
import rasterio
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


class SimpleVisualizer:
    """
    Create simple, interpretable visualizations for biodiversity data.
    No cartopy dependency - uses matplotlib directly.
    """
    
    def __init__(self, output_dir: str = "outputs/biodiversity_analysis"):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        output_dir : str
            Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_data_coverage(self, file_path: str, title: str = None, 
                          save_name: str = None) -> plt.Figure:
        """
        Create a simple presence/absence map showing data coverage.
        
        Parameters
        ----------
        file_path : str
            Path to GeoTIFF file
        title : str
            Title for the plot
        save_name : str
            Name for saving the figure
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        with rasterio.open(file_path) as src:
            data = src.read(1)
            bounds = src.bounds
            
            # Create binary coverage mask
            if src.nodata is not None:
                coverage = (data != src.nodata).astype(float)
                valid_data = data[data != src.nodata]
            else:
                coverage = (~np.isnan(data)).astype(float)
                valid_data = data[~np.isnan(data)]
                
            # Mark zero values differently (0.5)
            coverage[(data == 0) & (coverage == 1)] = 0.5
            
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left panel: Coverage map
        # Custom colormap: white=no data, yellow=zero, green=data present
        colors = ['white', 'gold', 'darkgreen']
        n_bins = 3
        cmap = mcolors.LinearSegmentedColormap.from_list('coverage', colors, N=n_bins)
        
        im = ax1.imshow(coverage, cmap=cmap, vmin=0, vmax=1, 
                       extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                       aspect='auto', interpolation='nearest')
        
        # Add colorbar with custom labels
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['No Data', 'Zero Values', 'Data Present'])
        cbar.set_label('Coverage Status', rotation=270, labelpad=20)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Title
        if title is None:
            title = f"Data Coverage: {Path(file_path).stem}"
        ax1.set_title(title, fontsize=12, fontweight='bold')
        
        # Add coverage statistics
        valid_percent = np.sum(coverage > 0) / coverage.size * 100
        zero_percent = np.sum(coverage == 0.5) / coverage.size * 100
        data_percent = np.sum(coverage == 1) / coverage.size * 100
        
        stats_text = f"Coverage Statistics:\n"
        stats_text += f"  Total coverage: {valid_percent:.1f}%\n"
        stats_text += f"  Data present: {data_percent:.1f}%\n"
        stats_text += f"  Zero values: {zero_percent:.1f}%\n"
        stats_text += f"  No data: {100-valid_percent:.1f}%"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               fontfamily='monospace')
        
        # Right panel: Histogram of values
        ax2.hist(valid_data[valid_data > 0], bins=50, color='steelblue', 
                alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Species Richness (excluding zeros)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Non-Zero Values')
        ax2.grid(True, alpha=0.3)
        
        if len(valid_data[valid_data > 0]) > 0:
            ax2.axvline(np.mean(valid_data[valid_data > 0]), color='red', 
                       linestyle='--', label=f'Mean: {np.mean(valid_data[valid_data > 0]):.1f}')
            ax2.axvline(np.median(valid_data[valid_data > 0]), color='green', 
                       linestyle='--', label=f'Median: {np.median(valid_data[valid_data > 0]):.1f}')
            ax2.legend()
        
        plt.tight_layout()
        
        # Save if requested
        if save_name:
            save_path = self.output_dir / f"{save_name}_coverage.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {save_path}")
            
        return fig
        
    def plot_richness_gradient(self, file_path: str, data_type: str = 'observed_counts',
                              title: str = None, save_name: str = None,
                              quantile_clip: float = 0.99) -> plt.Figure:
        """
        Create a richness gradient heatmap with analysis panels.
        
        Parameters
        ----------
        file_path : str
            Path to GeoTIFF file
        data_type : str
            Either 'observed_counts' or 'sdm_prediction'
        title : str
            Title for the plot
        save_name : str
            Name for saving the figure
        quantile_clip : float
            Clip values at this quantile for better visualization
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        with rasterio.open(file_path) as src:
            data = src.read(1)
            bounds = src.bounds
            
            # Handle nodata
            if src.nodata is not None:
                mask = (data == src.nodata)
                data_clean = np.ma.masked_array(data, mask=mask)
            else:
                data_clean = np.ma.masked_invalid(data)
                
        # Get valid data for statistics
        valid_data = data_clean.compressed()
        
        # Clip extreme values for better visualization
        if quantile_clip and len(valid_data) > 0:
            vmax = np.percentile(valid_data, quantile_clip * 100)
            data_display = np.ma.clip(data_clean, 0, vmax)
        else:
            data_display = data_clean
            
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1], width_ratios=[3, 1, 1],
                     hspace=0.3, wspace=0.3)
        
        # Main richness map
        ax_map = fig.add_subplot(gs[0, :2])
        
        # Choose colormap based on data type
        if data_type == 'sdm_prediction':
            cmap = 'viridis'  # SDM predictions
            cbar_label = 'Predicted Species Richness'
            method_text = "⚠️ SDM PREDICTION\n(Potential habitat)"
        else:
            cmap = 'YlOrRd'  # Observed counts
            cbar_label = 'Observed Species Richness'
            method_text = "✓ OBSERVED COUNTS\n(Actual occurrences)"
            
        im = ax_map.imshow(data_display, cmap=cmap,
                          extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                          aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_map, shrink=0.8, pad=0.02)
        cbar.set_label(cbar_label, rotation=270, labelpad=20)
        
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        ax_map.grid(True, alpha=0.3, linestyle='--')
        
        # Title
        if title is None:
            title = f"Species Richness: {Path(file_path).stem}"
        ax_map.set_title(title, fontsize=14, fontweight='bold')
        
        # Latitudinal gradient
        ax_lat = fig.add_subplot(gs[0, 2])
        
        if not data_clean.mask.all():
            # Calculate mean richness by latitude
            lat_means = np.nanmean(data_clean, axis=1)
            latitudes = np.linspace(bounds.bottom, bounds.top, len(lat_means))
            
            ax_lat.plot(lat_means, latitudes, 'b-', linewidth=2)
            ax_lat.fill_betweenx(latitudes, 0, lat_means, alpha=0.3)
            ax_lat.set_ylabel('Latitude')
            ax_lat.set_xlabel('Mean\nRichness')
            ax_lat.set_title('Latitudinal\nGradient')
            ax_lat.grid(True, alpha=0.3)
            ax_lat.set_ylim(bounds.bottom, bounds.top)
            
        # Histogram
        ax_hist = fig.add_subplot(gs[1, :2])
        
        if len(valid_data) > 0:
            ax_hist.hist(valid_data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax_hist.axvline(np.mean(valid_data), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(valid_data):.1f}', linewidth=2)
            ax_hist.axvline(np.median(valid_data), color='green', linestyle='--',
                          label=f'Median: {np.median(valid_data):.1f}', linewidth=2)
            ax_hist.set_xlabel('Species Richness')
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title('Distribution of Richness Values')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            
        # Summary statistics box
        ax_stats = fig.add_subplot(gs[1:, 2])
        ax_stats.axis('off')
        
        if len(valid_data) > 0:
            stats_text = "SUMMARY STATISTICS\n" + "="*25 + "\n\n"
            stats_text += f"Min:      {np.min(valid_data):>10.1f}\n"
            stats_text += f"Max:      {np.max(valid_data):>10.1f}\n"
            stats_text += f"Mean:     {np.mean(valid_data):>10.1f}\n"
            stats_text += f"Median:   {np.median(valid_data):>10.1f}\n"
            stats_text += f"Std Dev:  {np.std(valid_data):>10.1f}\n\n"
            stats_text += f"Quantiles:\n"
            stats_text += f"  5%:     {np.percentile(valid_data, 5):>10.1f}\n"
            stats_text += f" 25%:     {np.percentile(valid_data, 25):>10.1f}\n"
            stats_text += f" 75%:     {np.percentile(valid_data, 75):>10.1f}\n"
            stats_text += f" 95%:     {np.percentile(valid_data, 95):>10.1f}\n\n"
            stats_text += f"Coverage: {(len(valid_data)/data.size)*100:>9.1f}%\n"
            stats_text += f"N cells:  {len(valid_data):>10,}\n\n"
            stats_text += "="*25 + "\n"
            stats_text += method_text
                
            ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # Longitudinal gradient
        ax_lon = fig.add_subplot(gs[2, :2])
        
        if not data_clean.mask.all():
            # Calculate mean richness by longitude
            lon_means = np.nanmean(data_clean, axis=0)
            longitudes = np.linspace(bounds.left, bounds.right, len(lon_means))
            
            ax_lon.plot(longitudes, lon_means, 'g-', linewidth=2)
            ax_lon.fill_between(longitudes, 0, lon_means, alpha=0.3, color='green')
            ax_lon.set_xlabel('Longitude')
            ax_lon.set_ylabel('Mean Richness')
            ax_lon.set_title('Longitudinal Gradient')
            ax_lon.grid(True, alpha=0.3)
            ax_lon.set_xlim(bounds.left, bounds.right)
        
        plt.suptitle(f"Biodiversity Analysis: {Path(file_path).stem}", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save if requested
        if save_name:
            save_path = self.output_dir / f"{save_name}_richness.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"    Saved: {save_path}")
            
        return fig
        
    def plot_data_comparison(self, datasets: Dict[str, str], 
                           save_name: str = "data_comparison") -> plt.Figure:
        """
        Create a grid comparison of multiple datasets.
        
        Parameters
        ----------
        datasets : Dict[str, str]
            Dictionary mapping dataset names to file paths
        save_name : str
            Name for saving the figure
            
        Returns
        -------
        plt.Figure
            The created comparison figure
        """
        n_datasets = len(datasets)
        
        # Determine grid layout
        if n_datasets <= 2:
            rows, cols = 1, n_datasets
        elif n_datasets <= 4:
            rows, cols = 2, 2
        else:
            rows = int(np.ceil(n_datasets / 2))
            cols = 2
            
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
        
        # Flatten axes for easier iteration
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for idx, (name, file_path) in enumerate(datasets.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            with rasterio.open(file_path) as src:
                data = src.read(1)
                bounds = src.bounds
                
                # Handle nodata
                if src.nodata is not None:
                    data = np.ma.masked_equal(data, src.nodata)
                else:
                    data = np.ma.masked_invalid(data)
                    
            # Clip extreme values
            if not data.mask.all():
                valid_data = data.compressed()
                if len(valid_data) > 0:
                    vmax = np.percentile(valid_data, 99)
                    data = np.ma.clip(data, 0, vmax)
                
            # Choose colormap based on name
            if 'fungi' in name.lower():
                cmap = 'viridis'
                data_type = "SDM Prediction"
            else:
                cmap = 'YlOrRd'
                data_type = "Observed Counts"
                
            im = ax.imshow(data, cmap=cmap,
                         extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                         aspect='auto', interpolation='nearest')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Richness', rotation=270, labelpad=15)
            
            # Title with data type indicator
            ax.set_title(f"{name.replace('_', ' ').title()}\n({data_type})", 
                        fontsize=11, fontweight='bold')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add basic stats
            if not data.mask.all() and len(data.compressed()) > 0:
                stats_text = f"Mean: {np.mean(data.compressed()):.1f}\n"
                stats_text += f"Max: {np.max(data.compressed()):.0f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused axes
        for idx in range(n_datasets, len(axes)):
            axes[idx].axis('off')
            
        plt.suptitle("Biodiversity Dataset Comparison", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / f"{save_name}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {save_path}")
        
        return fig