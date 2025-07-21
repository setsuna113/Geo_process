# src/spatial_analysis/gwpca/local_stats_mapper.py
"""Create maps and visualizations for GWPCA local statistics."""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import xarray as xr

from src.spatial_analysis.base_analyzer import AnalysisResult

logger = logging.getLogger(__name__)

class LocalStatsMapper:
    """Create maps and visualizations for GWPCA results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
    
    def plot_local_r2_map(self, result: AnalysisResult,
                         save_path: Optional[str] = None,
                         show_blocks: bool = True) -> plt.Figure:
        """
        Plot map of local R² values.
        
        Args:
            result: GWPCA analysis result
            save_path: Optional path to save figure
            show_blocks: Whether to show block boundaries (if applicable)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get local R² map
        r2_map = result.spatial_output
        
        # Check if block-based analysis
        is_block_based = 'block_size_km' in result.metadata.parameters and \
                        result.metadata.parameters.get('use_block_aggregation', False)
        
        # Plot with appropriate colormap
        if is_block_based:
            # Use nearest neighbor interpolation for blocks
            im = ax.imshow(r2_map.values, cmap='YlOrRd', aspect='auto',
                         vmin=0, vmax=1, interpolation='nearest')
            
            # Add block boundaries if requested
            if show_blocks:
                height, width = r2_map.shape
                for i in range(height + 1):
                    ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
                for j in range(width + 1):
                    ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        else:
            # Smooth interpolation for pixel-based
            im = ax.imshow(r2_map.values, cmap='YlOrRd', aspect='auto',
                         vmin=0, vmax=1, interpolation='bilinear')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Local R²')
        
        # Labels
        scale_text = f" ({result.statistics.get('analysis_scale', 'pixels')})" if is_block_based else ""
        ax.set_title(f'Local R² - Spatial Variation in PCA Performance{scale_text}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add statistics as text
        stats = result.statistics
        text = (f"Mean: {stats['local_r2_mean']:.3f}\n"
               f"Std: {stats['local_r2_std']:.3f}\n"
               f"Range: [{stats['local_r2_min']:.3f}, {stats['local_r2_max']:.3f}]")
        
        if is_block_based:
            text += f"\nBlocks: {stats['n_blocks_analyzed']}"
            text += f"\nReduction: {stats['data_reduction_factor']:.0f}×"
        
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_block_analysis_summary(self, result: AnalysisResult,
                                  save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Create summary figure specifically for block-based analysis.
        
        Args:
            result: GWPCA analysis result
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure or None if not block-based
        """
        # Check if this is block-based analysis
        if not result.metadata.parameters.get('use_block_aggregation', False):
            return None
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Local R² map with block visualization
        ax1 = fig.add_subplot(gs[0, :2])
        r2_map = result.spatial_output
        im = ax1.imshow(r2_map.values, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=1, interpolation='nearest')
        
        # Add block grid
        height, width = r2_map.shape
        for i in range(height + 1):
            ax1.axhline(i - 0.5, color='black', linewidth=1, alpha=0.3)
        for j in range(width + 1):
            ax1.axvline(j - 0.5, color='black', linewidth=1, alpha=0.3)
        
        plt.colorbar(im, ax=ax1, label='Local R²')
        ax1.set_title(f'Local R² ({result.metadata.parameters["block_size_km"]}km blocks)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Block Column')
        ax1.set_ylabel('Block Row')
        
        # 2. Block pixel counts (if available)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'block_pixel_counts' in result.additional_outputs:
            counts = result.additional_outputs['block_pixel_counts']
            im2 = ax2.imshow(counts.values, cmap='Blues', aspect='auto')
            plt.colorbar(im2, ax=ax2, label='Pixels per Block')
            ax2.set_title('Data Density', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No density data', ha='center', va='center',
                    transform=ax2.transAxes)
            ax2.set_title('Data Density', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Block Column')
        ax2.set_ylabel('Block Row')
        
        # 3. R² distribution comparison
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(result.labels, bins=30, density=True, alpha=0.7,
                edgecolor='black', label='Block R²')
        ax3.axvline(result.statistics['local_r2_mean'], color='red',
                   linestyle='--', label='Mean')
        
        # Add global R² if available
        if 'global_variance_explained' in result.statistics:
            global_r2 = result.statistics['global_variance_explained'][0]
            ax3.axvline(global_r2, color='blue', linestyle='--',
                       label=f'Global R² ({global_r2:.3f})')
        
        ax3.set_xlabel('Local R²')
        ax3.set_ylabel('Density')
        ax3.set_title('R² Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        
        # 4. Spatial heterogeneity assessment
        ax4 = fig.add_subplot(gs[1, 1])
        # Create a simple heterogeneity map by calculating local variance
        r2_values = r2_map.values
        heterogeneity = np.zeros_like(r2_values)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                # Calculate variance in 3x3 neighborhood
                neighborhood = r2_values[max(0,i-1):i+2, max(0,j-1):j+2]
                heterogeneity[i, j] = np.var(neighborhood)
        
        im4 = ax4.imshow(heterogeneity, cmap='hot', aspect='auto')
        plt.colorbar(im4, ax=ax4, label='Local Variance')
        ax4.set_title('Spatial Heterogeneity', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Block Column')
        ax4.set_ylabel('Block Row')
        
        # 5. Summary statistics
        ax5 = fig.add_subplot(gs[1, 2])
        stats_text = (
            f"Analysis Summary\n"
            f"{'='*20}\n"
            f"Block Size: {result.metadata.parameters['block_size_km']} km\n"
            f"Total Blocks: {result.statistics['n_blocks_analyzed']}\n"
            f"Original Pixels: {result.statistics['original_pixels']:,}\n"
            f"Reduction Factor: {result.statistics['data_reduction_factor']:.0f}×\n"
            f"\nBandwidth: {result.statistics['bandwidth']:.1f}\n"
            f"Kernel: {result.metadata.parameters['kernel']}\n"
            f"Adaptive: {result.metadata.parameters['adaptive']}\n"
            f"\nSpatial Heterogeneity: {result.statistics['spatial_heterogeneity']:.3f}"
        )
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax5.axis('off')
        
        plt.suptitle('Block-based GWPCA Analysis Summary', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_figure(self, result: AnalysisResult,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive summary figure."""
        # Check if block-based and use appropriate visualization
        if result.metadata.parameters.get('use_block_aggregation', False):
            return self.plot_block_analysis_summary(result, save_path)
        
        # Original implementation for pixel-based analysis
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Local R² map (large)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        r2_map = result.spatial_output
        im1 = ax1.imshow(r2_map.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax1.set_title('Local R² Map', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        plt.colorbar(im1, ax=ax1, label='Local R²')
        
        # 2. R² histogram
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(result.labels, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(result.statistics['local_r2_mean'], 
                   color='red', linestyle='--', label='Mean')
        ax2.set_xlabel('Local R²')
        ax2.set_ylabel('Frequency')
        ax2.set_title('R² Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        
        # 3. Global vs Local comparison
        ax3 = fig.add_subplot(gs[1, 2])
        global_var = result.statistics['global_variance_explained']
        local_mean = result.statistics['local_r2_mean']
        
        categories = ['Global\nPC1', 'Local\n(mean)']
        values = [global_var[0] if isinstance(global_var, list) else global_var, 
                 local_mean]
        
        bars = ax3.bar(categories, values, color=['blue', 'orange'])
        ax3.set_ylabel('Variance Explained')
        ax3.set_title('Global vs Local', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 4. PC1 loading variation
        ax4 = fig.add_subplot(gs[2, :])
        band_names = result.metadata.input_bands
        loading_var = result.statistics['loading_variation']
        
        if len(loading_var) > 0 and len(loading_var[0]) > 0:
            # First component loadings
            pc1_var = [var[0] for var in loading_var[:len(band_names)]]
            
            bars = ax4.bar(band_names, pc1_var)
            ax4.set_xlabel('Variable')
            ax4.set_ylabel('Loading Std Dev')
            ax4.set_title('Spatial Variation in PC1 Loadings', 
                         fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
        
        # Add summary text
        summary_text = (
            f"Bandwidth: {result.statistics['bandwidth']:.2f}\n"
            f"Spatial Heterogeneity: {result.statistics['spatial_heterogeneity']:.3f}\n"
            f"Kernel: {result.metadata.parameters['kernel']}\n"
            f"Adaptive: {result.metadata.parameters['adaptive']}"
        )
        
        fig.text(0.98, 0.02, summary_text, transform=fig.transFigure,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('GWPCA Analysis Summary', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig