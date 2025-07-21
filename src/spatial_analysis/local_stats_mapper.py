# src/spatial_analysis/gwpca/local_stats_mapper.py
# type: ignore
"""Create maps and visualizations for GWPCA local statistics."""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import xarray as xr

from src.spatial_analysis.base_analyzer import AnalysisResult

logger = logging.getLogger(__name__)

class LocalStatsMapper:
    """Create maps and visualizations for GWPCA results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
    
    def plot_local_r2_map(self, result: AnalysisResult,
                         save_path: Optional[str] = None) -> Figure:
        """
        Plot map of local R² values.
        
        Args:
            result: GWPCA analysis result
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get local R² map
        r2_map = result.spatial_output
        
        # Plot with appropriate colormap
        im = ax.imshow(r2_map.values, cmap='YlOrRd', aspect='auto',
                      vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Local R²')
        
        # Labels
        ax.set_title('Local R² - Spatial Variation in PCA Performance', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add statistics as text
        stats = result.statistics
        text = (f"Mean: {stats['local_r2_mean']:.3f}\n"
               f"Std: {stats['local_r2_std']:.3f}\n"
               f"Range: [{stats['local_r2_min']:.3f}, {stats['local_r2_max']:.3f}]")
        
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_component_loadings(self, result: AnalysisResult, component: int,
                        save_path: Optional[str] = None) -> Figure:
        """
        Plot loading maps for a specific component.
        
        Args:
            result: GWPCA analysis result
            component: Which PC to plot (1-indexed)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        band_names = result.metadata.input_bands
        n_bands = len(band_names)
        
        # Create subplot grid
        ncols = min(3, n_bands)
        nrows = (n_bands + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if n_bands == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 else axes
        
        # Plot each variable's loading
        for idx, band in enumerate(band_names[:n_bands]):
            ax = axes[idx] if n_bands > 1 else axes
            
            # Get loading map
            loading_key = f'loading_{band}_pc{component}'
            if loading_key in result.additional_outputs:
                loading_map = result.additional_outputs[loading_key]
                
                # Use diverging colormap centered at 0
                norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
                im = ax.imshow(loading_map.values, cmap='RdBu_r', 
                             norm=norm, aspect='auto')
                
                plt.colorbar(im, ax=ax, label='Loading')
                
                ax.set_title(f'{band} Loading on PC{component}', 
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
        
        # Hide extra subplots
        if n_bands > 1:
            for idx in range(n_bands, len(axes)):
                axes[idx].set_visible(False)
        
        plt.suptitle(f'Local PCA Loadings - Component {component}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_eigenvalue_maps(self, result: AnalysisResult,
                           save_path: Optional[str] = None) -> Figure:
        """
        Plot eigenvalue maps for all components.
        
        Args:
            result: GWPCA analysis result
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_components = result.metadata.parameters['n_components']
        
        fig, axes = plt.subplots(1, n_components, figsize=(5*n_components, 4))
        if n_components == 1:
            axes = [axes]
        
        for i in range(n_components):
            ax = axes[i]
            
            # Get eigenvalue map
            eigenval_key = f'eigenvalue_pc{i+1}'
            if eigenval_key in result.additional_outputs:
                eigenval_map = result.additional_outputs[eigenval_key]
                
                im = ax.imshow(eigenval_map.values, cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=ax, label='Eigenvalue')
                
                ax.set_title(f'PC{i+1} Eigenvalue', fontsize=12, fontweight='bold')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
        
        plt.suptitle('Local Eigenvalue Maps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_figure(self, result: AnalysisResult,
                            save_path: Optional[str] = None) -> Figure:
        """Create comprehensive summary figure."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Return the figure (placeholder for now)
        return fig