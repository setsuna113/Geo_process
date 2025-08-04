"""Reporter for spatial pattern visualization in CP-WOPT results.

This module creates maps and spatial visualizations of the tensor decomposition
results, showing geographic patterns in biodiversity data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import warnings

# Try to import cartopy for map plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn("Cartopy not available, map plots will be simplified")

from src.abstractions.types.tensor_types import CPFactors, TensorData

logger = logging.getLogger(__name__)


class SpatialReporter:
    """Creates spatial visualizations of CP-WOPT results."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """Initialize spatial reporter.
        
        Args:
            output_dir: Directory to save plots
            config: Reporting configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.plot_format = config.get('plot_format', 'png')
        self.plot_dpi = config.get('plot_dpi', 150)
        self.colormap = config.get('colormap', 'viridis')
        self.n_components_to_plot = config.get('plot_components', 5)
        
    def report(self, factors: CPFactors,
               tensor_data: TensorData,
               coordinates: List[Tuple[float, float]],
               lat_bins: np.ndarray,
               lon_bins: np.ndarray) -> Dict[str, Path]:
        """Generate spatial visualizations.
        
        Args:
            factors: CP factors
            tensor_data: Original tensor data
            coordinates: Original data coordinates
            lat_bins: Latitude bin edges
            lon_bins: Longitude bin edges
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plots = {}
        
        # Plot spatial factors on maps
        if HAS_CARTOPY:
            plots['spatial_factors_maps'] = self._plot_spatial_factors_maps(
                factors, lat_bins, lon_bins
            )
        else:
            logger.warning("Cartopy not available, skipping map plots")
        
        # Plot spatial factor grids
        plots['spatial_factor_grids'] = self._plot_spatial_factor_grids(
            factors, lat_bins, lon_bins
        )
        
        # Plot data coverage map
        plots['data_coverage'] = self._plot_data_coverage(
            tensor_data, lat_bins, lon_bins
        )
        
        # Plot component activation maps
        plots['component_activations'] = self._plot_component_activations(
            factors, tensor_data, lat_bins, lon_bins
        )
        
        # Plot spatial autocorrelation
        plots['spatial_autocorrelation'] = self._plot_spatial_autocorrelation(
            factors
        )
        
        return plots
    
    def _plot_spatial_factors_maps(self, factors: CPFactors,
                                  lat_bins: np.ndarray,
                                  lon_bins: np.ndarray) -> Path:
        """Plot spatial factors on geographic maps using cartopy.
        
        Args:
            factors: CP factors
            lat_bins: Latitude bin edges
            lon_bins: Longitude bin edges
            
        Returns:
            Path to saved plot
        """
        # Get spatial factors (assuming first two modes are lat/lon)
        lat_factors = factors.factors[0]
        lon_factors = factors.factors[1]
        
        n_comps = min(self.n_components_to_plot, factors.rank)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(5 * n_comps, 8))
        
        for comp in range(n_comps):
            # Create geographic projection
            ax = fig.add_subplot(2, n_comps, comp + 1, 
                               projection=ccrs.PlateCarree())
            
            # Create 2D grid from outer product of spatial factors
            spatial_pattern = np.outer(lat_factors[:, comp], lon_factors[:, comp])
            
            # Get coordinate grids
            lon_grid, lat_grid = np.meshgrid(
                (lon_bins[:-1] + lon_bins[1:]) / 2,
                (lat_bins[:-1] + lat_bins[1:]) / 2
            )
            
            # Plot the pattern
            im = ax.pcolormesh(lon_grid, lat_grid, spatial_pattern,
                              cmap=self.colormap, 
                              transform=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, alpha=0.5)
            ax.add_feature(cfeature.BORDERS, alpha=0.3)
            ax.gridlines(draw_labels=True, alpha=0.3)
            
            ax.set_title(f'Component {comp + 1}')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                               pad=0.05, shrink=0.8)
            cbar.set_label('Factor Loading')
        
        # Add overall title
        fig.suptitle('Spatial Patterns by Component', fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'spatial_factors_maps.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved spatial factor maps to: {plot_path}")
        return plot_path
    
    def _plot_spatial_factor_grids(self, factors: CPFactors,
                                  lat_bins: np.ndarray,
                                  lon_bins: np.ndarray) -> Path:
        """Plot spatial factors as simple grids (no map).
        
        Args:
            factors: CP factors
            lat_bins: Latitude bin edges
            lon_bins: Longitude bin edges
            
        Returns:
            Path to saved plot
        """
        lat_factors = factors.factors[0]
        lon_factors = factors.factors[1]
        
        n_comps = min(self.n_components_to_plot, factors.rank)
        
        # Create subplots
        fig, axes = plt.subplots(2, n_comps, figsize=(4 * n_comps, 8))
        if n_comps == 1:
            axes = axes.reshape(2, 1)
        
        for comp in range(n_comps):
            # Plot latitude factor
            lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
            axes[0, comp].plot(lat_centers, lat_factors[:, comp], 'b-', linewidth=2)
            axes[0, comp].fill_between(lat_centers, lat_factors[:, comp], 
                                      alpha=0.3, color='blue')
            axes[0, comp].set_xlabel('Latitude')
            axes[0, comp].set_ylabel('Loading')
            axes[0, comp].set_title(f'Component {comp + 1} - Latitude')
            axes[0, comp].grid(True, alpha=0.3)
            axes[0, comp].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            
            # Plot longitude factor  
            lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
            axes[1, comp].plot(lon_centers, lon_factors[:, comp], 'r-', linewidth=2)
            axes[1, comp].fill_between(lon_centers, lon_factors[:, comp],
                                      alpha=0.3, color='red')
            axes[1, comp].set_xlabel('Longitude')
            axes[1, comp].set_ylabel('Loading')
            axes[1, comp].set_title(f'Component {comp + 1} - Longitude')
            axes[1, comp].grid(True, alpha=0.3)
            axes[1, comp].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'spatial_factor_grids.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved spatial factor grids to: {plot_path}")
        return plot_path
    
    def _plot_data_coverage(self, tensor_data: TensorData,
                          lat_bins: np.ndarray,
                          lon_bins: np.ndarray) -> Path:
        """Plot data coverage map.
        
        Args:
            tensor_data: Tensor data
            lat_bins: Latitude bin edges
            lon_bins: Longitude bin edges
            
        Returns:
            Path to saved plot
        """
        # Calculate coverage (proportion of non-missing data)
        coverage = np.mean(tensor_data.weights > 0, axis=2)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get coordinate grids
        lon_grid, lat_grid = np.meshgrid(
            (lon_bins[:-1] + lon_bins[1:]) / 2,
            (lat_bins[:-1] + lat_bins[1:]) / 2
        )
        
        # Plot coverage
        im = ax.pcolormesh(lon_grid, lat_grid, coverage,
                          cmap='YlOrRd', vmin=0, vmax=1)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Data Coverage (Proportion of Non-Missing Features)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        total_coverage = np.mean(tensor_data.weights > 0)
        ax.text(0.02, 0.98, f'Overall Coverage: {total_coverage:.1%}',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'data_coverage.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved data coverage map to: {plot_path}")
        return plot_path
    
    def _plot_component_activations(self, factors: CPFactors,
                                   tensor_data: TensorData,
                                   lat_bins: np.ndarray,
                                   lon_bins: np.ndarray) -> Path:
        """Plot component activation strength across space.
        
        Args:
            factors: CP factors
            tensor_data: Tensor data
            lat_bins: Latitude bin edges
            lon_bins: Longitude bin edges
            
        Returns:
            Path to saved plot
        """
        n_comps = min(self.n_components_to_plot, factors.rank)
        
        # Calculate activation strength for each component
        # This is simplified - full version would use Tucker3 core
        lat_factors = factors.factors[0]
        lon_factors = factors.factors[1]
        
        fig, axes = plt.subplots(2, (n_comps + 1) // 2, 
                                figsize=(6 * ((n_comps + 1) // 2), 10))
        axes = axes.flatten()
        
        # Get coordinate grids
        lon_grid, lat_grid = np.meshgrid(
            (lon_bins[:-1] + lon_bins[1:]) / 2,
            (lat_bins[:-1] + lat_bins[1:]) / 2
        )
        
        for comp in range(n_comps):
            ax = axes[comp]
            
            # Calculate activation as outer product weighted by component importance
            activation = np.outer(np.abs(lat_factors[:, comp]), 
                                np.abs(lon_factors[:, comp]))
            
            # Normalize
            activation = activation / (np.max(activation) + 1e-10)
            
            # Plot
            im = ax.pcolormesh(lon_grid, lat_grid, activation,
                              cmap='hot', vmin=0, vmax=1)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'Component {comp + 1} Activation Strength')
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(im, ax=ax)
        
        # Remove extra subplots
        for i in range(n_comps, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'component_activations.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved component activations to: {plot_path}")
        return plot_path
    
    def _plot_spatial_autocorrelation(self, factors: CPFactors) -> Path:
        """Plot spatial autocorrelation of factors.
        
        Args:
            factors: CP factors
            
        Returns:
            Path to saved plot
        """
        # Calculate spatial autocorrelation for each component
        lat_factors = factors.factors[0]
        lon_factors = factors.factors[1]
        
        n_comps = min(self.n_components_to_plot, factors.rank)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Latitude autocorrelation
        max_lag = min(20, lat_factors.shape[0] // 4)
        lags = np.arange(1, max_lag + 1)
        
        lat_autocorr = np.zeros((n_comps, max_lag))
        for comp in range(n_comps):
            factor = lat_factors[:, comp]
            for lag in range(max_lag):
                if lag == 0:
                    lat_autocorr[comp, lag] = 1.0
                else:
                    lat_autocorr[comp, lag] = np.corrcoef(
                        factor[:-lag-1], factor[lag+1:]
                    )[0, 1]
        
        # Plot latitude autocorrelation
        for comp in range(n_comps):
            axes[0, 0].plot(lags, lat_autocorr[comp, :], 
                           label=f'Component {comp + 1}')
        axes[0, 0].set_xlabel('Lag')
        axes[0, 0].set_ylabel('Autocorrelation')
        axes[0, 0].set_title('Latitude Factor Autocorrelation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Longitude autocorrelation
        lon_autocorr = np.zeros((n_comps, max_lag))
        for comp in range(n_comps):
            factor = lon_factors[:, comp]
            for lag in range(max_lag):
                if lag == 0:
                    lon_autocorr[comp, lag] = 1.0
                else:
                    lon_autocorr[comp, lag] = np.corrcoef(
                        factor[:-lag-1], factor[lag+1:]
                    )[0, 1]
        
        # Plot longitude autocorrelation
        for comp in range(n_comps):
            axes[0, 1].plot(lags, lon_autocorr[comp, :],
                           label=f'Component {comp + 1}')
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('Autocorrelation')
        axes[0, 1].set_title('Longitude Factor Autocorrelation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Spatial smoothness (first differences)
        lat_smoothness = []
        lon_smoothness = []
        
        for comp in range(factors.rank):
            # Calculate roughness as variance of first differences
            lat_diff = np.diff(lat_factors[:, comp])
            lon_diff = np.diff(lon_factors[:, comp])
            
            lat_smoothness.append(np.var(lat_diff))
            lon_smoothness.append(np.var(lon_diff))
        
        # Plot smoothness by component
        components = np.arange(factors.rank) + 1
        axes[1, 0].bar(components[:n_comps], lat_smoothness[:n_comps], 
                      alpha=0.7, color='blue')
        axes[1, 0].set_xlabel('Component')
        axes[1, 0].set_ylabel('Roughness (Var of Differences)')
        axes[1, 0].set_title('Latitude Factor Roughness')
        axes[1, 0].set_yscale('log')
        
        axes[1, 1].bar(components[:n_comps], lon_smoothness[:n_comps],
                      alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Component')
        axes[1, 1].set_ylabel('Roughness (Var of Differences)')
        axes[1, 1].set_title('Longitude Factor Roughness')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'spatial_autocorrelation.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved spatial autocorrelation to: {plot_path}")
        return plot_path