"""Reporter for visualizing CP-WOPT factor matrices.

This module creates visualizations of the learned factor matrices,
showing spatial patterns, feature loadings, and component structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.core_abstractions import CPFactors
from ..constants import FIGURE_DPI, FIGURE_SIZE, COLORMAP, TOP_N_FEATURES

logger = logging.getLogger(__name__)


class FactorReporter:
    """Creates visualizations of CP-WOPT factor matrices."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """Initialize factor reporter.
        
        Args:
            output_dir: Directory to save plots
            config: Reporting configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.plot_format = config.get('plot_format', 'png')
        self.plot_dpi = config.get('plot_dpi', FIGURE_DPI)
        self.colormap = config.get('colormap', COLORMAP)
        self.n_components_to_plot = config.get('plot_components', 5)
        
    def report(self, factors: CPFactors, tensor_shape: Tuple[int, ...],
               feature_names: Optional[List[str]] = None,
               coordinates: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Path]:
        """Generate factor visualizations.
        
        Args:
            factors: CP factors to visualize
            tensor_shape: Original tensor shape
            feature_names: Names of features
            coordinates: Spatial coordinates (lat, lon)
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plots = {}
        
        # Plot factor matrices heatmaps
        plots['factor_heatmaps'] = self._plot_factor_heatmaps(factors)
        
        # Plot spatial factors if we have coordinates
        if coordinates is not None and len(factors.mode_names) >= 2:
            plots['spatial_patterns'] = self._plot_spatial_patterns(
                factors, coordinates
            )
        
        # Plot feature loadings
        if feature_names is not None:
            plots['feature_loadings'] = self._plot_feature_loadings(
                factors, feature_names
            )
        
        # Plot component weights
        plots['component_weights'] = self._plot_component_weights(factors)
        
        # Plot factor correlations
        # TODO: Fix factor correlations for different mode sizes
        # plots['factor_correlations'] = self._plot_factor_correlations(factors)
        
        return plots
    
    def _plot_factor_heatmaps(self, factors: CPFactors) -> Path:
        """Plot heatmaps of factor matrices.
        
        Args:
            factors: CP factors
            
        Returns:
            Path to saved plot
        """
        n_modes = factors.n_modes
        n_comps = min(self.n_components_to_plot, factors.rank)
        
        fig, axes = plt.subplots(n_modes, 1, figsize=(12, 4 * n_modes))
        if n_modes == 1:
            axes = [axes]
        
        for mode, (mode_name, factor) in enumerate(zip(factors.mode_names, 
                                                      factors.factors)):
            # Select components to plot
            factor_subset = factor[:, :n_comps]
            
            # Create heatmap
            im = axes[mode].imshow(factor_subset.T, aspect='auto', 
                                  cmap=self.colormap)
            axes[mode].set_title(f'{mode_name.capitalize()} Factors')
            axes[mode].set_xlabel(f'{mode_name.capitalize()} Index')
            axes[mode].set_ylabel('Component')
            axes[mode].set_yticks(range(n_comps))
            axes[mode].set_yticklabels([f'C{i+1}' for i in range(n_comps)])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[mode])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'factor_heatmaps.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved factor heatmaps to: {plot_path}")
        return plot_path
    
    def _plot_spatial_patterns(self, factors: CPFactors,
                             coordinates: List[Tuple[float, float]]) -> Path:
        """Plot spatial patterns for each component.
        
        Args:
            factors: CP factors
            coordinates: Spatial coordinates
            
        Returns:
            Path to saved plot
        """
        # Assume first two modes are spatial (lat, lon)
        lat_factors = factors.factors[0]
        lon_factors = factors.factors[1]
        
        n_comps = min(self.n_components_to_plot, factors.rank)
        
        # Create subplots
        fig, axes = plt.subplots(2, n_comps, figsize=(4 * n_comps, 8))
        if n_comps == 1:
            axes = axes.reshape(2, 1)
        
        for comp in range(n_comps):
            # Latitude pattern
            axes[0, comp].plot(lat_factors[:, comp])
            axes[0, comp].set_title(f'Component {comp+1} - Latitude')
            axes[0, comp].set_xlabel('Latitude Index')
            axes[0, comp].set_ylabel('Loading')
            axes[0, comp].grid(True, alpha=0.3)
            
            # Longitude pattern
            axes[1, comp].plot(lon_factors[:, comp])
            axes[1, comp].set_title(f'Component {comp+1} - Longitude')
            axes[1, comp].set_xlabel('Longitude Index')
            axes[1, comp].set_ylabel('Loading')
            axes[1, comp].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'spatial_patterns.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved spatial patterns to: {plot_path}")
        return plot_path
    
    def _plot_feature_loadings(self, factors: CPFactors,
                             feature_names: List[str]) -> Path:
        """Plot top feature loadings for each component.
        
        Args:
            factors: CP factors
            feature_names: Names of features
            
        Returns:
            Path to saved plot
        """
        # Get feature factor (usually last mode)
        feature_factor = factors.factors[-1]
        n_comps = min(self.n_components_to_plot, factors.rank)
        
        fig, axes = plt.subplots(1, n_comps, figsize=(5 * n_comps, 6))
        if n_comps == 1:
            axes = [axes]
        
        for comp in range(n_comps):
            # Get top features for this component
            loadings = feature_factor[:, comp]
            top_idx = np.argsort(np.abs(loadings))[-TOP_N_FEATURES:][::-1]
            
            # Create bar plot
            top_features = [feature_names[i] for i in top_idx]
            top_loadings = loadings[top_idx]
            
            y_pos = np.arange(len(top_features))
            axes[comp].barh(y_pos, top_loadings)
            axes[comp].set_yticks(y_pos)
            axes[comp].set_yticklabels(top_features)
            axes[comp].set_xlabel('Loading')
            axes[comp].set_title(f'Component {comp+1} - Top Features')
            axes[comp].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            
            # Color positive/negative differently
            colors = ['red' if x < 0 else 'blue' for x in top_loadings]
            for i, (bar, color) in enumerate(zip(axes[comp].patches, colors)):
                bar.set_color(color)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'feature_loadings.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature loadings to: {plot_path}")
        return plot_path
    
    def _plot_component_weights(self, factors: CPFactors) -> Path:
        """Plot component importance weights.
        
        Args:
            factors: CP factors
            
        Returns:
            Path to saved plot
        """
        # Calculate component weights as product of factor norms
        weights = np.zeros(factors.rank)
        for r in range(factors.rank):
            weight = 1.0
            for factor in factors.factors:
                weight *= np.linalg.norm(factor[:, r])
            weights[r] = weight
        
        # Normalize
        weights /= weights.sum()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = np.arange(factors.rank) + 1
        bars = ax.bar(components, weights)
        
        # Color top components differently
        n_highlight = min(5, factors.rank)
        top_idx = np.argsort(weights)[-n_highlight:]
        for i in top_idx:
            bars[i].set_color('darkblue')
        
        ax.set_xlabel('Component')
        ax.set_ylabel('Relative Weight')
        ax.set_title('Component Importance Weights')
        ax.set_xticks(components[::max(1, factors.rank // 20)])
        
        # Add cumulative line
        ax2 = ax.twinx()
        cumulative = np.cumsum(weights[np.argsort(weights)[::-1]])
        ax2.plot(components, cumulative, 'r-', linewidth=2, label='Cumulative')
        ax2.set_ylabel('Cumulative Weight', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0, 1.05)
        ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'component_weights.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved component weights to: {plot_path}")
        return plot_path
    
    def _plot_factor_correlations(self, factors: CPFactors) -> Path:
        """Plot correlations between factors across modes.
        
        Args:
            factors: CP factors
            
        Returns:
            Path to saved plot
        """
        n_modes = factors.n_modes
        
        fig, axes = plt.subplots(n_modes, n_modes, 
                                figsize=(4 * n_modes, 4 * n_modes))
        
        for i in range(n_modes):
            for j in range(n_modes):
                ax = axes[i, j] if n_modes > 1 else axes
                
                if i == j:
                    # Plot factor correlation matrix for this mode
                    corr = np.corrcoef(factors.factors[i].T)
                    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
                    ax.set_title(f'{factors.mode_names[i]} Correlations')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    # Plot cross-mode component alignment
                    # This shows how well components align across modes
                    factor_i = factors.factors[i]
                    factor_j = factors.factors[j]
                    
                    # Normalize factors
                    factor_i_norm = factor_i / (np.linalg.norm(factor_i, axis=0) + 1e-10)
                    factor_j_norm = factor_j / (np.linalg.norm(factor_j, axis=0) + 1e-10)
                    
                    # Compute alignment (simplified - full version would use Tucker congruence)
                    alignment = np.abs(factor_i_norm.T @ factor_j_norm.T)
                    
                    im = ax.imshow(alignment, cmap='viridis', vmin=0, vmax=1)
                    ax.set_title(f'{factors.mode_names[i]} vs {factors.mode_names[j]}')
                    ax.set_xlabel(f'{factors.mode_names[j]} Components')
                    ax.set_ylabel(f'{factors.mode_names[i]} Components')
                    
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'factor_correlations.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved factor correlations to: {plot_path}")
        return plot_path