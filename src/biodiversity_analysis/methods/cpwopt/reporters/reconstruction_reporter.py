"""Reporter for tensor reconstruction quality visualization.

This module creates visualizations showing how well the CP decomposition
reconstructs the original tensor data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from src.core_abstractions import TensorMetrics, CPFactors

logger = logging.getLogger(__name__)


class ReconstructionReporter:
    """Creates visualizations of tensor reconstruction quality."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """Initialize reconstruction reporter.
        
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
        
    def report(self, original_tensor: np.ndarray,
               reconstructed_tensor: np.ndarray,
               weights: np.ndarray,
               metrics: TensorMetrics,
               slice_indices: Optional[Dict[str, int]] = None) -> Dict[str, Path]:
        """Generate reconstruction quality visualizations.
        
        Args:
            original_tensor: Original tensor data
            reconstructed_tensor: Reconstructed tensor
            weights: Weight mask
            metrics: Tensor metrics
            slice_indices: Specific slices to visualize
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        # Convert CuPy arrays to NumPy if needed (early conversion)
        if hasattr(original_tensor, 'get'):
            original_tensor = original_tensor.get()
        if hasattr(reconstructed_tensor, 'get'):
            reconstructed_tensor = reconstructed_tensor.get()
        if hasattr(weights, 'get'):
            weights = weights.get()
            
        plots = {}
        
        # Plot convergence history
        if len(metrics.convergence_history) > 0:
            plots['convergence'] = self._plot_convergence(metrics.convergence_history)
        
        # Plot reconstruction vs original slices
        plots['reconstruction_slices'] = self._plot_reconstruction_slices(
            original_tensor, reconstructed_tensor, weights, slice_indices
        )
        
        # Plot residuals
        plots['residuals'] = self._plot_residuals(
            original_tensor, reconstructed_tensor, weights
        )
        
        # Plot reconstruction error distribution
        plots['error_distribution'] = self._plot_error_distribution(
            original_tensor, reconstructed_tensor, weights
        )
        
        # Plot fit by data density
        plots['fit_by_density'] = self._plot_fit_by_density(
            original_tensor, reconstructed_tensor, weights
        )
        
        return plots
    
    def _plot_convergence(self, convergence_history: List[float]) -> Path:
        """Plot convergence history.
        
        Args:
            convergence_history: List of errors by iteration
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        iterations = np.arange(len(convergence_history))
        
        # Plot absolute error
        ax1.plot(iterations, convergence_history, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Reconstruction Error')
        ax1.set_title('Convergence History')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot relative improvement
        if len(convergence_history) > 1:
            improvements = []
            for i in range(1, len(convergence_history)):
                if convergence_history[i-1] > 0:
                    rel_imp = abs(convergence_history[i] - convergence_history[i-1]) / convergence_history[i-1]
                    improvements.append(rel_imp)
                else:
                    improvements.append(0)
            
            ax2.plot(iterations[1:], improvements, 'r-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Relative Improvement')
            ax2.set_title('Convergence Rate')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            ax2.axhline(y=1e-6, color='k', linestyle='--', alpha=0.5, 
                       label='Convergence Threshold')
            ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'convergence.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved convergence plot to: {plot_path}")
        return plot_path
    
    def _plot_reconstruction_slices(self, original: np.ndarray,
                                  reconstructed: np.ndarray,
                                  weights: np.ndarray,
                                  slice_indices: Optional[Dict[str, int]] = None) -> Path:
        """Plot original vs reconstructed tensor slices.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
            weights: Weight mask
            slice_indices: Which slices to plot
            
        Returns:
            Path to saved plot
        """
        # Convert CuPy arrays to NumPy if needed
        if hasattr(original, 'get'):
            original = original.get()
        if hasattr(reconstructed, 'get'):
            reconstructed = reconstructed.get()
        if hasattr(weights, 'get'):
            weights = weights.get()
        
        # Default to middle slices if not specified
        if slice_indices is None:
            slice_indices = {
                'mode_0': original.shape[0] // 2,
                'mode_1': original.shape[1] // 2,
                'mode_2': min(3, original.shape[2] - 1)  # Feature slice
            }
        
        n_modes = len(original.shape)
        fig, axes = plt.subplots(n_modes, 3, figsize=(15, 5 * n_modes))
        
        for mode in range(n_modes):
            # Get slice along this mode
            slice_idx = slice_indices.get(f'mode_{mode}', original.shape[mode] // 2)
            
            # Create slice tuple
            slice_tuple = [slice(None)] * n_modes
            slice_tuple[mode] = slice_idx
            slice_tuple = tuple(slice_tuple)
            
            # Extract slices
            orig_slice = original[slice_tuple]
            recon_slice = reconstructed[slice_tuple]
            weight_slice = weights[slice_tuple]
            
            # Mask by weights
            orig_masked = np.where(weight_slice > 0, orig_slice, np.nan)
            recon_masked = np.where(weight_slice > 0, recon_slice, np.nan)
            
            # Determine common color scale
            vmin = np.nanmin([orig_masked, recon_masked])
            vmax = np.nanmax([orig_masked, recon_masked])
            
            # Plot original
            im1 = axes[mode, 0].imshow(orig_masked, cmap=self.colormap,
                                      vmin=vmin, vmax=vmax, aspect='auto')
            axes[mode, 0].set_title(f'Original (Mode {mode}, Slice {slice_idx})')
            plt.colorbar(im1, ax=axes[mode, 0])
            
            # Plot reconstructed
            im2 = axes[mode, 1].imshow(recon_masked, cmap=self.colormap,
                                      vmin=vmin, vmax=vmax, aspect='auto')
            axes[mode, 1].set_title(f'Reconstructed')
            plt.colorbar(im2, ax=axes[mode, 1])
            
            # Plot difference
            diff = orig_masked - recon_masked
            im3 = axes[mode, 2].imshow(diff, cmap='RdBu_r', 
                                      vmin=-np.nanmax(np.abs(diff)),
                                      vmax=np.nanmax(np.abs(diff)),
                                      aspect='auto')
            axes[mode, 2].set_title(f'Residual')
            plt.colorbar(im3, ax=axes[mode, 2])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'reconstruction_slices.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved reconstruction slices to: {plot_path}")
        return plot_path
    
    def _plot_residuals(self, original: np.ndarray,
                       reconstructed: np.ndarray,
                       weights: np.ndarray) -> Path:
        """Plot residual patterns.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
            weights: Weight mask
            
        Returns:
            Path to saved plot
        """
        # Convert CuPy arrays to NumPy if needed
        if hasattr(original, 'get'):
            original = original.get()
        if hasattr(reconstructed, 'get'):
            reconstructed = reconstructed.get()
        if hasattr(weights, 'get'):
            weights = weights.get()
        
        # Calculate residuals
        residuals = (original - reconstructed) * weights
        
        # Flatten for analysis
        valid_mask = weights.flatten() > 0
        residuals_flat = residuals.flatten()[valid_mask]
        original_flat = original.flatten()[valid_mask]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residual distribution
        axes[0, 0].hist(residuals_flat, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Residual Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Residual Distribution')
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals_flat, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')
        
        # Residuals vs fitted values
        reconstructed_flat = reconstructed.flatten()[valid_mask]
        axes[1, 0].scatter(reconstructed_flat, residuals_flat, alpha=0.1, s=1)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals vs Fitted')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Residuals vs original values
        axes[1, 1].scatter(original_flat, residuals_flat, alpha=0.1, s=1)
        axes[1, 1].set_xlabel('Original Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Original')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'residuals.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved residuals plot to: {plot_path}")
        return plot_path
    
    def _plot_error_distribution(self, original: np.ndarray,
                                reconstructed: np.ndarray,
                                weights: np.ndarray) -> Path:
        """Plot distribution of reconstruction errors.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
            weights: Weight mask
            
        Returns:
            Path to saved plot
        """
        # Convert CuPy arrays to NumPy if needed
        if hasattr(original, 'get'):
            original = original.get()
        if hasattr(reconstructed, 'get'):
            reconstructed = reconstructed.get()
        if hasattr(weights, 'get'):
            weights = weights.get()
        
        # Calculate errors
        valid_mask = weights > 0
        errors = np.abs(original - reconstructed)[valid_mask]
        relative_errors = errors / (np.abs(original[valid_mask]) + 1e-10)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Absolute error distribution
        axes[0, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Absolute Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Absolute Error Distribution')
        axes[0, 0].set_yscale('log')
        
        # Relative error distribution
        axes[0, 1].hist(relative_errors[relative_errors < 2], bins=50, 
                       alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Relative Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Relative Error Distribution (< 200%)')
        axes[0, 1].set_yscale('log')
        
        # Error by value magnitude
        value_bins = np.percentile(original[valid_mask], np.linspace(0, 100, 11))
        bin_indices = np.digitize(original[valid_mask], value_bins) - 1
        
        mean_errors = []
        std_errors = []
        bin_centers = []
        
        for i in range(len(value_bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                mean_errors.append(np.mean(errors[mask]))
                std_errors.append(np.std(errors[mask]))
                bin_centers.append((value_bins[i] + value_bins[i+1]) / 2)
        
        axes[1, 0].errorbar(bin_centers, mean_errors, yerr=std_errors, 
                           fmt='o-', capsize=5)
        axes[1, 0].set_xlabel('Value Magnitude (Bin Center)')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('Error by Value Magnitude')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        
        # Cumulative error distribution
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        axes[1, 1].plot(sorted_errors, cumulative)
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Error Distribution')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add percentile lines
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(errors, p)
            axes[1, 1].axvline(x=val, linestyle='--', alpha=0.5, 
                              label=f'{p}th: {val:.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'error_distribution.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved error distribution to: {plot_path}")
        return plot_path
    
    def _plot_fit_by_density(self, original: np.ndarray,
                            reconstructed: np.ndarray,
                            weights: np.ndarray) -> Path:
        """Plot reconstruction quality by data density.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
            weights: Weight mask
            
        Returns:
            Path to saved plot
        """
        # Convert CuPy arrays to NumPy if needed
        if hasattr(original, 'get'):
            original = original.get()
        if hasattr(reconstructed, 'get'):
            reconstructed = reconstructed.get()
        if hasattr(weights, 'get'):
            weights = weights.get()
        
        # Calculate local density (simplified - count neighbors within each slice)
        densities = []
        fits = []
        
        # For each spatial location (assuming first two dimensions are spatial)
        if len(original.shape) >= 2:
            for i in range(original.shape[0]):
                for j in range(original.shape[1]):
                    # Get data slice at this location
                    if len(original.shape) == 3:
                        data_slice = original[i, j, :]
                        recon_slice = reconstructed[i, j, :]
                        weight_slice = weights[i, j, :]
                    else:
                        data_slice = original[i, j]
                        recon_slice = reconstructed[i, j]
                        weight_slice = weights[i, j]
                    
                    # Calculate density (proportion of non-zero weights)
                    density = np.mean(weight_slice > 0)
                    
                    # Calculate local fit
                    if np.sum(weight_slice) > 0:
                        local_error = np.sum((data_slice - recon_slice)**2 * weight_slice)
                        local_norm = np.sum(data_slice**2 * weight_slice)
                        if local_norm > 0:
                            local_fit = 1 - np.sqrt(local_error / local_norm)
                            densities.append(density)
                            fits.append(local_fit)
        
        if len(densities) > 0:
            densities = np.array(densities)
            fits = np.array(fits)
            
            # Create density bins
            density_bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(densities, density_bins) - 1
            
            # Calculate mean fit per bin
            mean_fits = []
            std_fits = []
            bin_centers = []
            
            for i in range(len(density_bins) - 1):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    mean_fits.append(np.mean(fits[mask]))
                    std_fits.append(np.std(fits[mask]))
                    bin_centers.append((density_bins[i] + density_bins[i+1]) / 2)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot with density coloring
            scatter = ax.scatter(densities, fits, c=densities, cmap='viridis',
                               alpha=0.3, s=1)
            plt.colorbar(scatter, ax=ax, label='Data Density')
            
            # Overlay binned means
            if len(mean_fits) > 0:
                ax.errorbar(bin_centers, mean_fits, yerr=std_fits,
                          fmt='ro-', linewidth=2, markersize=8,
                          capsize=5, label='Binned Mean ± Std')
            
            ax.set_xlabel('Data Density')
            ax.set_ylabel('Local Fit Score')
            ax.set_title('Reconstruction Quality by Data Density')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            # Fallback plot if density calculation fails
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Insufficient data for density analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Reconstruction Quality by Data Density')
        
        # Save plot
        plot_path = self.output_dir / f'fit_by_density.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved fit by density plot to: {plot_path}")
        return plot_path