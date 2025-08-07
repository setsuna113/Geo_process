#!/usr/bin/env python
"""
Trust Map Visualizer: Shows where biodiversity datasets agree or conflict.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from pathlib import Path
import rasterio
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TrustMapVisualizer:
    """Create trust maps showing data agreement and conflict areas"""
    
    def __init__(self, output_dir: str = "outputs/biodiversity_analysis/trust_maps"):
        """Initialize visualizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_align_data(self, datasets: Dict[str, str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load and align all datasets to common dimensions"""
        aligned_data = {}
        min_height, min_width = float('inf'), float('inf')
        
        # First pass: find minimum dimensions
        for name, path in datasets.items():
            with rasterio.open(path) as src:
                min_height = min(min_height, src.height)
                min_width = min(min_width, src.width)
        
        # Second pass: load and crop to common dimensions
        for name, path in datasets.items():
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                # Crop to common dimensions
                data = data[:min_height, :min_width]
                aligned_data[name] = data
        
        # Create combined valid mask
        valid_mask = None
        for data in aligned_data.values():
            mask = ~np.isnan(data)
            valid_mask = mask if valid_mask is None else valid_mask & mask
        
        return aligned_data, valid_mask
    
    def create_trust_map(self, datasets: Dict[str, str]) -> None:
        """Create comprehensive trust map visualization"""
        print("Creating trust map...")
        
        # Load and align data
        data, valid_mask = self.load_and_align_data(datasets)
        
        # Separate observed vs SDM
        observed_data = {}
        sdm_data = {}
        
        for name, array in data.items():
            if 'fungi' in name:
                sdm_data[name] = array
            else:
                observed_data[name] = array
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle("Biodiversity Data Trust Map: Agreement and Conflict Areas", 
                     fontsize=16, fontweight='bold', y=0.98)
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.2)
        
        # 1. Data availability trust (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_data_availability_trust(data, valid_mask, ax1)
        
        # 2. Observed data agreement (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_observed_agreement(observed_data, ax2)
        
        # 3. SDM consistency (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_sdm_consistency(sdm_data, ax3)
        
        # 4. Cross-method agreement (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_cross_method_agreement(observed_data, sdm_data, ax4)
        
        # 5. Hotspot consensus (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_hotspot_consensus(data, ax5)
        
        # 6. Conflict zones (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_conflict_zones(data, ax6)
        
        # 7. Combined trust score (bottom, spanning 2 columns)
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_combined_trust_score(data, ax7)
        
        # 8. Trust legend and statistics (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_trust_statistics(data, ax8)
        
        # Save figure
        output_path = self.output_dir / "comprehensive_trust_map.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def _plot_data_availability_trust(self, data: Dict, valid_mask: np.ndarray, ax: plt.Axes) -> None:
        """Plot data availability trust levels"""
        # Count how many datasets have data at each location
        availability = np.zeros_like(list(data.values())[0])
        
        for array in data.values():
            availability += ~np.isnan(array)
        
        availability = availability.astype(float)
        availability[availability == 0] = np.nan
        
        im = ax.imshow(availability, cmap='RdYlGn', vmin=1, vmax=len(data), aspect='auto')
        ax.set_title("Data Availability Trust\n(# of datasets with data)", fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label("Number of datasets")
        cbar.set_ticks([1, 2, 3, 4])
        cbar.set_ticklabels(['Low\n(1)', 'Medium\n(2)', 'Good\n(3)', 'High\n(4)'])
    
    def _plot_observed_agreement(self, observed_data: Dict, ax: plt.Axes) -> None:
        """Plot agreement between observed datasets"""
        if len(observed_data) < 2:
            ax.text(0.5, 0.5, "Insufficient observed data", ha='center', va='center')
            ax.axis('off')
            return
        
        # Normalize observed data to percentiles
        normalized = {}
        for name, array in observed_data.items():
            valid = ~np.isnan(array)
            norm_array = np.full_like(array, np.nan)
            if valid.any():
                percentiles = np.percentile(array[valid], [25, 50, 75])
                norm_array[valid] = np.digitize(array[valid], percentiles)
            normalized[name] = norm_array
        
        # Calculate agreement
        arrays = list(normalized.values())
        agreement = np.full_like(arrays[0], np.nan)
        
        valid = ~np.isnan(arrays[0]) & ~np.isnan(arrays[1])
        agreement[valid] = (arrays[0][valid] == arrays[1][valid]).astype(float)
        
        im = ax.imshow(agreement, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_title("Observed Data Agreement\n(Plants vs Terrestrial)", fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label("Agreement Level")
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Disagree', 'Partial', 'Agree'])
    
    def _plot_sdm_consistency(self, sdm_data: Dict, ax: plt.Axes) -> None:
        """Plot consistency between SDM predictions"""
        if len(sdm_data) < 2:
            ax.text(0.5, 0.5, "Insufficient SDM data", ha='center', va='center')
            ax.axis('off')
            return
        
        # Calculate coefficient of variation between SDMs
        arrays = list(sdm_data.values())
        
        # Stack arrays and calculate statistics
        stack = np.stack([a for a in arrays], axis=0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_sdm = np.nanmean(stack, axis=0)
            std_sdm = np.nanstd(stack, axis=0)
            cv = np.where(mean_sdm > 0, std_sdm / mean_sdm, np.nan)
        
        # Invert CV so high values = high consistency
        consistency = np.where(~np.isnan(cv), 1 - np.clip(cv, 0, 1), np.nan)
        
        im = ax.imshow(consistency, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_title("SDM Consistency\n(AM vs EcM Fungi)", fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label("Consistency (1 - CV)")
    
    def _plot_cross_method_agreement(self, observed: Dict, sdm: Dict, ax: plt.Axes) -> None:
        """Plot agreement between observed and SDM data"""
        # Use plants vs AM fungi as example
        plants = observed.get('plants')
        am_fungi = sdm.get('am_fungi')
        
        if plants is None or am_fungi is None:
            ax.text(0.5, 0.5, "Data not available", ha='center', va='center')
            ax.axis('off')
            return
        
        # Normalize both to percentiles
        valid_p = ~np.isnan(plants)
        valid_f = ~np.isnan(am_fungi)
        valid = valid_p & valid_f
        
        agreement_map = np.full_like(plants, np.nan)
        
        if valid.any():
            # Normalize to z-scores
            plants_z = np.zeros_like(plants)
            fungi_z = np.zeros_like(am_fungi)
            
            plants_z[valid] = (plants[valid] - np.mean(plants[valid])) / np.std(plants[valid])
            fungi_z[valid] = (am_fungi[valid] - np.mean(am_fungi[valid])) / np.std(am_fungi[valid])
            
            # Calculate agreement as inverse of absolute difference
            diff = np.abs(plants_z - fungi_z)
            agreement_map[valid] = 1 - np.clip(diff[valid] / 4, 0, 1)  # Scale difference
        
        im = ax.imshow(agreement_map, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_title("Cross-Method Agreement\n(Observed vs SDM)", fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label("Agreement Level")
    
    def _plot_hotspot_consensus(self, data: Dict, ax: plt.Axes) -> None:
        """Plot consensus on biodiversity hotspots"""
        # Define hotspots as top 10% for each dataset
        hotspot_count = np.zeros_like(list(data.values())[0])
        
        for array in data.values():
            valid = ~np.isnan(array)
            if valid.any():
                threshold = np.percentile(array[valid], 90)
                hotspot_count += (array >= threshold).astype(float)
        
        hotspot_count[hotspot_count == 0] = np.nan
        
        im = ax.imshow(hotspot_count, cmap='YlOrRd', vmin=0, vmax=len(data), aspect='auto')
        ax.set_title("Hotspot Consensus\n(Top 10% agreement)", fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label("# Datasets identifying as hotspot")
    
    def _plot_conflict_zones(self, data: Dict, ax: plt.Axes) -> None:
        """Identify and plot zones of high disagreement"""
        # Calculate pairwise disagreements
        arrays = list(data.values())
        n_datasets = len(arrays)
        
        # Normalize all arrays to [0, 1]
        normalized = []
        for array in arrays:
            valid = ~np.isnan(array)
            norm = np.full_like(array, np.nan)
            if valid.any():
                min_val = np.min(array[valid])
                max_val = np.max(array[valid])
                if max_val > min_val:
                    norm[valid] = (array[valid] - min_val) / (max_val - min_val)
            normalized.append(norm)
        
        # Calculate variance as measure of disagreement
        stack = np.stack(normalized, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variance = np.nanvar(stack, axis=0)
        
        # Highlight high variance areas
        conflict = np.where(variance > np.nanpercentile(variance, 75), variance, np.nan)
        
        im = ax.imshow(conflict, cmap='Reds', aspect='auto')
        ax.set_title("Conflict Zones\n(High disagreement areas)", fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label("Disagreement Level")
    
    def _plot_combined_trust_score(self, data: Dict, ax: plt.Axes) -> None:
        """Calculate and plot combined trust score"""
        # Combine multiple trust metrics
        trust_score = np.zeros_like(list(data.values())[0])
        weight_sum = 0
        
        # 1. Data availability (weight: 0.3)
        availability = np.zeros_like(trust_score)
        for array in data.values():
            availability += ~np.isnan(array)
        trust_score += (availability / len(data)) * 0.3
        weight_sum += 0.3
        
        # 2. Low variance (weight: 0.4)
        arrays = []
        for array in data.values():
            valid = ~np.isnan(array)
            norm = np.full_like(array, np.nan)
            if valid.any():
                norm[valid] = (array[valid] - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
            arrays.append(norm)
        
        stack = np.stack(arrays, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variance = np.nanvar(stack, axis=0)
        
        low_var_score = 1 - np.clip(variance * 4, 0, 1)  # Scale and invert
        trust_score += np.nan_to_num(low_var_score) * 0.4
        weight_sum += 0.4
        
        # 3. Spatial consistency (weight: 0.3) - low spatial gradient
        gradient_score = np.zeros_like(trust_score)
        for array in data.values():
            gy, gx = np.gradient(np.nan_to_num(array))
            grad_mag = np.sqrt(gy**2 + gx**2)
            grad_mag = 1 - np.clip(grad_mag / np.nanpercentile(grad_mag, 95), 0, 1)
            gradient_score += grad_mag / len(data)
        
        trust_score += gradient_score * 0.3
        weight_sum += 0.3
        
        # Normalize trust score
        trust_score = trust_score / weight_sum
        trust_score[availability == 0] = np.nan
        
        # Create categorical trust levels
        trust_levels = np.full_like(trust_score, np.nan)
        valid = ~np.isnan(trust_score)
        trust_levels[valid] = np.digitize(trust_score[valid], [0.25, 0.5, 0.75])
        
        # Custom colormap for trust levels
        colors = ['#d73027', '#fee08b', '#a6d96a', '#1a9850']  # Red to green
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        im = ax.imshow(trust_levels, cmap=cmap, vmin=0, vmax=3, aspect='auto')
        ax.set_title("Combined Trust Score\n(Integrating all metrics)", fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
        cbar.set_label("Trust Level", fontsize=10)
        cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
        cbar.set_ticklabels(['Low', 'Medium', 'Good', 'High'])
    
    def _plot_trust_statistics(self, data: Dict, ax: plt.Axes) -> None:
        """Plot trust statistics and legend"""
        ax.axis('off')
        
        # Calculate statistics
        stats_text = "TRUST MAP STATISTICS\n" + "="*25 + "\n\n"
        
        # Data coverage
        total_pixels = list(data.values())[0].size
        coverage_stats = {}
        for name, array in data.items():
            valid_pixels = np.sum(~np.isnan(array))
            coverage_stats[name] = (valid_pixels / total_pixels) * 100
        
        stats_text += "Data Coverage:\n"
        for name, cov in coverage_stats.items():
            stats_text += f"  {name}: {cov:.1f}%\n"
        
        # Overlap statistics
        all_valid = np.ones_like(list(data.values())[0], dtype=bool)
        for array in data.values():
            all_valid &= ~np.isnan(array)
        
        overlap_pct = (np.sum(all_valid) / total_pixels) * 100
        stats_text += f"\nComplete overlap: {overlap_pct:.1f}%\n"
        
        # Agreement metrics
        stats_text += "\n" + "="*25 + "\n"
        stats_text += "INTERPRETATION:\n\n"
        stats_text += "High Trust Areas:\n"
        stats_text += "• All datasets present\n"
        stats_text += "• Low variance\n"
        stats_text += "• Spatial consistency\n\n"
        
        stats_text += "Low Trust Areas:\n"
        stats_text += "• Missing data\n"
        stats_text += "• High disagreement\n"
        stats_text += "• Method conflicts\n\n"
        
        stats_text += "Use Cases:\n"
        stats_text += "• Conservation: Focus on\n"
        stats_text += "  high-trust hotspots\n"
        stats_text += "• Research: Investigate\n"
        stats_text += "  conflict zones\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')


def create_method_comparison_fixed(datasets: Dict[str, str], output_dir: str = "outputs/biodiversity_analysis/visualizations") -> None:
    """Create fixed method comparison visualization"""
    print("\nCreating method comparison (SDM vs Observed)...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate observed vs SDM
    observed_data = {}
    sdm_data = {}
    
    for name, path in datasets.items():
        if Path(path).exists():
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                if 'fungi' in name:
                    sdm_data[name] = data
                else:
                    observed_data[name] = data
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Method Comparison: SDM Predictions vs Observed Counts", 
                 fontsize=14, fontweight='bold')
    
    # Plot observed data
    for idx, (name, data) in enumerate(observed_data.items()):
        if idx < 2:
            ax = axes[0, idx]
            valid = ~np.isnan(data)
            
            im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
            ax.set_title(f"Observed: {name.replace('_', ' ').title()}")
            ax.axis('off')
            
            # Add stats
            if valid.any():
                stats = f"Range: {np.min(data[valid]):.0f}-{np.max(data[valid]):.0f}\n"
                stats += f"Mean: {np.mean(data[valid]):.1f}"
                ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9, va='top')
    
    # Plot SDM data
    for idx, (name, data) in enumerate(sdm_data.items()):
        if idx < 2:
            ax = axes[1, idx]
            valid = ~np.isnan(data)
            
            im = ax.imshow(data, cmap='viridis', aspect='auto')
            ax.set_title(f"SDM: {name.replace('_', ' ').title()}")
            ax.axis('off')
            
            # Add stats
            if valid.any():
                stats = f"Range: {np.min(data[valid]):.0f}-{np.max(data[valid]):.0f}\n"
                stats += f"Mean: {np.mean(data[valid]):.1f}"
                ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=9, va='top')
    
    # Comparison panel
    ax = axes[:, 2].flatten()[0]
    axes[1, 2].axis('off')
    
    # Make comparison text box span both cells
    ax.axis('off')
    comparison_text = """
KEY DIFFERENCES:

OBSERVED COUNTS:
• Direct observations
• Sampling bias present
• Reflects actual occurrences
• Limited by survey effort
• Higher confidence in values

SDM PREDICTIONS:
• Modeled potential habitat
• Complete spatial coverage
• Environmental suitability
• May overpredict range
• Uncertainty in predictions

IMPLICATIONS:
• Cannot directly compare values
• Different ecological meaning
• SDM shows potential
• Observed shows realized
• Need different analyses
    """
    
    ax.text(0.1, 0.5, comparison_text, transform=ax.transAxes,
            fontsize=11, va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    output_path = output_dir / "5_method_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Test with example datasets
    datasets = {
        'plants': "/scratch/yl998/downsampled_rasters/plants-richness_downsampled_0.16667.tif",
        'terrestrial': "/scratch/yl998/downsampled_rasters/terrestrial-richness_downsampled_0.16667.tif",
        'am_fungi': "/scratch/yl998/downsampled_rasters/am-fungi-richness_downsampled_0.16667.tif",
        'ecm_fungi': "/scratch/yl998/downsampled_rasters/ecm-fungi-richness_downsampled_0.16667.tif"
    }
    
    # Check which files exist
    existing = {}
    for name, path in datasets.items():
        if Path(path).exists():
            existing[name] = path
            print(f"Found: {name}")
    
    if existing:
        visualizer = TrustMapVisualizer()
        visualizer.create_trust_map(existing)
        create_method_comparison_fixed(existing)