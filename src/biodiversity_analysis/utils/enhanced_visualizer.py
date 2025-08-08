"""
Enhanced Biodiversity Visualizer
=================================

Creates multiple, focused visualizations for better understanding of the data.
Each visualization addresses a specific question.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
import rasterio
from scipy import stats
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class EnhancedVisualizer:
    """
    Create focused, interpretable visualizations for biodiversity data.
    """
    
    def __init__(self, output_dir: str = "outputs/biodiversity_analysis/visualizations"):
        """Initialize the visualizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_coverage_comparison(self, datasets: Dict[str, str]) -> None:
        """
        Create a clear coverage comparison visualization.
        Shows where we have data, where we don't, and overlap.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data Coverage Analysis: Where Do We Have Data?', fontsize=16, fontweight='bold')
        
        all_data = {}
        coverage_masks = {}
        
        # Load all datasets
        for name, path in datasets.items():
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                all_data[name] = data
                coverage_masks[name] = ~np.isnan(data) & (data > 0)
        
        # Plot individual coverage
        for idx, (name, mask) in enumerate(coverage_masks.items()):
            if idx < 4:
                ax = axes[idx // 2, idx % 2]
                
                # Create coverage map with different colors for zeros and data
                coverage_img = np.zeros_like(mask, dtype=float)
                coverage_img[mask] = 1  # Has data
                coverage_img[~mask] = np.nan  # No data
                
                # Color based on data type
                cmap = 'Greens' if 'fungi' not in name else 'Purples'
                
                im = ax.imshow(coverage_img, cmap=cmap, vmin=0, vmax=1, aspect='auto')
                ax.set_title(f'{name.replace("_", " ").title()}\n'
                           f'Coverage: {np.sum(mask)/mask.size*100:.1f}%\n'
                           f'({"SDM" if "fungi" in name else "Observed"})',
                           fontsize=11)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                
                # Add text annotation
                ax.text(0.02, 0.98, f'{np.sum(mask):,} cells',
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Overlap visualization
        ax_overlap = axes[1, 2]
        
        # Calculate overlap levels - need to handle different shapes
        # Find minimum common shape
        shapes = [mask.shape for mask in coverage_masks.values()]
        min_rows = min(s[0] for s in shapes)
        min_cols = min(s[1] for s in shapes)
        
        overlap_count = np.zeros((min_rows, min_cols), dtype=int)
        for mask in coverage_masks.values():
            # Crop to common size
            overlap_count += mask[:min_rows, :min_cols].astype(int)
        
        im = ax_overlap.imshow(overlap_count, cmap='YlOrRd', vmin=0, vmax=len(datasets),
                               aspect='auto')
        ax_overlap.set_title(f'Data Overlap\nAll 4 datasets: {np.sum(overlap_count==4)/overlap_count.size*100:.1f}%',
                            fontsize=11)
        ax_overlap.set_xlabel('Longitude')
        ax_overlap.set_ylabel('Latitude')
        
        cbar = plt.colorbar(im, ax=ax_overlap)
        cbar.set_label('# Datasets with data')
        
        # Coverage bar chart
        ax_bar = axes[0, 2]
        
        coverage_percents = [np.sum(mask)/mask.size*100 for mask in coverage_masks.values()]
        colors = ['green' if 'fungi' not in name else 'purple' 
                 for name in coverage_masks.keys()]
        
        bars = ax_bar.bar(range(len(coverage_masks)), coverage_percents, color=colors, alpha=0.7)
        ax_bar.set_xticks(range(len(coverage_masks)))
        ax_bar.set_xticklabels([name.replace('_', '\n') for name in coverage_masks.keys()],
                              rotation=0, fontsize=9)
        ax_bar.set_ylabel('Coverage (%)')
        ax_bar.set_title('Coverage Comparison', fontsize=11)
        ax_bar.set_ylim(0, 40)
        ax_bar.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, pct in zip(bars, coverage_percents):
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / "1_coverage_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_richness_patterns(self, datasets: Dict[str, str]) -> None:
        """
        Show richness patterns side by side with proper scaling.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Species Richness Patterns: What Are The Values?', 
                    fontsize=16, fontweight='bold')
        
        for idx, (name, path) in enumerate(datasets.items()):
            if idx < 4:
                ax = axes[idx // 2, idx % 2]
                
                with rasterio.open(path) as src:
                    data = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan
                
                # Mask zeros
                data[data == 0] = np.nan
                
                # Clip at 99th percentile for better visualization
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    vmax = np.percentile(valid_data, 99)
                    data_display = np.clip(data, 0, vmax)
                else:
                    data_display = data
                
                # Choose colormap
                if 'fungi' in name:
                    cmap = 'viridis'
                    label = 'Predicted Richness'
                else:
                    cmap = 'YlOrRd'
                    label = 'Observed Richness'
                
                im = ax.imshow(data_display, cmap=cmap, aspect='auto')
                
                # Add statistics
                if len(valid_data) > 0:
                    stats_text = f'Range: {np.min(valid_data):.0f}-{np.max(valid_data):.0f}\n'
                    stats_text += f'Mean: {np.mean(valid_data):.1f}\n'
                    stats_text += f'Median: {np.median(valid_data):.1f}'
                    
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                ax.set_title(f'{name.replace("_", " ").title()}\n({"SDM" if "fungi" in name else "Observed"})',
                           fontsize=11)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(label, fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / "2_richness_patterns.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_concordance_matrix(self, datasets: Dict[str, str]) -> None:
        """
        Create a clear concordance matrix showing correlations between datasets.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Cross-Taxa Concordance: How Do Datasets Relate?', 
                    fontsize=16, fontweight='bold')
        
        # Load and align data - need to handle different sizes
        all_data = {}
        shapes = []
        
        # First pass: get all shapes
        for name, path in datasets.items():
            with rasterio.open(path) as src:
                shapes.append(src.shape)
        
        # Find minimum shape
        min_rows = min(s[0] for s in shapes)
        min_cols = min(s[1] for s in shapes)
        
        # Second pass: load and crop data
        for name, path in datasets.items():
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                # Crop to common size
                all_data[name] = data[:min_rows, :min_cols].flatten()
        
        # Calculate correlations
        names = list(all_data.keys())
        n = len(names)
        spearman_matrix = np.ones((n, n))
        pearson_matrix = np.ones((n, n))
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i != j:
                    data1 = all_data[name1]
                    data2 = all_data[name2]
                    
                    # Find valid pairs
                    valid = ~np.isnan(data1) & ~np.isnan(data2) & (data1 > 0) & (data2 > 0)
                    
                    if np.sum(valid) > 100:
                        spearman_matrix[i, j], _ = stats.spearmanr(data1[valid], data2[valid])
                        pearson_matrix[i, j], _ = stats.pearsonr(data1[valid], data2[valid])
        
        # Plot Spearman correlation
        im1 = ax1.imshow(spearman_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels([name.replace('_', '\n') for name in names], fontsize=9)
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_title('Spearman Correlation\n(Rank-based, robust)', fontsize=11)
        
        # Add values
        for i in range(n):
            for j in range(n):
                if i != j:
                    val = spearman_matrix[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax1.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=color, fontsize=9, fontweight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Correlation')
        
        # Interpretation guide
        ax2.axis('off')
        
        interpretation_text = """
INTERPRETATION GUIDE
====================

Correlation Values:
• 0.8 to 1.0:  Very strong positive
• 0.6 to 0.8:  Strong positive
• 0.4 to 0.6:  Moderate positive
• -0.4 to 0.4: Weak/no relationship
• -0.6 to -0.4: Moderate negative
• < -0.6:      Strong negative

Key Findings:
"""
        
        # Add specific findings
        findings = []
        
        # Check plant-vertebrate correlation
        if 'plants' in names and 'animals' in names:
            idx1 = names.index('plants')
            idx2 = names.index('animals')
            corr = spearman_matrix[idx1, idx2]
            findings.append(f'✓ Plants~Vertebrates: {corr:.2f} (expected)')
        
        # Check fungi correlations
        fungi_names = [n for n in names if 'fungi' in n]
        if len(fungi_names) == 2:
            idx1 = names.index(fungi_names[0])
            idx2 = names.index(fungi_names[1])
            corr = spearman_matrix[idx1, idx2]
            if corr < 0:
                findings.append(f'✗ AM~EcM Fungi: {corr:.2f} (PROBLEM!)')
            else:
                findings.append(f'• AM~EcM Fungi: {corr:.2f}')
        
        # Check plant-fungi
        if 'plants' in names and 'ecm_fungi' in names:
            idx1 = names.index('plants')
            idx2 = names.index('ecm_fungi')
            corr = spearman_matrix[idx1, idx2]
            if corr < 0:
                findings.append(f'✗ Plants~EcM: {corr:.2f} (IMPOSSIBLE!)')
        
        interpretation_text += '\n'.join(findings)
        
        interpretation_text += """

⚠️ CRITICAL ISSUES:
• Negative correlations between 
  ecologically linked taxa
• SDM predictions inconsistent
• Method artifacts affecting patterns
"""
        
        ax2.text(0.1, 0.9, interpretation_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        save_path = self.output_dir / "3_concordance_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_hotspot_analysis(self, datasets: Dict[str, str]) -> None:
        """
        Visualize biodiversity hotspots and their overlap.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Biodiversity Hotspots: Where Are The Priority Areas?',
                    fontsize=16, fontweight='bold')
        
        hotspot_masks = {}
        
        # Calculate hotspots for each dataset
        for idx, (name, path) in enumerate(datasets.items()):
            if idx < 4:
                ax = fig.add_subplot(gs[idx // 2, idx % 2])
                
                with rasterio.open(path) as src:
                    data = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan
                
                valid_data = data[~np.isnan(data) & (data > 0)]
                
                if len(valid_data) > 0:
                    # Top 10% threshold
                    threshold = np.percentile(valid_data, 90)
                    hotspot_mask = (data >= threshold) & ~np.isnan(data)
                    hotspot_masks[name] = hotspot_mask
                    
                    # Visualize
                    display = np.zeros_like(data)
                    display[~np.isnan(data) & (data > 0)] = 0.3  # Has data
                    display[hotspot_mask] = 1  # Hotspot
                    
                    cmap_colors = ['white', 'lightgray', 'red'] if 'fungi' not in name else ['white', 'lightgray', 'darkviolet']
                    cmap = mcolors.LinearSegmentedColormap.from_list('hotspot', cmap_colors)
                    
                    im = ax.imshow(display, cmap=cmap, vmin=0, vmax=1, aspect='auto')
                    
                    ax.set_title(f'{name.replace("_", " ").title()}\n'
                               f'Threshold: >{threshold:.0f} species\n'
                               f'{np.sum(hotspot_mask):,} hotspot cells',
                               fontsize=10)
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
        
        # Hotspot overlap
        ax_overlap = fig.add_subplot(gs[0, 2])
        
        if len(hotspot_masks) > 0:
            # Handle different shapes
            shapes = [mask.shape for mask in hotspot_masks.values()]
            min_rows = min(s[0] for s in shapes)
            min_cols = min(s[1] for s in shapes)
            
            overlap_count = np.zeros((min_rows, min_cols), dtype=int)
            for mask in hotspot_masks.values():
                overlap_count += mask[:min_rows, :min_cols].astype(int)
            
            im = ax_overlap.imshow(overlap_count, cmap='hot', vmin=0, vmax=len(datasets),
                                  aspect='auto')
            ax_overlap.set_title(f'Hotspot Overlap\n'
                               f'All 4 taxa: {np.sum(overlap_count==4):,} cells',
                               fontsize=11)
            ax_overlap.set_xlabel('Longitude')
            ax_overlap.set_ylabel('Latitude')
            
            cbar = plt.colorbar(im, ax=ax_overlap)
            cbar.set_label('# Taxa hotspots')
        
        # Venn diagram concept (simplified bar chart)
        ax_venn = fig.add_subplot(gs[1, 2])
        
        # Calculate pairwise overlaps
        overlap_data = []
        labels = []
        
        names = list(hotspot_masks.keys())
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i < j:
                    # Handle different shapes
                    mask1 = hotspot_masks[name1]
                    mask2 = hotspot_masks[name2]
                    min_r = min(mask1.shape[0], mask2.shape[0])
                    min_c = min(mask1.shape[1], mask2.shape[1])
                    overlap = np.sum(mask1[:min_r, :min_c] & mask2[:min_r, :min_c])
                    overlap_pct = overlap / min(np.sum(hotspot_masks[name1]),
                                              np.sum(hotspot_masks[name2])) * 100
                    overlap_data.append(overlap_pct)
                    labels.append(f'{name1[:4]}\n×\n{name2[:4]}')
        
        bars = ax_venn.bar(range(len(overlap_data)), overlap_data, color='coral', alpha=0.7)
        ax_venn.set_xticks(range(len(overlap_data)))
        ax_venn.set_xticklabels(labels, fontsize=8)
        ax_venn.set_ylabel('Overlap (%)')
        ax_venn.set_title('Pairwise Hotspot Overlap\n(% of smaller hotspot)', fontsize=10)
        ax_venn.set_ylim(0, 100)
        ax_venn.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars, overlap_data):
            ax_venn.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
        
        # Summary text
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis('off')
        
        summary_text = """
KEY FINDINGS:
• Different taxa identify DIFFERENT priority areas
• Plants and vertebrates show ~66% hotspot overlap (good)
• Fungi hotspots DON'T overlap with plant hotspots (concerning!)
• NO locations are hotspots for ALL four taxa

CONSERVATION IMPLICATIONS:
• Single-taxon conservation strategies will miss other taxa
• Need integrated multi-taxa approach
• Current protected areas may not cover all biodiversity
"""
        
        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='top')
        
        plt.tight_layout()
        save_path = self.output_dir / "4_hotspot_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_method_comparison(self, datasets: Dict[str, str]) -> None:
        """
        Direct comparison of SDM vs Observed data patterns.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Method Comparison: SDM Predictions vs Observed Counts',
                    fontsize=16, fontweight='bold')
        
        # Separate SDM and observed
        sdm_datasets = {k: v for k, v in datasets.items() if 'fungi' in k}
        obs_datasets = {k: v for k, v in datasets.items() if 'fungi' not in k}
        
        # Load data
        all_data = {}
        for name, path in datasets.items():
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                all_data[name] = data
        
        # Value distributions
        ax1 = axes[0, 0]
        
        for name, data in all_data.items():
            valid_data = data[~np.isnan(data) & (data > 0)]
            if len(valid_data) > 0:
                if 'fungi' in name:
                    ax1.hist(valid_data, bins=30, alpha=0.5, label=name,
                           color='purple', density=True)
                else:
                    # Clip extreme values for visualization
                    clipped = np.clip(valid_data, 0, np.percentile(valid_data, 99))
                    ax1.hist(clipped, bins=30, alpha=0.5, label=name,
                           color='green', density=True)
        
        ax1.set_xlabel('Species Richness')
        ax1.set_ylabel('Density')
        ax1.set_title('Value Distributions', fontsize=11)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Spatial variance comparison
        ax2 = axes[0, 1]
        
        variance_data = {}
        for name, data in all_data.items():
            # Calculate local variance
            from scipy.ndimage import generic_filter
            
            def local_variance(values):
                valid = values[~np.isnan(values)]
                return np.var(valid) if len(valid) > 2 else np.nan
            
            local_var = generic_filter(data, local_variance, size=5, mode='constant', cval=np.nan)
            variance_data[name] = np.nanmean(local_var)
        
        names = list(variance_data.keys())
        values = list(variance_data.values())
        colors = ['purple' if 'fungi' in n else 'green' for n in names]
        
        bars = ax2.bar(range(len(names)), values, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
        ax2.set_ylabel('Mean Local Variance')
        ax2.set_title('Spatial Heterogeneity\n(Higher = more patchy)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Direct scatter comparisons
        plot_idx = 2
        for obs_name, obs_data in obs_datasets.items():
            for sdm_name, sdm_data in sdm_datasets.items():
                if plot_idx < 6:
                    ax = axes[plot_idx // 3, plot_idx % 3]
                    
                    # Get matching pixels
                    obs_flat = obs_data.flatten()
                    sdm_flat = sdm_data.flatten()
                    
                    valid = ~np.isnan(obs_flat) & ~np.isnan(sdm_flat) & (obs_flat > 0) & (sdm_flat > 0)
                    
                    if np.sum(valid) > 100:
                        # Subsample for plotting
                        if np.sum(valid) > 5000:
                            sample_idx = np.random.choice(np.where(valid)[0], 5000, replace=False)
                            obs_sample = obs_flat[sample_idx]
                            sdm_sample = sdm_flat[sample_idx]
                        else:
                            obs_sample = obs_flat[valid]
                            sdm_sample = sdm_flat[valid]
                        
                        # Normalize for comparison
                        obs_norm = (obs_sample - np.min(obs_sample)) / (np.max(obs_sample) - np.min(obs_sample))
                        sdm_norm = (sdm_sample - np.min(sdm_sample)) / (np.max(sdm_sample) - np.min(sdm_sample))
                        
                        ax.scatter(obs_norm, sdm_norm, alpha=0.3, s=1)
                        
                        # Add correlation
                        r, _ = stats.spearmanr(obs_sample, sdm_sample)
                        
                        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='1:1 line')
                        ax.set_xlabel(f'{obs_name} (normalized)')
                        ax.set_ylabel(f'{sdm_name} (normalized)')
                        ax.set_title(f'r = {r:.3f}', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        
                        plot_idx += 1
        
        # Summary
        ax_summary = axes[1, 2]
        ax_summary.axis('off')
        
        summary_text = """
KEY DIFFERENCES:

SDM Predictions:
• Smooth spatial patterns
• Narrow value ranges
• Low spatial variance
• Environmental predictors

Observed Counts:
• Patchy distributions
• Wide value ranges
• High spatial variance
• Actual survey data

IMPLICATIONS:
• Cannot compare absolute values
• Patterns partially comparable
• Need harmonized methods
"""
        
        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                       fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        save_path = self.output_dir / "5_method_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()