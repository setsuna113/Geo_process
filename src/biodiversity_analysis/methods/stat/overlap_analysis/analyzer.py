"""
Overlap Analysis for Biodiversity Datasets
===========================================

Analyzes spatial overlap, cross-taxa concordance, and data agreement between
multiple biodiversity datasets with different data types (observed vs predicted).
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class OverlapAnalyzer:
    """
    Analyze overlap and concordance between biodiversity datasets.
    """
    
    def __init__(self, output_dir: str = "outputs/biodiversity_analysis"):
        """Initialize the overlap analyzer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        self.aligned_data = {}
        
    def load_and_align_datasets(self, dataset_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Load multiple datasets and align them to common grid.
        
        Parameters
        ----------
        dataset_paths : Dict[str, str]
            Dictionary mapping dataset names to file paths
            
        Returns
        -------
        Dict[str, np.ndarray]
            Aligned datasets
        """
        print("Loading and aligning datasets...")
        print("-" * 40)
        
        # First, load all datasets and find common extent
        bounds_list = []
        resolutions = []
        shapes = []
        
        for name, path in dataset_paths.items():
            with rasterio.open(path) as src:
                bounds_list.append(src.bounds)
                resolutions.append((src.transform[0], abs(src.transform[4])))
                shapes.append(src.shape)
                print(f"  {name}: {src.shape}, resolution: {src.transform[0]:.6f}")
        
        # Use the coarsest resolution (maximum pixel size)
        target_res = max(resolutions, key=lambda x: x[0])
        print(f"\nTarget resolution: {target_res[0]:.6f} degrees")
        
        # Find common extent (intersection of all bounds)
        common_bounds = {
            'left': max(b.left for b in bounds_list),
            'right': min(b.right for b in bounds_list),
            'bottom': max(b.bottom for b in bounds_list),
            'top': min(b.top for b in bounds_list)
        }
        
        print(f"Common extent: {common_bounds}")
        
        # Calculate common shape
        width = int((common_bounds['right'] - common_bounds['left']) / target_res[0])
        height = int((common_bounds['top'] - common_bounds['bottom']) / target_res[1])
        
        print(f"Common grid size: {height} x {width}")
        
        # Load and align each dataset
        for name, path in dataset_paths.items():
            with rasterio.open(path) as src:
                # Read data
                data = src.read(1).astype(np.float32)
                
                # Handle nodata
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                
                # If resolution matches, just crop
                if abs(src.transform[0] - target_res[0]) < 1e-6:
                    # Calculate crop indices
                    col_off = int((common_bounds['left'] - src.bounds.left) / src.transform[0])
                    row_off = int((src.bounds.top - common_bounds['top']) / abs(src.transform[4]))
                    
                    # Crop to common extent
                    aligned = data[row_off:row_off+height, col_off:col_off+width]
                else:
                    # Need to resample
                    print(f"  Resampling {name}...")
                    # This is simplified - in production you'd use rasterio's reproject
                    from scipy import ndimage
                    zoom_factors = (
                        height / data.shape[0],
                        width / data.shape[1]
                    )
                    aligned = ndimage.zoom(data, zoom_factors, order=1)
                
                self.aligned_data[name] = aligned
                self.datasets[name] = {
                    'data': aligned,
                    'original_path': path,
                    'shape': aligned.shape,
                    'bounds': common_bounds
                }
        
        print(f"\n✓ Aligned {len(self.datasets)} datasets to common grid")
        return self.aligned_data
    
    def calculate_overlap_statistics(self) -> Dict:
        """
        Calculate overlap statistics between all dataset pairs.
        
        Returns
        -------
        Dict
            Overlap statistics for each dataset pair
        """
        results = {}
        dataset_names = list(self.aligned_data.keys())
        
        print("\nCalculating overlap statistics...")
        print("-" * 40)
        
        # Individual dataset coverage
        for name, data in self.aligned_data.items():
            valid_mask = ~np.isnan(data) & (data > 0)
            coverage_percent = np.sum(valid_mask) / data.size * 100
            
            results[name] = {
                'total_cells': data.size,
                'valid_cells': np.sum(valid_mask),
                'coverage_percent': coverage_percent,
                'mean_value': np.nanmean(data[valid_mask]) if np.any(valid_mask) else 0,
                'max_value': np.nanmax(data[valid_mask]) if np.any(valid_mask) else 0
            }
            
            print(f"  {name}: {coverage_percent:.1f}% coverage")
        
        # Pairwise overlap
        print("\nPairwise overlap:")
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i < j:
                    data1 = self.aligned_data[name1]
                    data2 = self.aligned_data[name2]
                    
                    mask1 = ~np.isnan(data1) & (data1 > 0)
                    mask2 = ~np.isnan(data2) & (data2 > 0)
                    
                    both_valid = mask1 & mask2
                    either_valid = mask1 | mask2
                    
                    overlap_cells = np.sum(both_valid)
                    union_cells = np.sum(either_valid)
                    
                    if union_cells > 0:
                        jaccard = overlap_cells / union_cells
                        overlap_percent = overlap_cells / data1.size * 100
                    else:
                        jaccard = 0
                        overlap_percent = 0
                    
                    pair_key = f"{name1}_vs_{name2}"
                    results[pair_key] = {
                        'overlap_cells': overlap_cells,
                        'union_cells': union_cells,
                        'jaccard_index': jaccard,
                        'overlap_percent': overlap_percent
                    }
                    
                    print(f"  {name1} ∩ {name2}: {overlap_percent:.1f}% overlap, Jaccard={jaccard:.3f}")
        
        # All datasets overlap
        all_masks = [~np.isnan(self.aligned_data[name]) & (self.aligned_data[name] > 0) 
                     for name in dataset_names]
        all_valid = np.logical_and.reduce(all_masks)
        all_overlap_percent = np.sum(all_valid) / data1.size * 100
        
        results['all_datasets'] = {
            'overlap_cells': np.sum(all_valid),
            'overlap_percent': all_overlap_percent
        }
        
        print(f"\nAll datasets overlap: {all_overlap_percent:.1f}%")
        
        return results
    
    def calculate_concordance(self) -> Dict:
        """
        Calculate cross-taxa concordance (correlation) between datasets.
        
        Returns
        -------
        Dict
            Concordance metrics between dataset pairs
        """
        results = {}
        dataset_names = list(self.aligned_data.keys())
        
        print("\nCalculating cross-taxa concordance...")
        print("-" * 40)
        
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i < j:
                    data1 = self.aligned_data[name1].flatten()
                    data2 = self.aligned_data[name2].flatten()
                    
                    # Find valid pairs
                    valid_mask = ~np.isnan(data1) & ~np.isnan(data2) & (data1 > 0) & (data2 > 0)
                    
                    if np.sum(valid_mask) > 10:
                        valid_data1 = data1[valid_mask]
                        valid_data2 = data2[valid_mask]
                        
                        # Pearson correlation
                        pearson_r, pearson_p = stats.pearsonr(valid_data1, valid_data2)
                        
                        # Spearman correlation (rank-based, better for different data types)
                        spearman_r, spearman_p = stats.spearmanr(valid_data1, valid_data2)
                        
                        # Determine if SDM vs observed comparison
                        is_mixed = ('fungi' in name1) != ('fungi' in name2)
                        
                        pair_key = f"{name1}_vs_{name2}"
                        results[pair_key] = {
                            'n_pairs': np.sum(valid_mask),
                            'pearson_r': pearson_r,
                            'pearson_p': pearson_p,
                            'spearman_r': spearman_r,
                            'spearman_p': spearman_p,
                            'is_mixed_method': is_mixed,
                            'interpretation': self._interpret_correlation(spearman_r, is_mixed)
                        }
                        
                        print(f"  {name1} ~ {name2}:")
                        print(f"    Spearman r = {spearman_r:.3f} (p = {spearman_p:.3e})")
                        print(f"    {results[pair_key]['interpretation']}")
        
        return results
    
    def _interpret_correlation(self, r: float, is_mixed: bool) -> str:
        """Interpret correlation coefficient"""
        if is_mixed:
            # SDM vs observed - expect lower correlation
            if abs(r) < 0.3:
                return "Weak/no concordance (expected for SDM vs observed)"
            elif abs(r) < 0.5:
                return "Moderate concordance (reasonable for SDM vs observed)"
            else:
                return "Strong concordance (surprising for SDM vs observed!)"
        else:
            # Same method - expect higher correlation
            if abs(r) < 0.3:
                return "Weak/no concordance (concerning for same method)"
            elif abs(r) < 0.6:
                return "Moderate concordance"
            else:
                return "Strong concordance (expected for same method)"
    
    def identify_hotspots(self, threshold_percentile: float = 90) -> Dict:
        """
        Identify biodiversity hotspots for each dataset and their overlap.
        
        Parameters
        ----------
        threshold_percentile : float
            Percentile threshold for defining hotspots
            
        Returns
        -------
        Dict
            Hotspot locations and overlap statistics
        """
        results = {}
        
        print(f"\nIdentifying hotspots (top {100-threshold_percentile}%)...")
        print("-" * 40)
        
        hotspot_masks = {}
        
        # Identify hotspots for each dataset
        for name, data in self.aligned_data.items():
            valid_data = data[~np.isnan(data) & (data > 0)]
            
            if len(valid_data) > 0:
                threshold = np.percentile(valid_data, threshold_percentile)
                hotspot_mask = (data >= threshold) & ~np.isnan(data)
                hotspot_masks[name] = hotspot_mask
                
                hotspot_cells = np.sum(hotspot_mask)
                hotspot_percent = hotspot_cells / np.sum(~np.isnan(data)) * 100
                
                results[name] = {
                    'threshold': threshold,
                    'hotspot_cells': hotspot_cells,
                    'hotspot_percent': hotspot_percent,
                    'mean_richness': np.mean(data[hotspot_mask])
                }
                
                print(f"  {name}: {hotspot_cells} cells (>{threshold:.1f} species)")
        
        # Calculate hotspot overlap
        print("\nHotspot overlap:")
        dataset_names = list(hotspot_masks.keys())
        
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i < j:
                    overlap = hotspot_masks[name1] & hotspot_masks[name2]
                    overlap_cells = np.sum(overlap)
                    
                    # Calculate overlap percentage relative to smaller hotspot
                    smaller_hotspot = min(np.sum(hotspot_masks[name1]), 
                                        np.sum(hotspot_masks[name2]))
                    
                    if smaller_hotspot > 0:
                        overlap_percent = overlap_cells / smaller_hotspot * 100
                    else:
                        overlap_percent = 0
                    
                    pair_key = f"{name1}_vs_{name2}_hotspots"
                    results[pair_key] = {
                        'overlap_cells': overlap_cells,
                        'overlap_percent': overlap_percent
                    }
                    
                    print(f"  {name1} ∩ {name2}: {overlap_percent:.1f}% of smaller hotspot")
        
        # All hotspots overlap
        if len(hotspot_masks) > 0:
            all_hotspots = np.logical_and.reduce(list(hotspot_masks.values()))
            results['all_hotspots_overlap'] = {
                'cells': np.sum(all_hotspots),
                'locations': np.where(all_hotspots)  # Row, col indices
            }
            print(f"\nAll hotspots overlap: {np.sum(all_hotspots)} cells")
        
        return results
    
    def create_overlap_visualizations(self) -> None:
        """Create comprehensive overlap visualizations"""
        
        if len(self.aligned_data) < 2:
            print("Need at least 2 datasets for overlap visualization")
            return
        
        # Create figure with multiple panels
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        dataset_names = list(self.aligned_data.keys())
        
        # Panel 1: Coverage overlap map
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Create coverage sum map
        coverage_sum = np.zeros_like(list(self.aligned_data.values())[0])
        for data in self.aligned_data.values():
            valid_mask = ~np.isnan(data) & (data > 0)
            coverage_sum += valid_mask.astype(int)
        
        im1 = ax1.imshow(coverage_sum, cmap='YlOrRd', vmin=0, vmax=len(self.aligned_data))
        ax1.set_title('Data Coverage Overlap')
        ax1.set_xlabel('Longitude index')
        ax1.set_ylabel('Latitude index')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('# Datasets')
        
        # Panel 2: All valid data mask
        ax2 = fig.add_subplot(gs[0, 1])
        
        all_valid = np.ones_like(coverage_sum, dtype=bool)
        for data in self.aligned_data.values():
            all_valid &= (~np.isnan(data) & (data > 0))
        
        ax2.imshow(all_valid, cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_title(f'All Datasets Valid\n({np.sum(all_valid)} cells, {np.sum(all_valid)/all_valid.size*100:.1f}%)')
        ax2.set_xlabel('Longitude index')
        ax2.set_ylabel('Latitude index')
        
        # Panel 3: Concordance heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Create correlation matrix
        n = len(dataset_names)
        corr_matrix = np.ones((n, n))
        
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i != j:
                    data1 = self.aligned_data[name1].flatten()
                    data2 = self.aligned_data[name2].flatten()
                    valid = ~np.isnan(data1) & ~np.isnan(data2) & (data1 > 0) & (data2 > 0)
                    
                    if np.sum(valid) > 10:
                        corr, _ = stats.spearmanr(data1[valid], data2[valid])
                        corr_matrix[i, j] = corr
        
        im3 = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax3.set_xticks(range(n))
        ax3.set_yticks(range(n))
        ax3.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax3.set_yticklabels(dataset_names)
        ax3.set_title('Cross-taxa Concordance\n(Spearman correlation)')
        
        # Add correlation values as text
        for i in range(n):
            for j in range(n):
                if i != j:
                    text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
        
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Correlation')
        
        # Panels 4-7: Individual datasets
        for idx, (name, data) in enumerate(self.aligned_data.items()):
            if idx < 4:
                ax = fig.add_subplot(gs[1, idx % 3])
                
                # Mask invalid data
                masked_data = np.ma.masked_invalid(data)
                masked_data = np.ma.masked_equal(masked_data, 0)
                
                # Determine colormap
                cmap = 'viridis' if 'fungi' in name else 'YlOrRd'
                
                im = ax.imshow(masked_data, cmap=cmap)
                ax.set_title(f'{name}\n({"SDM" if "fungi" in name else "Observed"})')
                ax.set_xlabel('Longitude index')
                if idx % 3 == 0:
                    ax.set_ylabel('Latitude index')
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Richness')
        
        # Panel 8: Hotspot overlap
        ax8 = fig.add_subplot(gs[2, 0])
        
        # Calculate hotspots
        hotspot_sum = np.zeros_like(coverage_sum)
        for name, data in self.aligned_data.items():
            valid_data = data[~np.isnan(data) & (data > 0)]
            if len(valid_data) > 0:
                threshold = np.percentile(valid_data, 90)
                hotspot_mask = (data >= threshold) & ~np.isnan(data)
                hotspot_sum += hotspot_mask.astype(int)
        
        im8 = ax8.imshow(hotspot_sum, cmap='hot', vmin=0, vmax=len(self.aligned_data))
        ax8.set_title('Hotspot Overlap\n(Top 10% richness)')
        ax8.set_xlabel('Longitude index')
        ax8.set_ylabel('Latitude index')
        cbar8 = plt.colorbar(im8, ax=ax8)
        cbar8.set_label('# Datasets')
        
        # Panel 9: Agreement map (SDM vs Observed)
        ax9 = fig.add_subplot(gs[2, 1])
        
        # Separate SDM and observed datasets
        sdm_data = [data for name, data in self.aligned_data.items() if 'fungi' in name]
        obs_data = [data for name, data in self.aligned_data.items() if 'fungi' not in name]
        
        if len(sdm_data) > 0 and len(obs_data) > 0:
            # Average within each group
            sdm_mean = np.nanmean(np.stack(sdm_data), axis=0)
            obs_mean = np.nanmean(np.stack(obs_data), axis=0)
            
            # Normalize to compare
            sdm_norm = (sdm_mean - np.nanmin(sdm_mean)) / (np.nanmax(sdm_mean) - np.nanmin(sdm_mean))
            obs_norm = (obs_mean - np.nanmin(obs_mean)) / (np.nanmax(obs_mean) - np.nanmin(obs_mean))
            
            # Calculate difference
            diff = sdm_norm - obs_norm
            
            im9 = ax9.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
            ax9.set_title('SDM vs Observed\n(Normalized difference)')
            ax9.set_xlabel('Longitude index')
            ax9.set_ylabel('Latitude index')
            cbar9 = plt.colorbar(im9, ax=ax9)
            cbar9.set_label('SDM - Observed')
        
        # Panel 10: Summary statistics
        ax10 = fig.add_subplot(gs[2, 2])
        ax10.axis('off')
        
        # Calculate summary stats
        stats_text = "OVERLAP SUMMARY\n" + "="*20 + "\n\n"
        stats_text += f"Total cells: {coverage_sum.size:,}\n"
        stats_text += f"Any data: {np.sum(coverage_sum > 0):,} ({np.sum(coverage_sum > 0)/coverage_sum.size*100:.1f}%)\n"
        stats_text += f"All data: {np.sum(all_valid):,} ({np.sum(all_valid)/all_valid.size*100:.1f}%)\n\n"
        
        stats_text += "Pairwise Overlap:\n"
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i < j:
                    mask1 = ~np.isnan(self.aligned_data[name1]) & (self.aligned_data[name1] > 0)
                    mask2 = ~np.isnan(self.aligned_data[name2]) & (self.aligned_data[name2] > 0)
                    overlap = np.sum(mask1 & mask2) / mask1.size * 100
                    stats_text += f"  {name1[:4]}-{name2[:4]}: {overlap:.1f}%\n"
        
        ax10.text(0.1, 0.9, stats_text, transform=ax10.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Biodiversity Dataset Overlap Analysis', fontsize=16, fontweight='bold')
        
        # Save figure
        save_path = self.output_dir / "overlap_analysis.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved overlap visualization to: {save_path}")
        
        plt.close()
    
    def generate_overlap_report(self) -> str:
        """Generate comprehensive overlap analysis report"""
        
        report = []
        report.append("=" * 80)
        report.append("BIODIVERSITY DATASET OVERLAP ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Coverage summary
        report.append("INDIVIDUAL DATASET COVERAGE")
        report.append("-" * 40)
        
        for name, data in self.aligned_data.items():
            valid_mask = ~np.isnan(data) & (data > 0)
            coverage = np.sum(valid_mask) / data.size * 100
            report.append(f"  {name:20s}: {coverage:6.2f}% ({np.sum(valid_mask):,} cells)")
        
        report.append("")
        
        # Overlap statistics
        report.append("PAIRWISE OVERLAP STATISTICS")
        report.append("-" * 40)
        
        dataset_names = list(self.aligned_data.keys())
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i < j:
                    mask1 = ~np.isnan(self.aligned_data[name1]) & (self.aligned_data[name1] > 0)
                    mask2 = ~np.isnan(self.aligned_data[name2]) & (self.aligned_data[name2] > 0)
                    
                    both = mask1 & mask2
                    either = mask1 | mask2
                    
                    overlap_pct = np.sum(both) / data.size * 100
                    jaccard = np.sum(both) / np.sum(either) if np.sum(either) > 0 else 0
                    
                    report.append(f"  {name1} ∩ {name2}:")
                    report.append(f"    Overlap: {overlap_pct:.2f}% of total area")
                    report.append(f"    Jaccard: {jaccard:.3f}")
        
        # All dataset overlap
        all_valid = np.ones_like(list(self.aligned_data.values())[0], dtype=bool)
        for data in self.aligned_data.values():
            all_valid &= (~np.isnan(data) & (data > 0))
        
        report.append("")
        report.append(f"All datasets overlap: {np.sum(all_valid):,} cells ({np.sum(all_valid)/all_valid.size*100:.2f}%)")
        
        report.append("")
        report.append("CROSS-TAXA CONCORDANCE")
        report.append("-" * 40)
        
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i < j:
                    data1 = self.aligned_data[name1].flatten()
                    data2 = self.aligned_data[name2].flatten()
                    
                    valid = ~np.isnan(data1) & ~np.isnan(data2) & (data1 > 0) & (data2 > 0)
                    
                    if np.sum(valid) > 10:
                        spearman_r, spearman_p = stats.spearmanr(data1[valid], data2[valid])
                        
                        is_mixed = ('fungi' in name1) != ('fungi' in name2)
                        method = "SDM vs Observed" if is_mixed else "Same method"
                        
                        report.append(f"  {name1} ~ {name2} ({method}):")
                        report.append(f"    Spearman r = {spearman_r:.3f} (p < {spearman_p:.2e})")
                        report.append(f"    Interpretation: {self._interpret_correlation(spearman_r, is_mixed)}")
        
        report.append("")
        report.append("KEY FINDINGS")
        report.append("-" * 40)
        
        # Find key patterns
        findings = []
        
        # Check if fungi overlap with plants (host relationships)
        if 'plants' in dataset_names and any('fungi' in n for n in dataset_names):
            plant_data = self.aligned_data['plants'].flatten()
            
            for fungi_name in [n for n in dataset_names if 'fungi' in n]:
                fungi_data = self.aligned_data[fungi_name].flatten()
                valid = ~np.isnan(plant_data) & ~np.isnan(fungi_data) & (plant_data > 0) & (fungi_data > 0)
                
                if np.sum(valid) > 10:
                    r, _ = stats.spearmanr(plant_data[valid], fungi_data[valid])
                    if abs(r) < 0.3:
                        findings.append(f"⚠️  Weak correlation between {fungi_name} and plants (r={r:.2f})")
                        findings.append("   This is concerning as fungi depend on plant hosts!")
        
        # Check method consistency
        sdm_names = [n for n in dataset_names if 'fungi' in n]
        obs_names = [n for n in dataset_names if 'fungi' not in n]
        
        if len(sdm_names) >= 2:
            # Check SDM consistency
            data1 = self.aligned_data[sdm_names[0]].flatten()
            data2 = self.aligned_data[sdm_names[1]].flatten()
            valid = ~np.isnan(data1) & ~np.isnan(data2) & (data1 > 0) & (data2 > 0)
            if np.sum(valid) > 10:
                r, _ = stats.spearmanr(data1[valid], data2[valid])
                findings.append(f"SDM consistency: {sdm_names[0]} ~ {sdm_names[1]} (r={r:.2f})")
        
        if len(obs_names) >= 2:
            # Check observed consistency
            data1 = self.aligned_data[obs_names[0]].flatten()
            data2 = self.aligned_data[obs_names[1]].flatten()
            valid = ~np.isnan(data1) & ~np.isnan(data2) & (data1 > 0) & (data2 > 0)
            if np.sum(valid) > 10:
                r, _ = stats.spearmanr(data1[valid], data2[valid])
                findings.append(f"Observed consistency: {obs_names[0]} ~ {obs_names[1]} (r={r:.2f})")
        
        # Coverage gaps
        if np.sum(all_valid) / all_valid.size < 0.15:
            findings.append(f"⚠️  Only {np.sum(all_valid)/all_valid.size*100:.1f}% of area has all datasets")
            findings.append("   Limited area for comprehensive analysis!")
        
        for finding in findings:
            report.append(finding)
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)