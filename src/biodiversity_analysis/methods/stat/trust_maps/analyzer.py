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
    
    def load_and_align_data(self, datasets: Dict[str, str], sample_factor: int = 10) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load and align all datasets to common dimensions (with spatial sampling for large files)"""
        aligned_data = {}
        min_height, min_width = float('inf'), float('inf')
        
        print(f"Loading datasets with {sample_factor}x spatial sampling...")
        
        # First pass: find minimum dimensions
        for name, path in datasets.items():
            with rasterio.open(path) as src:
                min_height = min(min_height, src.height)
                min_width = min(min_width, src.width)
        
        print(f"Original dimensions: {min_height} x {min_width}")
        
        # Apply spatial sampling
        sample_height = max(500, min_height // sample_factor)
        sample_width = max(500, min_width // sample_factor)
        
        print(f"Sampled dimensions: {sample_height} x {sample_width} (factor: {sample_factor}x)")
        
        # Second pass: load and sample data
        for name, path in datasets.items():
            with rasterio.open(path) as src:
                data = src.read(1).astype(np.float32)
                
                # Crop to common dimensions first
                data = data[:min_height, :min_width]
                
                # Apply spatial sampling (every nth pixel)
                step_h = max(1, min_height // sample_height)
                step_w = max(1, min_width // sample_width)
                
                sampled_data = data[::step_h, ::step_w]
                aligned_data[name] = sampled_data
                
                print(f"  {name}: {data.shape} -> {sampled_data.shape}")
        
        # Create combined valid mask
        valid_mask = None
        for data in aligned_data.values():
            mask = ~np.isnan(data) & (data > 0)
            valid_mask = mask if valid_mask is None else valid_mask & mask
        
        return aligned_data, valid_mask
    
    def create_trust_maps(self, datasets: Dict[str, str]) -> None:
        """Create individual trust map visualizations as separate PNG files"""
        print("Creating individual trust maps...")
        
        # Load and align data
        data, valid_mask = self.load_and_align_data(datasets)
        
        # Generate individual trust maps
        self.create_data_coverage_map(data, valid_mask)
        self.create_pairwise_agreement_maps(data)
        self.create_hotspot_agreement_map(data)
        self.create_method_compatibility_map(data)
        self.create_combined_trust_score_map(data)
        self.create_conflict_zones_map(data)
        
        print(f"All trust maps saved to: {self.output_dir}")
    
    def create_data_coverage_map(self, data: Dict[str, np.ndarray], valid_mask: np.ndarray) -> None:
        """Create map showing data coverage overlap"""
        print("  Creating data coverage trust map...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate coverage overlap
        coverage_score = np.zeros_like(valid_mask, dtype=float)
        
        for name, array in data.items():
            has_data = (array > 0) & ~np.isnan(array)
            coverage_score += has_data.astype(float)
        
        # Normalize to 0-100%
        coverage_score = (coverage_score / len(data)) * 100
        
        # Create the map
        im = ax.imshow(coverage_score, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        
        ax.set_title('Data Coverage Trust Map\n(% of datasets with valid data)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Coverage Score (%)', fontsize=12)
        
        # Add statistics
        high_coverage = np.sum(coverage_score >= 75)
        medium_coverage = np.sum((coverage_score >= 50) & (coverage_score < 75))
        low_coverage = np.sum((coverage_score > 0) & (coverage_score < 50))
        
        stats_text = f"High trust (≥75%): {high_coverage:,} pixels\nMedium trust (50-75%): {medium_coverage:,} pixels\nLow trust (<50%): {low_coverage:,} pixels"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        ax.set_xlabel('Longitude (pixel)')
        ax.set_ylabel('Latitude (pixel)')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "data_coverage_trust_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
    
    def create_pairwise_agreement_maps(self, data: Dict[str, np.ndarray]) -> None:
        """Create maps for each pairwise dataset comparison"""
        print("  Creating pairwise agreement maps...")
        
        dataset_names = list(data.keys())
        
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i >= j:  # Skip duplicate comparisons
                    continue
                
                self._create_single_pairwise_map(data[name1], data[name2], name1, name2)
    
    def _create_single_pairwise_map(self, data1: np.ndarray, data2: np.ndarray, 
                                   name1: str, name2: str) -> None:
        """Create a single pairwise comparison map"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate agreement/disagreement
        valid_both = (data1 > 0) & (data2 > 0) & ~np.isnan(data1) & ~np.isnan(data2)
        
        # Create agreement score based on relative values
        agreement_map = np.full_like(data1, np.nan)
        
        # Where both have data, calculate relative agreement
        where_valid = valid_both
        if np.any(where_valid):
            # Normalize each dataset to 0-1 scale within valid region
            d1_norm = np.zeros_like(data1)
            d2_norm = np.zeros_like(data2)
            
            d1_valid = data1[where_valid]
            d2_valid = data2[where_valid]
            
            if len(d1_valid) > 0 and np.max(d1_valid) > 0:
                d1_norm[where_valid] = d1_valid / np.max(d1_valid)
            if len(d2_valid) > 0 and np.max(d2_valid) > 0:
                d2_norm[where_valid] = d2_valid / np.max(d2_valid)
            
            # Calculate agreement (0 = perfect disagreement, 1 = perfect agreement)
            diff = np.abs(d1_norm - d2_norm)
            agreement_map[where_valid] = 1 - diff[where_valid]
        
        # Create discrete agreement categories
        agreement_categories = np.full_like(agreement_map, 0, dtype=int)  # 0 = no data
        agreement_categories[valid_both & (agreement_map >= 0.8)] = 4  # High agreement
        agreement_categories[valid_both & (agreement_map >= 0.6) & (agreement_map < 0.8)] = 3  # Medium agreement
        agreement_categories[valid_both & (agreement_map >= 0.4) & (agreement_map < 0.6)] = 2  # Low agreement  
        agreement_categories[valid_both & (agreement_map < 0.4)] = 1  # Disagreement
        
        # Create custom colormap
        colors = ['white', 'red', 'orange', 'yellow', 'green']
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        im = ax.imshow(agreement_categories, cmap=cmap, vmin=0, vmax=4, aspect='auto')
        
        # Clean up names for display
        display_name1 = name1.replace('_', ' ').title()
        display_name2 = name2.replace('_', ' ').title()
        
        ax.set_title(f'Agreement Map: {display_name1} vs {display_name2}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Custom legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', label='No Data'),
            plt.Rectangle((0,0),1,1, facecolor='red', label='Strong Disagreement'),
            plt.Rectangle((0,0),1,1, facecolor='orange', label='Moderate Disagreement'), 
            plt.Rectangle((0,0),1,1, facecolor='yellow', label='Weak Agreement'),
            plt.Rectangle((0,0),1,1, facecolor='green', label='Strong Agreement')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add statistics
        total_valid = np.sum(valid_both)
        if total_valid > 0:
            high_agree = np.sum(agreement_categories == 4)
            disagree = np.sum(agreement_categories == 1)
            
            stats_text = f"Valid pixels: {total_valid:,}\nStrong agreement: {high_agree:,} ({high_agree/total_valid*100:.1f}%)\nStrong disagreement: {disagree:,} ({disagree/total_valid*100:.1f}%)"
        else:
            stats_text = "No overlapping valid data"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        ax.set_xlabel('Longitude (pixel)')
        ax.set_ylabel('Latitude (pixel)')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"agreement_map_{name1}_vs_{name2}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
    
    def create_hotspot_agreement_map(self, data: Dict[str, np.ndarray]) -> None:
        """Create map showing where multiple datasets identify hotspots"""
        print("  Creating hotspot agreement map...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Identify hotspots in each dataset (top 10%)
        hotspot_map = np.zeros_like(list(data.values())[0], dtype=int)
        
        for name, array in data.items():
            valid_data = array[(array > 0) & ~np.isnan(array)]
            if len(valid_data) > 0:
                threshold = np.percentile(valid_data, 90)  # Top 10%
                is_hotspot = (array >= threshold) & (array > 0)
                hotspot_map += is_hotspot.astype(int)
        
        # Create the map
        im = ax.imshow(hotspot_map, cmap='Reds', vmin=0, vmax=len(data), aspect='auto')
        
        ax.set_title('Biodiversity Hotspot Agreement Map\n(Number of datasets identifying each area as high-diversity)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Number of Datasets Agreeing', fontsize=12)
        
        # Add statistics
        unanimous = np.sum(hotspot_map == len(data))
        majority = np.sum(hotspot_map >= len(data) // 2)
        
        stats_text = f"Unanimous hotspots ({len(data)}/4): {unanimous:,} pixels\nMajority agreement (≥{len(data)//2}): {majority:,} pixels"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        ax.set_xlabel('Longitude (pixel)')
        ax.set_ylabel('Latitude (pixel)')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "hotspot_agreement_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
    
    def create_method_compatibility_map(self, data: Dict[str, np.ndarray]) -> None:
        """Create map showing observed data vs SDM compatibility"""
        print("  Creating method compatibility map...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Separate observed vs SDM
        observed_data = {}
        sdm_data = {}
        
        for name, array in data.items():
            if 'fungi' in name:
                sdm_data[name] = array
            else:
                observed_data[name] = array
        
        # Calculate method compatibility
        compatibility_map = np.full_like(list(data.values())[0], np.nan)
        
        if len(observed_data) > 0 and len(sdm_data) > 0:
            # Average observed and SDM data
            obs_avg = np.zeros_like(list(observed_data.values())[0])
            obs_count = np.zeros_like(obs_avg)
            
            for array in observed_data.values():
                valid = (array > 0) & ~np.isnan(array)
                obs_avg[valid] += array[valid]
                obs_count[valid] += 1
            
            obs_avg = np.divide(obs_avg, obs_count, out=np.zeros_like(obs_avg), where=obs_count>0)
            
            sdm_avg = np.zeros_like(list(sdm_data.values())[0])
            sdm_count = np.zeros_like(sdm_avg)
            
            for array in sdm_data.values():
                valid = (array > 0) & ~np.isnan(array)
                sdm_avg[valid] += array[valid]
                sdm_count[valid] += 1
            
            sdm_avg = np.divide(sdm_avg, sdm_count, out=np.zeros_like(sdm_avg), where=sdm_count>0)
            
            # Calculate compatibility where both methods have data
            both_valid = (obs_count > 0) & (sdm_count > 0)
            
            if np.any(both_valid):
                obs_norm = obs_avg / np.max(obs_avg[both_valid]) if np.max(obs_avg[both_valid]) > 0 else obs_avg
                sdm_norm = sdm_avg / np.max(sdm_avg[both_valid]) if np.max(sdm_avg[both_valid]) > 0 else sdm_avg
                
                compatibility_map[both_valid] = 1 - np.abs(obs_norm[both_valid] - sdm_norm[both_valid])
        
        # Create the map
        im = ax.imshow(compatibility_map, cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')
        
        ax.set_title('Method Compatibility Map\n(Observed Counts vs SDM Predictions)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Compatibility Score (0=incompatible, 1=perfect)', fontsize=12)
        
        # Add statistics
        valid_comparisons = ~np.isnan(compatibility_map)
        if np.any(valid_comparisons):
            high_compat = np.sum(compatibility_map >= 0.8)
            low_compat = np.sum(compatibility_map <= 0.2)
            total_valid = np.sum(valid_comparisons)
            
            stats_text = f"Comparable pixels: {total_valid:,}\nHigh compatibility (≥0.8): {high_compat:,} ({high_compat/total_valid*100:.1f}%)\nLow compatibility (≤0.2): {low_compat:,} ({low_compat/total_valid*100:.1f}%)"
        else:
            stats_text = "No comparable data between methods"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        ax.set_xlabel('Longitude (pixel)')
        ax.set_ylabel('Latitude (pixel)')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "method_compatibility_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
    
    def create_combined_trust_score_map(self, data: Dict[str, np.ndarray]) -> None:
        """Create combined trust score integrating multiple metrics"""
        print("  Creating combined trust score map...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Initialize combined score
        combined_score = np.zeros_like(list(data.values())[0])
        weight_sum = np.zeros_like(combined_score)
        
        # 1. Data availability score (40% weight)
        availability_score = np.zeros_like(combined_score)
        for array in data.values():
            has_data = (array > 0) & ~np.isnan(array)
            availability_score += has_data.astype(float)
        
        availability_score = availability_score / len(data)  # Normalize to 0-1
        combined_score += availability_score * 0.4
        weight_sum += 0.4
        
        # 2. Agreement score (30% weight) - calculate pairwise agreements
        if len(data) > 1:
            agreement_scores = []
            dataset_arrays = list(data.values())
            
            for i in range(len(dataset_arrays)):
                for j in range(i + 1, len(dataset_arrays)):
                    d1, d2 = dataset_arrays[i], dataset_arrays[j]
                    
                    # Find overlap
                    valid_both = (d1 > 0) & (d2 > 0) & ~np.isnan(d1) & ~np.isnan(d2)
                    
                    if np.any(valid_both):
                        # Normalize and calculate agreement
                        d1_norm = np.zeros_like(d1)
                        d2_norm = np.zeros_like(d2)
                        
                        d1_valid = d1[valid_both]
                        d2_valid = d2[valid_both]
                        
                        if np.max(d1_valid) > 0:
                            d1_norm[valid_both] = d1_valid / np.max(d1_valid)
                        if np.max(d2_valid) > 0:
                            d2_norm[valid_both] = d2_valid / np.max(d2_valid)
                        
                        # Agreement as inverse of absolute difference
                        agreement = np.full_like(d1, 0.0)
                        agreement[valid_both] = 1 - np.abs(d1_norm[valid_both] - d2_norm[valid_both])
                        agreement_scores.append(agreement)
            
            if agreement_scores:
                avg_agreement = np.mean(agreement_scores, axis=0)
                combined_score += avg_agreement * 0.3
                weight_sum += 0.3
        
        # 3. Hotspot consensus score (20% weight)
        hotspot_score = np.zeros_like(combined_score)
        for array in data.values():
            valid_data = array[(array > 0) & ~np.isnan(array)]
            if len(valid_data) > 0:
                threshold = np.percentile(valid_data, 85)  # Top 15% as hotspots
                is_hotspot = (array >= threshold) & (array > 0)
                hotspot_score += is_hotspot.astype(float)
        
        hotspot_score = hotspot_score / len(data)  # Normalize
        combined_score += hotspot_score * 0.2
        weight_sum += 0.2
        
        # 4. Data quality score (10% weight) - based on value ranges
        quality_score = np.zeros_like(combined_score)
        for array in data.values():
            valid = (array > 0) & ~np.isnan(array)
            if np.any(valid):
                # Higher values generally indicate better data quality
                normalized_values = array / np.max(array[valid]) if np.max(array[valid]) > 0 else array
                quality_score += normalized_values * valid.astype(float)
        
        if len(data) > 0:
            quality_score = quality_score / len(data)
            combined_score += quality_score * 0.1
            weight_sum += 0.1
        
        # Normalize by total weights
        final_score = np.divide(combined_score, weight_sum, 
                               out=np.zeros_like(combined_score), where=weight_sum>0)
        
        # Create the map
        im = ax.imshow(final_score, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        ax.set_title('Combined Trust Score Map\n(Weighted integration of coverage, agreement, hotspots, quality)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Combined Trust Score (0=low trust, 1=high trust)', fontsize=12)
        
        # Add statistics
        valid_scores = final_score[final_score > 0]
        if len(valid_scores) > 0:
            high_trust = np.sum(final_score >= 0.7)
            medium_trust = np.sum((final_score >= 0.4) & (final_score < 0.7))
            low_trust = np.sum((final_score > 0) & (final_score < 0.4))
            
            stats_text = f"High trust (≥0.7): {high_trust:,} pixels\nMedium trust (0.4-0.7): {medium_trust:,} pixels\nLow trust (<0.4): {low_trust:,} pixels\n\nWeights: Coverage(40%) + Agreement(30%) + Hotspots(20%) + Quality(10%)"
        else:
            stats_text = "No trust scores calculated"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        ax.set_xlabel('Longitude (pixel)')
        ax.set_ylabel('Latitude (pixel)')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "combined_trust_score_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
    
    def create_conflict_zones_map(self, data: Dict[str, np.ndarray]) -> None:
        """Create map highlighting areas with high disagreement between datasets"""
        print("  Creating conflict zones map...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if len(data) < 2:
            ax.text(0.5, 0.5, "Need at least 2 datasets to identify conflicts", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Conflict Zones Map', fontsize=14, fontweight='bold')
            plt.savefig(self.output_dir / "conflict_zones_map.png", dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Calculate disagreement metrics
        dataset_arrays = list(data.values())
        
        # Normalize all datasets to 0-1 scale
        normalized_arrays = []
        for array in dataset_arrays:
            valid = (array > 0) & ~np.isnan(array)
            norm_array = np.full_like(array, np.nan)
            
            if np.any(valid):
                min_val = np.min(array[valid])
                max_val = np.max(array[valid])
                if max_val > min_val:
                    norm_array[valid] = (array[valid] - min_val) / (max_val - min_val)
                else:
                    norm_array[valid] = 0.5  # If all values same, set to middle
            
            normalized_arrays.append(norm_array)
        
        # Calculate coefficient of variation as disagreement measure
        stack = np.stack(normalized_arrays, axis=0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_vals = np.nanmean(stack, axis=0)
            std_vals = np.nanstd(stack, axis=0)
            
            # Coefficient of variation (higher = more disagreement)
            cv = np.divide(std_vals, mean_vals, out=np.full_like(mean_vals, np.nan), 
                          where=(mean_vals > 0))
        
        # Identify conflict zones (high CV areas with data from multiple sources)
        data_count = np.sum(~np.isnan(stack), axis=0)
        conflict_threshold = np.nanpercentile(cv[data_count >= 2], 75)  # Top 25% of CV values
        
        conflict_zones = np.full_like(cv, 0.0)
        
        # Strong conflicts: high CV, many datasets
        strong_conflict = (cv >= conflict_threshold) & (data_count >= 3)
        moderate_conflict = (cv >= conflict_threshold * 0.7) & (data_count >= 2) & ~strong_conflict
        
        conflict_zones[strong_conflict] = 2.0  # Strong conflict
        conflict_zones[moderate_conflict] = 1.0  # Moderate conflict
        
        # Create discrete conflict map
        from matplotlib.colors import ListedColormap
        colors = ['white', 'orange', 'red']
        cmap = ListedColormap(colors)
        
        im = ax.imshow(conflict_zones, cmap=cmap, vmin=0, vmax=2, aspect='auto')
        
        ax.set_title('Conflict Zones Map\n(Areas of high disagreement between datasets)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Custom legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', label='No Conflict/Data'),
            plt.Rectangle((0,0),1,1, facecolor='orange', label='Moderate Conflict'),
            plt.Rectangle((0,0),1,1, facecolor='red', label='Strong Conflict')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add statistics
        strong_conflicts = np.sum(conflict_zones == 2.0)
        moderate_conflicts = np.sum(conflict_zones == 1.0)
        total_with_data = np.sum(data_count >= 2)
        
        if total_with_data > 0:
            stats_text = f"Areas with multiple datasets: {total_with_data:,} pixels\nModerate conflicts: {moderate_conflicts:,} ({moderate_conflicts/total_with_data*100:.1f}%)\nStrong conflicts: {strong_conflicts:,} ({strong_conflicts/total_with_data*100:.1f}%)\n\nConflict = High coefficient of variation"
        else:
            stats_text = "No areas with multiple datasets for conflict analysis"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        ax.set_xlabel('Longitude (pixel)')
        ax.set_ylabel('Latitude (pixel)')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "conflict_zones_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
    
    # Keep the old method name for compatibility but redirect to new method
    def create_trust_map(self, datasets: Dict[str, str]) -> None:
        """Legacy method name - redirects to create_trust_maps"""
        return self.create_trust_maps(datasets)


def main():
    """Main function to create trust maps."""
    datasets = {
        'plants': '/scratch/yl998/downsampled_rasters/plants-richness_downsampled_0.16667.tif',
        'terrestrial': '/scratch/yl998/downsampled_rasters/terrestrial-richness_downsampled_0.16667.tif',
        'am_fungi': '/scratch/yl998/downsampled_rasters/am-fungi-richness_downsampled_0.16667.tif',
        'ecm_fungi': '/scratch/yl998/downsampled_rasters/ecm-fungi-richness_downsampled_0.16667.tif'
    }
    
    visualizer = TrustMapVisualizer()
    visualizer.create_trust_maps(datasets)


if __name__ == "__main__":
    main()
