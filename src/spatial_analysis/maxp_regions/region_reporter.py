# src/spatial_analysis/maxp_regions/region_reporter.py
"""Generate reports for Max-p regionalization results."""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.spatial_analysis.base_analyzer import AnalysisResult

logger = logging.getLogger(__name__)

class RegionReporter:
    """Generate statistical reports for Max-p regionalization."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path('reports/maxp')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_region_summary(self, result: AnalysisResult) -> pd.DataFrame:
        """
        Generate summary statistics for each region.
        
        Args:
            result: Max-p analysis result
            
        Returns:
            DataFrame with region statistics
        """
        region_stats = result.statistics['region_statistics']
        band_names = result.metadata.input_bands
        
        # Build summary data
        summary_data = []
        compactness = {}
        homogeneity = {}
        if result.additional_outputs is not None:
            compactness = result.additional_outputs.get('compactness_scores', {})
            homogeneity = result.additional_outputs.get('homogeneity_scores', {})
        
        for region_id, stats in region_stats.items():
            row = {
                'region_id': region_id,
                'area_km2': stats['area_km2'],
                'pixel_count': stats['pixel_count'],
                'percentage': stats['percentage_of_total'],
                'within_variance': stats['within_variance'],
                'compactness': compactness.get(region_id, np.nan),
                'homogeneity': homogeneity.get(region_id, np.nan)
            }
            
            # Add mean values for each band
            for i, band in enumerate(band_names):
                if i < len(stats['mean']):
                    row[f'{band}_mean'] = stats['mean'][i]
                    row[f'{band}_std'] = stats['std'][i]
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('area_km2', ascending=False)
        
        return df
    
    def generate_quality_report(self, result: AnalysisResult) -> Dict[str, Any]:
        """
        Generate quality metrics for the regionalization.
        
        Args:
            result: Max-p analysis result
            
        Returns:
            Dictionary with quality metrics
        """
        stats = result.statistics
        
        # Area-based metrics
        areas = [r['area_km2'] for r in stats['region_statistics'].values()]
        area_cv = stats.get('area_coefficient_variation', 0.0)  # Use fallback if missing
        
        # Calculate average compactness
        compactness_scores = []
        if result.additional_outputs is not None and 'compactness_scores' in result.additional_outputs:
            compactness_scores = list(result.additional_outputs['compactness_scores'].values())
        avg_compactness = np.mean(compactness_scores) if compactness_scores else 0.0
        
        # Calculate average homogeneity
        homogeneity_scores = []
        if result.additional_outputs is not None and 'homogeneity_scores' in result.additional_outputs:
            homogeneity_scores = list(result.additional_outputs['homogeneity_scores'].values())
        avg_homogeneity = np.mean(homogeneity_scores) if homogeneity_scores else 0.0
        
        quality = {
            'variance_explained': stats['variance_explained'],
            'n_regions': stats['n_regions'],
            'ecological_scale': stats['ecological_scale'],
            'min_area_threshold_km2': stats['min_area_threshold_km2'],
            'threshold_satisfied': stats['threshold_satisfied'],
            'size_balance': 1 - area_cv,  # Higher is better
            'avg_compactness': avg_compactness,
            'avg_homogeneity': avg_homogeneity,
            'smallest_region_km2': stats['smallest_region_km2'],
            'largest_region_km2': stats['largest_region_km2'],
            'mean_region_area_km2': stats['mean_region_area_km2']
        }
        
        # Add perturbation analysis results if available
        if result.additional_outputs is not None and 'perturbation_results' in result.additional_outputs:
            pert = result.additional_outputs['perturbation_results']
            if pert:
                quality['stability_assessment'] = pert['stability_assessment']
                quality['avg_boundary_stability'] = np.mean(pert['boundary_stability'])
        
        return quality
    
    def plot_perturbation_analysis(self, result: AnalysisResult,
                                 save_path: Optional[Path] = None) -> Optional[plt.Figure]:
        """
        Plot perturbation analysis results.
        
        Args:
            result: Max-p analysis result
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure or None if no perturbation results
        """
        if result.additional_outputs is None or 'perturbation_results' not in result.additional_outputs:
            return None
        
        pert = result.additional_outputs['perturbation_results']
        if not pert:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Number of regions vs threshold
        ax = axes[0, 0]
        ax.plot(pert['tested_areas_km2'], pert['n_regions'], 'o-', markersize=8)
        ax.axvline(pert['baseline_area_km2'], color='red', linestyle='--', 
                  label='Baseline threshold')
        ax.set_xlabel('Minimum Area Threshold (km²)')
        ax.set_ylabel('Number of Regions')
        ax.set_title('Region Count Sensitivity')
        ax.legend()
        
        # 2. Boundary stability
        ax = axes[0, 1]
        ax.plot(pert['tested_areas_km2'], pert['boundary_stability'], 'o-', markersize=8)
        ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='High stability')
        ax.axhline(0.6, color='orange', linestyle='--', alpha=0.5, label='Moderate stability')
        ax.set_xlabel('Minimum Area Threshold (km²)')
        ax.set_ylabel('Boundary Stability')
        ax.set_title('Boundary Stability Analysis')
        ax.set_ylim(0, 1)
        ax.legend()
        
        # 3. Region correspondence
        ax = axes[1, 0]
        ax.plot(pert['tested_areas_km2'], pert['region_correspondence'], 'o-', markersize=8)
        ax.set_xlabel('Minimum Area Threshold (km²)')
        ax.set_ylabel('Rand Index')
        ax.set_title('Region Correspondence')
        ax.set_ylim(0, 1)
        
        # 4. Summary text
        ax = axes[1, 1]
        ax.text(0.1, 0.9, f"Stability Assessment:\n{pert['stability_assessment']}", 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top')
        
        avg_stability = np.mean(pert['boundary_stability'])
        summary_text = (
            f"\nAverage Boundary Stability: {avg_stability:.3f}\n"
            f"Tested Range: {min(pert['tested_areas_km2']):.0f} - "
            f"{max(pert['tested_areas_km2']):.0f} km²\n"
            f"Region Count Range: {min(pert['n_regions'])} - {max(pert['n_regions'])}"
        )
        
        ax.text(0.1, 0.6, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top')
        ax.axis('off')
        
        plt.suptitle('Perturbation Analysis Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_region_characteristics(self, result: AnalysisResult,
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot characteristics of regions.
        
        Args:
            result: Max-p analysis result
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        df = self.generate_region_summary(result)
        
        # 1. Region sizes (in km²)
        ax = axes[0, 0]
        df_sorted = df.sort_values('area_km2', ascending=True)
        bars = ax.barh(range(len(df_sorted)), df_sorted['area_km2'])
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([f"R{id}" for id in df_sorted['region_id']])
        ax.set_xlabel('Area (km²)')
        ax.set_title('Region Sizes')
        
        # Add threshold line
        threshold = result.statistics['min_area_threshold_km2']
        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.0f} km²')
        ax.legend()
        
        # Color bars based on threshold satisfaction
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            if row['area_km2'] < threshold:
                bars[i].set_color('red')
            else:
                bars[i].set_color('green')
        
        # 2. Compactness vs Homogeneity
        ax = axes[0, 1]
        scatter = ax.scatter(df['compactness'], df['homogeneity'], 
                           s=df['area_km2']/10, alpha=0.6, c=df['area_km2'],
                           cmap='viridis')
        ax.set_xlabel('Compactness')
        ax.set_ylabel('Homogeneity')
        ax.set_title('Region Quality Metrics')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Area (km²)')
        
        # Add region labels
        for _, row in df.iterrows():
            ax.annotate(f"R{row['region_id']}", 
                       (row['compactness'], row['homogeneity']),
                       fontsize=8)
        
        # 3. Within-region variance
        ax = axes[1, 0]
        ax.bar(range(len(df)), df['within_variance'])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f"R{id}" for id in df['region_id']], rotation=45)
        ax.set_ylabel('Within-region Variance')
        ax.set_title('Region Homogeneity (Lower is Better)')
        
        # 4. Area distribution
        ax = axes[1, 1]
        ax.hist(df['area_km2'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(threshold, color='red', linestyle='--', 
                  label=f'Threshold: {threshold:.0f} km²')
        ax.set_xlabel('Area (km²)')
        ax.set_ylabel('Number of Regions')
        ax.set_title('Distribution of Region Areas')
        ax.legend()
        
        # Add scale annotation
        scale = result.statistics['ecological_scale']
        ax.text(0.95, 0.95, f"Scale: {scale}", transform=ax.transAxes,
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _generate_interpretation(self, result: AnalysisResult) -> Dict[str, str]:
        """Generate automated interpretation of results."""
        quality = self.generate_quality_report(result)
        
        interp = {}
        
        # Interpret variance explained
        ve = quality['variance_explained']
        if ve > 0.8:
            ve_text = "Excellent - regions capture most biodiversity variation"
        elif ve > 0.6:
            ve_text = "Good - regions explain substantial variation"
        elif ve > 0.4:
            ve_text = "Moderate - consider adjusting threshold for better separation"
        else:
            ve_text = "Poor - regions may be too constrained"
        interp['variance_quality'] = ve_text
        
        # Interpret scale appropriateness
        scale = quality['ecological_scale']
        n_regions = quality['n_regions']
        if scale == 'ecoregion':
            if 10 <= n_regions <= 50:
                scale_text = "Appropriate number of regions for ecoregion scale"
            elif n_regions < 10:
                scale_text = "Few regions - consider reducing area threshold"
            else:
                scale_text = "Many regions - consider increasing area threshold"
        else:
            scale_text = f"Using {scale} scale with {n_regions} regions"
        interp['scale_assessment'] = scale_text
        
        # Interpret threshold satisfaction
        if quality['threshold_satisfied']:
            threshold_text = "All regions meet minimum area requirement"
        else:
            smallest = quality['smallest_region_km2']
            threshold = quality['min_area_threshold_km2']
            threshold_text = f"WARNING: Smallest region ({smallest:.0f} km²) below threshold ({threshold:.0f} km²)"
        interp['threshold_check'] = threshold_text
        
        # Interpret stability if available
        if 'stability_assessment' in quality:
            interp['robustness'] = quality['stability_assessment']
        
        return interp