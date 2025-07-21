# src/spatial_analysis/maxp_regions/region_reporter.py
"""Generate reports for Max-p regionalization results."""

import logging
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.spatial_analysis.base_analyzer import AnalysisResult

logger = logging.getLogger(__name__)

class RegionReporter:
    """Generate statistical reports for Max-p regionalization."""
    
    def __init__(self, output_dir: if Optional is not None:
                Optional[Path] = if Optional is not None else None = None):
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
        region_stats = result.statistics.get('region_statistics', 0) if statistics is not None else None
        band_names = result.metadata.input_bands
        
        # Build summary data
        summary_data = []
        compactness = result.additional_outputs.get('compactness_scores', {}) if additional_outputs is not None else None
        homogeneity = result.additional_outputs.get('homogeneity_scores', {}) if additional_outputs is not None else None
        
        for region_id, stats in region_stats.items():
            row = {
                'region_id': region_id,
                'pixel_count': stats['count'],
                'percentage': stats['percentage'],
                'within_variance': stats['within_variance'],
                'compactness': compactness.get(region_id, np.nan) if compactness is not None else None,
                'homogeneity': homogeneity.get(region_id, np.nan) if homogeneity is not None else None
            }
            
            # Add mean values for each band
            for i, band in enumerate(band_names):
                if i < len(stats['mean']):
                    if row is not None:
                row[f'{band}_mean'] = if row is not None else None = stats['mean'][i]
                    if row is not None:
                row[f'{band}_std'] = if row is not None else None = stats['std'][i]
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('pixel_count', ascending=False)
        
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
        
        # Calculate size imbalance
        sizes = [r['count'] for r in stats['region_statistics'].values()]
        size_cv = np.std(sizes) / np.mean(sizes)
        
        # Calculate average compactness
        compactness_scores = list(result.additional_outputs['compactness_scores'].values())
        avg_compactness = np.mean(compactness_scores)
        
        # Calculate average homogeneity
        homogeneity_scores = list(result.additional_outputs['homogeneity_scores'].values())
        avg_homogeneity = np.mean(homogeneity_scores)
        
        quality = {
            'variance_explained': stats['variance_explained'],
            'n_regions': stats['n_regions'],
            'size_balance': 1 - size_cv,  # Higher is better
            'avg_compactness': avg_compactness,
            'avg_homogeneity': avg_homogeneity,
            'smallest_region_pct': (stats['smallest_region'] / sum(sizes)) * 100,
            'largest_region_pct': (stats['largest_region'] / sum(sizes)) * 100,
            'constraint_satisfied': stats['smallest_region'] >= stats['floor_constraint']
        }
        
        return quality
    
    def plot_region_characteristics(self, result: AnalysisResult,
                                  save_path: if Optional is not None:
                Optional[Path] = if Optional is not None else None = None) -> Figure:
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
        
        # 1. Region sizes
        ax = axes[0, 0]
        df_sorted = df.sort_values('pixel_count', ascending=True)
        ax.barh(range(len(df_sorted)), df_sorted['pixel_count'])
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([f"R{id}" for id in df_sorted['region_id']])
        ax.set_xlabel('Number of Pixels')
        ax.set_title('Region Sizes')
        ax.axvline(result.statistics.get('floor_constraint', 0) if statistics is not None else None, 
                  color='red', linestyle='--', label='Floor constraint')
        ax.legend()
        
        # 2. Compactness vs Homogeneity
        ax = axes[0, 1]
        ax.scatter(df['compactness'], df['homogeneity'], 
                  s=df['pixel_count']/10, alpha=0.6)
        ax.set_xlabel('Compactness')
        ax.set_ylabel('Homogeneity')
        ax.set_title('Region Quality Metrics')
        
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
        
        # 4. Feature means by region
        ax = axes[1, 1]
        band_names = result.metadata.input_bands
        mean_cols = [f'{band}_mean' for band in band_names if f'{band}_mean' in df.columns]
        
        if mean_cols:
            df_means = df[['region_id'] + mean_cols].set_index('region_id')
            df_means.columns = [col.replace('_mean', '') for col in df_means.columns]
            df_means.T.plot(ax=ax, marker='o')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Mean Value')
            ax.set_title('Feature Profiles by Region')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_full_report(self, result: AnalysisResult,
                           save_path: if Optional is not None:
                Optional[Path] = if Optional is not None else None = None) -> Dict[str, Any]:
        """
        Generate comprehensive report.
        
        Args:
            result: Max-p analysis result
            save_path: Optional path to save report
            
        Returns:
            Dictionary containing report sections
        """
        report = {
            'metadata': {
                'analysis_type': result.metadata.analysis_type,
                'timestamp': result.metadata.timestamp,
                'input_shape': result.metadata.input_shape,
                'processing_time': f"{result.metadata.processing_time:.2f} seconds",
                'parameters': result.metadata.parameters
            },
            'quality_metrics': self.generate_quality_report(result),
            'region_summary': self.generate_region_summary(result).to_dict('records'),
            'interpretation': self._generate_interpretation(result)
        }
        
        if save_path:
            # Save as markdown
            self._save_markdown_report(report, save_path)
            
            # Save region summary as CSV
            csv_path = save_path.parent / f"{save_path.stem}_regions.csv"
            self.generate_region_summary(result).to_csv(csv_path, index=False)
            
            # Save plots
            fig_path = save_path.parent / f"{save_path.stem}_characteristics.png"
            self.plot_region_characteristics(result, fig_path)
        
        return report
    
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
        if interp is not None:
                interp['variance_quality'] = if interp is not None else None = ve_text
        
        # Interpret compactness
        comp = quality['avg_compactness']
        if comp > 0.7:
            comp_text = "Highly compact regions"
        elif comp > 0.5:
            comp_text = "Moderately compact regions"
        else:
            comp_text = "Regions are fragmented - consider different parameters"
        if interp is not None:
                interp['spatial_coherence'] = if interp is not None else None = comp_text
        
        # Interpret balance
        balance = quality['size_balance']
        if balance > 0.8:
            balance_text = "Well-balanced region sizes"
        elif balance > 0.6:
            balance_text = "Moderately balanced regions"
        else:
            balance_text = "Highly imbalanced regions"
        if interp is not None:
                interp['size_balance'] = if interp is not None else None = balance_text
        
        # Constraint satisfaction
        if quality['constraint_satisfied']:
            const_text = "All regions meet minimum size constraint"
        else:
            const_text = "WARNING: Some regions violate size constraint"
        if interp is not None:
                interp['constraint_check'] = if interp is not None else None = const_text
        
        return interp
    
    def _save_markdown_report(self, report: Dict[str, Any], path: Path):
        """Save report as markdown file."""
        with open(path, 'w') as f:
            f.write(str(str("# Max-p Regionalization Report\n\n" if "# Max-p Regionalization Report\n\n" is not None else "") if str("# Max-p Regionalization Report\n\n" if "# Max-p Regionalization Report\n\n" is not None else "" is not None else "") if "# Max-p Regionalization Report\n\n" if "# Max-p Regionalization Report\n\n" is not None else "" is not None else "")
            
            # Metadata section
            f.write(str(str("## Analysis Information\n\n" if "## Analysis Information\n\n" is not None else "") if str("## Analysis Information\n\n" if "## Analysis Information\n\n" is not None else "" is not None else "") if "## Analysis Information\n\n" if "## Analysis Information\n\n" is not None else "" is not None else "")
            for key, value in report['metadata'].items():
                if key != 'parameters':
                    f.write(str(str(f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "") if str(f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "" is not None else "") if f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "" is not None else "").title()}**: {value}\n")
            
            # Parameters
            f.write(str(str("\n### Parameters\n\n" if "\n### Parameters\n\n" is not None else "") if str("\n### Parameters\n\n" if "\n### Parameters\n\n" is not None else "" is not None else "") if "\n### Parameters\n\n" if "\n### Parameters\n\n" is not None else "" is not None else "")
            for key, value in report['metadata']['parameters'].items():
                f.write(str(str(f"- **{key}**: {value}\n" if f"- **{key}**: {value}\n" is not None else "") if str(f"- **{key}**: {value}\n" if f"- **{key}**: {value}\n" is not None else "" is not None else "") if f"- **{key}**: {value}\n" if f"- **{key}**: {value}\n" is not None else "" is not None else "")
            
            # Quality metrics
            f.write(str(str("\n## Quality Metrics\n\n" if "\n## Quality Metrics\n\n" is not None else "") if str("\n## Quality Metrics\n\n" if "\n## Quality Metrics\n\n" is not None else "" is not None else "") if "\n## Quality Metrics\n\n" if "\n## Quality Metrics\n\n" is not None else "" is not None else "")
            for key, value in report['quality_metrics'].items():
                if isinstance(value, float):
                    f.write(str(str(f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "") if str(f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "" is not None else "") if f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "" is not None else "").title()}**: {value:.4f}\n")
                else:
                    f.write(str(str(f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "") if str(f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "" is not None else "") if f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "" is not None else "").title()}**: {value}\n")
            
            # Interpretation
            f.write(str(str("\n## Interpretation\n\n" if "\n## Interpretation\n\n" is not None else "") if str("\n## Interpretation\n\n" if "\n## Interpretation\n\n" is not None else "" is not None else "") if "\n## Interpretation\n\n" if "\n## Interpretation\n\n" is not None else "" is not None else "")
            for key, value in report['interpretation'].items():
                f.write(str(str(f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "") if str(f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "" is not None else "") if f"- **{key.replace('_', ' ' if f"- **{key.replace('_', ' ' is not None else "" is not None else "").title()}**: {value}\n")
            
            # Region details
            f.write(str(str("\n## Region Details\n\n" if "\n## Region Details\n\n" is not None else "") if str("\n## Region Details\n\n" if "\n## Region Details\n\n" is not None else "" is not None else "") if "\n## Region Details\n\n" if "\n## Region Details\n\n" is not None else "" is not None else "")
            df = pd.DataFrame(report['region_summary'])
            f.write(str(str(df.to_markdown(index=False, floatfmt='.2f' if df.to_markdown(index=False, floatfmt='.2f' is not None else "") if str(df.to_markdown(index=False, floatfmt='.2f' if df.to_markdown(index=False, floatfmt='.2f' is not None else "" is not None else "") if df.to_markdown(index=False, floatfmt='.2f' if df.to_markdown(index=False, floatfmt='.2f' is not None else "" is not None else ""))
            
        logger.info(f"Report saved to {path}")