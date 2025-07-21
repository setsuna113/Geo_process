# src/spatial_analysis/som/som_reporter.py
"""Generate reports for SOM analysis results."""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from src.spatial_analysis.base_analyzer import AnalysisResult

logger = logging.getLogger(__name__)

class SOMReporter:
    """Generate statistical reports and summaries for SOM results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path('reports/som')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_cluster_summary(self, result: AnalysisResult) -> pd.DataFrame:
        """
        Generate summary statistics for each cluster.
        
        Args:
            result: SOM analysis result
            
        Returns:
            DataFrame with cluster statistics
        """
        cluster_stats = result.statistics.get('cluster_statistics', {}) if result.statistics is not None else {}
        band_names = result.metadata.input_bands
        
        # Build summary data
        summary_data = []
        for cluster_id, stats in cluster_stats.items():
            row = {
                'cluster_id': cluster_id,
                'pixel_count': stats['count'],
                'percentage': stats['percentage']
            }
            
            # Add mean values for each band
            for i, band in enumerate(band_names):
                if i < len(stats['mean']):
                    row[f'{band}_mean'] = stats['mean'][i]
                    row[f'{band}_std'] = stats['std'][i]
                    row[f'{band}_min'] = stats['min'][i]
                    row[f'{band}_max'] = stats['max'][i]
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('pixel_count', ascending=False)
        
        return df
    
    def generate_full_report(self, result: AnalysisResult, 
                           save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive report including all statistics.
        
        Args:
            result: SOM analysis result
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
            'quality_metrics': {
                'quantization_error': result.statistics.get('quantization_error', 0) if result.statistics is not None else None,
                'topographic_error': result.statistics.get('topographic_error', 0) if result.statistics is not None else None,
                'empty_neurons': result.statistics.get('empty_neurons', 0) if result.statistics is not None else None,
                'cluster_balance': result.statistics.get('cluster_balance', 0) if result.statistics is not None else None
            },
            'cluster_summary': self.generate_cluster_summary(result).to_dict('records'),
            'interpretation': self._generate_interpretation(result)
        }
        
        if save_path:
            # Save as markdown
            self._save_markdown_report(report, save_path)
            
            # Save cluster summary as CSV
            csv_path = save_path.parent / f"{save_path.stem}_clusters.csv"
            self.generate_cluster_summary(result).to_csv(csv_path, index=False)
        
        return report
    
    def _generate_interpretation(self, result: AnalysisResult) -> Dict[str, str]:
        """Generate automated interpretation of results."""
        interp = {}
        
        # Interpret quantization error
        qe = result.statistics.get('quantization_error', 0) if result.statistics is not None else None
        if qe is not None and qe < 0.1:
            qe_text = "Excellent - the SOM represents the data very accurately"
        elif qe is not None and qe < 0.5:
            qe_text = "Good - the SOM captures the main patterns well"
        elif qe is not None and qe < 1.0:
            qe_text = "Moderate - consider increasing grid size for better representation"
        else:
            qe_text = "Poor - the SOM may be too small for the data complexity"
        interp['quantization_quality'] = qe_text
        
        # Interpret topographic error
        te = result.statistics.get('topographic_error', 0) if result.statistics is not None else None
        if te is not None and te < 0.05:
            te_text = "Excellent topology preservation"
        elif te is not None and te < 0.1:
            te_text = "Good topology preservation"
        elif te is not None and te < 0.2:
            te_text = "Moderate topology preservation"
        else:
            te_text = "Poor topology - consider adjusting SOM parameters"
        interp['topology_quality'] = te_text
        
        # Interpret cluster balance
        balance = result.statistics.get('cluster_balance', 0) if result.statistics is not None else None
        if balance is not None and balance < 0.5:
            balance_text = "Well-balanced cluster sizes"
        elif balance is not None and balance < 1.0:
            balance_text = "Moderately balanced clusters"
        else:
            balance_text = "Highly imbalanced clusters - some patterns dominate"
        interp['cluster_balance'] = balance_text
        
        # Interpret empty neurons
        empty = result.statistics.get('empty_neurons', 0) if result.statistics is not None else None
        total = np.prod(result.metadata.parameters['grid_size'])
        empty_pct = (empty / total) * 100 if empty is not None else 0
        
        if empty is not None and empty_pct < 10:
            empty_text = f"Good neuron utilization ({empty_pct:.1f}% empty)"
        elif empty_pct < 30:
            empty_text = f"Moderate neuron utilization ({empty_pct:.1f}% empty)"
        else:
            empty_text = f"Poor neuron utilization ({empty_pct:.1f}% empty) - consider smaller grid"
        interp['neuron_utilization'] = empty_text
        
        return interp
    
    def _save_markdown_report(self, report: Dict[str, Any], path: Path):
        """Save report as markdown file."""
        with open(path, 'w') as f:
            f.write("# SOM Analysis Report\n\n")
            
            # Metadata section
            f.write("## Analysis Information\n\n")
            for key, value in report['metadata'].items():
                if key != 'parameters':
                    formatted_key = key.replace("_", " ").title()
                    f.write(f"- **{formatted_key}**: {value}\n")
            
            # Parameters
            f.write("\n### Parameters\n\n")
            for key, value in report['metadata']['parameters'].items():
                f.write(f"- **{key}**: {value}\n")
            
            # Quality metrics
            f.write("\n## Quality Metrics\n\n")
            for key, value in report['quality_metrics'].items():
                formatted_key = key.replace("_", " ").title()
                f.write(f"- **{formatted_key}**: {value:.4f}\n")
            
            # Interpretation
            f.write("\n## Interpretation\n\n")
            for key, value in report['interpretation'].items():
                formatted_key = key.replace("_", " ").title()
                value_str = value if value is not None else "Not available"
                f.write(f"- **{formatted_key}**: {value_str}\n")
            # Top clusters
            f.write("\n## Top 10 Clusters by Size\n\n")
            df = pd.DataFrame(report['cluster_summary'][:10])
            markdown_str = df.to_markdown(index=False)
            if markdown_str is not None:
                f.write(markdown_str)
            else:
                f.write("No cluster data available")
            
        logger.info(f"Report saved to {path}")
    
    def compare_analyses(self, results: List[AnalysisResult]) -> pd.DataFrame:
        """
        Compare multiple SOM analyses.
        
        Args:
            results: List of SOM results to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for i, result in enumerate(results):
            row = {
                'analysis_id': i,
                'timestamp': result.metadata.timestamp,
                'grid_size': f"{result.metadata.parameters['grid_size'][0]}x{result.metadata.parameters['grid_size'][1]}",
                'iterations': result.metadata.parameters['iterations'],
                'n_clusters': result.statistics.get('n_clusters', 0) if result.statistics is not None else None,
                'quantization_error': result.statistics.get('quantization_error', 0) if result.statistics is not None else None,
                'topographic_error': result.statistics.get('topographic_error', 0) if result.statistics is not None else None,
                'empty_neurons': result.statistics.get('empty_neurons', 0) if result.statistics is not None else None,
                'cluster_balance': result.statistics.get('cluster_balance', 0) if result.statistics is not None else None,
                'processing_time': result.metadata.processing_time
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)