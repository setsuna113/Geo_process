"""
Clean SOM Analysis Reporter

Generates clear, actionable reports for biodiversity SOM analysis results.
Replaces deprecated complex reporting with simple, understandable output.
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from src.abstractions.interfaces.analyzer import AnalysisResult
from .simple_vlrsom import VLRSOMResult
from .spatial_validation import SpatialDataSplit

logger = logging.getLogger(__name__)


class SOMAnalysisReporter:
    """
    Clean reporter for SOM analysis results.
    
    Generates:
    1. Summary report (JSON)
    2. Detailed analysis (CSV)
    3. Training history (CSV)
    4. Cluster analysis (CSV)
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize SOM reporter.
        
        Args:
            output_dir: Directory for saving reports (default: ./som_reports)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("som_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SOM reporter, output dir: {self.output_dir}")
    
    def generate_report(self, 
                       result: AnalysisResult, 
                       report_name: str = None) -> Dict[str, Path]:
        """
        Generate complete SOM analysis report.
        
        Args:
            result: SOM analysis result
            report_name: Base name for report files
            
        Returns:
            Dictionary mapping report type to file path
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"som_analysis_{timestamp}"
        
        logger.info(f"Generating SOM report: {report_name}")
        
        report_files = {}
        
        # 1. Summary report (JSON)
        summary_file = self._generate_summary_report(result, report_name)
        report_files['summary'] = summary_file
        
        # 2. Cluster analysis (CSV)
        cluster_file = self._generate_cluster_analysis(result, report_name)
        report_files['clusters'] = cluster_file
        
        # 3. Training history (CSV) - if available
        if 'vlrsom_result' in result.additional_outputs:
            history_file = self._generate_training_history(result, report_name)
            report_files['training_history'] = history_file
        
        # 4. Spatial validation report (CSV) - if available
        if result.additional_outputs.get('data_split') is not None:
            spatial_file = self._generate_spatial_report(result, report_name)
            report_files['spatial_validation'] = spatial_file
        
        logger.info(f"Generated {len(report_files)} report files")
        
        return report_files
    
    def _generate_summary_report(self, result: AnalysisResult, report_name: str) -> Path:
        """Generate JSON summary report."""
        summary_file = self.output_dir / f"{report_name}_summary.json"
        
        # Extract key information
        vlrsom_result = result.additional_outputs.get('vlrsom_result')
        data_split = result.additional_outputs.get('data_split')
        
        summary = {
            'report_metadata': {
                'report_name': report_name,
                'generated_at': datetime.now().isoformat(),
                'analysis_type': result.metadata.analysis_type,
                'processing_time': result.metadata.processing_time
            },
            'dataset_info': {
                'input_shape': result.metadata.input_shape,
                'total_samples': result.metadata.input_shape[0] if result.metadata.input_shape else 0,
                'n_features': result.metadata.input_shape[1] if len(result.metadata.input_shape) > 1 else 0,
                'data_source': result.metadata.data_source
            },
            'som_configuration': {
                'grid_size': result.metadata.parameters.get('grid_size'),
                'max_iterations': result.metadata.parameters.get('max_iterations'),
                'qe_threshold': result.metadata.parameters.get('qe_threshold'),
                'te_threshold': result.metadata.parameters.get('te_threshold')
            },
            'clustering_results': {
                'n_clusters': result.statistics.get('n_clusters'),
                'quantization_error': result.statistics.get('som_metrics', {}).get('quantization_error'),
                'topographic_error': result.statistics.get('som_metrics', {}).get('topographic_error')
            },
            'training_results': self._extract_training_summary(vlrsom_result),
            'spatial_validation': self._extract_spatial_summary(data_split, result.statistics),
            'cluster_distribution': self._calculate_cluster_distribution(result.statistics)
        }
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved summary report: {summary_file}")
        return summary_file
    
    def _generate_cluster_analysis(self, result: AnalysisResult, report_name: str) -> Path:
        """Generate detailed cluster analysis CSV."""
        cluster_file = self.output_dir / f"{report_name}_clusters.csv"
        
        cluster_stats = result.statistics.get('cluster_statistics', {})
        feature_names = result.additional_outputs.get('feature_names', [])
        
        # Convert cluster statistics to DataFrame
        cluster_rows = []
        for cluster_id, stats in cluster_stats.items():
            row = {
                'cluster_id': cluster_id,
                'sample_count': stats['count'],
                'percentage': stats['percentage']
            }
            
            # Add feature means and stds
            means = stats.get('mean', [])
            stds = stats.get('std', [])
            
            for i, (mean_val, std_val) in enumerate(zip(means, stds)):
                feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                row[f"{feature_name}_mean"] = mean_val
                row[f"{feature_name}_std"] = std_val
            
            cluster_rows.append(row)
        
        # Create DataFrame and save
        if cluster_rows:
            cluster_df = pd.DataFrame(cluster_rows)
            cluster_df = cluster_df.sort_values('cluster_id')
            cluster_df.to_csv(cluster_file, index=False)
            
            logger.info(f"Saved cluster analysis: {cluster_file}")
        else:
            logger.warning("No cluster statistics available")
        
        return cluster_file
    
    def _generate_training_history(self, result: AnalysisResult, report_name: str) -> Path:
        """Generate training history CSV."""
        history_file = self.output_dir / f"{report_name}_training_history.csv"
        
        vlrsom_result = result.additional_outputs.get('vlrsom_result')
        if not isinstance(vlrsom_result, VLRSOMResult):
            logger.warning("No VLRSOM training history available")
            return history_file
        
        # Create training history DataFrame
        max_len = max(
            len(vlrsom_result.qe_history),
            len(vlrsom_result.te_history),
            len(vlrsom_result.learning_rate_history)
        )
        
        history_data = {
            'iteration': list(range(0, max_len * 10, 10)),  # Assuming 10-iteration intervals
            'quantization_error': self._pad_list(vlrsom_result.qe_history, max_len),
            'topographic_error': self._pad_list(vlrsom_result.te_history, max_len),
            'learning_rate': self._pad_list(vlrsom_result.learning_rate_history, max_len)
        }
        
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(history_file, index=False)
        
        logger.info(f"Saved training history: {history_file}")
        return history_file
    
    def _generate_spatial_report(self, result: AnalysisResult, report_name: str) -> Path:
        """Generate spatial validation report CSV."""
        spatial_file = self.output_dir / f"{report_name}_spatial_validation.csv"
        
        data_split = result.additional_outputs.get('data_split')
        spatial_stats = result.statistics.get('spatial_validation', {})
        
        if not isinstance(data_split, SpatialDataSplit):
            logger.warning("No spatial validation data available")
            return spatial_file
        
        # Create spatial validation report
        spatial_data = []
        
        # Split metadata
        split_metadata = spatial_stats.get('split_metadata', {})
        for split_name in ['train', 'validation', 'test']:
            stats_key = f"{split_name}_stats"
            if stats_key in split_metadata:
                stats = split_metadata[stats_key]
                row = {
                    'split': split_name,
                    'sample_count': stats.get('count', 0),
                    'longitude_min': stats.get('lon_range', [None, None])[0],
                    'longitude_max': stats.get('lon_range', [None, None])[1],
                    'latitude_min': stats.get('lat_range', [None, None])[0],
                    'latitude_max': stats.get('lat_range', [None, None])[1],
                    'longitude_center': stats.get('lon_center'),
                    'latitude_center': stats.get('lat_center'),
                    'quantization_error': spatial_stats.get(f"{split_name}_qe")
                }
                spatial_data.append(row)
        
        # Add overall statistics
        overall_row = {
            'split': 'overall',
            'sample_count': split_metadata.get('total_samples'),
            'split_strategy': spatial_stats.get('split_strategy'),
            'longitude_min': None,
            'longitude_max': None,
            'latitude_min': None,
            'latitude_max': None,
            'longitude_center': None,
            'latitude_center': None,
            'quantization_error': result.statistics.get('som_metrics', {}).get('quantization_error')
        }
        spatial_data.append(overall_row)
        
        # Save spatial report
        if spatial_data:
            spatial_df = pd.DataFrame(spatial_data)
            spatial_df.to_csv(spatial_file, index=False)
            
            logger.info(f"Saved spatial validation report: {spatial_file}")
        
        return spatial_file
    
    def _extract_training_summary(self, vlrsom_result: Optional[VLRSOMResult]) -> Dict[str, Any]:
        """Extract training summary from VLRSOM result."""
        if not isinstance(vlrsom_result, VLRSOMResult):
            return {}
        
        return {
            'converged': vlrsom_result.converged,
            'total_iterations': vlrsom_result.total_iterations,
            'training_time_seconds': vlrsom_result.training_time,
            'final_quantization_error': vlrsom_result.final_quantization_error,
            'final_topographic_error': vlrsom_result.final_topographic_error,
            'learning_rate_final': vlrsom_result.learning_rate_history[-1] if vlrsom_result.learning_rate_history else None,
            'learning_rate_initial': vlrsom_result.learning_rate_history[0] if vlrsom_result.learning_rate_history else None
        }
    
    def _extract_spatial_summary(self, data_split: Optional[SpatialDataSplit], statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract spatial validation summary."""
        if not isinstance(data_split, SpatialDataSplit):
            return {'enabled': False}
        
        spatial_stats = statistics.get('spatial_validation', {})
        
        return {
            'enabled': True,
            'split_strategy': data_split.split_strategy,
            'train_samples': len(data_split.train_data),
            'validation_samples': len(data_split.validation_data),
            'test_samples': len(data_split.test_data),
            'train_qe': spatial_stats.get('train_qe'),
            'validation_qe': spatial_stats.get('validation_qe'),
            'test_qe': spatial_stats.get('test_qe')
        }
    
    def _calculate_cluster_distribution(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cluster distribution statistics."""
        cluster_stats = statistics.get('cluster_statistics', {})
        
        if not cluster_stats:
            return {}
        
        # Extract cluster sizes
        cluster_sizes = [stats['count'] for stats in cluster_stats.values()]
        
        return {
            'total_clusters': len(cluster_sizes),
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'std_cluster_size': np.std(cluster_sizes) if cluster_sizes else 0,
            'cluster_balance_coefficient': np.std(cluster_sizes) / np.mean(cluster_sizes) if cluster_sizes and np.mean(cluster_sizes) > 0 else 0
        }
    
    def _pad_list(self, lst: List, target_length: int, pad_value=None):
        """Pad list to target length."""
        if len(lst) >= target_length:
            return lst[:target_length]
        return lst + [pad_value] * (target_length - len(lst))
    
    def print_summary(self, result: AnalysisResult) -> None:
        """Print a concise summary to console."""
        print("\n" + "="*60)
        print("BIODIVERSITY SOM ANALYSIS SUMMARY")
        print("="*60)
        
        # Dataset info
        shape = result.metadata.input_shape
        print(f"Dataset: {shape[0]:,} samples, {shape[1]} features" if len(shape) > 1 else f"Dataset: {shape}")
        print(f"Processing time: {result.metadata.processing_time:.2f} seconds")
        
        # SOM results
        n_clusters = result.statistics.get('n_clusters', 0)
        grid_size = result.metadata.parameters.get('grid_size', [0, 0])
        print(f"SOM Grid: {grid_size[0]}x{grid_size[1]} → {n_clusters} active clusters")
        
        # Quality metrics
        som_metrics = result.statistics.get('som_metrics', {})
        qe = som_metrics.get('quantization_error', 0)
        te = som_metrics.get('topographic_error', 0)
        print(f"Quantization Error: {qe:.6f}")
        print(f"Topographic Error: {te:.4f}")
        
        # Training results
        training_stats = result.statistics.get('training_statistics', {})
        converged = training_stats.get('converged', False)
        iterations = training_stats.get('total_iterations', 0)
        print(f"Training: {'Converged' if converged else 'Not converged'} after {iterations} iterations")
        
        # Spatial validation
        spatial_stats = result.statistics.get('spatial_validation', {})
        if spatial_stats:
            strategy = spatial_stats.get('split_strategy', 'unknown')
            print(f"Spatial validation: {strategy} split")
            val_qe = spatial_stats.get('validation_qe', 0)
            test_qe = spatial_stats.get('test_qe', 0)
            print(f"Validation QE: {val_qe:.6f}, Test QE: {test_qe:.6f}")
        
        print("="*60)


def create_som_reporter(output_dir: Optional[Path] = None) -> SOMAnalysisReporter:
    """Factory function for creating SOM reporter."""
    return SOMAnalysisReporter(output_dir)