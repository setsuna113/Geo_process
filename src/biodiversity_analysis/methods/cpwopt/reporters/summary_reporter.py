"""Summary reporter for CP-WOPT analysis results.

This module creates a comprehensive summary report combining all aspects
of the tensor factorization analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime

from src.core_abstractions import CPFactors, TensorMetrics

logger = logging.getLogger(__name__)


class SummaryReporter:
    """Creates comprehensive summary reports for CP-WOPT analysis."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """Initialize summary reporter.
        
        Args:
            output_dir: Directory to save outputs
            config: Reporting configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.plot_format = config.get('plot_format', 'png')
        self.plot_dpi = config.get('plot_dpi', 150)
        self.generate_html = config.get('generate_html', True)
        
    def generate_summary_report(self, 
                              decomposition_result: Dict[str, Any],
                              experiment_metadata: Dict[str, Any],
                              all_reporter_results: Dict[str, Any]) -> Path:
        """Generate comprehensive summary report.
        
        Args:
            decomposition_result: Complete decomposition result
            experiment_metadata: Experiment metadata
            all_reporter_results: Results from all other reporters
            
        Returns:
            Path to summary report
        """
        # Create summary dashboard plot
        dashboard_path = self._create_summary_dashboard(
            decomposition_result.factors,
            decomposition_result.metrics,
            experiment_metadata
        )
        
        # Generate text summary
        text_summary = self._generate_text_summary(
            decomposition_result,
            experiment_metadata,
            all_reporter_results
        )
        
        # Save text summary
        summary_path = self.output_dir / 'summary_report.txt'
        with open(summary_path, 'w') as f:
            f.write(text_summary)
        
        # Generate JSON summary
        json_summary = self._generate_json_summary(
            decomposition_result,
            experiment_metadata,
            all_reporter_results
        )
        
        json_path = self.output_dir / 'summary_report.json'
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        # Generate HTML report if requested
        if self.generate_html:
            html_path = self._generate_html_report(
                text_summary,
                json_summary,
                dashboard_path,
                all_reporter_results
            )
            logger.info(f"Generated HTML report: {html_path}")
        
        logger.info(f"Generated summary report: {summary_path}")
        return summary_path
    
    def _create_summary_dashboard(self, factors: CPFactors,
                                 metrics: TensorMetrics,
                                 metadata: Dict[str, Any]) -> Path:
        """Create summary dashboard visualization.
        
        Args:
            factors: CP factors
            metrics: Tensor metrics
            metadata: Experiment metadata
            
        Returns:
            Path to dashboard plot
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('CP-WOPT Analysis Summary Dashboard', fontsize=20)
        
        # 1. Convergence plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if len(metrics.convergence_history) > 0:
            ax1.plot(metrics.convergence_history, 'b-', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Reconstruction Error')
            ax1.set_title('Convergence History')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No convergence history',
                    ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Component weights (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if metrics.component_weights is not None:
            weights = metrics.component_weights[:20]  # Show top 20
            ax2.bar(range(len(weights)), weights)
            ax2.set_xlabel('Component')
            ax2.set_ylabel('Weight')
            ax2.set_title('Component Weights')
        else:
            ax2.text(0.5, 0.5, 'No component weights',
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Key metrics (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        metrics_text = f"""Key Metrics:
        
Reconstruction Error: {metrics.reconstruction_error:.4f}
Relative Error: {metrics.relative_error:.4f}
Fit Score: {metrics.fit:.4f}
Explained Variance: {metrics.explained_variance:.4f}

Model Parameters:
Rank: {factors.rank}
Tensor Modes: {factors.n_modes}
Iterations: {len(metrics.convergence_history)}
        """
        ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        # 4. Factor correlation heatmap (middle row, spanning 2 cols)
        ax4 = fig.add_subplot(gs[1, :2])
        # Average correlation across modes
        avg_corr = np.zeros((factors.rank, factors.rank))
        for factor in factors.factors:
            factor_norm = factor / (np.linalg.norm(factor, axis=0) + 1e-10)
            avg_corr += np.abs(factor_norm.T @ factor_norm) / factors.n_modes
        
        im = ax4.imshow(avg_corr, cmap='RdBu_r', vmin=0, vmax=1)
        ax4.set_title('Average Component Correlations')
        ax4.set_xlabel('Component')
        ax4.set_ylabel('Component')
        plt.colorbar(im, ax=ax4)
        
        # 5. Data characteristics (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Extract tensor info from metadata
        tensor_shape = metadata.get('tensor_shape', 'Unknown')
        sparsity = metadata.get('sparsity', 0.0)
        n_samples = metadata.get('n_samples', 0)
        
        data_text = f"""Data Characteristics:

Tensor Shape: {tensor_shape}
Sparsity: {sparsity:.1%}
Total Samples: {n_samples:,}
Features: {metadata.get('n_features', 'Unknown')}

Processing:
Runtime: {metadata.get('runtime_seconds', 0):.1f}s
Memory Peak: {metadata.get('peak_memory_mb', 0):.1f}MB
        """
        ax5.text(0.1, 0.9, data_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        # 6. Cumulative variance explained (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        if metrics.component_weights is not None:
            sorted_weights = np.sort(metrics.component_weights)[::-1]
            cumvar = np.cumsum(sorted_weights) / np.sum(sorted_weights)
            ax6.plot(range(1, len(cumvar) + 1), cumvar, 'g-', linewidth=2)
            ax6.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
            ax6.set_xlabel('Number of Components')
            ax6.set_ylabel('Cumulative Variance')
            ax6.set_title('Variance Explained')
            ax6.grid(True, alpha=0.3)
            
            # Find 90% threshold
            n_90 = np.argmax(cumvar >= 0.9) + 1
            ax6.axvline(x=n_90, color='r', linestyle='--', alpha=0.5)
            ax6.text(n_90, 0.5, f'{n_90} components\nfor 90%', 
                    ha='center', va='center')
        
        # 7. Mode-wise factor norms (bottom middle)
        ax7 = fig.add_subplot(gs[2, 1])
        mode_norms = []
        for i, (name, factor) in enumerate(zip(factors.mode_names, factors.factors)):
            norms = np.linalg.norm(factor, axis=0)
            mode_norms.append(norms)
        
        # Box plot of norms by mode
        ax7.boxplot(mode_norms, labels=factors.mode_names)
        ax7.set_ylabel('Factor Norm')
        ax7.set_title('Factor Norms by Mode')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Experiment info (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        exp_text = f"""Experiment Info:

Name: {metadata.get('experiment_name', 'Unknown')}
Method: CP-WOPT
Date: {metadata.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}

Configuration:
Init Method: {metadata.get('init_method', 'Unknown')}
Optimizer: {metadata.get('optimizer', 'ALS')}
Convergence: {metadata.get('convergence_threshold', 1e-6)}
        """
        ax8.text(0.1, 0.9, exp_text, transform=ax8.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.output_dir / f'summary_dashboard.{self.plot_format}'
        plt.savefig(dashboard_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created summary dashboard: {dashboard_path}")
        return dashboard_path
    
    def _generate_text_summary(self, decomposition_result: Dict[str, Any],
                             metadata: Dict[str, Any],
                             all_reporter_results: Dict[str, Any]) -> str:
        """Generate comprehensive text summary.
        
        Args:
            decomposition_result: Decomposition result
            metadata: Experiment metadata
            all_reporter_results: Results from all reporters
            
        Returns:
            Text summary
        """
        summary_lines = []
        
        # Header
        summary_lines.append("=" * 80)
        summary_lines.append("CP-WOPT TENSOR FACTORIZATION ANALYSIS SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Experiment: {metadata.get('experiment_name', 'Unknown')}")
        summary_lines.append("")
        
        # Model Overview
        summary_lines.append("MODEL OVERVIEW")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Method: CP-WOPT (Weighted CP Decomposition)")
        factors = decomposition_result.get('factors', {})
        summary_lines.append(f"Tensor Rank: {factors.get('rank', 'Unknown')}")
        summary_lines.append(f"Tensor Shape: {metadata.get('tensor_shape', 'Unknown')}")
        summary_lines.append(f"Number of Modes: {factors.get('n_modes', 'Unknown')}")
        mode_names = factors.get('mode_names', [])
        summary_lines.append(f"Mode Names: {', '.join(mode_names) if mode_names else 'Unknown'}")
        summary_lines.append("")
        
        # Data Characteristics
        summary_lines.append("DATA CHARACTERISTICS")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Sparsity: {metadata.get('sparsity', 0.0):.1%}")
        summary_lines.append(f"Total Entries: {metadata.get('total_entries', 0):,}")
        summary_lines.append(f"Non-zero Entries: {metadata.get('nonzero_entries', 0):,}")
        summary_lines.append(f"Features: {metadata.get('n_features', 'Unknown')}")
        summary_lines.append("")
        
        # Model Performance
        summary_lines.append("MODEL PERFORMANCE")
        summary_lines.append("-" * 40)
        metrics = decomposition_result.get('metrics', {})
        summary_lines.append(f"Reconstruction Error: {metrics.get('reconstruction_error', 0.0):.6f}")
        summary_lines.append(f"Relative Error: {metrics.get('relative_error', 0.0):.6f}")
        summary_lines.append(f"Fit Score: {metrics.get('fit', 0.0):.4f}")
        summary_lines.append(f"Explained Variance: {metrics.get('explained_variance', 0.0):.4f}")
        convergence_history = metrics.get('convergence_history', [])
        summary_lines.append(f"Iterations: {len(convergence_history)}")
        summary_lines.append(f"Converged: {metadata.get('converged', 'Unknown')}")
        summary_lines.append(f"Runtime: {metadata.get('runtime_seconds', 0):.1f} seconds")
        summary_lines.append("")
        
        # Component Analysis
        summary_lines.append("COMPONENT ANALYSIS")
        summary_lines.append("-" * 40)
        
        component_weights = metrics.get('component_weights')
        if component_weights is not None:
            if isinstance(component_weights, np.ndarray):
                weights_array = component_weights
            else:
                weights_array = np.array(component_weights)
            
            # Top components
            sorted_idx = np.argsort(weights_array)[::-1][:5]
            summary_lines.append("Top 5 Components by Weight:")
            for i, idx in enumerate(sorted_idx):
                weight = weights_array[idx]
                summary_lines.append(f"  {i+1}. Component {idx+1}: {weight:.4f}")
            
            # Variance concentration
            sorted_weights = np.sort(weights_array)[::-1]
            cumvar = np.cumsum(sorted_weights) / np.sum(sorted_weights)
            n_90 = np.argmax(cumvar >= 0.9) + 1
            summary_lines.append(f"\nComponents needed for 90% variance: {n_90}")
        
        summary_lines.append("")
        
        # Key Findings from Component Reporter
        if 'component_reporter' in all_reporter_results:
            comp_results = all_reporter_results['component_reporter']
            if 'interpretations' in comp_results:
                summary_lines.append("TOP COMPONENT INTERPRETATIONS")
                summary_lines.append("-" * 40)
                
                # Show top 3 components
                for i in range(min(3, len(comp_results['interpretations']))):
                    comp_key = f'component_{i}'
                    if comp_key in comp_results['interpretations']:
                        comp_info = comp_results['interpretations'][comp_key]
                        summary_lines.append(f"\nComponent {i+1}:")
                        
                        # Top features
                        if 'top_positive_features' in comp_info and comp_info['top_positive_features']:
                            top_3 = comp_info['top_positive_features'][:3]
                            features = [f['name'] for f in top_3]
                            summary_lines.append(f"  High: {', '.join(features)}")
                        
                        if 'top_negative_features' in comp_info and comp_info['top_negative_features']:
                            top_3 = comp_info['top_negative_features'][:3]
                            features = [f['name'] for f in top_3]
                            summary_lines.append(f"  Low: {', '.join(features)}")
                        
                        # Spatial pattern
                        if 'spatial_pattern' in comp_info:
                            pattern = comp_info['spatial_pattern'].get('overall', 'Unknown')
                            summary_lines.append(f"  Spatial: {pattern}")
        
        summary_lines.append("")
        
        # Processing Details
        summary_lines.append("PROCESSING DETAILS")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Initialization: {metadata.get('init_method', 'Unknown')}")
        summary_lines.append(f"Optimizer: {metadata.get('optimizer', 'ALS')}")
        summary_lines.append(f"Mini-batch: {metadata.get('minibatch_enabled', False)}")
        if metadata.get('minibatch_enabled'):
            summary_lines.append(f"  Batch Size: {metadata.get('batch_size', 'Unknown')}")
        summary_lines.append(f"Sparse Mode: {metadata.get('sparse_mode', False)}")
        summary_lines.append(f"Memory Peak: {metadata.get('peak_memory_mb', 0):.1f} MB")
        summary_lines.append("")
        
        # Output Files
        summary_lines.append("OUTPUT FILES")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Results Directory: {self.output_dir}")
        summary_lines.append("Generated Files:")
        summary_lines.append("  - summary_report.txt (this file)")
        summary_lines.append("  - summary_report.json")
        summary_lines.append("  - summary_dashboard.png")
        summary_lines.append("  - factors/")
        summary_lines.append("  - plots/")
        summary_lines.append("  - component_interpretations.json")
        
        return "\n".join(summary_lines)
    
    def _generate_json_summary(self, decomposition_result: Dict[str, Any],
                             metadata: Dict[str, Any],
                             all_reporter_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON summary for programmatic access.
        
        Args:
            decomposition_result: Decomposition result
            metadata: Experiment metadata
            all_reporter_results: Results from all reporters
            
        Returns:
            JSON-serializable dictionary
        """
        metrics = decomposition_result.metrics
        
        summary = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'experiment_name': metadata.get('experiment_name', 'Unknown'),
                'method': 'CP-WOPT',
                'version': '1.0.0'
            },
            'model': {
                'rank': decomposition_result.factors.rank,
                'n_modes': decomposition_result.factors.n_modes,
                'mode_names': decomposition_result.factors.mode_names,
                'tensor_shape': metadata.get('tensor_shape'),
                'initialization': metadata.get('init_method'),
                'optimizer': metadata.get('optimizer')
            },
            'data': {
                'sparsity': metadata.get('sparsity'),
                'total_entries': metadata.get('total_entries'),
                'nonzero_entries': metadata.get('nonzero_entries'),
                'n_features': metadata.get('n_features')
            },
            'performance': {
                'reconstruction_error': float(metrics.reconstruction_error),
                'relative_error': float(metrics.relative_error),
                'fit_score': float(metrics.fit),
                'explained_variance': float(metrics.explained_variance) if metrics.explained_variance else None,
                'iterations': len(metrics.convergence_history),
                'converged': metadata.get('converged'),
                'runtime_seconds': metadata.get('runtime_seconds')
            },
            'components': {
                'weights': metrics.component_weights.tolist() if metrics.component_weights is not None else None,
                'n_components_90_variance': self._find_n_components_for_variance(metrics.component_weights, 0.9)
            },
            'processing': {
                'minibatch_enabled': metadata.get('minibatch_enabled'),
                'batch_size': metadata.get('batch_size'),
                'sparse_mode': metadata.get('sparse_mode'),
                'peak_memory_mb': metadata.get('peak_memory_mb'),
                'n_workers': metadata.get('n_workers')
            },
            'outputs': {
                'directory': str(self.output_dir),
                'plots': self._list_plot_files(all_reporter_results),
                'data_files': self._list_data_files()
            }
        }
        
        # Add component interpretations if available
        if 'component_reporter' in all_reporter_results:
            comp_results = all_reporter_results['component_reporter']
            if 'statistics' in comp_results:
                summary['component_statistics'] = comp_results['statistics']
        
        return summary
    
    def _generate_html_report(self, text_summary: str,
                            json_summary: Dict[str, Any],
                            dashboard_path: Path,
                            all_reporter_results: Dict[str, Any]) -> Path:
        """Generate HTML report with embedded visualizations.
        
        Args:
            text_summary: Text summary
            json_summary: JSON summary
            dashboard_path: Path to dashboard plot
            all_reporter_results: Results from all reporters
            
        Returns:
            Path to HTML report
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CP-WOPT Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CP-WOPT Tensor Factorization Report</h1>
        <p>Generated: {json_summary['metadata']['generated_at']}</p>
        
        <h2>Key Metrics</h2>
        <div>
            <div class="metric">
                <div class="metric-value">{json_summary['performance']['fit_score']:.3f}</div>
                <div class="metric-label">Fit Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{json_summary['model']['rank']}</div>
                <div class="metric-label">Components</div>
            </div>
            <div class="metric">
                <div class="metric-value">{json_summary['data']['sparsity']:.1%}</div>
                <div class="metric-label">Sparsity</div>
            </div>
            <div class="metric">
                <div class="metric-value">{json_summary['performance']['runtime_seconds']:.1f}s</div>
                <div class="metric-label">Runtime</div>
            </div>
        </div>
        
        <h2>Summary Dashboard</h2>
        <img src="{dashboard_path.name}" alt="Summary Dashboard">
        
        <h2>Detailed Summary</h2>
        <pre>{text_summary}</pre>
        
        <h2>Additional Plots</h2>
        <div class="plot-grid">
            {self._generate_plot_html(all_reporter_results)}
        </div>
    </div>
</body>
</html>
        """
        
        html_path = self.output_dir / 'summary_report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _find_n_components_for_variance(self, weights: Optional[np.ndarray], 
                                      threshold: float) -> Optional[int]:
        """Find number of components needed for variance threshold.
        
        Args:
            weights: Component weights
            threshold: Variance threshold
            
        Returns:
            Number of components or None
        """
        if weights is None:
            return None
        
        sorted_weights = np.sort(weights)[::-1]
        cumvar = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        return int(np.argmax(cumvar >= threshold) + 1)
    
    def _list_plot_files(self, all_reporter_results: Dict[str, Any]) -> List[str]:
        """List all generated plot files.
        
        Args:
            all_reporter_results: Results from all reporters
            
        Returns:
            List of plot file names
        """
        plot_files = []
        
        for reporter_name, results in all_reporter_results.items():
            if 'plots' in results:
                for plot_name, plot_path in results['plots'].items():
                    if isinstance(plot_path, Path):
                        plot_files.append(plot_path.name)
        
        return plot_files
    
    def _list_data_files(self) -> List[str]:
        """List all data files in output directory.
        
        Returns:
            List of data file names
        """
        data_extensions = ['.npy', '.json', '.csv', '.txt']
        data_files = []
        
        for ext in data_extensions:
            data_files.extend([f.name for f in self.output_dir.glob(f'*{ext}')])
        
        return sorted(data_files)
    
    def _generate_plot_html(self, all_reporter_results: Dict[str, Any]) -> str:
        """Generate HTML for embedding plots.
        
        Args:
            all_reporter_results: Results from all reporters
            
        Returns:
            HTML string
        """
        html_parts = []
        
        # Select key plots to show
        key_plots = [
            ('factor_reporter', 'feature_loadings', 'Feature Loadings'),
            ('spatial_reporter', 'spatial_factor_grids', 'Spatial Patterns'),
            ('reconstruction_reporter', 'convergence', 'Convergence'),
            ('component_reporter', 'component_similarities', 'Component Similarities')
        ]
        
        for reporter, plot_key, title in key_plots:
            if reporter in all_reporter_results:
                results = all_reporter_results[reporter]
                if 'plots' in results and plot_key in results['plots']:
                    plot_path = results['plots'][plot_key]
                    if isinstance(plot_path, Path):
                        html_parts.append(f'''
                        <div>
                            <h3>{title}</h3>
                            <img src="{plot_path.name}" alt="{title}">
                        </div>
                        ''')
        
        return '\n'.join(html_parts)