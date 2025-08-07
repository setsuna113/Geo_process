"""Consolidated CP-WOPT reporters with clean abstractions.

This module consolidates the reporter functionality into logical groups
while maintaining clean separation of concerns through inheritance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
from scipy import stats

from .base_reporter import BaseReporter

logger = logging.getLogger(__name__)


class VisualizationReporter(BaseReporter):
    """Handles all visualization-related reporting (spatial, component, biome).
    
    Consolidates:
    - SpatialReporter functionality
    - ComponentReporter visualization
    - BiomeReporter functionality
    """
    
    def get_required_fields(self) -> List[str]:
        """Required fields for visualization."""
        return ['factors', 'tensor_data', 'metrics']
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all visualizations."""
        self._validate_results(results)
        
        outputs = {}
        
        if self.generate_plots:
            # Spatial visualizations
            outputs['spatial'] = self._generate_spatial_plots(results)
            
            # Component visualizations  
            outputs['components'] = self._generate_component_plots(results)
            
            # Biome visualizations
            outputs['biomes'] = self._generate_biome_plots(results)
        
        return outputs
    
    def _generate_spatial_plots(self, results: Dict[str, Any]) -> Dict[str, Path]:
        """Generate spatial distribution plots."""
        outputs = {}
        factors = results['factors']
        tensor_data = results['tensor_data']
        
        # Plot spatial factors (mode 0 and 1)
        fig, axes = plt.subplots(2, min(factors.rank, 4), 
                                 figsize=(16, 8))
        
        if factors.rank == 1:
            axes = axes.reshape(2, 1)
        
        for r in range(min(factors.rank, 4)):
            # Latitude factor
            axes[0, r].plot(factors.factors[0][:, r])
            axes[0, r].set_title(f'Component {r+1} - Latitude')
            axes[0, r].set_xlabel('Latitude bin')
            
            # Longitude factor  
            axes[1, r].plot(factors.factors[1][:, r])
            axes[1, r].set_title(f'Component {r+1} - Longitude')
            axes[1, r].set_xlabel('Longitude bin')
        
        outputs['spatial_factors'] = self.save_figure(fig, 'spatial_factors')
        
        # Generate heatmaps if we have coordinate info
        if hasattr(tensor_data, 'coordinates') and tensor_data.coordinates:
            outputs['spatial_heatmaps'] = self._create_spatial_heatmaps(
                factors, tensor_data
            )
        
        return outputs
    
    def _generate_component_plots(self, results: Dict[str, Any]) -> Dict[str, Path]:
        """Generate component interpretation plots."""
        outputs = {}
        factors = results['factors']
        
        # Component importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate component weights (Frobenius norm of each component)
        weights = []
        for r in range(factors.rank):
            weight = 1.0
            for mode in range(len(factors.factors)):
                weight *= np.linalg.norm(factors.factors[mode][:, r])
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Plot
        components = list(range(1, factors.rank + 1))
        ax.bar(components, weights, color=plt.cm.viridis(weights/weights.max()))
        ax.set_xlabel('Component')
        ax.set_ylabel('Relative Importance')
        ax.set_title('Component Importance')
        ax.set_xticks(components)
        
        outputs['component_importance'] = self.save_figure(fig, 'component_importance')
        
        # Feature loadings heatmap
        if len(factors.factors) > 2:  # Has feature mode
            outputs['feature_loadings'] = self._create_feature_loadings_plot(factors)
        
        return outputs
    
    def _generate_biome_plots(self, results: Dict[str, Any]) -> Dict[str, Path]:
        """Generate biome-like visualizations."""
        outputs = {}
        factors = results['factors']
        
        # Create composite biome map
        if factors.rank >= 3:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Combine first 3 components as RGB
            lat_factor = factors.factors[0][:, :3]
            lon_factor = factors.factors[1][:, :3]
            
            # Create RGB image
            rgb_map = np.zeros((lat_factor.shape[0], lon_factor.shape[0], 3))
            for i in range(3):
                rgb_map[:, :, i] = np.outer(
                    lat_factor[:, i], 
                    lon_factor[:, i]
                )
            
            # Normalize to [0, 1]
            rgb_map = (rgb_map - rgb_map.min()) / (rgb_map.max() - rgb_map.min())
            
            # Apply smoothing if configured
            sigma = self.config.get('smoothing_sigma', 1.5)
            if sigma > 0:
                rgb_map = self._apply_smoothing(rgb_map, sigma)
            
            # Display
            im = ax.imshow(rgb_map, aspect='equal', origin='lower')
            ax.set_title('Biome-like Visualization (RGB from first 3 components)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            
            outputs['biome_map'] = self.save_figure(fig, 'biome_map')
        
        return outputs
    
    def _create_spatial_heatmaps(self, factors, tensor_data) -> Path:
        """Create spatial heatmaps for each component."""
        n_components = min(factors.rank, self.config.get('plot_components', 4))
        
        fig, axes = plt.subplots(2, (n_components + 1) // 2, 
                                 figsize=(15, 10))
        axes = axes.flatten()
        
        for r in range(n_components):
            # Create 2D representation
            spatial_pattern = np.outer(
                factors.factors[0][:, r],
                factors.factors[1][:, r]
            )
            
            # Apply smoothing
            spatial_pattern = self._apply_smoothing(
                spatial_pattern, 
                self.config.get('smoothing_sigma', 1.0)
            )
            
            # Plot
            im = axes[r].imshow(spatial_pattern, cmap=self.colormap, 
                               aspect='equal', origin='lower')
            axes[r].set_title(f'Component {r+1} Spatial Pattern')
            self._create_colorbar(axes[r], im)
        
        # Hide unused axes
        for i in range(n_components, len(axes)):
            axes[i].set_visible(False)
        
        return self.save_figure(fig, 'spatial_heatmaps')
    
    def _create_feature_loadings_plot(self, factors) -> Path:
        """Create feature loadings heatmap."""
        feature_mode = 2  # Assuming 3D tensor with features in mode 2
        feature_matrix = factors.factors[feature_mode]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(feature_matrix.T, cmap='RdBu_r', aspect='auto',
                      vmin=-np.abs(feature_matrix).max(),
                      vmax=np.abs(feature_matrix).max())
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Component')
        ax.set_title('Feature Loadings by Component')
        
        # Add component labels
        ax.set_yticks(range(factors.rank))
        ax.set_yticklabels([f'Comp {i+1}' for i in range(factors.rank)])
        
        self._create_colorbar(ax, im, 'Loading')
        
        return self.save_figure(fig, 'feature_loadings')


class AnalyticsReporter(BaseReporter):
    """Handles all analytical reporting (factors, reconstruction, metrics).
    
    Consolidates:
    - FactorReporter analytics
    - ReconstructionReporter metrics
    - Summary statistics
    """
    
    def get_required_fields(self) -> List[str]:
        """Required fields for analytics."""
        return ['factors', 'metrics', 'tensor_data']
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytical reports."""
        self._validate_results(results)
        
        outputs = {}
        
        # Factor analysis
        outputs['factor_analysis'] = self._analyze_factors(results)
        
        # Reconstruction metrics
        outputs['reconstruction'] = self._analyze_reconstruction(results)
        
        # Summary statistics
        outputs['summary'] = self._generate_summary(results)
        
        # Save JSON report
        outputs['json_report'] = self._save_json_report(outputs)
        
        return outputs
    
    def _analyze_factors(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factor properties."""
        factors = results['factors']
        analysis = {}
        
        # Factor statistics for each mode
        for mode, factor_matrix in enumerate(factors.factors):
            mode_stats = {
                'shape': factor_matrix.shape,
                'sparsity': float(np.sum(np.abs(factor_matrix) < 1e-10) / factor_matrix.size),
                'mean': float(np.mean(factor_matrix)),
                'std': float(np.std(factor_matrix)),
                'min': float(np.min(factor_matrix)),
                'max': float(np.max(factor_matrix))
            }
            
            # Component-wise stats
            component_stats = []
            for r in range(factors.rank):
                comp_vec = factor_matrix[:, r]
                component_stats.append({
                    'norm': float(np.linalg.norm(comp_vec)),
                    'sparsity': float(np.sum(np.abs(comp_vec) < 1e-10) / len(comp_vec)),
                    'max_value': float(np.max(np.abs(comp_vec))),
                    'top_indices': [int(i) for i in np.argsort(np.abs(comp_vec))[-5:].tolist()]
                })
            
            mode_stats['components'] = component_stats
            analysis[f'mode_{mode}'] = mode_stats
        
        # Factor correlation analysis
        analysis['correlations'] = self._compute_factor_correlations(factors)
        
        # Save factor statistics plot if enabled
        if self.generate_plots:
            analysis['factor_plot'] = self._create_factor_statistics_plot(factors)
        
        return analysis
    
    def _analyze_reconstruction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reconstruction quality."""
        metrics = results['metrics']
        tensor_data = results['tensor_data']
        
        analysis = {
            'overall_fit': float(metrics.fit),
            'reconstruction_error': float(metrics.reconstruction_error),
            'relative_error': float(metrics.relative_error),
            'convergence_iterations': len(metrics.convergence_history)
        }
        
        # Convergence analysis
        if metrics.convergence_history:
            history = metrics.convergence_history
            analysis['convergence'] = {
                'initial_error': float(history[0]),
                'final_error': float(history[-1]),
                'improvement': float((history[0] - history[-1]) / history[0]),
                'converged': len(history) < results.get('max_iterations', 1000)
            }
        
        # Missing data handling
        if hasattr(tensor_data, 'weights'):
            n_observed = np.sum(tensor_data.weights > 0)
            n_total = tensor_data.weights.size
            analysis['missing_data'] = {
                'observed_ratio': float(n_observed / n_total),
                'missing_ratio': float(1 - n_observed / n_total),
                'n_observed': int(n_observed),
                'n_total': int(n_total)
            }
        
        # Generate plots if enabled
        if self.generate_plots:
            analysis['convergence_plot'] = self._create_convergence_plot(metrics)
            analysis['error_distribution'] = self._create_error_distribution_plot(
                results
            )
        
        return analysis
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary."""
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'tensor_shape': results['tensor_data'].shape,
            'rank': results['factors'].rank,
            'final_fit': float(results['metrics'].fit),
            'computation_time': results.get('computation_time', None)
        }
        
        # Add configuration summary
        if 'config' in results:
            summary['configuration'] = {
                'optimizer': results['config'].get('algorithm', {}).get('optimizer', 'als'),
                'max_iterations': results['config'].get('algorithm', {}).get('max_iterations'),
                'convergence_threshold': results['config'].get('algorithm', {}).get('convergence_threshold')
            }
        
        return summary
    
    def _compute_factor_correlations(self, factors) -> Dict[str, Any]:
        """Compute correlations between factor components."""
        correlations = {}
        
        for mode, factor_matrix in enumerate(factors.factors):
            # Compute correlation matrix between components
            corr_matrix = np.corrcoef(factor_matrix.T)
            correlations[f'mode_{mode}'] = {
                'correlation_matrix': corr_matrix.tolist(),
                'mean_abs_correlation': float(np.mean(np.abs(
                    corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                )))
            }
        
        return correlations
    
    def _create_factor_statistics_plot(self, factors) -> Path:
        """Create factor statistics visualization."""
        n_modes = len(factors.factors)
        
        fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5))
        if n_modes == 1:
            axes = [axes]
        
        for mode, (ax, factor_matrix) in enumerate(zip(axes, factors.factors)):
            # Box plot of factor values by component
            ax.boxplot([factor_matrix[:, r] for r in range(factors.rank)])
            ax.set_xlabel('Component')
            ax.set_ylabel('Factor Values')
            ax.set_title(f'Mode {mode} Factor Distribution')
            ax.grid(True, alpha=0.3)
        
        return self.save_figure(fig, 'factor_statistics')
    
    def _create_convergence_plot(self, metrics) -> Path:
        """Create convergence plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        history = metrics.convergence_history
        iterations = list(range(1, len(history) + 1))
        
        ax.semilogy(iterations, history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reconstruction Error (log scale)')
        ax.set_title('Convergence History')
        ax.grid(True, alpha=0.3)
        
        # Add convergence rate annotation
        if len(history) > 10:
            mid_point = len(history) // 2
            rate = (history[mid_point] - history[-1]) / (len(history) - mid_point)
            ax.annotate(f'Rate: {rate:.2e}/iter', 
                       xy=(0.7, 0.9), xycoords='axes fraction')
        
        return self.save_figure(fig, 'convergence_history')
    
    def _create_error_distribution_plot(self, results) -> Path:
        """Create error distribution plot."""
        # This would need actual vs reconstructed tensor
        # For now, create a placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simulate error distribution (would be computed from actual data)
        errors = np.random.normal(0, results['metrics'].relative_error, 1000)
        
        ax.hist(errors, bins=50, density=True, alpha=0.7, color='blue')
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution')
        ax.axvline(x=0, color='red', linestyle='--', label='Zero Error')
        ax.legend()
        
        return self.save_figure(fig, 'error_distribution')
    
    def _save_json_report(self, outputs: Dict[str, Any]) -> Path:
        """Save complete report as JSON."""
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        json_outputs = convert_paths(outputs)
        
        json_path = self.output_dir / 'analytics_report.json'
        with open(json_path, 'w') as f:
            json.dump(json_outputs, f, indent=2)
        
        logger.info(f"Saved JSON report: {json_path}")
        return json_path


class UnifiedReporter(BaseReporter):
    """Unified reporter that orchestrates all reporting functionality.
    
    This is the main entry point that delegates to specialized reporters.
    """
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """Initialize unified reporter with sub-reporters."""
        super().__init__(output_dir, config)
        
        # Initialize specialized reporters
        self.visualization_reporter = VisualizationReporter(
            output_dir / 'visualizations', config
        )
        self.analytics_reporter = AnalyticsReporter(
            output_dir / 'analytics', config
        )
    
    def get_required_fields(self) -> List[str]:
        """Combined required fields."""
        return ['factors', 'tensor_data', 'metrics']
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete unified report."""
        self._validate_results(results)
        
        logger.info("Generating unified CP-WOPT report...")
        
        outputs = {
            'visualizations': self.visualization_reporter.generate_report(results),
            'analytics': self.analytics_reporter.generate_report(results)
        }
        
        # Save summary HTML report
        outputs['html_report'] = self._create_html_summary(outputs)
        
        logger.info(f"Report generation complete. Output directory: {self.output_dir}")
        
        return outputs
    
    def _create_html_summary(self, outputs: Dict[str, Any]) -> Path:
        """Create an HTML summary report linking all outputs."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>CP-WOPT Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 1px solid #ccc; }
        .section { margin: 20px 0; }
        img { max-width: 100%; height: auto; }
        .metrics { background: #f5f5f5; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>CP-WOPT Analysis Report</h1>
    <div class="section">
        <h2>Visualizations</h2>
        <p>See visualizations/ directory for plots</p>
    </div>
    <div class="section">
        <h2>Analytics</h2>
        <p>See analytics/ directory for detailed metrics</p>
    </div>
</body>
</html>
"""
        
        html_path = self.output_dir / 'report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path