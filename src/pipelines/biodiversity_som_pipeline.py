"""Complete pipeline for biodiversity SOM analysis."""

from typing import Dict, Any, Optional, List
from src.pipelines.base_pipeline import BasePipeline
from src.pipelines.stages.load_stage import LoadStage
from src.pipelines.stages.analysis_stage import AnalysisStage
from src.biodiversity_analysis.methods.som.visualization import SOMVisualizer
import logging
import os
import json

logger = logging.getLogger(__name__)


class BiodiversitySOMPipeline(BasePipeline):
    """Complete pipeline for biodiversity SOM analysis.
    
    This pipeline orchestrates the entire SOM analysis workflow:
    1. Load biodiversity data from parquet files
    2. Run SOM analysis with spatial validation
    3. Generate visualizations and reports
    
    Example usage:
        config = {
            'biodiversity_data_path': '/path/to/data.parquet',
            'som_analysis': {
                'grid_size': [10, 10],
                'distance_metric': 'bray_curtis',
                'epochs': 100
            }
        }
        
        pipeline = BiodiversitySOMPipeline(config)
        results = pipeline.run(generate_plots=True)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline.
        
        Args:
            config: Configuration dictionary containing:
                - biodiversity_data_path: Path to parquet file
                - som_analysis: SOM configuration parameters
                - output_dir: Optional output directory
        """
        super().__init__(config)
        self.output_dir = config.get('output_dir', './som_results')
        self.setup_stages()
    
    def setup_stages(self):
        """Configure pipeline stages."""
        # Stage 1: Data loading
        self.add_stage(LoadStage(
            self.config,
            stage_config={
                'file_path': self.config.get('biodiversity_data_path'),
                'file_type': 'parquet',
                'dataset_name': 'biodiversity_data'
            }
        ))
        
        # Stage 2: SOM analysis
        som_config = self.config.get('som_analysis', {})
        self.add_stage(AnalysisStage(
            self.config,
            stage_config={
                'analysis_method': 'som',
                'parameters': som_config,
                'spatial_validation': som_config.get('spatial_validation', True),
                'spatial_strategy': som_config.get('spatial_strategy', 'random_blocks'),
                'save_results': True,
                'output_dir': os.path.join(self.output_dir, 'analysis')
            }
        ))
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the pipeline.
        
        Args:
            **kwargs: Additional arguments:
                - generate_plots: Whether to generate visualizations (default: True)
                - generate_report: Whether to generate HTML report (default: True)
                - plot_formats: List of formats for plots (default: ['png'])
                
        Returns:
            Dictionary with pipeline results including:
                - loaded_data: Loaded dataset
                - analysis_result: SOM analysis results
                - visualizations: Generated plot paths
                - report_path: Path to generated report
        """
        logger.info("Starting Biodiversity SOM Pipeline")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Run all stages
        results = super().run(**kwargs)
        
        # Extract results
        pipeline_results = {
            'loaded_data': results.get('dataset'),
            'analysis_result': results.get('analysis_result')
        }
        
        # Generate visualizations if requested
        if kwargs.get('generate_plots', True):
            visualization_paths = self._generate_visualizations(
                pipeline_results['analysis_result'],
                pipeline_results['loaded_data'],
                plot_formats=kwargs.get('plot_formats', ['png'])
            )
            pipeline_results['visualizations'] = visualization_paths
        
        # Generate report if requested
        if kwargs.get('generate_report', True):
            report_path = self._generate_report(pipeline_results)
            pipeline_results['report_path'] = report_path
        
        # Save pipeline metadata
        self._save_pipeline_metadata(pipeline_results)
        
        logger.info(f"Pipeline completed. Results saved to {self.output_dir}")
        return pipeline_results
    
    def _generate_visualizations(self, analysis_result: Any, 
                               loaded_data: Any,
                               plot_formats: List[str] = ['png']) -> Dict[str, str]:
        """Generate SOM visualizations.
        
        Args:
            analysis_result: SOM analysis result
            loaded_data: Loaded dataset
            plot_formats: Formats to save plots
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("Generating SOM visualizations")
        
        visualizer = SOMVisualizer()
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        paths = {}
        
        # Extract data from results
        weights = analysis_result.metadata.get('weights')
        if weights is None:
            # Load from saved file if available
            weights_path = os.path.join(self.output_dir, 'analysis', 'som_weights.npy')
            if os.path.exists(weights_path):
                import numpy as np
                weights = np.load(weights_path)
        
        if weights is not None:
            # 1. U-matrix
            for fmt in plot_formats:
                u_matrix_path = os.path.join(viz_dir, f'u_matrix.{fmt}')
                fig = visualizer.plot_u_matrix(weights, save_path=u_matrix_path)
                paths['u_matrix'] = u_matrix_path
                plt.close(fig)
            
            # 2. Component planes
            feature_names = analysis_result.metadata['data_info'].get('feature_names', [])
            if feature_names:
                for fmt in plot_formats:
                    comp_path = os.path.join(viz_dir, f'component_planes.{fmt}')
                    fig = visualizer.plot_component_planes(
                        weights, feature_names[:30], save_path=comp_path
                    )
                    paths['component_planes'] = comp_path
                    plt.close(fig)
            
            # 3. Hit map
            if analysis_result.labels is not None:
                for fmt in plot_formats:
                    hit_path = os.path.join(viz_dir, f'hit_map.{fmt}')
                    fig = visualizer.plot_hit_map(
                        analysis_result.labels, 
                        weights.shape[:2],
                        save_path=hit_path
                    )
                    paths['hit_map'] = hit_path
                    plt.close(fig)
            
            # 4. Training history
            training_metrics = analysis_result.metadata.get('training_metrics', {})
            if training_metrics.get('quantization_errors'):
                # Create a mock training result object
                from src.abstractions.types.som_types import SOMTrainingResult
                training_result = SOMTrainingResult(
                    weights=weights,
                    quantization_errors=training_metrics['quantization_errors'],
                    topographic_errors=training_metrics['topographic_errors'],
                    training_time=training_metrics.get('training_time', 0),
                    convergence_epoch=training_metrics.get('convergence_epoch'),
                    final_learning_rate=0.01,
                    final_neighborhood_radius=0.1,
                    n_samples_trained=0
                )
                
                for fmt in plot_formats:
                    history_path = os.path.join(viz_dir, f'training_history.{fmt}')
                    fig = visualizer.plot_training_history(
                        training_result, save_path=history_path
                    )
                    paths['training_history'] = history_path
                    plt.close(fig)
            
            # 5. Biodiversity patterns (if we have the original data)
            # This would require access to the original species data
            # which might not be available in the pipeline results
        
        logger.info(f"Generated {len(paths)} visualizations")
        return paths
    
    def _generate_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report summarizing the analysis.
        
        Args:
            results: Pipeline results
            
        Returns:
            Path to generated report
        """
        logger.info("Generating analysis report")
        
        report_path = os.path.join(self.output_dir, 'som_analysis_report.html')
        
        # Extract key metrics
        analysis_result = results['analysis_result']
        training_metrics = analysis_result.metadata.get('training_metrics', {})
        validation_metrics = analysis_result.metadata.get('validation_metrics', {})
        biodiv_metrics = analysis_result.metadata.get('biodiversity_metrics', {})
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Biodiversity SOM Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .metric {{ margin: 10px 0; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ color: #0066cc; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Biodiversity SOM Analysis Report</h1>
            
            <h2>Dataset Information</h2>
            <div class="metric">
                <span class="metric-name">Number of samples:</span>
                <span class="metric-value">{analysis_result.metadata['data_info'].get('n_samples', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Number of features:</span>
                <span class="metric-value">{analysis_result.metadata['data_info'].get('n_features', 'N/A')}</span>
            </div>
            
            <h2>SOM Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Grid Size</td><td>{analysis_result.parameters.get('grid_size', 'N/A')}</td></tr>
                <tr><td>Distance Metric</td><td>{analysis_result.parameters.get('distance_metric', 'N/A')}</td></tr>
                <tr><td>Epochs</td><td>{analysis_result.parameters.get('epochs', 'N/A')}</td></tr>
                <tr><td>Initialization Method</td><td>{analysis_result.parameters.get('initialization_method', 'N/A')}</td></tr>
            </table>
            
            <h2>Training Results</h2>
            <div class="metric">
                <span class="metric-name">Convergence Epoch:</span>
                <span class="metric-value">{training_metrics.get('convergence_epoch', 'Did not converge')}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Training Time:</span>
                <span class="metric-value">{training_metrics.get('training_time', 0):.2f} seconds</span>
            </div>
            
            <h2>Validation Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Train</th><th>Validation</th><th>Test</th></tr>
                <tr>
                    <td>Quantization Error</td>
                    <td>{validation_metrics.get('train_qe', 0):.4f}</td>
                    <td>{validation_metrics.get('val_qe', 0):.4f}</td>
                    <td>{validation_metrics.get('test_qe', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Topographic Error</td>
                    <td>{validation_metrics.get('train_te', 0):.4f}</td>
                    <td>{validation_metrics.get('val_te', 0):.4f}</td>
                    <td>{validation_metrics.get('test_te', 0):.4f}</td>
                </tr>
            </table>
            
            <h2>Biodiversity Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Species Coherence</td><td>{biodiv_metrics.get('species_coherence', 0):.3f}</td></tr>
                <tr><td>Beta Diversity Preservation</td><td>{biodiv_metrics.get('beta_diversity_preservation', 0):.3f}</td></tr>
                <tr><td>Rare Species Representation</td><td>{biodiv_metrics.get('rare_species_representation', 0):.3f}</td></tr>
                <tr><td>Spatial Autocorrelation Preserved</td><td>{biodiv_metrics.get('spatial_autocorrelation_preserved', 0):.3f}</td></tr>
            </table>
            
            <h2>Visualizations</h2>
            {self._generate_visualization_section(results.get('visualizations', {}))}
            
            <p><em>Report generated on {self._get_timestamp()}</em></p>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {report_path}")
        return report_path
    
    def _generate_visualization_section(self, viz_paths: Dict[str, str]) -> str:
        """Generate HTML for visualization section."""
        html = ""
        
        viz_titles = {
            'u_matrix': 'U-Matrix',
            'component_planes': 'Component Planes',
            'hit_map': 'Hit Map',
            'training_history': 'Training History'
        }
        
        for viz_type, path in viz_paths.items():
            if os.path.exists(path):
                rel_path = os.path.relpath(path, self.output_dir)
                title = viz_titles.get(viz_type, viz_type.replace('_', ' ').title())
                html += f'<h3>{title}</h3>\n<img src="{rel_path}" alt="{title}">\n'
        
        return html
    
    def _save_pipeline_metadata(self, results: Dict[str, Any]):
        """Save pipeline execution metadata."""
        metadata = {
            'pipeline': 'BiodiversitySOMPipeline',
            'version': '1.0.0',
            'execution_time': self._get_timestamp(),
            'config': self.config,
            'stages_executed': [stage.__class__.__name__ for stage in self.stages],
            'output_files': {
                'visualizations': results.get('visualizations', {}),
                'report': results.get('report_path', ''),
                'analysis_output': os.path.join(self.output_dir, 'analysis')
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'pipeline_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Pipeline metadata saved to {metadata_path}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Import matplotlib here to avoid import errors if not needed
try:
    import matplotlib.pyplot as plt
except ImportError:
    logger.warning("Matplotlib not available. Visualization features will be limited.")