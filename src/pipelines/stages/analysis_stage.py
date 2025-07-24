# src/pipelines/stages/analysis_stage.py
"""Analysis stage for SOM."""

from typing import List, Tuple
import logging

from .base_stage import PipelineStage, StageResult
from src.spatial_analysis.som.som_trainer import SOMAnalyzer

logger = logging.getLogger(__name__)


class AnalysisStage(PipelineStage):
    """Stage for SOM analysis."""
    
    @property
    def name(self) -> str:
        return "analysis"
    
    @property
    def dependencies(self) -> List[str]:
        return ["merge"]
    
    @property
    def memory_requirements(self) -> float:
        # Based on SOM memory requirements
        return 8.0  # GB
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate analysis configuration."""
        return True, []
    
    def execute(self, context) -> StageResult:
        """Execute SOM analysis."""
        logger.info("Starting analysis stage")
        
        try:
            # Get merged dataset
            merged_dataset = context.get('merged_dataset')
            if merged_dataset is None:
                return StageResult(
                    success=False,
                    data={},
                    metrics={},
                    warnings=['No merged dataset available']
                )
            
            # Initialize SOM analyzer
            analyzer = SOMAnalyzer(context.config, context.db)
            
            # Get SOM parameters
            som_config = context.config.get('som_analysis', {})
            som_params = {
                'grid_size': som_config.get('default_grid_size', [8, 8]),
                'iterations': som_config.get('iterations', 1000),
                'sigma': som_config.get('sigma', 1.5),
                'learning_rate': som_config.get('learning_rate', 0.5),
                'neighborhood_function': som_config.get('neighborhood_function', 'gaussian'),
                'random_seed': som_config.get('random_seed', 42)
            }
            
            logger.info(f"Running SOM analysis with parameters: {som_params}")
            
            # Run analysis
            results = analyzer.analyze(
                data=merged_dataset,
                **som_params
            )
            
            # Save results
            output_name = f"SOM_Analysis_{context.experiment_id}"
            saved_path = analyzer.save_results(results, output_name)
            
            # Store results in context
            context.set('som_results', results)
            context.set('som_output_path', saved_path)
            
            # Clean NaN values for metrics
            from src.pipelines.unified_resampling.pipeline_orchestrator import clean_nan_for_json
            
            metrics = {
                'grid_size': som_params['grid_size'],
                'iterations': som_params['iterations'],
                'n_samples': getattr(results.metadata, 'n_samples', 0),
                'n_features': getattr(results.metadata, 'n_features', 0),
                'output_path': str(saved_path)
            }
            
            # Add statistics if available
            if hasattr(results, 'statistics'):
                clean_stats = clean_nan_for_json(results.statistics)
                metrics['analysis_stats'] = clean_stats
            
            return StageResult(
                success=True,
                data={'analysis_results_path': str(saved_path)},
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Analysis stage failed: {e}")
            raise