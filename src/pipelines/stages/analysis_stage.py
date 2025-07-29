# src/pipelines/stages/analysis_stage.py
"""Analysis stage for multiple spatial analysis methods."""

from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
import pandas as pd
import xarray as xr
import gc

from .base_stage import PipelineStage, StageResult
from .analyzer_factory import AnalyzerFactory
from src.infrastructure.logging import get_logger
from src.processors.data_preparation.analysis_data_source import (
    ParquetAnalysisDataset, DatabaseAnalysisDataset
)
from src.base.dataset import BaseDataset

logger = get_logger(__name__)


class AnalysisStage(PipelineStage):
    """Stage for spatial analysis (SOM, MaxP, GWPCA)."""
    
    def __init__(self, analysis_method: str = 'som'):
        """
        Initialize analysis stage.
        
        Args:
            analysis_method: Type of analysis ('som', 'maxp_regions', 'gwpca')
        """
        super().__init__()
        self.analysis_method = analysis_method.lower()
        self._analyzer = None
        self._validate_method()
    
    @property
    def name(self) -> str:
        return f"analysis_{self.analysis_method}"
    
    @property
    def dependencies(self) -> List[str]:
        # Analysis happens after export (so it can use CSV if needed)
        return ["export"]
    
    @property
    def memory_requirements(self) -> float:
        # Different methods have different requirements
        requirements = {
            'som': 8.0,
            'maxp_regions': 12.0,  # More memory intensive
            'gwpca': 10.0
        }
        return requirements.get(self.analysis_method, 8.0)
    
    def _validate_method(self):
        """Validate analysis method is supported."""
        available_methods = AnalyzerFactory.available_methods()
        if self.analysis_method not in available_methods:
            raise ValueError(
                f"Unsupported analysis method: {self.analysis_method}. "
                f"Available: {available_methods}"
            )
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate analysis configuration."""
        errors = []
        
        # Validate factory can create analyzer
        try:
            available = AnalyzerFactory.available_methods()
            if self.analysis_method not in available:
                errors.append(f"Analysis method '{self.analysis_method}' not available")
        except Exception as e:
            errors.append(f"Factory validation failed: {e}")
        
        return len(errors) == 0, errors
    
    def execute(self, context) -> StageResult:
        """Execute analysis with full integration."""
        logger.info(
            f"Starting {self.analysis_method} analysis",
            extra={
                'experiment_id': context.experiment_id,
                'stage': self.name,
                'method': self.analysis_method
            }
        )
        
        try:
            # Step 1: Load dataset
            logger.debug("Loading dataset")
            dataset = self._load_dataset(context)
            dataset_info = dataset.load_info()
            
            logger.info(
                f"Dataset loaded: {dataset_info.record_count:,} records, "
                f"{dataset_info.size_mb:.2f} MB"
            )
            
            # Step 2: Create analyzer via factory
            logger.debug("Creating analyzer via factory")
            self._analyzer = AnalyzerFactory.create(
                self.analysis_method,
                context.config,
                context.db
            )
            
            # Step 3: Set up progress tracking
            self._setup_progress_tracking(context)
            
            # Step 4: Get parameters
            params = self._get_analysis_params(context)
            logger.debug(f"Analysis parameters: {params}")
            
            # Step 5: Perform analysis
            logger.info("Starting analysis computation")
            results = self._analyzer.analyze(dataset, **params)
            
            # Step 6: Save results if configured
            saved_path = None
            if context.config.get('analysis.save_results.enabled', True):
                output_dir = context.output_dir / 'analysis' / self.analysis_method
                saved_path = self._analyzer.save_results(
                    results,
                    f"{self.analysis_method}_{context.experiment_id}",
                    output_dir
                )
                context.set(f'{self.analysis_method}_output_path', str(saved_path))
                
                logger.info(f"Results saved to: {saved_path}")
            
            # Step 7: Store in context if configured
            if context.config.get('analysis.keep_results_in_memory', False):
                context.set(f'{self.analysis_method}_results', results)
            
            # Step 8: Extract metrics
            metrics = self._extract_metrics(results, params)
            if saved_path:
                metrics['output_path'] = str(saved_path)
            
            # Step 9: Update progress to complete
            if hasattr(context, 'progress_tracker') and context.progress_tracker:
                context.progress_tracker.update_progress(
                    node_id=f"analysis/{self.analysis_method}",
                    completed_units=100,
                    status="completed"
                )
            
            logger.info(
                f"Analysis completed successfully",
                extra={'metrics': metrics}
            )
            
            return StageResult(
                success=True,
                data={
                    'analysis_method': self.analysis_method,
                    'output_path': str(saved_path) if saved_path else None
                },
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(
                f"{self.analysis_method} analysis failed: {e}",
                exc_info=True,
                extra={
                    'experiment_id': context.experiment_id,
                    'stage': self.name,
                    'error_type': type(e).__name__
                }
            )
            
            # Update progress to failed
            if hasattr(context, 'progress_tracker') and context.progress_tracker:
                context.progress_tracker.update_progress(
                    node_id=f"analysis/{self.analysis_method}",
                    completed_units=0,
                    status="failed",
                    metadata={"error": str(e)}
                )
            
            raise
    
    def _load_dataset(self, context) -> BaseDataset:
        """Load dataset using appropriate source."""
        data_source = context.config.get('analysis.data_source', 'parquet')
        
        if data_source == 'parquet':
            parquet_path = Path(context.get('ml_ready_path'))
            if not parquet_path or not parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            
            return ParquetAnalysisDataset(
                parquet_path,
                chunk_size=context.config.get('analysis.chunk_size', 10000)
            )
        
        elif data_source == 'database':
            return DatabaseAnalysisDataset(
                context.db,
                experiment_id=context.experiment_id,
                chunk_size=context.config.get('analysis.chunk_size', 10000)
            )
        
        elif data_source == 'csv':
            # Legacy CSV support
            csv_path = Path(context.get('exported_csv_path'))
            if not csv_path or not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # Create a simple CSV dataset wrapper
            # This is a fallback for backward compatibility
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            # Convert to parquet temporarily for uniform handling
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                df.to_parquet(tmp.name, index=False)
                return ParquetAnalysisDataset(Path(tmp.name))
        
        else:
            raise ValueError(f"Unknown data source: {data_source}")
    
    def _setup_progress_tracking(self, context):
        """Set up progress tracking integration."""
        if hasattr(context, 'progress_tracker') and context.progress_tracker:
            # Create progress node
            context.progress_tracker.create_node(
                node_id=f"analysis/{self.analysis_method}",
                parent_id="analysis",
                level="step",
                name=f"{self.analysis_method} analysis",
                total_units=100
            )
            
            # Create callback
            def progress_callback(current: int, total: int, message: str):
                context.progress_tracker.update_progress(
                    node_id=f"analysis/{self.analysis_method}",
                    completed_units=int((current / total) * 100),
                    status="running",
                    metadata={"message": message, "current": current, "total": total}
                )
                
                # Also log for debugging
                logger.debug(f"Analysis progress: {message} ({current}/{total})")
            
            self._analyzer.set_progress_callback(progress_callback)
    
    def cleanup(self, context):
        """Clean up resources after execution."""
        logger.debug("Starting cleanup")
        
        # Clean up analyzer
        if self._analyzer:
            if hasattr(self._analyzer, 'cleanup'):
                try:
                    self._analyzer.cleanup()
                except Exception as e:
                    logger.warning(f"Analyzer cleanup failed: {e}")
            self._analyzer = None
        
        # Remove large data from context
        if context.config.get('analysis.memory.cleanup_after_stage', True):
            keys_to_remove = [
                f'{self.analysis_method}_results',
                'merged_dataset',
                'spatial_coordinates',
                'exported_csv_path'  # If loaded from CSV
            ]
            
            removed_count = 0
            for key in keys_to_remove:
                if key in context.shared_data:
                    del context.shared_data[key]
                    removed_count += 1
            
            if removed_count > 0:
                logger.debug(f"Removed {removed_count} items from context")
        
        # Force garbage collection
        collected = gc.collect()
        
        logger.info(
            f"Cleanup completed",
            extra={
                'stage': self.name,
                'gc_collected': collected
            }
        )
    
    
    def _get_analysis_params(self, context) -> Dict[str, Any]:
        """Get analysis-specific parameters."""
        # Get base config
        base_config = context.config.get(f'{self.analysis_method}_analysis', {})
        
        if self.analysis_method == 'som':
            return {
                'grid_size': base_config.get('default_grid_size', [8, 8]),
                'iterations': base_config.get('iterations', 1000),
                'sigma': base_config.get('sigma', 1.5),
                'learning_rate': base_config.get('learning_rate', 0.5),
                'neighborhood_function': base_config.get('neighborhood_function', 'gaussian'),
                'random_seed': base_config.get('random_seed', 42)
            }
        
        elif self.analysis_method == 'maxp_regions':
            return {
                'n_regions': base_config.get('n_regions', 10),
                'min_region_size': base_config.get('min_region_size', 5),
                'method': base_config.get('method', 'ward'),
                'spatial_weights': base_config.get('spatial_weights', 'queen'),
                'random_seed': base_config.get('random_seed', 42)
            }
        
        elif self.analysis_method == 'gwpca':
            return {
                'n_components': base_config.get('n_components', 3),
                'bandwidth': base_config.get('bandwidth', 'adaptive'),
                'kernel': base_config.get('kernel', 'gaussian'),
                'adaptive_bw': base_config.get('adaptive_bw', 50)  # number of nearest neighbors
            }
        
        return {}
    
    def _extract_metrics(self, results: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract method-specific metrics from results."""
        from src.pipelines.unified_resampling.pipeline_orchestrator import clean_nan_for_json
        
        metrics = {
            'analysis_method': self.analysis_method,
            'parameters': params
        }
        
        if self.analysis_method == 'som':
            if hasattr(results, 'statistics'):
                metrics['analysis_stats'] = clean_nan_for_json(results.statistics)
            if hasattr(results, 'metadata'):
                metrics.update({
                    'n_samples': results.metadata.get('n_samples', 0),
                    'n_features': results.metadata.get('n_features', 0)
                })
        
        elif self.analysis_method == 'maxp_regions':
            if hasattr(results, 'statistics'):
                metrics['region_stats'] = clean_nan_for_json({
                    'n_regions': results.statistics.get('n_regions'),
                    'region_sizes': results.statistics.get('region_sizes'),
                    'objective_value': results.statistics.get('objective_value')
                })
        
        elif self.analysis_method == 'gwpca':
            if hasattr(results, 'statistics'):
                metrics['gwpca_stats'] = clean_nan_for_json({
                    'explained_variance': results.statistics.get('explained_variance'),
                    'bandwidth_used': results.statistics.get('bandwidth_used'),
                    'local_r2': results.statistics.get('local_r2_stats')
                })
        
        return metrics