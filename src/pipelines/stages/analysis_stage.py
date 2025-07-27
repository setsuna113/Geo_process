# src/pipelines/stages/analysis_stage.py
"""Analysis stage for multiple spatial analysis methods."""

from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
import pandas as pd
import xarray as xr

from .base_stage import PipelineStage, StageResult

logger = logging.getLogger(__name__)


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
        supported_methods = {'som', 'maxp_regions', 'gwpca'}
        if self.analysis_method not in supported_methods:
            raise ValueError(
                f"Unsupported analysis method: {self.analysis_method}. "
                f"Supported: {supported_methods}"
            )
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate analysis configuration."""
        errors = []
        
        # Check if analyzer module exists
        try:
            self._get_analyzer_class()
        except ImportError as e:
            errors.append(f"Analyzer not available: {e}")
        
        return len(errors) == 0, errors
    
    def execute(self, context) -> StageResult:
        """Execute spatial analysis."""
        logger.info(f"Starting {self.analysis_method} analysis")
        
        try:
            # Determine data source (CSV or database)
            data_source = context.config.get('analysis.data_source', 'database')
            
            # Load data
            if data_source == 'csv':
                data = self._load_from_csv(context)
            else:
                data = self._load_from_database(context)
            
            if data is None:
                return StageResult(
                    success=False,
                    data={},
                    metrics={},
                    warnings=['No data available for analysis']
                )
            
            # Get analyzer
            analyzer = self._create_analyzer(context)
            
            # Get analysis parameters
            params = self._get_analysis_params(context)
            
            # Run analysis
            logger.info(f"Running {self.analysis_method} with parameters: {params}")
            results = analyzer.analyze(data=data, **params)
            
            # Save results
            output_name = f"{self.analysis_method}_Analysis_{context.experiment_id}"
            saved_path = analyzer.save_results(results, output_name)
            
            # Store results in context
            context.set(f'{self.analysis_method}_results', results)
            context.set(f'{self.analysis_method}_output_path', saved_path)
            
            # Prepare metrics
            metrics = self._extract_metrics(results, params)
            metrics['output_path'] = str(saved_path)
            
            return StageResult(
                success=True,
                data={f'{self.analysis_method}_results_path': str(saved_path)},
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"{self.analysis_method} analysis failed: {e}")
            raise
    
    def _load_from_csv(self, context) -> Optional[Any]:
        """Load data from exported CSV."""
        # Get exported file path from context
        export_path = context.get('exported_csv_path')
        
        if not export_path or not Path(export_path).exists():
            logger.warning("No exported CSV found, falling back to database")
            return self._load_from_database(context)
        
        logger.info(f"Loading data from CSV: {export_path}")
        
        try:
            # Read CSV
            df = pd.read_csv(export_path)
            
            # Convert to format expected by analyzers
            # For spatial analysis, we need coordinates and features
            
            # Extract coordinates
            if 'x' in df.columns and 'y' in df.columns:
                coords = df[['x', 'y']].values
            else:
                logger.error("CSV missing coordinate columns")
                return None
            
            # Extract feature columns (everything except cell_id, x, y)
            feature_cols = [col for col in df.columns 
                           if col not in ['cell_id', 'x', 'y']]
            
            if not feature_cols:
                logger.error("No feature columns found in CSV")
                return None
            
            features = df[feature_cols].values
            
            # Create xarray dataset for compatibility
            data = xr.Dataset({
                'features': xr.DataArray(
                    features,
                    dims=['location', 'feature'],
                    coords={
                        'location': range(len(features)),
                        'feature': feature_cols
                    }
                ),
                'x': xr.DataArray(coords[:, 0] if hasattr(coords, '__getitem__') else [c[0] for c in coords], dims=['location']),
                'y': xr.DataArray(coords[:, 1] if hasattr(coords, '__getitem__') else [c[1] for c in coords], dims=['location'])
            })
            
            # Store coordinates in context for spatial methods
            context.set('spatial_coordinates', coords)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return None
    
    def _load_from_database(self, context) -> Optional[Any]:
        """Load merged dataset from database/context."""
        # Try to get from context first (in-memory)
        merged_dataset = context.get('merged_dataset')
        
        if merged_dataset is not None:
            logger.info("Using in-memory merged dataset")
            
            # Extract coordinates for spatial analysis
            if 'x' in merged_dataset.coords and 'y' in merged_dataset.coords:
                x_vals = merged_dataset.coords['x'].values
                y_vals = merged_dataset.coords['y'].values
                
                # Create coordinate pairs
                import numpy as np
                xx, yy = np.meshgrid(x_vals, y_vals)
                coords = np.column_stack([xx.ravel(), yy.ravel()])
                context.set('spatial_coordinates', coords)
            
            return merged_dataset
        
        # Otherwise load from database
        logger.info("Loading merged dataset from database")
        
        # This would be implemented based on your database schema
        # For now, returning None to indicate not implemented
        logger.warning("Database loading not implemented in analysis stage")
        return None
    
    def _get_analyzer_class(self):
        """Get the appropriate analyzer class."""
        if self.analysis_method == 'som':
            from src.spatial_analysis.som.som_trainer import SOMAnalyzer
            return SOMAnalyzer
        elif self.analysis_method == 'maxp_regions':
            try:
                from src.spatial_analysis.regionalization.maxp_regions import MaxPRegionsAnalyzer
                return MaxPRegionsAnalyzer
            except ImportError:
                raise ImportError(f"MaxP regions analysis not available - missing module")
        elif self.analysis_method == 'gwpca':
            try:
                from src.spatial_analysis.multivariate.gwpca import GWPCAAnalyzer
                return GWPCAAnalyzer
            except ImportError:
                raise ImportError(f"GWPCA analysis not available - missing module")
        else:
            raise ValueError(f"Unknown analysis method: {self.analysis_method}")
    
    def _create_analyzer(self, context):
        """Create analyzer instance."""
        analyzer_class = self._get_analyzer_class()
        # SOMAnalyzer only takes db_connection as optional argument
        if self.analysis_method == 'som':
            return analyzer_class(db_connection=context.db)
        else:
            return analyzer_class(context.config, context.db)
    
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