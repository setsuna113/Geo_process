# src/pipelines/stages/load_stage.py
"""Data loading stage."""

from typing import List, Tuple
import logging
from pathlib import Path

from .base_stage import PipelineStage, StageResult
from src.raster_data.catalog import RasterCatalog
from src.config.dataset_utils import DatasetPathResolver

logger = logging.getLogger(__name__)


class DataLoadStage(PipelineStage):
    """Stage for loading and validating input datasets."""
    
    @property
    def name(self) -> str:
        return "data_load"
    
    @property
    def dependencies(self) -> List[str]:
        return []  # First stage, no dependencies
    
    @property
    def memory_requirements(self) -> float:
        return 2.0  # GB - just for metadata, not loading full rasters
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate data loading configuration."""
        errors = []
        
        # This validation will be done using context during execution
        # For now, just return success
        return True, errors
    
    def execute(self, context) -> StageResult:
        """Load and validate datasets."""
        logger.info("Starting data load stage")
        
        try:
            # Get dataset configurations
            datasets_config = context.config.get('datasets.target_datasets', [])
            enabled_datasets = [ds for ds in datasets_config if ds.get('enabled', True)]
            
            if not enabled_datasets:
                return StageResult(
                    success=False,
                    data={},
                    metrics={'datasets_found': 0},
                    warnings=['No enabled datasets found']
                )
            
            # Resolve dataset paths
            resolver = DatasetPathResolver(context.config)
            catalog = RasterCatalog(context.db, context.config)
            
            loaded_datasets = []
            metrics = {
                'total_datasets': len(enabled_datasets),
                'loaded_successfully': 0,
                'total_size_mb': 0
            }
            
            for dataset_config in enabled_datasets:
                try:
                    # Validate and resolve path
                    normalized_config = resolver.validate_dataset_config(dataset_config)
                    dataset_path = Path(normalized_config['resolved_path'])
                    
                    # Register in catalog if needed
                    raster_info = {"name": normalized_config["name"], "path": str(dataset_path)}
                    
                    loaded_datasets.append({
                        'name': normalized_config['name'],
                        'path': str(dataset_path),
                        'config': normalized_config,
                        'raster_info': raster_info
                    })
                    
                    metrics['loaded_successfully'] += 1
                    if dataset_path.exists():
                        metrics['total_size_mb'] += dataset_path.stat().st_size // (1024**2)
                    
                except Exception as e:
                    logger.error(f"Failed to load dataset {dataset_config.get('name')}: {e}")
                    continue
            
            # Store in context for next stages
            context.set('loaded_datasets', loaded_datasets)
            context.set('dataset_catalog', catalog)
            
            return StageResult(
                success=True,
                data={'datasets': loaded_datasets},
                metrics=metrics,
                warnings=[] if metrics['loaded_successfully'] == metrics['total_datasets'] 
                        else [f"Only {metrics['loaded_successfully']}/{metrics['total_datasets']} datasets loaded"]
            )
            
        except Exception as e:
            logger.error(f"Data load stage failed: {e}")
            raise
