# src/pipelines/stages/load_stage.py
"""Data loading stage."""

from typing import List, Tuple
from pathlib import Path

from .base_stage import PipelineStage, StageResult
from src.domain.raster.catalog import RasterCatalog
from src.config.dataset_utils import DatasetPathResolver
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_stage

logger = get_logger(__name__)


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
    
    @log_stage("data_load")
    def execute(self, context) -> StageResult:
        """Load and validate datasets."""
        logger.info("Starting data load stage")
        
        try:
            # Get dataset configurations
            logger.debug("📂 Getting dataset configurations from context.config")
            datasets_config = context.config.get('datasets.target_datasets', [])
            logger.debug(f"📊 Found {len(datasets_config)} datasets in config")
            enabled_datasets = [ds for ds in datasets_config if ds.get('enabled', True)]
            
            if not enabled_datasets:
                return StageResult(
                    success=False,
                    data={},
                    metrics={'datasets_found': 0},
                    warnings=['No enabled datasets found']
                )
            
            # Resolve dataset paths
            logger.debug("🔧 Creating DatasetPathResolver")
            resolver = DatasetPathResolver(context.config)
            logger.debug("🔧 Creating RasterCatalog")
            catalog = RasterCatalog(context.db, context.config)
            logger.debug("✅ Resolver and Catalog created")
            
            loaded_datasets = []
            metrics = {
                'total_datasets': len(enabled_datasets),
                'loaded_successfully': 0,
                'total_size_mb': 0
            }
            
            for i, dataset_config in enumerate(enabled_datasets):
                logger.debug(f"📦 Processing dataset {i+1}/{len(enabled_datasets)}: {dataset_config.get('name', 'unknown')}")
                try:
                    # Validate and resolve path
                    logger.debug(f"🔍 Validating dataset config for {dataset_config.get('name')}")
                    normalized_config = resolver.validate_dataset_config(dataset_config)
                    logger.debug(f"✅ Dataset config validated")
                    dataset_path = Path(normalized_config['resolved_path'])
                    
                    # Register in catalog if needed
                    raster_info = {"name": normalized_config["name"], "path": str(dataset_path)}
                    
                    # Check if resampling is needed by comparing resolutions
                    needs_resampling = True  # Default to true
                    source_resolution = None
                    
                    try:
                        # Get raster metadata to check resolution
                        import rasterio
                        with rasterio.open(dataset_path) as src:
                            # Get resolution in degrees (assuming geographic CRS)
                            source_resolution = abs(src.transform[0])  # pixel width in degrees
                            
                            # Get target resolution from config
                            target_resolution = context.config.get('resampling.target_resolution', 0.0416667)
                            tolerance = context.config.get('resampling.resolution_tolerance', 0.001)
                            
                            # Check if resolution matches (within tolerance)
                            if abs(source_resolution - target_resolution) < tolerance:
                                needs_resampling = False
                                logger.info(f"Dataset {normalized_config['name']} resolution {source_resolution:.6f}° matches target {target_resolution:.6f}° (within tolerance {tolerance})")
                            else:
                                logger.info(f"Dataset {normalized_config['name']} resolution {source_resolution:.6f}° differs from target {target_resolution:.6f}°, resampling needed")
                    except Exception as e:
                        logger.warning(f"Could not check resolution for {normalized_config['name']}: {e}")
                    
                    loaded_datasets.append({
                        'name': normalized_config['name'],
                        'path': str(dataset_path),
                        'config': normalized_config,
                        'raster_info': raster_info,
                        'needs_resampling': needs_resampling,
                        'source_resolution': source_resolution
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
