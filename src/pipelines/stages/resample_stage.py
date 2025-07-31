# src/pipelines/stages/resample_stage.py
"""Resampling stage."""

from typing import List, Tuple
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_stage

logger = get_logger(__name__)

from .base_stage import PipelineStage, StageResult
logger.debug("🔍 About to import ResamplingProcessor")

# Configure GDAL to use exceptions
from osgeo import gdal
gdal.UseExceptions()

from src.processors.data_preparation.resampling_processor import ResamplingProcessor
logger.debug("🔍 ResamplingProcessor imported successfully")

logger.debug("🔍 resample_stage.py module loaded successfully")


class ResampleStage(PipelineStage):
    """Stage for resampling datasets to target resolution."""
    
    @property
    def name(self) -> str:
        return "resample"
    
    @property
    def dependencies(self) -> List[str]:
        return ["data_load"]
    
    @property
    def memory_requirements(self) -> float:
        # Estimate based on target resolution
        return 8.0  # GB
    
    @property
    def estimated_operations(self) -> int:
        # Will be updated based on number of datasets
        return 1000
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate resampling configuration."""
        errors = []
        # Validation done in context
        return True, errors
    
    @log_stage("resample")
    def execute(self, context) -> StageResult:
        """Execute resampling for all datasets."""
        logger.info("Starting resample stage")
        
        try:
            # Get loaded datasets from previous stage
            loaded_datasets = context.get('loaded_datasets', [])
            logger.info(f"📊 DEBUG: Found {len(loaded_datasets)} loaded datasets in context")
            if not loaded_datasets:
                return StageResult(
                    success=False,
                    data={},
                    metrics={'datasets_resampled': 0},
                    warnings=['No datasets available for resampling']
                )
            
            # Initialize resampling processor
            logger.info("🔧 DEBUG: Creating ResamplingProcessor...")
            processor = ResamplingProcessor(context.config, context.db)
            logger.info("✅ DEBUG: ResamplingProcessor created")
            
            resampled_datasets = []
            metrics = {
                'total_datasets': len(loaded_datasets),
                'resampled_successfully': 0,
                'skipped_existing': 0,
                'passthrough_datasets': 0,
                'total_pixels_processed': 0
            }
            
            for idx, dataset_info in enumerate(loaded_datasets):
                try:
                    dataset_config = dataset_info['config']
                    logger.info(f"Processing dataset {idx+1}/{len(loaded_datasets)}: {dataset_config['name']}")
                    
                    # Check if already resampled (including passthrough datasets)
                    logger.debug(f"🔍 Checking for existing resampled dataset: {dataset_config['name']}")
                    existing = processor.get_resampled_dataset(dataset_config['name'])
                    logger.debug(f"✅ get_resampled_dataset returned: {existing is not None}")
                    target_resolution = context.config.get('resampling.target_resolution')
                    tolerance = context.config.get('resampling.resolution_tolerance', 0.001)
                    
                    if existing:
                        # Check if resolution matches target (for both resampled and passthrough)
                        res_matches = abs(existing.target_resolution - target_resolution) <= tolerance
                        
                        if res_matches:
                            is_passthrough = existing.metadata.get('passthrough', False)
                            dataset_type = "passthrough" if is_passthrough else "resampled"
                            logger.info(f"✓ Using existing {dataset_type} dataset: {dataset_config['name']}")
                            logger.info(f"  Resolution: {existing.target_resolution:.6f}° (matches target)")
                            
                            resampled_datasets.append(existing)
                            metrics['skipped_existing'] += 1
                            continue
                        else:
                            logger.info(f"Existing dataset {dataset_config['name']} has different resolution, reprocessing")
                    
                    # Use needs_resampling flag from load stage if available
                    needs_resampling = dataset_info.get('needs_resampling', True)
                    if not needs_resampling:
                        logger.info(f"Dataset {dataset_config['name']} already at target resolution (detected in load stage)")
                    
                    # Resample dataset using memory-aware processing (handles skip-resampling automatically)
                    logger.info(f"Processing dataset: {dataset_config['name']}")
                    
                    # Use memory-efficient processing with context for adaptive behavior
                    resampled_info = processor.resample_dataset_memory_aware(dataset_config, context=context)
                    resampled_datasets.append(resampled_info)
                    
                    # Track whether it was actually resampled or skipped via passthrough
                    if resampled_info.metadata.get('passthrough', False):
                        logger.info(f"✓ Skipped resampling via passthrough for: {dataset_config['name']}")
                        metrics['passthrough_datasets'] += 1
                    else:
                        logger.info(f"✓ Successfully resampled: {dataset_config['name']}")
                        metrics['resampled_successfully'] += 1
                        metrics['total_pixels_processed'] += resampled_info.shape[0] * resampled_info.shape[1]
                    
                except Exception as e:
                    import traceback
                    logger.error(f"Failed to process {dataset_config.get('name')}: {e}")
                    logger.error(f"Stack trace:\n{traceback.format_exc()}")
                    continue
            
            # Store results in context
            context.set('resampled_datasets', resampled_datasets)
            
            # Calculate success
            total_processed = metrics['resampled_successfully'] + metrics['skipped_existing'] + metrics['passthrough_datasets']
            success = len(resampled_datasets) > 0
            warnings = [] if total_processed == metrics['total_datasets'] else [f"Some datasets failed to process"]
            
            return StageResult(
                success=success,
                data={'resampled_datasets': resampled_datasets},
                metrics=metrics,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Resample stage failed: {e}")
            raise