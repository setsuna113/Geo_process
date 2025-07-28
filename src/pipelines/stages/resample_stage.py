# src/pipelines/stages/resample_stage.py
"""Resampling stage."""

from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

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
    
    def execute(self, context) -> StageResult:
        """Execute resampling for all datasets."""
        logger.info("Starting resample stage")
        logger.info("🔍 DEBUG: ResampleStage.execute() called - entering try block")
        
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
            
            # Initialize skip controller for intelligent skip decisions
            from .skip_control import StageSkipController
            skip_controller = StageSkipController(context.config, context.db)
            
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
                    
                    # Enhanced skip logic with DB status detection
                    can_skip, skip_reason = skip_controller.can_skip_dataset_processing(dataset_config['name'])
                    
                    if can_skip:
                        # Load metadata from DB instead of reprocessing
                        existing = processor.get_resampled_dataset(dataset_config['name'])
                        if existing:
                            is_passthrough = existing.metadata.get('passthrough', False)
                            dataset_type = "passthrough" if is_passthrough else "resampled"
                            logger.info(f"✓ Skipping {dataset_config['name']} - {skip_reason}")
                            logger.info(f"  Using existing {dataset_type} data from DB")
                            
                            resampled_datasets.append(existing)
                            metrics['skipped_existing'] += 1
                            continue
                    else:
                        # Cannot skip - check if we need to clean up partial data
                        if "error" in skip_reason or "incomplete" in skip_reason:
                            logger.warning(f"DB status {skip_reason}, cleaning and reprocessing")
                            skip_controller.cleanup_partial_data(dataset_config['name'])
                    
                    # Resample dataset (this will handle skip-resampling automatically)
                    logger.info(f"Processing dataset: {dataset_config['name']}")
                    resampled_info = processor.resample_dataset(dataset_config)
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
                    logger.error(f"Failed to process {dataset_config.get('name')}: {e}")
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