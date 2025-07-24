# src/pipelines/stages/resample_stage.py
"""Resampling stage."""

from typing import List, Tuple
import logging

from .base_stage import PipelineStage, StageResult
from src.processors.data_preparation.resampling_processor import ResamplingProcessor

logger = logging.getLogger(__name__)


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
        
        try:
            # Get loaded datasets from previous stage
            loaded_datasets = context.get('loaded_datasets', [])
            if not loaded_datasets:
                return StageResult(
                    success=False,
                    data={},
                    metrics={'datasets_resampled': 0},
                    warnings=['No datasets available for resampling']
                )
            
            # Initialize resampling processor
            processor = ResamplingProcessor(context.config, context.db)
            
            resampled_datasets = []
            metrics = {
                'total_datasets': len(loaded_datasets),
                'resampled_successfully': 0,
                'skipped_existing': 0,
                'total_pixels_processed': 0
            }
            
            for dataset_info in loaded_datasets:
                try:
                    dataset_config = dataset_info['config']
                    
                    # Check if already resampled
                    existing = processor.get_resampled_dataset(dataset_config['name'])
                    target_resolution = context.config.get('resampling.target_resolution')
                    
                    if existing and existing.target_resolution == target_resolution:
                        logger.info(f"Using existing resampled dataset: {dataset_config['name']}")
                        resampled_datasets.append(existing)
                        metrics['skipped_existing'] += 1
                    else:
                        # Resample dataset
                        logger.info(f"Resampling dataset: {dataset_config['name']}")
                        resampled_info = processor.resample_dataset(dataset_config)
                        resampled_datasets.append(resampled_info)
                        metrics['resampled_successfully'] += 1
                        metrics['total_pixels_processed'] += resampled_info.shape[0] * resampled_info.shape[1]
                    
                except Exception as e:
                    logger.error(f"Failed to resample {dataset_config.get('name')}: {e}")
                    continue
            
            # Store results in context
            context.set('resampled_datasets', resampled_datasets)
            
            return StageResult(
                success=True,
                data={'resampled_datasets': resampled_datasets},
                metrics=metrics,
                warnings=[] if metrics['resampled_successfully'] + metrics['skipped_existing'] == metrics['total_datasets']
                        else [f"Some datasets failed to resample"]
            )
            
        except Exception as e:
            logger.error(f"Resample stage failed: {e}")
            raise