# src/pipelines/stages/merge_stage.py
"""Dataset merging stage - orchestration only."""

from typing import List, Tuple
import logging
from pathlib import Path
from .base_stage import PipelineStage, StageResult
from src.processors.data_preparation.coordinate_merger import CoordinateMerger

logger = logging.getLogger(__name__)


class MergeStage(PipelineStage):
    """Orchestrates dataset merging using CoordinateMerger processor."""
    
    @property
    def name(self) -> str:
        return "merge"
    
    @property
    def dependencies(self) -> List[str]:
        return ["resample"]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate merge configuration."""
        return True, []
    
    def execute(self, context) -> StageResult:
        """Orchestrate merge process."""
        logger.info("Starting merge stage orchestration")
        
        try:
            # Get resampled datasets from context
            resampled_datasets = context.get('resampled_datasets', [])
            
            if len(resampled_datasets) < 2:
                return StageResult(
                    success=False,
                    warnings=['Need at least 2 datasets for merging']
                )
            
            # Convert ResampledDatasetInfo objects to dict format for the merger
            dataset_dicts = []
            for info in resampled_datasets:
                dataset_dict = {
                    'name': info.name,
                    'table_name': f"passthrough_{info.name.replace('-', '_')}" if info.metadata.get('passthrough', False) else f"resampled_{info.name.replace('-', '_')}",
                    'source_path': str(info.source_path),
                    'bounds': list(info.bounds),  # Ensure it's a list
                    'resolution': info.target_resolution,
                    'passthrough': info.metadata.get('passthrough', False)
                }
                dataset_dicts.append(dataset_dict)
            
            # Delegate all work to processor
            merger = CoordinateMerger(context.config, context.db)
            ml_ready_path = merger.create_ml_ready_parquet(
                dataset_dicts,
                context.output_dir
            )
            
            # Update context with results
            context.set('ml_ready_path', str(ml_ready_path))
            
            # Return success metrics
            return StageResult(
                success=True,
                data={
                    'ml_ready_path': str(ml_ready_path),
                    'file_size_mb': ml_ready_path.stat().st_size / (1024**2)
                },
                metrics={
                    'datasets_merged': len(resampled_datasets),
                    'output_format': 'parquet'
                }
            )
            
        except Exception as e:
            logger.error(f"Merge stage failed: {e}")
            return StageResult(
                success=False,
                error=str(e)
            )