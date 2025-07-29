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
            
            try:
                ml_ready_path = merger.create_ml_ready_parquet(
                    dataset_dicts,
                    context.output_dir
                )
                
                # Get validation results
                validation_results = merger.get_validation_results()
                
                # Process validation results
                total_errors = sum(v['result'].error_count for v in validation_results)
                total_warnings = sum(v['result'].warning_count for v in validation_results)
                failed_validations = sum(1 for v in validation_results if not v['result'].is_valid)
                
                # Log validation summary
                logger.info(f"Merge validation results: {len(validation_results)} checks, "
                           f"{failed_validations} failures, {total_errors} errors, {total_warnings} warnings")
                
                # Create warnings list for validation issues
                warnings = []
                if total_warnings > 0:
                    warnings.append(f"Merge process generated {total_warnings} validation warnings")
                if failed_validations > 0:
                    warnings.append(f"{failed_validations} validation checks failed during merge")
                
                # Update context with results
                context.set('ml_ready_path', str(ml_ready_path))
                context.set('merge_validation_results', validation_results)
                
                # Return success metrics with validation info
                return StageResult(
                    success=True,
                    data={
                        'ml_ready_path': str(ml_ready_path),
                        'file_size_mb': ml_ready_path.stat().st_size / (1024**2)
                    },
                    metrics={
                        'datasets_merged': len(resampled_datasets),
                        'output_format': 'parquet',
                        'validation_checks': len(validation_results),
                        'validation_errors': total_errors,
                        'validation_warnings': total_warnings,
                        'validation_failures': failed_validations
                    },
                    warnings=warnings
                )
                
            except ValueError as e:
                # Handle validation errors specifically
                error_msg = str(e)
                if "validation" in error_msg.lower():
                    logger.error(f"Merge failed due to validation error: {error_msg}")
                    
                    # Get partial validation results if available
                    validation_results = merger.get_validation_results()
                    
                    return StageResult(
                        success=False,
                        error=f"Validation error during merge: {error_msg}",
                        data={'validation_results': validation_results}
                    )
                else:
                    # Re-raise non-validation errors
                    raise
            
        except Exception as e:
            logger.error(f"Merge stage failed: {e}")
            return StageResult(
                success=False,
                error=str(e)
            )