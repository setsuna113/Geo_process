# src/pipelines/stages/merge_stage.py
"""Dataset merging stage - orchestration only."""

from typing import List, Tuple, Dict, Any
from pathlib import Path
from .base_stage import PipelineStage, StageResult
from src.processors.data_preparation.coordinate_merger import CoordinateMerger
from src.processors.data_preparation.raster_alignment import RasterAligner
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_stage

logger = get_logger(__name__)


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
        errors = []
        
        # Basic validation - detailed config validation happens in execute()
        # when we have access to the context
        
        return len(errors) == 0, errors
    
    @log_stage("merge")
    def execute(self, context) -> StageResult:
        """Orchestrate merge process."""
        logger.info("Starting merge stage orchestration")
        
        # Validate configuration now that we have context
        if context.config.get('merge.enable_streaming', False):
            # Verify export formats are compatible
            export_formats = context.config.get('export.formats', ['csv'])
            if isinstance(export_formats, str):
                export_formats = [export_formats]
            
            # Currently streaming only supports CSV
            unsupported = [fmt for fmt in export_formats if fmt != 'csv']
            if unsupported:
                return StageResult(
                    success=False,
                    data={'error': f"Streaming export only supports CSV format, but found: {unsupported}"},
                    metrics={},
                    warnings=[]
                )
        
        # Validate chunk size
        chunk_size = context.config.get('merge.streaming_chunk_size', 5000)
        if chunk_size < 100 or chunk_size > 1000000:
            return StageResult(
                success=False,
                data={'error': f"Invalid streaming chunk size: {chunk_size} (must be between 100 and 1000000)"},
                metrics={},
                warnings=[]
            )
        
        # Setup memory pressure callbacks if available
        if hasattr(context, 'memory_monitor') and context.memory_monitor:
            self._setup_memory_callbacks(context)
        
        try:
            # Get resampled datasets from context
            resampled_datasets = context.get('resampled_datasets', [])
            
            if len(resampled_datasets) < 2:
                return StageResult(
                    success=False,
                    data={'error': 'Insufficient datasets'},
                    metrics={},
                    warnings=['Need at least 2 datasets for merging']
                )
            
            # Convert ResampledDatasetInfo objects to dict format for the merger
            dataset_dicts = []
            for info in resampled_datasets:
                # Detect storage type from metadata
                if info.metadata.get('memory_aware', False):
                    # New windowed storage format
                    table_name = info.metadata.get('storage_table')
                    if not table_name:
                        logger.warning(f"No storage_table found for memory-aware dataset {info.name}")
                        table_name = f"windowed_{info.name.replace('-', '_')}"
                else:
                    # Legacy naming
                    table_name = f"passthrough_{info.name.replace('-', '_')}" if info.metadata.get('passthrough', False) else f"resampled_{info.name.replace('-', '_')}"
                
                dataset_dict = {
                    'name': info.name,
                    'table_name': table_name,
                    'source_path': str(info.source_path),
                    'bounds': list(info.bounds),  # Ensure it's a list
                    'resolution': info.target_resolution,
                    'passthrough': info.metadata.get('passthrough', False),
                    'memory_aware': info.metadata.get('memory_aware', False)
                }
                dataset_dicts.append(dataset_dict)
            
            # Check alignment using RasterAligner
            logger.info("Checking dataset alignment...")
            aligner = RasterAligner()
            alignment_report = aligner.calculate_grid_shifts(resampled_datasets)
            
            # Log alignment issues
            alignment_metrics = {
                'datasets_requiring_alignment': 0,
                'max_shift_degrees': 0.0
            }
            
            for alignment in alignment_report:
                if alignment.requires_shift:
                    alignment_metrics['datasets_requiring_alignment'] += 1
                    max_shift = max(abs(alignment.x_shift), abs(alignment.y_shift))
                    alignment_metrics['max_shift_degrees'] = max(
                        alignment_metrics['max_shift_degrees'], 
                        max_shift
                    )
                    logger.info(f"Dataset {alignment.aligned_dataset} requires shift: "
                              f"x={alignment.x_shift:.6f}, y={alignment.y_shift:.6f} degrees")
            
            # Determine chunk size based on config and dataset sizes
            chunk_size = None
            if context.config.get('merge.enable_chunked_processing', False):
                chunk_size = context.config.get('merge.chunk_size', 5000)
                logger.info(f"Using chunked processing with chunk_size={chunk_size}")
            
            # Check if streaming mode is enabled
            enable_streaming = context.config.get('merge.enable_streaming', False)
            export_formats = context.config.get('export.formats', ['csv'])
            
            # Delegate all work to processor
            merger = CoordinateMerger(context.config, context.db)
            
            try:
                # If streaming mode is enabled and we're only exporting to CSV, skip in-memory merge
                if enable_streaming and export_formats == ['csv']:
                    logger.info("Streaming mode enabled - configuring for memory-efficient export")
                    
                    # Store configuration for ExportStage
                    context.set('merge_mode', 'streaming')
                    context.set('merge_config', {
                        'dataset_dicts': dataset_dicts,
                        'chunk_size': chunk_size or context.config.get('merge.streaming_chunk_size', 5000),
                        'merger': merger  # Store merger instance for use in ExportStage
                    })
                    
                    # We don't create the merged dataset, just validate the inputs
                    for dataset_dict in dataset_dicts:
                        bounds = dataset_dict.get('bounds')
                        if not bounds:
                            raise ValueError(f"Dataset {dataset_dict['name']} missing bounds")
                    
                    # Return success without creating merged dataset
                    return StageResult(
                        success=True,
                        data={
                            'mode': 'streaming',
                            'datasets': len(dataset_dicts),
                            'chunk_size': chunk_size or context.config.get('merge.streaming_chunk_size', 5000)
                        },
                        metrics={
                            'datasets_configured': len(resampled_datasets),
                            'streaming_enabled': True,
                            'datasets_requiring_alignment': alignment_metrics['datasets_requiring_alignment'],
                            'max_alignment_shift_degrees': alignment_metrics['max_shift_degrees']
                        }
                    )
                
                # Otherwise, create merged dataset in memory as before
                merged_dataset = merger.create_merged_dataset(
                    dataset_dicts,
                    chunk_size=chunk_size,
                    return_as='xarray',  # Return as xarray for compatibility with ExportStage
                    context=context
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
                
                # Update context with merged dataset for ExportStage
                context.set('merged_dataset', merged_dataset)
                context.set('merge_validation_results', validation_results)
                
                # Calculate dataset size estimate
                estimated_size_mb = (merged_dataset.nbytes if hasattr(merged_dataset, 'nbytes') else 0) / (1024**2)
                
                # Return success metrics with validation info
                return StageResult(
                    success=True,
                    data={
                        'dataset_shape': dict(merged_dataset.dims),
                        'variables': list(merged_dataset.data_vars),
                        'estimated_size_mb': estimated_size_mb,
                        'alignment_report': alignment_report
                    },
                    metrics={
                        'datasets_merged': len(resampled_datasets),
                        'output_format': 'xarray',  # Now returning xarray, not file
                        'validation_checks': len(validation_results),
                        'validation_errors': total_errors,
                        'validation_warnings': total_warnings,
                        'validation_failures': failed_validations,
                        'datasets_requiring_alignment': alignment_metrics['datasets_requiring_alignment'],
                        'max_alignment_shift_degrees': alignment_metrics['max_shift_degrees'],
                        'chunked_processing': chunk_size is not None
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
                        data={
                            'error': f"Validation error during merge: {error_msg}",
                            'validation_results': validation_results
                        },
                        metrics={},
                        warnings=[f"Validation error: {error_msg}"]
                    )
                else:
                    # Re-raise non-validation errors
                    raise
            
        except Exception as e:
            logger.error(f"Merge stage failed: {e}")
            return StageResult(
                success=False,
                data={'error': str(e)},
                metrics={},
                warnings=[f"Merge stage failed: {str(e)}"]
            )
    
    def _setup_memory_callbacks(self, context):
        """Setup memory pressure callbacks for adaptive behavior."""
        memory_monitor = context.memory_monitor
        
        # Warning callback - reduce chunk sizes
        def on_memory_warning(usage):
            logger.warning(f"Memory pressure warning: {usage:.1f}% used")
            # Update config to use smaller chunks
            current_chunk = context.config.get('merge.streaming_chunk_size', 5000)
            new_chunk = max(1000, current_chunk // 2)
            context.config.set('merge.streaming_chunk_size', new_chunk)
            logger.info(f"Reduced streaming chunk size to {new_chunk}")
        
        # Critical callback - switch to streaming mode
        def on_memory_critical(usage):
            logger.critical(f"Memory pressure critical: {usage:.1f}% used")
            # Force streaming mode
            context.config.set('merge.enable_streaming', True)
            logger.info("Switched to streaming mode due to memory pressure")
        
        memory_monitor.register_warning_callback(on_memory_warning)
        memory_monitor.register_critical_callback(on_memory_critical)