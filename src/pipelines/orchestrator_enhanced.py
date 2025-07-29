"""Enhanced pipeline orchestrator with integrated monitoring and logging."""

import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.schema import DatabaseSchema
from src.infrastructure.logging import (
    get_logger, setup_logging, LoggingContext, 
    log_operation, log_stage
)
from src.infrastructure.monitoring import UnifiedMonitor
from src.pipelines.orchestrator import PipelineOrchestrator
from src.pipelines.enhanced_context import EnhancedPipelineContext
from src.pipelines.stages.base_stage import PipelineStage
from src.core.signal_handler import SignalHandler

logger = get_logger(__name__)


class EnhancedPipelineOrchestrator(PipelineOrchestrator):
    """Enhanced pipeline orchestrator with full monitoring and logging integration.
    
    Features:
    - Structured logging with context propagation
    - Real-time progress tracking with persistence
    - Performance metrics collection
    - Automatic error capture with tracebacks
    - Resource monitoring during execution
    """
    
    def __init__(self, config: Config, db: DatabaseManager, 
                 signal_handler: Optional[SignalHandler] = None):
        """Initialize enhanced orchestrator.
        
        Args:
            config: Configuration object
            db: Database connection manager
            signal_handler: Optional signal handler
        """
        super().__init__(config, db, signal_handler)
        
        # Setup structured logging
        setup_logging(
            config=config,
            db_manager=db,
            console=True,
            database=True,
            log_level=config.get('logging.level', 'INFO')
        )
        
        logger.info("Enhanced pipeline orchestrator initialized")
    
    @log_operation("run_pipeline")
    def run_pipeline(self, 
                    experiment_name: str,
                    description: Optional[str] = None,
                    checkpoint_dir: Optional[Path] = None,
                    output_dir: Optional[Path] = None,
                    resume_from_checkpoint: bool = True,
                    **kwargs) -> Dict[str, Any]:
        """Run pipeline with enhanced monitoring and logging.
        
        Args:
            experiment_name: Name for this pipeline run
            description: Optional description
            checkpoint_dir: Directory for checkpoints
            output_dir: Directory for outputs
            resume_from_checkpoint: Whether to resume from checkpoints
            **kwargs: Additional configuration
            
        Returns:
            Pipeline execution results with metrics
        """
        start_time = time.time()
        
        try:
            # Initialize context (creates experiment in DB)
            self._initialize_context(experiment_name, checkpoint_dir, output_dir, **kwargs)
            
            # Create enhanced context
            enhanced_context = EnhancedPipelineContext(
                config=self.context.config,
                db=self.context.db,
                experiment_id=self.context.experiment_id,
                checkpoint_dir=self.context.checkpoint_dir,
                output_dir=self.context.output_dir,
                metadata=self.context.metadata,
                shared_data=self.context.shared_data,
                quality_metrics=self.context.quality_metrics
            )
            
            # Start monitoring
            enhanced_context.start_monitoring()
            
            # Use the enhanced context
            self.context = enhanced_context
            
            # Run pipeline with logging context
            with enhanced_context.logging_context.pipeline(experiment_name, **kwargs):
                logger.info(
                    f"Starting pipeline execution: {experiment_name}",
                    extra={
                        'context': {
                            'description': description,
                            'checkpoint_dir': str(checkpoint_dir),
                            'output_dir': str(output_dir),
                            'resume': resume_from_checkpoint
                        }
                    }
                )
                
                # Initialize progress tracking
                total_stages = len(self.stages)
                enhanced_context.progress_manager.create_pipeline(
                    pipeline_id=f"pipeline/{experiment_name}",
                    total_phases=total_stages,
                    metadata={'description': description}
                )
                
                # Validate pipeline
                is_valid, errors = self.validate_pipeline()
                if not is_valid:
                    error_msg = f"Pipeline validation failed: {'; '.join(errors)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Execute stages
                completed_stages = 0
                for stage in self.stages:
                    stage_name = stage.__class__.__name__
                    
                    # Skip if completed and resuming
                    if resume_from_checkpoint and self._is_stage_completed(stage_name):
                        logger.info(f"Skipping completed stage: {stage_name}")
                        completed_stages += 1
                        continue
                    
                    # Execute stage with monitoring
                    with enhanced_context.monitor.track_stage(stage_name) as progress:
                        logger.info(f"Executing stage: {stage_name}")
                        
                        try:
                            # Run stage
                            stage_start = time.time()
                            stage.execute(self.context)
                            stage_duration = time.time() - stage_start
                            
                            # Log performance
                            logger.log_performance(
                                f"stage_{stage_name}",
                                stage_duration,
                                status='completed'
                            )
                            
                            # Record metrics
                            enhanced_context.record_metrics(
                                stage_duration_seconds=stage_duration,
                                stage_name=stage_name
                            )
                            
                            completed_stages += 1
                            
                            # Update pipeline progress
                            enhanced_context.progress_manager.update_progress(
                                node_id=f"pipeline/{experiment_name}",
                                completed_units=completed_stages,
                                status='running'
                            )
                            
                        except Exception as e:
                            logger.error(
                                f"Stage {stage_name} failed",
                                exc_info=True,
                                extra={
                                    'context': {
                                        'stage': stage_name,
                                        'error_type': type(e).__name__
                                    }
                                }
                            )
                            
                            # Update experiment status
                            self._update_experiment_status('failed', str(e))
                            raise
                
                # Pipeline completed successfully
                duration = time.time() - start_time
                
                # Finalize and get results
                results = self._finalize_pipeline()
                
                # Add execution metrics
                results['execution_metrics'] = {
                    'total_duration_seconds': duration,
                    'completed_stages': completed_stages,
                    'total_stages': total_stages,
                    'quality_metrics': enhanced_context.get_quality_metrics()
                }
                
                logger.info(
                    f"Pipeline completed successfully",
                    extra={
                        'performance': {
                            'duration_seconds': duration,
                            'stages_completed': completed_stages
                        }
                    }
                )
                
                return results
                
        except Exception as e:
            # Log pipeline failure
            duration = time.time() - start_time
            logger.error(
                f"Pipeline execution failed",
                exc_info=True,
                extra={
                    'performance': {
                        'duration_seconds': duration,
                        'stages_completed': completed_stages if 'completed_stages' in locals() else 0
                    }
                }
            )
            
            # Update status if context exists
            if hasattr(self, 'context') and self.context:
                self._update_experiment_status('failed', str(e))
            
            raise
            
        finally:
            # Stop monitoring
            if hasattr(self, 'context') and isinstance(self.context, EnhancedPipelineContext):
                self.context.stop_monitoring()
    
    def _execute_stage(self, stage: PipelineStage, 
                      stage_registry: Dict[str, PipelineStage]) -> bool:
        """Execute a single stage with enhanced logging context.
        
        Args:
            stage: Stage to execute
            stage_registry: Registry of all stages
            
        Returns:
            True if successful
        """
        # Wrap stage execution with logging context
        stage_name = stage.name
        
        # Special handling for merge stage
        if stage_name == "merge" and isinstance(self.context, EnhancedPipelineContext):
            with self.context.logging_context.operation("merge_datasets"):
                logger.info(f"Executing merge stage with enhanced monitoring")
                result = super()._execute_stage(stage, stage_registry)
                
                # Log merge-specific metrics if available
                if hasattr(self.context, 'shared_data'):
                    merge_validation = self.context.shared_data.get('merge_validation_results')
                    if merge_validation:
                        logger.info(
                            "Merge validation summary",
                            extra={
                                'validation_results': len(merge_validation),
                                'validation_errors': sum(v['result'].error_count for v in merge_validation),
                                'validation_warnings': sum(v['result'].warning_count for v in merge_validation)
                            }
                        )
                    
                    # Track merge metrics in monitoring
                    if hasattr(self.context, 'monitor') and self.context.monitor:
                        self.context.monitor.track_metric(
                            'merge_datasets_count',
                            len(self.context.shared_data.get('resampled_datasets', []))
                        )
                        
                        # Track alignment metrics if available
                        stage_results = self.context.get_stage_results()
                        merge_result = stage_results.get('merge', {})
                        if merge_result and 'metrics' in merge_result:
                            metrics = merge_result['metrics']
                            self.context.monitor.track_metric(
                                'datasets_requiring_alignment',
                                metrics.get('datasets_requiring_alignment', 0)
                            )
                            self.context.monitor.track_metric(
                                'max_alignment_shift_degrees',
                                metrics.get('max_alignment_shift_degrees', 0.0)
                            )
                
                return result
        else:
            # Standard stage execution
            return super()._execute_stage(stage, stage_registry)
    
    def _is_stage_completed(self, stage_name: str) -> bool:
        """Check if stage was previously completed.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            True if stage was completed
        """
        # Check in progress tracking
        if hasattr(self.context, 'progress_manager'):
            node_id = f"pipeline/{self.context.experiment_id}/{stage_name}"
            node = self.context.progress_manager.backend.get_node(
                self.context.experiment_id, node_id
            )
            if node and node['status'] == 'completed':
                return True
        
        # Fallback to checkpoint check
        return super()._is_stage_completed(stage_name)
    
    def _finalize_pipeline(self) -> Dict[str, Any]:
        """Finalize pipeline execution with enhanced metrics."""
        results = super()._finalize_pipeline()
        
        # Add monitoring data if available
        if isinstance(self.context, EnhancedPipelineContext):
            # Get timing data from logging context
            if self.context.logging_context:
                results['stage_timings'] = self.context.logging_context.get_timings()
            
            # Get final progress state
            if self.context.progress_manager:
                results['progress_summary'] = self.context.progress_manager.get_experiment_progress()
        
        return results
    
    def _handle_signal(self, signum: int, frame):
        """Enhanced signal handling with proper logging."""
        signal_name = signal.Signals(signum).name
        
        logger.warning(
            f"Received signal {signal_name}, initiating graceful shutdown",
            extra={
                'context': {
                    'signal': signal_name,
                    'signal_number': signum
                }
            }
        )
        
        # Call parent handler
        super()._handle_signal(signum, frame)