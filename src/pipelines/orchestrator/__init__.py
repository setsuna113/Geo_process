# src/pipelines/orchestrator/__init__.py
"""Modular pipeline orchestrator with focused responsibilities."""

import logging
from typing import Dict, Any, List, Optional, Type, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

from src.config.config import Config
from src.database.connection import DatabaseManager
from ..stages.base_stage import PipelineStage, StageStatus, StageResult
from src.core.signal_handler import SignalHandler, create_signal_handler

from .stage_manager import StageManager
from .monitor_manager import MonitorManager

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class PipelineContext:
    """Shared context for pipeline execution."""
    config: Config
    db: DatabaseManager
    experiment_id: str
    checkpoint_dir: Path
    output_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    memory_monitor: Optional[Any] = None  # MemoryMonitor instance
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from shared data."""
        return self.shared_data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in shared data."""
        self.shared_data[key] = value
    
    def update_metadata(self, **kwargs):
        """Update metadata."""
        self.metadata.update(kwargs)


class PipelineOrchestrator:
    """
    Modular pipeline orchestrator with focused responsibilities.
    
    This replaces the monolithic orchestrator with a composition of focused managers:
    - StageManager: Handles stage registration, validation, and execution
    - MonitorManager: Handles monitoring and metrics collection
    - RecoveryManager: Handles failure detection and recovery (TODO)
    """
    
    def __init__(self, config: Config, db_connection: DatabaseManager, 
                 signal_handler: Optional[SignalHandler] = None):
        self.config = config
        self.db = db_connection
        
        # Pipeline state
        self.status = PipelineStatus.INITIALIZING
        self.context: Optional[PipelineContext] = None
        
        # Focused managers - composition over inheritance
        self.stage_manager = StageManager()
        self.monitor_manager = MonitorManager(config)
        # TODO: Add recovery_manager when created
        
        # Signal handling for pause/resume - support dependency injection
        self.signal_handler = signal_handler or create_signal_handler()
        self.signal_handler.register_pause_callback(self.pause_pipeline)
        self.signal_handler.register_resume_callback(self.resume_pipeline)
        
        # Execution control
        self._is_paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially
        
        logger.info("Modular pipeline orchestrator initialized")
        
    def register_stage(self, stage: PipelineStage):
        """Register a pipeline stage - delegates to stage manager."""
        self.stage_manager.register_stage(stage)
        
    def configure_pipeline(self, stages: List[Type[PipelineStage]]):
        """Configure pipeline with stages - delegates to stage manager.""" 
        self.stage_manager.configure_pipeline(stages)
        
    def validate_pipeline(self) -> Tuple[bool, List[str]]:
        """Validate pipeline configuration - delegates to stage manager."""
        return self.stage_manager.validate_pipeline()
        
    def run_pipeline(self, experiment_name: str, output_dir: Path, 
                     checkpoint_dir: Optional[Path] = None, 
                     resume: bool = False) -> Dict[str, Any]:
        """
        Execute the complete pipeline with monitoring and recovery.
        
        This is the main orchestration method that coordinates all managers.
        """
        logger.info(f"Starting pipeline execution: {experiment_name}")
        
        try:
            # Initialize pipeline context
            self.context = PipelineContext(
                config=self.config,
                db=self.db,
                experiment_id=experiment_name,
                checkpoint_dir=checkpoint_dir or Path.cwd() / "checkpoint_outputs",
                output_dir=output_dir
            )
            
            # Start monitoring first to get memory monitor
            self.monitor_manager.start_monitoring()
            
            # Add memory monitor to context
            self.context.memory_monitor = self.monitor_manager.memory_monitor
            
            # Inject context into stage manager
            self.stage_manager.context = self.context
            
            # Validate pipeline before execution
            is_valid, errors = self.validate_pipeline()
            if not is_valid:
                raise RuntimeError(f"Pipeline validation failed: {errors}")
            
            try:
                # Execute pipeline
                self.status = PipelineStatus.RUNNING
                result = self._execute_pipeline()
                
                self.status = PipelineStatus.COMPLETED
                logger.info(f"Pipeline {experiment_name} completed successfully")
                
                return result
                
            finally:
                # Always stop monitoring
                self.monitor_manager.stop_monitoring()
                
        except Exception as e:
            self.status = PipelineStatus.FAILED
            logger.error(f"Pipeline {experiment_name} failed: {e}")
            
            # TODO: Attempt recovery if recovery manager is available
            # if hasattr(self, 'recovery_manager'):
            #     recovery_result = self.recovery_manager.attempt_recovery()
            
            raise
            
    def _execute_pipeline(self) -> Dict[str, Any]:
        """Execute pipeline stages in proper order."""
        results = {}
        
        try:
            # Get execution order from stage manager
            ordered_stages = self.stage_manager.get_execution_order()
            
            logger.info(f"Executing {len(ordered_stages)} stages in order")
            
            for stage in ordered_stages:
                # Check for pause
                self._pause_event.wait()
                
                if self.signal_handler.is_shutdown_requested():
                    logger.info("Shutdown requested - stopping pipeline")
                    break
                    
                # Execute stage with monitoring
                with self.monitor_manager.memory_monitoring_context(stage):
                    # Set up progress callback
                    def progress_callback(progress_info):
                        self.monitor_manager.handle_stage_progress(stage.stage_name, progress_info)
                        
                    # Execute the stage
                    result = self.stage_manager.execute_stage(stage, progress_callback=progress_callback)
                    results[stage.stage_name] = result
                    
                    if not result.success:
                        logger.error(f"Stage {stage.stage_name} failed: {result.error_message}")
                        # TODO: Use recovery manager to handle failure
                        raise RuntimeError(f"Stage {stage.stage_name} failed: {result.error_message}")
                        
            # Collect final metrics
            final_metrics = self.monitor_manager.get_current_metrics()
            
            return {
                'status': 'completed',
                'stage_results': results,
                'metrics': final_metrics,
                'experiment_id': self.context.experiment_id if self.context else None
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
            
    def pause_pipeline(self):
        """Pause pipeline execution."""
        if not self._is_paused:
            self._is_paused = True
            self._pause_event.clear()
            self.status = PipelineStatus.PAUSED
            logger.info("Pipeline paused")
            
    def resume_pipeline(self):
        """Resume pipeline execution."""
        if self._is_paused:
            self._is_paused = False
            self._pause_event.set()
            self.status = PipelineStatus.RUNNING
            logger.info("Pipeline resumed")
            
    def get_current_stage(self) -> Optional[PipelineStage]:
        """Get currently executing stage."""
        if self.monitor_manager.state.current_stage:
            return self.stage_manager.get_stage_by_name(self.monitor_manager.state.current_stage)
        return None
        
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        current_metrics = self.monitor_manager.get_current_metrics()
        completed_stages = self.stage_manager.get_completed_stages()
        
        return {
            'status': self.status.value,
            'is_paused': self._is_paused,
            'current_stage': self.monitor_manager.state.current_stage,
            'completed_stages': completed_stages,
            'total_stages': len(self.stage_manager.stages),
            'metrics': current_metrics,
            'experiment_id': self.context.experiment_id if self.context else None
        }
        
    def reset_pipeline(self):
        """Reset pipeline to initial state."""
        self.status = PipelineStatus.INITIALIZING
        self._is_paused = False
        self._pause_event.set()
        
        # Reset managers
        self.stage_manager.reset_pipeline()
        self.monitor_manager.reset_metrics()
        
        self.context = None
        logger.info("Pipeline reset to initial state")


# Import threading at module level to avoid issues
import threading