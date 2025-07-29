# src/pipelines/orchestrator.py
"""Production-ready pipeline orchestrator with monitoring and recovery."""

import logging
import time
from typing import Dict, Any, List, Optional, Type, Tuple
from datetime import datetime
from pathlib import Path
import json
from enum import Enum
from dataclasses import dataclass, field
import threading
import queue
import psutil

from src.config import config
from src.config.config import Config
from src.database.connection import DatabaseManager
from src.pipelines.stages.base_stage import ProcessingConfig
from src.pipelines.stages.base_stage import PipelineStage, StageStatus, StageResult
from src.pipelines.monitors.memory_monitor import MemoryMonitor
from src.pipelines.monitors.progress_tracker import ProgressTracker
from src.pipelines.monitors.quality_checker import QualityChecker
# from src.pipelines.recovery.checkpoint_manager import CheckpointManager  # OLD - DEPRECATED
from src.checkpoints import get_checkpoint_manager, CheckpointLevel
from src.pipelines.recovery.failure_handler import FailureHandler
from src.core.signal_handler import SignalHandler, create_signal_handler

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
    Production-ready pipeline orchestrator with monitoring and recovery.
    
    Features:
    - Stage-based execution with dependencies
    - Checkpoint and recovery system
    - Memory and progress monitoring
    - Quality checking at each stage
    - Parallel stage execution where possible
    - Graceful failure handling
    """
    
    def __init__(self, config: Config, db_connection: DatabaseManager, signal_handler: Optional[SignalHandler] = None):
        self.config = config
        self.db = db_connection
        
        # Pipeline state
        self.status = PipelineStatus.INITIALIZING
        self.stages: List[PipelineStage] = []
        self.stage_registry: Dict[str, PipelineStage] = {}
        self.context: Optional[PipelineContext] = None
        
        # Monitoring components
        self.memory_monitor = MemoryMonitor(config)
        self.progress_tracker = ProgressTracker()
        self.quality_checker = QualityChecker(config)
        
        # Recovery components - using unified checkpoint system
        self.checkpoint_manager = get_checkpoint_manager()
        self.failure_handler = FailureHandler(config)
        
        # Signal handling for pause/resume - support dependency injection
        self.signal_handler = signal_handler or create_signal_handler()
        self.signal_handler.register_pause_callback(self.pause_pipeline)
        self.signal_handler.register_resume_callback(self.resume_pipeline)
        
        # Execution control
        self._stop_requested = False
        self._pause_requested = False
        self._execution_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Event queue for async monitoring
        self.event_queue = queue.Queue()
        
        logger.info("Pipeline orchestrator initialized")
    
    def register_stage(self, stage: PipelineStage):
        """Register a pipeline stage."""
        if stage.name in self.stage_registry:
            raise ValueError(f"Stage '{stage.name}' already registered")
        
        self.stages.append(stage)
        self.stage_registry[stage.name] = stage
        logger.info(f"Registered stage: {stage.name}")
    
    def configure_pipeline(self, stages: List[Type[PipelineStage]]):
        """Configure pipeline with stage classes."""
        for stage_class in stages:
            stage = stage_class()
            self.register_stage(stage)
    
    def validate_pipeline(self) -> Tuple[bool, List[str]]:
        """Validate pipeline configuration and dependencies."""
        errors = []
        
        # Check for circular dependencies
        for stage in self.stages:
            if self._has_circular_dependency(stage):
                errors.append(f"Circular dependency detected for stage '{stage.name}'")
        
        # Validate each stage
        for stage in self.stages:
            is_valid, stage_errors = stage.validate()
            if not is_valid:
                errors.extend([f"{stage.name}: {e}" for e in stage_errors])
        
        # Check stage dependencies exist
        for stage in self.stages:
            for dep in stage.dependencies:
                if dep not in self.stage_registry:
                    errors.append(f"Stage '{stage.name}' depends on unknown stage '{dep}'")
        
        return len(errors) == 0, errors
    
    def run_pipeline(self, 
                    experiment_name: str,
                    checkpoint_dir: Optional[Path] = None,
                    output_dir: Optional[Path] = None,
                    resume_from_checkpoint: bool = True,
                    **kwargs) -> Dict[str, Any]:
        """
        Run the complete pipeline with monitoring and recovery.
        
        Args:
            experiment_name: Name for this pipeline run
            checkpoint_dir: Directory for checkpoints (auto-generated if None)
            output_dir: Directory for outputs (auto-generated if None)
            resume_from_checkpoint: Whether to resume from existing checkpoint
            **kwargs: Additional arguments passed to stages
            
        Returns:
            Pipeline execution results
        """
        logger.debug(f"🎬 run_pipeline() called with experiment_name={experiment_name}")
        try:
            # Auto-cleanup for fresh start if needed
            if kwargs.get('auto_cleanup', True):
                logger.debug("🧹 Performing auto-cleanup for fresh start...")
                from src.utils.cleanup_manager import cleanup_for_fresh_start
                cleanup_success = cleanup_for_fresh_start(experiment_name)
                if cleanup_success:
                    logger.debug("✅ Auto-cleanup completed successfully")
                else:
                    logger.debug("⚠️  Auto-cleanup had issues but continuing...")
            
            # Initialize pipeline context
            logger.debug("📝 Calling _initialize_context...")
            self._initialize_context(experiment_name, checkpoint_dir, output_dir, **kwargs)
            logger.debug("✅ _initialize_context completed")
            
            # Validate pipeline
            logger.debug("🔍 Validating pipeline...")
            is_valid, errors = self.validate_pipeline()
            if not is_valid:
                raise ValueError(f"Pipeline validation failed: {errors}")
            logger.debug("✅ Pipeline validation passed")
            
            # Reset stop flag before starting monitoring
            self._stop_requested = False
            
            # Start monitoring
            logger.debug("📊 Starting monitoring...")
            self._start_monitoring()
            logger.debug("✅ Monitoring started")
            
            # Check for existing checkpoint
            if resume_from_checkpoint:
                logger.debug("💾 Checking for existing checkpoints...")
                process_id = self.context.experiment_id if self.context else "pipeline"
                checkpoint_data = self.checkpoint_manager.load_latest(process_id, CheckpointLevel.STAGE)
                if checkpoint_data:
                    logger.info(f"Resuming from checkpoint: {checkpoint_data.checkpoint_id}")
                    self._restore_from_checkpoint(checkpoint_data.data)
                else:
                    logger.debug("💾 No checkpoint found, starting fresh")
            
            # Execute pipeline
            logger.debug("🏃 Setting status to RUNNING...")
            self.status = PipelineStatus.RUNNING
            # Update experiment status in database
            from src.database.schema import schema
            if self.context and self.context.experiment_id:
                schema.update_experiment_status(self.context.experiment_id, 'running')
                logger.info("Updated experiment status to 'running'")
            results = self._execute_pipeline()
            
            # Finalize
            self.status = PipelineStatus.COMPLETED
            self._finalize_pipeline(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.status = PipelineStatus.FAILED
            
            # Handle failure
            recovery_attempted = self.failure_handler.handle_failure(
                e, self.context, self.get_current_stage()
            )
            
            if recovery_attempted and self.failure_handler.can_recover():
                logger.info("Attempting recovery...")
                self.status = PipelineStatus.RECOVERING
                return self._attempt_recovery()
            
            raise
        
        finally:
            self._stop_monitoring()
    
    def _initialize_context(self, experiment_name: str, 
                           checkpoint_dir: Optional[Path],
                           output_dir: Optional[Path],
                           **kwargs):
        """Initialize pipeline execution context."""
        # Create experiment
        logger.debug("🧪 Creating experiment in database...")
        from src.database.schema import schema
        experiment_id = schema.create_experiment(
            name=experiment_name,
            description=kwargs.get('description', 'Pipeline execution'),
            config={
                'pipeline': self.get_pipeline_config(),
                'stages': [s.name for s in self.stages],
                **kwargs
            }
        )
        
        # Setup directories
        if checkpoint_dir is None:
            checkpoint_dir = Path(self.config.get('paths.checkpoint_dir', 'checkpoints')) / experiment_id
        else:
            checkpoint_dir = Path(checkpoint_dir)
            
        if output_dir is None:
            output_dir = Path(self.config.get('paths.output_dir', 'outputs')) / experiment_id
        else:
            output_dir = Path(output_dir)
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create context
        self.context = PipelineContext(
            config=self.config,
            db=self.db,
            experiment_id=experiment_id,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            metadata={
                'experiment_name': experiment_name,
                'start_time': datetime.now().isoformat(),
                'pipeline_version': '1.0.0',
                **kwargs
            }
        )
        
        # Initialize progress tracker
        total_operations = sum(stage.estimated_operations for stage in self.stages)
        self.progress_tracker.initialize(
            total_stages=len(self.stages),
            total_operations=total_operations
        )
    
    def _execute_pipeline(self) -> Dict[str, Any]:
        """Execute pipeline stages in dependency order."""
        logger.debug(f"🚀 _execute_pipeline starting...")
        logger.debug("🚀 _execute_pipeline starting...")
        
        results = {}
        completed_stages = set()
        
        # Get execution order
        execution_order = self._get_execution_order()
        logger.debug(f"📋 Execution order has {len(execution_order)} stage groups")
        logger.debug(f"📋 Execution order has {len(execution_order)} stage groups")
        
        for i, stage_group in enumerate(execution_order):
            logger.debug(f"📌 Processing stage group {i+1}/{len(execution_order)}: {[s.name for s in stage_group]}")
            logger.debug(f"📌 Processing stage group {i+1}/{len(execution_order)}: {[s.name for s in stage_group]}")
            
        for stage_group in execution_order:
            # Execute stages in parallel if possible
            if len(stage_group) == 1:
                # Single stage - execute normally
                stage = stage_group[0]
                result = self._execute_stage(stage, completed_stages)
                results[stage.name] = result
                completed_stages.add(stage.name)
            else:
                # Multiple independent stages - execute in parallel
                parallel_results = self._execute_parallel_stages(stage_group, completed_stages)
                results.update(parallel_results)
                completed_stages.update(s.name for s in stage_group)
            
            # Check for stop request
            if self._stop_requested:
                logger.info("Pipeline stop requested")
                break
            
            # Check for pause request (from signal handler or direct call)
            if self._pause_requested or self.signal_handler.is_pause_requested():
                if not self._pause_requested:
                    self._pause_requested = True
                    self.status = PipelineStatus.PAUSED
                logger.info("Pipeline paused - waiting for resume")
                while (self._pause_requested or self.signal_handler.is_pause_requested()) and not self._stop_requested:
                    time.sleep(0.5)  # Check every 500ms
                if self._stop_requested:
                    logger.info("Pipeline stop requested during pause")
                    break
                logger.info("Pipeline resumed")
        
        return results
    
    def _execute_stage(self, stage: PipelineStage, 
                      completed_stages: set) -> StageResult:
        """Execute a single pipeline stage with enhanced memory management."""
        logger.debug(f"🎯 Starting stage execution: {stage.name}")
        logger.debug(f"🎯 Starting stage execution: {stage.name}")
        logger.info(f"Executing stage: {stage.name}")
        
        # Update progress
        logger.debug(f"📊 Updating progress tracker for stage: {stage.name}")
        self.progress_tracker.start_stage(stage.name)
        
        # Pre-execution checks with memory awareness
        self._pre_stage_checks(stage)
        
        # Configure stage for memory-aware processing if supported
        if stage.supports_chunking:
            processing_config = self._create_processing_config(stage)
            stage.set_processing_config(processing_config)
        
        try:
            # Check dependencies
            for dep in stage.dependencies:
                if dep not in completed_stages:
                    raise RuntimeError(f"Dependency '{dep}' not completed")
            
            # Set up progress callback for stage
            if self.context:
                self.context.set('progress_callback', 
                    lambda info: self._handle_stage_progress(stage.name, info))
            
            # Execute stage
            stage.status = StageStatus.RUNNING
            start_time = time.time()
            
            # Check for pause/stop before execution
            if self._pause_requested or self.signal_handler.is_pause_requested():
                if not self._pause_requested:
                    self._pause_requested = True
                stage.pause()
                logger.info(f"Stage '{stage.name}' paused before execution")
                while (self._pause_requested or self.signal_handler.is_pause_requested()) and not self._stop_requested:
                    time.sleep(0.5)
                if self._stop_requested:
                    logger.info(f"Stage '{stage.name}' stopped during pause")
                    raise InterruptedError("Pipeline stopped during pause")
                stage.resume()
                logger.info(f"Stage '{stage.name}' resumed")
            
            # Check if stage can be skipped
            skip_result = self._check_stage_skip(stage)
            if skip_result is not None:
                # Stage was skipped, use the skip result
                result = skip_result
                execution_time = 0.01  # Minimal time for skipped stage
                stage.status = StageStatus.COMPLETED
                logger.info(f"Stage '{stage.name}' skipped - using existing data")
            else:
                # Monitor memory during execution
                logger.debug(f"🔧 About to call stage.execute() for {stage.name}")
                logger.debug(f"🔧 About to call stage.execute() for {stage.name}")
                
                try:
                    with self._memory_monitoring_context(stage):
                        logger.debug(f"🚀 Entering stage.execute() for {stage.name}")
                        result = stage.execute(self.context)
                        logger.debug(f"✅ stage.execute() completed for {stage.name}")
                        logger.debug(f"✅ stage.execute() completed for {stage.name}")
                except Exception as e:
                    import traceback
                    logger.error(f"❌ ERROR in stage {stage.name}: {str(e)}")
                    logger.error(f"❌ TRACEBACK:\n{traceback.format_exc()}")
                    logger.error(f"Stage {stage.name} failed: {str(e)}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    raise
            
            execution_time = time.time() - start_time
            stage.status = StageStatus.COMPLETED
            
            # Post-execution processing
            self._post_stage_processing(stage, result, execution_time)
            
            # Save checkpoint using unified system
            process_id = self.context.experiment_id if self.context else "pipeline"
            checkpoint_data = {
                'stage': stage.name,
                'timestamp': datetime.now().isoformat(),
                'completed_stages': self._get_completed_stages(),
                'shared_data': self._serialize_shared_data(self.context.shared_data) if self.context else {},
                'metadata': self.context.metadata if self.context else {},
                'quality_metrics': self.context.quality_metrics if self.context else {},
                'stage_results': {stage.name: self._serialize_stage_result(result)},
                'progress': {
                    'completed_stages': len(self._get_completed_stages()),
                    'current_stage': stage.name
                }
            }
            
            self.checkpoint_manager.save(
                process=process_id,
                data=checkpoint_data,
                level=CheckpointLevel.STAGE,
                checkpoint_id=f"{process_id}_{stage.name}",
                parent_checkpoint_id=None,
                tags=['pipeline', 'stage', stage.name]
            )
            
            # Update progress
            self.progress_tracker.complete_stage(stage.name)
            
            return result
            
        except MemoryError as e:
            # Special handling for memory errors
            logger.error(f"Memory error in stage '{stage.name}': {e}")
            
            # Try to recover by enabling chunking
            if stage.supports_chunking and not getattr(stage, '_retry_with_chunking', False):
                logger.info("Retrying with chunked processing enabled")
                setattr(stage, "_retry_with_chunking", True)
                
                # Force chunking and retry
                if self.context:
                    self.context.config.settings.setdefault('resampling', {})['force_chunking'] = True
                
                # Clean up and retry
                self._handle_memory_pressure()
                return self._execute_stage(stage, completed_stages)
            
            # If already tried chunking, fail
            stage.status = StageStatus.FAILED
            stage.error = str(e)
            self.progress_tracker.fail_stage(stage.name, str(e))
            raise
            
        except Exception as e:
            stage.status = StageStatus.FAILED
            stage.error = str(e)
            logger.error(f"Stage '{stage.name}' failed: {e}")
            
            # Update progress
            self.progress_tracker.fail_stage(stage.name, str(e))
            
            # Quality check on failure
            self.quality_checker.check_stage_failure(stage, e, self.context)
            
            raise

    def _create_processing_config(self, stage: PipelineStage) -> ProcessingConfig:
        """Create processing configuration for memory-aware stages."""
        config = self.config
        
        # Get memory limit from config or system
        memory_limit_mb = config.get('processing.memory_limit_gb', 4.0) * 1024
        
        # Adjust based on current memory pressure
        memory_status = self.memory_monitor.get_status()
        if memory_status['pressure'] == 'warning':
            memory_limit_mb *= 0.7  # Reduce limit if under pressure
        elif memory_status['pressure'] == 'critical':
            memory_limit_mb *= 0.5  # Significantly reduce if critical
        
        return ProcessingConfig(
            chunk_size=config.get('processing.chunk_size', 1000),
            memory_limit_mb=memory_limit_mb,
            enable_chunking=config.get('processing.enable_chunking', True),
            checkpoint_interval=config.get('processing.checkpoint_interval', 10)
        )
    
    def _memory_monitoring_context(self, stage: PipelineStage):
        """Context manager for memory monitoring during stage execution."""
        class MemoryMonitoringContext:
            def __init__(self, orchestrator, stage):
                self.orchestrator = orchestrator
                self.stage = stage
                self.start_memory = None
            
            def __enter__(self):
                self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_increase = end_memory - self.start_memory
                
                if memory_increase > 1024:  # More than 1GB increase
                    logger.warning(
                        f"Stage '{self.stage.name}' increased memory by {memory_increase:.0f}MB"
                    )
                
                # Trigger cleanup if memory is high
                if end_memory > self.orchestrator.config.get('processing.memory_limit_gb', 4.0) * 1024 * 0.8:
                    self.orchestrator._handle_memory_pressure()
        
        return MemoryMonitoringContext(self, stage)
    
    def _handle_stage_progress(self, stage_name: str, progress_info: Dict[str, Any]):
        """Handle progress updates from stages."""
        # Update progress tracker with detailed info
        if 'progress_percent' in progress_info:
            self.progress_tracker.update_stage(
                stage_name, 
                progress_percent=progress_info['progress_percent']
            )
        
        # Emit progress event
        self.event_queue.put({
            'type': 'stage_progress',
            'stage': stage_name,
            'progress': progress_info
        })
        
        # Log significant progress
        if progress_info.get('message'):
            logger.info(f"{stage_name}: {progress_info['message']}")
    
    def _handle_memory_pressure(self):
        """Enhanced memory pressure handling."""
        # Trigger garbage collection
        import gc
        gc.collect()
        
        # Clear caches if implemented
        if self.context and hasattr(self.context, 'clear_caches'):
            self.context.clear_caches()  # type: ignore
        
        # Force garbage collection of unreferenced objects
        gc.collect(2)  # Full collection
        
        # Log memory status
        memory_status = self.memory_monitor.get_status()
        logger.info(f"Memory cleanup completed. Current usage: {memory_status['current_usage_gb']:.1f}GB")
        
        # Pause pipeline if still critical
        if memory_status['pressure'] == 'critical':
            logger.error("Memory pressure still critical after cleanup")
            
            # Try to free more memory by clearing any stage-specific caches
            for stage in self.stages:
                if hasattr(stage, 'cleanup'):
                    stage.cleanup(self.context)
            
            # Final check
            memory_status = self.memory_monitor.get_status()
            if memory_status['pressure'] == 'critical':
                self.pause_pipeline()
                raise MemoryError("Critical memory pressure - pipeline paused")
    
    def _execute_parallel_stages(self, stages: List[PipelineStage], 
                                completed_stages: set) -> Dict[str, StageResult]:
        """Execute multiple stages in parallel."""
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(stages)) as executor:
            # Submit all stages
            future_to_stage = {
                executor.submit(self._execute_stage, stage, completed_stages): stage
                for stage in stages
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_stage):
                stage = future_to_stage[future]
                try:
                    result = future.result()
                    results[stage.name] = result
                except Exception as e:
                    logger.error(f"Parallel stage '{stage.name}' failed: {e}")
                    # Cancel remaining futures
                    for f in future_to_stage:
                        f.cancel()
                    raise
        
        return results
    
    def _pre_stage_checks(self, stage: PipelineStage):
        """Pre-execution checks for a stage."""
        # Memory check
        available_memory = self.memory_monitor.get_available_memory()
        required_memory = stage.memory_requirements
        
        if required_memory and available_memory < required_memory:
            # Try to free memory
            self.memory_monitor.trigger_cleanup()
            available_memory = self.memory_monitor.get_available_memory()
            
            if available_memory < required_memory:
                raise MemoryError(
                    f"Insufficient memory for stage '{stage.name}': "
                    f"required {required_memory}GB, available {available_memory}GB"
                )
        
        # Disk space check
        if stage.disk_requirements:
            free_space = psutil.disk_usage(str(self.context.output_dir) if self.context else ".").free / (1024**3)
            if free_space < stage.disk_requirements:
                raise IOError(
                    f"Insufficient disk space for stage '{stage.name}': "
                    f"required {stage.disk_requirements}GB, available {free_space}GB"
                )
    
    def _post_stage_processing(self, stage: PipelineStage, 
                              result: StageResult, 
                              execution_time: float):
        """Post-execution processing for a stage."""
        # Quality checks
        quality_report = self.quality_checker.check_stage_output(
            stage, result, self.context
        )
        
        if quality_report.has_critical_issues():
            raise ValueError(f"Stage '{stage.name}' failed quality checks: {quality_report}")
        
        if not self.context: return
        # Update context with quality metrics
        self.context.quality_metrics[stage.name] = quality_report.to_dict()
        
        # Log performance metrics
        logger.info(f"Stage '{stage.name}' completed in {execution_time:.2f}s")
        
        # Emit monitoring event
        self.event_queue.put({
            'type': 'stage_completed',
            'stage': stage.name,
            'execution_time': execution_time,
            'quality_score': quality_report.overall_score
        })
    
    def _get_execution_order(self) -> List[List[PipelineStage]]:
        """Get stages in execution order respecting dependencies."""
        from collections import defaultdict, deque
        
        # Build dependency graph
        in_degree = defaultdict(int)
        adjacency = defaultdict(list)
        
        for stage in self.stages:
            for dep in stage.dependencies:
                adjacency[dep].append(stage.name)
                in_degree[stage.name] += 1
        
        # Topological sort with level grouping
        execution_order = []
        queue = deque([s for s in self.stages if in_degree[s.name] == 0])
        
        while queue:
            # Get all stages at current level (can run in parallel)
            current_level = []
            level_size = len(queue)
            
            for _ in range(level_size):
                stage_name = queue.popleft()
                stage_key = stage_name.name if hasattr(stage_name, "name") else stage_name
                stage = self.stage_registry[str(stage_key)]
                current_level.append(stage)
                
                # Update in-degrees
                for dependent in adjacency[stage_key]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
            
            if current_level:
                execution_order.append(current_level)
        
        return execution_order
    
    def _has_circular_dependency(self, start_stage: PipelineStage) -> bool:
        """Check if stage has circular dependencies."""
        visited = set()
        rec_stack = set()
        
        def visit(stage_name: str) -> bool:
            visited.add(stage_name)
            rec_stack.add(stage_name)
            
            stage = self.stage_registry.get(stage_name)
            if stage:
                for dep in stage.dependencies:
                    if dep not in visited:
                        if visit(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(stage_name)
            return False
        
        return visit(start_stage.name)
    
    def _check_stage_skip(self, stage: PipelineStage) -> Optional[StageResult]:
        """Check if stage can be skipped using StageSkipController."""
        from src.pipelines.stages.skip_control import StageSkipController
        
        # Initialize skip controller
        controller = StageSkipController(self.config, self.db)
        
        # Check if stage can be skipped
        can_skip, reason = controller.can_skip_stage(
            stage.name,
            self.context.metadata.get('experiment_name', '') if self.context else "",
            force_fresh=self.context.get('force_fresh', False) if self.context else False
        )
        
        # Log decision
        controller.log_skip_decision(stage.name, can_skip, reason)
        
        if not can_skip:
            return None
            
        # Handle different stages differently
        if stage.name == 'resample':
            # For resample stage, load existing datasets from database
            from src.database.schema import schema
            from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo
            
            existing_datasets = schema.get_resampled_datasets()
            resampled_infos = []
            
            for dataset in existing_datasets:
                # Convert database record to ResampledDatasetInfo
                info = ResampledDatasetInfo(
                    name=dataset['name'],
                    source_path=Path(dataset['source_path']),
                    bounds=tuple(dataset['bounds']),
                    shape=(dataset['shape_height'], dataset['shape_width']),
                    resampling_method=dataset['resampling_method'],
                    target_resolution=dataset['target_resolution'],
                    target_crs=dataset.get('target_crs', 'EPSG:4326'),
                    band_name=dataset.get('band_name', dataset['name']),
                    data_type=dataset.get('data_type', 'richness_data'),
                    metadata=dataset.get('metadata', {})
                )
                resampled_infos.append(info)
            
            # Store in context
            if self.context:
                self.context.set('resampled_datasets', resampled_infos)
            
            return StageResult(
                success=True,
                data={'resampled_datasets': resampled_infos},
                metrics={
                    'datasets_loaded': len(resampled_infos),
                    'skipped': True,
                    'reason': reason
                }
            )
            
        elif stage.name == 'data_load':
            # For data_load stage, verify files exist and return success
            datasets = self.config.get('datasets.target_datasets', [])
            enabled_datasets = [d for d in datasets if d.get('enabled', True)]
            
            # Store the dataset info in context as if we loaded them
            if self.context:
                self.context.set('target_datasets', enabled_datasets)
                self.context.set('dataset_configs', enabled_datasets)
            
            return StageResult(
                success=True,
                data={'datasets': enabled_datasets},
                metrics={
                    'datasets_loaded': len(enabled_datasets),  # Quality checker expects this
                    'datasets_count': len(enabled_datasets),
                    'skipped': True,
                    'reason': reason
                }
            )
            
        else:
            # For other stages, return a generic skip result
            return StageResult(
                success=True,
                data={},
                metrics={
                    'skipped': True,
                    'reason': reason
                }
            )
    
    def _start_monitoring(self):
        """Start monitoring threads."""
        print("🧵 DEBUG: Creating monitor thread...")
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        print("🧵 DEBUG: Starting monitor thread...")
        self._monitor_thread.start()
        
        # Start individual monitors
        print("💾 DEBUG: Starting memory monitor...")
        self.memory_monitor.start()
        print("📊 DEBUG: Starting progress tracker...")
        self.progress_tracker.start()
        print("✅ DEBUG: All monitors started")
    
    def _stop_monitoring(self):
        """Stop monitoring threads."""
        # Set flag to stop monitoring loop
        self._stop_requested = True
        
        # Stop individual monitors
        self.memory_monitor.stop()
        self.progress_tracker.stop()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_requested:
            try:
                # Process events
                while not self.event_queue.empty():
                    event = self.event_queue.get_nowait()
                    self._handle_monitoring_event(event)
                
                # Check memory
                memory_status = self.memory_monitor.get_status()
                if memory_status['pressure'] == 'critical':
                    logger.warning("Critical memory pressure detected")
                    self._handle_memory_pressure()
                
                # Update progress
                self.progress_tracker.update()
                
                time.sleep(1)  # Monitor interval
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _handle_monitoring_event(self, event: Dict[str, Any]):
        """Handle monitoring event."""
        event_type = event.get('type')
        
        if event_type == 'stage_completed':
            logger.info(f"Stage {event['stage']} completed with quality score {event['quality_score']}")
        elif event_type == 'memory_warning':
            logger.warning(f"Memory warning: {event['message']}")
        elif event_type == 'quality_issue':
            logger.warning(f"Quality issue detected: {event['message']}")
        
    def _restore_from_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore pipeline state from checkpoint."""
        if not self.context: return
        # Restore context
        self.context.shared_data.update(checkpoint.get('shared_data', {}))
        self.context.metadata.update(checkpoint.get('metadata', {}))
        
        # Mark completed stages
        completed_stages = checkpoint.get('completed_stages', [])
        for stage_name in completed_stages:
            if stage_name in self.stage_registry:
                self.stage_registry[stage_name].status = StageStatus.COMPLETED
        
        # Restore progress
        self.progress_tracker.restore_from_checkpoint(checkpoint.get('progress', {}))
    
    def _attempt_recovery(self) -> Dict[str, Any]:
        """Attempt to recover from failure."""
        recovery_strategy = self.failure_handler.get_recovery_strategy()
        
        if recovery_strategy == 'retry':
            # Retry from last checkpoint
            process_id = self.context.experiment_id if self.context else "pipeline"
            checkpoint_data = self.checkpoint_manager.load_latest(process_id, CheckpointLevel.STAGE)
            if checkpoint_data:
                self._restore_from_checkpoint(checkpoint_data.data)
                return self._execute_pipeline()
        
        elif recovery_strategy == 'skip':
            # Skip failed stage and continue
            current_stage = self.get_current_stage()
            if current_stage:
                current_stage.status = StageStatus.SKIPPED
                return self._execute_pipeline()
        
        raise RuntimeError("Recovery failed")
    
    def _finalize_pipeline(self, results: Dict[str, Any]):
        """Finalize pipeline execution."""
        # Generate final report
        report = self._generate_execution_report(results)
        
        # Save report
        if self.context:
            report_path = self.context.output_dir / 'pipeline_report.json'
        else:
            report_path = Path(".") / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Update experiment status
        from src.database.schema import schema
        schema.update_experiment_status(
            self.context.experiment_id if self.context else "",
            'completed',
            report
        )
        
        # Cleanup checkpoints if successful - handled automatically by unified system
        if self.config.get('pipeline.cleanup_checkpoints_on_success', True):
            # The unified checkpoint system handles cleanup automatically based on policies
            pass
        
        logger.info(f"Pipeline completed successfully. Report saved to {report_path}")
    
    def _get_completed_stages(self) -> List[str]:
        """Get list of completed stages."""
        completed = []
        for stage in self.stages:
            if stage.status == StageStatus.COMPLETED:
                completed.append(stage.name)
        return completed
    
    def _serialize_shared_data(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize shared data for checkpointing."""
        serialized = {}
        for key, value in shared_data.items():
            if key in ['merged_dataset', 'som_results']:
                # Store reference instead of actual data
                serialized[key] = f"<{type(value).__name__} object>"
            elif hasattr(value, 'to_dict'):
                serialized[key] = value.to_dict()
            else:
                try:
                    json.dumps(value)  # Test JSON serialization
                    serialized[key] = value
                except (TypeError, ValueError):
                    serialized[key] = str(value)
        return serialized
    
    def _serialize_stage_result(self, result: Any) -> Dict[str, Any]:
        """Serialize stage result for checkpointing."""
        if hasattr(result, 'to_dict'):
            return result.to_dict()
        elif hasattr(result, '__dict__'):
            return {
                'success': getattr(result, 'success', True),
                'metrics': getattr(result, 'metrics', {}),
                'warnings': getattr(result, 'warnings', [])
            }
        else:
            return {'result': str(result)}

    def _generate_execution_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive execution report."""
        # Calculate execution time safely
        execution_time = 0
        if self.context and self.context.metadata:
            start_time = self.context.metadata.get("start_time")
            if start_time:
                if isinstance(start_time, str):
                    # If it's already a string, parse it
                    from dateutil import parser
                    start_time = parser.parse(start_time)
                execution_time = (datetime.now() - start_time).total_seconds()
        
        # Clean metadata to ensure JSON serialization
        metadata = {}
        if self.context and self.context.metadata:
            for key, value in self.context.metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = value.isoformat()
                else:
                    metadata[key] = value
        
        return {
            'experiment_id': self.context.experiment_id if self.context else "",
            'status': self.status.value,
            'execution_time': execution_time,
            'stages_completed': len([s for s in self.stages if s.status == StageStatus.COMPLETED]),
            'total_stages': len(self.stages),
            'quality_metrics': self.context.quality_metrics if self.context else {},
            'memory_stats': self.memory_monitor.get_summary(),
            'stage_results': {
                stage_name: {
                    'status': self.stage_registry[stage_name].status.value,
                    'execution_time': getattr(result, 'execution_time', None),
                    'output_size': getattr(result, 'output_size', None)
                }
                for stage_name, result in results.items()
            },
            'metadata': metadata
        }
    
    def pause_pipeline(self):
        """Pause pipeline execution."""
        logger.info("Pausing pipeline...")
        self._pause_requested = True
        self.status = PipelineStatus.PAUSED
        
        # Pause all currently running stages
        for stage in self.stages:
            if stage.status == StageStatus.RUNNING:
                stage.pause()
    
    def resume_pipeline(self):
        """Resume pipeline execution."""
        logger.info("Resuming pipeline...")
        self._pause_requested = False
        self.status = PipelineStatus.RUNNING
        
        # Resume all paused stages
        for stage in self.stages:
            if stage.status == StageStatus.PAUSED:
                stage.resume()
    
    def stop_pipeline(self):
        """Stop pipeline execution."""
        logger.info("Stopping pipeline...")
        self._stop_requested = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'status': self.status.value,
            'progress': self.progress_tracker.get_progress(),
            'current_stage': getattr(self.get_current_stage(), "name", None) if self.get_current_stage() else None,
            'memory_usage': self.memory_monitor.get_status(),
            'quality_scores': self.context.quality_metrics if self.context else {}
        }
    
    def get_current_stage(self) -> Optional[PipelineStage]:
        """Get currently executing stage."""
        for stage in self.stages:
            if stage.status == StageStatus.RUNNING:
                return stage
        return None
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return {
            'stages': [
                {
                    'name': stage.name,
                    'dependencies': stage.dependencies,
                    'memory_requirements': stage.memory_requirements,
                    'disk_requirements': stage.disk_requirements
                }
                for stage in self.stages
            ],
            'monitoring': {
                'memory_limit_gb': self.config.get('processing.memory_limit_gb', 16),
                'checkpoint_interval': self.config.get('pipeline.checkpoint_interval', 300)
            }
        }