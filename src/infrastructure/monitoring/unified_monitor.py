"""Unified monitor that combines progress, metrics, and logging."""

import time
import threading
import psutil
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from contextlib import contextmanager

from src.base.monitoring import ProgressBackend, MetricsBackend
from src.infrastructure.logging import get_logger, LoggingContext
from .database_progress_backend import DatabaseProgressBackend
from .database_metrics_backend import DatabaseMetricsBackend
from .memory_progress_backend import MemoryProgressBackend

logger = get_logger(__name__)


class UnifiedMonitor:
    """Unified monitoring system for pipeline execution.
    
    Combines:
    - Progress tracking (hierarchical)
    - Metrics collection (performance)
    - Structured logging (context-aware)
    - Resource monitoring (CPU/memory)
    """
    
    def __init__(self, config: Dict[str, Any], db_manager = None):
        """Initialize unified monitor.
        
        Args:
            config: Configuration dict
            db_manager: Optional DatabaseManager for persistence
        """
        self.config = config
        self.db = db_manager
        
        # Setup backends based on configuration
        if db_manager and config.get('monitoring.enable_database_backend', True):
            self.progress_backend = DatabaseProgressBackend(db_manager)
            self.metrics_backend = DatabaseMetricsBackend(db_manager)
            logger.info("Using database backends for monitoring")
        else:
            self.progress_backend = MemoryProgressBackend()
            self.metrics_backend = None  # No metrics without database
            logger.info("Using memory backend for monitoring (no persistence)")
        
        # Monitoring state
        self.experiment_id: Optional[str] = None
        self.logging_context: Optional[LoggingContext] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._monitor_interval = config.get('monitoring.metrics_interval', 10)
        
        # Callbacks
        self._progress_callbacks: List[Callable] = []
        self._metrics_callbacks: List[Callable] = []
    
    def start(self, experiment_id: str, job_id: Optional[str] = None):
        """Start monitoring for an experiment.
        
        Args:
            experiment_id: Experiment UUID
            job_id: Optional job UUID
        """
        self.experiment_id = experiment_id
        self.logging_context = LoggingContext(experiment_id, job_id)
        
        # Start resource monitoring thread
        if self.metrics_backend:
            self._stop_event.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitor_resources,
                name=f"ResourceMonitor-{experiment_id[:8]}",
                daemon=True
            )
            self._monitoring_thread.start()
            logger.info(f"Started monitoring for experiment {experiment_id}")
        
    def stop(self):
        """Stop monitoring."""
        # Stop resource monitoring
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_event.set()
            self._monitoring_thread.join(timeout=5)
        
        logger.info(f"Stopped monitoring for experiment {self.experiment_id}")
        
        self.experiment_id = None
        self.logging_context = None
    
    @contextmanager
    def track_stage(self, stage_name: str, total_units: int = 100, **metadata):
        """Context manager for tracking a pipeline stage.
        
        Args:
            stage_name: Name of the stage
            total_units: Total units for progress tracking
            **metadata: Additional stage metadata
        """
        if not self.experiment_id:
            raise RuntimeError("Monitor not started. Call start() first.")
        
        # Use logging context
        with self.logging_context.stage(stage_name, **metadata):
            node_id = self.logging_context.current_node
            
            # Create progress node
            self.progress_backend.create_node(
                experiment_id=self.experiment_id,
                node_id=node_id,
                parent_id=self.logging_context.node_stack[-2] if len(self.logging_context.node_stack) > 1 else None,
                level='stage',
                name=stage_name,
                total_units=total_units,
                metadata=metadata
            )
            
            # Update to running
            self.progress_backend.update_progress(
                self.experiment_id, node_id, 0, 'running'
            )
            
            try:
                yield ProgressTracker(self, node_id, total_units)
                
                # Mark as completed
                self.progress_backend.update_progress(
                    self.experiment_id, node_id, total_units, 'completed'
                )
                
            except Exception as e:
                # Mark as failed
                self.progress_backend.update_progress(
                    self.experiment_id, node_id, 
                    self.progress_backend.get_node(self.experiment_id, node_id)['completed_units'],
                    'failed',
                    metadata={'error': str(e)}
                )
                raise
    
    def update_progress(self, node_id: str, completed_units: int, 
                       message: Optional[str] = None):
        """Update progress for a node.
        
        Args:
            node_id: Node identifier
            completed_units: Number of completed units
            message: Optional progress message
        """
        if not self.experiment_id:
            return
        
        # Update backend
        self.progress_backend.update_progress(
            self.experiment_id, node_id, completed_units, 'running'
        )
        
        # Log progress if message provided
        if message:
            node = self.progress_backend.get_node(self.experiment_id, node_id)
            if node:
                self.logging_context.log_progress(
                    completed_units, node['total_units'], message
                )
        
        # Notify callbacks
        for callback in self._progress_callbacks:
            try:
                callback(node_id, completed_units)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def record_metrics(self, metrics: Dict[str, float], node_id: Optional[str] = None):
        """Record performance metrics.
        
        Args:
            metrics: Dict of metric name to value
            node_id: Optional node context
        """
        if not self.experiment_id or not self.metrics_backend:
            return
        
        # Record to backend
        self.metrics_backend.record_metrics(
            self.experiment_id, node_id, metrics
        )
        
        # Log significant metrics
        if metrics.get('memory_mb', 0) > self.config.get('monitoring.memory_warning_mb', 10000):
            logger.warning(
                f"High memory usage: {metrics['memory_mb']:.1f} MB",
                extra={'performance': metrics}
            )
        
        # Notify callbacks
        for callback in self._metrics_callbacks:
            try:
                callback(metrics, node_id)
            except Exception as e:
                logger.error(f"Metrics callback error: {e}")
    
    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates."""
        self._progress_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """Add callback for metrics updates."""
        self._metrics_callbacks.append(callback)
    
    def _monitor_resources(self):
        """Background thread to monitor system resources."""
        process = psutil.Process()
        
        while not self._stop_event.wait(self._monitor_interval):
            try:
                # Collect metrics
                metrics = {
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(interval=1),
                    'num_threads': process.num_threads()
                }
                
                # Add system-wide metrics
                system_memory = psutil.virtual_memory()
                metrics['system_memory_percent'] = system_memory.percent
                metrics['system_memory_available_mb'] = system_memory.available / 1024 / 1024
                
                # Record metrics
                self.record_metrics(metrics, node_id=self.logging_context.current_node)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")


class ProgressTracker:
    """Helper class for tracking progress within a context."""
    
    def __init__(self, monitor: UnifiedMonitor, node_id: str, total_units: int):
        """Initialize progress tracker.
        
        Args:
            monitor: Parent UnifiedMonitor
            node_id: Node being tracked
            total_units: Total units for progress
        """
        self.monitor = monitor
        self.node_id = node_id
        self.total_units = total_units
        self._completed = 0
    
    def update(self, completed: int, message: Optional[str] = None):
        """Update progress.
        
        Args:
            completed: Number of completed units
            message: Optional progress message
        """
        self._completed = completed
        self.monitor.update_progress(self.node_id, completed, message)
    
    def increment(self, delta: int = 1, message: Optional[str] = None):
        """Increment progress.
        
        Args:
            delta: Amount to increment
            message: Optional progress message
        """
        self._completed += delta
        self.monitor.update_progress(self.node_id, self._completed, message)
    
    @property
    def progress_percent(self) -> float:
        """Get current progress percentage."""
        if self.total_units <= 0:
            return 0.0
        return min(100.0, (self._completed / self.total_units) * 100.0)