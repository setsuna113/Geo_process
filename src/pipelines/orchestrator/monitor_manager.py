# src/pipelines/orchestrator/monitor_manager.py
"""Monitoring coordination for pipeline orchestration."""

import logging
import threading
import time
import psutil
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..monitors.memory_monitor import MemoryMonitor
from ..monitors.progress_tracker import ProgressTracker
from ..monitors.quality_checker import QualityChecker
from ..stages.base_stage import PipelineStage

logger = logging.getLogger(__name__)


@dataclass
class MonitoringState:
    """Current monitoring state."""
    is_active: bool = False
    start_time: Optional[float] = None
    current_stage: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class MonitorManager:
    """Manages monitoring and metrics collection during pipeline execution."""
    
    def __init__(self, config):
        """Initialize monitoring manager."""
        self.config = config
        
        # Monitoring components
        self.memory_monitor = MemoryMonitor(config)
        self.progress_tracker = ProgressTracker()
        self.quality_checker = QualityChecker(config)
        
        # Monitoring state
        self.state = MonitoringState()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._metrics_lock = threading.RLock()
        
        # Event callbacks
        self._progress_callbacks: list[Callable] = []
        self._metrics_callbacks: list[Callable] = []
        
    def start_monitoring(self):
        """Start monitoring systems."""
        if self.state.is_active:
            logger.warning("Monitoring already active")
            return
            
        logger.info("Starting pipeline monitoring")
        self.state.is_active = True
        self.state.start_time = time.time()
        self._stop_event.clear()
        
        # Start monitoring components
        self.memory_monitor.start()
        self.progress_tracker.start()
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PipelineMonitoring",
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info("Pipeline monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring systems."""
        if not self.state.is_active:
            return
            
        logger.info("Stopping pipeline monitoring")
        self.state.is_active = False
        self._stop_event.set()
        
        # Stop monitoring components
        self.memory_monitor.stop()
        self.progress_tracker.stop()
        
        # Wait for monitoring thread to finish
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
            
        # Record final metrics
        if self.state.start_time:
            total_time = time.time() - self.state.start_time
            with self._metrics_lock:
                self.state.metrics['total_execution_time'] = total_time
                
        logger.info("Pipeline monitoring stopped")
        
    @contextmanager
    def memory_monitoring_context(self, stage: PipelineStage):
        """Context manager for stage-specific memory monitoring."""
        monitor_context = self._MemoryMonitorContext(self, stage)
        
        try:
            monitor_context.start()
            yield monitor_context
        finally:
            monitor_context.stop()
            
    class _MemoryMonitorContext:
        """Context for monitoring memory during stage execution."""
        
        def __init__(self, monitor_manager, stage):
            self.monitor_manager = monitor_manager
            self.stage = stage
            self.start_memory = None
            self.peak_memory = 0
            self.monitoring_active = False
            
        def start(self):
            """Start memory monitoring for this stage."""
            self.start_memory = psutil.virtual_memory().used
            self.peak_memory = self.start_memory
            self.monitoring_active = True
            
            logger.debug(f"Started memory monitoring for stage {self.stage.stage_name}")
            
        def stop(self):
            """Stop memory monitoring and record metrics."""
            if not self.monitoring_active:
                return
                
            self.monitoring_active = False
            current_memory = psutil.virtual_memory().used
            
            # Calculate memory usage
            memory_used = current_memory - self.start_memory if self.start_memory else 0
            memory_peak = self.peak_memory - self.start_memory if self.start_memory else 0
            
            # Store metrics
            stage_metrics = {
                'memory_used_mb': memory_used / (1024 * 1024),
                'memory_peak_mb': memory_peak / (1024 * 1024),
                'final_memory_mb': current_memory / (1024 * 1024)
            }
            
            with self.monitor_manager._metrics_lock:
                stage_key = f"stage_{self.stage.stage_name}_memory"
                self.monitor_manager.state.metrics[stage_key] = stage_metrics
                
            logger.debug(f"Memory monitoring stopped for stage {self.stage.stage_name}: {stage_metrics}")
            
        def update_peak_memory(self):
            """Update peak memory usage."""
            if self.monitoring_active:
                current_memory = psutil.virtual_memory().used
                self.peak_memory = max(self.peak_memory, current_memory)
                
    def handle_stage_progress(self, stage_name: str, progress_info: Dict[str, Any]):
        """Handle progress updates from stages."""
        with self._metrics_lock:
            self.state.current_stage = stage_name
            
            # Update progress metrics
            progress_key = f"stage_{stage_name}_progress"
            self.state.metrics[progress_key] = {
                'timestamp': time.time(),
                'progress_percent': progress_info.get('progress_percent', 0),
                'items_processed': progress_info.get('items_processed', 0),
                'estimated_completion': progress_info.get('estimated_completion'),
                **progress_info
            }
            
        # Notify progress callbacks
        for callback in self._progress_callbacks:
            try:
                callback(stage_name, progress_info)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
                
        logger.debug(f"Progress update for {stage_name}: {progress_info}")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.debug("Monitoring loop started")
        
        while not self._stop_event.wait(timeout=5.0):  # Check every 5 seconds
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Update memory monitor contexts
                self._update_memory_contexts()
                
                # Process monitoring events
                self._process_monitoring_events()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
        logger.debug("Monitoring loop ended")
        
    def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        try:
            # System resource usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metrics = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_free_gb': disk.free / (1024 * 1024 * 1024)
            }
            
            with self._metrics_lock:
                self.state.metrics['system'] = system_metrics
                
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            
    def _update_memory_contexts(self):
        """Update peak memory for active monitoring contexts."""
        # This would be called by active contexts
        # Implementation depends on how contexts are tracked
        pass
        
    def _process_monitoring_events(self):
        """Process any pending monitoring events."""
        # Check for memory pressure
        memory = psutil.virtual_memory()
        if memory.percent > 85:  # High memory usage
            event = {
                'type': 'memory_pressure',
                'severity': 'high' if memory.percent > 95 else 'medium',
                'memory_percent': memory.percent,
                'timestamp': time.time()
            }
            self._handle_monitoring_event(event)
            
    def _handle_monitoring_event(self, event: Dict[str, Any]):
        """Handle monitoring events."""
        event_type = event.get('type')
        
        if event_type == 'memory_pressure':
            severity = event.get('severity', 'low')
            memory_percent = event.get('memory_percent', 0)
            
            logger.warning(f"Memory pressure detected: {severity} ({memory_percent:.1f}% used)")
            
            # Store event in metrics
            with self._metrics_lock:
                events = self.state.metrics.setdefault('events', [])
                events.append(event)
                
                # Keep only recent events (last 100)
                if len(events) > 100:
                    events[:] = events[-100:]
                    
            # Notify callbacks
            for callback in self._metrics_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Metrics callback failed: {e}")
                    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        with self._metrics_lock:
            return {
                'state': {
                    'is_active': self.state.is_active,
                    'current_stage': self.state.current_stage,
                    'uptime': time.time() - self.state.start_time if self.state.start_time else 0
                },
                'metrics': self.state.metrics.copy()
            }
            
    def add_progress_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for progress updates."""
        self._progress_callbacks.append(callback)
        
    def add_metrics_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for monitoring events."""
        self._metrics_callbacks.append(callback)
        
    def reset_metrics(self):
        """Reset all collected metrics."""
        with self._metrics_lock:
            self.state.metrics.clear()
            self.state.current_stage = None
            
        logger.info("Monitoring metrics reset")