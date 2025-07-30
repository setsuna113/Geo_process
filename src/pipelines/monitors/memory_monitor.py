# src/pipelines/monitors/memory_monitor.py
"""Memory monitoring for pipeline execution."""

import psutil
import logging
import threading
import time
from typing import Dict, Any, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during pipeline execution."""
    
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db = db_manager  # Store database reference
        self.memory_limit_gb = config.get('pipeline.memory_limit_gb', 16)
        self.warning_threshold = 0.8  # 80% of limit
        self.critical_threshold = 0.9  # 90% of limit
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self.memory_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.peak_memory_gb = 0
        
        # Context tracking
        self.experiment_id: Optional[str] = None
        self.current_stage: Optional[str] = None
        self.current_operation: Optional[str] = None
        
        # Callbacks
        self.warning_callbacks = []
        self.critical_callbacks = []
    
    def start(self):
        """Start memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Get current memory usage
                memory_info = psutil.virtual_memory()
                process_info = psutil.Process()
                
                current_usage = {
                    'timestamp': datetime.now(),
                    'system_total_gb': memory_info.total / (1024**3),
                    'system_available_gb': memory_info.available / (1024**3),
                    'system_percent': memory_info.percent,
                    'process_rss_gb': process_info.memory_info().rss / (1024**3),
                    'process_vms_gb': process_info.memory_info().vms / (1024**3)
                }
                
                self.memory_history.append(current_usage)
                
                # Update peak memory
                if current_usage['process_rss_gb'] > self.peak_memory_gb:
                    self.peak_memory_gb = current_usage['process_rss_gb']
                
                # Store to database if available
                if self.db and self.experiment_id:
                    self._store_memory_snapshot(current_usage)
                
                # Check thresholds
                usage_ratio = current_usage['process_rss_gb'] / self.memory_limit_gb
                
                if usage_ratio > self.critical_threshold:
                    self._trigger_critical(current_usage)
                elif usage_ratio > self.warning_threshold:
                    self._trigger_warning(current_usage)
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(5)
    
    def _trigger_warning(self, usage: Dict[str, Any]):
        """Trigger memory warning."""
        logger.warning(f"Memory usage warning: {usage['process_rss_gb']:.2f}GB / {self.memory_limit_gb}GB")
        
        for callback in self.warning_callbacks:
            try:
                callback(usage)
            except Exception as e:
                logger.error(f"Warning callback error: {e}")
    
    def _trigger_critical(self, usage: Dict[str, Any]):
        """Trigger critical memory alert."""
        logger.error(f"Critical memory usage: {usage['process_rss_gb']:.2f}GB / {self.memory_limit_gb}GB")
        
        for callback in self.critical_callbacks:
            try:
                callback(usage)
            except Exception as e:
                logger.error(f"Critical callback error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        memory_info = psutil.virtual_memory()
        process_info = psutil.Process()
        
        current_usage_gb = process_info.memory_info().rss / (1024**3)
        usage_ratio = current_usage_gb / self.memory_limit_gb
        
        if usage_ratio > self.critical_threshold:
            pressure = 'critical'
        elif usage_ratio > self.warning_threshold:
            pressure = 'warning'
        else:
            pressure = 'normal'
        
        return {
            'current_usage_gb': current_usage_gb,
            'limit_gb': self.memory_limit_gb,
            'usage_percent': usage_ratio * 100,
            'peak_usage_gb': self.peak_memory_gb,
            'system_available_gb': memory_info.available / (1024**3),
            'pressure': pressure
        }
    
    def get_available_memory(self) -> float:
        """Get available memory in GB."""
        return psutil.virtual_memory().available / (1024**3)
    
    def trigger_cleanup(self):
        """Trigger memory cleanup."""
        import gc
        
        logger.info("Triggering memory cleanup")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Clear any caches (implement as needed)
        # This is a placeholder for cache clearing logic
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_history:
            return {}
        
        recent_history = list(self.memory_history)[-100:]  # Last 100 measurements
        
        avg_usage = sum(h['process_rss_gb'] for h in recent_history) / len(recent_history)
        max_usage = max(h['process_rss_gb'] for h in recent_history)
        
        return {
            'average_usage_gb': avg_usage,
            'peak_usage_gb': self.peak_memory_gb,
            'max_recent_usage_gb': max_usage,
            'measurements': len(self.memory_history)
        }
    
    def register_warning_callback(self, callback):
        """Register callback for memory warnings."""
        self.warning_callbacks.append(callback)
    
    def register_critical_callback(self, callback):
        """Register callback for critical memory alerts."""
        self.critical_callbacks.append(callback)
    
    def set_experiment_id(self, experiment_id: str):
        """Set the experiment ID for tracking."""
        self.experiment_id = experiment_id
        logger.debug(f"Memory monitor tracking experiment: {experiment_id}")
    
    def set_stage(self, stage: str):
        """Set current pipeline stage."""
        self.current_stage = stage
        logger.debug(f"Memory monitor tracking stage: {stage}")
    
    def set_operation(self, operation: str):
        """Set current operation within stage."""
        self.current_operation = operation
        logger.debug(f"Memory monitor tracking operation: {operation}")
    
    def _store_memory_snapshot(self, usage: Dict[str, Any]):
        """Store memory snapshot to database."""
        if not self.db or not self.experiment_id:
            return
            
        try:
            import json
            # Convert experiment_id to UUID format if needed
            experiment_uuid = self.experiment_id
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # First check if experiment exists
                    cur.execute("SELECT id FROM experiments WHERE id = %s::uuid", (experiment_uuid,))
                    if not cur.fetchone():
                        logger.warning(f"Experiment {experiment_uuid} not found in database")
                        return
                    
                    # Use the existing pipeline_metrics table structure
                    custom_metrics = {
                        'memory_type': 'process_rss',
                        'process_rss_gb': usage['process_rss_gb'],
                        'process_vms_gb': usage.get('process_vms_gb', 0),
                        'system_percent': usage['system_percent'],
                        'system_available_gb': usage['system_available_gb'],
                        'stage': self.current_stage,
                        'operation': self.current_operation,
                        'peak_memory_gb': self.peak_memory_gb
                    }
                    
                    cur.execute("""
                        INSERT INTO pipeline_metrics 
                        (experiment_id, node_id, memory_mb, cpu_percent, custom_metrics)
                        VALUES (%s::uuid, %s, %s, %s, %s)
                    """, (
                        experiment_uuid,
                        f"memory_monitor_{self.current_stage or 'pipeline'}",
                        usage['process_rss_gb'] * 1024,  # Convert GB to MB
                        usage.get('system_percent', 0),
                        json.dumps(custom_metrics)
                    ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to store memory snapshot: {e}")