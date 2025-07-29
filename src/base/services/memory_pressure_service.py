"""Memory pressure monitoring and callback service."""

import threading
import time
import logging
from typing import Callable, List, Dict, Any
from .memory_tracker_service import MemoryPressureLevel

logger = logging.getLogger(__name__)


class MemoryPressureService:
    """Service for monitoring memory pressure and triggering callbacks."""
    
    def __init__(self, memory_tracker):
        self._memory_tracker = memory_tracker
        self._pressure_callbacks: List[Callable[[MemoryPressureLevel], None]] = []
        self._callback_lock = threading.RLock()
        
        # Monitoring configuration
        self._config = {
            'monitoring_interval': 5.0,  # seconds
            'enable_auto_cleanup': True,
            'callback_timeout': 10.0  # seconds
        }
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._last_pressure_level = MemoryPressureLevel.NORMAL
    
    def register_pressure_callback(self, callback: Callable[[MemoryPressureLevel], None]) -> None:
        """Register a callback for memory pressure events."""
        with self._callback_lock:
            self._pressure_callbacks.append(callback)
            logger.debug(f"Registered pressure callback: {callback.__name__}")
    
    def unregister_pressure_callback(self, callback: Callable[[MemoryPressureLevel], None]) -> bool:
        """Unregister a pressure callback."""
        with self._callback_lock:
            try:
                self._pressure_callbacks.remove(callback)
                logger.debug(f"Unregistered pressure callback: {callback.__name__}")
                return True
            except ValueError:
                logger.warning(f"Callback not found: {callback.__name__}")
                return False
    
    def start_monitoring(self) -> None:
        """Start background memory pressure monitoring."""
        if self._monitoring:
            logger.warning("Memory pressure monitoring already running")
            return
        
        self._monitoring = True
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory_pressure,
            name="MemoryPressureMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info("Started memory pressure monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background memory pressure monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logger.info("Stopped memory pressure monitoring")
    
    def check_pressure_now(self) -> MemoryPressureLevel:
        """Check current memory pressure and trigger callbacks if needed."""
        current_level = self._memory_tracker.get_memory_pressure_level()
        
        # Trigger callbacks if pressure level changed
        if current_level != self._last_pressure_level:
            self._handle_memory_pressure(current_level)
            self._last_pressure_level = current_level
        
        return current_level
    
    def configure(self, **config) -> None:
        """Update configuration."""
        self._config.update(config)
        logger.debug(f"Updated pressure service config: {config}")
    
    def get_pressure_stats(self) -> Dict[str, Any]:
        """Get pressure monitoring statistics."""
        return {
            'monitoring_active': self._monitoring,
            'current_pressure_level': self._last_pressure_level.value,
            'registered_callbacks': len(self._pressure_callbacks),
            'config': self._config.copy()
        }
    
    def _monitor_memory_pressure(self) -> None:
        """Background memory pressure monitoring loop."""
        while not self._stop_event.wait(self._config['monitoring_interval']):
            try:
                self.check_pressure_now()
                
            except Exception as e:
                logger.error(f"Memory pressure monitoring error: {e}")
    
    def _handle_memory_pressure(self, pressure_level: MemoryPressureLevel) -> None:
        """Handle memory pressure by triggering callbacks."""
        logger.info(f"Memory pressure level changed to: {pressure_level.value}")
        
        # Auto cleanup on high pressure
        if (pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL] 
            and self._config['enable_auto_cleanup']):
            try:
                freed_mb = self._memory_tracker.trigger_cleanup(force_gc=True)
                if freed_mb > 0:
                    logger.info(f"Auto cleanup freed {freed_mb:.1f}MB")
            except Exception as e:
                logger.error(f"Auto cleanup failed: {e}")
        
        # Trigger registered callbacks
        with self._callback_lock:
            for callback in self._pressure_callbacks:
                try:
                    # Run callback with timeout
                    callback_thread = threading.Thread(
                        target=callback,
                        args=(pressure_level,),
                        name=f"PressureCallback-{callback.__name__}",
                        daemon=True
                    )
                    callback_thread.start()
                    callback_thread.join(timeout=self._config['callback_timeout'])
                    
                    if callback_thread.is_alive():
                        logger.warning(f"Pressure callback {callback.__name__} timed out")
                    
                except Exception as e:
                    logger.error(f"Pressure callback {callback.__name__} failed: {e}")
    
    def __del__(self):
        """Cleanup when service is destroyed."""
        try:
            self.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping pressure monitoring: {e}")