"""Mixin for lazy-loading support with resource management."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, Union, ClassVar
from contextlib import contextmanager
import weakref
import threading
import logging

logger = logging.getLogger(__name__)


class LazyLoadable(ABC):
    """
    Mixin for lazy-loading support.
    
    Provides context manager for resource management and memory pressure response.
    """
    
    _instances: ClassVar[weakref.WeakSet] = weakref.WeakSet()
    """
    Mixin for lazy-loading support.
    
    Provides context manager for resource management and memory pressure response.
    """
    
    def __init__(self):
        self._is_loaded = False
        self._load_lock = threading.RLock()
        self._resource_handle: Optional[Any] = None
        self._memory_pressure_callbacks = []
        self._lazy_config = {
            'auto_unload': True,
            'unload_delay_seconds': 300,  # 5 minutes
            'max_idle_time': 1800,  # 30 minutes
        }
        
        # Register for memory pressure notifications
        self._register_memory_pressure()
        
    @abstractmethod
    def _load_resource(self) -> Any:
        """Load the actual resource. Override in subclasses."""
        pass
        
    @abstractmethod
    def _unload_resource(self) -> None:
        """Unload the resource and free memory. Override in subclasses."""
        pass
        
    def ensure_loaded(self) -> Any:
        """Ensure resource is loaded and return handle."""
        with self._load_lock:
            if not self._is_loaded:
                logger.debug(f"Lazy loading resource for {self.__class__.__name__}")
                self._resource_handle = self._load_resource()
                self._is_loaded = True
                
            return self._resource_handle
            
    def is_loaded(self) -> bool:
        """Check if resource is currently loaded."""
        return self._is_loaded
        
    def unload(self) -> None:
        """Manually unload the resource."""
        with self._load_lock:
            if self._is_loaded:
                logger.debug(f"Unloading resource for {self.__class__.__name__}")
                self._unload_resource()
                self._resource_handle = None
                self._is_loaded = False
                
    @contextmanager
    def lazy_context(self):
        """Context manager for automatic resource management."""
        try:
            handle = self.ensure_loaded()
            yield handle
        finally:
            if self._lazy_config.get('auto_unload', True):
                # Schedule unload after delay
                self._schedule_unload()
                
    def _schedule_unload(self) -> None:
        """Schedule automatic unload after delay."""
        import threading
        import time
        
        def delayed_unload():
            time.sleep(self._lazy_config.get('unload_delay_seconds', 300))
            if self._is_loaded:
                self.unload()
                
        thread = threading.Thread(target=delayed_unload, daemon=True)
        thread.start()
        
    def on_memory_pressure(self, pressure_level: str) -> None:
        """Handle memory pressure notifications."""
        if pressure_level in ['high', 'critical']:
            logger.info(f"Memory pressure ({pressure_level}), unloading {self.__class__.__name__}")
            self.unload()
            
    def _register_memory_pressure(self) -> None:
        """Register for memory pressure notifications."""
        # This would integrate with a system memory monitor
        # For now, just store a weak reference
        if not hasattr(LazyLoadable, '_instances'):
            LazyLoadable._instances = weakref.WeakSet()
        LazyLoadable._instances.add(self)
        
    def configure_lazy_loading(self, **config) -> None:
        """Configure lazy loading behavior."""
        self._lazy_config.update(config)
        
    @classmethod
    def trigger_memory_pressure(cls, pressure_level: str) -> None:
        """Trigger memory pressure response for all instances."""
        if hasattr(cls, '_instances'):
            for instance in cls._instances:
                try:
                    instance.on_memory_pressure(pressure_level)
                except Exception as e:
                    logger.error(f"Error in memory pressure callback: {e}")
                    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB. Override in subclasses."""
        return 0.0
        
    def estimate_load_time_seconds(self) -> float:
        """Estimate time to load resource in seconds. Override in subclasses."""
        return 1.0
