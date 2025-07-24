# src/base/memory_manager.py
"""Centralized memory management with dynamic chunk sizing and pressure handling."""

import gc
import psutil
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import weakref
from contextlib import contextmanager

from .memory_tracker import get_memory_tracker, MemorySnapshot
from ..config.processing_config import ProcessingConfig, ChunkInfo

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    NORMAL = "normal"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryAllocation:
    """Track memory allocations."""
    name: str
    size_mb: float
    timestamp: float
    owner: Optional[str] = None
    can_release: bool = True
    priority: int = 0  # Higher priority = keep longer


class MemoryManager:
    """
    Centralized memory management with dynamic chunk sizing and pressure handling.
    
    Integrates with existing MemoryTracker for monitoring while adding:
    - Dynamic chunk size calculation
    - Memory allocation tracking
    - Pressure response coordination
    - Integration with ProcessingConfig
    """
    
    _instance: Optional['MemoryManager'] = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        """Initialize memory manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            
            # Use existing memory tracker
            self._memory_tracker = get_memory_tracker()
            
            # Memory allocations
            self._allocations: Dict[str, MemoryAllocation] = {}
            self._allocation_lock = threading.RLock()
            
            # Pressure callbacks
            self._pressure_callbacks: List[Callable[[MemoryPressureLevel], None]] = []
            
            # Configuration
            self._config = {
                'target_memory_usage_percent': 70.0,
                'warning_threshold_percent': 70.0,
                'high_threshold_percent': 85.0,
                'critical_threshold_percent': 95.0,
                'min_chunk_size': 100,
                'max_chunk_size': 10000,
                'gc_threshold_mb': 500,
                'check_interval_seconds': 5.0
            }
            
            # Monitoring state
            self._last_gc_time = time.time()
            self._gc_count = 0
            
            # Weak references to managed objects
            self._managed_objects = weakref.WeakSet()
            
            # Register with memory tracker for pressure callbacks
            self._memory_tracker.add_pressure_callback(self._handle_memory_pressure)
            
            logger.info("Memory manager initialized")
    
    def configure(self, **config) -> None:
        """Update configuration."""
        self._config.update(config)
    
    def register_pressure_callback(self, callback: Callable[[MemoryPressureLevel], None]) -> None:
        """Register callback for memory pressure events."""
        self._pressure_callbacks.append(callback)
    
    def register_managed_object(self, obj: Any) -> None:
        """Register an object for memory management."""
        self._managed_objects.add(obj)
    
    def allocate_memory(self, 
                       name: str, 
                       size_mb: float,
                       owner: Optional[str] = None,
                       can_release: bool = True,
                       priority: int = 0) -> bool:
        """
        Track a memory allocation.
        
        Args:
            name: Allocation name
            size_mb: Size in MB
            owner: Owner identifier
            can_release: Whether this can be released under pressure
            priority: Priority (higher = keep longer)
            
        Returns:
            True if allocation successful
        """
        with self._allocation_lock:
            # Check if allocation would exceed limits
            current_usage = self.get_current_memory_usage()
            if current_usage['usage_percent'] > self._config['critical_threshold_percent']:
                logger.warning(f"Cannot allocate {size_mb}MB for {name}: critical memory pressure")
                return False
            
            allocation = MemoryAllocation(
                name=name,
                size_mb=size_mb,
                timestamp=time.time(),
                owner=owner,
                can_release=can_release,
                priority=priority
            )
            
            self._allocations[name] = allocation
            logger.debug(f"Allocated {size_mb}MB for {name}")
            return True
    
    def release_memory(self, name: str) -> None:
        """Release a tracked memory allocation."""
        with self._allocation_lock:
            if name in self._allocations:
                allocation = self._allocations.pop(name)
                logger.debug(f"Released {allocation.size_mb}MB from {name}")
    
    def calculate_optimal_chunk_size(self, 
                                   data_size_mb: float,
                                   dtype_size: int = 4,
                                   processing_overhead: float = 1.5) -> int:
        """
        Calculate optimal chunk size based on available memory.
        
        Args:
            data_size_mb: Total data size in MB
            dtype_size: Size of each element in bytes
            processing_overhead: Multiplier for processing memory overhead
            
        Returns:
            Optimal chunk size (number of elements)
        """
        # Get current memory state from tracker
        snapshot = self._memory_tracker.get_current_snapshot()
        available_mb = snapshot.system_available_mb
        
        # Target usage
        target_percent = self._config['target_memory_usage_percent']
        target_mb = available_mb * (target_percent / 100.0)
        
        # Account for existing allocations
        allocated_mb = sum(a.size_mb for a in self._allocations.values())
        usable_mb = max(10, target_mb - allocated_mb)
        
        # Calculate chunk size
        chunk_memory_mb = usable_mb / processing_overhead
        elements_per_mb = (1024 * 1024) / dtype_size
        chunk_size = int(chunk_memory_mb * elements_per_mb)
        
        # Apply bounds
        min_chunk = self._config['min_chunk_size']
        max_chunk = self._config['max_chunk_size']
        chunk_size = max(min_chunk, min(chunk_size, max_chunk))
        
        logger.debug(f"Calculated chunk size: {chunk_size} elements "
                    f"(available: {available_mb:.1f}MB, target: {chunk_memory_mb:.1f}MB)")
        
        return chunk_size
    
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        snapshot = self._memory_tracker.get_current_snapshot()
        
        # Calculate tracked allocations
        with self._allocation_lock:
            tracked_mb = sum(a.size_mb for a in self._allocations.values())
        
        # Determine pressure level
        pressure_level = self._get_pressure_level(snapshot.system_used_percentage)
        
        return {
            'heap_mb': snapshot.heap_memory_mb,
            'virtual_mb': snapshot.virtual_memory_mb,
            'available_mb': snapshot.system_available_mb,
            'total_mb': snapshot.system_total_mb,
            'usage_percent': snapshot.system_used_percentage,
            'tracked_allocations_mb': tracked_mb,
            'pressure_level': pressure_level
        }
    
    def get_memory_pressure_level(self) -> MemoryPressureLevel:
        """Get current memory pressure level."""
        usage = self.get_current_memory_usage()
        return usage['pressure_level']
    
    def trigger_cleanup(self, force_gc: bool = True) -> float:
        """
        Trigger memory cleanup.
        
        Args:
            force_gc: Whether to force garbage collection
            
        Returns:
            Amount of memory freed in MB
        """
        logger.info("Triggering memory cleanup")
        
        # Use memory tracker's garbage collection method
        freed_mb = self._memory_tracker.force_garbage_collection() if force_gc else 0
        
        # Clear caches in managed objects
        for obj in self._managed_objects:
            if hasattr(obj, 'clear_cache'):
                try:
                    obj.clear_cache()
                except Exception as e:
                    logger.error(f"Error clearing cache: {e}")
            
            if hasattr(obj, 'on_memory_pressure'):
                try:
                    obj.on_memory_pressure('high')
                except Exception as e:
                    logger.error(f"Error in memory pressure callback: {e}")
        
        # Release low-priority allocations
        with self._allocation_lock:
            releasable = [
                (name, alloc) for name, alloc in self._allocations.items()
                if alloc.can_release
            ]
            # Sort by priority (lower first) and age (older first)
            releasable.sort(key=lambda x: (x[1].priority, x[1].timestamp))
            
            # Release bottom 25% of releasable allocations
            release_count = max(1, len(releasable) // 4)
            for name, _ in releasable[:release_count]:
                self.release_memory(name)
        
        self._gc_count += 1
        self._last_gc_time = time.time()
        
        logger.info(f"Memory cleanup freed {freed_mb:.1f}MB")
        return freed_mb
    
    def create_processing_config(self, 
                               data_size_mb: float,
                               operation: str = "processing") -> ProcessingConfig:
        """
        Create ProcessingConfig with optimal settings.
        
        Args:
            data_size_mb: Size of data to process
            operation: Type of operation
            
        Returns:
            Configured ProcessingConfig
        """
        # Calculate optimal chunk size
        chunk_size = self.calculate_optimal_chunk_size(data_size_mb)
        
        # Get current memory state
        memory_info = self.get_current_memory_usage()
        pressure_level = memory_info['pressure_level']
        
        # Adjust based on pressure
        if pressure_level == MemoryPressureLevel.CRITICAL:
            chunk_size = self._config['min_chunk_size']
            enable_chunking = True
            memory_limit_mb = memory_info['available_mb'] * 0.3
        elif pressure_level == MemoryPressureLevel.HIGH:
            chunk_size = min(chunk_size, 500)
            enable_chunking = True
            memory_limit_mb = memory_info['available_mb'] * 0.5
        else:
            enable_chunking = data_size_mb > 100  # Enable for large datasets
            memory_limit_mb = memory_info['available_mb'] * 0.7
        
        config = ProcessingConfig(
            chunk_size=chunk_size,
            memory_limit_mb=memory_limit_mb,
            enable_chunking=enable_chunking,
            checkpoint_interval=max(1, chunk_size // 100),
            use_memory_mapping=pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL],
            retry_on_memory_error=True,
            min_chunk_size=self._config['min_chunk_size']
        )
        
        # Adjust for memory pressure
        if pressure_level != MemoryPressureLevel.NORMAL:
            config.adjust_for_memory_pressure(memory_info['available_mb'])
        
        return config
    
    @contextmanager
    def memory_context(self, 
                      name: str,
                      estimated_mb: float,
                      owner: Optional[str] = None):
        """
        Context manager for memory allocation tracking.
        
        Args:
            name: Allocation name
            estimated_mb: Estimated size in MB
            owner: Owner identifier
        """
        allocated = self.allocate_memory(name, estimated_mb, owner)
        try:
            yield allocated
        finally:
            if allocated:
                self.release_memory(name)
    
    def monitor_operation(self, 
                         operation_name: str,
                         data_size_mb: float) -> 'MemoryMonitor':
        """
        Create a memory monitor for an operation.
        
        Args:
            operation_name: Name of operation
            data_size_mb: Size of data being processed
            
        Returns:
            MemoryMonitor instance
        """
        return MemoryMonitor(self, operation_name, data_size_mb)
    
    def _get_pressure_level(self, usage_percent: float) -> MemoryPressureLevel:
        """Determine pressure level from usage percentage."""
        if usage_percent >= self._config['critical_threshold_percent']:
            return MemoryPressureLevel.CRITICAL
        elif usage_percent >= self._config['high_threshold_percent']:
            return MemoryPressureLevel.HIGH
        elif usage_percent >= self._config['warning_threshold_percent']:
            return MemoryPressureLevel.WARNING
        else:
            return MemoryPressureLevel.NORMAL
    
    def _handle_memory_pressure(self, pressure_level_str: str) -> None:
        """Handle memory pressure from memory tracker."""
        # Convert string to enum
        level_map = {
            'low': MemoryPressureLevel.NORMAL,
            'medium': MemoryPressureLevel.WARNING,
            'high': MemoryPressureLevel.HIGH,
            'critical': MemoryPressureLevel.CRITICAL
        }
        
        pressure_level = level_map.get(pressure_level_str, MemoryPressureLevel.WARNING)
        
        logger.info(f"Memory pressure detected: {pressure_level.value}")
        
        # Notify callbacks
        for callback in self._pressure_callbacks:
            try:
                callback(pressure_level)
            except Exception as e:
                logger.error(f"Error in pressure callback: {e}")
        
        # Auto cleanup on high pressure
        if pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            self.trigger_cleanup(force_gc=pressure_level == MemoryPressureLevel.CRITICAL)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        with self._allocation_lock:
            allocations = list(self._allocations.values())
        
        # Get tracker statistics
        tracker_summary = self._memory_tracker.get_memory_summary()
        
        return {
            'current_usage': self.get_current_memory_usage(),
            'allocations': {
                'count': len(allocations),
                'total_mb': sum(a.size_mb for a in allocations),
                'by_owner': self._group_allocations_by_owner(allocations)
            },
            'gc_stats': {
                'gc_count': self._gc_count,
                'last_gc_time': self._last_gc_time,
                'time_since_gc': time.time() - self._last_gc_time
            },
            'managed_objects': len(self._managed_objects),
            'tracker_summary': tracker_summary,
            'config': self._config
        }
    
    def _group_allocations_by_owner(self, allocations: List[MemoryAllocation]) -> Dict[str, float]:
        """Group allocations by owner."""
        by_owner: Dict[str, float] = {}
        for alloc in allocations:
            owner = alloc.owner or 'unknown'
            by_owner[owner] = by_owner.get(owner, 0) + alloc.size_mb
        return by_owner


class MemoryMonitor:
    """Monitor memory usage for a specific operation."""
    
    def __init__(self, manager: MemoryManager, operation_name: str, data_size_mb: float):
        self.manager = manager
        self.operation_name = operation_name
        self.data_size_mb = data_size_mb
        self.start_time = time.time()
        self.peak_usage_mb = 0.0
        self._monitoring = True
        
        # Use memory tracker's prediction
        self.prediction = manager._memory_tracker.predict_memory_usage(
            data_size_mb, 
            operation_name.lower(),
            {'source': 'memory_monitor'}
        )
        
    def update_progress(self, progress_percent: float, current_chunk: Optional[int] = None) -> None:
        """Update operation progress."""
        current_usage = self.manager.get_current_memory_usage()
        self.peak_usage_mb = max(self.peak_usage_mb, current_usage['heap_mb'])
        
        # Log if memory usage is concerning
        if current_usage['pressure_level'] in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            logger.warning(
                f"{self.operation_name}: High memory usage at {progress_percent:.1f}% "
                f"(current: {current_usage['heap_mb']:.1f}MB, peak: {self.peak_usage_mb:.1f}MB)"
            )
    
    def complete(self) -> Dict[str, Any]:
        """Mark operation as complete and return statistics."""
        self._monitoring = False
        duration = time.time() - self.start_time
        
        return {
            'operation': self.operation_name,
            'duration_seconds': duration,
            'data_size_mb': self.data_size_mb,
            'peak_usage_mb': self.peak_usage_mb,
            'mb_per_second': self.data_size_mb / duration if duration > 0 else 0,
            'prediction_accuracy': abs(self.peak_usage_mb - self.prediction.predicted_peak_mb) / self.prediction.predicted_peak_mb if self.prediction.predicted_peak_mb > 0 else 0
        }


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager