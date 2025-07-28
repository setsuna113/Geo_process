"""Facade for memory services - maintains backward compatibility.

This facade provides the same interface as the original MemoryManager
but delegates to focused, single-responsibility services.
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from dataclasses import dataclass

from .services import (
    MemoryTrackerService, 
    MemoryOptimizerService, 
    MemoryPressureService,
    MemoryPressureLevel,
    MemoryAllocation
)
from .memory_tracker import get_memory_tracker
from ..config.processing_config import ProcessingConfig

logger = logging.getLogger(__name__)


@dataclass
class MemoryOperationContext:
    """Context for memory operations."""
    manager: 'MemoryService'
    operation_name: str
    data_size_mb: float
    start_time: float
    allocation_name: str
    
    def update_progress(self, progress_percent: float, current_chunk: Optional[int] = None) -> None:
        """Update operation progress."""
        metadata = {
            'progress_percent': progress_percent,
            'elapsed_time': time.time() - self.start_time
        }
        if current_chunk is not None:
            metadata['current_chunk'] = current_chunk
        
        # Update allocation metadata
        with self.manager._tracker._allocation_lock:
            if self.allocation_name in self.manager._tracker._allocations:
                self.manager._tracker._allocations[self.allocation_name].metadata.update(metadata)
    
    def complete(self) -> Dict[str, Any]:
        """Complete the operation and return statistics."""
        elapsed_time = time.time() - self.start_time
        
        # Release memory allocation
        self.manager._tracker.release_memory(self.allocation_name)
        
        return {
            'operation_name': self.operation_name,
            'data_size_mb': self.data_size_mb,
            'elapsed_time': elapsed_time,
            'completed_at': time.time()
        }


class MemoryService:
    """
    Facade for decomposed memory services.
    
    This class provides the same interface as the original MemoryManager
    but delegates to focused, single-responsibility services.
    
    The original manager (482 lines) has been decomposed into:
    - MemoryTrackerService: Memory allocation tracking (148 lines)
    - MemoryOptimizerService: Memory optimization and chunking (113 lines)
    - MemoryPressureService: Pressure monitoring and callbacks (140 lines)
    """
    
    _instance: Optional['MemoryService'] = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        """Initialize memory service."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            
            # Initialize services
            self._tracker = MemoryTrackerService()
            self._optimizer = MemoryOptimizerService()
            self._pressure = MemoryPressureService(self._tracker)
            
            # Use existing memory tracker for compatibility
            self._memory_tracker = get_memory_tracker()
            
            # Start pressure monitoring
            self._pressure.start_monitoring()
            
            logger.info("MemoryService initialized with decomposed services")
    
    def configure(self, **config) -> None:
        """Configure memory services."""
        # Distribute config to appropriate services
        if any(key.startswith('tracker_') for key in config.keys()):
            tracker_config = {k[8:]: v for k, v in config.items() if k.startswith('tracker_')}
            self._tracker.configure(**tracker_config)
        
        if any(key.startswith('optimizer_') for key in config.keys()):
            optimizer_config = {k[10:]: v for k, v in config.items() if k.startswith('optimizer_')}
            self._optimizer.configure(**optimizer_config)
        
        if any(key.startswith('pressure_') for key in config.keys()):
            pressure_config = {k[9:]: v for k, v in config.items() if k.startswith('pressure_')}
            self._pressure.configure(**pressure_config)
        
        # Pass all config to all services for backward compatibility
        self._tracker.configure(**config)
        self._optimizer.configure(**config)
        self._pressure.configure(**config)
    
    def register_pressure_callback(self, callback: Callable[[MemoryPressureLevel], None]) -> None:
        """Register a callback for memory pressure events."""
        self._pressure.register_pressure_callback(callback)
    
    def register_managed_object(self, obj: Any) -> None:
        """Register an object for memory management."""
        self._tracker.register_managed_object(obj)
    
    def allocate_memory(self, name: str, size_mb: float, owner: str = "unknown",
                       description: str = "", track_allocator: bool = True) -> None:
        """Track a memory allocation."""
        self._tracker.allocate_memory(name, size_mb, owner, description)
    
    def release_memory(self, name: str) -> None:
        """Release a tracked memory allocation."""
        self._tracker.release_memory(name)
    
    def calculate_optimal_chunk_size(self, total_items: int, 
                                   memory_per_item_bytes: float,
                                   available_memory_mb: Optional[float] = None,
                                   target_memory_mb: Optional[float] = None) -> int:
        """Calculate optimal chunk size for processing."""
        config = self._optimizer.calculate_optimal_chunk_size(
            total_items, memory_per_item_bytes, available_memory_mb, target_memory_mb
        )
        return config.chunk_size
    
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        return self._tracker.get_current_memory_usage()
    
    def get_memory_pressure_level(self) -> MemoryPressureLevel:
        """Get current memory pressure level."""
        return self._tracker.get_memory_pressure_level()
    
    def trigger_cleanup(self, force_gc: bool = True) -> float:
        """Trigger memory cleanup and return freed memory."""
        return self._tracker.trigger_cleanup(force_gc)
    
    def create_processing_config(self, total_items: int, 
                               memory_per_item_bytes: float,
                               processing_name: str = "processing",
                               target_memory_mb: Optional[float] = None) -> ProcessingConfig:
        """Create a processing configuration with optimal chunking."""
        return self._optimizer.create_processing_config(
            total_items, memory_per_item_bytes, processing_name, target_memory_mb
        )
    
    @contextmanager
    def memory_context(self, operation_name: str, estimated_size_mb: float):
        """Context manager for memory operations."""
        allocation_name = f"context_{operation_name}_{int(time.time())}"
        
        # Allocate memory
        self.allocate_memory(allocation_name, estimated_size_mb, "memory_context", operation_name)
        
        context = MemoryOperationContext(
            manager=self,
            operation_name=operation_name,
            data_size_mb=estimated_size_mb,
            start_time=time.time(),
            allocation_name=allocation_name
        )
        
        try:
            yield context
        finally:
            # Ensure memory is released
            self.release_memory(allocation_name)
    
    def monitor_operation(self, operation_name: str, data_size_mb: float) -> MemoryOperationContext:
        """Create a memory operation monitor."""
        allocation_name = f"monitor_{operation_name}_{int(time.time())}"
        
        self.allocate_memory(allocation_name, data_size_mb, "operation_monitor", operation_name)
        
        return MemoryOperationContext(
            manager=self,
            operation_name=operation_name,
            data_size_mb=data_size_mb,
            start_time=time.time(),
            allocation_name=allocation_name
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        tracker_stats = self._tracker.get_statistics()
        pressure_stats = self._pressure.get_pressure_stats()
        optimizer_config = self._optimizer.get_config()
        
        return {
            'tracker': tracker_stats,
            'pressure': pressure_stats,
            'optimizer_config': optimizer_config,
            'service_info': {
                'services_active': True,
                'monitoring_active': pressure_stats['monitoring_active']
            }
        }
    
    # Backward compatibility methods
    def _get_pressure_level(self, usage_percent: float) -> MemoryPressureLevel:
        """Get pressure level from usage percentage (backward compatibility)."""
        return self._tracker._get_pressure_level(usage_percent)
    
    def _handle_memory_pressure(self, pressure_level_str: str) -> None:
        """Handle memory pressure (backward compatibility)."""
        try:
            pressure_level = MemoryPressureLevel(pressure_level_str)
            self._pressure._handle_memory_pressure(pressure_level)
        except ValueError:
            logger.warning(f"Invalid pressure level: {pressure_level_str}")
    
    def _group_allocations_by_owner(self, allocations: List[MemoryAllocation]) -> Dict[str, float]:
        """Group allocations by owner (backward compatibility)."""
        return self._tracker._group_allocations_by_owner(allocations)
    
    def __del__(self):
        """Cleanup when service is destroyed."""
        try:
            if hasattr(self, '_pressure'):
                self._pressure.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping memory service: {e}")


# Singleton instance getter for backward compatibility
def get_memory_manager() -> MemoryService:
    """Get the global memory service instance."""
    return MemoryService()


# Backward compatibility alias
MemoryManager = MemoryService