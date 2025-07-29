"""Memory tracking and allocation service."""

import gc
import psutil
import threading
import weakref
from typing import Dict, Any, Optional, List
import logging

from src.abstractions.types.memory_types import MemoryPressureLevel, MemoryAllocation

logger = logging.getLogger(__name__)


# MemoryPressureLevel moved to abstractions.types.memory_types


# MemoryAllocation moved to abstractions.types.memory_types


class MemoryTrackerService:
    """Service for tracking memory allocations and pressure."""
    
    def __init__(self):
        self._allocations: Dict[str, MemoryAllocation] = {}
        self._allocation_lock = threading.RLock()
        self._managed_objects = []  # Weak references to managed objects
        
        # Configuration
        self._config = {
            'warning_threshold_percent': 70.0,
            'high_threshold_percent': 80.0,
            'critical_threshold_percent': 90.0,
            'enable_gc_on_pressure': True
        }
    
    def allocate_memory(self, name: str, size_mb: float, owner: str = "unknown",
                       description: str = "") -> None:
        """Track a memory allocation."""
        import time
        
        with self._allocation_lock:
            allocation = MemoryAllocation(
                name=name,
                owner=owner,
                size_mb=size_mb,
                allocated_at=time.time(),
                description=description
            )
            
            self._allocations[name] = allocation
            logger.debug(f"Allocated {size_mb:.1f}MB for {name} (owner: {owner})")
    
    def release_memory(self, name: str) -> None:
        """Release a tracked memory allocation."""
        with self._allocation_lock:
            if name in self._allocations:
                allocation = self._allocations.pop(name)
                logger.debug(f"Released {allocation.size_mb:.1f}MB for {name}")
            else:
                logger.warning(f"Attempted to release unknown allocation: {name}")
    
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        # System memory info
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # Tracked allocations
        with self._allocation_lock:
            tracked_total = sum(alloc.size_mb for alloc in self._allocations.values())
            allocation_count = len(self._allocations)
        
        return {
            'system_total_gb': memory.total / (1024**3),
            'system_available_gb': memory.available / (1024**3),
            'system_used_percent': memory.percent,
            'process_rss_mb': process_memory.rss / (1024**2),
            'process_vms_mb': process_memory.vms / (1024**2),
            'tracked_allocations_mb': tracked_total,
            'allocation_count': allocation_count,
            'managed_objects': len(self._managed_objects)
        }
    
    def get_memory_pressure_level(self) -> MemoryPressureLevel:
        """Get current memory pressure level."""
        memory_info = self.get_current_memory_usage()
        usage_percent = memory_info['system_used_percent']
        
        return self._get_pressure_level(usage_percent)
    
    def register_managed_object(self, obj: Any) -> None:
        """Register an object for memory management."""
        self._managed_objects.append(weakref.ref(obj))
    
    def trigger_cleanup(self, force_gc: bool = True) -> float:
        """Trigger memory cleanup and return freed memory."""
        initial_usage = self.get_current_memory_usage()['process_rss_mb']
        
        # Clean up dead weak references
        self._managed_objects = [ref for ref in self._managed_objects if ref() is not None]
        
        # Force garbage collection if enabled
        if force_gc and self._config['enable_gc_on_pressure']:
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
        
        final_usage = self.get_current_memory_usage()['process_rss_mb']
        freed_mb = initial_usage - final_usage
        
        if freed_mb > 0:
            logger.info(f"Memory cleanup freed {freed_mb:.1f}MB")
        
        return freed_mb
    
    def get_allocations_by_owner(self) -> Dict[str, List[MemoryAllocation]]:
        """Get allocations grouped by owner."""
        with self._allocation_lock:
            by_owner = {}
            for allocation in self._allocations.values():
                owner = allocation.owner
                if owner not in by_owner:
                    by_owner[owner] = []
                by_owner[owner].append(allocation)
            
            return by_owner
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        memory_info = self.get_current_memory_usage()
        pressure_level = self.get_memory_pressure_level()
        
        # Allocation statistics
        with self._allocation_lock:
            allocations = list(self._allocations.values())
        
        allocation_stats = {
            'total_tracked_mb': sum(alloc.size_mb for alloc in allocations),
            'count': len(allocations),
            'by_owner': self._group_allocations_by_owner(allocations)
        }
        
        return {
            'memory_info': memory_info,
            'pressure_level': pressure_level.value,
            'allocations': allocation_stats,
            'config': self._config.copy()
        }
    
    def configure(self, **config) -> None:
        """Update configuration."""
        self._config.update(config)
        logger.debug(f"Updated memory tracker config: {config}")
    
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
    
    def _group_allocations_by_owner(self, allocations: List[MemoryAllocation]) -> Dict[str, float]:
        """Group allocations by owner and sum sizes."""
        by_owner = {}
        for allocation in allocations:
            owner = allocation.owner
            if owner not in by_owner:
                by_owner[owner] = 0.0
            by_owner[owner] += allocation.size_mb
        
        return by_owner