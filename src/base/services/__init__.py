"""Base memory services - domain-driven service architecture.

This module provides focused services to replace the MemoryManager
anti-pattern with proper domain-driven services:

- MemoryTrackerService: Memory allocation tracking and monitoring
- MemoryOptimizerService: Memory optimization and chunk size calculation
- MemoryPressureService: Memory pressure monitoring and callbacks
"""

from .memory_tracker_service import (
    MemoryTrackerService, 
    MemoryAllocation, 
    MemoryPressureLevel
)
from .memory_optimizer_service import (
    MemoryOptimizerService, 
    OptimalChunkConfig
)
from .memory_pressure_service import MemoryPressureService

__all__ = [
    'MemoryTrackerService',
    'MemoryAllocation',
    'MemoryPressureLevel',
    'MemoryOptimizerService', 
    'OptimalChunkConfig',
    'MemoryPressureService'
]