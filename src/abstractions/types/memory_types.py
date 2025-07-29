# src/abstractions/types/memory_types.py
"""Memory management type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    NORMAL = "normal"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryAllocation:
    """Track a memory allocation."""
    name: str
    size_mb: float
    timestamp: float  # When allocated
    owner: Optional[str] = None
    can_release: bool = True
    priority: int = 0  # Higher priority = keep longer
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryState:
    """Current memory state snapshot."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    pressure_level: MemoryPressureLevel
    allocations: Dict[str, MemoryAllocation] = field(default_factory=dict)