"""Foundation layer - pure abstractions and types with no src dependencies."""

# Re-export key interfaces and types for convenience
from .interfaces import IProcessor, IGrid, GridCell
from .types import CheckpointLevel, CheckpointStatus, CheckpointData
from .mixins import Cacheable, LazyLoadable, Tileable

__all__ = [
    'IProcessor',
    'IGrid', 
    'GridCell',
    'CheckpointLevel',
    'CheckpointStatus',
    'CheckpointData',
    'Cacheable',
    'LazyLoadable',
    'Tileable'
]