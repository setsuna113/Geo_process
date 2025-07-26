"""Foundation types - core data structures with no dependencies."""

from .checkpoint_types import (
    CheckpointLevel,
    CheckpointStatus,
    CheckpointData,
    CheckpointMetadata,
    StorageConfig,
    CheckpointError
)

__all__ = [
    'CheckpointLevel',
    'CheckpointStatus', 
    'CheckpointData',
    'CheckpointMetadata',
    'StorageConfig',
    'CheckpointError'
]