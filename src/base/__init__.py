"""
Compatibility shim for base module.

All functionality has been moved to abstractions where it belongs.
This module now just re-exports from the correct locations for backward compatibility.
"""

# Re-export interfaces as "Base" classes for compatibility
from src.abstractions.interfaces.dataset import IDataset as BaseDataset
from src.abstractions.interfaces.analyzer import IAnalyzer as BaseAnalyzer
from src.abstractions.interfaces import IProcessor as BaseProcessor, IGrid as BaseGrid

# Re-export types from abstractions
from src.abstractions.types import (
    ProcessingResult, 
    GridCell, 
    DatasetInfo,
    DataType,
    CheckpointData,
    CheckpointMetadata,
    CheckpointLevel,
    CheckpointStatus,
    CheckpointFilter,
    StorageConfig,
    StorageBackend
)

# Feature-related types (if they exist in abstractions)
try:
    from src.abstractions.types import FeatureResult
except ImportError:
    FeatureResult = None

# Create placeholder for BaseFeature if needed
try:
    from src.abstractions.interfaces.feature import IFeature as BaseFeature
except ImportError:
    from abc import ABC, abstractmethod
    class BaseFeature(ABC):
        @abstractmethod
        def extract(self, data):
            pass

# Legacy MemoryTracker alias (if still needed)
class MemoryTracker:
    """Placeholder for legacy MemoryTracker - no longer used."""
    def __init__(self, *args, **kwargs):
        pass
    def track(self, *args, **kwargs):
        pass
    def get_usage(self):
        return 0

# Version info
__version__ = "2.0.0"  # Bumped to indicate major refactor
__author__ = "Jason"

# Public API - maintain compatibility
__all__ = [
    'BaseProcessor',
    'BaseGrid', 
    'BaseFeature',
    'BaseDataset',
    'BaseAnalyzer',
    'ProcessingResult',
    'GridCell',
    'DatasetInfo',
    'DataType',
    'MemoryTracker',
    'CheckpointData',
    'CheckpointMetadata',
    'CheckpointLevel',
    'CheckpointStatus',
    'CheckpointFilter',
    'StorageConfig',
    'StorageBackend'
]

# Add FeatureResult if it exists
if FeatureResult is not None:
    __all__.append('FeatureResult')