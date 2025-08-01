# src/abstractions/types/__init__.py
"""Type definitions for the abstractions layer."""

# Checkpoint types
from .checkpoint_types import (
    CheckpointData, CheckpointMetadata, CheckpointLevel, CheckpointStatus,
    CheckpointFilter, StorageConfig, StorageBackend,
    CheckpointError, CheckpointNotFoundError, CheckpointCorruptedError,
    CheckpointValidationError, StorageBackendError
)

# Dataset types
from .dataset_types import DataType, DatasetInfo

# Memory types
from .memory_types import MemoryPressureLevel, MemoryAllocation, MemoryState

# Processing types
from .processing_types import (
    ProcessingStatus, ProcessingResult, ProcessorConfig, TileProgress
)

# Resampling types
from .resampling_types import (
    ResamplingMethod, AggregationMethod, ResamplingConfidence
)

# Feature types
from .feature_types import SourceType, FeatureResult

# Grid types
from .grid_types import GridCell

# Validation types
from .validation_types import (
    BoundsValidation, CoordinateValidation, ValueRangeValidation,
    SpatialIntegrityValidation, ValidationErrorType, ValidationContext
)

# Biodiversity types
from .biodiversity_types import (
    CoordinateSystem, SpatialData, BiodiversityData
)

__all__ = [
    # Checkpoint
    'CheckpointData', 'CheckpointMetadata', 'CheckpointLevel', 'CheckpointStatus',
    'CheckpointFilter', 'StorageConfig', 'StorageBackend',
    'CheckpointError', 'CheckpointNotFoundError', 'CheckpointCorruptedError',
    'CheckpointValidationError', 'StorageBackendError',
    
    # Dataset
    'DataType', 'DatasetInfo',
    
    # Memory
    'MemoryPressureLevel', 'MemoryAllocation', 'MemoryState',
    
    # Processing
    'ProcessingStatus', 'ProcessingResult', 'ProcessorConfig', 'TileProgress',
    
    # Resampling
    'ResamplingMethod', 'AggregationMethod', 'ResamplingConfidence',
    
    # Feature
    'SourceType', 'FeatureResult',
    
    # Grid
    'GridCell',
    
    # Validation
    'BoundsValidation', 'CoordinateValidation', 'ValueRangeValidation',
    'SpatialIntegrityValidation', 'ValidationErrorType', 'ValidationContext',
    
    # Biodiversity
    'CoordinateSystem', 'SpatialData', 'BiodiversityData'
]