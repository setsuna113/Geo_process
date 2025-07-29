"""
Base classes for the biodiversity geoprocessing framework.

This module provides abstract base classes that define the interfaces and common
functionality for all processing components in the pipeline:

- BaseProcessor: Memory-tracked, batch-capable processing with progress callbacks
- BaseGrid: Spatial grid generation and cell management  
- BaseFeature: Feature extraction from spatial data
- BaseDataset: Data loading and validation with filtering capabilities

Key Features:
- ✅ Abstract method enforcement for consistent interfaces
- ✅ Memory tracking and performance monitoring
- ✅ Progress callbacks for async/parallel processing
- ✅ Database integration ready
- ✅ Configuration-driven with defaults
- ✅ Type-safe with full annotations
- ✅ Extensible through inheritance

Compatibility:
- ✅ Core module integration (registry, builder)
- ✅ Config module integration (settings, defaults)
- ✅ Database module integration (persistence)
- ✅ Multiprocessing ready (stateless design)
- ✅ Future module expansion ready

Usage Example:
    from src.base import BaseProcessor
    from src.core import component_registry
    
    @component_registry.processors.register_decorator()
    class SpeciesProcessor(BaseProcessor):
        def process_single(self, species_data):
            # Process single species record
            return processed_result
            
        def validate_input(self, item):
            return True, None
            
    processor = SpeciesProcessor(batch_size=1000, max_workers=4)
    results = processor.process_batch(species_records)
"""

from .processor import BaseProcessor, ProcessingResult, LegacyMemoryTracker as MemoryTracker
from .grid import BaseGrid, GridCell
from .feature import BaseFeature, FeatureResult
from .dataset import BaseDataset, DatasetInfo

# Checkpoint system abstractions
from src.foundations.types.checkpoint_types import (
    CheckpointData, CheckpointMetadata, CheckpointLevel, CheckpointStatus,
    CheckpointFilter, StorageConfig, StorageBackend,
    CheckpointError, CheckpointNotFoundError, CheckpointCorruptedError,
    CheckpointValidationError, StorageBackendError
)
from .checkpoint import (
    CheckpointStorage, CheckpointValidator, DefaultCheckpointValidator,
    CheckpointCompressor, NoopCompressor, CheckpointMetrics,
    generate_checkpoint_id, parse_checkpoint_id, checkpoint_operation_timer
)
from .checkpointable import (
    CheckpointPolicy, ResumableProcess, CheckpointableProcess,
    SimpleCheckpointableProcess, make_checkpointable, checkpoint_on_interval
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Jason"

# Public API
__all__ = [
    # Processor classes
    'BaseProcessor', 
    'ProcessingResult', 
    'MemoryTracker',
    
    # Grid classes
    'BaseGrid', 
    'GridCell',
    
    # Feature classes  
    'BaseFeature', 
    'FeatureResult',
    
    # Dataset classes
    'BaseDataset', 
    'DatasetInfo',
    
    # Checkpoint data types
    'CheckpointData',
    'CheckpointMetadata', 
    'CheckpointLevel',
    'CheckpointStatus',
    'CheckpointFilter',
    'StorageConfig',
    'StorageBackend',
    
    # Checkpoint exceptions
    'CheckpointError',
    'CheckpointNotFoundError',
    'CheckpointCorruptedError', 
    'CheckpointValidationError',
    'StorageBackendError',
    
    # Checkpoint abstractions
    'CheckpointStorage',
    'CheckpointValidator',
    'DefaultCheckpointValidator',
    'CheckpointCompressor',
    'NoopCompressor',
    'CheckpointMetrics',
    
    # Checkpoint utilities
    'generate_checkpoint_id',
    'parse_checkpoint_id',
    'checkpoint_operation_timer',
    
    # Checkpointable process classes
    'CheckpointPolicy',
    'ResumableProcess',
    'CheckpointableProcess',
    'SimpleCheckpointableProcess',
    
    # Checkpoint decorators
    'make_checkpointable',
    'checkpoint_on_interval'
]

def get_base_class_info():
    """Get information about all base classes."""
    import inspect
    
    classes = [
        ('BaseProcessor', BaseProcessor),
        ('BaseGrid', BaseGrid),
        ('BaseFeature', BaseFeature), 
        ('BaseDataset', BaseDataset),
        ('CheckpointStorage', CheckpointStorage),
        ('CheckpointValidator', CheckpointValidator),
        ('CheckpointCompressor', CheckpointCompressor),
        ('ResumableProcess', ResumableProcess)
    ]
    
    info = {}
    for name, cls in classes:
        info[name] = {
            'abstract_methods': list(cls.__abstractmethods__) if hasattr(cls, '__abstractmethods__') else [],
            'module': cls.__module__,
            'doc': cls.__doc__
        }
    
    return info

# Module initialization check
def _validate_base_module():
    """Validate base module integrity."""
    try:
        # Check all core imports work
        assert BaseProcessor is not None
        assert BaseGrid is not None  
        assert BaseFeature is not None
        assert BaseDataset is not None
        
        # Check checkpoint imports work
        assert CheckpointData is not None
        assert CheckpointStorage is not None
        assert CheckpointableProcess is not None
        assert CheckpointPolicy is not None
        
        # Check abstract methods are defined for core classes
        for cls in [BaseProcessor, BaseGrid, BaseFeature, BaseDataset]:
            assert hasattr(cls, '__abstractmethods__')
            assert len(cls.__abstractmethods__) > 0
        
        # Check abstract methods are defined for checkpoint classes    
        for cls in [CheckpointStorage, CheckpointValidator, CheckpointCompressor, ResumableProcess]:
            assert hasattr(cls, '__abstractmethods__')
            assert len(cls.__abstractmethods__) > 0
            
        return True
    except Exception:
        return False

# Validate on import
if not _validate_base_module():
    import warnings
    warnings.warn("Base module validation failed", UserWarning)
