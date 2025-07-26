# src/base/processor_compat.py
"""Backward compatibility layer for BaseProcessor."""

import warnings
import sys
from pathlib import Path

# Import the original processor classes for backward compatibility
# This maintains the existing interface while the architecture is being refactored
try:
    from .processor import BaseProcessor as OriginalBaseProcessor
    from .processor import ProcessingResult, LegacyMemoryTracker
    
    # Re-export everything for backward compatibility
    BaseProcessor = OriginalBaseProcessor
    
    # Add deprecation warning for new usage
    class DeprecatedBaseProcessor(OriginalBaseProcessor):
        """Deprecated wrapper that issues warnings for new usage."""
        
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "BaseProcessor from src.base.processor is deprecated. "
                "Use src.infrastructure.processors.base_processor.EnhancedBaseProcessor "
                "for new code. This compatibility layer will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )
            super().__init__(*args, **kwargs)
    
    # For new code, recommend the enhanced version
    def create_processor(*args, **kwargs):
        """Factory function to create processors with deprecation guidance."""
        warnings.warn(
            "Consider using src.infrastructure.processors.base_processor.EnhancedBaseProcessor "
            "for new processors to follow the clean architecture pattern.",
            FutureWarning,
            stacklevel=2
        )
        return BaseProcessor(*args, **kwargs)
    
except ImportError as e:
    # Fallback if original processor can't be imported
    print(f"Warning: Could not import original BaseProcessor: {e}")
    
    # Create a minimal fallback
    class BaseProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "BaseProcessor could not be imported. "
                "Use src.infrastructure.processors.base_processor.EnhancedBaseProcessor instead."
            )
    
    class ProcessingResult:
        pass
    
    class LegacyMemoryTracker:
        pass

# Export everything for backward compatibility
__all__ = ['BaseProcessor', 'ProcessingResult', 'LegacyMemoryTracker', 'create_processor']