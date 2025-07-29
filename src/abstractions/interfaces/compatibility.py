"""Compatibility adapters for smooth transition to foundation layer."""

import warnings
from typing import Any, Optional, Dict, Tuple
from .processor import IProcessor


class BaseProcessor:
    """Compatibility layer for existing BaseProcessor implementations.
    
    This class provides backward compatibility during the transition to the
    new foundation architecture. It will be removed in Phase 6.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BaseProcessor is deprecated. Use foundations.interfaces.IProcessor "
            "for interfaces or infrastructure.processors.EnhancedProcessor for "
            "implementation. This compatibility layer will be removed in Phase 6.",
            DeprecationWarning,
            stacklevel=2
        )
        # Initialize with minimal functionality for compatibility
        self._config = kwargs.get('config', {})
        self._initialized = True
    
    def process_single(self, item: Any) -> Any:
        """Compatibility method - subclasses should override."""
        raise NotImplementedError("Subclasses must implement process_single")
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Compatibility method - basic validation."""
        if item is None:
            return False, "Input cannot be None"
        return True, None