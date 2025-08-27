"""
Result types for statistical methods.

Following Linus: "Keep it simple, stupid."
No inheritance hierarchies, no abstract base classes, just a simple container.
"""

from typing import Dict, Any, Optional


class MethodResult:
    """Simple result container. That's it.
    
    Why a class instead of dict? Type checking. 
    Why not dataclass? This is simpler.
    """
    
    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        method: Optional[str] = None,
        warnings: Optional[list] = None
    ):
        """Create a result. No magic."""
        self.success = success
        self.data = data or {}
        self.error = error
        self.method = method
        self.warnings = warnings or []
    
    def __bool__(self):
        """Result is truthy if successful."""
        return self.success
    
    def __repr__(self):
        """Simple representation."""
        status = "✓" if self.success else "✗"
        return f"MethodResult({status} {self.method or 'unknown'})"