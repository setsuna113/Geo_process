"""
Biodiversity Analysis Utilities

General utility modules for biodiversity analysis.
"""

from .import_helpers import safe_import, safe_import_class, ImportContext

# Try to import other modules but don't fail if they don't exist
try:
    from .data_inspector import DataInspector
except ImportError:
    DataInspector = None

try:
    from .enhanced_visualizer import EnhancedVisualizer
except ImportError:
    EnhancedVisualizer = None

__all__ = ['safe_import', 'safe_import_class', 'ImportContext']

# Add available classes to __all__
if DataInspector:
    __all__.append('DataInspector')
if EnhancedVisualizer:
    __all__.append('EnhancedVisualizer')