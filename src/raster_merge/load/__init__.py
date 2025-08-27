"""
Load module - separates path-based and memory-based loading.

Following FarmVibes pattern:
- path_loader: For operations that work with file paths (merge, VRT)
- memory_loader: For operations that need arrays (gradient, clip)

NO ORCHESTRATION - operations know what they need and call the right loader.
"""

from .path_loader import (
    validate_paths,
    get_raster_info,
    check_alignment,
    prepare_for_merge
)

from .memory_loader import (
    load_to_array,
    load_single_raster,
    load_with_callback,
    estimate_memory_usage
)

__all__ = [
    # Path operations (no loading)
    'validate_paths',
    'get_raster_info', 
    'check_alignment',
    'prepare_for_merge',
    
    # Memory operations (when needed)
    'load_to_array',
    'load_single_raster',
    'load_with_callback',
    'estimate_memory_usage'
]