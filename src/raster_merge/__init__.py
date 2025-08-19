"""
Industrial-grade raster merging - simplified and correct.

No abstractions. No magic. Just functions that work.

Key Components:
1. pipeline.py - Unix-style functions (merge_rasters, clip_raster, etc.)
2. vrt_merger.py - VRT-based merging with proper batching
3. config.py - Simple configuration, one input format

Design Principles (Linus-approved):
- Functions over classes
- Paths in, paths out
- No hidden state
- No special cases
"""

from .config import RasterConfig
from .pipeline import (
    merge_rasters,
    clip_raster,
    compute_gradient,
    build_overviews,
    pipe,
    merge_and_export
)
# RasterData removed - not needed with functional approach
from .merger_raster.vrt_merger import VRTMerger

__all__ = [
    # Configuration
    'RasterConfig',
    
    # Core functions (Unix-style)
    'merge_rasters',
    'clip_raster', 
    'compute_gradient',
    'build_overviews',
    'pipe',
    'merge_and_export',
    
    # VRT merger for advanced use
    'VRTMerger',
]