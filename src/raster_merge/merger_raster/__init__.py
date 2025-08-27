"""
Merger module implementing FarmVibes' VRT-based merging.

No fallbacks - all errors are raised.
"""

from .vrt_merger import VRTMerger, merge_rasters_vrt

__all__ = [
    "VRTMerger",
    "merge_rasters_vrt"
]