"""Machine learning module for biodiversity analysis."""

# Since Julia handles all preprocessing, we don't import preprocessing modules
# This avoids polars and other preprocessing dependencies

# Import CV strategies (if available)
try:
    from .validation.spatial_cv import (
        SpatialBlockCV,
        SpatialBufferCV,
        EnvironmentalBlockCV
    )
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

__all__ = []

if CV_AVAILABLE:
    __all__.extend([
        'SpatialBlockCV',
        'SpatialBufferCV',
        'EnvironmentalBlockCV'
    ])