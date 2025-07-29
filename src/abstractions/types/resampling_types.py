# src/abstractions/types/resampling_types.py
"""Resampling-related type definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class ResamplingMethod(Enum):
    """Available resampling methods."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    AVERAGE = "average"
    MODE = "mode"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    MEDIAN = "median"
    Q1 = "q1"
    Q3 = "q3"


class AggregationMethod(Enum):
    """Available aggregation methods for downsampling."""
    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    MODE = "mode"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"


@dataclass
class ResamplingConfidence:
    """Confidence metrics for resampling operation."""
    overall_confidence: float  # 0.0 to 1.0
    spatial_confidence: float  # Based on resolution change
    data_confidence: float     # Based on data type compatibility
    method_confidence: float   # Based on method suitability
    
    # Quality metrics
    information_loss: float    # Estimated information loss (0.0 to 1.0)
    aliasing_risk: float      # Risk of aliasing artifacts (0.0 to 1.0)
    interpolation_accuracy: float  # Expected interpolation accuracy
    
    # Recommendations
    recommended_method: Optional[ResamplingMethod] = None
    warnings: List[str] = field(default_factory=list)