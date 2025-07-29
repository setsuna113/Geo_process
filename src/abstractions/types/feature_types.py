# src/abstractions/types/feature_types.py
"""Feature extraction type definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Union
import numpy as np


class SourceType(Enum):
    """Types of data sources for feature extraction."""
    RASTER = "raster"
    VECTOR = "vector"
    TABULAR = "tabular"
    API = "api"
    COMPUTED = "computed"
    MIXED = "mixed"


@dataclass
class FeatureResult:
    """Standard feature extraction result."""
    feature_name: str
    feature_type: str
    value: Union[float, np.ndarray]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        value = self.value
        if isinstance(value, np.ndarray):
            value = value.tolist()
            
        return {
            'feature_name': self.feature_name,
            'feature_type': self.feature_type,
            'feature_value': value,
            'computation_metadata': self.metadata or {}
        }