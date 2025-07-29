# src/abstractions/types/dataset_types.py
"""Dataset-related type definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Tuple


class DataType(Enum):
    """Supported data types for datasets."""
    RASTER = "raster"
    VECTOR = "vector"
    TABULAR = "tabular"
    POINT_CLOUD = "point_cloud"
    TIME_SERIES = "time_series"
    MIXED = "mixed"


@dataclass
class DatasetInfo:
    """Enhanced dataset metadata."""
    name: str
    source: str
    format: str
    size_mb: float
    record_count: int
    bounds: Optional[Tuple[float, float, float, float]]
    crs: Optional[str]
    metadata: Dict[str, Any]
    data_type: DataType
    resolution: Optional[Tuple[float, float]] = None  # For raster data
    tile_count: Optional[int] = None  # For tiled data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'source': self.source,
            'format': self.format,
            'size_mb': self.size_mb,
            'record_count': self.record_count,
            'bounds': self.bounds,
            'crs': self.crs,
            'metadata': self.metadata,
            'data_type': self.data_type.value,
            'resolution': self.resolution,
            'tile_count': self.tile_count
        }