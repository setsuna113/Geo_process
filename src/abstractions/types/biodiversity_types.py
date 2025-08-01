"""
Data type definitions for biodiversity analysis.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
from enum import Enum


class CoordinateSystem(Enum):
    """Supported coordinate systems."""
    WGS84 = "EPSG:4326"
    WEB_MERCATOR = "EPSG:3857"
    

@dataclass
class SpatialData:
    """
    Container for spatial data with coordinates.
    """
    features: np.ndarray  # Shape: (n_samples, n_features)
    coordinates: np.ndarray  # Shape: (n_samples, 2) for lat/lon
    feature_names: List[str]
    coordinate_system: CoordinateSystem = CoordinateSystem.WGS84
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def n_samples(self) -> int:
        """Number of spatial samples."""
        return self.features.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.features.shape[1]
    
    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get spatial bounds (min_lon, min_lat, max_lon, max_lat).
        
        Assumes coordinates are stored as [longitude, latitude].
        """
        return (
            self.coordinates[:, 0].min(),  # min_lon
            self.coordinates[:, 1].min(),  # min_lat
            self.coordinates[:, 0].max(),  # max_lon
            self.coordinates[:, 1].max()   # max_lat
        )


@dataclass 
class BiodiversityData(SpatialData):
    """
    Specialized container for biodiversity data.
    
    Extends SpatialData with biodiversity-specific attributes.
    """
    species_names: Optional[List[str]] = None
    has_abundance: bool = False  # True if abundance data, False if presence/absence
    zero_inflated: bool = False  # True if data has many zeros
    
    @property
    def richness(self) -> np.ndarray:
        """Calculate species richness per sample."""
        if self.has_abundance:
            return (self.features > 0).sum(axis=1)
        else:
            return self.features.sum(axis=1)
    
    @property
    def is_species_data(self) -> bool:
        """Check if this is species occurrence/abundance data."""
        return self.species_names is not None