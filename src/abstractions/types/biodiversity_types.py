"""
Data type definitions for biodiversity analysis.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
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
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get spatial bounds (min_lon, min_lat, max_lon, max_lat).
        
        Convention: coordinates are ALWAYS [longitude, latitude] per WGS84 standard.
        """
        # Import here to avoid circular dependency
        from src.biodiversity_analysis.shared.coordinate_convention import LON_IDX, LAT_IDX
        
        return (
            self.coordinates[:, LON_IDX].min(),  # min_lon
            self.coordinates[:, LAT_IDX].min(),  # min_lat
            self.coordinates[:, LON_IDX].max(),  # max_lon
            self.coordinates[:, LAT_IDX].max()   # max_lat
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