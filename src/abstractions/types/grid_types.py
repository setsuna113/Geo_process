# src/abstractions/types/grid_types.py
"""Grid system type definitions."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, Point


@dataclass
class GridCell:
    """Standard grid cell representation."""
    cell_id: str
    geometry: Polygon
    centroid: Point
    area_km2: float
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'cell_id': self.cell_id,
            'geometry_wkt': self.geometry.wkt,
            'centroid_wkt': self.centroid.wkt,
            'area_km2': self.area_km2,
            'bounds': self.bounds,
            'metadata': self.metadata or {}
        }
    
    def intersects(self, geometry: Polygon) -> bool:
        """Check if cell intersects with geometry."""
        return self.geometry.intersects(geometry)
    
    def intersection_area(self, geometry: Polygon) -> float:
        """Calculate intersection area in km²."""
        if not self.intersects(geometry):
            return 0.0
        intersection = self.geometry.intersection(geometry)
        # Assuming geographic coordinates, approximate area calculation
        # In production, use proper projection for accurate area
        return intersection.area * 111.32 * 111.32  # Rough conversion to km²