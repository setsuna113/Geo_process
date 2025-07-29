# src/grid_systems/bounds_calculator.py
"""Pure bounds calculation utilities - replacing BoundsManager anti-pattern."""

import math
import logging
from typing import Tuple, List, Optional, TYPE_CHECKING
from shapely.geometry import Polygon, Point, box
from shapely.ops import transform
import pyproj

from .bounds_manager import BoundsDefinition  # Import the data class

if TYPE_CHECKING:
    from ..base import GridCell

logger = logging.getLogger(__name__)


class BoundsCalculator:
    """
    Pure calculation utilities for spatial bounds.
    
    This replaces the procedural 'Manager' pattern with focused domain calculations.
    Follows single responsibility principle - only handles mathematical operations.
    """
    
    @staticmethod
    def calculate_area_km2(bounds: Tuple[float, float, float, float], crs: str = "EPSG:4326") -> float:
        """Calculate area in km² for given bounds."""
        minx, miny, maxx, maxy = bounds
        
        if crs == "EPSG:4326":
            # Simple approximation using WGS84 degrees to km conversion
            lat_center = (miny + maxy) / 2
            width_km = (maxx - minx) * 111.0 * abs(math.cos(math.radians(lat_center)))
            height_km = (maxy - miny) * 111.0
            return width_km * height_km
        else:
            # For projected coordinates, use shapely for accurate calculation
            polygon = box(*bounds)
            try:
                transformer = pyproj.Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
                projected_polygon = transform(transformer.transform, polygon)
                return projected_polygon.area / 1_000_000  # m² to km²
            except Exception as e:
                logger.warning(f"Failed to project bounds for area calculation: {e}")
                return (maxx - minx) * (maxy - miny)
    
    @staticmethod
    def calculate_intersection(bounds1: Tuple[float, float, float, float], 
                               bounds2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
        """Calculate intersection of two bounds."""
        minx1, miny1, maxx1, maxy1 = bounds1
        minx2, miny2, maxx2, maxy2 = bounds2
        
        intersect_minx = max(minx1, minx2)
        intersect_miny = max(miny1, miny2)
        intersect_maxx = min(maxx1, maxx2)
        intersect_maxy = min(maxy1, maxy2)
        
        if intersect_minx < intersect_maxx and intersect_miny < intersect_maxy:
            return (intersect_minx, intersect_miny, intersect_maxx, intersect_maxy)
        
        return None
    
    @staticmethod
    def check_intersection(bounds1: Tuple[float, float, float, float], 
                          bounds2: Tuple[float, float, float, float]) -> bool:
        """Check if two bounds intersect."""
        return BoundsCalculator.calculate_intersection(bounds1, bounds2) is not None
    
    @staticmethod
    def contains_point(bounds: Tuple[float, float, float, float], x: float, y: float) -> bool:
        """Check if bounds contain a point."""
        minx, miny, maxx, maxy = bounds
        return minx <= x <= maxx and miny <= y <= maxy


class GridClipper:
    """Focused service for clipping grid cells by bounds."""
    
    def __init__(self, bounds_calculator: BoundsCalculator = None):
        """Initialize with optional bounds calculator."""
        self.bounds_calculator = bounds_calculator or BoundsCalculator()
    
    def clip_cells_by_bounds(self, cells: List['GridCell'], 
                           bounds: BoundsDefinition) -> List['GridCell']:
        """Clip grid cells to specified bounds."""
        clipped_cells = []
        bounds_tuple = bounds.bounds
        
        for cell in cells:
            try:
                # Check if cell intersects with bounds
                if hasattr(cell, 'bounds') and cell.bounds:
                    cell_bounds = cell.bounds
                    if self.bounds_calculator.check_intersection(cell_bounds, bounds_tuple):
                        clipped_cells.append(cell)
                elif hasattr(cell, 'geometry') and cell.geometry:
                    # Check geometry intersection
                    cell_geom = cell.geometry
                    bounds_geom = bounds.polygon
                    
                    if cell_geom.intersects(bounds_geom):
                        clipped_cells.append(cell)
                        
            except Exception as e:
                logger.warning(f"Failed to clip cell {getattr(cell, 'cell_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Clipped {len(cells)} cells to {len(clipped_cells)} within bounds {bounds.name}")
        return clipped_cells