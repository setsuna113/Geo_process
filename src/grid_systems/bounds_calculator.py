"""Bounds calculation service for spatial grid generation."""

from typing import Tuple, Dict, List, Optional, Union, cast, TYPE_CHECKING
from dataclasses import dataclass  
from shapely.geometry import Polygon, Point, box, shape
from shapely.ops import transform
import pyproj
import json
import math
import logging
from pathlib import Path

from src.config import config
from ..database.schema import schema
from ..database.connection import db

if TYPE_CHECKING:
    from ..base import GridCell

logger = logging.getLogger(__name__)

@dataclass
class BoundsDefinition:
    """Structured bounds definition."""
    name: str
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    crs: str = "EPSG:4326"
    category: str = "custom"  # continent, country, region, custom
    metadata: Optional[Dict] = None
    
    @property
    def polygon(self) -> Polygon:
        """Get bounds as polygon."""
        return box(*self.bounds)
    
    @property
    def area_km2(self) -> float:
        """Calculate area in km² using simple approximation."""
        # Simple approximation using WGS84 degrees to km conversion
        # This is less accurate but works without specific projections
        minx, miny, maxx, maxy = self.bounds
        
        # Average latitude for better approximation
        avg_lat = (miny + maxy) / 2
        lat_rad = math.radians(avg_lat)
        
        # Degrees to kilometers conversion
        km_per_deg_lat = 111.32  # ~constant
        km_per_deg_lon = 111.32 * math.cos(lat_rad)
        
        width_km = (maxx - minx) * km_per_deg_lon
        height_km = (maxy - miny) * km_per_deg_lat
        
        return abs(width_km * height_km)


class BoundsCalculator:
    """Service for calculating and managing spatial bounds."""
    
    def __init__(self):
        self._predefined_bounds = self._load_predefined_bounds()
    
    def get_bounds_definition(self, bounds_name: str) -> Optional[BoundsDefinition]:
        """Get bounds definition by name."""
        # Check predefined bounds first
        if bounds_name in self._predefined_bounds:
            return self._predefined_bounds[bounds_name]
        
        # Check database for custom bounds
        try:
            with db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT name, bounds, crs, category, metadata
                    FROM spatial_bounds 
                    WHERE name = %s
                """, (bounds_name,))
                
                result = cursor.fetchone()
                if result:
                    return BoundsDefinition(
                        name=result['name'],
                        bounds=tuple(result['bounds']),
                        crs=result['crs'],
                        category=result['category'],
                        metadata=result['metadata']
                    )
        except Exception as e:
            logger.warning(f"Failed to query bounds from database: {e}")
        
        return None
    
    def calculate_union_bounds(self, bounds_list: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        """Calculate union of multiple bounds."""
        if not bounds_list:
            raise ValueError("Cannot calculate union of empty bounds list")
        
        min_x = min(bounds[0] for bounds in bounds_list)
        min_y = min(bounds[1] for bounds in bounds_list)
        max_x = max(bounds[2] for bounds in bounds_list)
        max_y = max(bounds[3] for bounds in bounds_list)
        
        return (min_x, min_y, max_x, max_y)
    
    def calculate_intersection_bounds(self, bounds1: Tuple[float, float, float, float],
                                    bounds2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
        """Calculate intersection of two bounds."""
        min_x = max(bounds1[0], bounds2[0])
        min_y = max(bounds1[1], bounds2[1])
        max_x = min(bounds1[2], bounds2[2])
        max_y = min(bounds1[3], bounds2[3])
        
        # Check if intersection is valid
        if min_x >= max_x or min_y >= max_y:
            return None
        
        return (min_x, min_y, max_x, max_y)
    
    def expand_bounds(self, bounds: Tuple[float, float, float, float],
                     buffer_degrees: float) -> Tuple[float, float, float, float]:
        """Expand bounds by a buffer in degrees."""
        minx, miny, maxx, maxy = bounds
        return (
            minx - buffer_degrees,
            miny - buffer_degrees,
            maxx + buffer_degrees,
            maxy + buffer_degrees
        )
    
    def validate_bounds(self, bounds: Tuple[float, float, float, float],
                       crs: str = "EPSG:4326") -> bool:
        """Validate bounds format and values."""
        if len(bounds) != 4:
            return False
        
        minx, miny, maxx, maxy = bounds
        
        # Basic validation
        if minx >= maxx or miny >= maxy:
            return False
        
        # CRS-specific validation
        if crs == "EPSG:4326":
            # WGS84 bounds validation
            if not (-180 <= minx <= 180) or not (-180 <= maxx <= 180):
                return False
            if not (-90 <= miny <= 90) or not (-90 <= maxy <= 90):
                return False
        
        return True
    
    def transform_bounds(self, bounds: Tuple[float, float, float, float],
                        from_crs: str, to_crs: str) -> Tuple[float, float, float, float]:
        """Transform bounds from one CRS to another."""
        transformer = pyproj.Transformer.from_crs(from_crs, to_crs, always_xy=True)
        
        minx, miny, maxx, maxy = bounds
        
        # Transform corner points
        corner_points = [
            (minx, miny), (minx, maxy),
            (maxx, miny), (maxx, maxy)
        ]
        
        transformed_points = [transformer.transform(x, y) for x, y in corner_points]
        
        # Calculate new bounds from transformed points
        xs, ys = zip(*transformed_points)
        return (min(xs), min(ys), max(xs), max(ys))
    
    def get_grid_cell_bounds(self, cell: 'GridCell') -> Tuple[float, float, float, float]:
        """Get bounds for a grid cell."""
        if hasattr(cell, 'bounds') and cell.bounds:
            return cell.bounds
        
        # Calculate from geometry if available
        if hasattr(cell, 'geometry') and cell.geometry:
            return cell.geometry.bounds
        
        # Fallback: calculate from center and resolution
        if hasattr(cell, 'center_x') and hasattr(cell, 'center_y'):
            # Estimate cell size (this is approximate)
            half_size = 0.01  # Default cell half-size in degrees
            return (
                cell.center_x - half_size,
                cell.center_y - half_size,
                cell.center_x + half_size,
                cell.center_y + half_size
            )
        
        raise ValueError("Cannot determine bounds for grid cell")
    
    def calculate_bounds_area(self, bounds: Tuple[float, float, float, float],
                            crs: str = "EPSG:4326") -> float:
        """Calculate area of bounds in km²."""
        if crs == "EPSG:4326":
            # Use the same approximation as BoundsDefinition
            bounds_def = BoundsDefinition("temp", bounds, crs)
            return bounds_def.area_km2
        else:
            # For projected CRS, use more accurate calculation
            polygon = box(*bounds)
            
            # Convert to appropriate equal-area projection for area calculation
            from pyproj import CRS
            crs_obj = CRS.from_string(crs)
            
            if crs_obj.is_geographic:
                # Use approximate method for geographic CRS
                bounds_def = BoundsDefinition("temp", bounds, crs)
                return bounds_def.area_km2
            else:
                # For projected CRS, assume units are meters
                area_m2 = polygon.area
                return area_m2 / 1000000  # Convert to km²
    
    def store_bounds_definition(self, bounds_def: BoundsDefinition) -> None:
        """Store bounds definition in database."""
        try:
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO spatial_bounds (name, bounds, crs, category, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        bounds = EXCLUDED.bounds,
                        crs = EXCLUDED.crs,
                        category = EXCLUDED.category,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    bounds_def.name,
                    list(bounds_def.bounds),
                    bounds_def.crs,
                    bounds_def.category,
                    json.dumps(bounds_def.metadata or {})
                ))
                
                logger.info(f"Stored bounds definition: {bounds_def.name}")
                
        except Exception as e:
            logger.error(f"Failed to store bounds definition: {e}")
            raise
    
    def list_available_bounds(self, category: Optional[str] = None) -> List[str]:
        """List available bounds definitions."""
        bounds_names = []
        
        # Add predefined bounds
        for name, bounds_def in self._predefined_bounds.items():
            if category is None or bounds_def.category == category:
                bounds_names.append(name)
        
        # Add database bounds
        try:
            with db.get_cursor() as cursor:
                query = "SELECT name FROM spatial_bounds"
                params = []
                
                if category:
                    query += " WHERE category = %s"
                    params.append(category)
                
                cursor.execute(query, params)
                db_bounds = [row['name'] for row in cursor.fetchall()]
                bounds_names.extend(db_bounds)
                
        except Exception as e:
            logger.warning(f"Failed to query bounds from database: {e}")
        
        return sorted(set(bounds_names))
    
    def _load_predefined_bounds(self) -> Dict[str, BoundsDefinition]:
        """Load predefined bounds definitions."""
        # These could be loaded from a config file or defined here
        predefined = {
            'global': BoundsDefinition(
                name='global',
                bounds=(-180.0, -90.0, 180.0, 90.0),
                crs='EPSG:4326',
                category='global'
            ),
            'north_america': BoundsDefinition(
                name='north_america',
                bounds=(-170.0, 25.0, -50.0, 75.0),
                crs='EPSG:4326',
                category='continent'
            ),
            'europe': BoundsDefinition(
                name='europe',
                bounds=(-25.0, 35.0, 45.0, 75.0),
                crs='EPSG:4326',
                category='continent'
            ),
            'africa': BoundsDefinition(
                name='africa',
                bounds=(-20.0, -35.0, 55.0, 40.0),
                crs='EPSG:4326',
                category='continent'
            )
        }
        
        # Load additional bounds from config if available
        config_bounds = config.get('spatial.predefined_bounds', {})
        for name, bounds_config in config_bounds.items():
            predefined[name] = BoundsDefinition(
                name=name,
                bounds=tuple(bounds_config['bounds']),
                crs=bounds_config.get('crs', 'EPSG:4326'),
                category=bounds_config.get('category', 'custom'),
                metadata=bounds_config.get('metadata')
            )
        
        return predefined


# For backward compatibility
BoundsManager = BoundsCalculator