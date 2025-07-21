"""Bounds management for spatial grid generation."""

from typing import Tuple, Dict, List, Optional, Union, cast, TYPE_CHECKING
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, box, shape
from shapely.ops import transform
import pyproj
import json
import math
import logging
from pathlib import Path

from ..config import config
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
        
        # Approximate conversion: 1 degree ≈ 111 km at equator
        # Adjust for latitude using cosine
        lat_center = (miny + maxy) / 2
        width_km = (maxx - minx) * 111.0 * abs(math.cos(math.radians(lat_center)))
        height_km = (maxy - miny) * 111.0
        
        return width_km * height_km
    
    def contains(self, x: float, y: float) -> bool:
        """Check if point is within bounds."""
        return (self.bounds[0] <= x <= self.bounds[2] and 
                self.bounds[1] <= y <= self.bounds[3])
    
    def intersects(self, other_bounds: Tuple[float, float, float, float]) -> bool:
        """Check if bounds intersect with other bounds."""
        other_box = box(*other_bounds)
        return self.polygon.intersects(other_box)
    
    def intersection(self, other_bounds: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
        """Get intersection of bounds."""
        other_box = box(*other_bounds)
        intersection = self.polygon.intersection(other_box)
        
        if intersection.is_empty:
            return None
            
        return intersection.bounds
    
    def buffer(self, distance_km: float) -> 'BoundsDefinition':
        """Buffer bounds by distance in km."""
        # Approximate degrees from km (at equator)
        degrees = distance_km / 111.0
        
        return BoundsDefinition(
            name=f"{self.name}_buffered",
            bounds=(
                self.bounds[0] - degrees,
                self.bounds[1] - degrees,
                self.bounds[2] + degrees,
                self.bounds[3] + degrees
            ),
            crs=self.crs,
            category=self.category,
            metadata={**self.metadata, 'buffered_km': distance_km} if self.metadata else {'buffered_km': distance_km}
        )


class BoundsManager:
    """Manage spatial bounds for grid generation."""
    
    # Predefined regions
    REGIONS = {
        # Global
        'global': BoundsDefinition('global', (-180, -90, 180, 90), category='global'),
        
        # Continents (simplified bounds)
        'africa': BoundsDefinition('africa', (-20, -35, 55, 37), category='continent'),
        'asia': BoundsDefinition('asia', (25, -10, 180, 80), category='continent'),
        'europe': BoundsDefinition('europe', (-25, 35, 50, 71), category='continent'),
        'north_america': BoundsDefinition('north_america', (-170, 15, -50, 85), category='continent'),
        'south_america': BoundsDefinition('south_america', (-85, -56, -35, 15), category='continent'),
        'oceania': BoundsDefinition('oceania', (110, -50, 180, -10), category='continent'),
        'antarctica': BoundsDefinition('antarctica', (-180, -90, 180, -60), category='continent'),
        
        # Major regions
        'tropical': BoundsDefinition('tropical', (-180, -23.5, 180, 23.5), category='region'),
        'arctic': BoundsDefinition('arctic', (-180, 66.5, 180, 90), category='region'),
        'temperate_north': BoundsDefinition('temperate_north', (-180, 23.5, 180, 66.5), category='region'),
        'temperate_south': BoundsDefinition('temperate_south', (-180, -66.5, 180, -23.5), category='region'),
        
        # Example countries (add more as needed)
        'usa': BoundsDefinition('usa', (-125, 24, -66, 49), category='country'),
        'brazil': BoundsDefinition('brazil', (-74, -34, -34, 5), category='country'),
        'australia': BoundsDefinition('australia', (113, -44, 154, -10), category='country'),
        'china': BoundsDefinition('china', (73, 18, 135, 54), category='country'),
    }
    
    def __init__(self):
        """Initialize bounds manager."""
        self.custom_regions: Dict[str, BoundsDefinition] = {}
        self._load_custom_regions()
        
    def _load_custom_regions(self):
        """Load custom regions from config."""
        custom_bounds = config.get('processing_bounds.custom', {})
        
        for name, bounds_config in custom_bounds.items():
            if isinstance(bounds_config, list) and len(bounds_config) == 4:
                self.custom_regions[name] = BoundsDefinition(
                    name=name,
                    bounds=tuple(bounds_config),
                    category='custom'
                )
            elif isinstance(bounds_config, dict):
                self.custom_regions[name] = BoundsDefinition(
                    name=name,
                    bounds=tuple(bounds_config['bounds']),
                    category=bounds_config.get('category', 'custom'),
                    metadata=bounds_config.get('metadata')
                )
    
    def get_bounds(self, name: str) -> BoundsDefinition:
        """
        Get bounds by name.
        
        Args:
            name: Region name or 'minx,miny,maxx,maxy' string
            
        Returns:
            BoundsDefinition object
        """
        # Check predefined regions
        if name in self.REGIONS:
            return self.REGIONS[name]
            
        # Check custom regions
        if name in self.custom_regions:
            return self.custom_regions[name]
            
        # Try to parse as bounds string
        if ',' in name:
            try:
                parts = [float(x.strip()) for x in name.split(',')]
                if len(parts) == 4:
                    return BoundsDefinition(
                        name='custom_bounds',
                        bounds=cast(Tuple[float, float, float, float], tuple(parts)),
                        category='custom'
                    )
            except ValueError:
                pass
                
        raise ValueError(f"Unknown bounds: {name}. Available: {self.list_available()}")
    
    def list_available(self) -> Dict[str, List[str]]:
        """List all available bounds grouped by category."""
        available: Dict[str, List[str]] = {}
        
        # Group predefined regions
        for name, bounds_def in self.REGIONS.items():
            category = bounds_def.category
            if category not in available:
                available[category] = []
            available[category].append(name)
            
        # Add custom regions
        if self.custom_regions:
            available['custom'] = list(self.custom_regions.keys())
            
        return available
    
    def validate_bounds_overlap(self, 
                               bounds: BoundsDefinition,
                               data_bounds: Tuple[float, float, float, float]) -> bool:
        """
        Validate that bounds overlap with data extent.
        
        Args:
            bounds: Bounds to validate
            data_bounds: Extent of data
            
        Returns:
            True if bounds overlap
        """
        return bounds.intersects(data_bounds)
    
    def get_species_data_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the extent of all species data in database."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    ST_XMin(ST_Extent(geometry)) as minx,
                    ST_YMin(ST_Extent(geometry)) as miny,
                    ST_XMax(ST_Extent(geometry)) as maxx,
                    ST_YMax(ST_Extent(geometry)) as maxy
                FROM species_ranges
                WHERE geometry IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result and all(v is not None for v in result.values()):
                return (result['minx'], result['miny'], result['maxx'], result['maxy'])
                
        return None
    
    def clip_grid_cells(self, cells: List['GridCell'], bounds: BoundsDefinition) -> List['GridCell']:
        """
        Clip grid cells to bounds.
        
        Args:
            cells: List of grid cells
            bounds: Bounds to clip to
            
        Returns:
            List of clipped cells
        """
        clipped_cells = []
        bounds_polygon = bounds.polygon
        
        for cell in cells:
            if cell.geometry.intersects(bounds_polygon):
                # Check if fully contained
                if bounds_polygon.contains(cell.geometry):
                    clipped_cells.append(cell)
                else:
                    # Clip the cell
                    clipped_geom = cell.geometry.intersection(bounds_polygon)
                    if not clipped_geom.is_empty:
                        # Create new cell with clipped geometry
                        from ..base import GridCell
                        clipped_cell = GridCell(
                            cell_id=f"{cell.cell_id}_clipped",
                            geometry=cast(Polygon, clipped_geom),
                            centroid=cast(Point, clipped_geom.centroid),
                            area_km2=clipped_geom.area / 1_000_000,
                            bounds=clipped_geom.bounds,
                            metadata={**cell.metadata, 'clipped': True} if cell.metadata else {'clipped': True}
                        )
                        clipped_cells.append(clipped_cell)
                        
        return clipped_cells
    
    def subdivide_bounds(self, 
                        bounds: BoundsDefinition,
                        max_size_degrees: float = 10.0) -> List[BoundsDefinition]:
        """
        Subdivide large bounds into smaller chunks for processing.
        
        Args:
            bounds: Bounds to subdivide
            max_size_degrees: Maximum size in degrees for each chunk
            
        Returns:
            List of subdivided bounds
        """
        minx, miny, maxx, maxy = bounds.bounds
        width = maxx - minx
        height = maxy - miny
        
        if width <= max_size_degrees and height <= max_size_degrees:
            return [bounds]
            
        # Calculate number of divisions needed
        x_divisions = int(width / max_size_degrees) + 1
        y_divisions = int(height / max_size_degrees) + 1
        
        chunk_width = width / x_divisions
        chunk_height = height / y_divisions
        
        chunks = []
        for i in range(x_divisions):
            for j in range(y_divisions):
                chunk_bounds = (
                    minx + i * chunk_width,
                    miny + j * chunk_height,
                    min(minx + (i + 1) * chunk_width, maxx),
                    min(miny + (j + 1) * chunk_height, maxy)
                )
                
                chunks.append(BoundsDefinition(
                    name=f"{bounds.name}_chunk_{i}_{j}",
                    bounds=chunk_bounds,
                    crs=bounds.crs,
                    category='chunk',
                    metadata={'parent': bounds.name, 'chunk_index': (i, j)}
                ))
                
        logger.info(f"Subdivided {bounds.name} into {len(chunks)} chunks")
        return chunks
    
    def save_bounds_definition(self, bounds: BoundsDefinition, name: str):
        """Save custom bounds definition."""
        self.custom_regions[name] = bounds
        
        # Optionally persist to config
        custom_bounds = config.get('processing_bounds.custom', {})
        custom_bounds[name] = {
            'bounds': list(bounds.bounds),
            'category': bounds.category,
            'metadata': bounds.metadata
        }
        # Note: Would need config.set() method to persist