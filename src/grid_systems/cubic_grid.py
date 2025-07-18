"""Cubic (square) grid implementation using PostGIS."""

from typing import List, Optional, Tuple, Iterator
import logging
from shapely.geometry import Polygon, Point
from shapely import wkt
import math

from ..base import BaseGrid, GridCell
from ..core.registry import component_registry
from ..database.schema import schema
from .bounds_manager import BoundsManager, BoundsDefinition

logger = logging.getLogger(__name__)

@component_registry.grids.register_decorator()
class CubicGrid(BaseGrid):
    """
    Cubic grid system implementation using PostGIS ST_SquareGrid.
    
    Efficient for large-scale analysis with uniform cell sizes.
    """
    
    def __init__(self,
                 resolution: int,
                 bounds: Optional[Union[str, Tuple[float, float, float, float], BoundsDefinition]] = None,
                 crs: str = "EPSG:4326",
                 use_postgis: bool = True,
                 **kwargs):
        """
        Initialize cubic grid.
        
        Args:
            resolution: Grid resolution in meters
            bounds: Bounds specification (name, tuple, or BoundsDefinition)
            crs: Coordinate reference system
            use_postgis: Whether to use PostGIS functions
            **kwargs: Additional parameters
        """
        # Convert bounds to BoundsDefinition if needed
        if bounds is not None and not isinstance(bounds, BoundsDefinition):
            bounds_manager = BoundsManager()
            if isinstance(bounds, str):
                self.bounds_def = bounds_manager.get_bounds(bounds)
            else:
                self.bounds_def = BoundsDefinition('custom', bounds)
        else:
            self.bounds_def = bounds or BoundsManager().get_bounds('global')
            
        super().__init__(
            resolution=resolution,
            bounds=self.bounds_def.bounds,
            crs=crs,
            **kwargs
        )
        
        self.use_postgis = use_postgis
        self.cell_size_degrees = self._calculate_cell_size_degrees()
        
    def _calculate_cell_size_degrees(self) -> float:
        """Calculate cell size in degrees (approximate at center latitude)."""
        center_lat = (self.bounds[1] + self.bounds[3]) / 2
        
        # Meters per degree latitude is constant
        meters_per_degree_lat = 111319.9
        
        # Meters per degree longitude varies with latitude
        meters_per_degree_lon = 111319.9 * math.cos(math.radians(center_lat))
        
        # Use average for square cells
        avg_meters_per_degree = (meters_per_degree_lat + meters_per_degree_lon) / 2
        
        return self.resolution / avg_meters_per_degree
    
    def generate_grid(self) -> List[GridCell]:
        """Generate cubic grid cells."""
        if self.use_postgis and schema.db.pool:
            return self._generate_grid_postgis()
        else:
            return self._generate_grid_python()
    
    def _generate_grid_postgis(self) -> List[GridCell]:
        """Generate grid using PostGIS ST_SquareGrid."""
        logger.info(f"Generating cubic grid with PostGIS (resolution: {self.resolution}m)")
        
        cells = []
        bounds_manager = BoundsManager()
        
        # Subdivide large bounds for memory efficiency
        chunks = bounds_manager.subdivide_bounds(self.bounds_def, max_size_degrees=20.0)
        
        for chunk in chunks:
            chunk_cells = self._generate_chunk_postgis(chunk)
            cells.extend(chunk_cells)
            logger.info(f"Generated {len(chunk_cells)} cells for chunk {chunk.name}")
            
        logger.info(f"Total cubic grid cells generated: {len(cells)}")
        return cells
    
    def _generate_chunk_postgis(self, chunk: BoundsDefinition) -> List[GridCell]:
        """Generate grid cells for a single chunk using PostGIS."""
        cells = []
        
        with schema.db.get_cursor() as cursor:
            # Create bounds polygon
            bounds_wkt = f"POLYGON(({chunk.bounds[0]} {chunk.bounds[1]}, " \
                        f"{chunk.bounds[2]} {chunk.bounds[1]}, " \
                        f"{chunk.bounds[2]} {chunk.bounds[3]}, " \
                        f"{chunk.bounds[0]} {chunk.bounds[3]}, " \
                        f"{chunk.bounds[0]} {chunk.bounds[1]}))"
            
            # Use ST_SquareGrid to generate grid
            # Note: ST_SquareGrid expects size in CRS units
            query = """
                WITH grid AS (
                    SELECT 
                        row_number() OVER () as gid,
                        cell
                    FROM (
                        SELECT (
                            ST_SquareGrid(
                                %s,  -- size in degrees
                                ST_GeomFromText(%s, 4326)
                            )
                        ).* 
                    ) AS cells
                )
                SELECT 
                    gid,
                    ST_AsText(cell) as geom_wkt,
                    ST_AsText(ST_Centroid(cell)) as centroid_wkt,
                    ST_Area(ST_Transform(cell, 54009)) / 1000000 as area_km2,
                    ST_XMin(cell) as minx,
                    ST_YMin(cell) as miny,
                    ST_XMax(cell) as maxx,
                    ST_YMax(cell) as maxy
                FROM grid
                WHERE ST_Intersects(cell, ST_GeomFromText(%s, 4326))
                ORDER BY gid
            """
            
            cursor.execute(query, (self.cell_size_degrees, bounds_wkt, bounds_wkt))
            
            for row in cursor:
                # Generate cell ID based on position
                cell_id = self._generate_cell_id(row['minx'], row['miny'])
                
                cell = GridCell(
                    cell_id=cell_id,
                    geometry=wkt.loads(row['geom_wkt']),
                    centroid=wkt.loads(row['centroid_wkt']),
                    area_km2=row['area_km2'],
                    bounds=(row['minx'], row['miny'], row['maxx'], row['maxy']),
                    metadata={
                        'grid_type': 'cubic',
                        'resolution_m': self.resolution,
                        'chunk': chunk.name
                    }
                )
                cells.append(cell)
                
        return cells
    
    def _generate_grid_python(self) -> List[GridCell]:
        """Generate grid using pure Python (fallback)."""
        logger.info(f"Generating cubic grid with Python (resolution: {self.resolution}m)")
        
        cells = []
        minx, miny, maxx, maxy = self.bounds
        
        # Calculate number of cells
        nx = int((maxx - minx) / self.cell_size_degrees)
        ny = int((maxy - miny) / self.cell_size_degrees)
        
        logger.info(f"Grid dimensions: {nx} x {ny} = {nx * ny} cells")
        
        # Generate cells
        for i in range(nx):
            for j in range(ny):
                cell_minx = minx + i * self.cell_size_degrees
                cell_miny = miny + j * self.cell_size_degrees
                cell_maxx = min(cell_minx + self.cell_size_degrees, maxx)
                cell_maxy = min(cell_miny + self.cell_size_degrees, maxy)
                
                # Create cell polygon
                cell_geom = Polygon([
                    (cell_minx, cell_miny),
                    (cell_maxx, cell_miny),
                    (cell_maxx, cell_maxy),
                    (cell_minx, cell_maxy),
                    (cell_minx, cell_miny)
                ])
                
                cell_id = self._generate_cell_id(cell_minx, cell_miny)
                
                cell = GridCell(
                    cell_id=cell_id,
                    geometry=cell_geom,
                    centroid=cell_geom.centroid,
                    area_km2=self._calculate_area_km2(cell_geom),
                    bounds=(cell_minx, cell_miny, cell_maxx, cell_maxy),
                    metadata={
                        'grid_type': 'cubic',
                        'resolution_m': self.resolution,
                        'grid_index': (i, j)
                    }
                )
                cells.append(cell)
                
        return cells
    
    def _generate_cell_id(self, minx: float, miny: float) -> str:
        """Generate unique cell ID based on position."""
        # Convert to grid indices
        i = int((minx - self.bounds[0]) / self.cell_size_degrees)
        j = int((miny - self.bounds[1]) / self.cell_size_degrees)
        
        # Include resolution in ID for multi-resolution support
        return f"C{self.resolution}_{i:05d}_{j:05d}"
    
    def _calculate_area_km2(self, polygon: Polygon) -> float:
        """Calculate polygon area in km²."""
        # Transform to equal area projection
        from shapely.ops import transform
        
        transformer = self.transformer_to_crs if self.crs != "EPSG:54009" else None
        if transformer:
            projected = transform(transformer.transform, polygon)
        else:
            projected = polygon
            
        return projected.area / 1_000_000
    
    def get_cell_id(self, x: float, y: float) -> str:
        """Get cell ID for a coordinate."""
        if not self.bounds_def.contains(x, y):
            raise ValueError(f"Coordinate ({x}, {y}) outside grid bounds")
            
        # Calculate grid indices
        i = int((x - self.bounds[0]) / self.cell_size_degrees)
        j = int((y - self.bounds[1]) / self.cell_size_degrees)
        
        return f"C{self.resolution}_{i:05d}_{j:05d}"
    
    def get_cell_by_id(self, cell_id: str) -> Optional[GridCell]:
        """Get cell by ID."""
        # Parse cell ID
        parts = cell_id.split('_')
        if len(parts) != 3 or not parts[0].startswith('C'):
            return None
            
        try:
            resolution = int(parts[0][1:])
            i = int(parts[1])
            j = int(parts[2])
            
            if resolution != self.resolution:
                return None
                
            # Calculate cell bounds
            cell_minx = self.bounds[0] + i * self.cell_size_degrees
            cell_miny = self.bounds[1] + j * self.cell_size_degrees
            
            if not self.bounds_def.contains(cell_minx, cell_miny):
                return None
                
            cell_maxx = min(cell_minx + self.cell_size_degrees, self.bounds[2])
            cell_maxy = min(cell_miny + self.cell_size_degrees, self.bounds[3])
            
            # Create cell
            cell_geom = Polygon([
                (cell_minx, cell_miny),
                (cell_maxx, cell_miny),
                (cell_maxx, cell_maxy),
                (cell_minx, cell_maxy),
                (cell_minx, cell_miny)
            ])
            
            return GridCell(
                cell_id=cell_id,
                geometry=cell_geom,
                centroid=cell_geom.centroid,
                area_km2=self._calculate_area_km2(cell_geom),
                bounds=(cell_minx, cell_miny, cell_maxx, cell_maxy),
                metadata={
                    'grid_type': 'cubic',
                    'resolution_m': self.resolution,
                    'grid_index': (i, j)
                }
            )
            
        except (ValueError, IndexError):
            return None
    
    def get_neighbor_ids(self, cell_id: str) -> List[str]:
        """Get IDs of neighboring cells (8-connected)."""
        parts = cell_id.split('_')
        if len(parts) != 3:
            return []
            
        try:
            i = int(parts[1])
            j = int(parts[2])
            
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                        
                    ni = i + di
                    nj = j + dj
                    
                    # Check if neighbor is within bounds
                    neighbor_x = self.bounds[0] + ni * self.cell_size_degrees
                    neighbor_y = self.bounds[1] + nj * self.cell_size_degrees
                    
                    if self.bounds_def.contains(neighbor_x, neighbor_y):
                        neighbors.append(f"C{self.resolution}_{ni:05d}_{nj:05d}")
                        
            return neighbors
            
        except ValueError:
            return []