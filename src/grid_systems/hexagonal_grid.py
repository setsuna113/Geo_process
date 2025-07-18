# src/grid_systems/hexagonal_grid.py
"""Hexagonal grid implementation using H3."""

from typing import List, Optional, Tuple, Iterator, Set
import logging
import h3
from shapely.geometry import Polygon, Point
import numpy as np

from ..base import BaseGrid, GridCell
from ..core.registry import component_registry
from .bounds_manager import BoundsManager, BoundsDefinition

logger = logging.getLogger(__name__)

@component_registry.grids.register_decorator()
class HexagonalGrid(BaseGrid):
    """
    Hexagonal grid system using Uber's H3 library.
    
    Efficient for spatial analysis with uniform neighbor distances.
    """
    
    # H3 resolution to approximate meters mapping
    H3_RESOLUTION_METERS = {
        0: 1107712,   # ~1107 km
        1: 418676,    # ~418 km
        2: 158244,    # ~158 km
        3: 59810,     # ~59 km
        4: 22606,     # ~22 km
        5: 8544,      # ~8.5 km
        6: 3229,      # ~3.2 km
        7: 1220,      # ~1.2 km
        8: 461,       # ~461 m
        9: 174,       # ~174 m
        10: 66,       # ~66 m
        11: 25,       # ~25 m
        12: 9,        # ~9 m
        13: 3,        # ~3 m
        14: 1,        # ~1 m
        15: 0.5       # ~0.5 m
    }
    
    def __init__(self,
                 resolution: int,
                 bounds: Optional[Union[str, Tuple[float, float, float, float], BoundsDefinition]] = None,
                 crs: str = "EPSG:4326",
                 h3_resolution: Optional[int] = None,
                 **kwargs):
        """
        Initialize hexagonal grid.
        
        Args:
            resolution: Target resolution in meters
            bounds: Bounds specification
            crs: Coordinate reference system (must be EPSG:4326 for H3)
            h3_resolution: Override H3 resolution selection
            **kwargs: Additional parameters
        """
        if crs != "EPSG:4326":
            raise ValueError("H3 only supports EPSG:4326 (WGS84)")
            
        # Convert bounds to BoundsDefinition
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
        
        # Determine H3 resolution
        self.h3_resolution = h3_resolution or self._select_h3_resolution(resolution)
        logger.info(f"Selected H3 resolution {self.h3_resolution} for {resolution}m target")
        
        # Memory management
        self.chunk_size = self.config.get('chunk_size', 100000)  # Max hexagons per chunk
        
    def _select_h3_resolution(self, target_meters: int) -> int:
        """Select appropriate H3 resolution for target size."""
        for res, meters in self.H3_RESOLUTION_METERS.items():
            if meters <= target_meters:
                return res
        return 15  # Maximum resolution
    
    def generate_grid(self) -> List[GridCell]:
        """Generate hexagonal grid cells."""
        logger.info(f"Generating hexagonal grid (H3 resolution: {self.h3_resolution})")
        
        # For large areas, process in chunks
        bounds_manager = BoundsManager()
        chunks = bounds_manager.subdivide_bounds(self.bounds_def, max_size_degrees=10.0)
        
        all_cells = []
        all_hexagons = set()  # Track unique hexagons
        
        for chunk in chunks:
            chunk_hexagons = self._get_hexagons_for_bounds(chunk)
            
            # Process in batches to manage memory
            for hex_batch in self._batch_iterator(chunk_hexagons, self.chunk_size):
                cells = self._create_cells_from_hexagons(hex_batch, all_hexagons)
                all_cells.extend(cells)
                
            logger.info(f"Processed chunk {chunk.name}: {len(chunk_hexagons)} hexagons")
            
        logger.info(f"Total hexagonal grid cells generated: {len(all_cells)}")
        return all_cells
    
    def _get_hexagons_for_bounds(self, bounds: BoundsDefinition) -> Set[str]:
        """Get H3 hexagons covering the bounds."""
        # Create a polygon from bounds
        poly_coords = [
            [bounds.bounds[0], bounds.bounds[1]],  # SW
            [bounds.bounds[2], bounds.bounds[1]],  # SE
            [bounds.bounds[2], bounds.bounds[3]],  # NE
            [bounds.bounds[0], bounds.bounds[3]],  # NW
            [bounds.bounds[0], bounds.bounds[1]]   # Close
        ]
        
        # Convert to H3 format (lat, lng)
        h3_poly = [[lat, lng] for lng, lat in poly_coords]
        
        try:
            # Get hexagons using polyfill
            hexagons = h3.polyfill(h3_poly, self.h3_resolution, geo_json_conformant=False)
            return hexagons
            
        except Exception as e:
            logger.warning(f"H3 polyfill failed for bounds {bounds.name}: {e}")
            # Fallback: sample points and get hexagons
            return self._get_hexagons_by_sampling(bounds)
    
    def _get_hexagons_by_sampling(self, bounds: BoundsDefinition) -> Set[str]:
        """Get hexagons by sampling points (fallback method)."""
        hexagons = set()
        
        # Estimate hexagon size in degrees
        hex_edge_km = self.H3_RESOLUTION_METERS[self.h3_resolution] / 1000
        hex_edge_deg = hex_edge_km / 111  # Rough approximation
        
        # Sample points
        minx, miny, maxx, maxy = bounds.bounds
        x_points = np.arange(minx, maxx, hex_edge_deg)
        y_points = np.arange(miny, maxy, hex_edge_deg)
        
        for x in x_points:
            for y in y_points:
                hex_id = h3.geo_to_h3(y, x, self.h3_resolution)  # lat, lng
                hexagons.add(hex_id)
                
        return hexagons
    
    def _batch_iterator(self, items: Set[str], batch_size: int) -> Iterator[List[str]]:
        """Iterate over items in batches."""
        items_list = list(items)
        for i in range(0, len(items_list), batch_size):
            yield items_list[i:i + batch_size]
    
    def _create_cells_from_hexagons(self, 
                                   hex_ids: List[str], 
                                   processed: Set[str]) -> List[GridCell]:
        """Create GridCell objects from H3 hexagon IDs."""
        cells = []
        
        for hex_id in hex_ids:
            if hex_id in processed:
                continue
                
            processed.add(hex_id)
            
            try:
                # Get hexagon boundary
                boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
                
                # Convert to shapely polygon (lng, lat -> x, y)
                coords = [(lng, lat) for lat, lng in boundary]
                coords.append(coords[0])  # Close polygon
                
                hex_polygon = Polygon(coords)
                
                # Get center
                lat, lng = h3.h3_to_geo(hex_id)
                center = Point(lng, lat)
                
                # Calculate area
                area_m2 = h3.cell_area(hex_id, unit='m^2')
                area_km2 = area_m2 / 1_000_000
                
                cell = GridCell(
                    cell_id=f"H{self.h3_resolution}_{hex_id}",
                    geometry=hex_polygon,
                    centroid=center,
                    area_km2=area_km2,
                    bounds=hex_polygon.bounds,
                    metadata={
                        'grid_type': 'hexagonal',
                        'h3_resolution': self.h3_resolution,
                        'h3_id': hex_id,
                        'resolution_m': self.resolution
                    }
                )
                
                cells.append(cell)
                
            except Exception as e:
                logger.error(f"Failed to create cell for hexagon {hex_id}: {e}")
                
        return cells
    
    def get_cell_id(self, x: float, y: float) -> str:
        """Get cell ID for a coordinate."""
        if not self.bounds_def.contains(x, y):
            raise ValueError(f"Coordinate ({x}, {y}) outside grid bounds")
            
        # Get H3 hexagon for coordinate
        hex_id = h3.geo_to_h3(y, x, self.h3_resolution)  # lat, lng
        return f"H{self.h3_resolution}_{hex_id}"
    
    def get_cell_by_id(self, cell_id: str) -> Optional[GridCell]:
        """Get cell by ID."""
        # Parse cell ID
        if not cell_id.startswith('H'):
            return None
            
        parts = cell_id.split('_', 1)
        if len(parts) != 2:
            return None
            
        try:
            resolution = int(parts[0][1:])
            hex_id = parts[1]
            
            if resolution != self.h3_resolution:
                return None
                
            # Validate hex ID
            if not h3.h3_is_valid(hex_id):
                return None
                
            # Create cell from hex ID
            cells = self._create_cells_from_hexagons([hex_id], set())
            return cells[0] if cells else None
            
        except (ValueError, Exception):
            return None
    
    def get_neighbor_ids(self, cell_id: str) -> List[str]:
        """Get IDs of neighboring cells."""
        # Parse cell ID
        parts = cell_id.split('_', 1)
        if len(parts) != 2:
            return []
            
        hex_id = parts[1]
        
        try:
            # Get H3 neighbors
            neighbor_hexes = h3.k_ring(hex_id, 1)
            neighbor_hexes.remove(hex_id)  # Remove center
            
            # Convert to cell IDs
            return [f"H{self.h3_resolution}_{hex_id}" for hex_id in neighbor_hexes]
            
        except Exception:
            return []
    
    def get_cells_at_resolution(self, parent_cell_id: str, target_resolution: int) -> List[str]:
        """Get child cells at a different resolution."""
        parts = parent_cell_id.split('_', 1)
        if len(parts) != 2:
            return []
            
        parent_hex = parts[1]
        
        try:
            if target_resolution > self.h3_resolution:
                # Get children
                children = h3.h3_to_children(parent_hex, target_resolution)
                return [f"H{target_resolution}_{hex_id}" for hex_id in children]
            elif target_resolution < self.h3_resolution:
                # Get parent
                parent = h3.h3_to_parent(parent_hex, target_resolution)
                return [f"H{target_resolution}_{parent}"]
            else:
                return [parent_cell_id]
                
        except Exception:
            return []