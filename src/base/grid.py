"""Base grid class for spatial grid systems."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Iterator
# GridCell moved to abstractions
from src.abstractions.types import GridCell
from shapely.geometry import Polygon, Point, box
from shapely.ops import transform
import pyproj
import logging

# Registry import removed - unused import violating base layer architecture
from src.config import config
# Database schema import removed - using dependency injection to avoid architectural violation

logger = logging.getLogger(__name__)

# GridCell moved to abstractions.types.grid_types


class BaseGrid(ABC):
    """
    Base class for all grid systems.
    
    Handles:
    - Grid generation
    - Cell management
    - Spatial operations
    - Storage integration
    """
    
    def __init__(self,
                 resolution: int,
                 bounds: Optional[Tuple[float, float, float, float]] = None,
                 crs: str = "EPSG:4326",
                 schema=None,
                 **kwargs):
        """
        Initialize grid system.
        
        Args:
            resolution: Grid resolution in meters
            bounds: (minx, miny, maxx, maxy) in CRS units
            crs: Coordinate reference system
            schema: Database schema instance (injected to avoid architectural violation)
            **kwargs: Grid-specific parameters
        """
        self.resolution = resolution
        self.bounds = bounds or self._get_default_bounds()
        self.crs = crs  
        self.schema = schema  # Injected dependency to avoid architectural violation
        self.config = self._merge_config(kwargs)
        
        # Setup projection transformers
        self._setup_projections()
        
        # Grid metadata
        self.grid_id: Optional[str] = None
        self._cells: Optional[List[GridCell]] = None
        self.specification: Optional[Any] = None  # Grid specification object
        
    def _merge_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge kwargs with default config."""
        grid_type = self.__class__.__name__.lower().replace('grid', '')
        default_config = config.get(f'grids.{grid_type}', {})
        return {**default_config, **kwargs}
    
    def _get_default_bounds(self) -> Tuple[float, float, float, float]:
        """Get default bounds from config."""
        return tuple(config.get('grids.default_bounds', [-180, -90, 180, 90]))
    
    def _setup_projections(self):
        """Setup coordinate transformers."""
        self.transformer_to_crs = pyproj.Transformer.from_crs(
            "EPSG:4326", self.crs, always_xy=True
        )
        self.transformer_from_crs = pyproj.Transformer.from_crs(
            self.crs, "EPSG:4326", always_xy=True
        )
    
    @abstractmethod
    def generate_grid(self) -> List[GridCell]:
        """
        Generate grid cells.
        
        Returns:
            List of GridCell objects
        """
        pass
    
    @abstractmethod
    def get_cell_id(self, x: float, y: float) -> str:
        """
        Get cell ID for a coordinate.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Cell ID
        """
        pass
    
    @abstractmethod
    def get_cell_by_id(self, cell_id: str) -> Optional[GridCell]:
        """
        Get cell by ID.
        
        Args:
            cell_id: Cell identifier
            
        Returns:
            GridCell or None
        """
        pass
    
    def get_cells(self) -> List[GridCell]:
        """Get all grid cells (generate if needed)."""
        if self._cells is None:
            self._cells = self.generate_grid()
        return self._cells
    
    def get_cell_count(self) -> int:
        """Get total number of cells."""
        return len(self.get_cells())
    
    def get_cells_in_bounds(self, 
                           bounds: Tuple[float, float, float, float]) -> List[GridCell]:
        """Get cells within bounds."""
        bbox = box(*bounds)
        return [cell for cell in self.get_cells() if cell.geometry.intersects(bbox)]
    
    def get_cells_for_geometry(self, geometry: Polygon) -> List[GridCell]:
        """Get cells that intersect with geometry."""
        # Use spatial index if available
        return [cell for cell in self.get_cells() if cell.intersects(geometry)]
    
    def store_grid(self, name: str, description: str = "") -> str:
        """
        Store grid definition in database.
        
        Args:
            name: Grid name
            description: Grid description
            
        Returns:
            Grid ID
        """
        logger.info(f"Storing grid '{name}'...")
        
        # Check if already stored
        if self.grid_id is not None:
            logger.warning(f"Grid '{name}' already stored with ID {self.grid_id}")
            return self.grid_id
        
        # Store grid metadata
        grid_type = self.__class__.__name__.lower().replace('grid', '')
        metadata = {
            'description': description,
            'cell_count': self.get_cell_count(),
            'config': self.config
        }
        
        if self.schema:
            self.grid_id = self.schema.store_grid_definition(
            name=name,
            grid_type=grid_type,
            resolution=self.resolution,
            bounds=f"POLYGON(({self.bounds[0]} {self.bounds[1]}, "
                   f"{self.bounds[2]} {self.bounds[1]}, "
                   f"{self.bounds[2]} {self.bounds[3]}, "
                   f"{self.bounds[0]} {self.bounds[3]}, "
                   f"{self.bounds[0]} {self.bounds[1]}))",
            metadata=metadata
        )
        
        # Store cells in batches
        cells = self.get_cells()
        batch_size = 10000
        
        for i in range(0, len(cells), batch_size):
            batch = cells[i:i + batch_size]
            cells_data = [cell.to_dict() for cell in batch]
            if self.schema:
                self.schema.store_grid_cells_batch(self.grid_id, cells_data)
            logger.info(f"Stored {min(i + batch_size, len(cells))}/{len(cells)} cells")
            
        return self.grid_id
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate grid statistics."""
        cells = self.get_cells()
        areas = [cell.area_km2 for cell in cells]
        
        return {
            'cell_count': len(cells),
            'total_area_km2': sum(areas),
            'avg_cell_area_km2': sum(areas) / len(areas) if areas else 0,
            'min_cell_area_km2': min(areas) if areas else 0,
            'max_cell_area_km2': max(areas) if areas else 0,
            'bounds': self.bounds,
            'crs': self.crs
        }
    
    # Enhanced methods for base class enhancements
    
    def get_tiles_for_bounds(self, 
                           bounds: Tuple[float, float, float, float],
                           tile_size: Optional[int] = None) -> Iterator[Tuple[str, Tuple[float, float, float, float]]]:
        """
        Get tile definitions for given bounds.
        
        Args:
            bounds: (minx, miny, maxx, maxy) bounding box
            tile_size: Size of tiles in grid units (None for auto)
            
        Yields:
            Tuples of (tile_id, tile_bounds)
        """
        actual_tile_size = tile_size or config.get('grids.default_tile_size', 10)  # 10 grid units per tile
            
        minx, miny, maxx, maxy = bounds
        
        # Calculate tile grid based on resolution
        tile_width = self.resolution * actual_tile_size
        tile_height = self.resolution * actual_tile_size
        
        # Calculate number of tiles needed
        x_tiles = int((maxx - minx) / tile_width) + 1
        y_tiles = int((maxy - miny) / tile_height) + 1
        
        for y in range(y_tiles):
            for x in range(x_tiles):
                tile_minx = minx + x * tile_width
                tile_maxx = min(tile_minx + tile_width, maxx)
                tile_miny = miny + y * tile_height
                tile_maxy = min(tile_miny + tile_height, maxy)
                
                tile_id = f"tile_{x}_{y}"
                tile_bounds = (tile_minx, tile_miny, tile_maxx, tile_maxy)
                
                yield tile_id, tile_bounds
                
    def supports_irregular_shapes(self) -> bool:
        """
        Check if grid supports irregular tile shapes.
        Default implementation returns False - override in subclasses.
        
        Returns:
            True if irregular shapes are supported
        """
        return False
        
    def create_irregular_tile(self, 
                            geometry: Polygon,
                            tile_id: str) -> List[GridCell]:
        """
        Create tile with irregular shape.
        
        Args:
            geometry: Shape of the irregular tile
            tile_id: Identifier for the tile
            
        Returns:
            List of grid cells within the tile
            
        Raises:
            NotImplementedError: If irregular shapes not supported
        """
        if not self.supports_irregular_shapes():
            raise NotImplementedError(f"{self.__class__.__name__} does not support irregular tile shapes")
            
        # Default implementation for grids that support irregular shapes
        return self.get_cells_for_geometry(geometry)
        
    def check_resolution_compatibility(self, 
                                     other_resolution: int,
                                     tolerance: float = 0.1) -> Tuple[bool, str]:
        """
        Check if this grid is compatible with another resolution.
        
        Args:
            other_resolution: Resolution to check compatibility with
            tolerance: Tolerance for floating point comparison
            
        Returns:
            Tuple of (is_compatible, reason)
        """
        ratio = max(self.resolution, other_resolution) / min(self.resolution, other_resolution)
        
        # Check if resolutions are multiples of each other
        if abs(ratio - round(ratio)) < tolerance:
            return True, f"Resolutions are compatible (ratio: {ratio:.1f})"
            
        # Check if they're close enough
        if abs(1.0 - self.resolution / other_resolution) < tolerance:
            return True, f"Resolutions are nearly identical"
            
        return False, f"Resolutions not compatible (ratio: {ratio:.2f}, tolerance: {tolerance})"
        
    def get_neighboring_cells(self, cell_id: str, distance: int = 1) -> List[GridCell]:
        """
        Get neighboring cells within specified distance.
        
        Args:
            cell_id: Central cell ID
            distance: Distance in number of cells
            
        Returns:
            List of neighboring cells
        """
        center_cell = self.get_cell_by_id(cell_id)
        if not center_cell:
            return []
            
        # Expand bounds by distance * resolution
        expand_by = distance * self.resolution
        minx, miny, maxx, maxy = center_cell.bounds
        expanded_bounds = (
            minx - expand_by,
            miny - expand_by, 
            maxx + expand_by,
            maxy + expand_by
        )
        
        # Get all cells in expanded bounds
        candidates = self.get_cells_in_bounds(expanded_bounds)
        
        # Filter by actual distance
        neighbors = []
        for candidate in candidates:
            if candidate.cell_id == cell_id:
                continue
                
            # Calculate distance between centroids
            distance_m = center_cell.centroid.distance(candidate.centroid)
            max_distance_m = distance * self.resolution
            
            if distance_m <= max_distance_m:
                neighbors.append(candidate)
                
        return neighbors
        
    def optimize_for_region(self, 
                          focus_bounds: Tuple[float, float, float, float],
                          density_multiplier: float = 2.0) -> 'BaseGrid':
        """
        Create an optimized grid for a specific region.
        
        Args:
            focus_bounds: Bounds of the region to optimize for
            density_multiplier: How much denser the grid should be in focus region
            
        Returns:
            New grid instance optimized for the region
        """
        # Default implementation creates a higher resolution grid for focus area
        # Subclasses can override for more sophisticated optimization
        
        new_resolution = int(self.resolution / density_multiplier)
        
        # Create new instance with same class and optimized resolution
        optimized_grid = self.__class__(
            resolution=new_resolution,
            bounds=focus_bounds,
            crs=self.crs,
            **self.config
        )
        
        logger.info(f"Created optimized grid: {self.resolution}m -> {new_resolution}m resolution")
        
        return optimized_grid