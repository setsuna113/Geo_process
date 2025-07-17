"""Base grid class for spatial grid systems."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, box
from shapely.ops import transform
import pyproj
import logging

from ..core.registry import component_registry
from ..config import config
from ..database.schema import schema

logger = logging.getLogger(__name__)

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
        return intersection.area / 1_000_000  # m² to km²


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
                 **kwargs):
        """
        Initialize grid system.
        
        Args:
            resolution: Grid resolution in meters
            bounds: (minx, miny, maxx, maxy) in CRS units
            crs: Coordinate reference system
            **kwargs: Grid-specific parameters
        """
        self.resolution = resolution
        self.bounds = bounds or self._get_default_bounds()
        self.crs = crs
        self.config = self._merge_config(kwargs)
        
        # Setup projection transformers
        self._setup_projections()
        
        # Grid metadata
        self.grid_id: Optional[str] = None
        self._cells: Optional[List[GridCell]] = None
        
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
        
        # Store grid metadata
        grid_type = self.__class__.__name__.lower().replace('grid', '')
        metadata = {
            'description': description,
            'cell_count': self.get_cell_count(),
            'config': self.config
        }
        
        self.grid_id = schema.store_grid_definition(
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
            schema.store_grid_cells_batch(self.grid_id, cells_data)
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