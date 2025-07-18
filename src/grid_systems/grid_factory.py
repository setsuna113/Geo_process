# src/grid_systems/grid_factory.py
"""Factory for creating and managing grid systems."""

from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

from ..base import BaseGrid
from ..core.registry import component_registry
from ..config import config
from ..database.schema import schema
from .bounds_manager import BoundsManager, BoundsDefinition

logger = logging.getLogger(__name__)

@dataclass
class GridSpecification:
    """Specification for grid creation."""
    grid_type: str
    resolution: int
    bounds: Union[str, BoundsDefinition]
    crs: str = "EPSG:4326"
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'grid_type': self.grid_type,
            'resolution': self.resolution,
            'bounds': self.bounds.name if isinstance(self.bounds, BoundsDefinition) else self.bounds,
            'crs': self.crs,
            'name': self.name,
            'description': self.description,
            'metadata': self.metadata
        }


class GridFactory:
    """
    Factory for creating and managing multi-resolution grid systems.
    
    Handles:
    - Grid creation from specifications
    - Resolution hierarchy management
    - Grid storage and retrieval
    - Upscaling/downscaling between resolutions
    """
    
    # Standard resolutions for biodiversity analysis (in meters)
    STANDARD_RESOLUTIONS = {
        'coarse': 100000,    # 100 km
        'medium': 50000,     # 50 km
        'fine': 25000,       # 25 km
        'very_fine': 10000,  # 10 km
        'ultra_fine': 5000   # 5 km
    }
    
    def __init__(self):
        """Initialize grid factory."""
        self.bounds_manager = BoundsManager()
        self._grid_cache: Dict[str, BaseGrid] = {}
        
    def create_grid(self, spec: Union[GridSpecification, Dict]) -> BaseGrid:
        """
        Create a grid from specification.
        
        Args:
            spec: Grid specification
            
        Returns:
            Created grid instance
        """
        if isinstance(spec, dict):
            spec = GridSpecification(**spec)
            
        # Get grid class from registry
        try:
            grid_class = component_registry.grids.get(f"{spec.grid_type.title()}Grid")
        except KeyError:
            available = component_registry.grids.list_registered()
            raise ValueError(f"Unknown grid type: {spec.grid_type}. Available: {available}")
            
        # Get bounds
        if isinstance(spec.bounds, str):
            bounds_def = self.bounds_manager.get_bounds(spec.bounds)
        else:
            bounds_def = spec.bounds
            
        # Create grid
        logger.info(f"Creating {spec.grid_type} grid at {spec.resolution}m resolution")
        
        grid = grid_class(
            resolution=spec.resolution,
            bounds=bounds_def,
            crs=spec.crs,
            **(spec.metadata or {})
        )
        
        # Store specification in grid metadata
        grid.specification = spec
        
        return grid
    
    def create_multi_resolution_grids(self,
                                     grid_type: str,
                                     resolutions: List[int],
                                     bounds: Union[str, BoundsDefinition],
                                     base_name: str) -> Dict[int, BaseGrid]:
        """
        Create multiple grids at different resolutions.
        
        Args:
            grid_type: Type of grid (cubic, hexagonal)
            resolutions: List of resolutions in meters
            bounds: Bounds specification
            base_name: Base name for grids
            
        Returns:
            Dictionary mapping resolution to grid
        """
        grids = {}
        
        for resolution in sorted(resolutions, reverse=True):  # Coarse to fine
            spec = GridSpecification(
                grid_type=grid_type,
                resolution=resolution,
                bounds=bounds,
                name=f"{base_name}_{resolution}m",
                description=f"{grid_type.title()} grid at {resolution}m resolution",
                metadata={'multi_resolution_set': base_name}
            )
            
            grid = self.create_grid(spec)
            grids[resolution] = grid
            
            # Cache the grid
            self._grid_cache[spec.name] = grid
            
        logger.info(f"Created {len(grids)} grids for {base_name}")
        return grids
    
    def store_grid(self, grid: BaseGrid, overwrite: bool = False) -> str:
        """
        Store grid in database.
        
        Args:
            grid: Grid to store
            overwrite: Whether to overwrite existing grid
            
        Returns:
            Grid ID
        """
        if not hasattr(grid, 'specification') or not grid.specification.name:
            raise ValueError("Grid must have a specification with name")
            
        # Check if grid exists
        existing = schema.get_grid_by_name(grid.specification.name)
        if existing and not overwrite:
            raise ValueError(f"Grid '{grid.specification.name}' already exists")
        elif existing:
            logger.warning(f"Overwriting existing grid '{grid.specification.name}'")
            schema.delete_grid(grid.specification.name)
            
        # Store grid
        return grid.store_grid(
            name=grid.specification.name,
            description=grid.specification.description or ""
        )
    
    def load_grid(self, name: str) -> Optional[BaseGrid]:
        """
        Load grid from database.
        
        Args:
            name: Grid name
            
        Returns:
            Grid instance or None
        """
        # Check cache first
        if name in self._grid_cache:
            return self._grid_cache[name]
            
        # Load from database
        grid_info = schema.get_grid_by_name(name)
        if not grid_info:
            return None
            
        # Recreate grid
        spec = GridSpecification(
            grid_type=grid_info['grid_type'],
            resolution=grid_info['resolution'],
            bounds=self.bounds_manager.get_bounds('global'),  # TODO: Store bounds
            name=name,
            metadata=grid_info.get('metadata', {})
        )
        
        grid = self.create_grid(spec)
        grid.grid_id = grid_info['id']
        
        # Cache it
        self._grid_cache[name] = grid
        
        return grid
    
    def get_resolution_hierarchy(self, base_name: str) -> Dict[int, str]:
        """
        Get all grids in a resolution hierarchy.
        
        Args:
            base_name: Base name of grid set
            
        Returns:
            Dictionary mapping resolution to grid name
        """
        hierarchy = {}
        
        # Query database for matching grids
        with schema.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT name, resolution, metadata
                FROM grids
                WHERE metadata->>'multi_resolution_set' = %s
                ORDER BY resolution DESC
            """, (base_name,))
            
            for row in cursor:
                hierarchy[row['resolution']] = row['name']
                
        return hierarchy
    
    def create_standard_grids(self,
                             grid_type: str = 'cubic',
                             bounds: str = 'global',
                             name_prefix: str = 'bio') -> Dict[str, BaseGrid]:
        """
        Create standard resolution grids for biodiversity analysis.
        
        Args:
            grid_type: Type of grid
            bounds: Bounds name
            name_prefix: Prefix for grid names
            
        Returns:
            Dictionary of resolution name to grid
        """
        resolutions = list(self.STANDARD_RESOLUTIONS.values())
        
        grids_by_res = self.create_multi_resolution_grids(
            grid_type=grid_type,
            resolutions=resolutions,
            bounds=bounds,
            base_name=f"{name_prefix}_{grid_type}_{bounds}"
        )
        
        # Map by name
        grids_by_name = {}
        for name, resolution in self.STANDARD_RESOLUTIONS.items():
            if resolution in grids_by_res:
                grids_by_name[name] = grids_by_res[resolution]
                
        return grids_by_name
    
    def upscale_data(self,
                     data: Dict[str, float],
                     source_grid: BaseGrid,
                     target_grid: BaseGrid,
                     aggregation: str = 'mean') -> Dict[str, float]:
        """
        Upscale data from fine to coarse resolution.
        
        Args:
            data: Dictionary mapping cell_id to value
            source_grid: Source (fine) resolution grid
            target_grid: Target (coarse) resolution grid
            aggregation: Aggregation method (mean, sum, max, min)
            
        Returns:
            Upscaled data
        """
        if source_grid.resolution >= target_grid.resolution:
            raise ValueError("Source resolution must be finer than target")
            
        upscaled = {}
        
        # For each target cell, find overlapping source cells
        for target_cell in target_grid.get_cells():
            overlapping_values = []
            
            for source_cell_id, value in data.items():
                source_cell = source_grid.get_cell_by_id(source_cell_id)
                if source_cell and target_cell.geometry.intersects(source_cell.geometry):
                    # Weight by intersection area
                    intersection = target_cell.geometry.intersection(source_cell.geometry)
                    weight = intersection.area / source_cell.geometry.area
                    overlapping_values.append((value, weight))
                    
            if overlapping_values:
                if aggregation == 'mean':
                    total_weight = sum(w for _, w in overlapping_values)
                    upscaled[target_cell.cell_id] = sum(v * w for v, w in overlapping_values) / total_weight
                elif aggregation == 'sum':
                    upscaled[target_cell.cell_id] = sum(v * w for v, w in overlapping_values)
                elif aggregation == 'max':
                    upscaled[target_cell.cell_id] = max(v for v, _ in overlapping_values)
                elif aggregation == 'min':
                    upscaled[target_cell.cell_id] = min(v for v, _ in overlapping_values)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")
                    
        return upscaled
    
    def validate_grid_compatibility(self, grid1: BaseGrid, grid2: BaseGrid) -> bool:
        """Check if two grids are compatible for operations."""
        # Same CRS
        if grid1.crs != grid2.crs:
            logger.warning(f"Grids have different CRS: {grid1.crs} vs {grid2.crs}")
            return False
            
        # Overlapping bounds
        bounds1 = BoundsDefinition('grid1', grid1.bounds)
        bounds2 = BoundsDefinition('grid2', grid2.bounds)
        
        if not bounds1.intersects(grid2.bounds):
            logger.warning("Grids do not overlap")
            return False
            
        return True


# Convenience functions
def create_standard_grids(grid_type: str = 'cubic',
                         bounds: str = 'global',
                         store: bool = True) -> Dict[str, BaseGrid]:
    """
    Create and optionally store standard resolution grids.
    
    Args:
        grid_type: Type of grid
        bounds: Bounds specification
        store: Whether to store in database
        
    Returns:
        Dictionary of grids by resolution name
    """
    factory = GridFactory()
    grids = factory.create_standard_grids(grid_type, bounds)
    
    if store:
        for name, grid in grids.items():
            try:
                factory.store_grid(grid)
                logger.info(f"Stored {name} grid")
            except ValueError as e:
                logger.warning(f"Could not store {name} grid: {e}")
                
    return grids


def get_or_create_grid(name: str,
                      grid_type: str,
                      resolution: int,
                      bounds: Union[str, BoundsDefinition] = 'global') -> BaseGrid:
    """
    Get existing grid or create new one.
    
    Args:
        name: Grid name
        grid_type: Type of grid
        resolution: Resolution in meters
        bounds: Bounds specification
        
    Returns:
        Grid instance
    """
    factory = GridFactory()
    
    # Try to load existing
    grid = factory.load_grid(name)
    if grid:
        return grid
        
    # Create new
    spec = GridSpecification(
        grid_type=grid_type,
        resolution=resolution,
        bounds=bounds,
        name=name
    )
    
    grid = factory.create_grid(spec)
    factory.store_grid(grid)
    
    return grid