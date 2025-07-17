# src/grid_systems/cubic_grid.py
from ..core.registry import component_registry
from .base_grid import BaseGrid

@component_registry.grids.register_decorator()
class CubicGrid(BaseGrid):
    """Cubic grid system implementation."""
    # ... existing implementation

# src/grid_systems/hexagonal_grid.py  
from ..core.registry import component_registry
from .base_grid import BaseGrid

@component_registry.grids.register_decorator()
class HexagonalGrid(BaseGrid):
    """Hexagonal grid system implementation."""
    # ... existing implementation