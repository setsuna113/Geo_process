# src/grid_systems/__init__.py
"""Grid system implementations."""

from .bounds_manager import BoundsManager, BoundsDefinition
from .cubic_grid import CubicGrid
from .hexagonal_grid import HexagonalGrid
from .grid_factory import (
    GridFactory, 
    GridSpecification,
    create_standard_grids,
    get_or_create_grid
)

__all__ = [
    'BoundsManager',
    'BoundsDefinition', 
    'CubicGrid',
    'HexagonalGrid',
    'GridFactory',
    'GridSpecification',
    'create_standard_grids',
    'get_or_create_grid'
]

# Register grids with component registry on import
# (This happens automatically with decorators)