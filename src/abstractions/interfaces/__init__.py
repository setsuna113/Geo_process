"""Foundation interfaces - pure abstractions with no dependencies."""

from .processor import IProcessor
from .grid import IGrid, GridCell

__all__ = ['IProcessor', 'IGrid', 'GridCell']