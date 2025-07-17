"""
Geoprocessing package for biodiversity analysis.

This package provides tools for processing spatial species data,
managing geographic grids, and analyzing biodiversity patterns.
"""

__version__ = "1.0.0"
__author__ = "Jason"
__description__ = "Geoprocessing tools for biodiversity analysis"

# Main modules
from . import config
from . import database
from . import grid_systems

__all__ = [
    'config',
    'database', 
    'grid_systems',
]
