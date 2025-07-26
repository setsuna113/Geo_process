"""Unified raster processing module - consolidated from raster/ and raster_data/."""

from .catalog import RasterCatalog, RasterEntry
from .loaders.base_loader import BaseRasterLoader, RasterMetadata, RasterWindow
from .loaders.geotiff_loader import GeoTIFFLoader
from .validators.coverage_validator import CoverageValidator
from .validators.value_validator import ValueValidator

__all__ = [
    'RasterCatalog',
    'RasterEntry', 
    'BaseRasterLoader',
    'RasterMetadata',
    'RasterWindow',
    'GeoTIFFLoader',
    'CoverageValidator',
    'ValueValidator'
]