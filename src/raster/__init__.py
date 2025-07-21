"""
Unified raster processing module.

This module provides comprehensive raster data processing capabilities including:
- File format support (GeoTIFF, NetCDF)
- Advanced metadata extraction
- Spatial tiling and indexing  
- Validation and quality checks
- Database integration and caching
- Grid resampling operations

The module integrates database compatibility from the original raster module
with advanced processing capabilities from the raster_data pipeline.
"""

# Core management and coordination
from .manager import RasterManager, raster_manager
from .processor import RasterProcessor
from .metadata import RasterMetadataExtractor

# Advanced loaders (absorbed from raster_data)
from .loaders.base_loader import BaseRasterLoader, RasterMetadata
from .loaders.geotiff_loader import GeoTIFFLoader
from .loaders.metadata_extractor import RasterMetadataExtractor as AdvancedMetadataExtractor

# Validation capabilities (absorbed from raster_data)
from .validators.coverage_validator import CoverageValidator
from .validators.value_validator import ValueValidator

# Catalog management (absorbed from raster_data)
from .catalog import RasterCatalog

__all__ = [
    # Core components (backward compatibility)
    'RasterManager',
    'raster_manager',
    'RasterProcessor', 
    'RasterMetadataExtractor',
    
    # Advanced loaders
    'BaseRasterLoader',
    'RasterMetadata',
    'GeoTIFFLoader',
    'AdvancedMetadataExtractor',
    
    # Validation
    'CoverageValidator',
    'ValueValidator',
    
    # Catalog
    'RasterCatalog',
]
