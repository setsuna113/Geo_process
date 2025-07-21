# src/raster/loaders/geotiff_loader.py
from pathlib import Path
from typing import Any, Optional, Dict
import numpy as np
from osgeo import gdal, osr
import logging
from functools import lru_cache

from .base_loader import (
    BaseRasterLoader, RasterMetadata, RasterWindow
)
from src.config.config import Config

logger = logging.getLogger(__name__)

# Configure GDAL for optimal performance
gdal.SetCacheMax(512 * 1024 * 1024)  # 512MB cache
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

class GeoTIFFLoader(BaseRasterLoader):
    """Loader for GeoTIFF raster files with optimization for large files."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self._metadata_cache = {}
        
    def can_handle(self, file_path: Path) -> bool:
        """Check if file is a GeoTIFF."""
        suffix = file_path.suffix.lower()
        if suffix in ['.tif', '.tiff', '.gtiff']:
            return True
        
        # Check by driver
        try:
            dataset = gdal.Open(str(file_path), gdal.GA_ReadOnly)
            if dataset:
                driver = dataset.GetDriver()
                dataset = None
                return driver.ShortName == 'GTiff'
        except:
            pass
        
        return False
    
    @lru_cache(maxsize=32)
    def _extract_metadata_cached(self, file_path_str: str) -> RasterMetadata:
        """Cached metadata extraction helper."""
        return self._extract_metadata_impl(Path(file_path_str))
    
    def extract_metadata(self, file_path: Path) -> RasterMetadata:
        """Extract metadata from GeoTIFF without loading data."""
        return self._extract_metadata_cached(str(file_path))
    
    def _extract_metadata_impl(self, file_path: Path) -> RasterMetadata:
        """Extract metadata from GeoTIFF without loading data."""
        if str(file_path) in self._metadata_cache:
            return self._metadata_cache[str(file_path)]
        
        dataset = gdal.Open(str(file_path), gdal.GA_ReadOnly)
        if not dataset:
            raise ValueError(f"Cannot open raster: {file_path}")
        
        try:
            # Get basic properties
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            band_count = dataset.RasterCount
            
            # Get geotransform
            gt = dataset.GetGeoTransform()
            pixel_size = (gt[1], gt[5])
            
            # Calculate bounds
            west = gt[0]
            north = gt[3]
            east = west + width * gt[1]
            south = north + height * gt[5]
            bounds = (west, south, east, north)
            
            # Get CRS
            srs = osr.SpatialReference()
            srs.ImportFromWkt(dataset.GetProjection())
            crs = srs.ExportToProj4()
            
            # Get data type and nodata
            band = dataset.GetRasterBand(1)
            dtype_name = gdal.GetDataTypeName(band.DataType)
            nodata = band.GetNoDataValue()
            
            metadata = RasterMetadata(
                width=width,
                height=height,
                bounds=bounds,
                crs=crs,
                pixel_size=pixel_size,
                data_type=dtype_name,
                nodata_value=nodata,
                band_count=band_count
            )
            
            self._metadata_cache[str(file_path)] = metadata
            
            # Log metadata
            logger.info(f"Loaded metadata for {file_path.name}:")
            logger.info(f"  Size: {width} x {height}")
            logger.info(f"  Resolution: {abs(pixel_size[0]):.6f}°")
            logger.info(f"  Data type: {dtype_name}")
            logger.info(f"  NoData: {nodata}")
            
            return metadata
            
        finally:
            dataset = None
    
    def _open_dataset(self, file_path: Path) -> gdal.Dataset:
        """Open GeoTIFF dataset for reading."""
        # Open with explicit read-only flag and optimal settings
        dataset = gdal.OpenEx(
            str(file_path),
            gdal.GA_ReadOnly,
            open_options=['NUM_THREADS=ALL_CPUS', 'GEOREF_SOURCES=NONE']
        )
        
        if not dataset:
            raise ValueError(f"Cannot open raster: {file_path}")
        
        return dataset
    
    def _read_window(self, dataset: gdal.Dataset, window: RasterWindow, band: int = 1) -> np.ndarray:
        """Read data from a specific window using GDAL."""
        raster_band = dataset.GetRasterBand(band)
        
        # Read data with explicit window parameters
        data = raster_band.ReadAsArray(
            xoff=window.col_off,
            yoff=window.row_off,
            win_xsize=window.width,
            win_ysize=window.height
        )
        
        if data is None:
            raise ValueError(f"Failed to read window: {window}")
        
        return data
    
    def _close_dataset(self, dataset: gdal.Dataset) -> None:
        """Close GDAL dataset."""
        # GDAL datasets are closed by setting to None
        del dataset
    
    def build_overviews(self, file_path: Path, levels: Optional[list] = None) -> None:
        """Build pyramid overviews for faster multi-resolution access."""
        if levels is None:
            levels = [2, 4, 8, 16, 32]
        
        logger.info(f"Building overviews for {file_path.name} at levels: {levels}")
        
        dataset = gdal.Open(str(file_path), gdal.GA_Update)
        if not dataset:
            raise ValueError(f"Cannot open raster for update: {file_path}")
        
        try:
            dataset.BuildOverviews("AVERAGE", levels)
            logger.info("Overviews built successfully")
        finally:
            dataset = None
    
    def create_vrt(self, file_paths: list[Path], output_path: Path) -> None:
        """Create a VRT (Virtual Raster) from multiple GeoTIFF files."""
        vrt_options = gdal.BuildVRTOptions(
            separate=False,
            allowProjectionDifference=False
        )
        
        vrt = gdal.BuildVRT(
            str(output_path),
            [str(p) for p in file_paths],
            options=vrt_options
        )
        
        vrt = None
        logger.info(f"Created VRT: {output_path}")
