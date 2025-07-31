# src/raster_data/loaders/lightweight_metadata.py
"""Lightweight metadata extractor that avoids GDAL hangs on large files."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import json
import signal
from datetime import datetime
from contextlib import contextmanager
from osgeo import gdal, osr
import numpy as np
import logging

from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when GDAL operation times out."""
    pass


@contextmanager
def gdal_timeout(seconds: int = 30):
    """Context manager for GDAL operations with timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"GDAL operation timed out after {seconds} seconds")
    
    # Set up the timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Clear the alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class LightweightMetadataExtractor:
    """Extract minimal raster metadata without full file scanning."""
    
    def __init__(self, db_connection: DatabaseManager, 
                 timeout_seconds: int = 30,
                 progress_callback: Optional[Callable[[str, float], None]] = None):
        self.db = db_connection
        self.timeout_seconds = timeout_seconds
        self.progress_callback = progress_callback or (lambda msg, pct: None)
        
    def extract_essential_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract only essential metadata needed for processing.
        
        This method is designed to be fast and not hang on large files.
        """
        logger.info(f"Extracting lightweight metadata from {file_path.name}")
        self.progress_callback(f"Opening {file_path.name}", 10)
        
        try:
            with gdal_timeout(self.timeout_seconds):
                dataset = gdal.Open(str(file_path), gdal.GA_ReadOnly)
                if not dataset:
                    raise ValueError(f"Cannot open raster: {file_path}")
                
                try:
                    metadata = {
                        'file_info': self._extract_file_info_fast(file_path),
                        'spatial_info': self._extract_spatial_info_fast(dataset),
                        'data_info': self._extract_data_info_fast(dataset),
                        'extraction_mode': 'lightweight',
                        'extracted_at': datetime.now().isoformat()
                    }
                    
                    self.progress_callback(f"Metadata extracted for {file_path.name}", 100)
                    logger.info(f"✅ Lightweight metadata extracted successfully")
                    return metadata
                    
                finally:
                    dataset = None
                    
        except TimeoutError as e:
            logger.error(f"❌ GDAL timeout extracting metadata from {file_path}: {e}")
            # Return minimal metadata based on file system info only
            return self._create_minimal_metadata(file_path, error=str(e))
        except Exception as e:
            logger.error(f"❌ Error extracting metadata from {file_path}: {e}")
            return self._create_minimal_metadata(file_path, error=str(e))
    
    def _extract_file_info_fast(self, file_path: Path) -> Dict[str, Any]:
        """Extract file system information quickly."""
        self.progress_callback("Reading file info", 20)
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'format': file_path.suffix,
            'exists': True
        }
    
    def _extract_spatial_info_fast(self, dataset: gdal.Dataset) -> Dict[str, Any]:
        """Extract spatial information without reading data."""
        self.progress_callback("Reading spatial info", 50)
        
        # Get basic dimensions
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands = dataset.RasterCount
        
        # Get geotransform
        transform = dataset.GetGeoTransform()
        
        # Calculate bounds from geotransform
        minx = transform[0]
        maxy = transform[3]
        maxx = minx + width * transform[1]
        miny = maxy + height * transform[5]
        
        # Get CRS information
        proj_ref = dataset.GetProjectionRef()
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromWkt(proj_ref)
        
        # Calculate resolution (assuming square pixels)
        resolution_x = abs(transform[1])
        resolution_y = abs(transform[5])
        
        # Apply tolerance-based clamping for geographic bounds
        # This handles floating point precision issues in GDAL transforms
        tolerance = 0.01  # Same tolerance used in validation
        if spatial_ref.IsGeographic():
            # Clamp to valid geographic ranges with tolerance
            minx = max(-180 - tolerance, min(180 + tolerance, minx))
            maxx = max(-180 - tolerance, min(180 + tolerance, maxx))
            miny = max(-90 - tolerance, min(90 + tolerance, miny))
            maxy = max(-90 - tolerance, min(90 + tolerance, maxy))
            
            # If still slightly outside after clamping, snap to exact bounds
            if minx < -180:
                minx = -180
            if maxx > 180:
                maxx = 180
            if miny < -90:
                miny = -90
            if maxy > 90:
                maxy = 90
        
        return {
            'width': width,
            'height': height,
            'bands': bands,
            'bounds': [minx, miny, maxx, maxy],  # [west, south, east, north]
            'transform': transform,
            'crs': proj_ref,
            'epsg_code': spatial_ref.GetAttrValue('AUTHORITY', 1) if spatial_ref.GetAttrValue('AUTHORITY') else None,
            'resolution_degrees': max(resolution_x, resolution_y)  # Use max for safety
        }
    
    def _extract_data_info_fast(self, dataset: gdal.Dataset) -> Dict[str, Any]:
        """Extract data type information without reading pixel values."""
        self.progress_callback("Reading data info", 80)
        
        band = dataset.GetRasterBand(1)
        data_type = gdal.GetDataTypeName(band.DataType)
        nodata = band.GetNoDataValue()
        
        # Get basic statistics if they're already computed (don't compute them)
        stats = band.GetStatistics(False, False)  # Don't compute, don't force
        
        return {
            'data_type': data_type,
            'nodata_value': nodata,
            'has_precomputed_stats': stats is not None,
            'pixel_count_estimate': dataset.RasterXSize * dataset.RasterYSize,
            'data_size_estimate_mb': (dataset.RasterXSize * dataset.RasterYSize * 
                                    dataset.RasterCount * 4) / (1024 * 1024)  # Assume 4 bytes per pixel
        }
    
    def _create_minimal_metadata(self, file_path: Path, error: str) -> Dict[str, Any]:
        """Create minimal metadata when GDAL operations fail."""
        logger.warning(f"Creating minimal metadata for {file_path} due to error: {error}")
        
        stat = file_path.stat()
        return {
            'file_info': {
                'name': file_path.name,
                'path': str(file_path),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'format': file_path.suffix,
                'exists': True
            },
            'spatial_info': {
                'width': None,
                'height': None,
                'bands': None,
                'bounds': None,
                'transform': None,
                'crs': None,
                'epsg_code': None,
                'resolution_degrees': None
            },
            'data_info': {
                'data_type': None,
                'nodata_value': None,
                'has_precomputed_stats': False,
                'pixel_count_estimate': None,
                'data_size_estimate_mb': stat.st_size / (1024 * 1024)  # Use file size as estimate
            },
            'extraction_mode': 'minimal_fallback',
            'extraction_error': error,
            'extracted_at': datetime.now().isoformat()
        }
    
    def store_lightweight_metadata(self, metadata: Dict[str, Any], 
                                  raster_name: str) -> str:
        """Store lightweight metadata in database with minimal footprint."""
        self.progress_callback("Storing metadata", 90)
        
        with self.db.get_cursor() as cursor:
            # Store in raster_sources table with lightweight flag
            # Insert using actual table schema (name, file_path, data_type, pixel_size_degrees, spatial_extent)
            bounds = metadata['spatial_info'].get('bounds')
            if bounds:
                # Create WKT polygon from bounds [minx, miny, maxx, maxy]
                minx, miny, maxx, maxy = bounds
                spatial_extent_wkt = f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},{minx} {miny}))"
            else:
                spatial_extent_wkt = None
            
            cursor.execute("""
                INSERT INTO raster_sources (
                    name, file_path, data_type, pixel_size_degrees, 
                    spatial_extent, file_size_mb, processing_status, 
                    metadata, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    pixel_size_degrees = EXCLUDED.pixel_size_degrees,
                    spatial_extent = EXCLUDED.spatial_extent,
                    file_size_mb = EXCLUDED.file_size_mb,
                    processing_status = EXCLUDED.processing_status,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (
                raster_name,
                metadata['file_info']['path'],
                'unknown',  # Will be determined later
                metadata['spatial_info'].get('resolution_degrees', 0.016666666666667),
                spatial_extent_wkt,
                metadata['file_info']['size_mb'],
                'pending',  # Use valid enum value  
                json.dumps(metadata),
                datetime.now()
            ))
            
            raster_id = cursor.fetchone()['id']
            logger.info(f"✅ Stored lightweight metadata with ID: {raster_id}")
            return str(raster_id)
    
    def can_extract_full_metadata_later(self, file_path: Path) -> bool:
        """Check if full metadata extraction might succeed later."""
        try:
            stat = file_path.stat()
            # If file is smaller than 1GB, full extraction is probably safe
            size_gb = stat.st_size / (1024**3)
            return size_gb < 1.0
        except Exception:
            return False
    
    def schedule_full_metadata_extraction(self, raster_id: str, 
                                        file_path: Path, priority: int = 5):
        """Schedule full metadata extraction for later processing."""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO processing_queue (
                    queue_type, parameters, priority, status, created_at
                ) VALUES (%s, %s, %s, %s, %s)
            """, (
                'full_metadata_extraction',
                json.dumps({
                    'raster_id': raster_id,
                    'file_path': str(file_path),
                    'extraction_mode': 'full'
                }),
                priority,
                'pending',
                datetime.now()
            ))
            
        logger.info(f"📋 Scheduled full metadata extraction for {file_path.name}")


def extract_metadata_with_progress(file_path: Path, 
                                 db_connection: DatabaseManager,
                                 timeout: int = 30,
                                 progress_callback: Optional[Callable[[str, float], None]] = None,
                                 raster_name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function for extracting lightweight metadata with progress.
    
    Returns:
        Tuple of (raster_id, metadata_dict)
    """
    extractor = LightweightMetadataExtractor(
        db_connection, 
        timeout_seconds=timeout,
        progress_callback=progress_callback
    )
    
    metadata = extractor.extract_essential_metadata(file_path)
    # Use provided name or file stem
    name = raster_name if raster_name is not None else file_path.stem
    raster_id = extractor.store_lightweight_metadata(metadata, name)
    
    # Schedule full extraction for smaller files
    if extractor.can_extract_full_metadata_later(file_path):
        extractor.schedule_full_metadata_extraction(raster_id, file_path)
    
    return raster_id, metadata