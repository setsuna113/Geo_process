# src/raster/loaders/metadata_extractor.py
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
from osgeo import gdal, osr
import numpy as np

from .base_loader import RasterMetadata
from src.database.connection import DatabaseManager
from src.database.utils import DatabaseSchemaUtils

class RasterMetadataExtractor:
    """Extract and store comprehensive metadata from raster files."""
    
    def __init__(self, db_connection: DatabaseManager):
        self.db = db_connection
        self.schema_utils = DatabaseSchemaUtils(db_connection)
        
    def extract_full_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract comprehensive metadata from raster."""
        dataset = gdal.Open(str(file_path), gdal.GA_ReadOnly)
        if not dataset:
            raise ValueError(f"Cannot open raster: {file_path}")
        
        try:
            metadata = {
                'file_info': self._extract_file_info(file_path),
                'spatial_info': self._extract_spatial_info(dataset),
                'data_info': self._extract_data_info(dataset),
                'statistics': self._extract_statistics(dataset),
                'pyramid_info': self._extract_pyramid_info(dataset)
            }
            
            return metadata
            
        finally:
            del dataset
    
    def _extract_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract file system information."""
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'format': file_path.suffix
        }
    
    def _extract_spatial_info(self, dataset: gdal.Dataset) -> Dict[str, Any]:
        """Extract spatial reference information."""
        gt = dataset.GetGeoTransform()
        
        # Get CRS info
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjection())
        
        # Calculate bounds
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        
        corners = {
            'upper_left': (gt[0], gt[3]),
            'upper_right': (gt[0] + width * gt[1], gt[3]),
            'lower_left': (gt[0], gt[3] + height * gt[5]),
            'lower_right': (gt[0] + width * gt[1], gt[3] + height * gt[5])
        }
        
        return {
            'crs': {
                'proj4': srs.ExportToProj4(),
                'wkt': srs.ExportToWkt(),
                'epsg': srs.GetAttrValue('AUTHORITY', 1) if srs.IsProjected() or srs.IsGeographic() else None
            },
            'geotransform': list(gt),
            'pixel_size': {
                'x': abs(gt[1]),
                'y': abs(gt[5]),
                'units': 'degrees' if srs.IsGeographic() else 'meters'
            },
            'extent': {
                'west': min(corners['upper_left'][0], corners['lower_left'][0]),
                'east': max(corners['upper_right'][0], corners['lower_right'][0]),
                'south': min(corners['lower_left'][1], corners['lower_right'][1]),
                'north': max(corners['upper_left'][1], corners['upper_right'][1])
            },
            'corners': corners,
            'dimensions': {
                'width': width,
                'height': height,
                'bands': dataset.RasterCount
            }
        }
    
    def _extract_data_info(self, dataset: gdal.Dataset) -> Dict[str, Any]:
        """Extract data type and band information."""
        bands_info = []
        
        for i in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(i)
            
            band_info = {
                'band_number': i,
                'data_type': gdal.GetDataTypeName(band.DataType),
                'data_type_size': gdal.GetDataTypeSize(band.DataType),
                'nodata_value': band.GetNoDataValue(),
                'color_interpretation': gdal.GetColorInterpretationName(band.GetColorInterpretation()),
                'has_overview': band.GetOverviewCount() > 0,
                'overview_count': band.GetOverviewCount()
            }
            
            bands_info.append(band_info)
        
        return {
            'bands': bands_info,
            'interleave': dataset.GetMetadata().get('INTERLEAVE', 'PIXEL')
        }
    
    def _extract_statistics(self, dataset: gdal.Dataset, sample_size: int = 10000) -> Dict[str, Any]:
        """Extract or compute band statistics."""
        stats_info = []
        
        for i in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(i)
            
            # Try to get existing statistics
            stats = band.GetStatistics(False, False)
            
            if stats is None:
                # GetStatistics returned None - compute from sample
                stats = self._compute_sample_statistics(band, sample_size)
            elif len(stats) >= 4 and stats[0] is None:
                # Statistics exist but are invalid - compute from sample
                stats = self._compute_sample_statistics(band, sample_size)
            
            # Ensure we have valid statistics
            if stats is None or len(stats) < 4:
                # Fallback to safe default values
                stats = [0.0, 0.0, 0.0, 0.0]
            
            stats_info.append({
                'band': i,
                'min': stats[0] if stats[0] is not None else 0.0,
                'max': stats[1] if stats[1] is not None else 0.0,
                'mean': stats[2] if stats[2] is not None else 0.0,
                'std_dev': stats[3] if stats[3] is not None else 0.0,
                'computed': stats[0] is not None
            })
        
        return {'band_statistics': stats_info}
    
    def _compute_sample_statistics(self, band: gdal.Band, sample_size: int) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Compute statistics from a sample of the data."""
        width = band.XSize
        height = band.YSize
        
        # Calculate sample interval
        total_pixels = width * height
        interval = max(1, int(np.sqrt(total_pixels / sample_size)))
        
        # Read sampled data
        sampled_data = band.ReadAsArray(0, 0, width, height, 
                                       buf_xsize=width//interval, 
                                       buf_ysize=height//interval)
        
        if sampled_data is None:
            return (None, None, None, None)
        
        # Remove nodata values
        nodata = band.GetNoDataValue()
        if nodata is not None:
            valid_data = sampled_data[sampled_data != nodata]
        else:
            valid_data = sampled_data.flatten()
        
        if len(valid_data) == 0:
            return (None, None, None, None)
        
        return (
            float(np.min(valid_data)),
            float(np.max(valid_data)),
            float(np.mean(valid_data)),
            float(np.std(valid_data))
        )
    
    def _extract_pyramid_info(self, dataset: gdal.Dataset) -> Dict[str, Any]:
        """Extract information about overview pyramids."""
        band = dataset.GetRasterBand(1)
        overview_count = band.GetOverviewCount()
        
        overviews = []
        for i in range(overview_count):
            overview = band.GetOverview(i)
            overviews.append({
                'level': i,
                'width': overview.XSize,
                'height': overview.YSize,
                'scale': dataset.RasterXSize / overview.XSize
            })
        
        return {
            'has_overviews': overview_count > 0,
            'overview_count': overview_count,
            'overviews': overviews
        }
    
    def store_in_database(self, metadata: Dict[str, Any], raster_name: str) -> int:
        """Store metadata in database and return raster_id."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            
            # Get schema-aware column names
            geometry_col = self.schema_utils.get_geometry_column('raster_sources')
            active_col = self.schema_utils.get_active_column('raster_sources')
            metadata_col = self.schema_utils.get_metadata_column('raster_sources')
            
            # Build INSERT query with schema-aware column names
            query = f"""
                INSERT INTO raster_sources 
                (name, file_path, pixel_size_degrees, data_type, nodata_value, 
                 {geometry_col}, file_size_mb, {metadata_col}, {active_col})
                VALUES (%s, %s, %s, %s, %s, ST_MakeEnvelope(%s, %s, %s, %s, 4326), %s, %s, true)
                RETURNING id
            """
            
            cur.execute(query, (
                raster_name,
                metadata['file_info']['path'],
                metadata['spatial_info']['pixel_size']['x'],
                metadata['data_info']['bands'][0]['data_type'],
                metadata['data_info']['bands'][0]['nodata_value'],
                metadata['spatial_info']['extent']['west'],
                metadata['spatial_info']['extent']['south'],
                metadata['spatial_info']['extent']['east'],
                metadata['spatial_info']['extent']['north'],
                metadata['file_info']['size_mb'],
                json.dumps(metadata)
            ))
            
            raster_id = cur.fetchone()[0]
            conn.commit()
            
            return raster_id
