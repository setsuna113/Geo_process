# src/processors/data_preparation/raster_cleaner.py
"""Clean and validate raster values for biodiversity analysis."""

import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import xarray as xr
from dataclasses import dataclass
from datetime import datetime

from src.config import config as global_config
from src.infrastructure.processors.base_processor import EnhancedBaseProcessor as BaseProcessor
from src.domain.raster.loaders.geotiff_loader import GeoTIFFLoader
from src.domain.raster.loaders.base_loader import RasterWindow
from src.database.connection import DatabaseManager
import json

logger = logging.getLogger(__name__)

@dataclass
class CleaningStats:
    """Statistics from cleaning operations."""
    total_pixels: int
    nodata_pixels: int
    outliers_removed: int
    negative_values_fixed: int
    capped_values: int
    final_valid_pixels: int
    value_range: Tuple[float, float]
    
    @property
    def cleaning_ratio(self) -> float:
        """Percentage of pixels that required cleaning."""
        if self.total_pixels == 0:
            return 0.0
        cleaned = self.outliers_removed + self.negative_values_fixed + self.capped_values
        return cleaned / self.total_pixels

class RasterCleaner(BaseProcessor):
    """Clean and validate raster data for species richness analysis."""
    
    def __init__(self, db_connection: DatabaseManager, app_config=None):
        # Use global config instance
        config = app_config if app_config is not None else global_config
        super().__init__(batch_size=1000, config=config)
        self.db = db_connection
        self.loader = GeoTIFFLoader(config)
        
        # Species-specific constraints
        self.value_constraints = {
            'plants': {'min': 0, 'max': 20000, 'outlier_std': 4},
            'vertebrates': {'min': 0, 'max': 2000, 'outlier_std': 4},
            'all': {'min': 0, 'max': 25000, 'outlier_std': 5}
        }
        
        # Cleaning parameters from config
        self.tile_size = config.raster_processing.get("tile_size", 512)
        self.log_operations = config.get('data_cleaning', {}).get('log_operations', True)
        
    def clean_raster(self, 
                    raster_path: Path,
                    dataset_type: str = 'all',
                    output_path: Optional[Path] = None,
                    in_place: bool = False) -> Dict[str, Any]:
        """
        Clean a raster file by removing outliers and invalid values.
        
        Args:
            raster_path: Path to input raster
            dataset_type: Type of dataset for constraints ('plants', 'vertebrates', 'all')
            output_path: Path for cleaned output (if None, returns xarray)
            in_place: Whether to modify the input file
            
        Returns:
            Dictionary with cleaned data and statistics
        """
        logger.info(f"Starting cleaning for {raster_path.name} (type: {dataset_type})")
        
        # Get constraints
        constraints = self.value_constraints.get(dataset_type, self.value_constraints['all'])
        
        # Load metadata
        metadata = self.loader.extract_metadata(raster_path)
        
        # Initialize statistics
        stats = CleaningStats(
            total_pixels=0,
            nodata_pixels=0,
            outliers_removed=0,
            negative_values_fixed=0,
            capped_values=0,
            final_valid_pixels=0,
            value_range=(float('inf'), float('-inf'))
        )
        
        # Process in tiles for memory efficiency
        cleaned_data = []
        cleaning_log = []
        
        with self.loader.open_lazy(raster_path) as reader:
            for window, tile_data in self.loader.iter_tiles(raster_path):
                # Clean tile
                cleaned_tile, tile_stats, tile_log = self._clean_tile(
                    tile_data, 
                    metadata.nodata_value,
                    constraints,
                    window
                )
                
                cleaned_data.append((window, cleaned_tile))
                cleaning_log.extend(tile_log)
                
                # Update statistics
                self._update_stats(stats, tile_stats)
        
        # Create output
        if output_path or in_place:
            self._write_cleaned_raster(
                cleaned_data,
                metadata,
                output_path if output_path else raster_path
            )
        
        # Log cleaning operations to database
        if self.log_operations:
            self._log_to_database(raster_path, dataset_type, stats, cleaning_log)
        
        # Create result
        result = {
            'statistics': stats,
            'metadata': {
                'dataset_type': dataset_type,
                'constraints_applied': constraints,
                'processing_date': datetime.now().isoformat()
            }
        }
        
        # Add xarray if not writing to file
        if not output_path and not in_place:
            result['data'] = self._tiles_to_xarray(cleaned_data, metadata)
        
        logger.info(f"Cleaning complete. {stats.cleaning_ratio:.1%} of pixels modified")
        
        return result
    
    def _clean_tile(self, 
                   tile_data: np.ndarray,
                   nodata_value: Optional[float],
                   constraints: Dict[str, Any],
                   window: RasterWindow) -> Tuple[np.ndarray, CleaningStats, List[Dict]]:
        """Clean a single tile of data."""
        tile_log = []
        cleaned = tile_data.copy()
        
        # Initialize tile statistics
        stats = CleaningStats(
            total_pixels=tile_data.size,
            nodata_pixels=0,
            outliers_removed=0,
            negative_values_fixed=0,
            capped_values=0,
            final_valid_pixels=0,
            value_range=(float('inf'), float('-inf'))
        )
        
        # Handle NoData
        if nodata_value is not None:
            nodata_mask = tile_data == nodata_value
            stats.nodata_pixels = np.sum(nodata_mask)
            valid_mask = ~nodata_mask
        else:
            valid_mask = np.ones_like(tile_data, dtype=bool)
        
        if np.any(valid_mask):
            valid_data = cleaned[valid_mask]
            
            # Fix negative values
            negative_mask = valid_data < 0
            if np.any(negative_mask):
                stats.negative_values_fixed = int(np.sum(negative_mask))
                valid_data[negative_mask] = 0
                tile_log.append({
                    'operation': 'fix_negative',
                    'window': window,
                    'count': int(stats.negative_values_fixed)
                })
            
            # Remove statistical outliers
            if constraints['outlier_std'] > 0:
                mean = np.mean(valid_data)
                std = np.std(valid_data)
                outlier_threshold = mean + constraints['outlier_std'] * std
                
                outlier_mask = valid_data > outlier_threshold
                if np.any(outlier_mask):
                    stats.outliers_removed = np.sum(outlier_mask)
                    valid_data[outlier_mask] = outlier_threshold
                    tile_log.append({
                        'operation': 'remove_outliers',
                        'window': window,
                        'count': int(stats.outliers_removed),
                        'threshold': float(outlier_threshold)
                    })
            
            # Apply min/max constraints
            below_min = valid_data < constraints['min']
            above_max = valid_data > constraints['max']
            
            if np.any(below_min):
                valid_data[below_min] = constraints['min']
                stats.capped_values += np.sum(below_min)
            
            if np.any(above_max):
                valid_data[above_max] = constraints['max']
                stats.capped_values += np.sum(above_max)
            
            if stats.capped_values > 0:
                tile_log.append({
                    'operation': 'apply_constraints',
                    'window': window,
                    'count': int(stats.capped_values),
                    'min_constraint': constraints['min'],
                    'max_constraint': constraints['max']
                })
            
            # Update cleaned data
            cleaned[valid_mask] = valid_data
            
            # Calculate final statistics
            stats.final_valid_pixels = np.sum(valid_mask)
            stats.value_range = (float(np.min(valid_data)), float(np.max(valid_data)))
        
        return cleaned, stats, tile_log
    
    def _update_stats(self, total_stats: CleaningStats, tile_stats: CleaningStats):
        """Update total statistics with tile statistics."""
        total_stats.total_pixels += tile_stats.total_pixels
        total_stats.nodata_pixels += tile_stats.nodata_pixels
        total_stats.outliers_removed += tile_stats.outliers_removed
        total_stats.negative_values_fixed += tile_stats.negative_values_fixed
        total_stats.capped_values += tile_stats.capped_values
        total_stats.final_valid_pixels += tile_stats.final_valid_pixels
        
        if tile_stats.final_valid_pixels > 0:
            total_stats.value_range = (
                min(total_stats.value_range[0], tile_stats.value_range[0]),
                max(total_stats.value_range[1], tile_stats.value_range[1])
            )
    
    def _write_cleaned_raster(self, cleaned_data: List[Tuple[RasterWindow, np.ndarray]], 
                            metadata, output_path: Path):
        """Write cleaned data back to raster file."""
        from osgeo import gdal
        
        # Create output dataset
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            str(output_path),
            metadata.width,
            metadata.height,
            1,
            gdal.GDT_Float32,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        
        # Copy spatial reference
        out_ds.SetProjection(metadata.crs)
        out_ds.SetGeoTransform(self._metadata_to_geotransform(metadata))
        
        # Write tiles
        band = out_ds.GetRasterBand(1)
        band.SetNoDataValue(metadata.nodata_value)
        
        for window, data in cleaned_data:
            band.WriteArray(
                data,
                xoff=window.col_off,
                yoff=window.row_off
            )
        
        # Compute statistics
        band.ComputeStatistics(False)
        band.FlushCache()
        out_ds = None
        
        logger.info(f"Cleaned raster written to {output_path}")
    
    def _tiles_to_xarray(self, tiles: List[Tuple[RasterWindow, np.ndarray]], 
                        metadata) -> xr.DataArray:
        """Convert tiles back to xarray DataArray."""
        # Reconstruct full array
        full_array = np.full((metadata.height, metadata.width), 
                           metadata.nodata_value, dtype=np.float32)
        
        for window, data in tiles:
            full_array[window.slice] = data
        
        # Create coordinate arrays
        west, south, east, north = metadata.bounds
        lons = np.linspace(west, east, metadata.width)
        lats = np.linspace(north, south, metadata.height)
        
        # Create xarray
        da = xr.DataArray(
            full_array,
            coords={'lat': lats, 'lon': lons},
            dims=['lat', 'lon'],
            attrs={
                'crs': metadata.crs,
                'nodata': metadata.nodata_value,
                'cleaned': True,
                'cleaning_date': datetime.now().isoformat()
            }
        )
        
        return da
    
    def _metadata_to_geotransform(self, metadata):
        """Convert metadata to GDAL geotransform."""
        west, south, east, north = metadata.bounds
        pixel_width = (east - west) / metadata.width
        pixel_height = (north - south) / metadata.height
        return [west, pixel_width, 0, north, 0, -pixel_height]
    
    def _log_to_database(self, raster_path: Path, dataset_type: str, 
                        stats: CleaningStats, operations: List[Dict]):
        """Log cleaning operations to database."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            
            # Insert cleaning record
            cur.execute("""
                INSERT INTO data_cleaning_log 
                (raster_name, dataset_type, total_pixels, pixels_cleaned, 
                 cleaning_ratio, value_range, operations, processing_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                str(raster_path.name),
                dataset_type,
                int(stats.total_pixels),
                int(stats.outliers_removed + stats.negative_values_fixed + stats.capped_values),
                float(stats.cleaning_ratio),
                list([float(v) if isinstance(v, (np.number, np.floating)) else v for v in stats.value_range]),
                json.dumps([
                    {**op, 'window': {'col_off': op['window'].col_off, 'row_off': op['window'].row_off, 
                                      'width': op['window'].width, 'height': op['window'].height} 
                     if 'window' in op and hasattr(op['window'], 'col_off') else op.get('window', {})}
                    for op in operations
                ]),
                datetime.now()
            ))
            
            cleaning_id = cur.fetchone()[0]
            conn.commit()
            
            logger.debug(f"Cleaning operations logged with ID: {cleaning_id}")
    def process_single(self, item: Any) -> Any:
        """Process a single item - implementation depends on specific use case."""
        return item
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input item."""
        return True, None

