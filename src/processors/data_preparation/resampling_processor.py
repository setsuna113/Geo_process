# src/processors/data_preparation/resampling_processor.py
"""Processor for resampling datasets to uniform resolution."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import xarray as xr
from dataclasses import dataclass
from datetime import datetime
import json

from src.config.config import Config
from src.base.processor import BaseProcessor
from src.database.connection import DatabaseManager
from src.raster_data.catalog import RasterCatalog, RasterEntry
from src.resampling.engines.base_resampler import ResamplingConfig
from src.resampling.engines.numpy_resampler import NumpyResampler
from src.resampling.engines.gdal_resampler import GDALResampler
from src.resampling.cache_manager import ResamplingCacheManager

logger = logging.getLogger(__name__)


@dataclass
class ResampledDatasetInfo:
    """Information about a resampled dataset."""
    name: str
    source_path: Path
    target_resolution: float
    target_crs: str
    bounds: Tuple[float, float, float, float]
    shape: Tuple[int, int]
    data_type: str
    resampling_method: str
    band_name: str
    metadata: Dict[str, Any]


class ResamplingProcessor(BaseProcessor):
    """Handles resampling of datasets to target resolution with database storage."""
    
    def __init__(self, config: Config, db_connection: DatabaseManager):
        super().__init__(batch_size=1000, config=config)
        self.db = db_connection
        self.config = config  # Store the original Config object
        self.catalog = RasterCatalog(db_connection, config)
        self.cache_manager = ResamplingCacheManager()
        
        # Get resampling configuration
        self.resampling_config = config.get('resampling', {})
        self.target_resolution = self.resampling_config.get('target_resolution', 0.05)
        self.target_crs = self.resampling_config.get('target_crs', 'EPSG:4326')
        self.strategies = self.resampling_config.get('strategies', {})
        self.engine_type = self.resampling_config.get('engine', 'numpy')
        
        # Dataset configuration
        self.datasets_config = config.get('datasets', {}).get('target_datasets', [])
        
        logger.info(f"ResamplingProcessor initialized with target resolution: {self.target_resolution}")
        logger.info(f"Using {self.engine_type} engine with {len(self.datasets_config)} configured datasets")
    
    def _create_resampler_engine(self, method: str, source_bounds: Tuple[float, float, float, float]) -> Union[NumpyResampler, GDALResampler]:
        """Create resampler engine with appropriate configuration."""
        # Create resampling config
        resampling_config = ResamplingConfig(
            source_resolution=self._estimate_source_resolution(source_bounds),
            target_resolution=self.target_resolution,
            method=method,
            bounds=source_bounds,
            source_crs=self.target_crs,  # Assume source is in target CRS for now
            target_crs=self.target_crs,
            chunk_size=self.resampling_config.get('chunk_size', 1000),
            cache_results=self.resampling_config.get('cache_resampled', True),
            validate_output=self.resampling_config.get('validate_output', True),
            preserve_sum=self.resampling_config.get('preserve_sum', True)
        )
        
        # Create engine
        if self.engine_type == 'gdal':
            return GDALResampler(resampling_config)
        else:
            return NumpyResampler(resampling_config)
    
    def _estimate_source_resolution(self, bounds: Tuple[float, float, float, float]) -> float:
        """Estimate source resolution from bounds - placeholder implementation."""
        # This is a simple estimation - in practice you'd get this from raster metadata
        minx, miny, maxx, maxy = bounds
        # Assume square pixels for simplification
        return abs(maxx - minx) / 1000  # Rough estimate
    
    def resample_dataset(self, dataset_config: dict) -> ResampledDatasetInfo:
        """
        Resample single dataset and store in database.
        
        Args:
            dataset_config: Dataset configuration dict with name, path_key, data_type, etc.
            
        Returns:
            ResampledDatasetInfo object with metadata
        """
        dataset_name = dataset_config['name']
        path_key = dataset_config['path_key']
        data_type = dataset_config['data_type']
        band_name = dataset_config['band_name']
        
        logger.info(f"Resampling dataset: {dataset_name}")
        
        # Get resampling method for this data type
        method = self.strategies.get(data_type, 'bilinear')
        
        # Load source raster from catalog
        try:
            raster_entry = self.catalog.get_raster(dataset_name)
            if raster_entry is None:
                # Try to add it to catalog first
                data_files = self.config.get('data_files', {})
                if path_key not in data_files:
                    raise ValueError(f"Path key '{path_key}' not found in data_files config")
                
                data_dir = Path(self.config.get('paths.data_dir', 'data'))
                raster_path = data_dir / data_files[path_key]
                
                if not raster_path.exists():
                    raise FileNotFoundError(f"Raster file not found: {raster_path}")
                
                # Add to catalog
                raster_entry = self.catalog.add_raster(
                    raster_path,
                    dataset_type=data_type,
                    validate=True
                )
                logger.info(f"Added {dataset_name} to catalog")
        
        except Exception as e:
            logger.error(f"Failed to load raster {dataset_name}: {e}")
            raise
        
        # Load raster data
        source_data = self._load_raster_data(raster_entry.path)
        source_bounds = raster_entry.bounds
        
        # Create resampler
        resampler = self._create_resampler_engine(method, source_bounds)
        
        # Progress callback
        def progress_callback(percent: float):
            if percent % 10 == 0:  # Log every 10%
                logger.info(f"Resampling {dataset_name}: {percent:.0f}% complete")
        
        # Perform resampling
        logger.info(f"Resampling {dataset_name} using {method} method")
        result = resampler.resample(
            source_data=source_data,
            source_bounds=source_bounds,
            progress_callback=progress_callback
        )
        
        # Create resampled dataset info
        resampled_info = ResampledDatasetInfo(
            name=dataset_name,
            source_path=raster_entry.path,
            target_resolution=self.target_resolution,
            target_crs=self.target_crs,
            bounds=result.bounds,
            shape=result.data.shape,
            data_type=data_type,
            resampling_method=method,
            band_name=band_name,
            metadata={
                'source_resolution': resampler.config.source_resolution,
                'scale_factor': getattr(resampler, 'scale_factor', 1.0),
                'engine': self.engine_type,
                'coverage_available': result.coverage_map is not None,
                'resampled_at': datetime.now().isoformat()
            }
        )
        
        # Store in database
        self._store_resampled_dataset(resampled_info, result.data)
        
        logger.info(f"✅ Successfully resampled {dataset_name} to {self.target_resolution}° resolution")
        return resampled_info
    
    def resample_all_datasets(self) -> List[ResampledDatasetInfo]:
        """Resample all configured datasets."""
        resampled_datasets = []
        
        # Filter enabled datasets
        enabled_datasets = [ds for ds in self.datasets_config if ds.get('enabled', True)]
        
        logger.info(f"Resampling {len(enabled_datasets)} datasets to uniform resolution")
        
        for i, dataset_config in enumerate(enabled_datasets, 1):
            logger.info(f"Processing dataset {i}/{len(enabled_datasets)}: {dataset_config['name']}")
            
            try:
                resampled_info = self.resample_dataset(dataset_config)
                resampled_datasets.append(resampled_info)
                
            except Exception as e:
                logger.error(f"Failed to resample {dataset_config['name']}: {e}")
                # Continue with other datasets
                continue
        
        logger.info(f"✅ Completed resampling pipeline: {len(resampled_datasets)}/{len(enabled_datasets)} datasets processed")
        return resampled_datasets
    
    def get_resampled_dataset(self, name: str) -> Optional[ResampledDatasetInfo]:
        """Retrieve resampled dataset info from database."""
        try:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    SELECT name, source_path, target_resolution, target_crs, 
                           bounds, shape_height, shape_width, data_type, 
                           resampling_method, band_name, metadata
                    FROM resampled_datasets 
                    WHERE name = %s AND created_at = (
                        SELECT MAX(created_at) FROM resampled_datasets WHERE name = %s
                    )
                """, (name, name))
                
                row = cur.fetchone()
                if row is None:
                    return None
                
                return ResampledDatasetInfo(
                    name=row[0],
                    source_path=Path(row[1]),
                    target_resolution=row[2],
                    target_crs=row[3],
                    bounds=tuple(row[4]),  # Assuming bounds stored as array
                    shape=(row[5], row[6]),
                    data_type=row[7],
                    resampling_method=row[8],
                    band_name=row[9],
                    metadata=row[10] or {}
                )
                
        except Exception as e:
            logger.error(f"Failed to retrieve resampled dataset {name}: {e}")
            return None
    
    def load_resampled_data(self, name: str) -> Optional[np.ndarray]:
        """Load resampled data array from database."""
        try:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                
                # Get table name
                cur.execute("""
                    SELECT data_table_name FROM resampled_datasets 
                    WHERE name = %s AND created_at = (
                        SELECT MAX(created_at) FROM resampled_datasets WHERE name = %s
                    )
                """, (name, name))
                
                row = cur.fetchone()
                if row is None or row[0] is None:
                    logger.warning(f"No data table found for resampled dataset: {name}")
                    return None
                
                table_name = row[0]
                
                # Load data
                cur.execute(f"""
                    SELECT row_idx, col_idx, value 
                    FROM {table_name} 
                    ORDER BY row_idx, col_idx
                """)
                
                data_rows = cur.fetchall()
                if not data_rows:
                    return None
                
                # Get shape
                max_row = max(row[0] for row in data_rows)
                max_col = max(row[1] for row in data_rows)
                
                # Reconstruct array
                data_array = np.full((max_row + 1, max_col + 1), np.nan)
                for row_idx, col_idx, value in data_rows:
                    if value is not None:
                        data_array[row_idx, col_idx] = value
                
                return data_array
                
        except Exception as e:
            logger.error(f"Failed to load resampled data for {name}: {e}")
            return None
    
    def _load_raster_data(self, raster_path: Path) -> xr.DataArray:
        """Load raster data as xarray DataArray."""
        import rioxarray
        
        # Use rioxarray for better CRS handling
        da = rioxarray.open_rasterio(raster_path, chunks={'x': 1000, 'y': 1000})
        
        # Handle multi-band case
        if 'band' in da.dims:
            da = da.sel(band=1)
        
        # Rename to standard coordinates
        if 'x' in da.dims:
            da = da.rename({'x': 'lon', 'y': 'lat'})
        
        return da
    
    def _store_resampled_dataset(self, info: ResampledDatasetInfo, data: np.ndarray):
        """Store resampled dataset metadata and data in database."""
        try:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                
                # Create data table name
                table_name = f"resampled_{info.name.replace('-', '_')}"
                
                # Insert metadata
                cur.execute("""
                    INSERT INTO resampled_datasets 
                    (name, source_path, target_resolution, target_crs, bounds, 
                     shape_height, shape_width, data_type, resampling_method, 
                     band_name, data_table_name, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    info.name,
                    str(info.source_path),
                    info.target_resolution,
                    info.target_crs,
                    list(info.bounds),  # Convert to list for JSON storage
                    info.shape[0],
                    info.shape[1],
                    info.data_type,
                    info.resampling_method,
                    info.band_name,
                    table_name,
                    json.dumps(info.metadata) if info.metadata else '{}'
                ))
                
                # Create data table if it doesn't exist
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        row_idx INTEGER NOT NULL,
                        col_idx INTEGER NOT NULL,
                        value FLOAT,
                        PRIMARY KEY (row_idx, col_idx)
                    )
                """)
                
                # Store data efficiently (only non-NaN values)
                if not np.all(np.isnan(data)):
                    valid_mask = ~np.isnan(data)
                    rows, cols = np.where(valid_mask)
                    values = data[valid_mask]
                    
                    # Batch insert
                    data_to_insert = [(int(r), int(c), float(v)) for r, c, v in zip(rows, cols, values)]
                    
                    cur.executemany(f"""
                        INSERT INTO {table_name} (row_idx, col_idx, value) 
                        VALUES (%s, %s, %s)
                    """, data_to_insert)
                
                conn.commit()
                logger.info(f"Stored resampled dataset {info.name} in table {table_name}")
                
        except Exception as e:
            logger.error(f"Failed to store resampled dataset {info.name}: {e}")
            raise
    
    def process_single(self, item: Any) -> Any:
        """Process a single item - implementation for BaseProcessor interface."""
        if isinstance(item, dict) and 'name' in item:
            return self.resample_dataset(item)
        return item
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input item."""
        if isinstance(item, dict):
            required_keys = ['name', 'path_key', 'data_type', 'band_name']
            missing = [key for key in required_keys if key not in item]
            if missing:
                return False, f"Missing required keys: {missing}"
            return True, None
        return False, "Item must be a dictionary"