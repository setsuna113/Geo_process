# src/processors/data_preparation/resampling_processor.py
"""Enhanced processor for resampling datasets with chunked loading and progress support."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path
import numpy as np
import xarray as xr
from dataclasses import dataclass
from datetime import datetime
import json
import time

from src.config import config
from src.config.config import Config
from src.infrastructure.processors.base_processor import EnhancedBaseProcessor as BaseProcessor
from src.base.memory_manager import get_memory_manager
from src.database.connection import DatabaseManager
from src.domain.raster.catalog import RasterCatalog, RasterEntry
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
    """Enhanced resampling processor with chunked loading and progress support."""
    
    def __init__(self, config: Config, db_connection: DatabaseManager):
        # Initialize with enhanced features
        super().__init__(
            batch_size=1000, 
            config=config,
            enable_progress=True,
            enable_checkpoints=True,
            checkpoint_interval=1,  # Checkpoint after each dataset
            supports_chunking=True
        )
        
        self.db = db_connection
        self.config = config
        self.catalog = RasterCatalog(db_connection, config)
        self.cache_manager = ResamplingCacheManager()
        self.memory_manager = get_memory_manager()
        
        # Get resampling configuration
        self.resampling_config = config.get('resampling', {})
        self.target_resolution = self.resampling_config.get('target_resolution', 0.05)
        self.target_crs = self.resampling_config.get('target_crs', 'EPSG:4326')
        self.strategies = self.resampling_config.get('strategies', {})
        self.engine_type = self.resampling_config.get('engine', 'numpy')
        
        # Chunking configuration
        self.chunk_config = {
            'max_chunk_size_mb': config.get('raster_processing.max_chunk_size_mb', 512),
            'tile_size': config.get('raster_processing.tile_size', 1024),
            'overlap': config.get('raster_processing.tile_overlap', 0),
            'auto_adjust': True
        }
        
        # Dataset configuration
        self.datasets_config = config.get('datasets', {}).get('target_datasets', [])
        
        logger.info(f"ResamplingProcessor initialized with target resolution: {self.target_resolution}")
        logger.info(f"Using {self.engine_type} engine with chunked processing")
    
    def resample_all_datasets(self, resume_from_checkpoint: Optional[str] = None) -> List[ResampledDatasetInfo]:
        """
        Resample all configured datasets with progress and checkpoint support.
        
        Args:
            resume_from_checkpoint: Optional checkpoint ID to resume from
            
        Returns:
            List of resampled dataset info
        """
        # Filter enabled datasets
        enabled_datasets = [ds for ds in self.datasets_config if ds.get('enabled', True)]
        
        # Start progress tracking
        self.start_progress("Resampling Datasets", len(enabled_datasets))
        
        # Resume from checkpoint if provided
        start_index = 0
        if resume_from_checkpoint:
            checkpoint_data = self.load_checkpoint(resume_from_checkpoint)
            start_index = checkpoint_data.get('last_completed_index', -1) + 1
            logger.info(f"Resuming from dataset index {start_index}")
        
        resampled_datasets = []
        
        try:
            for i, dataset_config in enumerate(enabled_datasets[start_index:], start_index):
                # Check for cancellation
                if self._should_stop.is_set():
                    logger.info("Resampling cancelled by user")
                    break
                
                # Wait if paused
                self._pause_event.wait()
                
                dataset_name = dataset_config['name']
                logger.info(f"Processing dataset {i+1}/{len(enabled_datasets)}: {dataset_name}")
                
                # Update progress
                self.update_progress(i, metadata={
                    'current_dataset': dataset_name,
                    'datasets_completed': len(resampled_datasets)
                })
                
                try:
                    # Progress callback for sub-operations
                    def dataset_progress(msg: str, percent: float):
                        self.update_progress(i, metadata={
                            'current_dataset': dataset_name,
                            'dataset_progress': percent,
                            'status': msg
                        })
                    
                    resampled_info = self.resample_dataset(dataset_config, dataset_progress)
                    resampled_datasets.append(resampled_info)
                    
                    # Save checkpoint after each dataset
                    self._checkpoint_data = {
                        'last_completed_index': i,
                        'completed_datasets': [d.name for d in resampled_datasets],
                        'total_datasets': len(enabled_datasets)
                    }
                    self.save_checkpoint()
                    
                except Exception as e:
                    logger.error(f"Failed to resample {dataset_name}: {e}")
                    # Save error state
                    self._checkpoint_data['error'] = {
                        'dataset': dataset_name,
                        'error': str(e),
                        'index': i
                    }
                    self.save_checkpoint(checkpoint_id=f"error_{dataset_name}_{int(time.time())}")
                    # Continue with other datasets
                    continue
            
            # Complete progress
            self.complete_progress(
                status="completed" if not self._should_stop.is_set() else "cancelled",
                metadata={
                    'total_resampled': len(resampled_datasets),
                    'total_attempted': len(enabled_datasets)
                }
            )
            
            logger.info(f"✅ Resampling pipeline completed: {len(resampled_datasets)}/{len(enabled_datasets)} datasets")
            return resampled_datasets
            
        except Exception as e:
            logger.error(f"Resampling pipeline failed: {e}")
            self.complete_progress(status="failed", metadata={'error': str(e)})
            raise
    
    def resample_dataset(self, dataset_config: dict, 
                        progress_callback: Optional[Callable[[str, float], None]] = None) -> ResampledDatasetInfo:
        """
        Resample single dataset with chunked loading and memory management.
        
        Args:
            dataset_config: Dataset configuration
            progress_callback: Progress callback
            
        Returns:
            ResampledDatasetInfo
        """
        from src.config.dataset_utils import DatasetPathResolver
        
        # Memory allocation tracking
        with self.memory_manager.memory_context(
            f"resample_{dataset_config['name']}", 
            estimated_mb=self.chunk_config['max_chunk_size_mb']
        ):
            # Resolve dataset path
            resolver = DatasetPathResolver(self.config)
            normalized_config = resolver.validate_dataset_config(dataset_config)
            
            dataset_name = normalized_config['name']
            raster_path = Path(normalized_config['resolved_path'])
            data_type = normalized_config['data_type']
            band_name = normalized_config['band_name']
            
            logger.info(f"Resampling dataset: {dataset_name}")
            if progress_callback:
                progress_callback(f"Initializing {dataset_name}", 5)
            
            # Get resampling method
            method = self.strategies.get(data_type, 'bilinear')
            
            # Get or register raster in catalog
            raster_entry = self._get_or_register_raster(
                dataset_name, raster_path, data_type, progress_callback
            )
            
            # Check if skip-resampling is enabled and resolution matches
            if self.config.get('resampling.allow_skip_resampling', False):
                if self._check_resolution_match(raster_entry):
                    logger.info(f"🚀 Skipping resampling for {dataset_name} - resolution matches target")
                    
                    if progress_callback:
                        progress_callback(f"Skipping resampling (resolution match)", 50)
                    
                    # Create passthrough dataset info
                    passthrough_info = self._create_passthrough_dataset_info(raster_entry, normalized_config)
                    
                    if progress_callback:
                        progress_callback(f"Storing passthrough metadata", 90)
                    
                    # Store metadata (no actual data processing needed)
                    self._store_resampled_dataset(passthrough_info, None)  # Pass None for data
                    
                    if progress_callback:
                        progress_callback(f"Completed {dataset_name} (skipped)", 100)
                    
                    logger.info(f"✅ Skip-resampling completed for {dataset_name}")
                    return passthrough_info
                else:
                    logger.info(f"Resolution does not match for {dataset_name}, proceeding with resampling")
            else:
                logger.debug(f"Skip-resampling disabled for {dataset_name}")
            
            # Estimate memory requirements
            estimated_memory = self._estimate_memory_requirements(raster_entry)
            logger.info(f"Estimated memory requirement: {estimated_memory:.1f} MB")
            
            # Determine if chunked processing is needed
            if estimated_memory > self.chunk_config['max_chunk_size_mb']:
                logger.info(f"Using chunked processing (data size: {estimated_memory:.1f} MB)")
                result_data = self._resample_chunked(
                    raster_entry, method, progress_callback
                )
            else:
                logger.info(f"Using single-pass processing (data size: {estimated_memory:.1f} MB)")
                result_data = self._resample_single(
                    raster_entry, method, progress_callback
                )
            
            # Create resampled dataset info
            resampled_info = ResampledDatasetInfo(
                name=dataset_name,
                source_path=raster_entry.path,
                target_resolution=self.target_resolution,
                target_crs=self.target_crs,
                bounds=raster_entry.bounds,
                shape=result_data.shape,
                data_type=data_type,
                resampling_method=method,
                band_name=band_name,
                metadata={
                    'source_resolution': raster_entry.resolution_degrees,
                    'engine': self.engine_type,
                    'chunked_processing': estimated_memory > self.chunk_config['max_chunk_size_mb'],
                    'resampled_at': datetime.now().isoformat()
                }
            )
            
            # Store in database
            self._store_resampled_dataset(resampled_info, result_data)
            
            if progress_callback:
                progress_callback(f"Completed {dataset_name}", 100)
            
            return resampled_info
    
    def _resample_chunked(self, 
                         raster_entry: RasterEntry,
                         method: str,
                         progress_callback: Optional[Callable[[str, float], None]] = None) -> np.ndarray:
        """
        Resample using chunked/tiled approach for memory efficiency.
        
        Args:
            raster_entry: Raster catalog entry
            method: Resampling method
            progress_callback: Progress callback
            
        Returns:
            Resampled data array
        """
        import rioxarray
        from dask.array import from_delayed
        import dask.array as da
        
        logger.info(f"Starting chunked resampling for {raster_entry.name}")
        
        # Calculate optimal chunk size based on available memory
        memory_info = self.memory_manager.get_current_memory_usage()
        available_mb = memory_info['available_mb'] * 0.5  # Use 50% of available
        
        # Adjust chunk size
        chunk_size = min(
            self.chunk_config['tile_size'],
            int(np.sqrt(available_mb * 1024 * 1024 / 8))  # Assuming float64
        )
        logger.info(f"Using chunk size: {chunk_size}x{chunk_size}")
        
        if progress_callback:
            progress_callback("Loading raster metadata", 10)
        
        # Open raster with explicit chunking
        with rioxarray.open_rasterio(
            raster_entry.path,
            chunks={'x': chunk_size, 'y': chunk_size},
            cache=False,
            lock=False
        ) as src:
            # Handle bands
            if 'band' in src.dims:
                src = src.sel(band=1)
            
            # Rename coordinates
            if 'x' in src.dims:
                src = src.rename({'x': 'lon', 'y': 'lat'})
            
            # Calculate output shape
            output_shape = self._calculate_output_shape(raster_entry.bounds)
            
            # Create resampler
            resampler = self._create_resampler_engine(method, raster_entry.bounds)
            
            # Initialize output array
            output_data = np.zeros(output_shape, dtype=np.float32)
            
            # Process chunks
            total_chunks = len(src.lon.chunks[0]) * len(src.lat.chunks[0])
            processed_chunks = 0
            
            if progress_callback:
                progress_callback("Processing chunks", 20)
            
            # Iterate over chunks
            for i, lon_slice in enumerate(src.lon.chunks):
                for j, lat_slice in enumerate(src.lat.chunks):
                    # Check memory pressure
                    if self.memory_manager.get_memory_pressure_level().value in ['high', 'critical']:
                        logger.warning("High memory pressure, triggering cleanup")
                        self.memory_manager.trigger_cleanup()
                    
                    # Extract chunk bounds
                    chunk_bounds = (
                        float(src.lon[sum(src.lon.chunks[:i])].values),
                        float(src.lat[sum(src.lat.chunks[:j])].values),
                        float(src.lon[sum(src.lon.chunks[:i+1])-1].values),
                        float(src.lat[sum(src.lat.chunks[:j+1])-1].values)
                    )
                    
                    # Load chunk data
                    chunk_data = src.isel(
                        lon=slice(sum(src.lon.chunks[:i]), sum(src.lon.chunks[:i+1])),
                        lat=slice(sum(src.lat.chunks[:j]), sum(src.lat.chunks[:j+1]))
                    ).compute()
                    
                    # Resample chunk
                    chunk_result = resampler.resample(
                        chunk_data,
                        source_bounds=chunk_bounds,
                        target_bounds=chunk_bounds
                    )
                    
                    # Calculate output indices
                    out_i_start = int((chunk_bounds[0] - raster_entry.bounds[0]) / self.target_resolution)
                    out_j_start = int((raster_entry.bounds[3] - chunk_bounds[3]) / self.target_resolution)
                    
                    # Place in output array
                    out_slice_i = slice(out_i_start, out_i_start + chunk_result.data.shape[1])
                    out_slice_j = slice(out_j_start, out_j_start + chunk_result.data.shape[0])
                    output_data[out_slice_j, out_slice_i] = chunk_result.data
                    
                    # Update progress
                    processed_chunks += 1
                    progress_percent = 20 + (processed_chunks / total_chunks) * 70
                    if progress_callback:
                        progress_callback(
                            f"Processing chunk {processed_chunks}/{total_chunks}", 
                            progress_percent
                        )
            
            if progress_callback:
                progress_callback("Finalizing", 95)
            
            return output_data
    
    def _resample_single(self,
                        raster_entry: RasterEntry,
                        method: str,
                        progress_callback: Optional[Callable[[str, float], None]] = None) -> np.ndarray:
        """
        Resample entire raster in single pass (for smaller datasets).
        
        Args:
            raster_entry: Raster catalog entry
            method: Resampling method
            progress_callback: Progress callback
            
        Returns:
            Resampled data array
        """
        if progress_callback:
            progress_callback("Loading raster data", 20)
        
        # Load with timeout protection
        source_data = self._load_raster_data_with_timeout(
            raster_entry.path,
            timeout_seconds=self.config.get('raster_processing.gdal_timeout', 60)
        )
        
        if progress_callback:
            progress_callback("Creating resampler", 30)
        
        # Create resampler
        resampler = self._create_resampler_engine(method, raster_entry.bounds)
        
        # Progress wrapper for resampler
        def resample_progress(percent: float):
            if progress_callback:
                # Map resampler progress (0-100) to overall progress (30-90)
                overall_percent = 30 + (percent * 0.6)
                progress_callback(f"Resampling ({percent:.0f}%)", overall_percent)
        
        # Perform resampling
        result = resampler.resample(
            source_data=source_data,
            source_bounds=raster_entry.bounds,
            progress_callback=resample_progress
        )
        
        if progress_callback:
            progress_callback("Resampling complete", 95)
        
        return result.data
    
    def _get_or_register_raster(self,
                               dataset_name: str,
                               raster_path: Path,
                               data_type: str,
                               progress_callback: Optional[Callable[[str, float], None]] = None) -> RasterEntry:
        """Get raster from catalog or register it."""
        raster_entry = self.catalog.get_raster(dataset_name)
        
        if raster_entry is None:
            if not raster_path.exists():
                raise FileNotFoundError(f"Raster file not found: {raster_path}")
            
            logger.info(f"Registering {dataset_name} in catalog...")
            if progress_callback:
                progress_callback(f"Registering {dataset_name}", 10)
            
            # Use lightweight registration
            raster_entry = self.catalog.add_raster_lightweight(
                raster_path,
                dataset_type=data_type,
                validate=False
            )
            logger.info(f"✅ Registered {dataset_name}")
        
        return raster_entry
    
    def _estimate_memory_requirements(self, raster_entry: RasterEntry) -> float:
        """Estimate memory requirements for resampling."""
        # Source data size
        source_pixels = raster_entry.metadata.get('width', 0) * raster_entry.metadata.get('height', 0)
        source_mb = source_pixels * 8 / (1024 * 1024)  # Assuming float64
        
        # Target data size
        target_shape = self._calculate_output_shape(raster_entry.bounds)
        target_pixels = target_shape[0] * target_shape[1]
        target_mb = target_pixels * 8 / (1024 * 1024)
        
        # Total with overhead (source + target + working memory)
        total_mb = (source_mb + target_mb) * 1.5
        
        return total_mb
    
    def _calculate_output_shape(self, bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
        """Calculate output shape from bounds and target resolution."""
        minx, miny, maxx, maxy = bounds
        width = int(np.ceil((maxx - minx) / self.target_resolution))
        height = int(np.ceil((maxy - miny) / self.target_resolution))
        return (height, width)
    
    def _create_resampler_engine(self, method: str, source_bounds: Tuple[float, float, float, float]) -> Union[NumpyResampler, GDALResampler]:
        """Create resampler engine with appropriate configuration."""
        # Create resampling config
        resampling_config = ResamplingConfig(
            source_resolution=self._estimate_source_resolution(source_bounds),
            target_resolution=self.target_resolution,
            method=method,
            bounds=source_bounds,
            source_crs=self.target_crs,
            target_crs=self.target_crs,
            chunk_size=self.chunk_config['tile_size'],
            cache_results=self.resampling_config.get('cache_resampled', True),
            validate_output=self.resampling_config.get('validate_output', True),
            preserve_sum=self.resampling_config.get('preserve_sum', True)
            # memory_limit_mb removed - not a valid field
        )
        
        # Create engine
        if self.engine_type == 'gdal':
            return GDALResampler(resampling_config)
        else:
            return NumpyResampler(resampling_config)
    
    def _estimate_source_resolution(self, bounds: Tuple[float, float, float, float]) -> float:
        """Estimate source resolution from bounds."""
        minx, miny, maxx, maxy = bounds
        return abs(maxx - minx) / 1000  # Rough estimate
    
    def _load_raster_data_with_timeout(self, 
                                      raster_path: Path,
                                      timeout_seconds: int = 60) -> xr.DataArray:
        """Load raster data with timeout protection."""
        import rioxarray
        from src.domain.raster.loaders.lightweight_metadata import gdal_timeout, TimeoutError
        
        logger.info(f"Loading raster data from {raster_path.name}")
        
        try:
            with gdal_timeout(timeout_seconds):
                # Use rioxarray with chunking
                da = rioxarray.open_rasterio(
                    raster_path,
                    chunks={'x': 1000, 'y': 1000},
                    cache=False
                )
                
                # Handle multi-band
                if 'band' in da.dims:
                    da = da.sel(band=1)
                
                # Rename coordinates
                if 'x' in da.dims:
                    da = da.rename({'x': 'lon', 'y': 'lat'})
                
                return da
                
        except TimeoutError:
            raise RuntimeError(f"Raster loading timed out after {timeout_seconds}s")
        except Exception as e:
            raise RuntimeError(f"Failed to load raster data: {e}")
    
    def _check_resolution_match(self, raster_entry: RasterEntry) -> bool:
        """
        Check if raster resolution matches target resolution within tolerance.
        
        Args:
            raster_entry: Raster catalog entry with resolution information
            
        Returns:
            True if resolution matches within tolerance, False otherwise
        """
        if not raster_entry.resolution_degrees:
            logger.warning(f"No resolution information for {raster_entry.name}, cannot skip resampling")
            return False
        
        source_resolution = abs(float(raster_entry.resolution_degrees))
        target_resolution = abs(float(self.target_resolution))
        tolerance = self.config.get('resampling.resolution_tolerance', 0.001)
        
        resolution_diff = abs(source_resolution - target_resolution)
        matches = resolution_diff <= tolerance
        
        logger.info(f"Resolution check for {raster_entry.name}:")
        logger.info(f"  Source: {source_resolution:.6f}°, Target: {target_resolution:.6f}°")
        logger.info(f"  Difference: {resolution_diff:.6f}°, Tolerance: {tolerance:.6f}°")
        logger.info(f"  Match: {'✓' if matches else '✗'}")
        
        return matches
    
    def _create_passthrough_dataset_info(self, raster_entry: RasterEntry, dataset_config: dict) -> ResampledDatasetInfo:
        """
        Create ResampledDatasetInfo for passthrough dataset (no actual resampling).
        
        Args:
            raster_entry: Raster catalog entry
            dataset_config: Dataset configuration
            
        Returns:
            ResampledDatasetInfo with passthrough metadata
        """
        logger.info(f"Creating passthrough dataset info for {raster_entry.name}")
        
        # Use actual source resolution instead of target
        actual_resolution = abs(float(raster_entry.resolution_degrees))
        
        # Create metadata with passthrough information
        passthrough_metadata = {
            'source_resolution': actual_resolution,
            'target_resolution': self.target_resolution,
            'passthrough': True,
            'skip_reason': 'resolution_match',
            'resolution_difference': abs(actual_resolution - self.target_resolution),
            'resolution_tolerance': self.config.get('resampling.resolution_tolerance', 0.001),
            'engine': self.engine_type,
            'chunked_processing': False,  # No processing needed
            'resampled_at': datetime.now().isoformat(),
            'processing_skipped': True
        }
        
        # Calculate expected shape based on bounds and actual resolution
        bounds = raster_entry.bounds
        width = int(np.ceil((bounds[2] - bounds[0]) / actual_resolution))
        height = int(np.ceil((bounds[3] - bounds[1]) / actual_resolution))
        
        passthrough_info = ResampledDatasetInfo(
            name=dataset_config['name'],
            source_path=raster_entry.path,
            target_resolution=actual_resolution,  # Use actual resolution
            target_crs=self.target_crs,
            bounds=bounds,
            shape=(height, width),
            data_type=dataset_config['data_type'],
            resampling_method='passthrough',  # Special method name
            band_name=dataset_config['band_name'],
            metadata=passthrough_metadata
        )
        
        logger.info(f"✅ Created passthrough dataset info: {passthrough_info.name}")
        logger.info(f"   Resolution: {actual_resolution:.6f}° (source), Shape: {passthrough_info.shape}")
        
        return passthrough_info

    def _store_resampled_dataset(self, info: ResampledDatasetInfo, data: Optional[np.ndarray]):
        """Store resampled dataset in database."""
        # Handle passthrough datasets differently
        if info.metadata.get('passthrough', False):
            logger.info(f"Storing passthrough dataset metadata for {info.name}")
            try:
                with self.db.get_connection() as conn:
                    cur = conn.cursor()
                    
                    # Store only metadata for passthrough datasets
                    # The data remains in the original file
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
                        list(info.bounds),
                        info.shape[0],
                        info.shape[1],
                        info.data_type,
                        info.resampling_method,
                        info.band_name,
                        f"passthrough_{info.name.replace('-', '_')}",  # Special table name
                        json.dumps(info.metadata)
                    ))
                    
                    conn.commit()
                    logger.info(f"✅ Stored passthrough dataset metadata: {info.name}")
                    
            except Exception as e:
                logger.error(f"Failed to store passthrough dataset metadata: {e}")
                raise
            return
        
        # Original implementation for resampled datasets
        if data is None:
            raise ValueError(f"Data cannot be None for non-passthrough dataset {info.name}")
        
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
                    list(info.bounds),
                    info.shape[0],
                    info.shape[1],
                    info.data_type,
                    info.resampling_method,
                    info.band_name,
                    table_name,
                    json.dumps(info.metadata)
                ))
                
                # Create and populate data table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        row_idx INTEGER NOT NULL,
                        col_idx INTEGER NOT NULL,
                        value FLOAT,
                        PRIMARY KEY (row_idx, col_idx)
                    )
                """)
                
                # Store non-NaN values
                if not np.all(np.isnan(data)):
                    valid_mask = ~np.isnan(data)
                    rows, cols = np.where(valid_mask)
                    values = data[valid_mask]
                    
                    data_to_insert = [(int(r), int(c), float(v)) for r, c, v in zip(rows, cols, values)]
                    
                    # Batch insert
                    from psycopg2.extras import execute_values
                    execute_values(
                        cur,
                        f"INSERT INTO {table_name} (row_idx, col_idx, value) VALUES %s",
                        data_to_insert,
                        page_size=10000
                    )
                
                conn.commit()
                logger.info(f"Stored resampled dataset {info.name}")
                
        except Exception as e:
            logger.error(f"Failed to store resampled dataset: {e}")
            raise
    
    # BaseProcessor abstract method implementations
    def process_single(self, item: Any) -> Any:
        """Process single item."""
        if isinstance(item, dict) and 'name' in item:
            return self.resample_dataset(item)
        return item
    
    def load_passthrough_data(self, info: ResampledDatasetInfo) -> Optional[np.ndarray]:
        """
        Load data directly from original file for passthrough datasets.
        
        Args:
            info: ResampledDatasetInfo for passthrough dataset
            
        Returns:
            Loaded numpy array or None if failed
        """
        if not info.metadata.get('passthrough', False):
            raise ValueError(f"Dataset {info.name} is not a passthrough dataset")
        
        logger.info(f"Loading passthrough data from: {info.source_path}")
        
        try:
            # Load data with timeout protection 
            data_array = self._load_raster_data_with_timeout(
                info.source_path,
                timeout_seconds=self.config.get('raster_processing.gdal_timeout', 60)
            )
            
            # Convert xarray to numpy array
            if hasattr(data_array, 'values'):
                array_data = data_array.values
            else:
                array_data = np.array(data_array)
            
            # Ensure 2D array
            if array_data.ndim == 3 and array_data.shape[0] == 1:
                array_data = array_data[0]
            elif array_data.ndim != 2:
                raise ValueError(f"Expected 2D data, got {array_data.ndim}D")
            
            logger.info(f"✅ Loaded passthrough data: {array_data.shape}")
            return array_data
            
        except Exception as e:
            logger.error(f"Failed to load passthrough data for {info.name}: {e}")
            return None

    def get_resampled_dataset(self, dataset_name: str) -> Optional[ResampledDatasetInfo]:
        """
        Get existing resampled dataset (including passthrough datasets).
        
        Args:
            dataset_name: Name of the dataset to retrieve
            
        Returns:
            ResampledDatasetInfo if found, None otherwise
        """
        try:
            from src.database.schema import schema
            
            # Query for the dataset
            datasets = schema.get_resampled_datasets({'name': dataset_name})
            
            if not datasets:
                return None
            
            # Take the most recent one
            dataset_row = datasets[0]
            
            # Convert to ResampledDatasetInfo
            resampled_info = ResampledDatasetInfo(
                name=dataset_row['name'],
                source_path=Path(dataset_row['source_path']),
                target_resolution=float(dataset_row['target_resolution']),
                target_crs=dataset_row['target_crs'],
                bounds=tuple(dataset_row['bounds']),
                shape=(int(dataset_row['shape_height']), int(dataset_row['shape_width'])),
                data_type=dataset_row['data_type'],
                resampling_method=dataset_row['resampling_method'],
                band_name=dataset_row['band_name'],
                metadata=dataset_row['metadata'] or {}
            )
            
            logger.info(f"Found existing dataset: {dataset_name} ({'passthrough' if resampled_info.metadata.get('passthrough') else 'resampled'})")
            return resampled_info
            
        except Exception as e:
            logger.error(f"Failed to retrieve dataset {dataset_name}: {e}")
            return None

    def load_resampled_data(self, dataset_name: str) -> Optional[np.ndarray]:
        """
        Load resampled data from database (for non-passthrough datasets).
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Numpy array or None if failed
        """
        try:
            from src.database.schema import schema
            
            # Get dataset info
            datasets = schema.get_resampled_datasets({'name': dataset_name})
            if not datasets:
                logger.error(f"Dataset {dataset_name} not found")
                return None
            
            dataset_info = datasets[0]
            
            # Check if it's a passthrough dataset
            metadata = dataset_info.get('metadata', {})
            if metadata.get('passthrough', False):
                logger.info(f"Dataset {dataset_name} is passthrough, use load_passthrough_data instead")
                # Create ResampledDatasetInfo and load via passthrough method
                info = ResampledDatasetInfo(
                    name=dataset_info['name'],
                    source_path=Path(dataset_info['source_path']),
                    target_resolution=float(dataset_info['target_resolution']),
                    target_crs=dataset_info['target_crs'],
                    bounds=tuple(dataset_info['bounds']),
                    shape=(int(dataset_info['shape_height']), int(dataset_info['shape_width'])),
                    data_type=dataset_info['data_type'],
                    resampling_method=dataset_info['resampling_method'],
                    band_name=dataset_info['band_name'],
                    metadata=metadata
                )
                return self.load_passthrough_data(info)
            
            # For non-passthrough, load from database table
            table_name = dataset_info['data_table_name']
            shape = (dataset_info['shape_height'], dataset_info['shape_width'])
            
            # Load data from table
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    SELECT row_idx, col_idx, value 
                    FROM {table_name} 
                    ORDER BY row_idx, col_idx
                """)
                
                rows = cur.fetchall()
                
                if not rows:
                    logger.warning(f"No data found in table {table_name}")
                    return np.full(shape, np.nan, dtype=np.float32)
                
                # Create array
                array_data = np.full(shape, np.nan, dtype=np.float32)
                
                for row in rows:
                    array_data[row[0], row[1]] = row[2]
                
                logger.info(f"✅ Loaded resampled data: {dataset_name}, shape: {shape}")
                return array_data
                
        except Exception as e:
            logger.error(f"Failed to load resampled data for {dataset_name}: {e}")
            return None

    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input item."""
        if not isinstance(item, dict):
            return False, "Item must be a dictionary"
        
        try:
            from src.config.dataset_utils import DatasetPathResolver
            resolver = DatasetPathResolver(self.config)
            resolver.validate_dataset_config(item)
            return True, None
        except ValueError as e:
            return False, str(e)