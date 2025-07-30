# src/processors/data_preparation/resampling_processor.py
"""Enhanced processor for resampling datasets with chunked loading and progress support."""
import logging

logger = logging.getLogger(__name__)
logger.debug("🔍 resampling_processor.py module loading...")
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path
import numpy as np
import xarray as xr
from dataclasses import dataclass
from datetime import datetime
import json
import time
import rasterio

from src.config import config
from src.config.config import Config
from src.base.processor import BaseProcessor
from src.base.memory_manager import get_memory_manager
from src.database.connection import DatabaseManager
from src.domain.raster.catalog import RasterCatalog, RasterEntry
from src.domain.resampling.engines.base_resampler import ResamplingConfig
from src.domain.resampling.engines.numpy_resampler import NumpyResampler
from src.domain.resampling.engines.gdal_resampler import GDALResampler
from src.domain.resampling.cache_manager import ResamplingCacheManager
from src.domain.validators.coordinate_integrity import (
    BoundsConsistencyValidator, CoordinateTransformValidator, ParquetValueValidator
)
from src.abstractions.interfaces.validator import ValidationSeverity
from src.processors.data_preparation.windowed_storage import WindowedStorageManager


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
        logger.info("🔧 DEBUG: ResamplingProcessor.__init__() called")
        # Initialize with enhanced features
        super().__init__(
            batch_size=1000, 
            config=config,
            enable_progress=True,
            enable_checkpoints=True,
            checkpoint_interval=1,  # Checkpoint after each dataset
            supports_chunking=True
        )
        logger.info("✅ DEBUG: BaseProcessor initialized")
        
        self.db = db_connection
        self.config = config
        logger.info("🔧 DEBUG: Creating RasterCatalog...")
        self.catalog = RasterCatalog(db_connection, config)
        logger.info("✅ DEBUG: RasterCatalog created")
        self.cache_manager = ResamplingCacheManager()
        self.memory_manager = get_memory_manager()
        
        # Initialize validators
        self.bounds_validator = BoundsConsistencyValidator(tolerance=1e-6)
        self.transform_validator = CoordinateTransformValidator(max_error_meters=5.0)  # More lenient for resampling
        self.value_validator = ParquetValueValidator(
            max_null_percentage=15.0,  # More lenient for resampled data
            outlier_std_threshold=3.0
        )
        
        # Validation results tracking
        self.validation_results: List[Dict] = []
        
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
        
        # Adaptive window size tracking
        self._adaptive_window_size = None
        
        logger.info(f"ResamplingProcessor initialized with target resolution: {self.target_resolution}")
        logger.info(f"Using {self.engine_type} engine with chunked processing")
    
    def _get_current_window_size(self) -> int:
        """Get current window size, using adaptive size if set."""
        if self._adaptive_window_size is not None:
            return self._adaptive_window_size
        return self.config.get('resampling.window_size', 2048)
    
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
            
            # Report validation summary
            self._report_validation_summary()
            
            return resampled_datasets
            
        except Exception as e:
            logger.error(f"Resampling pipeline failed: {e}")
            self.complete_progress(status="failed", metadata={'error': str(e)})
            raise
    
    def resample_dataset(self, dataset_config: dict, 
                        progress_callback: Optional[Callable[[str, float], None]] = None) -> ResampledDatasetInfo:
        """
        Resample single dataset with chunked loading and memory management.
        
        DEPRECATED: This method loads entire datasets into memory for passthrough cases. 
        Use resample_dataset_memory_aware() instead for memory-efficient processing.
        
        .. deprecated:: 2.0
           Use :meth:`resample_dataset_memory_aware` instead.
        
        Args:
            dataset_config: Dataset configuration
            progress_callback: Progress callback
            
        Returns:
            ResampledDatasetInfo
        """
        import warnings
        warnings.warn(
            "resample_dataset() is deprecated and may load entire datasets into memory. "
            "Use resample_dataset_memory_aware() for memory-efficient processing.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.info(f"🔍 DEBUG: resample_dataset() called for {dataset_config.get('name', 'unknown')}")
        logger.debug(f"🔍 resample_dataset() called for {dataset_config.get('name', 'unknown')}")
        from src.config.dataset_utils import DatasetPathResolver
        
        # Memory allocation tracking
        logger.info(f"🔍 DEBUG: Entering memory context for {dataset_config['name']}")
        logger.debug(f"🔍 Entering memory context for {dataset_config['name']}")
        with self.memory_manager.memory_context(
            f"resample_{dataset_config['name']}", 
            estimated_mb=self.chunk_config['max_chunk_size_mb']
        ):
            logger.info(f"🔍 DEBUG: Inside memory context, creating DatasetPathResolver")
            logger.debug(f"🔍 Inside memory context, creating DatasetPathResolver")
            # Resolve dataset path
            resolver = DatasetPathResolver(self.config)
            logger.info(f"🔍 DEBUG: DatasetPathResolver created, validating config")
            logger.debug(f"🔍 DatasetPathResolver created, validating config")
            normalized_config = resolver.validate_dataset_config(dataset_config)
            logger.info(f"🔍 DEBUG: Config validated, extracting paths")
            logger.debug(f"🔍 Config validated, extracting paths")
            
            dataset_name = normalized_config['name']
            raster_path = Path(normalized_config['resolved_path'])
            data_type = normalized_config['data_type']
            band_name = normalized_config['band_name']
            
            logger.info(f"🔍 DEBUG: Paths extracted - dataset: {dataset_name}, path: {raster_path}")
            logger.debug(f"🔍 Paths extracted - dataset: {dataset_name}, path: {raster_path}")
            
            logger.info(f"Resampling dataset: {dataset_name}")
            if progress_callback:
                progress_callback(f"Initializing {dataset_name}", 5)
            
            # Get resampling method
            method = self.strategies.get(data_type, 'bilinear')
            logger.info(f"🔍 DEBUG: Resampling method: {method}")
            logger.debug(f"🔍 Resampling method: {method}")
            
            # Get or register raster in catalog
            logger.info(f"🔍 DEBUG: About to call _get_or_register_raster")
            logger.debug(f"🔍 About to call _get_or_register_raster")
            raster_entry = self._get_or_register_raster(
                dataset_name, raster_path, data_type, progress_callback
            )
            logger.info(f"🔍 DEBUG: _get_or_register_raster completed")
            logger.debug(f"🔍 _get_or_register_raster completed")
            
            # Validate source bounds
            self._validate_source_bounds(raster_entry, dataset_name)
            
            # Validate coordinate transformation if needed
            source_crs = raster_entry.crs or 'EPSG:4326'
            self._validate_coordinate_transformation(source_crs, dataset_name)
            
            # Check if skip-resampling is enabled and resolution matches
            logger.info(f"🔍 DEBUG: Checking skip-resampling - enabled: {self.config.get('resampling.allow_skip_resampling', False)}")
            logger.debug(f"🔍 Checking skip-resampling - enabled: {self.config.get('resampling.allow_skip_resampling', False)}")
            if self.config.get('resampling.allow_skip_resampling', False):
                logger.info(f"🔍 DEBUG: About to check resolution match")
                logger.debug(f"🔍 About to check resolution match")
                if self._check_resolution_match(raster_entry):
                    logger.info(f"🔍 DEBUG: Resolution matches! Will skip resampling")
                    logger.debug(f"🔍 Resolution matches! Will skip resampling")
                    print(f"🚨 IMMEDIATE: About to log skipping message", flush=True)
                    logger.info(f"🚀 Skipping resampling for {dataset_name} - resolution matches target")
                    print(f"🚨 IMMEDIATE: Logged skipping message", flush=True)
                    
                    if progress_callback:
                        progress_callback(f"Skipping resampling (resolution match)", 50)
                    
                    # Check if memory-aware processing is enabled
                    use_memory_aware = self.config.get('resampling.enable_memory_aware_processing', False)
                    
                    if use_memory_aware:
                        # Use memory-aware passthrough that doesn't load data
                        logger.info(f"Using memory-aware passthrough for {dataset_name}")
                        return self._handle_passthrough_memory_aware(raster_entry, normalized_config, progress_callback)
                    
                    # Create passthrough dataset info (legacy path)
                    passthrough_info = self._create_passthrough_dataset_info(raster_entry, normalized_config)
                    
                    if progress_callback:
                        progress_callback(f"Loading passthrough data for storage", 70)
                    
                    # Load the actual data for database storage
                    # Even for passthrough, we need to store in DB for consistent lazy processing
                    logger.info(f"Loading raster data from {raster_path} (this may take time for large files)")
                    logger.info(f"About to open raster file: {raster_path}")
                    
                    # Check memory requirements first
                    logger.info("Opening file for size check...")
                    with rasterio.open(raster_path) as src:
                        shape = (src.height, src.width)
                        estimated_mb = (shape[0] * shape[1] * 4) / (1024 * 1024)  # float32
                        logger.info(f"Raster shape: {shape}, estimated memory: {estimated_mb:.1f}MB")
                        
                        if estimated_mb > 5000:  # More than 5GB
                            logger.warning(f"Large raster detected ({estimated_mb:.1f}MB), this may take several minutes")
                    
                    logger.info("Size check complete, now loading data...")
                    start_time = time.time()
                    with rasterio.open(raster_path) as src:
                        logger.info("File opened, reading band 1...")
                        data = src.read(1).astype(np.float32)
                        logger.info(f"Data read, shape: {data.shape}")
                        # Handle nodata values
                        if src.nodata is not None:
                            logger.info(f"Handling nodata value: {src.nodata}")
                            data[data == src.nodata] = np.nan
                    load_time = time.time() - start_time
                    data_mb = data.nbytes / (1024 * 1024)
                    logger.info(f"Loaded {data.shape} array ({data_mb:.1f}MB) in {load_time:.1f}s")
                    
                    if progress_callback:
                        progress_callback(f"Storing passthrough data to database", 90)
                    
                    # Store both metadata AND data for passthrough datasets
                    try:
                        self._store_resampled_dataset(passthrough_info, data)
                    except Exception as e:
                        logger.error(f"Failed to store passthrough data for {dataset_name}: {e}")
                        # Clean up any partial data
                        try:
                            with self.db.get_connection() as conn:
                                with conn.cursor() as cur:
                                    cur.execute("DELETE FROM resampled_datasets WHERE name = %s", (dataset_name,))
                                    table_name = f"passthrough_{dataset_name.replace('-', '_')}"
                                    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                                    conn.commit()
                        except:
                            pass
                        raise
                    
                    if progress_callback:
                        progress_callback(f"Completed {dataset_name} (skipped)", 100)
                    
                    logger.info(f"✅ Skip-resampling completed for {dataset_name}")
                    return passthrough_info
                else:
                    logger.info(f"Resolution does not match for {dataset_name}, proceeding with resampling")
            else:
                logger.debug(f"Skip-resampling disabled for {dataset_name}")
            
            # Check if memory-aware processing is enabled for resampling
            use_memory_aware = self.config.get('resampling.enable_memory_aware_processing', False)
            
            if use_memory_aware:
                # Use memory-aware resampling that doesn't load data
                logger.info(f"Using memory-aware resampling for {dataset_name}")
                return self._handle_resampling_memory_aware(raster_entry, normalized_config, progress_callback)
            
            # Legacy path: estimate memory requirements
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
            
            # Validate resampled data
            self._validate_resampled_data(result_data, dataset_name)
            
            # Validate output bounds consistency
            calculated_bounds = self._calculate_output_bounds_from_data(result_data, raster_entry.bounds)
            self._validate_output_bounds(calculated_bounds, raster_entry.bounds, dataset_name)
            
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
    
    def resample_dataset_memory_aware(self, dataset_config: dict, 
                                    progress_callback: Optional[Callable[[str, float], None]] = None,
                                    context: Optional[Any] = None) -> ResampledDatasetInfo:
        """
        Memory-aware resampling that uses windowed processing for both passthrough and resampling.
        
        This method replaces the legacy resample_dataset() and provides:
        - Windowed processing for large datasets
        - No full dataset loading for passthrough cases
        - Efficient memory usage with configurable window sizes
        - Adaptive window sizing based on memory pressure
        
        Args:
            dataset_config: Dataset configuration
            progress_callback: Progress callback
            context: Optional pipeline context with memory monitor
            
        Returns:
            ResampledDatasetInfo with dataset metadata
        """
        dataset_name = dataset_config['name']
        logger.info(f"Starting memory-aware processing for {dataset_name}")
        
        # Get memory monitor if available
        memory_monitor = None
        if context and hasattr(context, 'memory_monitor'):
            memory_monitor = context.memory_monitor
        
        # Store original window size
        original_window_size = self.config.get('resampling.window_size', 2048)
        self._adaptive_window_size = original_window_size
        
        # Define pressure callbacks
        def on_memory_warning(usage):
            self._adaptive_window_size = max(512, self._adaptive_window_size // 2)
            logger.warning(f"Memory pressure (warning): reducing window size to {self._adaptive_window_size}")
        
        def on_memory_critical(usage):
            self._adaptive_window_size = 256  # Minimum viable size
            logger.error(f"Memory pressure (critical): window size set to minimum {self._adaptive_window_size}")
            # Force garbage collection
            import gc
            gc.collect()
        
        # Register callbacks
        if memory_monitor:
            memory_monitor.register_warning_callback(on_memory_warning)
            memory_monitor.register_critical_callback(on_memory_critical)
        
        try:
            # Get or register raster in catalog
            raster_entry = self._get_or_register_raster(
                dataset_name,
                dataset_config.get('path', dataset_config.get('resolved_path'))
            )
            
            # Determine if passthrough is appropriate
            if self._check_resolution_match(raster_entry):
                logger.info(f"Dataset {dataset_name} resolution matches target, using windowed passthrough")
                return self._handle_passthrough_memory_aware(
                    raster_entry, dataset_config, progress_callback
                )
            else:
                logger.info(f"Dataset {dataset_name} needs resampling, using windowed processing")
                return self._handle_resampling_memory_aware(
                    raster_entry, dataset_config, progress_callback
                )
                
        except Exception as e:
            logger.error(f"Memory-aware processing failed for {dataset_name}: {e}")
            raise
        finally:
            # Restore original window size
            self._adaptive_window_size = original_window_size
    
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
        logger.info(f"🔍 DEBUG: _get_or_register_raster called for {dataset_name}")
        logger.debug(f"🔍 _get_or_register_raster called for {dataset_name}")
        raster_entry = self.catalog.get_raster(dataset_name)
        logger.info(f"🔍 DEBUG: catalog.get_raster returned: {raster_entry is not None}")
        logger.debug(f"🔍 catalog.get_raster returned: {raster_entry is not None}")
        
        if raster_entry is None:
            logger.info(f"🔍 DEBUG: Raster entry is None, need to register")
            logger.debug(f"🔍 Raster entry is None, need to register")
            if not raster_path.exists():
                raise FileNotFoundError(f"Raster file not found: {raster_path}")
            
            logger.info(f"Registering {dataset_name} in catalog...")
            if progress_callback:
                progress_callback(f"Registering {dataset_name}", 10)
            
            logger.info(f"🔍 DEBUG: About to call catalog.add_raster_lightweight")
            logger.debug(f"🔍 About to call catalog.add_raster_lightweight")
            # Use lightweight registration
            raster_entry = self.catalog.add_raster_lightweight(
                raster_path,
                dataset_type=data_type,
                validate=False
            )
            logger.info(f"🔍 DEBUG: catalog.add_raster_lightweight completed")
            logger.debug(f"🔍 catalog.add_raster_lightweight completed")
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
        logger.info(f"🔍 DEBUG: _check_resolution_match called for {raster_entry.name}")
        logger.debug(f"🔍 _check_resolution_match called for {raster_entry.name}")
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
    
    def _handle_passthrough_memory_aware(self, raster_entry: RasterEntry, dataset_config: dict,
                                       progress_callback: Optional[Callable[[str, float], None]] = None) -> ResampledDatasetInfo:
        """Handle passthrough dataset without loading data into memory.
        
        This method creates metadata and uses windowed storage to copy data
        directly to the database without loading the entire dataset.
        
        Args:
            raster_entry: Raster catalog entry
            dataset_config: Dataset configuration
            progress_callback: Optional progress callback
            
        Returns:
            ResampledDatasetInfo for the passthrough dataset
        """
        logger.info(f"Processing passthrough dataset (memory-aware): {dataset_config['name']}")
        
        # Create passthrough metadata
        passthrough_info = self._create_passthrough_dataset_info(raster_entry, dataset_config)
        
        # Register metadata in database
        self._register_dataset_metadata(passthrough_info)
        
        # Create table for windowed storage
        table_name = f"passthrough_{passthrough_info.name.replace('-', '_')}"
        
        # Use windowed storage manager with current window size
        storage_manager = WindowedStorageManager(
            window_size=self._get_current_window_size()
        )
        
        # Create the storage table
        storage_manager.create_storage_table(table_name, self.db)
        
        if progress_callback:
            progress_callback("Starting windowed passthrough storage", 30)
        
        # Stream copy data using windowed storage
        stats = storage_manager.store_passthrough_windowed(
            str(raster_entry.path),
            table_name,
            self.db,
            raster_entry.bounds,
            progress_callback
        )
        
        logger.info(f"✅ Memory-aware passthrough completed for {dataset_config['name']}")
        logger.info(f"   Stored {stats['stored_pixels']:,} pixels in {stats['processed_windows']} windows")
        
        # Update metadata with storage info
        passthrough_info.metadata.update({
            'storage_table': table_name,
            'storage_stats': stats,
            'memory_aware': True
        })
        
        return passthrough_info
    
    def _register_dataset_metadata(self, info: ResampledDatasetInfo) -> None:
        """Register dataset metadata in database without storing data.
        
        Args:
            info: Dataset information to register
        """
        from src.database.schema import schema
        
        # Prepare metadata
        metadata = info.metadata.copy()
        metadata['registered_at'] = datetime.now().isoformat()
        
        # Store metadata entry
        schema.store_resampled_dataset(
            name=info.name,
            source_path=str(info.source_path),
            target_resolution=info.target_resolution,
            target_crs=info.target_crs,
            bounds=info.bounds,
            shape_height=info.shape[0],
            shape_width=info.shape[1],
            data_type=info.data_type,
            resampling_method=info.resampling_method,
            band_name=info.band_name,
            data_table_name=metadata.get('storage_table'),
            metadata=metadata
        )
        
        logger.info(f"Registered dataset metadata: {info.name}")
    
    def _handle_resampling_memory_aware(self, raster_entry: RasterEntry, dataset_config: dict,
                                      progress_callback: Optional[Callable[[str, float], None]] = None) -> ResampledDatasetInfo:
        """Handle resampling dataset without loading data into memory.
        
        This method uses windowed processing to resample data directly
        to the database without loading the entire dataset.
        
        Args:
            raster_entry: Raster catalog entry
            dataset_config: Dataset configuration
            progress_callback: Optional progress callback
            
        Returns:
            ResampledDatasetInfo for the resampled dataset
        """
        logger.info(f"Processing resampling dataset (memory-aware): {dataset_config['name']}")
        
        # Create output table name
        table_name = f"resampled_{dataset_config['name'].replace('-', '_')}"
        
        # Get resampling engine
        if self.engine_type == 'numpy':
            from src.domain.resampling.engines.numpy_resampler import NumpyResampler
            from src.domain.resampling.engines.base_resampler import ResamplingConfig
            
            # Create resampling config
            resample_config = ResamplingConfig(
                source_resolution=raster_entry.resolution_degrees,
                target_resolution=self.target_resolution,
                method=dataset_config.get('resampling_method', self._get_resampling_method(dataset_config['data_type'])),
                bounds=raster_entry.bounds,
                source_crs=raster_entry.crs or 'EPSG:4326',
                target_crs=self.target_crs,
                chunk_size=self._get_current_window_size(),
                preserve_sum=dataset_config['data_type'] == 'richness_data',
                nodata_value=raster_entry.metadata.get('nodata_value')
            )
            
            resampler = NumpyResampler(resample_config)
        else:
            raise ValueError(f"Unsupported engine for windowed resampling: {self.engine_type}")
        
        if progress_callback:
            progress_callback("Starting windowed resampling", 30)
        
        # Perform windowed resampling
        stats = resampler.resample_windowed(
            str(raster_entry.path),
            table_name,
            self.db,
            progress_callback
        )
        
        logger.info(f"✅ Memory-aware resampling completed for {dataset_config['name']}")
        logger.info(f"   Processed {stats['processed_windows']} windows, stored {stats['stored_pixels']:,} pixels")
        
        # Calculate expected shape based on target resolution
        bounds = raster_entry.bounds
        width = int(np.ceil((bounds[2] - bounds[0]) / self.target_resolution))
        height = int(np.ceil((bounds[3] - bounds[1]) / self.target_resolution))
        
        # Create resampled dataset info
        resampled_info = ResampledDatasetInfo(
            name=dataset_config['name'],
            source_path=raster_entry.path,
            target_resolution=self.target_resolution,
            target_crs=self.target_crs,
            bounds=bounds,
            shape=(height, width),
            data_type=dataset_config['data_type'],
            resampling_method=dataset_config.get('resampling_method', self._get_resampling_method(dataset_config['data_type'])),
            band_name=dataset_config['band_name'],
            metadata={
                'storage_table': table_name,
                'resampling_stats': stats,
                'memory_aware': True,
                'engine': self.engine_type,
                'source_resolution': raster_entry.resolution_degrees,
                'target_resolution': self.target_resolution
            }
        )
        
        # Register metadata
        self._register_dataset_metadata(resampled_info)
        
        return resampled_info

    def _store_resampled_dataset(self, info: ResampledDatasetInfo, data: Optional[np.ndarray]):
        """Store resampled dataset in database.
        
        DEPRECATED: This method requires the entire dataset in memory.
        Use windowed storage methods for memory-efficient processing.
        """
        import warnings
        warnings.warn(
            "_store_resampled_dataset() is deprecated as it requires entire dataset in memory. "
            "Use windowed storage methods for memory-efficient processing.",
            DeprecationWarning,
            stacklevel=2
        )
        # For consistency, store all data in DB (including passthrough)
        # This ensures lazy loading works uniformly for all datasets
        if data is None:
            raise ValueError(f"Data cannot be None for dataset {info.name}")
        
        # CRITICAL FIX: Get actual bounds from raster file for all datasets
        actual_bounds = self.catalog.get_raster_bounds(info.source_path)
        
        # Update info with actual bounds and store in metadata for reference
        info.bounds = actual_bounds
        info.metadata['actual_bounds'] = list(actual_bounds)
        info.metadata['bounds'] = list(actual_bounds)  # For backward compatibility
        
        try:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                
                # Create data table name based on type
                if info.metadata.get('passthrough', False):
                    table_name = f"passthrough_{info.name.replace('-', '_')}"
                    logger.info(f"Storing passthrough dataset {info.name} to database table {table_name}")
                else:
                    table_name = f"resampled_{info.name.replace('-', '_')}"
                    logger.info(f"Storing resampled dataset {info.name} to database table {table_name}")
                
                logger.info(f"Using actual raster bounds: {actual_bounds}")
                
                # Insert or update metadata
                cur.execute("""
                    INSERT INTO resampled_datasets 
                    (name, source_path, target_resolution, target_crs, bounds, 
                     shape_height, shape_width, data_type, resampling_method, 
                     band_name, data_table_name, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        source_path = EXCLUDED.source_path,
                        target_resolution = EXCLUDED.target_resolution,
                        target_crs = EXCLUDED.target_crs,
                        bounds = EXCLUDED.bounds,
                        shape_height = EXCLUDED.shape_height,
                        shape_width = EXCLUDED.shape_width,
                        data_type = EXCLUDED.data_type,
                        resampling_method = EXCLUDED.resampling_method,
                        band_name = EXCLUDED.band_name,
                        data_table_name = EXCLUDED.data_table_name,
                        metadata = EXCLUDED.metadata
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
                
                # Truncate existing data to avoid conflicts
                logger.info(f"Truncating existing data in {table_name}")
                cur.execute(f"TRUNCATE TABLE {table_name}")
                
                # Store non-NaN values
                if not np.all(np.isnan(data)):
                    valid_mask = ~np.isnan(data)
                    rows, cols = np.where(valid_mask)
                    values = data[valid_mask]
                    
                    data_to_insert = [(int(r), int(c), float(v)) for r, c, v in zip(rows, cols, values)]
                    
                    # Batch insert
                    from psycopg2.extras import execute_values
                    logger.info(f"Inserting {len(data_to_insert):,} values into {table_name}...")
                    insert_start = time.time()
                    execute_values(
                        cur,
                        f"INSERT INTO {table_name} (row_idx, col_idx, value) VALUES %s",
                        data_to_insert,
                        page_size=10000
                    )
                    insert_time = time.time() - insert_start
                    logger.info(f"Insert completed in {insert_time:.1f}s ({len(data_to_insert)/insert_time:.0f} values/sec)")
                
                conn.commit()
                logger.info(f"✅ Stored dataset {info.name} ({len(data_to_insert) if 'data_to_insert' in locals() else 0} non-NaN values)")
                
        except Exception as e:
            logger.error(f"Failed to store dataset: {e}")
            raise
    
    # BaseProcessor abstract method implementations
    def process_single(self, item: Any) -> Any:
        """Process single item."""
        if isinstance(item, dict) and 'name' in item:
            return self.resample_dataset(item)
        return item
    
    def load_passthrough_data(self, info: ResampledDatasetInfo) -> Optional[np.ndarray]:
        print(f"🔄 DEBUG: load_passthrough_data called for {info.name}")
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
            
            # Verify the data table actually exists
            table_name = dataset_row.get('data_table_name')
            if table_name:
                with self.db.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT 1 
                                FROM information_schema.tables 
                                WHERE table_schema = 'public' 
                                AND table_name = %s
                            )
                        """, (table_name,))
                        table_exists = cur.fetchone()[0]
                        
                        if not table_exists:
                            logger.warning(f"Data table {table_name} missing for {dataset_name}, will reprocess")
                            # Delete the orphaned metadata entry
                            cur.execute("DELETE FROM resampled_datasets WHERE name = %s", (dataset_name,))
                            conn.commit()
                            return None
            
            return resampled_info
            
        except Exception as e:
            logger.error(f"Failed to retrieve dataset {dataset_name}: {e}")
            return None

    def load_resampled_data_chunk(self, dataset_name: str,
                                  row_start: int, row_end: int,
                                  col_start: int, col_end: int) -> Optional[np.ndarray]:
        """
        Load a spatial chunk of resampled data from database.
        
        Args:
            dataset_name: Name of the dataset
            row_start: Starting row index (inclusive)
            row_end: Ending row index (exclusive)
            col_start: Starting column index (inclusive)
            col_end: Ending column index (exclusive)
            
        Returns:
            Numpy array chunk or None if failed
        """
        try:
            from src.database.schema import schema
            
            # Get dataset info
            datasets = schema.get_resampled_datasets({'name': dataset_name})
            if not datasets:
                logger.error(f"Dataset {dataset_name} not found")
                return None
            
            dataset_info = datasets[0]
            table_name = dataset_info['data_table_name']
            
            # Create chunk array
            chunk_shape = (row_end - row_start, col_end - col_start)
            chunk_data = np.full(chunk_shape, np.nan, dtype=np.float32)
            
            # Load only the chunk data from table
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(f"""
                    SELECT row_idx - %s as rel_row, col_idx - %s as rel_col, value 
                    FROM {table_name} 
                    WHERE row_idx >= %s AND row_idx < %s
                    AND col_idx >= %s AND col_idx < %s
                    ORDER BY row_idx, col_idx
                """, (row_start, col_start, row_start, row_end, col_start, col_end))
                
                rows = cur.fetchall()
                
                # Fill chunk array
                for row in rows:
                    rel_row, rel_col, value = row
                    if 0 <= rel_row < chunk_shape[0] and 0 <= rel_col < chunk_shape[1]:
                        chunk_data[rel_row, rel_col] = value
                
                logger.debug(f"Loaded chunk [{row_start}:{row_end}, {col_start}:{col_end}] for {dataset_name}: {len(rows)} values")
                return chunk_data
                
        except Exception as e:
            logger.error(f"Failed to load chunk for {dataset_name}: {e}")
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
            
            # Load data from database table (works for both passthrough and resampled)
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
    
    # Validation methods
    def _validate_source_bounds(self, raster_entry, dataset_name: str) -> None:
        """Validate source raster bounds."""
        validation_data = {
            'bounds': raster_entry.bounds,
            'crs': raster_entry.crs or 'EPSG:4326'
        }
        
        result = self.bounds_validator.validate(validation_data)
        self.validation_results.append({
            'stage': 'source_bounds',
            'dataset': dataset_name,
            'result': result
        })
        
        if not result.is_valid:
            error_messages = [issue.message for issue in result.issues 
                            if issue.severity == ValidationSeverity.ERROR]
            logger.error(f"Source bounds validation failed for {dataset_name}: {error_messages}")
            if result.has_errors:
                raise ValueError(f"Invalid source bounds for dataset {dataset_name}: {error_messages}")
        
        # Log warnings
        warnings = [issue.message for issue in result.issues 
                   if issue.severity == ValidationSeverity.WARNING]
        for warning in warnings:
            logger.warning(f"Source bounds validation warning for {dataset_name}: {warning}")
    
    def _validate_coordinate_transformation(self, source_crs: str, dataset_name: str) -> None:
        """Validate coordinate transformation from source to target CRS."""
        if source_crs == self.target_crs:
            logger.info(f"No CRS transformation needed for {dataset_name}")
            return
        
        # Create sample points for transformation validation
        sample_points = [
            (0.0, 0.0),      # Origin
            (1.0, 1.0),      # Unit point
            (-1.0, -1.0),    # Negative point
            (180.0, 85.0),   # Near-pole point (if geographic)
        ]
        
        validation_data = {
            'source_crs': source_crs,
            'target_crs': self.target_crs,
            'sample_points': sample_points
        }
        
        result = self.transform_validator.validate(validation_data)
        self.validation_results.append({
            'stage': 'coordinate_transform',
            'dataset': dataset_name,
            'result': result
        })
        
        if not result.is_valid:
            error_messages = [issue.message for issue in result.issues 
                            if issue.severity == ValidationSeverity.ERROR]
            logger.error(f"Coordinate transformation validation failed for {dataset_name}: {error_messages}")
            if result.has_errors:
                raise ValueError(f"Invalid coordinate transformation for dataset {dataset_name}: {error_messages}")
        
        # Log warnings
        warnings = [issue.message for issue in result.issues 
                   if issue.severity == ValidationSeverity.WARNING]
        for warning in warnings:
            logger.warning(f"Coordinate transformation validation warning for {dataset_name}: {warning}")
    
    def _validate_resampled_data(self, resampled_data: np.ndarray, dataset_name: str) -> None:
        """Validate resampled data quality."""
        if resampled_data is None:
            return
        
        # Convert to DataFrame for validation
        data_flat = resampled_data.flatten()
        data_flat = data_flat[~np.isnan(data_flat)]  # Remove NaN values
        
        if len(data_flat) == 0:
            logger.warning(f"All resampled data is NaN for {dataset_name}")
            return
        
        import pandas as pd
        df = pd.DataFrame({dataset_name: data_flat})
        
        result = self.value_validator.validate(df)
        self.validation_results.append({
            'stage': 'resampled_data',
            'dataset': dataset_name,
            'result': result
        })
        
        if not result.is_valid:
            error_messages = [issue.message for issue in result.issues 
                            if issue.severity == ValidationSeverity.ERROR]
            logger.error(f"Resampled data validation failed for {dataset_name}: {error_messages}")
            # Don't raise error for data quality issues - just log
        
        # Log warnings and info
        warnings = [issue.message for issue in result.issues 
                   if issue.severity == ValidationSeverity.WARNING]
        for warning in warnings:
            logger.warning(f"Resampled data validation warning for {dataset_name}: {warning}")
        
        # Log data statistics
        logger.info(f"Resampled data stats for {dataset_name}: "
                   f"min={data_flat.min():.4f}, max={data_flat.max():.4f}, "
                   f"mean={data_flat.mean():.4f}, std={data_flat.std():.4f}")
    
    def _validate_output_bounds(self, calculated_bounds: Tuple[float, float, float, float], 
                               expected_bounds: Tuple[float, float, float, float], 
                               dataset_name: str) -> None:
        """Validate output bounds consistency."""
        validation_data = {
            'bounds': calculated_bounds,
            'metadata_bounds': expected_bounds,
            'crs': self.target_crs
        }
        
        result = self.bounds_validator.validate(validation_data)
        self.validation_results.append({
            'stage': 'output_bounds',
            'dataset': dataset_name,
            'result': result
        })
        
        if not result.is_valid:
            error_messages = [issue.message for issue in result.issues 
                            if issue.severity == ValidationSeverity.ERROR]
            logger.error(f"Output bounds validation failed for {dataset_name}: {error_messages}")
            # Don't raise error for bounds mismatch - just log for investigation
        
        # Log warnings
        warnings = [issue.message for issue in result.issues 
                   if issue.severity == ValidationSeverity.WARNING]
        for warning in warnings:
            logger.warning(f"Output bounds validation warning for {dataset_name}: {warning}")
    
    def get_validation_results(self) -> List[Dict]:
        """Get all validation results for external reporting."""
        return self.validation_results.copy()
    
    def _report_validation_summary(self) -> None:
        """Report summary of all validation results."""
        if not self.validation_results:
            logger.info("No validation results to report")
            return
        
        total_validations = len(self.validation_results)
        failed_validations = sum(1 for v in self.validation_results if not v['result'].is_valid)
        
        total_errors = sum(v['result'].error_count for v in self.validation_results)
        total_warnings = sum(v['result'].warning_count for v in self.validation_results)
        
        logger.info("=== RESAMPLING VALIDATION SUMMARY ===")
        logger.info(f"Total validations: {total_validations}")
        logger.info(f"Failed validations: {failed_validations}")
        logger.info(f"Total errors: {total_errors}")
        logger.info(f"Total warnings: {total_warnings}")
        
        # Report by stage
        stages = {}
        for v in self.validation_results:
            stage = v['stage']
            if stage not in stages:
                stages[stage] = {'count': 0, 'errors': 0, 'warnings': 0}
            stages[stage]['count'] += 1
            stages[stage]['errors'] += v['result'].error_count
            stages[stage]['warnings'] += v['result'].warning_count
        
        for stage, metrics in stages.items():
            logger.info(f"Stage '{stage}': {metrics['count']} validations, "
                       f"{metrics['errors']} errors, {metrics['warnings']} warnings")
        
        logger.info("=== END RESAMPLING VALIDATION SUMMARY ===")
    
    def _calculate_output_bounds_from_data(self, data: np.ndarray, original_bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Calculate output bounds based on resampled data shape and target resolution."""
        if data is None:
            return original_bounds
        
        height, width = data.shape
        
        # Calculate bounds based on target resolution
        minx = original_bounds[0]
        maxy = original_bounds[3]
        maxx = minx + width * self.target_resolution
        miny = maxy - height * self.target_resolution
        
        return (minx, miny, maxx, maxy)