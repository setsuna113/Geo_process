# src/resampling/engines/gdal_resampler.py
"""Enhanced GDAL-based resampling with timeout and progress support."""

import numpy as np
from osgeo import gdal, osr
from typing import Union, Optional, Tuple, Callable
import xarray as xr
import logging
import signal
import threading
from contextlib import contextmanager

from .base_resampler import BaseResampler, ResamplingResult
from ..strategies.area_weighted import AreaWeightedStrategy
from ..strategies.sum_aggregation import SumAggregationStrategy
from ..strategies.majority_vote import MajorityVoteStrategy
from src.base.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


class GDALTimeoutError(Exception):
    """GDAL operation timeout error."""
    pass


@contextmanager
def gdal_timeout(seconds: int):
    """Context manager for GDAL operation timeouts."""
    def timeout_handler(signum, frame):
        raise GDALTimeoutError(f"GDAL operation timed out after {seconds} seconds")
    
    # Set signal alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class GDALResampler(BaseResampler):
    """Enhanced GDAL-based resampling engine with progress and timeout support."""
    
    # GDAL resampling method mapping
    GDAL_METHODS = {
        'nearest': gdal.GRA_NearestNeighbour,
        'bilinear': gdal.GRA_Bilinear,
        'cubic': gdal.GRA_Cubic,
        'cubicspline': gdal.GRA_CubicSpline,
        'lanczos': gdal.GRA_Lanczos,
        'average': gdal.GRA_Average,
        'mode': gdal.GRA_Mode,
        'max': gdal.GRA_Max,
        'min': gdal.GRA_Min,
        'median': gdal.GRA_Med,
        'q1': gdal.GRA_Q1,
        'q3': gdal.GRA_Q3
    }
    
    def __init__(self, config):
        """Initialize enhanced GDAL resampler."""
        super().__init__(config)
        self.memory_manager = get_memory_manager()
        
        # Timeout configuration
        self.timeout_seconds = getattr(config, 'gdal_timeout', 300)  # 5 minutes default
        
        # GDAL configuration
        gdal.SetConfigOption('GDAL_CACHEMAX', str(getattr(config, 'gdal_cache_mb', 512)))
        gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
        
        self._register_strategies()
    
    def _register_strategies(self):
        """Register custom resampling strategies."""
        self.strategies = {
            'area_weighted': AreaWeightedStrategy(),
            'sum': SumAggregationStrategy(),
            'majority': MajorityVoteStrategy(),
            # GDAL built-in methods
            **{name: None for name in self.GDAL_METHODS}
        }
    
    def resample(self, 
                 source_data: Union[np.ndarray, xr.DataArray],
                 source_bounds: Optional[Tuple[float, float, float, float]] = None,
                 target_bounds: Optional[Tuple[float, float, float, float]] = None,
                 progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """Enhanced resample with timeout protection."""
        self.validate_config()
        
        # Convert xarray to numpy if needed
        if isinstance(source_data, xr.DataArray):
            source_array = source_data.values
            if source_bounds is None:
                source_bounds = (
                    float(source_data.lon.min()),
                    float(source_data.lat.min()),
                    float(source_data.lon.max()),
                    float(source_data.lat.max())
                )
        else:
            source_array = source_data
        
        if source_bounds is None:
            raise ValueError("Source bounds must be provided for numpy arrays")
        
        if target_bounds is None:
            target_bounds = source_bounds
        
        # Check if we need custom strategy or can use GDAL
        if self.config.method in self.GDAL_METHODS:
            return self._resample_gdal_with_timeout(
                source_array, source_bounds, target_bounds, progress_callback
            )
        else:
            return self._resample_custom(
                source_array, source_bounds, target_bounds, progress_callback
            )
    
    def _resample_gdal_with_timeout(self,
                                   source_array: np.ndarray,
                                   source_bounds: Tuple[float, float, float, float],
                                   target_bounds: Tuple[float, float, float, float],
                                   progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """GDAL resampling with timeout protection."""
        try:
            with gdal_timeout(self.timeout_seconds):
                return self._resample_gdal(
                    source_array, source_bounds, target_bounds, progress_callback
                )
        except GDALTimeoutError as e:
            logger.error(f"GDAL resampling timed out: {e}")
            raise RuntimeError(f"Resampling timed out after {self.timeout_seconds} seconds")
    
    def _resample_gdal(self, 
                       source_array: np.ndarray,
                       source_bounds: Tuple[float, float, float, float],
                       target_bounds: Tuple[float, float, float, float],
                       progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """Enhanced GDAL resampling with progress monitoring."""
        
        # Monitor memory before operation
        memory_monitor = self.memory_manager.monitor_operation(
            "gdal_resample",
            source_array.nbytes / (1024 * 1024)
        )
        
        # Create temporary source dataset
        driver = gdal.GetDriverByName('MEM')
        src_height, src_width = source_array.shape
        src_ds = driver.Create('', src_width, src_height, 1, gdal.GDT_Float32)
        
        # Set geotransform
        minx, miny, maxx, maxy = source_bounds
        src_transform = [
            minx,
            (maxx - minx) / src_width,
            0,
            maxy,
            0,
            -(maxy - miny) / src_height
        ]
        src_ds.SetGeoTransform(src_transform)
        
        # Set projection
        srs = osr.SpatialReference()
        srs.SetFromUserInput(self.config.source_crs)
        src_ds.SetProjection(srs.ExportToWkt())
        
        # Write data
        src_band = src_ds.GetRasterBand(1)
        src_band.WriteArray(source_array)
        if self.config.nodata_value is not None:
            src_band.SetNoDataValue(self.config.nodata_value)
        
        # Calculate target dimensions
        target_shape = self.calculate_output_shape(target_bounds)
        tgt_height, tgt_width = target_shape
        
        # Create target dataset
        tgt_ds = driver.Create('', tgt_width, tgt_height, 1, gdal.GDT_Float32)
        
        # Set target geotransform
        tgt_minx, _, _, tgt_maxy = target_bounds
        tgt_transform = [
            tgt_minx,
            self.config.target_resolution,
            0,
            tgt_maxy,
            0,
            -self.config.target_resolution
        ]
        tgt_ds.SetGeoTransform(tgt_transform)
        tgt_ds.SetProjection(srs.ExportToWkt())
        
        # Perform resampling with progress
        gdal_method = self.GDAL_METHODS[self.config.method]
        
        # Enhanced progress callback
        self._last_progress = 0
        
        def gdal_progress(complete, message, user_data):
            progress_pct = int(complete * 100)
            
            # Update memory monitor
            memory_monitor.update_progress(progress_pct)
            
            # Call user callback
            if progress_callback and progress_pct > self._last_progress:
                progress_callback(progress_pct)
                self._last_progress = progress_pct
            
            # Check for cancellation
            if hasattr(self, '_should_stop') and self._should_stop.is_set():
                return 0  # Cancel GDAL operation
            
            return 1  # Continue
        
        # Setup warp options with memory limit
        warp_options = gdal.WarpOptions(
            resampleAlg=gdal_method,
            callback=gdal_progress if progress_callback else None,
            warpMemoryLimit=self.config.memory_limit_mb * 1024 * 1024,
            multithread=True
        )
        
        # Perform resampling
        try:
            gdal.Warp(
                tgt_ds,
                src_ds,
                options=warp_options
            )
        except Exception as e:
            logger.error(f"GDAL warp failed: {e}")
            raise
        
        # Read result
        result_array = tgt_ds.GetRasterBand(1).ReadAsArray()
        
        # Handle data type conversion
        result_array = self.handle_dtype_conversion(result_array)
        
        # Calculate coverage if requested
        coverage_map = None
        if self.config.validate_output:
            coverage_map = self._calculate_gdal_coverage(src_ds, tgt_ds)
        
        # Cleanup
        src_ds = None
        tgt_ds = None
        
        # Complete memory monitoring
        memory_stats = memory_monitor.complete()
        
        return ResamplingResult(
            data=result_array,
            bounds=target_bounds,
            resolution=self.config.target_resolution,
            crs=self.config.target_crs,
            method=self.config.method,
            coverage_map=coverage_map,
            metadata={
                'source_shape': source_array.shape,
                'scale_factor': self.scale_factor,
                'engine': 'gdal',
                'memory_stats': memory_stats
            }
        )
    
    def _resample_custom(self,
                        source_array: np.ndarray,
                        source_bounds: Tuple[float, float, float, float],
                        target_bounds: Tuple[float, float, float, float],
                        progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """Enhanced custom strategy resampling."""
        strategy = self.strategies[self.config.method]
        
        # Calculate target shape
        target_shape = self.calculate_output_shape(target_bounds)
        
        # Check memory requirements
        memory_estimate = self._estimate_memory_usage(source_array.shape, target_shape)
        
        if memory_estimate > self.config.memory_limit_mb:
            logger.info("Using chunked custom resampling due to memory constraints")
            return self._resample_custom_chunked(
                source_array, source_bounds, target_bounds, 
                target_shape, strategy, progress_callback
            )
        else:
            # Single pass processing
            mapping = self._build_pixel_mapping(
                source_array.shape, source_bounds,
                target_shape, target_bounds
            )
            
            # Apply strategy with progress
            result_array = strategy.resample(
                source_array, 
                target_shape,
                mapping,
                self.config,
                progress_callback
            )
            
            # Handle data type
            result_array = self.handle_dtype_conversion(result_array)
            
            # Calculate coverage
            coverage_map = None
            if self.config.validate_output:
                coverage_map = self.calculate_coverage(source_array.shape, target_shape, mapping)
            
            return ResamplingResult(
                data=result_array,
                bounds=target_bounds,
                resolution=self.config.target_resolution,
                crs=self.config.target_crs,
                method=self.config.method,
                coverage_map=coverage_map,
                metadata={
                    'source_shape': source_array.shape,
                    'scale_factor': self.scale_factor,
                    'engine': 'gdal_custom'
                }
            )
    
    def _resample_custom_chunked(self,
                               source_array: np.ndarray,
                               source_bounds: Tuple[float, float, float, float],
                               target_bounds: Tuple[float, float, float, float],
                               target_shape: Tuple[int, int],
                               strategy,
                               progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """Chunked custom strategy resampling."""
        logger.info("Starting chunked custom resampling")
        
        # Initialize output
        result_array = np.zeros(target_shape, dtype=np.float32)
        
        # Process in chunks similar to numpy resampler
        chunk_size = min(self.config.chunk_size, target_shape[0] * target_shape[1] // 4)
        total_pixels = target_shape[0] * target_shape[1]
        
        for start_idx in range(0, total_pixels, chunk_size):
            end_idx = min(start_idx + chunk_size, total_pixels)
            
            # Build mapping for this chunk
            chunk_mapping = self._build_chunk_pixel_mapping(
                source_array.shape, source_bounds,
                target_shape, target_bounds,
                start_idx, end_idx
            )
            
            # Apply strategy to chunk
            chunk_result = strategy.resample_chunk(
                source_array,
                target_shape,
                chunk_mapping,
                self.config,
                start_idx,
                end_idx
            )
            
            # Place chunk result
            for i in range(start_idx, end_idx):
                row = i // target_shape[1]
                col = i % target_shape[1]
                result_array[row, col] = chunk_result[i - start_idx]
            
            # Update progress
            if progress_callback:
                progress_pct = (end_idx / total_pixels) * 100
                progress_callback(progress_pct)
        
        # Handle data type
        result_array = self.handle_dtype_conversion(result_array)
        
        return ResamplingResult(
            data=result_array,
            bounds=target_bounds,
            resolution=self.config.target_resolution,
            crs=self.config.target_crs,
            method=self.config.method,
            coverage_map=None,  # Skip coverage for chunked processing
            metadata={
                'source_shape': source_array.shape,
                'scale_factor': self.scale_factor,
                'engine': 'gdal_custom_chunked'
            }
        )
    
    def _build_chunk_pixel_mapping(self,
                                 source_shape: Tuple[int, int],
                                 source_bounds: Tuple[float, float, float, float],
                                 target_shape: Tuple[int, int],
                                 target_bounds: Tuple[float, float, float, float],
                                 start_idx: int,
                                 end_idx: int) -> np.ndarray:
        """Build pixel mapping for a specific chunk."""
        src_height, src_width = source_shape
        tgt_height, tgt_width = target_shape
        
        src_minx, src_miny, src_maxx, src_maxy = source_bounds
        tgt_minx, _, _, tgt_maxy = target_bounds
        
        # Source pixel size
        src_pixel_width = (src_maxx - src_minx) / src_width
        src_pixel_height = (src_maxy - src_miny) / src_height
        
        # Build mapping for chunk
        mapping_list = []
        
        for idx in range(start_idx, end_idx):
            tgt_row = idx // tgt_width
            tgt_col = idx % tgt_width
            
            # Target pixel bounds
            tgt_pixel_minx = tgt_minx + tgt_col * self.config.target_resolution
            tgt_pixel_maxx = tgt_pixel_minx + self.config.target_resolution
            tgt_pixel_maxy = tgt_maxy - tgt_row * self.config.target_resolution
            tgt_pixel_miny = tgt_pixel_maxy - self.config.target_resolution
            
            # Find overlapping source pixels
            src_col_min = max(0, int((tgt_pixel_minx - src_minx) / src_pixel_width))
            src_col_max = min(src_width, int(np.ceil((tgt_pixel_maxx - src_minx) / src_pixel_width)))
            src_row_min = max(0, int((src_maxy - tgt_pixel_maxy) / src_pixel_height))
            src_row_max = min(src_height, int(np.ceil((src_maxy - tgt_pixel_miny) / src_pixel_height)))
            
            # Add mappings
            for src_row in range(src_row_min, src_row_max):
                for src_col in range(src_col_min, src_col_max):
                    source_idx = src_row * src_width + src_col
                    mapping_list.append([idx - start_idx, source_idx])
        
        return np.array(mapping_list, dtype=np.int64)
    
    def _estimate_memory_usage(self, source_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> float:
        """Estimate memory usage in MB."""
        # Similar to numpy resampler
        bytes_per_element = 8
        source_mb = (source_shape[0] * source_shape[1] * bytes_per_element) / (1024 * 1024)
        target_mb = (target_shape[0] * target_shape[1] * bytes_per_element) / (1024 * 1024)
        working_mb = max(source_mb, target_mb) * 0.5
        
        return source_mb + target_mb + working_mb
    
    def _build_pixel_mapping(self,
                            source_shape: Tuple[int, int],
                            source_bounds: Tuple[float, float, float, float],
                            target_shape: Tuple[int, int],
                            target_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Build mapping between source and target pixels."""
        # Implementation remains the same as original
        src_height, src_width = source_shape
        tgt_height, tgt_width = target_shape
        
        src_minx, src_miny, src_maxx, src_maxy = source_bounds
        tgt_minx, _, _, tgt_maxy = target_bounds
        
        # Source pixel size
        src_pixel_width = (src_maxx - src_minx) / src_width
        src_pixel_height = (src_maxy - src_miny) / src_height
        
        # Build mapping list
        mapping_list = []
        
        for tgt_row in range(tgt_height):
            for tgt_col in range(tgt_width):
                # Target pixel bounds
                tgt_pixel_minx = tgt_minx + tgt_col * self.config.target_resolution
                tgt_pixel_maxx = tgt_pixel_minx + self.config.target_resolution
                tgt_pixel_maxy = tgt_maxy - tgt_row * self.config.target_resolution
                tgt_pixel_miny = tgt_pixel_maxy - self.config.target_resolution
                
                # Find overlapping source pixels
                src_col_min = max(0, int((tgt_pixel_minx - src_minx) / src_pixel_width))
                src_col_max = min(src_width, int(np.ceil((tgt_pixel_maxx - src_minx) / src_pixel_width)))
                src_row_min = max(0, int((src_maxy - tgt_pixel_maxy) / src_pixel_height))
                src_row_max = min(src_height, int(np.ceil((src_maxy - tgt_pixel_miny) / src_pixel_height)))
                
                # Add mappings
                target_idx = tgt_row * tgt_width + tgt_col
                for src_row in range(src_row_min, src_row_max):
                    for src_col in range(src_col_min, src_col_max):
                        source_idx = src_row * src_width + src_col
                        mapping_list.append([target_idx, source_idx])
        
        return np.array(mapping_list, dtype=np.int64)
    
    def _calculate_gdal_coverage(self, src_ds, tgt_ds) -> np.ndarray:
        """Calculate coverage map from GDAL datasets."""
        # Create validity mask from source
        src_band = src_ds.GetRasterBand(1)
        src_array = src_band.ReadAsArray()
        
        if self.config.nodata_value is not None:
            validity_mask = (src_array != self.config.nodata_value).astype(np.float32)
        else:
            validity_mask = np.ones_like(src_array, dtype=np.float32)
        
        # Resample validity mask
        driver = gdal.GetDriverByName('MEM')
        mask_ds = driver.Create('', src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Float32)
        mask_ds.SetGeoTransform(src_ds.GetGeoTransform())
        mask_ds.SetProjection(src_ds.GetProjection())
        mask_ds.GetRasterBand(1).WriteArray(validity_mask)
        
        coverage_ds = driver.Create('', tgt_ds.RasterXSize, tgt_ds.RasterYSize, 1, gdal.GDT_Float32)
        coverage_ds.SetGeoTransform(tgt_ds.GetGeoTransform())
        coverage_ds.SetProjection(tgt_ds.GetProjection())
        
        gdal.ReprojectImage(mask_ds, coverage_ds, None, None, gdal.GRA_Average)
        
        coverage_map = coverage_ds.GetRasterBand(1).ReadAsArray() * 100
        
        # Cleanup
        mask_ds = None
        coverage_ds = None
        
        return coverage_map