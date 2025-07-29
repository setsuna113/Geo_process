# src/resampling/engines/numpy_resampler.py
"""Enhanced NumPy resampling with chunked processing and progress support."""

import numpy as np
from typing import Union, Optional, Tuple, Callable, Iterator
import xarray as xr
from scipy import ndimage
import logging
import gc

from .base_resampler import BaseResampler, ResamplingResult
from ..strategies.area_weighted import AreaWeightedStrategy
from ..strategies.sum_aggregation import SumAggregationStrategy
from ..strategies.majority_vote import MajorityVoteStrategy
from ...base.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


class NumpyResampler(BaseResampler):
    """Enhanced NumPy/SciPy resampling engine with memory-aware processing."""
    
    def __init__(self, config):
        """Initialize enhanced NumPy resampler."""
        super().__init__(config)
        self.config = config
        self.memory_manager = get_memory_manager()
        
        # Initialize attributes
        self.is_upsampling = False
        self.scale_factor = 1.0
        
        # Chunking configuration
        self.chunk_size = getattr(config, 'chunk_size', 1000)
        self.memory_limit_mb = getattr(config, 'memory_limit_mb', 512)
        
        self._register_strategies()
        
        # Cache for intermediate results
        self._intermediate_cache = {}
        self._cache_enabled = getattr(config, 'cache_intermediate', True)
    
    def _register_strategies(self):
        """Register resampling strategies."""
        self.strategies = {
            'area_weighted': AreaWeightedStrategy(),
            'sum': SumAggregationStrategy(),
            'majority': MajorityVoteStrategy(),
            'nearest': self._nearest_neighbor,
            'bilinear': self._bilinear,
            'mean': self._mean_aggregate,
            'max': self._max_aggregate,
            'min': self._min_aggregate
        }
    
    def resample(self,
                 source_data: Union[np.ndarray, xr.DataArray],
                 source_bounds: Optional[Tuple[float, float, float, float]] = None,
                 target_bounds: Optional[Tuple[float, float, float, float]] = None,
                 progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """Enhanced resample with memory management and progress support."""
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
        
        # Calculate target shape and check memory requirements
        target_shape = self.calculate_output_shape(target_bounds)
        memory_estimate = self._estimate_memory_usage(source_array.shape, target_shape)
        
        logger.info(f"Estimated memory usage: {memory_estimate:.1f} MB (limit: {self.memory_limit_mb} MB)")
        
        # Decide on processing strategy
        if memory_estimate > self.memory_limit_mb:
            logger.info("Using chunked processing due to memory constraints")
            return self._resample_chunked(
                source_array, source_bounds, target_bounds, 
                target_shape, progress_callback
            )
        else:
            logger.info("Using single-pass processing")
            return self._resample_single(
                source_array, source_bounds, target_bounds,
                target_shape, progress_callback
            )
    
    def _resample_single(self,
                        source_array: np.ndarray,
                        source_bounds: Tuple[float, float, float, float],
                        target_bounds: Tuple[float, float, float, float],
                        target_shape: Tuple[int, int],
                        progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """Single-pass resampling for smaller datasets."""
        # Get strategy
        strategy = self.strategies.get(self.config.method)
        
        # Progress wrapper
        def wrapped_callback(percent: float):
            if progress_callback:
                progress_callback(percent)
        
        # Apply resampling
        result_array: np.ndarray
        if callable(strategy):
            # Built-in method
            result_array = np.asarray(strategy(source_array, target_shape, wrapped_callback))
        elif strategy is not None:
            # Custom strategy
            mapping = self._build_pixel_mapping(
                source_array.shape, source_bounds,
                target_shape, target_bounds
            )
            result_array = strategy.resample(
                source_array,
                target_shape,
                mapping,
                self.config,
                wrapped_callback
            )
        else:
            raise ValueError(f"Strategy {self.config.method} is not implemented")
        
        # Handle data type
        result_array = self.handle_dtype_conversion(result_array)
        
        # Calculate coverage
        coverage_map = None
        if self.config.validate_output:
            coverage_map = self._calculate_coverage_map(
                source_array, source_bounds,
                target_shape, target_bounds
            )
        
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
                'engine': 'numpy',
                'processing_mode': 'single'
            }
        )
    
    def _resample_chunked(self,
                         source_array: np.ndarray,
                         source_bounds: Tuple[float, float, float, float],
                         target_bounds: Tuple[float, float, float, float],
                         target_shape: Tuple[int, int],
                         progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """Chunked resampling for memory efficiency."""
        logger.info(f"Starting chunked resampling with chunk size: {self.chunk_size}")
        
        # Initialize output array
        result_array = np.zeros(target_shape, dtype=source_array.dtype)
        
        # Get chunks
        chunks = list(self._generate_chunks(source_array.shape, source_bounds))
        total_chunks = len(chunks)
        
        logger.info(f"Processing {total_chunks} chunks")
        
        # Process each chunk
        for i, (chunk_slice, chunk_bounds) in enumerate(chunks):
            # Check memory pressure
            memory_info = self.memory_manager.get_current_memory_usage()
            if memory_info['pressure_level'].value in ['high', 'critical']:
                logger.warning("High memory pressure detected, triggering cleanup")
                self.memory_manager.trigger_cleanup()
                gc.collect()
            
            # Extract chunk data
            chunk_data = source_array[chunk_slice]
            
            # Calculate target bounds for this chunk
            target_chunk_bounds = self._calculate_chunk_target_bounds(
                chunk_bounds, source_bounds, target_bounds
            )
            
            # Calculate target shape for chunk
            chunk_target_shape = self._calculate_chunk_target_shape(target_chunk_bounds)
            
            # Resample chunk
            chunk_result = self._resample_chunk(
                chunk_data, chunk_bounds, target_chunk_bounds, chunk_target_shape
            )
            
            # Place in output array
            self._place_chunk_result(
                result_array, chunk_result, target_chunk_bounds, target_bounds
            )
            
            # Update progress
            if progress_callback:
                progress_percent = ((i + 1) / total_chunks) * 100
                progress_callback(progress_percent)
            
            # Cache intermediate result if enabled
            if self._cache_enabled and i % 10 == 0:
                self._save_intermediate_result(result_array, i)
        
        # Final cleanup
        self._clear_intermediate_cache()
        
        # Handle data type
        result_array = self.handle_dtype_conversion(result_array)
        
        # Calculate coverage
        coverage_map = None
        if self.config.validate_output:
            coverage_map = self._calculate_coverage_map(
                source_array, source_bounds,
                target_shape, target_bounds
            )
        
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
                'engine': 'numpy',
                'processing_mode': 'chunked',
                'chunks_processed': total_chunks
            }
        )
    
    def _generate_chunks(self, 
                        shape: Tuple[int, int],
                        bounds: Tuple[float, float, float, float]) -> Iterator[Tuple[Tuple[slice, slice], Tuple[float, float, float, float]]]:
        """Generate chunk slices and their bounds."""
        height, width = shape
        minx, miny, maxx, maxy = bounds
        
        # Calculate pixel size
        pixel_width = (maxx - minx) / width
        pixel_height = (maxy - miny) / height
        
        # Generate chunks
        for row_start in range(0, height, self.chunk_size):
            row_end = min(row_start + self.chunk_size, height)
            
            for col_start in range(0, width, self.chunk_size):
                col_end = min(col_start + self.chunk_size, width)
                
                # Create slice
                chunk_slice = (slice(row_start, row_end), slice(col_start, col_end))
                
                # Calculate bounds for this chunk
                chunk_minx = minx + col_start * pixel_width
                chunk_maxx = minx + col_end * pixel_width
                chunk_maxy = maxy - row_start * pixel_height
                chunk_miny = maxy - row_end * pixel_height
                
                chunk_bounds = (chunk_minx, chunk_miny, chunk_maxx, chunk_maxy)
                
                yield chunk_slice, chunk_bounds
    
    def _resample_chunk(self,
                       chunk_data: np.ndarray,
                       chunk_bounds: Tuple[float, float, float, float],
                       target_bounds: Tuple[float, float, float, float],
                       target_shape: Tuple[int, int]) -> np.ndarray:
        """Resample a single chunk."""
        strategy = self.strategies.get(self.config.method)
        
        if callable(strategy):
            # Built-in method
            return strategy(chunk_data, target_shape, None)
        elif strategy is not None:
            # Custom strategy
            mapping = self._build_pixel_mapping(
                chunk_data.shape, chunk_bounds,
                target_shape, target_bounds
            )
            return strategy.resample(
                chunk_data,
                target_shape,
                mapping,
                self.config,
                None
            )
        else:
            raise ValueError(f"Strategy {self.config.method} is not implemented")
    
    def _calculate_chunk_target_bounds(self,
                                     chunk_bounds: Tuple[float, float, float, float],
                                     source_bounds: Tuple[float, float, float, float],
                                     target_bounds: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Calculate target bounds for a chunk."""
        # Map chunk bounds to target coordinate system
        # This is simplified - in practice you might need coordinate transformation
        src_minx, src_miny, src_maxx, src_maxy = source_bounds
        tgt_minx, tgt_miny, tgt_maxx, tgt_maxy = target_bounds
        chunk_minx, chunk_miny, chunk_maxx, chunk_maxy = chunk_bounds
        
        # Calculate relative position
        rel_minx = (chunk_minx - src_minx) / (src_maxx - src_minx)
        rel_maxx = (chunk_maxx - src_minx) / (src_maxx - src_minx)
        rel_miny = (chunk_miny - src_miny) / (src_maxy - src_miny)
        rel_maxy = (chunk_maxy - src_miny) / (src_maxy - src_miny)
        
        # Map to target bounds
        target_chunk_minx = tgt_minx + rel_minx * (tgt_maxx - tgt_minx)
        target_chunk_maxx = tgt_minx + rel_maxx * (tgt_maxx - tgt_minx)
        target_chunk_miny = tgt_miny + rel_miny * (tgt_maxy - tgt_miny)
        target_chunk_maxy = tgt_miny + rel_maxy * (tgt_maxy - tgt_miny)
        
        return (target_chunk_minx, target_chunk_miny, target_chunk_maxx, target_chunk_maxy)
    
    def _calculate_chunk_target_shape(self, chunk_bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
        """Calculate target shape for a chunk."""
        minx, miny, maxx, maxy = chunk_bounds
        width = int(np.ceil((maxx - minx) / self.config.target_resolution))
        height = int(np.ceil((maxy - miny) / self.config.target_resolution))
        return (height, width)
    
    def _place_chunk_result(self,
                           output_array: np.ndarray,
                           chunk_result: np.ndarray,
                           chunk_bounds: Tuple[float, float, float, float],
                           full_bounds: Tuple[float, float, float, float]) -> None:
        """Place chunk result in the output array."""
        # Calculate indices in output array
        minx, miny, maxx, maxy = chunk_bounds
        full_minx, full_miny, full_maxx, full_maxy = full_bounds
        
        col_start = int((minx - full_minx) / self.config.target_resolution)
        col_end = col_start + chunk_result.shape[1]
        row_start = int((full_maxy - maxy) / self.config.target_resolution)
        row_end = row_start + chunk_result.shape[0]
        
        # Place result
        output_array[row_start:row_end, col_start:col_end] = chunk_result
    
    def _estimate_memory_usage(self, source_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> float:
        """Estimate memory usage in MB."""
        # Assume float64 for estimation
        bytes_per_element = 8
        
        # Source array
        source_mb = (source_shape[0] * source_shape[1] * bytes_per_element) / (1024 * 1024)
        
        # Target array
        target_mb = (target_shape[0] * target_shape[1] * bytes_per_element) / (1024 * 1024)
        
        # Working memory (conservative estimate)
        working_mb = max(source_mb, target_mb) * 0.5
        
        return source_mb + target_mb + working_mb
    
    def _save_intermediate_result(self, array: np.ndarray, checkpoint_id: int) -> None:
        """Save intermediate result for recovery."""
        if self._cache_enabled:
            self._intermediate_cache[checkpoint_id] = array.copy()
            logger.debug(f"Saved intermediate result at checkpoint {checkpoint_id}")
    
    def _clear_intermediate_cache(self) -> None:
        """Clear intermediate cache."""
        self._intermediate_cache.clear()
        gc.collect()
    
    # Keep existing resampling methods (_nearest_neighbor, _bilinear, etc.)
    # These remain the same as in the original implementation
    
    def _nearest_neighbor(self, source: np.ndarray, target_shape: Tuple[int, int],
                         progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Nearest neighbor resampling."""
        zoom_factors = (target_shape[0] / source.shape[0], target_shape[1] / source.shape[1])
        return np.asarray(ndimage.zoom(source, zoom_factors, order=0))
    
    def _bilinear(self, source: np.ndarray, target_shape: Tuple[int, int],
                  progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Bilinear interpolation."""
        zoom_factors = (target_shape[0] / source.shape[0], target_shape[1] / source.shape[1])
        return np.asarray(ndimage.zoom(source, zoom_factors, order=1))
    
    def _mean_aggregate(self, source: np.ndarray, target_shape: Tuple[int, int],
                       progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Mean aggregation for downsampling."""
        if self.is_upsampling:
            return self._bilinear(source, target_shape, progress_callback)
        
        # Downsampling implementation remains the same
        src_height, src_width = source.shape
        tgt_height, tgt_width = target_shape
        
        block_height = src_height // tgt_height
        block_width = src_width // tgt_width
        
        trimmed = source[:tgt_height * block_height, :tgt_width * block_width]
        reshaped = trimmed.reshape(tgt_height, block_height, tgt_width, block_width)
        
        if self.config.nodata_value is not None:
            mask = reshaped != self.config.nodata_value
            sums = np.sum(np.where(mask, reshaped, 0), axis=(1, 3))
            counts = np.sum(mask, axis=(1, 3))
            # Use appropriate fill value based on data type
            if np.issubdtype(source.dtype, np.integer):
                fill_value = self.config.nodata_value if self.config.nodata_value is not None else 0
                out_array = np.full((tgt_height, tgt_width), fill_value, dtype=source.dtype)
            else:
                out_array = np.full((tgt_height, tgt_width), np.nan, dtype=np.float64)
            result = np.divide(sums, counts, where=counts > 0, out=out_array)
        else:
            result = np.mean(reshaped, axis=(1, 3))
        
        return result
    
    def _max_aggregate(self, source: np.ndarray, target_shape: Tuple[int, int],
                      progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Maximum aggregation."""
        if self.is_upsampling:
            return self._nearest_neighbor(source, target_shape, progress_callback)
        
        # Implementation remains the same
        src_height, src_width = source.shape
        tgt_height, tgt_width = target_shape
        
        block_height = src_height // tgt_height
        block_width = src_width // tgt_width
        
        trimmed = source[:tgt_height * block_height, :tgt_width * block_width]
        reshaped = trimmed.reshape(tgt_height, block_height, tgt_width, block_width)
        
        if self.config.nodata_value is not None:
            mask = reshaped != self.config.nodata_value
            result = np.where(
                np.any(mask, axis=(1, 3), keepdims=True),
                np.nanmax(np.where(mask, reshaped, np.nan), axis=(1, 3)),
                self.config.nodata_value
            ).squeeze()
        else:
            result = np.max(reshaped, axis=(1, 3))
        
        return result
    
    def _min_aggregate(self, source: np.ndarray, target_shape: Tuple[int, int],
                      progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Minimum aggregation."""
        if self.is_upsampling:
            return self._nearest_neighbor(source, target_shape, progress_callback)
        
        # Implementation remains the same
        src_height, src_width = source.shape
        tgt_height, tgt_width = target_shape
        
        block_height = src_height // tgt_height
        block_width = src_width // tgt_width
        
        trimmed = source[:tgt_height * block_height, :tgt_width * block_width]
        reshaped = trimmed.reshape(tgt_height, block_height, tgt_width, block_width)
        
        if self.config.nodata_value is not None:
            mask = reshaped != self.config.nodata_value
            result = np.where(
                np.any(mask, axis=(1, 3), keepdims=True),
                np.nanmin(np.where(mask, reshaped, np.nan), axis=(1, 3)),
                self.config.nodata_value
            ).squeeze()
        else:
            result = np.min(reshaped, axis=(1, 3))
        
        return result
    
    def _build_pixel_mapping(self,
                            source_shape: Tuple[int, int],
                            source_bounds: Tuple[float, float, float, float],
                            target_shape: Tuple[int, int],
                            target_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Build pixel mapping for custom strategies."""
        # Implementation remains the same but with chunked processing
        src_height, src_width = source_shape
        tgt_height, tgt_width = target_shape
        
        src_minx, src_miny, src_maxx, src_maxy = source_bounds
        tgt_minx, _, _, tgt_maxy = target_bounds
        
        src_res_x = (src_maxx - src_minx) / src_width
        src_res_y = (src_maxy - src_miny) / src_height
        
        # For chunked processing
        chunk_size = min(self.config.chunk_size, tgt_height * tgt_width)
        mapping_list = []
        
        for tgt_idx in range(0, tgt_height * tgt_width, chunk_size):
            chunk_mappings = []
            
            for idx in range(tgt_idx, min(tgt_idx + chunk_size, tgt_height * tgt_width)):
                tgt_row = idx // tgt_width
                tgt_col = idx % tgt_width
                
                # Target pixel bounds
                tgt_pixel_minx = tgt_minx + tgt_col * self.config.target_resolution
                tgt_pixel_maxx = tgt_pixel_minx + self.config.target_resolution
                tgt_pixel_maxy = tgt_maxy - tgt_row * self.config.target_resolution
                tgt_pixel_miny = tgt_pixel_maxy - self.config.target_resolution
                
                # Find overlapping source pixels
                src_col_min = max(0, int((tgt_pixel_minx - src_minx) / src_res_x))
                src_col_max = min(src_width, int(np.ceil((tgt_pixel_maxx - src_minx) / src_res_x)))
                src_row_min = max(0, int((src_maxy - tgt_pixel_maxy) / src_res_y))
                src_row_max = min(src_height, int(np.ceil((src_maxy - tgt_pixel_miny) / src_res_y)))
                
                # Add mappings with area weights
                for src_row in range(src_row_min, src_row_max):
                    for src_col in range(src_col_min, src_col_max):
                        src_idx = src_row * src_width + src_col
                        
                        # Calculate overlap area
                        src_pixel_minx = src_minx + src_col * src_res_x
                        src_pixel_maxx = src_pixel_minx + src_res_x
                        src_pixel_maxy = src_maxy - src_row * src_res_y
                        src_pixel_miny = src_pixel_maxy - src_res_y
                        
                        overlap_minx = max(src_pixel_minx, tgt_pixel_minx)
                        overlap_maxx = min(src_pixel_maxx, tgt_pixel_maxx)
                        overlap_miny = max(src_pixel_miny, tgt_pixel_miny)
                        overlap_maxy = min(src_pixel_maxy, tgt_pixel_maxy)
                        
                        overlap_area = (overlap_maxx - overlap_minx) * (overlap_maxy - overlap_miny)
                        src_area = src_res_x * src_res_y
                        weight = overlap_area / src_area
                        
                        chunk_mappings.append([idx, src_idx, weight])
            
            mapping_list.extend(chunk_mappings)
        
        return np.array(mapping_list, dtype=np.float64)
    
    def _calculate_coverage_map(self,
                               source: np.ndarray,
                               source_bounds: Tuple[float, float, float, float],
                               target_shape: Tuple[int, int],
                               target_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Calculate coverage percentage for validation."""
        # Create validity mask
        if self.config.nodata_value is not None:
            validity = (source != self.config.nodata_value).astype(np.float32)
        else:
            validity = np.ones_like(source, dtype=np.float32)
        
        # Resample validity mask
        coverage = self._mean_aggregate(validity, target_shape) * 100
        
        return coverage