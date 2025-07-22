# src/resampling/engines/numpy_resampler.py
"""Pure NumPy resampling implementation."""

import numpy as np
from typing import Union, Optional, Tuple, Callable
import xarray as xr
from scipy import ndimage
import logging

from .base_resampler import BaseResampler, ResamplingResult
from ..strategies.area_weighted import AreaWeightedStrategy
from ..strategies.sum_aggregation import SumAggregationStrategy
from ..strategies.majority_vote import MajorityVoteStrategy

logger = logging.getLogger(__name__)


class NumpyResampler(BaseResampler):
    """Pure NumPy/SciPy resampling engine."""
    
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
        """Resample using NumPy/SciPy."""
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
        
        # Get strategy
        strategy = self.strategies.get(self.config.method)
        
        # Calculate target shape
        target_shape = self.calculate_output_shape(target_bounds)
        
        # Apply resampling
        result_array: np.ndarray
        if callable(strategy):
            # Built-in method
            result_array = np.asarray(strategy(source_array, target_shape, progress_callback))
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
                progress_callback
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
                'engine': 'numpy'
            }
        )
    
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
        
        # Downsampling - use strided operations
        src_height, src_width = source.shape
        tgt_height, tgt_width = target_shape
        
        # Calculate block size
        block_height = src_height // tgt_height
        block_width = src_width // tgt_width
        
        # Trim source to fit exactly
        trimmed = source[:tgt_height * block_height, :tgt_width * block_width]
        
        # Reshape and aggregate
        reshaped = trimmed.reshape(tgt_height, block_height, tgt_width, block_width)
        
        # Handle nodata
        if self.config.nodata_value is not None:
            mask = reshaped != self.config.nodata_value
            sums = np.sum(np.where(mask, reshaped, 0), axis=(1, 3))
            counts = np.sum(mask, axis=(1, 3))
            result = np.divide(sums, counts, where=counts > 0, out=np.full((tgt_height, tgt_width), np.nan))
        else:
            result = np.mean(reshaped, axis=(1, 3))
        
        return result
    
    def _max_aggregate(self, source: np.ndarray, target_shape: Tuple[int, int],
                      progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Maximum aggregation."""
        if self.is_upsampling:
            return self._nearest_neighbor(source, target_shape, progress_callback)
        
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
        # Similar to GDAL implementation
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
                        
                        # Calculate overlap area (for area-weighted strategies)
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