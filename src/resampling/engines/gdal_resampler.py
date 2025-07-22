# src/resampling/engines/gdal_resampler.py
"""GDAL-based resampling implementation."""

import numpy as np
from osgeo import gdal, osr
from typing import Union, Optional, Tuple, Callable
import xarray as xr
import logging

from .base_resampler import BaseResampler, ResamplingResult
from ..strategies.area_weighted import AreaWeightedStrategy
from ..strategies.sum_aggregation import SumAggregationStrategy
from ..strategies.majority_vote import MajorityVoteStrategy

logger = logging.getLogger(__name__)


class GDALResampler(BaseResampler):
    """GDAL-based resampling engine."""
    
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
        """Resample using GDAL."""
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
            return self._resample_gdal(source_array, source_bounds, target_bounds, progress_callback)
        else:
            return self._resample_custom(source_array, source_bounds, target_bounds, progress_callback)
    
    def _resample_gdal(self, 
                       source_array: np.ndarray,
                       source_bounds: Tuple[float, float, float, float],
                       target_bounds: Tuple[float, float, float, float],
                       progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """Use GDAL's built-in resampling."""
        
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
        
        # Perform resampling
        gdal_method = self.GDAL_METHODS[self.config.method]
        
        if progress_callback:
            def gdal_progress(complete, message, user_data):
                progress_callback(int(complete * 100))
                return 1
            callback = gdal_progress
        else:
            callback = None
        
        gdal.ReprojectImage(
            src_ds, tgt_ds,
            None, None,
            gdal_method,
            callback=callback
        )
        
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
                'engine': 'gdal'
            }
        )
    
    def _resample_custom(self,
                        source_array: np.ndarray,
                        source_bounds: Tuple[float, float, float, float],
                        target_bounds: Tuple[float, float, float, float],
                        progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """Use custom resampling strategy."""
        strategy = self.strategies[self.config.method]
        
        # Calculate target shape
        target_shape = self.calculate_output_shape(target_bounds)
        
        # Build pixel mapping
        mapping = self._build_pixel_mapping(
            source_array.shape, source_bounds,
            target_shape, target_bounds
        )
        
        # Apply strategy
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
    
    def _build_pixel_mapping(self,
                            source_shape: Tuple[int, int],
                            source_bounds: Tuple[float, float, float, float],
                            target_shape: Tuple[int, int],
                            target_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Build mapping between source and target pixels."""
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