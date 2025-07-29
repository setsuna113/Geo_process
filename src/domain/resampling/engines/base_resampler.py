# src/resampling/engines/base_resampler.py
"""Abstract base class for resampling engines."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Any, Callable, Iterator
import numpy as np
import xarray as xr
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResamplingConfig:
    """Configuration for resampling operation."""
    source_resolution: float  # degrees
    target_resolution: float  # degrees or meters
    method: str  # 'area_weighted', 'sum', 'majority', 'bilinear', etc.
    bounds: Optional[Tuple[float, float, float, float]] = None  # minx, miny, maxx, maxy
    source_crs: str = 'EPSG:4326'
    target_crs: str = 'EPSG:4326'
    chunk_size: int = 1000  # pixels
    cache_results: bool = True
    validate_output: bool = True
    preserve_sum: bool = False  # For count data
    nodata_value: Optional[float] = None
    dtype: Optional[np.dtype] = None


@dataclass
class ResamplingResult:
    """Result of resampling operation."""
    data: np.ndarray
    bounds: Tuple[float, float, float, float]
    resolution: float
    crs: str
    method: str
    coverage_map: Optional[np.ndarray] = None  # Percentage of valid source pixels
    uncertainty_map: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_xarray(self) -> xr.DataArray:
        """Convert to xarray DataArray."""
        minx, miny, maxx, maxy = self.bounds
        
        # Create coordinates
        lons = np.linspace(minx, maxx, self.data.shape[1])
        lats = np.linspace(maxy, miny, self.data.shape[0])  # Note: reversed for north-up
        
        # Create DataArray
        attrs = {
            'crs': self.crs,
            'resolution': self.resolution,
            'resampling_method': self.method,
        }
        if self.metadata:
            attrs.update(self.metadata)
            
        da = xr.DataArray(
            self.data,
            dims=['lat', 'lon'],
            coords={'lat': lats, 'lon': lons},
            attrs=attrs
        )
        
        return da


class BaseResampler(ABC):
    """Abstract base class for resampling engines."""
    
    def __init__(self, config: ResamplingConfig):
        self.config = config
        self.strategies = {}
        self._register_strategies()
    
    @abstractmethod
    def _register_strategies(self):
        """Register available resampling strategies."""
        pass
    
    @abstractmethod
    def resample(self, 
                 source_data: Union[np.ndarray, xr.DataArray],
                 source_bounds: Optional[Tuple[float, float, float, float]] = None,
                 target_bounds: Optional[Tuple[float, float, float, float]] = None,
                 progress_callback: Optional[Callable[[float], None]] = None) -> ResamplingResult:
        """
        Resample source data to target resolution.
        
        Args:
            source_data: Input data array
            source_bounds: Source data bounds (minx, miny, maxx, maxy)
            target_bounds: Target output bounds (defaults to source bounds)
            progress_callback: Function to report progress (0-100)
            
        Returns:
            ResamplingResult object
        """
        pass
    
    @abstractmethod
    def resample_windowed(self,
                         source_path: str,
                         target_table: str,
                         db_connection,
                         progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Resample raster data using windowed processing without loading full dataset.
        
        This method reads the source raster in windows, resamples each window,
        and stores results directly to database without accumulating in memory.
        
        Args:
            source_path: Path to source raster file
            target_table: Database table for storing results
            db_connection: Database connection
            progress_callback: Optional callback for progress reporting
            
        Returns:
            Dictionary with processing statistics
        """
        pass
    
    def validate_config(self):
        """Validate resampling configuration."""
        if self.config.source_resolution <= 0:
            raise ValueError("Source resolution must be positive")
        
        if self.config.target_resolution <= 0:
            raise ValueError("Target resolution must be positive")
        
        if self.config.method not in self.strategies:
            raise ValueError(f"Unknown resampling method: {self.config.method}")
        
        # Check if upsampling or downsampling
        self.is_upsampling = self.config.target_resolution < self.config.source_resolution
        self.scale_factor = self.config.source_resolution / self.config.target_resolution
        
        logger.info(f"Resampling from {self.config.source_resolution} to {self.config.target_resolution}")
        logger.info(f"Scale factor: {self.scale_factor:.2f} ({'upsampling' if self.is_upsampling else 'downsampling'})")
    
    def calculate_output_shape(self, bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
        """Calculate output array shape from bounds and target resolution."""
        minx, miny, maxx, maxy = bounds
        width = int(np.ceil((maxx - minx) / self.config.target_resolution))
        height = int(np.ceil((maxy - miny) / self.config.target_resolution))
        return (height, width)
    
    def calculate_coverage(self, 
                          source_shape: Tuple[int, int],
                          target_shape: Tuple[int, int],
                          mapping: np.ndarray) -> np.ndarray:
        """Calculate coverage percentage for each target pixel."""
        coverage = np.zeros(target_shape, dtype=np.float32)
        
        # Count valid source pixels per target pixel
        for target_idx in range(target_shape[0] * target_shape[1]):
            source_indices = mapping[mapping[:, 0] == target_idx, 1]
            if len(source_indices) > 0:
                expected_pixels = int(1 / (self.scale_factor ** 2))
                coverage.flat[target_idx] = len(source_indices) / expected_pixels * 100
        
        return coverage
    
    def handle_dtype_conversion(self, data: np.ndarray) -> np.ndarray:
        """Handle data type conversion with overflow checking."""
        if self.config.dtype is None:
            return data
        
        # Check for potential overflow
        if np.issubdtype(self.config.dtype, np.integer):
            dtype_info = np.iinfo(self.config.dtype)
            if data.min() < dtype_info.min or data.max() > dtype_info.max:
                logger.warning(f"Data range [{data.min()}, {data.max()}] exceeds {self.config.dtype} range")
                logger.warning("Clipping values to prevent overflow")
                data = np.clip(data, dtype_info.min, dtype_info.max)
        
        return data.astype(self.config.dtype)
    
    def generate_resampling_windows(self, source_shape: Tuple[int, int], 
                                  window_size: int = 2048,
                                  overlap: int = 128) -> Iterator[Tuple[Tuple[int, int, int, int], int]]:
        """
        Generate windows for resampling with overlap to avoid edge artifacts.
        
        Args:
            source_shape: (height, width) of source raster
            window_size: Size of windows in pixels
            overlap: Overlap between windows
            
        Yields:
            Tuple of ((row_start, row_end, col_start, col_end), window_index)
        """
        height, width = source_shape
        step_size = window_size - overlap
        
        window_idx = 0
        for row_start in range(0, height, step_size):
            row_end = min(row_start + window_size, height)
            
            for col_start in range(0, width, step_size):
                col_end = min(col_start + window_size, width)
                
                yield (row_start, row_end, col_start, col_end), window_idx
                window_idx += 1