"""Base class for raster resampling with data type conversion and confidence calculation."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass, field

from .cacheable import Cacheable

logger = logging.getLogger(__name__)


class ResamplingMethod(Enum):
    """Available resampling methods."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    AVERAGE = "average"
    MODE = "mode"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    MEDIAN = "median"
    Q1 = "q1"
    Q3 = "q3"


class AggregationMethod(Enum):
    """Available aggregation methods for downsampling."""
    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    MODE = "mode"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"


@dataclass
class ResamplingConfidence:
    """Confidence metrics for resampling operation."""
    overall_confidence: float  # 0.0 to 1.0
    spatial_confidence: float  # Based on resolution change
    data_confidence: float     # Based on data type compatibility
    method_confidence: float   # Based on method suitability
    
    # Quality metrics
    information_loss: float    # Estimated information loss (0.0 to 1.0)
    aliasing_risk: float      # Risk of aliasing artifacts (0.0 to 1.0)
    interpolation_accuracy: float  # Expected interpolation accuracy
    
    # Recommendations
    recommended_method: Optional[ResamplingMethod] = None
    warnings: List[str] = field(default_factory=list)


class BaseResampler(Cacheable, ABC):
    """
    Abstract base class for raster resampling.
    
    Supports different aggregation methods, data type conversions,
    and resampling confidence calculation.
    """
    
    def __init__(self, 
                 cache_results: bool = True,
                 preserve_nodata: bool = True,
                 **kwargs):
        """
        Initialize resampler.
        
        Args:
            cache_results: Whether to cache resampling results
            preserve_nodata: Whether to preserve no-data values
            **kwargs: Additional resampler-specific parameters
        """
        super().__init__()
        
        self.cache_results = cache_results
        self.preserve_nodata = preserve_nodata
        
        # Configuration
        self._resampler_config = {
            'default_method': ResamplingMethod.BILINEAR,
            'chunk_size': 1024,
            'max_memory_mb': 512,
            'quality_threshold': 0.7,
            'warn_on_upsampling': True,
            'warn_on_extreme_scaling': True,
            'extreme_scale_threshold': 10.0
        }
        self._resampler_config.update(kwargs)
        
        # Supported methods - override in subclasses
        self._supported_methods = [ResamplingMethod.NEAREST, ResamplingMethod.BILINEAR]
        self._supported_dtypes = [np.float32, np.float64, np.int16, np.int32, np.uint16, np.uint32]
        
    @abstractmethod
    def _resample_array(self,
                       data: np.ndarray,
                       target_shape: Tuple[int, int],
                       method: ResamplingMethod,
                       **kwargs) -> np.ndarray:
        """
        Core resampling implementation.
        
        Args:
            data: Input data array
            target_shape: Target (height, width)
            method: Resampling method
            **kwargs: Additional parameters
            
        Returns:
            Resampled array
        """
        pass
        
    def resample(self,
                data: np.ndarray,
                target_shape: Tuple[int, int],
                method: Optional[ResamplingMethod] = None,
                target_dtype: Optional[np.dtype] = None,
                nodata: Optional[Union[int, float]] = None,
                **kwargs) -> Tuple[np.ndarray, ResamplingConfidence]:
        """
        Resample data array to target shape.
        
        Args:
            data: Input data array
            target_shape: Target (height, width)
            method: Resampling method (auto-select if None)
            target_dtype: Target data type (preserve if None)
            nodata: No-data value
            **kwargs: Additional resampling parameters
            
        Returns:
            Tuple of (resampled_data, confidence_metrics)
        """
        # Validate inputs
        if len(data.shape) < 2:
            raise ValueError("Input data must be at least 2D")
            
        # Auto-select method if not provided
        if method is None:
            method = self._select_optimal_method(data, target_shape)
            
        # Validate method support
        if method not in self._supported_methods:
            logger.warning(f"Method {method} not supported, falling back to {self._supported_methods[0]}")
            method = self._supported_methods[0]
            
        # Check cache if enabled
        cache_key = None
        if self.cache_results:
            cache_key = self._generate_resample_cache_key(data, target_shape, method, target_dtype, nodata)
            cached_result = self.get_cached(cache_key)
            if cached_result is not None:
                return cached_result
                
        # Calculate confidence before resampling
        confidence = self._calculate_resampling_confidence(data, target_shape, method, target_dtype)
        
        # Perform resampling
        try:
            # Handle multi-band data
            if len(data.shape) == 3:  # (bands, height, width)
                resampled_bands = []
                for band_idx in range(data.shape[0]):
                    band_data = data[band_idx]
                    resampled_band = self._resample_array(band_data, target_shape, method, **kwargs)
                    resampled_bands.append(resampled_band)
                resampled_data = np.stack(resampled_bands, axis=0)
            else:  # Single band (height, width)
                resampled_data = self._resample_array(data, target_shape, method, **kwargs)
                
            # Handle data type conversion
            if target_dtype is not None and resampled_data.dtype != target_dtype:
                resampled_data = self._convert_dtype(resampled_data, target_dtype, nodata)
                
            # Handle no-data values
            if self.preserve_nodata and nodata is not None:
                resampled_data = self._preserve_nodata(resampled_data, nodata)
                
            # Cache result if enabled
            if self.cache_results and cache_key:
                self.cache(cache_key, (resampled_data, confidence))
                
            return resampled_data, confidence
            
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            confidence.overall_confidence = 0.0
            confidence.warnings.append(f"Resampling failed: {str(e)}")
            raise
            
    def aggregate(self,
                 data: np.ndarray,
                 scale_factor: Union[int, Tuple[int, int]],
                 method: AggregationMethod = AggregationMethod.MEAN,
                 nodata: Optional[Union[int, float]] = None) -> np.ndarray:
        """
        Aggregate data by downsampling.
        
        Args:
            data: Input data array
            scale_factor: Downsampling factor
            method: Aggregation method
            nodata: No-data value to exclude from aggregation
            
        Returns:
            Aggregated array
        """
        if isinstance(scale_factor, int):
            scale_y = scale_x = scale_factor
        else:
            scale_y, scale_x = scale_factor
            
        if len(data.shape) == 2:  # Single band
            return self._aggregate_2d(data, scale_y, scale_x, method, nodata)
        elif len(data.shape) == 3:  # Multi-band
            aggregated_bands = []
            for band_idx in range(data.shape[0]):
                agg_band = self._aggregate_2d(data[band_idx], scale_y, scale_x, method, nodata)
                aggregated_bands.append(agg_band)
            return np.stack(aggregated_bands, axis=0)
        else:
            raise ValueError("Data must be 2D or 3D")
            
    def upsample(self,
                data: np.ndarray,
                scale_factor: Union[int, Tuple[int, int]],
                method: ResamplingMethod = ResamplingMethod.BILINEAR,
                **kwargs) -> np.ndarray:
        """
        Upsample data by increasing resolution.
        
        Args:
            data: Input data array
            scale_factor: Upsampling factor
            method: Interpolation method
            **kwargs: Additional parameters
            
        Returns:
            Upsampled array
        """
        if isinstance(scale_factor, int):
            scale_y = scale_x = scale_factor
        else:
            scale_y, scale_x = scale_factor
            
        if len(data.shape) == 2:
            height, width = data.shape
            target_shape = (height * scale_y, width * scale_x)
        elif len(data.shape) == 3:
            _, height, width = data.shape
            target_shape = (height * scale_y, width * scale_x)
        else:
            raise ValueError("Data must be 2D or 3D")
            
        result, _ = self.resample(data, target_shape, method, **kwargs)
        return result
        
    def get_supported_methods(self) -> List[ResamplingMethod]:
        """Get list of supported resampling methods."""
        return self._supported_methods.copy()
        
    def get_supported_dtypes(self) -> List[np.dtype]:
        """Get list of supported data types."""
        return self._supported_dtypes.copy()
        
    def is_method_suitable(self,
                          data: np.ndarray,
                          target_shape: Tuple[int, int],
                          method: ResamplingMethod) -> bool:
        """
        Check if resampling method is suitable for the given transformation.
        
        Args:
            data: Input data
            target_shape: Target shape
            method: Resampling method
            
        Returns:
            True if method is suitable
        """
        if method not in self._supported_methods:
            return False
            
        # Calculate scale factors
        scale_y = target_shape[0] / data.shape[-2]
        scale_x = target_shape[1] / data.shape[-1]
        
        # Check method suitability based on scale
        if method == ResamplingMethod.NEAREST:
            return True  # Always suitable
        elif method in [ResamplingMethod.BILINEAR, ResamplingMethod.BICUBIC]:
            return scale_x > 0.1 and scale_y > 0.1  # Not too much downsampling
        elif method == ResamplingMethod.LANCZOS:
            return scale_x > 0.5 and scale_y > 0.5  # Prefer for moderate scaling
        elif method in [ResamplingMethod.AVERAGE, ResamplingMethod.MODE]:
            return scale_x <= 1.0 and scale_y <= 1.0  # Only for downsampling
        else:
            return True
            
    def _select_optimal_method(self,
                              data: np.ndarray,
                              target_shape: Tuple[int, int]) -> ResamplingMethod:
        """Select optimal resampling method based on data and transformation."""
        scale_y = target_shape[0] / data.shape[-2]
        scale_x = target_shape[1] / data.shape[-1]
        avg_scale = (scale_x + scale_y) / 2
        
        # Check data type
        is_categorical = data.dtype in [np.int8, np.uint8, np.int16, np.uint16] and np.max(data) < 256
        
        # Select method based on scale and data type
        if is_categorical:
            return ResamplingMethod.NEAREST
        elif avg_scale > 2.0:  # Significant upsampling
            if ResamplingMethod.BICUBIC in self._supported_methods:
                return ResamplingMethod.BICUBIC
            elif ResamplingMethod.BILINEAR in self._supported_methods:
                return ResamplingMethod.BILINEAR
            else:
                return ResamplingMethod.NEAREST
        elif avg_scale < 0.5:  # Significant downsampling
            if ResamplingMethod.AVERAGE in self._supported_methods:
                return ResamplingMethod.AVERAGE
            elif ResamplingMethod.BILINEAR in self._supported_methods:
                return ResamplingMethod.BILINEAR
            else:
                return ResamplingMethod.NEAREST
        else:  # Moderate scaling
            if ResamplingMethod.BILINEAR in self._supported_methods:
                return ResamplingMethod.BILINEAR
            else:
                return ResamplingMethod.NEAREST
                
    def _calculate_resampling_confidence(self,
                                       data: np.ndarray,
                                       target_shape: Tuple[int, int],
                                       method: ResamplingMethod,
                                       target_dtype: Optional[np.dtype]) -> ResamplingConfidence:
        """Calculate confidence metrics for resampling operation."""
        scale_y = target_shape[0] / data.shape[-2]
        scale_x = target_shape[1] / data.shape[-1]
        avg_scale = (scale_x + scale_y) / 2
        
        # Spatial confidence based on scale factor
        if 0.8 <= avg_scale <= 1.2:  # Near 1:1
            spatial_confidence = 1.0
        elif 0.5 <= avg_scale <= 2.0:  # Moderate scaling
            spatial_confidence = 0.8
        elif 0.1 <= avg_scale <= 5.0:  # Significant scaling
            spatial_confidence = 0.6
        else:  # Extreme scaling
            spatial_confidence = 0.3
            
        # Data confidence based on dtype compatibility
        data_confidence = 1.0
        if target_dtype is not None:
            if target_dtype != data.dtype:
                if np.can_cast(data.dtype, target_dtype):
                    data_confidence = 0.9
                else:
                    data_confidence = 0.7
                    
        # Method confidence based on suitability
        method_confidence = 1.0 if self.is_method_suitable(data, target_shape, method) else 0.6
        
        # Overall confidence
        overall_confidence = (spatial_confidence * data_confidence * method_confidence) ** (1/3)
        
        # Information loss estimate
        if avg_scale < 1.0:  # Downsampling
            information_loss = min(1.0, 1.0 - avg_scale)
        else:  # Upsampling
            information_loss = 0.1  # Minimal loss for upsampling
            
        # Aliasing risk
        if avg_scale < 0.5 and method not in [ResamplingMethod.AVERAGE, ResamplingMethod.MODE]:
            aliasing_risk = 0.7
        else:
            aliasing_risk = 0.1
            
        # Interpolation accuracy
        if method == ResamplingMethod.NEAREST:
            interpolation_accuracy = 0.6
        elif method == ResamplingMethod.BILINEAR:
            interpolation_accuracy = 0.8
        elif method == ResamplingMethod.BICUBIC:
            interpolation_accuracy = 0.9
        else:
            interpolation_accuracy = 0.7
            
        # Generate warnings
        warnings = []
        if avg_scale > 5.0:
            warnings.append("Extreme upsampling may cause pixelation")
        elif avg_scale < 0.1:
            warnings.append("Extreme downsampling may cause significant information loss")
        if aliasing_risk > 0.5:
            warnings.append("High risk of aliasing artifacts")
            
        return ResamplingConfidence(
            overall_confidence=overall_confidence,
            spatial_confidence=spatial_confidence,
            data_confidence=data_confidence,
            method_confidence=method_confidence,
            information_loss=information_loss,
            aliasing_risk=aliasing_risk,
            interpolation_accuracy=interpolation_accuracy,
            recommended_method=self._select_optimal_method(data, target_shape),
            warnings=warnings
        )
        
    def _convert_dtype(self,
                      data: np.ndarray,
                      target_dtype: np.dtype,
                      nodata: Optional[Union[int, float]]) -> np.ndarray:
        """Convert data to target dtype with proper scaling and nodata handling."""
        if data.dtype == target_dtype:
            return data
            
        # Handle no-data mask
        nodata_mask = None
        if nodata is not None:
            nodata_mask = (data == nodata)
            
        # Convert data type
        if np.can_cast(data.dtype, target_dtype):
            # Safe cast
            converted = data.astype(target_dtype)
        else:
            # Scale conversion needed
            if np.issubdtype(data.dtype, np.floating) and np.issubdtype(target_dtype, np.integer):
                # Float to int - clip and round
                info = np.iinfo(target_dtype)
                clipped = np.clip(data, info.min, info.max)
                converted = np.round(clipped).astype(target_dtype)
            elif np.issubdtype(data.dtype, np.integer) and np.issubdtype(target_dtype, np.floating):
                # Int to float - direct conversion
                converted = data.astype(target_dtype)
            else:
                # Other conversions - scale to 0-1 range then to target
                data_min, data_max = np.min(data), np.max(data)
                if data_max > data_min:
                    normalized = (data - data_min) / (data_max - data_min)
                else:
                    normalized = np.zeros_like(data)
                    
                if np.issubdtype(target_dtype, np.integer):
                    info = np.iinfo(target_dtype)
                    converted = (normalized * (info.max - info.min) + info.min).astype(target_dtype)
                else:
                    converted = normalized.astype(target_dtype)
                    
        # Restore no-data values
        if nodata_mask is not None and nodata is not None:
            if np.issubdtype(target_dtype, np.integer):
                target_nodata = int(nodata) if np.can_cast(nodata, target_dtype) else np.iinfo(target_dtype).min
            else:
                target_nodata = float(nodata)
            converted[nodata_mask] = target_nodata
            
        return converted
        
    def _preserve_nodata(self,
                        data: np.ndarray,
                        nodata: Union[int, float]) -> np.ndarray:
        """Ensure no-data values are preserved after resampling."""
        # This is a simple implementation - override for more sophisticated handling
        return data
        
    def _aggregate_2d(self,
                     data: np.ndarray,
                     scale_y: int,
                     scale_x: int,
                     method: AggregationMethod,
                     nodata: Optional[Union[int, float]]) -> np.ndarray:
        """Aggregate 2D array using specified method."""
        height, width = data.shape
        new_height = height // scale_y
        new_width = width // scale_x
        
        # Reshape for aggregation
        reshaped = data[:new_height*scale_y, :new_width*scale_x]
        reshaped = reshaped.reshape(new_height, scale_y, new_width, scale_x)
        
        # Apply aggregation method
        if method == AggregationMethod.MEAN:
            if nodata is not None:
                masked = np.ma.masked_equal(reshaped, nodata)
                result = np.ma.mean(masked, axis=(1, 3)).filled(nodata)
            else:
                result = np.mean(reshaped, axis=(1, 3))
        elif method == AggregationMethod.SUM:
            if nodata is not None:
                masked = np.ma.masked_equal(reshaped, nodata)
                result = np.ma.sum(masked, axis=(1, 3)).filled(nodata)
            else:
                result = np.sum(reshaped, axis=(1, 3))
        elif method == AggregationMethod.MIN:
            if nodata is not None:
                masked = np.ma.masked_equal(reshaped, nodata)
                result = np.ma.min(masked, axis=(1, 3)).filled(nodata)
            else:
                result = np.min(reshaped, axis=(1, 3))
        elif method == AggregationMethod.MAX:
            if nodata is not None:
                masked = np.ma.masked_equal(reshaped, nodata)
                result = np.ma.max(masked, axis=(1, 3)).filled(nodata)
            else:
                result = np.max(reshaped, axis=(1, 3))
        elif method == AggregationMethod.MEDIAN:
            if nodata is not None:
                masked = np.ma.masked_equal(reshaped, nodata)
                result = np.ma.median(masked, axis=(1, 3)).filled(nodata)
            else:
                result = np.median(reshaped, axis=(1, 3))
        else:
            # Default to mean
            result = np.mean(reshaped, axis=(1, 3))
            
        return result
        
    def _generate_resample_cache_key(self,
                                   data: np.ndarray,
                                   target_shape: Tuple[int, int],
                                   method: ResamplingMethod,
                                   target_dtype: Optional[np.dtype],
                                   nodata: Optional[Union[int, float]]) -> str:
        """Generate cache key for resampling operation."""
        return self.generate_cache_key(
            data.shape,
            data.dtype,
            target_shape,
            method.value,
            target_dtype,
            nodata
        )
