# src/domain/resampling/strategies/average_aggregation.py
"""Average aggregation resampling strategy for SDM predictions."""

import numpy as np
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class AverageAggregationStrategy:
    """Average aggregation for continuous predicted data (e.g., SDM predictions)."""
    
    def resample(self,
                 source: np.ndarray,
                 target_shape: tuple,
                 mapping: np.ndarray,
                 config,
                 progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Apply average aggregation resampling.
        
        For downsampling: Compute area-weighted average
        For upsampling: Use bilinear interpolation
        """
        result = np.zeros(target_shape, dtype=np.float64)
        
        # For upsampling with continuous data, use interpolation
        # Note: smaller resolution value = finer resolution
        # So if source < target, source is finer and we're downsampling
        if config.source_resolution > config.target_resolution:
            # Upsampling - use bilinear interpolation
            logger.info(f"Upsampling detected (source_res={config.source_resolution} > target_res={config.target_resolution}) - using bilinear interpolation for SDM predictions")
            return self._upsample_interpolate(source, target_shape, mapping, config, progress_callback)
        
        # Downsampling - compute area-weighted average
        unique_targets = np.unique(mapping[:, 0])
        n_targets = len(unique_targets)
        
        for i, target_idx in enumerate(unique_targets):
            if progress_callback and i % 100 == 0:
                progress_callback(int(i / n_targets * 100))
            
            # Get source pixels and their weights
            mask = mapping[:, 0] == target_idx
            source_indices = mapping[mask, 1].astype(int)
            
            # Get area weights if available (column 2)
            if mapping.shape[1] > 2:
                weights = mapping[mask, 2]
            else:
                # Equal weights if no area information
                weights = np.ones(len(source_indices))
            
            # Get values
            values = source.flat[source_indices]
            
            # Handle nodata
            if config.nodata_value is not None:
                valid_mask = values != config.nodata_value
                values = values[valid_mask]
                weights = weights[valid_mask]
            
            if len(values) > 0:
                # Compute weighted average
                result.flat[int(target_idx)] = np.average(values, weights=weights)
        
        if progress_callback:
            progress_callback(100)
        
        return result
    
    def _upsample_interpolate(self, source, target_shape, mapping, config, progress_callback):
        """Handle upsampling using bilinear interpolation for continuous data."""
        # Use scipy's zoom for proper bilinear interpolation
        from scipy import ndimage
        
        src_height, src_width = source.shape
        tgt_height, tgt_width = target_shape
        
        # Calculate zoom factors
        zoom_y = tgt_height / src_height
        zoom_x = tgt_width / src_width
        
        # Handle nodata by masking
        if config.nodata_value is not None:
            # Create mask for valid data
            valid_mask = source != config.nodata_value
            
            # Replace nodata with 0 for interpolation
            source_clean = np.where(valid_mask, source, 0)
            
            # Interpolate data and mask separately
            result = ndimage.zoom(source_clean, (zoom_y, zoom_x), order=1)  # order=1 for bilinear
            mask_interp = ndimage.zoom(valid_mask.astype(float), (zoom_y, zoom_x), order=1)
            
            # Restore nodata where mask is near 0
            result = np.where(mask_interp > 0.01, result / np.maximum(mask_interp, 0.01), config.nodata_value)
        else:
            # Simple bilinear interpolation
            result = ndimage.zoom(source, (zoom_y, zoom_x), order=1)
        
        if progress_callback:
            progress_callback(100)
        
        return result.astype(np.float64)