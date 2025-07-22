# src/resampling/strategies/area_weighted.py
"""Area-weighted resampling strategy."""

import numpy as np
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class AreaWeightedStrategy:
    """Area-weighted average resampling for continuous data."""
    
    def resample(self,
                 source: np.ndarray,
                 target_shape: tuple,
                 mapping: np.ndarray,
                 config,
                 progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Apply area-weighted resampling."""
        result = np.zeros(target_shape, dtype=np.float64)
        weights = np.zeros(target_shape, dtype=np.float64)
        
        # Group by target pixel
        unique_targets = np.unique(mapping[:, 0])
        n_targets = len(unique_targets)
        
        for i, target_idx in enumerate(unique_targets):
            if progress_callback and i % 100 == 0:
                progress_callback(int(i / n_targets * 100))
            
            # Get source pixels for this target
            mask = mapping[:, 0] == target_idx
            source_indices = mapping[mask, 1].astype(int)
            pixel_weights = mapping[mask, 2] if mapping.shape[1] > 2 else np.ones(len(source_indices))
            
            # Extract values
            values = source.flat[source_indices]
            
            # Handle nodata
            if config.nodata_value is not None:
                valid_mask = values != config.nodata_value
                values = values[valid_mask]
                pixel_weights = pixel_weights[valid_mask]
            
            if len(values) > 0:
                # Weighted average
                weighted_sum = np.sum(values * pixel_weights)
                weight_sum = np.sum(pixel_weights)
                
                if weight_sum > 0:
                    result.flat[int(target_idx)] = weighted_sum / weight_sum
                    weights.flat[int(target_idx)] = weight_sum
        
        if progress_callback:
            progress_callback(100)
        
        # Set nodata where no valid data
        if config.nodata_value is not None:
            result[weights == 0] = config.nodata_value
        
        return result