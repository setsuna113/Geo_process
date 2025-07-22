# src/resampling/strategies/sum_aggregation.py
"""Sum aggregation resampling strategy."""

import numpy as np
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SumAggregationStrategy:
    """Sum aggregation for count data (e.g., species richness)."""
    
    def resample(self,
                 source: np.ndarray,
                 target_shape: tuple,
                 mapping: np.ndarray,
                 config,
                 progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Apply sum aggregation resampling."""
        result = np.zeros(target_shape, dtype=np.float64)
        
        # For upsampling with count data, we need to distribute counts
        if config.source_resolution < config.target_resolution:
            # Upsampling - distribute counts proportionally
            return self._upsample_counts(source, target_shape, mapping, config, progress_callback)
        
        # Downsampling - sum counts
        unique_targets = np.unique(mapping[:, 0])
        n_targets = len(unique_targets)
        
        for i, target_idx in enumerate(unique_targets):
            if progress_callback and i % 100 == 0:
                progress_callback(int(i / n_targets * 100))
            
            # Get source pixels
            mask = mapping[:, 0] == target_idx
            source_indices = mapping[mask, 1].astype(int)
            
            # Sum values
            values = source.flat[source_indices]
            
            # Handle nodata
            if config.nodata_value is not None:
                values = values[values != config.nodata_value]
            
            if len(values) > 0:
                result.flat[int(target_idx)] = np.sum(values)
        
        if progress_callback:
            progress_callback(100)
        
        return result
    
    def _upsample_counts(self, source, target_shape, mapping, config, progress_callback):
        """Handle upsampling of count data."""
        result = np.zeros(target_shape, dtype=np.float64)
        
        # Count how many target pixels each source pixel maps to
        source_to_target_count = {}
        for target_idx, source_idx in mapping[:, :2].astype(int):
            if source_idx not in source_to_target_count:
                source_to_target_count[source_idx] = []
            source_to_target_count[source_idx].append(target_idx)
        
        # Distribute source counts to target pixels
        for source_idx, target_indices in source_to_target_count.items():
            source_value = source.flat[source_idx]
            
            if config.nodata_value is not None and source_value == config.nodata_value:
                continue
            
            # Distribute equally (or by area if weights provided)
            if mapping.shape[1] > 2:
                # Use area weights
                weights = []
                for t_idx in target_indices:
                    mask = (mapping[:, 0] == t_idx) & (mapping[:, 1] == source_idx)
                    weight = mapping[mask, 2][0] if np.any(mask) else 1.0
                    weights.append(weight)
                
                weights = np.array(weights)
                weights = weights / weights.sum()
            else:
                # Equal distribution
                weights = np.ones(len(target_indices)) / len(target_indices)
            
            # Distribute value
            for t_idx, weight in zip(target_indices, weights):
                result.flat[t_idx] += source_value * weight
        
        return result