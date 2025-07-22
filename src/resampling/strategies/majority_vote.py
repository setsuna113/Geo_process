# src/resampling/strategies/majority_vote.py
"""Majority vote resampling strategy."""

import numpy as np
from typing import Optional, Callable
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class MajorityVoteStrategy:
    """Majority vote resampling for categorical data."""
    
    def resample(self,
                 source: np.ndarray,
                 target_shape: tuple,
                 mapping: np.ndarray,
                 config,
                 progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Apply majority vote resampling."""
        result = np.zeros(target_shape, dtype=source.dtype)
        
        unique_targets = np.unique(mapping[:, 0])
        n_targets = len(unique_targets)
        
        for i, target_idx in enumerate(unique_targets):
            if progress_callback and i % 100 == 0:
                progress_callback(int(i / n_targets * 100))
            
            # Get source pixels
            mask = mapping[:, 0] == target_idx
            source_indices = mapping[mask, 1].astype(int)
            
            # Get values
            values = source.flat[source_indices]
            
            # Handle nodata
            if config.nodata_value is not None:
                values = values[values != config.nodata_value]
            
            if len(values) > 0:
                # Find most common value
                if mapping.shape[1] > 2:
                    # Weighted voting
                    weights = mapping[mask, 2]
                    weighted_counts = {}
                    
                    for val, weight in zip(values, weights):
                        if val in weighted_counts:
                            weighted_counts[val] += weight
                        else:
                            weighted_counts[val] = weight
                    
                    # Get value with highest weight
                    majority_value = max(weighted_counts.keys(), key=lambda k: weighted_counts[k])
                else:
                    # Simple majority
                    counter = Counter(values)
                    majority_value = counter.most_common(1)[0][0]
                
                result.flat[int(target_idx)] = majority_value
            elif config.nodata_value is not None:
                result.flat[int(target_idx)] = config.nodata_value
        
        if progress_callback:
            progress_callback(100)
        
        return result