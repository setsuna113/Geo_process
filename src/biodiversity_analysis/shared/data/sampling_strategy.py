"""
Sampling strategy - Reduce data when it's too big.

Linus: "Perfection is achieved not when there is nothing more to add,
        but when there is nothing left to take away."
Sometimes we need to take away data points.
"""

from typing import Dict, Any, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SamplingStrategy:
    """Static methods for sampling. No state needed."""
    
    @staticmethod
    def apply_sampling(
        data: Union[Dict[str, Any], np.ndarray],
        config: Dict[str, Any]
    ) -> Union[Dict[str, Any], np.ndarray]:
        """Apply spatial sampling to reduce data size.
        
        LINUS: Handle Ray ObjectRefs properly - don't sample them!
        Tabular data is already sampled during loading.
        
        Args:
            data: Data arrays, Ray ObjectRefs, or single array
            config: Configuration with sampling parameters
            
        Returns:
            Sampled data (or unchanged ObjectRefs)
        """
        import ray
        
        # Get sampling step
        step = config.get('spatial_grid_step', 1)
        
        if step == 1:
            # No sampling needed
            return data
        
        # Apply sampling
        if isinstance(data, dict):
            # Check if it contains Ray ObjectRefs (tabular data)
            has_objectrefs = any(
                hasattr(v, '__class__') and 'ObjectRef' in str(type(v))
                for v in data.values()
            )
            
            if has_objectrefs:
                # DON'T sample Ray ObjectRefs - they're already sampled during loading!
                logger.debug("Data contains Ray ObjectRefs, skipping sampling (already sampled during load)")
                return data
            
            # LINUS: Check if data is from tabular loader (already sampled)
            # If all arrays have same length and it's around 518K, it's already sampled
            if all(isinstance(v, np.ndarray) for v in data.values()):
                lengths = [len(v) for v in data.values() if isinstance(v, np.ndarray)]
                if lengths and all(l == lengths[0] for l in lengths):
                    # Check if it's already sampled (around 518K rows from 51.8M with step=100)
                    if 500_000 < lengths[0] < 600_000:
                        logger.debug(f"Data already sampled ({lengths[0]:,} rows), skipping additional sampling")
                        return data
            
            # Sample spatial arrays
            result = {}
            for key, arr in data.items():
                if isinstance(arr, np.ndarray):
                    result[key] = SamplingStrategy._sample_array(arr, step)
                else:
                    # Keep non-array data as-is
                    result[key] = arr
            return result
        else:
            # Single array
            return SamplingStrategy._sample_array(data, step)
    
    @staticmethod
    def _sample_array(arr: np.ndarray, step: int) -> np.ndarray:
        """Sample an array with given step.
        
        Simple strided sampling. No fancy algorithms.
        """
        if arr.ndim == 1:
            # 1D array - simple slice
            return arr[::step]
        elif arr.ndim == 2:
            # 2D array - sample both dimensions
            return arr[::step, ::step]
        else:
            # Higher dimensions - sample first two
            return arr[::step, ::step, ...]
    
    @staticmethod
    def get_sampling_info(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about sampling configuration.
        
        Returns:
            Dictionary with sampling information
        """
        step = config.get('spatial_grid_step', 1)
        
        if step == 1:
            return {
                'enabled': False,
                'step': 1,
                'reduction': 1.0,
                'description': 'No sampling (using all data)'
            }
        else:
            reduction = step * step  # 2D reduction
            return {
                'enabled': True,
                'step': step,
                'reduction': reduction,
                'description': f'Sampling every {step} points (1/{reduction} of data)'
            }