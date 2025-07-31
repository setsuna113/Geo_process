# src/domain/resampling/strategies/block_sum_aggregation.py
"""Efficient block-based sum aggregation for downsampling."""

import numpy as np
from typing import Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class BlockSumAggregationStrategy:
    """Efficient block-based sum aggregation for downsampling."""
    
    def is_block_based(self):
        """Indicate this is a block-based strategy that doesn't need pixel mapping."""
        return True
    
    def resample_direct(self,
                       source: np.ndarray,
                       source_bounds: Tuple[float, float, float, float],
                       target_shape: Tuple[int, int],
                       target_bounds: Tuple[float, float, float, float],
                       config,
                       progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Direct resampling without pixel mapping."""
        src_height, src_width = source.shape
        tgt_height, tgt_width = target_shape
        
        # Calculate the downsampling factors
        factor_y = src_height / tgt_height
        factor_x = src_width / tgt_width
        
        logger.info(f"Block sum aggregation: {src_height}x{src_width} -> {tgt_height}x{tgt_width}")
        logger.info(f"Downsampling factors: {factor_y:.2f}x{factor_x:.2f}")
        
        # For near-integer factors, use efficient block processing
        if abs(factor_y - 2.0) < 0.1 and abs(factor_x - 2.0) < 0.1:
            return self._block_sum_2x2(source, target_shape, config, progress_callback)
        else:
            # For non-integer factors, use block approximation
            return self._block_sum_general(source, target_shape, factor_y, factor_x, 
                                         config, progress_callback)
    
    def _block_sum_2x2(self, source, target_shape, config, progress_callback):
        """Efficient 2x2 block sum for factor-2 downsampling."""
        tgt_height, tgt_width = target_shape
        
        # Handle nodata
        if config.nodata_value is not None:
            source_masked = np.where(source == config.nodata_value, 0, source)
        else:
            source_masked = source
        
        # Ensure source dimensions are even (crop if needed)
        crop_height = tgt_height * 2
        crop_width = tgt_width * 2
        source_cropped = source_masked[:crop_height, :crop_width]
        
        # Reshape into 2x2 blocks and sum
        result = source_cropped.reshape(tgt_height, 2, tgt_width, 2).sum(axis=(1, 3))
        
        if progress_callback:
            progress_callback(100)
        
        return result
    
    def _block_sum_general(self, source, target_shape, factor_y, factor_x, 
                          config, progress_callback):
        """General block sum for arbitrary downsampling factors."""
        tgt_height, tgt_width = target_shape
        src_height, src_width = source.shape
        
        result = np.zeros(target_shape, dtype=np.float64)
        
        # Process in chunks for memory efficiency
        chunk_size = min(100, tgt_height)
        
        for chunk_start in range(0, tgt_height, chunk_size):
            chunk_end = min(chunk_start + chunk_size, tgt_height)
            
            for tgt_y in range(chunk_start, chunk_end):
                # Calculate source pixel range for this target row
                src_y_start = int(tgt_y * factor_y)
                src_y_end = min(int((tgt_y + 1) * factor_y), src_height)
                
                for tgt_x in range(tgt_width):
                    # Calculate source pixel range for this target column
                    src_x_start = int(tgt_x * factor_x)
                    src_x_end = min(int((tgt_x + 1) * factor_x), src_width)
                    
                    # Extract block and sum
                    block = source[src_y_start:src_y_end, src_x_start:src_x_end]
                    
                    # Handle nodata
                    if config.nodata_value is not None:
                        valid_mask = block != config.nodata_value
                        if np.any(valid_mask):
                            result[tgt_y, tgt_x] = np.sum(block[valid_mask])
                        else:
                            result[tgt_y, tgt_x] = config.nodata_value
                    else:
                        result[tgt_y, tgt_x] = np.sum(block)
            
            if progress_callback:
                progress = (chunk_end / tgt_height) * 100
                progress_callback(int(progress))
        
        return result
    
    def resample(self, source, target_shape, mapping, config, progress_callback=None):
        """Legacy interface - redirects to resample_direct."""
        # For compatibility - just ignore mapping and use direct approach
        # Extract bounds from config if available
        if hasattr(config, 'bounds') and config.bounds:
            source_bounds = target_bounds = config.bounds
        else:
            # Assume full bounds
            h, w = source.shape
            source_bounds = target_bounds = (0, 0, w, h)
        
        return self.resample_direct(source, source_bounds, target_shape, 
                                   target_bounds, config, progress_callback)