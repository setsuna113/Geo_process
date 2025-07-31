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
        
        # Handle empty arrays
        if src_height == 0 or src_width == 0 or tgt_height == 0 or tgt_width == 0:
            return np.zeros(target_shape, dtype=np.float64)
        
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
        """General block sum for arbitrary downsampling factors - OPTIMIZED.
        
        This implementation reduces redundant operations and improves cache efficiency.
        Key optimizations:
        1. Pre-compute all block boundaries
        2. Process by rows to improve cache locality
        3. Minimize coordinate calculations in inner loops
        """
        tgt_height, tgt_width = target_shape
        src_height, src_width = source.shape
        
        result = np.zeros(target_shape, dtype=np.float64)
        
        # Pre-compute all block boundaries (vectorized)
        y_starts = (np.arange(tgt_height) * factor_y).astype(int)
        y_ends = np.minimum(((np.arange(tgt_height) + 1) * factor_y).astype(int), src_height)
        x_starts = (np.arange(tgt_width) * factor_x).astype(int)
        x_ends = np.minimum(((np.arange(tgt_width) + 1) * factor_x).astype(int), src_width)
        
        # Special optimization for integer or near-integer factors
        if abs(factor_y - round(factor_y)) < 0.01 and abs(factor_x - round(factor_x)) < 0.01:
            # Use more efficient processing for regular grids
            fy = int(round(factor_y))
            fx = int(round(factor_x))
            
            # Process in larger chunks when blocks are regular
            chunk_size = min(200, tgt_height)
        else:
            # Standard chunk size for irregular grids
            chunk_size = min(100, tgt_height)
        
        # Process in row chunks for memory efficiency
        for chunk_start in range(0, tgt_height, chunk_size):
            chunk_end = min(chunk_start + chunk_size, tgt_height)
            
            if config.nodata_value is None:
                # Fast path: no nodata handling needed
                for i in range(chunk_start, chunk_end):
                    y_start = y_starts[i]
                    y_end = y_ends[i]
                    
                    # Extract the rows we need once
                    row_slice = source[y_start:y_end, :]
                    
                    # Vectorized operations on columns where possible
                    for j in range(tgt_width):
                        result[i, j] = np.sum(row_slice[:, x_starts[j]:x_ends[j]])
            else:
                # Slower path with nodata handling
                nodata = config.nodata_value
                for i in range(chunk_start, chunk_end):
                    y_start = y_starts[i]
                    y_end = y_ends[i]
                    row_slice = source[y_start:y_end, :]
                    
                    for j in range(tgt_width):
                        block = row_slice[:, x_starts[j]:x_ends[j]]
                        valid_mask = block != nodata
                        if np.any(valid_mask):
                            result[i, j] = np.sum(block[valid_mask])
                        else:
                            result[i, j] = nodata
            
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