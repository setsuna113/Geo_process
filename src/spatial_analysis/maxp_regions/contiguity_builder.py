# src/spatial_analysis/maxp_regions/contiguity_builder.py
"""Build spatial contiguity structures for raster data."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import libpysal
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

class ContiguityBuilder:
    """Build various types of spatial contiguity for grid data."""
    
    @staticmethod
    def build_rook_contiguity(height: int, width: int) -> libpysal.weights.W:
        """
        Build rook contiguity (4-connected neighbors).
        
        Args:
            height: Grid height
            width: Grid width
            
        Returns:
            Spatial weights object
        """
        return libpysal.weights.lat2W(height, width, rook=True)
    
    @staticmethod
    def build_queen_contiguity(height: int, width: int) -> libpysal.weights.W:
        """
        Build queen contiguity (8-connected neighbors).
        
        Args:
            height: Grid height
            width: Grid width
            
        Returns:
            Spatial weights object
        """
        return libpysal.weights.lat2W(height, width, rook=False)
    
    @staticmethod
    def build_distance_band(coordinates: np.ndarray, 
                          threshold: float) -> libpysal.weights.W:
        """
        Build distance band contiguity.
        
        Args:
            coordinates: Array of (x, y) coordinates
            threshold: Distance threshold
            
        Returns:
            Spatial weights object
        """
        return libpysal.weights.DistanceBand.from_array(
            coordinates, threshold=threshold
        )
    
    @staticmethod
    def build_knn_contiguity(coordinates: np.ndarray, 
                           k: int) -> libpysal.weights.W:
        """
        Build k-nearest neighbors contiguity.
        
        Args:
            coordinates: Array of (x, y) coordinates
            k: Number of nearest neighbors
            
        Returns:
            Spatial weights object
        """
        return libpysal.weights.KNN.from_array(coordinates, k=k)
    
    @staticmethod
    def build_custom_contiguity(height: int, width: int,
                              kernel: np.ndarray) -> libpysal.weights.W:
        """
        Build custom contiguity based on a kernel.
        
        Args:
            height: Grid height
            width: Grid width
            kernel: Boolean kernel defining neighborhood
            
        Returns:
            Spatial weights object
        """
        n = height * width
        neighbors = {}
        weights = {}
        
        # Kernel center
        ky, kx = kernel.shape
        cy, cx = ky // 2, kx // 2
        
        # Build neighbor lists
        for i in range(height):
            for j in range(width):
                idx = i * width + j
                neighbors[idx] = []
                weights[idx] = []
                
                # Apply kernel
                for ki in range(ky):
                    for kj in range(kx):
                        if kernel[ki, kj] and (ki != cy or kj != cx):
                            # Neighbor position
                            ni = i + ki - cy
                            nj = j + kj - cx
                            
                            # Check bounds
                            if 0 <= ni < height and 0 <= nj < width:
                                nidx = ni * width + nj
                                neighbors[idx].append(nidx)
                                weights[idx].append(1.0)
        
        return libpysal.weights.W(neighbors, weights)
    
    @staticmethod
    def add_higher_order_neighbors(w: libpysal.weights.W, 
                                 order: int) -> libpysal.weights.W:
        """
        Add higher order neighbors to existing weights.
        
        Args:
            w: Base spatial weights
            order: Maximum order of neighbors to include
            
        Returns:
            Extended spatial weights
        """
        return libpysal.weights.higher_order(w, order)
    
    @staticmethod
    def create_block_weights(height: int, width: int,
                           block_size: int) -> libpysal.weights.W:
        """
        Create weights for block-based analysis.
        
        Args:
            height: Grid height
            width: Grid width
            block_size: Size of blocks
            
        Returns:
            Block-based spatial weights
        """
        # Calculate number of blocks
        n_blocks_y = (height + block_size - 1) // block_size
        n_blocks_x = (width + block_size - 1) // block_size
        
        # Map pixels to blocks
        pixel_to_block = {}
        for i in range(height):
            for j in range(width):
                pixel_idx = i * width + j
                block_y = i // block_size
                block_x = j // block_size
                block_idx = block_y * n_blocks_x + block_x
                pixel_to_block[pixel_idx] = block_idx
        
        # Build weights based on block membership
        neighbors = {}
        weights = {}
        
        for pixel_idx, block_idx in pixel_to_block.items():
            neighbors[pixel_idx] = []
            weights[pixel_idx] = []
            
            # Add all pixels in same block as neighbors
            for other_pixel, other_block in pixel_to_block.items():
                if other_block == block_idx and other_pixel != pixel_idx:
                    neighbors[pixel_idx].append(other_pixel)
                    weights[pixel_idx].append(1.0)
        
        return libpysal.weights.W(neighbors, weights)