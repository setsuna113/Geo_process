"""Nystrom approximation eigensolver for large-scale MESF.

Implements the three-stage Nystrom approximation:
1. Landmark eigendecomposition (with sparse matrix support)
2. Extension to samples
3. Interpolation to full dataset
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional, Union
import logging

from .spatial_weights import haversine_distance_km

logger = logging.getLogger(__name__)


class NystromEigensolver:
    """Efficient Nystrom approximation for spatial eigendecomposition with sparse matrix support."""
    
    def __init__(
        self,
        n_eigenvectors: int = 10,
        distance_threshold: float = 100.0,
        extension_bandwidth: float = 100.0,
        use_sparse: bool = True,
        kernel_threshold: float = 0.01
    ):
        """Initialize Nystrom eigensolver.
        
        Args:
            n_eigenvectors: Number of eigenvectors to compute
            distance_threshold: Distance threshold in km for spatial weights
            extension_bandwidth: Gaussian bandwidth for landmark-sample extension
            use_sparse: Whether to use sparse matrices for efficiency
            kernel_threshold: Threshold for sparse kernel (values below are set to 0)
        """
        self.n_eigenvectors = n_eigenvectors
        self.distance_threshold = distance_threshold
        self.extension_bandwidth = extension_bandwidth
        self.use_sparse = use_sparse
        self.kernel_threshold = kernel_threshold
    
    def compute_landmark_eigenvectors(
        self,
        coords_landmark: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvectors on landmark points using sparse matrices.
        
        Args:
            coords_landmark: Landmark coordinates (n_landmark, 2)
            
        Returns:
            (eigenvalues, eigenvectors) tuple
        """
        n_landmark = len(coords_landmark)
        logger.info(f"Computing eigendecomposition for {n_landmark:,} landmarks...")
        
        if self.use_sparse:
            # Create sparse spatial weights matrix
            W_sparse = self._create_sparse_spatial_weights(coords_landmark)
            
            # Use sparse eigendecomposition (much faster for sparse matrices)
            # eigsh finds k largest eigenvalues of sparse matrix
            try:
                # Request a few extra to ensure we get enough positive ones
                k = min(self.n_eigenvectors + 5, n_landmark - 1)
                eigenvalues, eigenvectors = eigsh(
                    W_sparse, 
                    k=k,
                    which='LM',  # Largest magnitude
                    return_eigenvectors=True
                )
                
                # Sort by eigenvalue magnitude (descending)
                idx = np.argsort(np.abs(eigenvalues))[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
            except Exception as e:
                logger.warning(f"Sparse eigendecomposition failed: {e}, falling back to dense")
                W_dense = W_sparse.toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(W_dense)
                idx = np.argsort(np.abs(eigenvalues))[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
        else:
            # Dense matrix computation (original approach)
            W = self._create_dense_spatial_weights(coords_landmark)
            eigenvalues, eigenvectors = np.linalg.eigh(W)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        
        # Select top eigenvectors
        selected_eigenvalues = eigenvalues[:self.n_eigenvectors]
        selected_eigenvectors = eigenvectors[:, :self.n_eigenvectors]
        
        # Log statistics
        positive_eigenvalues = eigenvalues[eigenvalues > 0]
        if len(positive_eigenvalues) > 0:
            variance_explained = (
                np.sum(np.abs(selected_eigenvalues)) / 
                np.sum(positive_eigenvalues) * 100
            )
            logger.info(
                f"Selected {self.n_eigenvectors} eigenvectors "
                f"explaining {variance_explained:.1f}% variance"
            )
        
        return selected_eigenvalues, selected_eigenvectors
    
    def _create_sparse_spatial_weights(self, coords: np.ndarray) -> csr_matrix:
        """Create sparse spatial weights matrix without computing full distance matrix.
        
        Uses BallTree for efficient neighbor search - only computes distances for actual neighbors.
        
        Args:
            coords: Coordinates (n, 2) with [lon, lat] in degrees
            
        Returns:
            Sparse spatial weights matrix
        """
        from sklearn.neighbors import BallTree
        
        n = len(coords)
        logger.info(f"Building sparse weights matrix for {n} points using BallTree...")
        
        # Convert to radians for BallTree with haversine metric
        coords_rad = np.radians(coords[:, [1, 0]])  # [lat, lon] order for sklearn
        
        # Build spatial index (BallTree is efficient for neighbor queries)
        tree = BallTree(coords_rad, metric='haversine')
        
        # Convert distance threshold from km to radians
        # Earth radius = 6371 km
        radius_rad = self.distance_threshold / 6371.0
        
        # Find all neighbors within threshold for each point
        # This is much more efficient than computing full distance matrix
        logger.info(f"Finding neighbors within {self.distance_threshold} km...")
        indices_list = tree.query_radius(coords_rad, r=radius_rad, 
                                         return_distance=False,
                                         sort_results=False)
        
        # Build sparse matrix directly (never create dense matrix)
        # Use lil_matrix for efficient construction
        W = lil_matrix((n, n), dtype=np.float32)
        
        # Populate sparse matrix with row-standardized weights
        total_neighbors = 0
        for i in range(n):
            # Get neighbors (excluding self)
            neighbors = indices_list[i]
            neighbors = neighbors[neighbors != i]
            
            if len(neighbors) > 0:
                # Row-standardize: each neighbor gets equal weight
                weight = 1.0 / len(neighbors)
                W[i, neighbors] = weight
                total_neighbors += len(neighbors)
        
        # Convert to CSR format (efficient for matrix operations and eigendecomposition)
        W_sparse = W.tocsr()
        
        # Log sparsity statistics
        nnz = W_sparse.nnz
        sparsity = 1 - (nnz / (n * n))
        avg_neighbors = total_neighbors / n
        logger.info(
            f"Sparse weights matrix created: {n}×{n}, "
            f"{nnz:,} non-zeros ({sparsity*100:.1f}% sparse), "
            f"avg {avg_neighbors:.1f} neighbors/point"
        )
        
        return W_sparse
    
    def _create_dense_spatial_weights(self, coords: np.ndarray) -> np.ndarray:
        """Create dense spatial weights matrix (fallback option).
        
        Args:
            coords: Coordinates (n, 2) with [lon, lat] in degrees
            
        Returns:
            Dense spatial weights matrix
        """
        n = len(coords)
        
        # Compute distances
        distances_km = haversine_distance_km(coords)
        
        # Create binary weights based on threshold
        W = (distances_km <= self.distance_threshold).astype(float)
        np.fill_diagonal(W, 0)  # Remove self-connections
        
        # Row-standardize
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W = W / row_sums[:, np.newaxis]
        
        return W
    
    def extend_to_samples(
        self,
        coords_sample: np.ndarray,
        coords_landmark: np.ndarray,
        eigenvectors_landmark: np.ndarray
    ) -> np.ndarray:
        """Extend eigenvectors from landmarks to samples using Nystrom.
        
        Can use sparse kernel matrix if memory is limited.
        
        Args:
            coords_sample: Sample coordinates (n_sample, 2)
            coords_landmark: Landmark coordinates (n_landmark, 2)
            eigenvectors_landmark: Landmark eigenvectors (n_landmark, n_eigenvectors)
            
        Returns:
            Sample eigenvectors (n_sample, n_eigenvectors)
        """
        n_sample = len(coords_sample)
        n_landmark = len(coords_landmark)
        
        logger.info(
            f"Extending {self.n_eigenvectors} eigenvectors from "
            f"{n_landmark:,} landmarks to {n_sample:,} samples..."
        )
        
        # Compute distances
        distances_km = haversine_distance_km(coords_sample, coords_landmark)
        
        # Compute Gaussian kernel
        kernel = np.exp(-distances_km / self.extension_bandwidth)
        
        if self.use_sparse and self.kernel_threshold > 0:
            # Threshold small values for sparsity
            logger.info(f"Creating sparse kernel with threshold {self.kernel_threshold}")
            
            # Count how many values will be kept
            mask = kernel > self.kernel_threshold
            nnz = np.sum(mask)
            sparsity = 1 - (nnz / (n_sample * n_landmark))
            
            logger.info(f"Kernel sparsity: {sparsity*100:.1f}% zeros, {nnz:,} non-zeros")
            
            # Apply threshold
            kernel[~mask] = 0
            
            # Convert to sparse and normalize rows
            kernel_sparse = csr_matrix(kernel)
            row_sums = np.array(kernel_sparse.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1
            
            # Normalize using sparse operations
            inv_row_sums = 1.0 / row_sums
            kernel_sparse = kernel_sparse.multiply(inv_row_sums[:, np.newaxis])
            
            # Project eigenvectors (sparse @ dense = dense)
            eigenvectors_sample = kernel_sparse @ eigenvectors_landmark
            
        else:
            # Dense kernel (original approach)
            # Normalize rows
            kernel = kernel / kernel.sum(axis=1, keepdims=True)
            
            # Project eigenvectors
            eigenvectors_sample = kernel @ eigenvectors_landmark
        
        return eigenvectors_sample
    
    @staticmethod
    def _orthonormalize(vectors: np.ndarray) -> np.ndarray:
        """Orthonormalize vectors using QR decomposition.
        
        Args:
            vectors: Input vectors (n, m)
            
        Returns:
            Orthonormalized vectors (n, m)
        """
        Q, _ = np.linalg.qr(vectors)
        return Q