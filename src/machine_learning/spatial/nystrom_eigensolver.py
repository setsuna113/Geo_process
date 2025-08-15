"""
Nystrom eigensolver - JAX implementation.

No fallbacks. If you don't have JAX, install it.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Configure JAX
import jax
# Use float32 for speed (2x faster than float64)
jax.config.update("jax_enable_x64", False)


class NystromEigensolver:
    """Nystrom approximation for spatial eigendecomposition."""
    
    def __init__(
        self,
        n_eigenvectors: int = 10,
        distance_threshold: float = 100.0,
        extension_bandwidth: float = 100.0,
        use_sparse: bool = True,  # Ignored - kept for compatibility
        kernel_threshold: float = 0.01  # Ignored - no special cases
    ):
        self.n_eigenvectors = n_eigenvectors
        self.distance_threshold = distance_threshold
        self.extension_bandwidth = extension_bandwidth
    
    @staticmethod
    def _haversine_distances(coords1: jnp.ndarray, coords2: jnp.ndarray) -> jnp.ndarray:
        """Compute haversine distances."""
        c1 = jnp.radians(coords1)
        c2 = jnp.radians(coords2)
        
        lat1 = c1[:, 1:2]
        lon1 = c1[:, 0:1]
        lat2 = c2[:, 1:2].T
        lon2 = c2[:, 0:1].T
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = jnp.sin(dlat/2)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon/2)**2
        c = 2 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0, 1)))
        
        return c * 6371.0
    
    def _compute_weights_matrix(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Compute spatial weights matrix."""
        distances = self._haversine_distances(coords, coords)
        W = (distances <= self.distance_threshold).astype(jnp.float32)
        W = W.at[jnp.diag_indices(len(W))].set(0)
        
        # Row standardize
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = jnp.where(row_sums == 0, 1, row_sums)
        
        return W / row_sums
    
    def compute_landmark_eigenvectors(
        self, 
        coords: np.ndarray,
        return_eigenvalues: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvectors for landmarks."""
        n = len(coords)
        logger.info(f"Computing eigendecomposition for {n:,} landmarks...")
        
        # Convert to JAX
        coords_jax = jnp.array(coords)
        
        # Compute weights
        W = self._compute_weights_matrix(coords_jax)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(W)
        
        # Select largest (skip trivial)
        idx = jnp.argsort(eigenvalues)[::-1][1:self.n_eigenvectors+1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Moran's I
        moran_i = jnp.diag(eigenvectors.T @ W @ eigenvectors)
        
        logger.info(
            f"Selected {self.n_eigenvectors} eigenvectors with "
            f"Moran's I range: [{moran_i.min():.3f}, {moran_i.max():.3f}]"
        )
        
        # Convert back to NumPy
        if return_eigenvalues:
            return np.array(eigenvalues), np.array(eigenvectors)
        return np.array(eigenvectors)
    
    def _extend_eigenvectors_jax(
        self,
        coords_sample: jnp.ndarray,
        coords_landmark: jnp.ndarray,
        eigenvectors_landmark: jnp.ndarray
    ) -> jnp.ndarray:
        """Extend eigenvectors using Nystrom."""
        # Compute distances
        distances = self._haversine_distances(coords_sample, coords_landmark)
        
        # Gaussian kernel
        kernel = jnp.exp(-distances / self.extension_bandwidth)
        
        # Row normalize
        kernel = kernel / kernel.sum(axis=1, keepdims=True)
        
        # Project
        return kernel @ eigenvectors_landmark
    
    def extend_to_samples(
        self,
        coords_sample: np.ndarray,
        coords_landmark: np.ndarray,
        eigenvectors_landmark: np.ndarray
    ) -> np.ndarray:
        """Extend eigenvectors from landmarks to samples."""
        n_sample = len(coords_sample)
        n_landmark = len(coords_landmark)
        
        logger.info(
            f"Extending {self.n_eigenvectors} eigenvectors from "
            f"{n_landmark:,} landmarks to {n_sample:,} samples..."
        )
        
        # Check memory
        memory_gb = (n_sample * n_landmark * 8) / (1024**3)
        
        if memory_gb > 50:  # Process in batches if too large
            return self._extend_in_batches(
                coords_sample, coords_landmark, eigenvectors_landmark
            )
        
        # Small enough - do it all at once
        coords_sample_jax = jnp.array(coords_sample)
        coords_landmark_jax = jnp.array(coords_landmark)
        eigenvectors_jax = jnp.array(eigenvectors_landmark)
        
        result = self._extend_eigenvectors_jax(
            coords_sample_jax, coords_landmark_jax, eigenvectors_jax
        )
        
        return np.array(result)
    
    def _extend_in_batches(
        self,
        coords_sample: np.ndarray,
        coords_landmark: np.ndarray,
        eigenvectors_landmark: np.ndarray,
        batch_size: int = 50000
    ) -> np.ndarray:
        """Process in batches when memory constrained."""
        n_sample = len(coords_sample)
        n_eigenvectors = eigenvectors_landmark.shape[1]
        result = np.empty((n_sample, n_eigenvectors), dtype=np.float64)
        
        coords_landmark_jax = jnp.array(coords_landmark)
        eigenvectors_jax = jnp.array(eigenvectors_landmark)
        
        for i in range(0, n_sample, batch_size):
            end = min(i + batch_size, n_sample)
            batch = jnp.array(coords_sample[i:end])
            
            result[i:end] = np.array(
                self._extend_eigenvectors_jax(batch, coords_landmark_jax, eigenvectors_jax)
            )
            
            if i % (batch_size * 10) == 0 and i > 0:
                logger.info(f"Processed {i:,}/{n_sample:,} samples")
        
        return result
    
    @staticmethod
    def _orthonormalize(vectors: np.ndarray) -> np.ndarray:
        """Orthonormalize vectors using QR decomposition."""
        Q, _ = np.linalg.qr(vectors)
        return Q


# Import for compatibility with old code that expects this
def haversine_distance_km(coords1: np.ndarray, coords2: np.ndarray = None) -> np.ndarray:
    """Compute haversine distances using JAX."""
    if coords2 is None:
        coords2 = coords1
    
    c1 = jnp.array(coords1)
    c2 = jnp.array(coords2)
    
    # Use the static method
    distances = NystromEigensolver._haversine_distances(c1, c2)
    
    return np.array(distances)