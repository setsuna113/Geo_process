"""Optimization strategies for sparse biodiversity data."""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from .kmeans_config import KMeansConfig
from .adaptive_distance import AdaptivePartialDistance

logger = logging.getLogger(__name__)


class SparseDataOptimizer:
    """Optimization strategies for k-means with sparse data.
    
    Handles:
    - Prefiltering of empty/near-empty grids
    - Sparse distance matrix computation
    - Hierarchical clustering by data quality
    - Parallel processing
    """
    
    def __init__(self, config: KMeansConfig):
        """Initialize optimizer.
        
        Args:
            config: K-means configuration
        """
        self.config = config
    
    def prefilter_data(self, data: np.ndarray, 
                      min_features: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Remove grids with insufficient data.
        
        Args:
            data: Feature data (n_samples, n_features)
            min_features: Minimum valid features required
            
        Returns:
            filtered_data: Data with empty grids removed
            valid_mask: Boolean mask of kept samples
        """
        # Count valid features per sample
        valid_counts = (~np.isnan(data)).sum(axis=1)
        
        # Create mask for samples with enough features
        valid_mask = valid_counts >= min_features
        
        n_original = len(data)
        n_kept = valid_mask.sum()
        n_removed = n_original - n_kept
        
        logger.info(f"Prefiltering: keeping {n_kept}/{n_original} grids "
                   f"({100*n_kept/n_original:.1f}%), removed {n_removed} empty grids")
        
        # Log statistics about removed grids
        if n_removed > 0:
            removed_counts = valid_counts[~valid_mask]
            logger.debug(f"Removed grids had {removed_counts.mean():.2f} avg valid features")
        
        return data[valid_mask], valid_mask
    
    def stratify_by_quality(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Stratify data by quality (number of valid features).
        
        Args:
            data: Feature data
            
        Returns:
            strata: Dictionary with 'high', 'medium', 'low' quality indices
        """
        valid_counts = (~np.isnan(data)).sum(axis=1)
        
        # Define strata
        high_mask = valid_counts >= 3
        medium_mask = (valid_counts == 2)
        low_mask = (valid_counts == 1)
        
        strata = {
            'high': np.where(high_mask)[0],
            'medium': np.where(medium_mask)[0],
            'low': np.where(low_mask)[0]
        }
        
        logger.info(f"Data stratification: high={len(strata['high'])}, "
                   f"medium={len(strata['medium'])}, low={len(strata['low'])}")
        
        return strata
    
    def compute_sparse_distances(self, data: np.ndarray,
                               distance_calculator: AdaptivePartialDistance,
                               coordinates: Optional[np.ndarray] = None,
                               threshold: float = np.inf) -> csr_matrix:
        """Compute sparse distance matrix.
        
        Only computes distances where samples have overlapping features.
        
        Args:
            data: Feature data
            distance_calculator: Distance calculation object
            coordinates: Geographic coordinates
            threshold: Maximum distance to store (for sparsity)
            
        Returns:
            distances: Sparse distance matrix
        """
        n_samples = len(data)
        
        # Pre-compute which pairs have overlap
        logger.info("Computing sample overlap matrix...")
        has_overlap = self._compute_overlap_matrix(data)
        
        # Compute distances only where overlap exists
        logger.info("Computing sparse distances...")
        
        if self.config.n_jobs == 1:
            distances = self._compute_distances_sequential(
                data, distance_calculator, coordinates, has_overlap, threshold
            )
        else:
            distances = self._compute_distances_parallel(
                data, distance_calculator, coordinates, has_overlap, threshold
            )
        
        # Convert to sparse matrix
        row_indices = []
        col_indices = []
        values = []
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if distances[i, j] < threshold:
                    row_indices.extend([i, j])
                    col_indices.extend([j, i])
                    values.extend([distances[i, j], distances[i, j]])
        
        sparse_distances = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )
        
        logger.info(f"Sparse distance matrix: {sparse_distances.nnz} non-zero entries "
                   f"out of {n_samples**2} ({100*sparse_distances.nnz/n_samples**2:.2f}%)")
        
        return sparse_distances
    
    def _compute_overlap_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute which sample pairs have sufficient overlap.
        
        Args:
            data: Feature data
            
        Returns:
            overlap: Boolean matrix of sample pairs with overlap
        """
        n_samples = len(data)
        overlap = np.zeros((n_samples, n_samples), dtype=bool)
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Check if samples have any common non-missing features
                valid_both = ~(np.isnan(data[i]) | np.isnan(data[j]))
                if valid_both.sum() >= 1:  # At least 1 common feature
                    overlap[i, j] = overlap[j, i] = True
        
        return overlap
    
    def _compute_distances_sequential(self, data: np.ndarray,
                                    distance_calculator: AdaptivePartialDistance,
                                    coordinates: Optional[np.ndarray],
                                    has_overlap: np.ndarray,
                                    threshold: float) -> np.ndarray:
        """Compute distances sequentially."""
        n_samples = len(data)
        distances = np.full((n_samples, n_samples), np.inf)
        np.fill_diagonal(distances, 0)
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if has_overlap[i, j]:
                    coord_i = coordinates[i] if coordinates is not None else None
                    coord_j = coordinates[j] if coordinates is not None else None
                    
                    d = distance_calculator.calculate_distance(
                        data[i], data[j], coord_i, coord_j
                    )
                    
                    if d < threshold:
                        distances[i, j] = distances[j, i] = d
        
        return distances
    
    def _compute_distances_parallel(self, data: np.ndarray,
                                  distance_calculator: AdaptivePartialDistance,
                                  coordinates: Optional[np.ndarray],
                                  has_overlap: np.ndarray,
                                  threshold: float) -> np.ndarray:
        """Compute distances in parallel."""
        n_samples = len(data)
        
        # Create chunks for parallel processing
        def compute_chunk(i_start, i_end):
            chunk_distances = []
            for i in range(i_start, i_end):
                for j in range(i+1, n_samples):
                    if has_overlap[i, j]:
                        coord_i = coordinates[i] if coordinates is not None else None
                        coord_j = coordinates[j] if coordinates is not None else None
                        
                        d = distance_calculator.calculate_distance(
                            data[i], data[j], coord_i, coord_j
                        )
                        
                        if d < threshold:
                            chunk_distances.append((i, j, d))
            return chunk_distances
        
        # Parallel computation
        n_chunks = min(self.config.n_jobs if self.config.n_jobs > 0 else 1, n_samples)
        chunk_size = n_samples // n_chunks + 1
        
        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(compute_chunk)(i, min(i + chunk_size, n_samples))
            for i in range(0, n_samples, chunk_size)
        )
        
        # Combine results
        distances = np.full((n_samples, n_samples), np.inf)
        np.fill_diagonal(distances, 0)
        
        for chunk_result in results:
            for i, j, d in chunk_result:
                distances[i, j] = distances[j, i] = d
        
        return distances
    
    def spatial_aggregation(self, data: np.ndarray, 
                           coordinates: np.ndarray,
                           radius_km: float = 50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate nearby sparse grids to improve data quality.
        
        Args:
            data: Feature data
            coordinates: Geographic coordinates
            radius_km: Aggregation radius
            
        Returns:
            aggregated_data: Aggregated features
            aggregated_coords: Aggregated coordinates
            aggregation_map: Mapping from original to aggregated indices
        """
        from ..som.spatial_utils import haversine_distance
        
        n_samples = len(data)
        processed = np.zeros(n_samples, dtype=bool)
        
        aggregated_data = []
        aggregated_coords = []
        aggregation_map = np.full(n_samples, -1, dtype=int)
        
        for i in range(n_samples):
            if processed[i]:
                continue
            
            # Find nearby grids
            distances = haversine_distance(coordinates[i], coordinates)
            nearby = (distances < radius_km) & (~processed)
            
            if nearby.sum() >= 3:  # Aggregate if enough neighbors
                # Aggregate features (nanmean)
                agg_features = np.nanmean(data[nearby], axis=0)
                
                # Aggregate coordinates (mean)
                agg_coord = np.mean(coordinates[nearby], axis=0)
                
                # Store aggregated result
                agg_idx = len(aggregated_data)
                aggregated_data.append(agg_features)
                aggregated_coords.append(agg_coord)
                
                # Update mapping
                aggregation_map[nearby] = agg_idx
                processed[nearby] = True
            else:
                # Keep as individual grid
                agg_idx = len(aggregated_data)
                aggregated_data.append(data[i])
                aggregated_coords.append(coordinates[i])
                aggregation_map[i] = agg_idx
                processed[i] = True
        
        aggregated_data = np.array(aggregated_data)
        aggregated_coords = np.array(aggregated_coords)
        
        logger.info(f"Spatial aggregation: {n_samples} grids → {len(aggregated_data)} aggregated units")
        
        return aggregated_data, aggregated_coords, aggregation_map