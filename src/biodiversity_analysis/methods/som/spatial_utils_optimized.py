"""Optimized spatial utilities for large-scale biodiversity data.

This module provides memory-efficient implementations for computing
spatial statistics on millions of points.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def approximate_morans_i_large(y: np.ndarray, coordinates: np.ndarray, 
                               k_neighbors: int = 30,
                               sample_size: Optional[int] = None) -> float:
    """Compute Moran's I using k-nearest neighbors approximation.
    
    This avoids creating the full n×n distance matrix by only considering
    k nearest neighbors for each point. Research shows k=30 provides
    excellent approximation to the full calculation.
    
    Args:
        y: Variable values (e.g., cluster assignments)
        coordinates: Geographic coordinates [lon, lat]
        k_neighbors: Number of neighbors to consider (default: 30)
        sample_size: If provided, use spatial sampling
        
    Returns:
        Moran's I statistic
    """
    n = len(y)
    
    # Apply sampling if needed
    if sample_size and n > sample_size:
        indices = np.random.choice(n, size=sample_size, replace=False)
        y = y[indices]
        coordinates = coordinates[indices]
        n = sample_size
        logger.info(f"Using {sample_size} samples for Moran's I calculation")
    
    # Standardize y
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    # Build KD-tree for efficient neighbor search
    tree = cKDTree(coordinates)
    
    # Compute local contributions to Moran's I
    sum_wy = 0.0
    sum_w = 0.0
    
    for i in range(n):
        # Find k nearest neighbors
        distances, indices = tree.query(coordinates[i], k=min(k_neighbors + 1, n))
        
        # Skip the first one (self)
        neighbor_indices = indices[1:]
        neighbor_distances = distances[1:]
        
        # Compute inverse distance weights
        weights = np.where(neighbor_distances > 0, 1.0 / neighbor_distances, 0)
        weights_sum = np.sum(weights)
        
        if weights_sum > 0:
            # Normalize weights to sum to 1
            weights = weights / weights_sum
            
            # Contribution to Moran's I
            sum_wy += y_centered[i] * np.sum(weights * y_centered[neighbor_indices])
            sum_w += weights_sum
    
    # Final Moran's I calculation
    variance = np.var(y)
    if variance > 0 and sum_w > 0:
        morans_i = (n / sum_w) * (sum_wy / (n * variance))
    else:
        morans_i = 0.0
    
    return morans_i


def stratified_spatial_sample(coordinates: np.ndarray, 
                            target_size: int,
                            grid_dims: Tuple[int, int] = (50, 50)) -> np.ndarray:
    """Perform stratified spatial sampling.
    
    Args:
        coordinates: Full set of coordinates [lon, lat]
        target_size: Target number of samples
        grid_dims: Grid dimensions for stratification
        
    Returns:
        Array of sampled indices
    """
    n_samples = len(coordinates)
    
    if n_samples <= target_size:
        return np.arange(n_samples)
    
    # Create spatial grid
    lon_min, lon_max = coordinates[:, 0].min(), coordinates[:, 0].max()
    lat_min, lat_max = coordinates[:, 1].min(), coordinates[:, 1].max()
    
    lon_bins = np.linspace(lon_min, lon_max, grid_dims[0] + 1)
    lat_bins = np.linspace(lat_min, lat_max, grid_dims[1] + 1)
    
    # Assign points to grid cells
    lon_idx = np.digitize(coordinates[:, 0], lon_bins) - 1
    lat_idx = np.digitize(coordinates[:, 1], lat_bins) - 1
    
    # Clip to valid range
    lon_idx = np.clip(lon_idx, 0, grid_dims[0] - 1)
    lat_idx = np.clip(lat_idx, 0, grid_dims[1] - 1)
    
    cell_idx = lon_idx * grid_dims[1] + lat_idx
    
    # Sample from each cell
    sample_indices = []
    unique_cells, cell_counts = np.unique(cell_idx, return_counts=True)
    
    # Proportional allocation
    cell_samples = np.round(target_size * cell_counts / n_samples).astype(int)
    cell_samples = np.maximum(cell_samples, 1)  # At least 1 sample per cell
    
    # Adjust to match target size
    while cell_samples.sum() > target_size:
        largest_idx = np.argmax(cell_samples)
        if cell_samples[largest_idx] > 1:
            cell_samples[largest_idx] -= 1
    
    while cell_samples.sum() < target_size:
        smallest_idx = np.argmin(cell_samples / cell_counts)
        cell_samples[smallest_idx] += 1
    
    # Perform sampling
    for cell, n_cell_samples in zip(unique_cells, cell_samples):
        cell_points = np.where(cell_idx == cell)[0]
        if len(cell_points) > 0:
            n_to_sample = min(len(cell_points), n_cell_samples)
            samples = np.random.choice(cell_points, size=n_to_sample, replace=False)
            sample_indices.extend(samples)
    
    return np.array(sample_indices)


def compute_spatial_metrics_large(cluster_assignments: np.ndarray,
                                coordinates: np.ndarray,
                                max_samples: int = 100000) -> Dict[str, float]:
    """Compute spatial metrics for large datasets efficiently.
    
    Args:
        cluster_assignments: Cluster ID for each sample
        coordinates: Geographic coordinates [lon, lat]
        max_samples: Maximum samples to use for expensive calculations
        
    Returns:
        Dictionary of spatial metrics
    """
    n_total = len(cluster_assignments)
    
    # Use stratified sampling for large datasets
    if n_total > max_samples:
        logger.info(f"Using stratified sampling: {max_samples} from {n_total} samples")
        sample_indices = stratified_spatial_sample(coordinates, max_samples)
        cluster_assignments = cluster_assignments[sample_indices]
        coordinates = coordinates[sample_indices]
    
    metrics = {}
    
    # 1. Moran's I with k-NN approximation
    metrics['morans_i'] = approximate_morans_i_large(
        cluster_assignments.astype(float), 
        coordinates,
        k_neighbors=30
    )
    
    # 2. Within-cluster cohesion (using centroids for efficiency)
    unique_clusters = np.unique(cluster_assignments)
    cluster_sizes = []
    cluster_spreads = []
    
    for cluster_id in unique_clusters:
        mask = cluster_assignments == cluster_id
        cluster_size = mask.sum()
        cluster_sizes.append(cluster_size)
        
        if cluster_size > 1:
            cluster_coords = coordinates[mask]
            # Use standard deviation as measure of spread
            lon_std = np.std(cluster_coords[:, 0])
            lat_std = np.std(cluster_coords[:, 1])
            # Convert to approximate km (1 degree ≈ 111 km)
            spread_km = np.sqrt(lon_std**2 + lat_std**2) * 111
            cluster_spreads.append(spread_km)
    
    metrics['mean_cluster_spread_km'] = np.mean(cluster_spreads) if cluster_spreads else 0.0
    metrics['median_cluster_size'] = np.median(cluster_sizes)
    
    # 3. Cluster separation using centroids
    if len(unique_clusters) > 1:
        centroids = []
        for cluster_id in unique_clusters:
            mask = cluster_assignments == cluster_id
            if mask.sum() > 0:
                centroids.append(coordinates[mask].mean(axis=0))
        
        centroids = np.array(centroids)
        
        # Average nearest-neighbor distance between centroids
        tree = cKDTree(centroids)
        nn_distances = []
        for i, centroid in enumerate(centroids):
            distances, _ = tree.query(centroid, k=min(2, len(centroids)))
            if len(distances) > 1:
                nn_distances.append(distances[1] * 111)  # Convert to km
        
        metrics['mean_centroid_separation_km'] = np.mean(nn_distances) if nn_distances else 0.0
    else:
        metrics['mean_centroid_separation_km'] = 0.0
    
    # 4. Spatial contiguity ratio
    if metrics['mean_cluster_spread_km'] > 0:
        metrics['spatial_contiguity_ratio'] = (
            metrics['mean_centroid_separation_km'] / metrics['mean_cluster_spread_km']
        )
    else:
        metrics['spatial_contiguity_ratio'] = 1.0
    
    # 5. Add sampling info
    metrics['n_samples_used'] = len(cluster_assignments)
    metrics['n_total_samples'] = n_total
    metrics['sampling_fraction'] = len(cluster_assignments) / n_total
    
    return metrics