"""Spatial utilities for GeoSOM implementation.

Handles geographic distance calculations, spatial weighting, and
spatial block creation for validation.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.spatial import cKDTree
from numba import jit


@jit(nopython=True)
def haversine_distance_numba(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Numba-optimized haversine distance calculation.
    
    Args:
        lon1, lat1: First coordinate in degrees
        lon2, lat2: Second coordinate in degrees
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in km
    r = 6371.0
    return c * r


class SpatialWeighting:
    """Handle spatial weighting for GeoSOM distance calculations."""
    
    def __init__(self, spatial_weight: float = 0.3, max_distance: float = 20000.0):
        """Initialize spatial weighting.
        
        Args:
            spatial_weight: Weight for spatial component (0-1)
            max_distance: Maximum distance for normalization (km)
        """
        self.spatial_weight = spatial_weight
        self.feature_weight = 1.0 - spatial_weight
        self.max_distance = max_distance
    
    def combine_distances(self, feature_dist: float, spatial_dist: float) -> float:
        """Combine feature and spatial distances.
        
        Args:
            feature_dist: Feature-based distance (0-1)
            spatial_dist: Spatial distance in km
            
        Returns:
            Combined weighted distance
        """
        # Normalize spatial distance
        spatial_dist_norm = min(spatial_dist / self.max_distance, 1.0)
        
        # Weighted combination
        return self.feature_weight * feature_dist + self.spatial_weight * spatial_dist_norm
    
    def normalize_geographic_distance(self, dist_km: float) -> float:
        """Normalize geographic distance to 0-1 range."""
        return min(dist_km / self.max_distance, 1.0)


class SpatialBlockGenerator:
    """Generate spatial blocks for cross-validation."""
    
    def __init__(self, block_size_km: float = 750.0, random_state: Optional[int] = None):
        """Initialize block generator.
        
        Args:
            block_size_km: Size of spatial blocks in kilometers
            random_state: Random seed for reproducibility
        """
        self.block_size_km = block_size_km
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def create_blocks(self, coordinates: np.ndarray) -> np.ndarray:
        """Create spatial blocks for given coordinates.
        
        Args:
            coordinates: Array of [lat, lon] coordinates
            
        Returns:
            Array of block assignments for each coordinate
        """
        n_samples = len(coordinates)
        
        # Convert block size to degrees (approximate)
        # 1 degree latitude ≈ 111 km
        block_size_deg = self.block_size_km / 111.0
        
        # Find bounds
        lat_min, lat_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        lon_min, lon_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        # Create grid
        n_lat_blocks = int(np.ceil((lat_max - lat_min) / block_size_deg))
        n_lon_blocks = int(np.ceil((lon_max - lon_min) / block_size_deg))
        
        # Assign points to blocks
        block_assignments = np.zeros(n_samples, dtype=int)
        
        for i, (lat, lon) in enumerate(coordinates):
            lat_block = int((lat - lat_min) / block_size_deg)
            lon_block = int((lon - lon_min) / block_size_deg)
            block_id = lat_block * n_lon_blocks + lon_block
            block_assignments[i] = block_id
        
        return block_assignments
    
    def create_cv_splits(self, coordinates: np.ndarray, n_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create spatial cross-validation splits.
        
        Args:
            coordinates: Array of [lat, lon] coordinates
            n_folds: Number of CV folds
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        # Create blocks
        block_assignments = self.create_blocks(coordinates)
        unique_blocks = np.unique(block_assignments)
        n_blocks = len(unique_blocks)
        
        # Shuffle blocks
        shuffled_blocks = unique_blocks.copy()
        np.random.shuffle(shuffled_blocks)
        
        # Create folds
        blocks_per_fold = n_blocks // n_folds
        splits = []
        
        for fold in range(n_folds):
            # Determine test blocks for this fold
            start_idx = fold * blocks_per_fold
            if fold == n_folds - 1:
                # Last fold gets remaining blocks
                test_blocks = shuffled_blocks[start_idx:]
            else:
                end_idx = start_idx + blocks_per_fold
                test_blocks = shuffled_blocks[start_idx:end_idx]
            
            # Create train/test masks
            test_mask = np.isin(block_assignments, test_blocks)
            train_mask = ~test_mask
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            splits.append((train_indices, test_indices))
        
        return splits


class GeographicCoherence:
    """Calculate geographic coherence metrics for SOM results."""
    
    @staticmethod
    def morans_i(values: np.ndarray, coordinates: np.ndarray, 
                 weight_type: str = 'inverse_distance') -> float:
        """Calculate Moran's I statistic for spatial autocorrelation.
        
        Args:
            values: Values at each location (e.g., cluster assignments)
            coordinates: Geographic coordinates
            weight_type: Type of spatial weights ('inverse_distance' or 'knn')
            
        Returns:
            Moran's I value (-1 to 1)
        """
        n = len(values)
        if n < 3:
            return 0.0
        
        # Create spatial weights matrix
        if weight_type == 'inverse_distance':
            W = GeographicCoherence._inverse_distance_weights(coordinates)
        elif weight_type == 'knn':
            W = GeographicCoherence._knn_weights(coordinates, k=8)
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
        
        # Ensure no self-connections
        np.fill_diagonal(W, 0)
        
        # Row-standardize weights
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W = W / row_sums[:, np.newaxis]
        
        # Calculate Moran's I
        mean_val = values.mean()
        deviations = values - mean_val
        
        numerator = 0.0
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * deviations[i] * deviations[j]
        
        denominator = np.sum(deviations**2) / n
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def _inverse_distance_weights(coordinates: np.ndarray, 
                                 max_distance: float = 1000.0) -> np.ndarray:
        """Create inverse distance spatial weights matrix."""
        n = len(coordinates)
        W = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = haversine_distance_numba(
                    coordinates[i, 0], coordinates[i, 1],  # lon1, lat1
                    coordinates[j, 0], coordinates[j, 1]   # lon2, lat2
                )
                
                if dist > 0 and dist <= max_distance:
                    weight = 1.0 / dist
                    W[i, j] = W[j, i] = weight
        
        return W
    
    @staticmethod
    def _knn_weights(coordinates: np.ndarray, k: int = 8) -> np.ndarray:
        """Create k-nearest neighbors spatial weights matrix."""
        n = len(coordinates)
        W = np.zeros((n, n))
        
        # Build KD-tree for efficient nearest neighbor search
        # Convert lat/lon to radians for proper distance calculation
        coords_rad = np.radians(coordinates)
        tree = cKDTree(coords_rad)
        
        for i in range(n):
            # Find k+1 nearest neighbors (including self)
            distances, indices = tree.query(coords_rad[i], k=min(k+1, n))
            
            # Exclude self (first neighbor)
            for j, idx in enumerate(indices[1:]):
                W[i, idx] = 1.0
        
        return W
    
    @staticmethod
    def geographic_cluster_quality(cluster_assignments: np.ndarray,
                                 coordinates: np.ndarray) -> Dict[str, float]:
        """Evaluate geographic quality of clustering.
        
        Args:
            cluster_assignments: Cluster ID for each sample
            coordinates: Geographic coordinates
            
        Returns:
            Dictionary of quality metrics
        """
        unique_clusters = np.unique(cluster_assignments)
        n_clusters = len(unique_clusters)
        
        metrics = {}
        
        # 1. Overall spatial autocorrelation
        metrics['morans_i'] = GeographicCoherence.morans_i(
            cluster_assignments, coordinates
        )
        
        # 2. Within-cluster spatial cohesion
        cohesions = []
        for cluster_id in unique_clusters:
            mask = cluster_assignments == cluster_id
            if mask.sum() > 2:
                cluster_coords = coordinates[mask]
                
                # Calculate average pairwise distance within cluster
                distances = []
                for i in range(len(cluster_coords)):
                    for j in range(i+1, len(cluster_coords)):
                        dist = haversine_distance_numba(
                            cluster_coords[i, 0], cluster_coords[i, 1],  # lon1, lat1
                            cluster_coords[j, 0], cluster_coords[j, 1]   # lon2, lat2
                        )
                        distances.append(dist)
                
                if distances:
                    cohesions.append(np.mean(distances))
        
        metrics['mean_within_cluster_distance'] = np.mean(cohesions) if cohesions else 0.0
        
        # 3. Between-cluster separation
        if n_clusters > 1:
            separations = []
            for i, cluster_i in enumerate(unique_clusters):
                for j, cluster_j in enumerate(unique_clusters[i+1:], i+1):
                    mask_i = cluster_assignments == cluster_i
                    mask_j = cluster_assignments == cluster_j
                    
                    if mask_i.sum() > 0 and mask_j.sum() > 0:
                        # Calculate centroid distance
                        centroid_i = coordinates[mask_i].mean(axis=0)
                        centroid_j = coordinates[mask_j].mean(axis=0)
                        
                        dist = haversine_distance_numba(
                            centroid_i[0], centroid_i[1],  # lon1, lat1
                            centroid_j[0], centroid_j[1]   # lon2, lat2
                        )
                        separations.append(dist)
            
            metrics['mean_between_cluster_distance'] = np.mean(separations) if separations else 0.0
        else:
            metrics['mean_between_cluster_distance'] = 0.0
        
        # 4. Spatial contiguity score
        # Ratio of between to within cluster distance
        if metrics['mean_within_cluster_distance'] > 0:
            metrics['spatial_contiguity_score'] = (
                metrics['mean_between_cluster_distance'] / 
                metrics['mean_within_cluster_distance']
            )
        else:
            metrics['spatial_contiguity_score'] = 1.0
        
        return metrics


def create_geographic_grid(bounds: Tuple[float, float, float, float],
                          grid_size: Tuple[int, int]) -> np.ndarray:
    """Create a geographic grid over specified bounds.
    
    Args:
        bounds: (lat_min, lat_max, lon_min, lon_max)
        grid_size: (n_rows, n_cols)
        
    Returns:
        Array of grid coordinates (n_rows * n_cols, 2)
    """
    lat_min, lat_max, lon_min, lon_max = bounds
    n_rows, n_cols = grid_size
    
    lat_range = np.linspace(lat_min, lat_max, n_rows)
    lon_range = np.linspace(lon_min, lon_max, n_cols)
    
    grid_coords = []
    for lat in lat_range:
        for lon in lon_range:
            grid_coords.append([lat, lon])
    
    return np.array(grid_coords)