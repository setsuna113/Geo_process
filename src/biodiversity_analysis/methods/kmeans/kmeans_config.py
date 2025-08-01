"""Configuration dataclass for k-means clustering."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
import numpy as np


@dataclass
class KMeansConfig:
    """Configuration for biodiversity k-means clustering.
    
    Attributes:
        n_clusters: Number of clusters
        init: Initialization method ('k-means++', 'random')
        n_init: Number of initializations to try
        max_iter: Maximum iterations per run
        tol: Convergence tolerance
        random_state: Random seed for reproducibility
        
        distance_metric: Distance metric ('bray_curtis', 'euclidean')
        weight_method: How to calculate feature weights ('auto', 'completeness', 'variance', 'fixed')
        fixed_weights: Fixed weights if weight_method='fixed'
        
        adaptive_mode: How to adapt minimum features ('latitude', 'density', 'both')
        arctic_boundary: Latitude boundary for Arctic (degrees)
        temperate_boundary: Latitude boundary for temperate zone
        arctic_min_features: Minimum features required in Arctic
        temperate_min_features: Minimum features in temperate zone
        tropical_min_features: Minimum features in tropical zone
        
        grid_size_km: Size of each grid cell in km
        neighborhood_radius_km: Radius for neighborhood calculations
        remote_threshold_km: Distance threshold for remote areas
        min_neighbors_remote: Minimum neighbors to not be considered remote
        
        transform: Data transformation ('log1p', 'sqrt', 'none')
        normalize: Normalization method ('standardize', 'minmax', 'none')
        handle_zeros: Whether to handle zero inflation
        
        prefilter_empty: Remove grids with no data
        min_features_prefilter: Minimum features to keep grid
        use_sparse_distances: Use sparse distance matrix
        chunk_size: Size of chunks for processing
        n_jobs: Number of parallel jobs (-1 for all cores)
        
        calculate_silhouette: Whether to calculate silhouette scores
        silhouette_sample_size: Sample size for silhouette calculation
        determine_k_method: Method for optimal k ('elbow', 'silhouette', 'both')
    """
    # Core k-means parameters
    n_clusters: int = 20
    init: Literal['k-means++', 'random'] = 'k-means++'
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    random_state: Optional[int] = 42
    
    # Distance and weights
    distance_metric: Literal['bray_curtis', 'euclidean'] = 'bray_curtis'
    weight_method: Literal['auto', 'completeness', 'variance', 'fixed'] = 'auto'
    fixed_weights: Optional[List[float]] = None
    
    # Adaptive thresholds
    adaptive_mode: Literal['latitude', 'density', 'both'] = 'latitude'
    arctic_boundary: float = 66.5
    temperate_boundary: float = 45.0
    arctic_min_features: int = 1
    temperate_min_features: int = 2
    tropical_min_features: int = 2
    
    # Spatial parameters
    grid_size_km: float = 18.0
    neighborhood_radius_km: float = 100.0
    remote_threshold_km: float = 200.0
    min_neighbors_remote: int = 10
    
    # Preprocessing
    transform: Literal['log1p', 'sqrt', 'none'] = 'log1p'
    normalize: Literal['standardize', 'minmax', 'none'] = 'standardize'
    handle_zeros: bool = True
    
    # Optimization
    prefilter_empty: bool = True
    min_features_prefilter: int = 1
    use_sparse_distances: bool = True
    chunk_size: int = 10000
    n_jobs: int = -1
    
    # Validation
    calculate_silhouette: bool = True
    silhouette_sample_size: int = 5000
    determine_k_method: Literal['elbow', 'silhouette', 'both'] = 'silhouette'
    
    def __post_init__(self):
        """Validate configuration."""
        if self.fixed_weights is None:
            self.fixed_weights = [1.0, 1.0, 1.0, 1.0]
        
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        
        if self.arctic_boundary <= self.temperate_boundary:
            raise ValueError("arctic_boundary must be greater than temperate_boundary")
        
        if self.min_features_prefilter < 1:
            raise ValueError("min_features_prefilter must be at least 1")