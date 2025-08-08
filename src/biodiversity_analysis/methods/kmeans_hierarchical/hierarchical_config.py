"""Configuration for two-stage k-means hierarchical clustering.

Clean configuration that extends base k-means config without redundancy.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
from .kmeans_config import KMeansConfig


@dataclass
class KMeansHierarchicalConfig(KMeansConfig):
    """Configuration for two-stage k-means → hierarchical clustering.
    
    Inherits k-means parameters and adds hierarchical-specific settings.
    """
    
    # Override n_clusters to be n_pre_clusters for clarity
    n_pre_clusters: int = 750  # Number of k-means pre-clusters
    
    # Variance preservation for automatic k selection
    use_variance_preservation: bool = True
    variance_threshold: float = 0.95
    variance_k_range: Tuple[int, int] = (100, 2000)
    
    # Hierarchical clustering parameters
    linkage_method: Literal['ward', 'complete', 'average', 'single'] = 'ward'
    final_k_range: Tuple[int, int] = (4, 8)
    final_k: Optional[int] = None  # If set, overrides range and selection
    
    # Method for selecting optimal final k
    selection_method: Literal['bootstrap', 'silhouette', 'davies_bouldin'] = 'bootstrap'
    n_bootstrap: int = 30
    bootstrap_sample_ratio: float = 0.9
    
    # Computational optimization
    use_minibatch: bool = True
    batch_size: int = 10000
    
    # Visualization settings
    generate_dendrogram: bool = True
    generate_geographic_maps: bool = True
    generate_3d_visualization: bool = True
    generate_validation_plots: bool = True
    n_samples_for_viz: int = 100000  # Max samples for expensive visualizations
    
    def __post_init__(self):
        """Validate hierarchical configuration."""
        # Call parent validation
        super().__post_init__()
        
        # Map n_clusters to n_pre_clusters for compatibility
        self.n_clusters = self.n_pre_clusters
        
        # Validate hierarchical-specific settings
        if self.variance_threshold <= 0 or self.variance_threshold > 1:
            raise ValueError("variance_threshold must be in (0, 1]")
            
        if self.final_k_range[0] >= self.final_k_range[1]:
            raise ValueError("final_k_range must have min < max")
            
        if self.final_k and (self.final_k < 2 or self.final_k > self.n_pre_clusters):
            raise ValueError(f"final_k must be between 2 and {self.n_pre_clusters}")
            
        if self.n_bootstrap < 10:
            raise ValueError("n_bootstrap should be at least 10 for stability")
            
        if self.linkage_method == 'ward' and self.distance_metric != 'euclidean':
            # Ward requires Euclidean distance
            self.distance_metric = 'euclidean'