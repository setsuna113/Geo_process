"""Configuration for two-stage k-means hierarchical clustering.

Standalone configuration without deprecated dependencies.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal


@dataclass
class KMeansHierarchicalConfig:
    """Configuration for two-stage k-means → hierarchical clustering.
    
    Standalone configuration for hierarchical clustering parameters.
    
    Note: Configuration precedence (highest to lowest):
    1. Runtime parameters passed to analyze()
    2. YAML configuration file (kmeans_hierarchical_config.yml)
    3. Default values in this dataclass
    """
    
    # K-means pre-clustering parameters
    n_pre_clusters: int = 750  # Number of k-means pre-clusters
    random_state: int = 42
    
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
    
    # Additional processing parameters
    silhouette_sample_size: int = 50000  # Sample size for silhouette calculation
    
    def __post_init__(self):
        """Validate hierarchical configuration."""
        # Validate hierarchical-specific settings
        if self.variance_threshold <= 0 or self.variance_threshold > 1:
            raise ValueError("variance_threshold must be in (0, 1]")
            
        if self.final_k_range[0] >= self.final_k_range[1]:
            raise ValueError("final_k_range must have min < max")
            
        if self.final_k and (self.final_k < 2 or self.final_k > self.n_pre_clusters):
            raise ValueError(f"final_k must be between 2 and {self.n_pre_clusters}")
            
        if self.n_bootstrap < 10:
            raise ValueError("n_bootstrap should be at least 10 for stability")