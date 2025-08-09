"""Simplified configuration for k-means clustering."""

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np


@dataclass
class KMeansConfig:
    """Essential configuration for biodiversity k-means clustering.
    
    Reduced from 25+ parameters to 12 essential ones.
    """
    
    # Core clustering parameters
    n_clusters: int = 50
    init: Literal['k-means++', 'random'] = 'k-means++'
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    random_state: Optional[int] = 42
    
    # Processing parameters
    transform: Literal['log1p', 'sqrt', 'none'] = 'log1p'
    normalize: Literal['standardize', 'minmax', 'none'] = 'standardize'
    handle_zeros: bool = True
    
    # Computational parameters
    chunk_size: int = 10000
    n_jobs: int = -1
    
    # Analysis parameters
    calculate_silhouette: bool = True
    silhouette_sample_size: int = 50000
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1")
            
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")
            
        if self.tol <= 0:
            raise ValueError("tol must be positive")
            
        if self.chunk_size < 100:
            raise ValueError("chunk_size should be at least 100")
