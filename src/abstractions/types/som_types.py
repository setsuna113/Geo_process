"""Type definitions for Self-Organizing Map analysis."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from enum import Enum


class DistanceMetric(Enum):
    """Available distance metrics for SOM."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    BRAY_CURTIS = "bray_curtis"  # For biodiversity data


class NeighborhoodFunction(Enum):
    """Neighborhood functions for SOM training."""
    GAUSSIAN = "gaussian"
    BUBBLE = "bubble"
    MEXICAN_HAT = "mexican_hat"


class InitializationMethod(Enum):
    """Weight initialization methods."""
    RANDOM = "random"
    PCA = "pca"
    SAMPLE = "sample"


@dataclass
class SOMConfig:
    """Configuration for SOM training."""
    grid_size: Tuple[int, int]
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    neighborhood_function: NeighborhoodFunction = NeighborhoodFunction.GAUSSIAN
    initialization_method: InitializationMethod = InitializationMethod.PCA
    learning_rate: float = 0.5
    min_learning_rate: float = 0.01
    neighborhood_radius: float = 1.0
    min_neighborhood_radius: float = 0.1
    epochs: int = 100
    batch_size: Optional[int] = None  # For batch training
    random_seed: Optional[int] = None
    convergence_threshold: float = 0.001  # Relative improvement threshold
    convergence_window: int = 5  # Number of epochs to check for convergence


@dataclass
class SOMTrainingResult:
    """Result of SOM training."""
    weights: np.ndarray  # Shape: (grid_rows, grid_cols, n_features)
    quantization_errors: List[float]
    topographic_errors: List[float]
    training_time: float
    convergence_epoch: Optional[int]
    final_learning_rate: float
    final_neighborhood_radius: float
    n_samples_trained: int
    
    @property
    def converged(self) -> bool:
        """Check if training converged before max epochs."""
        return self.convergence_epoch is not None
    
    @property
    def final_quantization_error(self) -> float:
        """Get final quantization error."""
        return self.quantization_errors[-1] if self.quantization_errors else float('inf')
    
    @property
    def final_topographic_error(self) -> float:
        """Get final topographic error."""
        return self.topographic_errors[-1] if self.topographic_errors else float('inf')


@dataclass
class SOMVisualizationConfig:
    """Configuration for SOM visualizations."""
    figure_size: Tuple[int, int] = (10, 10)
    colormap: str = "viridis"
    show_grid_lines: bool = True
    component_planes_cols: int = 4
    u_matrix_interpolation: str = "bilinear"
    save_format: str = "png"
    dpi: int = 300


@dataclass
class BiodiversityMetrics:
    """Biodiversity-specific metrics for SOM evaluation."""
    species_coherence: float
    beta_diversity_preservation: float
    environmental_gradient_detection: Dict[str, float]
    rare_species_representation: float
    spatial_autocorrelation_preserved: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'species_coherence': self.species_coherence,
            'beta_diversity_preservation': self.beta_diversity_preservation,
            'environmental_gradient_detection': self.environmental_gradient_detection,
            'rare_species_representation': self.rare_species_representation,
            'spatial_autocorrelation_preserved': self.spatial_autocorrelation_preserved
        }