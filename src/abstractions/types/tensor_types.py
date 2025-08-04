"""Type definitions for tensor factorization methods.

This module defines data structures for tensor-based biodiversity analysis,
particularly for CP-WOPT (CANDECOMP/PARAFAC with weighted optimization).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from enum import Enum


class TensorMode(Enum):
    """Modes (dimensions) of a biodiversity tensor."""
    LATITUDE = 0
    LONGITUDE = 1
    FEATURES = 2  # Species or environmental features
    TIME = 3  # Optional temporal dimension


class FactorInitMethod(Enum):
    """Initialization methods for factor matrices."""
    RANDOM = "random"
    SVD = "svd"
    NNDSVD = "nndsvd"  # Non-negative Double SVD
    PCA_NOISE = "pca_noise"


class TensorOptimizer(Enum):
    """Optimization methods for tensor factorization."""
    ALS = "als"  # Alternating Least Squares
    LBFGS = "lbfgs"  # Limited-memory BFGS
    SGD = "sgd"  # Stochastic Gradient Descent
    ADMM = "admm"  # Alternating Direction Method of Multipliers


@dataclass
class TensorData:
    """Represents a tensor with associated metadata for biodiversity analysis.
    
    Attributes:
        tensor: The main data tensor (3D or 4D)
        weights: Binary mask for missing values (same shape as tensor)
        coordinates: Grid cell centers [(lat, lon), ...]
        feature_names: Names of features/species
        metadata: Additional metadata (grid resolution, CRS, etc.)
        is_sparse: Whether to use sparse representation
    """
    tensor: np.ndarray
    weights: np.ndarray
    coordinates: List[Tuple[float, float]]
    feature_names: List[str]
    metadata: Dict[str, Any]
    is_sparse: bool = False
    
    def __post_init__(self):
        """Validate tensor data after initialization."""
        if self.tensor.shape != self.weights.shape:
            raise ValueError(f"Tensor shape {self.tensor.shape} != weights shape {self.weights.shape}")
        
        # Coordinates are optional for grid-based tensor data
        if self.coordinates and len(self.coordinates) > 0:
            if len(self.coordinates) != self.tensor.shape[0] * self.tensor.shape[1]:
                raise ValueError(f"Number of coordinates {len(self.coordinates)} doesn't match spatial dimensions")
        
        if len(self.feature_names) != self.tensor.shape[2]:
            raise ValueError(f"Number of feature names {len(self.feature_names)} doesn't match feature dimension")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self.tensor.shape
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self.tensor.ndim
    
    @property
    def n_observed(self) -> int:
        """Get number of observed (non-missing) entries."""
        return int(np.sum(self.weights))
    
    @property
    def sparsity(self) -> float:
        """Get sparsity level (proportion of missing values)."""
        return 1.0 - (self.n_observed / self.tensor.size)


@dataclass
class CPFactors:
    """Factor matrices for CP decomposition.
    
    Attributes:
        factors: List of factor matrices, one per mode
        rank: Number of components
        mode_names: Names of each mode for interpretability
    """
    factors: List[np.ndarray]
    rank: int
    mode_names: List[str]
    
    def __post_init__(self):
        """Validate factors after initialization."""
        if len(self.factors) != len(self.mode_names):
            raise ValueError("Number of factors must match number of mode names")
        
        # Check rank consistency
        for i, factor in enumerate(self.factors):
            if factor.shape[1] != self.rank:
                raise ValueError(f"Factor {i} has {factor.shape[1]} components, expected {self.rank}")
    
    @property
    def n_modes(self) -> int:
        """Get number of modes."""
        return len(self.factors)
    
    def get_factor(self, mode: Union[int, str]) -> np.ndarray:
        """Get factor matrix by index or name."""
        if isinstance(mode, str):
            try:
                mode = self.mode_names.index(mode)
            except ValueError:
                raise ValueError(f"Mode '{mode}' not found. Available: {self.mode_names}")
        return self.factors[mode]


@dataclass
class TensorMetrics:
    """Metrics for evaluating tensor factorization quality.
    
    Attributes:
        reconstruction_error: Weighted Frobenius norm of residual
        relative_error: Reconstruction error / norm of original
        fit: 1 - relative_error (goodness of fit)
        convergence_history: List of errors over iterations
        component_weights: Importance of each component
        explained_variance: Variance explained by the decomposition
    """
    reconstruction_error: float
    relative_error: float
    fit: float
    convergence_history: List[float]
    component_weights: Optional[np.ndarray] = None
    explained_variance: Optional[float] = None
    
    @classmethod
    def from_errors(cls, recon_error: float, original_norm: float, 
                    history: List[float]) -> 'TensorMetrics':
        """Create metrics from reconstruction error and original norm."""
        relative_error = recon_error / original_norm if original_norm > 0 else np.inf
        fit = 1.0 - relative_error
        return cls(
            reconstruction_error=recon_error,
            relative_error=relative_error,
            fit=fit,
            convergence_history=history
        )


@dataclass
class TensorDecompositionResult:
    """Complete result of tensor factorization.
    
    Attributes:
        factors: The decomposed factor matrices
        metrics: Quality metrics
        metadata: Algorithm parameters and runtime info
        spatial_patterns: Extracted spatial patterns (optional)
        feature_loadings: Feature importance/loadings (optional)
    """
    factors: CPFactors
    metrics: TensorMetrics
    metadata: Dict[str, Any]
    spatial_patterns: Optional[np.ndarray] = None
    feature_loadings: Optional[np.ndarray] = None
    
    def reconstruct(self, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct the tensor from factors.
        
        Args:
            weights: Optional weight mask to apply
            
        Returns:
            Reconstructed tensor
        """
        # This is a placeholder - actual implementation will use tensor operations
        # Will be implemented in the core algorithm
        raise NotImplementedError("Reconstruction will be implemented in core.py")


@dataclass
class MiniBatchConfig:
    """Configuration for mini-batch processing.
    
    Attributes:
        enabled: Whether to use mini-batching
        batch_size: Number of tensor slices per batch
        mode: Which mode to batch along (usually spatial)
        shuffle: Whether to shuffle batches
        drop_last: Whether to drop incomplete last batch
    """
    enabled: bool = True
    batch_size: int = 10000
    mode: int = 0  # Batch along first mode (latitude) by default
    shuffle: bool = True
    drop_last: bool = False