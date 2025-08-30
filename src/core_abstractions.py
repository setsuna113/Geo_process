"""
Core abstractions - ONLY things used multiple times.

Following the principle: Abstract if used multiple times, don't if used once.
Usage counts:
- TensorData: 97 uses
- CPFactors: 65 uses  
- AnalysisResult: 33 uses
- BiodiversityData: 23 uses
- IMethodConfig: 7 implementations
- IAnalyzer: 4 implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from enum import Enum


# ============== INTERFACES (Multi-Implementation) ==============

class IAnalyzer(ABC):
    """Base analyzer interface - 4 implementations."""
    
    @abstractmethod
    def analyze(self, data: Any) -> 'AnalysisResult':
        """Perform analysis on data."""
        pass


class IMethodConfig(ABC):
    """Configuration interface - 7 implementations."""
    
    @abstractmethod
    def load_config(self, preset: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration for the method."""
        pass


# ============== DATA TYPES (Multi-Use) ==============

@dataclass
class TensorData:
    """Tensor data structure - 97 uses across cpwopt module."""
    tensor: np.ndarray
    weights: np.ndarray
    coordinates: List[Tuple[float, float]]
    feature_names: List[str]
    metadata: Optional[Dict[str, Any]] = None
    is_sparse: bool = False


@dataclass
class CPFactors:
    """CP decomposition factors - 65 uses in tensor operations."""
    spatial_factors: np.ndarray  # Lat/lon factors
    feature_factors: np.ndarray  # Species/feature factors
    weights: np.ndarray          # Factor weights
    rank: int
    n_iterations: int
    final_error: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisMetadata:
    """Metadata for analysis results."""
    method: str
    parameters: Dict[str, Any]
    timestamp: Optional[str] = None
    version: Optional[str] = None

@dataclass
class AnalysisResult:
    """Analysis result container - 33 uses across all analyzers."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None


@dataclass
class BiodiversityData:
    """Biodiversity data structure - 23 uses."""
    features: np.ndarray
    coordinates: np.ndarray
    species_names: List[str]
    metadata: Dict[str, Any]


@dataclass
class TensorMetrics:
    """Metrics for evaluating tensor factorization quality."""
    reconstruction_error: float
    relative_error: float
    fit: float
    convergence_history: List[float]
    component_weights: np.ndarray
    explained_variance: float


@dataclass
class SOMConfig:
    """Configuration for SOM training - 8 uses."""
    grid_size: Tuple[int, int]
    distance_metric: str = 'euclidean'
    neighborhood_function: str = 'gaussian'
    initialization_method: str = 'pca'
    batch_size: Optional[int] = None
    max_iterations: int = 1000
    initial_neighborhood_size: float = 1.0
    final_neighborhood_size: float = 0.1
    momentum: float = 0.0


@dataclass
class SOMTrainingResult:
    """Result of SOM training - 8 uses."""
    weights: np.ndarray
    quantization_errors: List[float]
    topographic_errors: List[float]
    training_time: float
    convergence_metric: float = 0.0
    n_iterations: int = 0
    metadata: Optional[Dict[str, Any]] = None


# ============== ENUMS (Multi-Use) ==============

class TensorMode(Enum):
    """Tensor dimensions - used in multiple tensor operations."""
    LATITUDE = 0
    LONGITUDE = 1
    FEATURES = 2
    TIME = 3  # Optional


class AnalysisStatus(Enum):
    """Analysis status - used across analyzers."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"