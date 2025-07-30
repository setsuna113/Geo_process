"""Type definitions for machine learning components."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from enum import Enum
import numpy as np
from datetime import datetime

from ..interfaces.analyzer import AnalysisResult, AnalysisMetadata


@dataclass
class ModelMetadata:
    """Metadata for trained ML models."""
    model_type: str  # "regression", "classification", "clustering"
    algorithm: str  # "ridge", "lightgbm", etc.
    training_samples: int
    training_features: int
    feature_names: List[str]
    target_name: str
    
    # Training info
    training_time: float
    training_date: datetime
    cv_score: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Model parameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    
    # Performance metrics
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Feature importance
    feature_importance: Optional[Dict[str, float]] = None
    
    # Data characteristics
    missing_value_handling: str = "none"
    scaling_method: Optional[str] = None
    class_balance: Optional[Dict[str, int]] = None  # For classification


@dataclass
class MLResult(AnalysisResult):
    """Extended analysis result for ML models."""
    # All fields must have defaults because parent class has fields with defaults
    predictions: Optional[np.ndarray] = None
    prediction_uncertainty: Optional[np.ndarray] = None
    model_metadata: Optional[ModelMetadata] = None
    model_path: Optional[str] = None
    cv_results: Optional['CVResult'] = None
    
    # Feature analysis
    feature_importance: Optional[Dict[str, float]] = None
    feature_statistics: Optional[Dict[str, Dict[str, float]]] = None
    
    # Spatial analysis
    spatial_autocorrelation: Optional[float] = None
    residual_clustering: Optional[Dict[str, float]] = None


@dataclass
class FeatureMetadata:
    """Metadata for engineered features."""
    name: str
    description: str
    category: str  # "spatial", "ecological", "richness", "temporal"
    dtype: str
    
    # Value characteristics
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    missing_rate: float = 0.0
    
    # Creation info
    source_columns: List[str] = field(default_factory=list)
    transformation: str = "identity"
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Importance
    prior_importance: Optional[float] = None
    correlation_with_target: Optional[float] = None


@dataclass
class CVFold:
    """Information about a single cross-validation fold."""
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    
    # Spatial information
    train_bounds: Optional[Tuple[float, float, float, float]] = None  # minx, miny, maxx, maxy
    test_bounds: Optional[Tuple[float, float, float, float]] = None
    
    # Fold statistics
    train_size: int = 0
    test_size: int = 0
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    training_time: float = 0.0
    prediction_time: float = 0.0


@dataclass
class CVResult:
    """Cross-validation results."""
    strategy: str  # "spatial_block", "environmental", "random"
    n_folds: int
    folds: List[CVFold]
    
    # Aggregate metrics
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Spatial characteristics
    spatial_autocorrelation_range: Optional[float] = None
    buffer_distance: Optional[float] = None
    block_size: Optional[float] = None
    
    # Overall timing
    total_time: float = 0.0
    
    # Feature importance stability
    feature_importance_per_fold: Optional[List[Dict[str, float]]] = None
    stable_features: Optional[List[str]] = None


class ImputationStrategy(Enum):
    """Available imputation strategies."""
    NONE = "none"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    KNN = "knn"
    ITERATIVE = "iterative"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    DROP = "drop"


class ModelType(Enum):
    """Types of ML models."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


class ScalingMethod(Enum):
    """Feature scaling methods."""
    NONE = "none"
    STANDARD = "standard"  # Zero mean, unit variance
    MINMAX = "minmax"  # Scale to [0, 1]
    ROBUST = "robust"  # Median and IQR
    MAXABS = "maxabs"  # Scale to [-1, 1]
    NORMALIZER = "normalizer"  # Unit norm


@dataclass
class ImputationResult:
    """Result of imputation process."""
    strategy: ImputationStrategy
    imputed_columns: List[str]
    imputation_values: Dict[str, Any]  # Column -> imputation value/model
    
    # Statistics
    missing_before: Dict[str, float]  # Column -> missing rate
    missing_after: Dict[str, float]
    
    # Quality metrics
    imputation_quality: Optional[Dict[str, float]] = None
    convergence_info: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    
    # Specific checks
    collinearity_issues: List[Tuple[str, str, float]] = field(default_factory=list)  # var1, var2, correlation
    scale_issues: List[Tuple[str, float, float]] = field(default_factory=list)  # var, min, max
    outlier_issues: List[Tuple[str, int]] = field(default_factory=list)  # var, count
    
    # Recommendations
    features_to_remove: List[str] = field(default_factory=list)
    features_to_transform: Dict[str, str] = field(default_factory=dict)  # feature -> transformation


@dataclass
class SpatialCVStrategy:
    """Configuration for spatial cross-validation."""
    strategy_type: str  # "block", "buffer", "environmental"
    
    # Spatial parameters
    block_size: Optional[float] = None  # in km
    buffer_distance: Optional[float] = None  # in km
    
    # Environmental parameters
    stratify_by: Optional[List[str]] = None  # ["latitude", "biome", etc.]
    n_strata: Optional[int] = None
    
    # General parameters
    n_folds: int = 5
    random_state: Optional[int] = None
    
    # Validation
    ensure_minimum_samples: int = 10
    allow_overlapping_buffers: bool = False