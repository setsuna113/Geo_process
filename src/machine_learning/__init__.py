"""Machine learning module for biodiversity analysis."""

# Import H2O models
from .models import H2ONegativeBinomialGLM, H2OGBM

# Import feature builders
from .preprocessing.feature_engineering import (
    RichnessFeatureBuilder,
    SpatialFeatureBuilder,
    EcologicalFeatureBuilder,
    CompositeFeatureBuilder
)

# Import imputation
from .preprocessing.imputation.knn_imputer import SpatialKNNImputer

# Import CV strategies
from .validation.spatial_cv import (
    SpatialBlockCV,
    SpatialBufferCV,
    EnvironmentalBlockCV
)

__all__ = [
    'H2ONegativeBinomialGLM',
    'H2OGBM',
    'RichnessFeatureBuilder',
    'SpatialFeatureBuilder',
    'EcologicalFeatureBuilder',
    'CompositeFeatureBuilder',
    'SpatialKNNImputer',
    'SpatialBlockCV',
    'SpatialBufferCV',
    'EnvironmentalBlockCV'
]