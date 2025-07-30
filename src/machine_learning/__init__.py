"""Machine learning module for biodiversity analysis."""

# Import models
from .models import LinearRegressionAnalyzer

# Import feature builders
from .preprocessing.feature_engineering import (
    RichnessFeatureBuilder,
    SpatialFeatureBuilder,
    EcologicalFeatureBuilder
)

# Import imputation
from .preprocessing.imputation.knn_imputer import SpatialKNNImputer

# Import CV strategies
from .validation.spatial_cv import (
    SpatialBlockCV,
    SpatialBufferCV,
    EnvironmentalBlockCV
)

# Try to import LightGBM model if available
try:
    from .models import LightGBMAnalyzer
    __all__ = [
        'LinearRegressionAnalyzer',
        'LightGBMAnalyzer',
        'RichnessFeatureBuilder',
        'SpatialFeatureBuilder',
        'EcologicalFeatureBuilder',
        'SpatialKNNImputer',
        'SpatialBlockCV',
        'SpatialBufferCV',
        'EnvironmentalBlockCV'
    ]
except ImportError:
    __all__ = [
        'LinearRegressionAnalyzer',
        'RichnessFeatureBuilder',
        'SpatialFeatureBuilder',
        'EcologicalFeatureBuilder',
        'SpatialKNNImputer',
        'SpatialBlockCV',
        'SpatialBufferCV',
        'EnvironmentalBlockCV'
    ]