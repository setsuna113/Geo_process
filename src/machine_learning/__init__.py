"""Machine learning module for biodiversity analysis."""

# Import H2O models
from .models import H2ONegativeBinomialGLM, H2OGBM

# Import preprocessing (H2O-optimized)
from .preprocessing import (
    MESFProcessor,
    preprocess_for_regression
)

# Import imputation (module not yet implemented)
# from .preprocessing.imputation.knn_imputer import SpatialKNNImputer

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
    # 'SpatialKNNImputer',  # Not yet implemented
    'SpatialBlockCV',
    'SpatialBufferCV',
    'EnvironmentalBlockCV'
]