"""Machine learning models for biodiversity analysis.

All models use H2O for massive parallelization on 256-core systems.
"""

# H2O GLM models
from .glm.h2o_negative_binomial import H2ONegativeBinomialGLM

# H2O GBM models
from .gbm.h2o_gbm import H2OGBM

__all__ = [
    'H2ONegativeBinomialGLM',
    'H2OGBM'
]