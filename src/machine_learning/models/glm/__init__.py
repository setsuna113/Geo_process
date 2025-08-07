"""H2O GLM models for massive parallel regression analysis.

Optimized for 256-core CPU systems and 180M+ point datasets.
"""

from .h2o_negative_binomial import H2ONegativeBinomialGLM

__all__ = [
    'H2ONegativeBinomialGLM'
]