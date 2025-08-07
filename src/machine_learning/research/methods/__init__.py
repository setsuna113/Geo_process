"""Research methods for biodiversity regression analysis."""

from .method1_continuous import Method1ContinuousGLM
from .method2_categorical import Method2CategoricalGLM
from .method3_dml import Method3DML

__all__ = [
    'Method1ContinuousGLM',
    'Method2CategoricalGLM',
    'Method3DML'
]