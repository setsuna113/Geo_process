"""Machine learning models for biodiversity analysis."""

from .linear_regression import LinearRegressionAnalyzer

# Import LightGBM only if available
try:
    from .lightgbm_regressor import LightGBMAnalyzer
    __all__ = ['LinearRegressionAnalyzer', 'LightGBMAnalyzer']
except ImportError:
    __all__ = ['LinearRegressionAnalyzer']