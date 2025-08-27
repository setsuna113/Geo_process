"""
DoubleML package initialization
This makes imports cleaner but doesn't solve the pickle module path problem
"""

# These imports make classes available at package level
# So you can do: from src.machine_learning.doubleML import OptimizedLGBMRegressor
from .optimized_lgbm import OptimizedLGBMRegressor
from .refactored_pipeline import (
    HeterogeneousEffectEstimator,
    HomogeneousEffectEstimator
)

# This helps with imports but NOT with pickle!
# Pickle still stores the ACTUAL module where class was defined
# Not where it was imported from

__all__ = [
    'OptimizedLGBMRegressor',
    'HeterogeneousEffectEstimator', 
    'HomogeneousEffectEstimator'
]