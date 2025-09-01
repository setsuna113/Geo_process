"""Evaluation modules for ML models."""

from .model_comparison import ModelComparison

# Lazy imports to avoid loading R when not needed
def get_emmeans_wrapper():
    from .emmeans_core import REMmeansWrapper
    return REMmeansWrapper

def get_julia_emmeans():
    from .emmeans_julia import JuliaEMmeans
    return JuliaEMmeans

__all__ = [
    'ModelComparison',
    'get_emmeans_wrapper',
    'get_julia_emmeans'
]