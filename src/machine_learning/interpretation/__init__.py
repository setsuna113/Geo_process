"""Interpretation methods for machine learning models.

Includes methods suitable for GLM (Methods 1 & 2) but NOT for DML (Method 3).
"""

from .marginal_effects import MarginalEffects, VariableImportance
from .pdp import PartialDependencePlot, InteractionStrengthAnalyzer, PartialDependence
from .effect_sizes import LargeSampleEffects
from .simple_slopes import SimpleSlopesInterpretation, analyze_simple_slopes
from .marginal_means import MarginalMeansInterpretation, analyze_marginal_means
# LRT is in hypothesis folder, imported where needed
# from .lrt import LikelihoodRatioTest, ModelComparison

__all__ = [
    'MarginalEffects',
    'VariableImportance',
    'PartialDependencePlot',
    'PartialDependence',
    'InteractionStrengthAnalyzer',
    'LargeSampleEffects',
    'SimpleSlopesInterpretation',
    'analyze_simple_slopes',
    'MarginalMeansInterpretation',
    'analyze_marginal_means'
]