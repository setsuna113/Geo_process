"""
Biodiversity Inference Module for Double Machine Learning

This module provides domain-specific interpretation and hypothesis testing
for biodiversity effects on fungi richness using EconML's built-in inference.

Key Components:
- BiodiversityInference: Main class for biodiversity hypothesis testing
- analyze_saved_results: Quick analysis of saved EconML results
- demonstrate_proper_usage: Examples of correct EconML API usage

Usage:
    from biodiversity_inference import BiodiversityInference
    
    # From saved results
    bio_inf = BiodiversityInference(results_path="path/to/results.json")
    results = bio_inf.test_biodiversity_hypothesis()
    
    # From fitted estimator
    bio_inf = BiodiversityInference(estimator=fitted_dml_model)
    results = bio_inf.test_biodiversity_hypothesis()
"""

from .inference import BiodiversityInference
from .analysis import analyze_saved_results, demonstrate_proper_econml_usage
from .hetero import (
    get_hetero_inference, 
    rank_regions_by_impact,
    analyze_gradient,
    interpret_biodiversity_effects
)

__version__ = "2.0.0"

__all__ = [
    # Constant effects (homogeneous)
    'BiodiversityInference',
    'analyze_saved_results',
    'demonstrate_proper_econml_usage',
    # Heterogeneous effects
    'get_hetero_inference',
    'rank_regions_by_impact', 
    'analyze_gradient',
    'interpret_biodiversity_effects'
]