"""Visualization modules for hypothesis testing and model comparison.

All plotting functionality extracted from hypothesis test modules.
"""

from .forest_plots import create_wald_visualization, create_method2_comparison
from .model_comparison_plots import plot_model_comparison

from .lrt_plots import (
    visualize_lrt_results,
    create_lrt_comparison_matrix,
    create_analysis_of_deviance_report
)

from .dml_magnitude_plots import visualize_dml_magnitude
from .bayesian_dml_plots import visualize_bayesian_dml

__all__ = [
    'create_wald_visualization',
    'create_method2_comparison',
    'plot_model_comparison',
    'visualize_lrt_results',
    'create_lrt_comparison_matrix',
    'create_analysis_of_deviance_report',
    'visualize_dml_magnitude',
    'visualize_bayesian_dml'
]