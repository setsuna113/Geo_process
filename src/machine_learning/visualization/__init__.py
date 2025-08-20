"""Visualization modules for hypothesis testing and model comparison.

All plotting functionality extracted from hypothesis test modules.
"""

from .forest_plots import create_wald_visualization

from .interaction_plots import (
    visualize_contrast_results,
    create_h2_publication_figure
)

from .model_comparison_plots import plot_model_comparison

from .lrt_plots import (
    visualize_lrt_results,
    create_lrt_comparison_matrix,
    create_analysis_of_deviance_report
)

__all__ = [
    'create_wald_visualization',
    'visualize_contrast_results', 
    'create_h2_publication_figure',
    'plot_model_comparison',
    'plot_deviance_comparison',
    'visualize_lrt_results',
    'create_lrt_comparison_matrix',
    'create_analysis_of_deviance_report'
]