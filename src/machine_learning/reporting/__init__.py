"""Reporting modules for hypothesis testing results.

All table generation and publication report functionality.
"""

from .latex_tables import (
    generate_wald_latex_table,
    generate_contrast_latex_table,
    generate_lrt_latex_table,
    generate_summary_table,
    generate_score_test_latex_table
)

from .publication_reports import (
    generate_wald_publication_report,
    generate_contrast_publication_report,
    generate_lrt_publication_report,
    generate_simple_slopes_publication_report,
    generate_score_test_publication_report
)

__all__ = [
    'generate_wald_latex_table',
    'generate_contrast_latex_table',
    'generate_lrt_latex_table',
    'generate_summary_table',
    'generate_score_test_latex_table',
    'generate_wald_publication_report',
    'generate_contrast_publication_report',
    'generate_lrt_publication_report',
    'generate_simple_slopes_publication_report',
    'generate_score_test_publication_report'
]