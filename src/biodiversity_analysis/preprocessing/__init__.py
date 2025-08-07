"""Shared preprocessing module for biodiversity analysis.

This module provides unified preprocessing for all analysis methods (SOM, CP-WOPT, K-means, etc.)
to avoid redundancy and ensure consistency.

The SharedPreprocessor class automatically configures itself based on the analysis method:
- SOM: minmax normalization, zero imputation
- CP-WOPT: minmax normalization (ensures non-negative), zero imputation  
- K-means: standardization, mean imputation
"""

from .pipeline import SharedPreprocessor

__all__ = [
    'SharedPreprocessor'  # The only preprocessor we need
]