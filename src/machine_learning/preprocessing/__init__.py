"""
Machine learning preprocessing module.

All preprocessing is now handled by the Julia pipeline.
This module provides compatibility stubs and specialized processors.
"""

# Note: All preprocessing has been moved to Julia
# The Julia pipeline (julia_complete_pipeline.jl) handles:
# - Data loading and sampling
# - Missing value removal
# - Standardization (2SD method)
# - Feature engineering (elevation squared, log transforms)
# - MESF spatial eigenvectors
# - Categorical feature creation

# Specialized processors still in Python
from .mesf_processor import MESFProcessor
from .fungi_processor import get_fungi_richness
from .column_standardizer import ColumnStandardizer

__all__ = [
    'MESFProcessor',
    'get_fungi_richness', 
    'ColumnStandardizer'
]