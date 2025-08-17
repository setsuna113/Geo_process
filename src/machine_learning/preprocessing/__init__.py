"""
Machine learning preprocessing module.

Provides H2O-optimized preprocessing functions for the ML pipeline.
All functions work directly with H2O frames to avoid conversions.
"""

# Note: H2O preprocessing has been replaced with Julia-based preprocessing
# These functions are now handled in JuliaManager
# Keeping stubs for backward compatibility

# Stub functions for backward compatibility
def remove_missing(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

def standardize_climate_2sd(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

def standardize_column_names(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

def add_elevation_squared(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

def log_transform_biodiversity(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

def add_fungi_richness(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

def create_temperate_categories(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

def create_biodiversity_categories(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

def add_categorical_interactions(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

def preprocess_for_regression(*args, **kwargs):
    raise NotImplementedError("Use JuliaManager for data preprocessing")

# Specialized processors
from .mesf_processor import MESFProcessor
from .fungi_processor import get_fungi_richness

# PyArrow-based utilities (for caching/parquet operations)
from .column_standardizer import ColumnStandardizer

__all__ = [
    # Stub functions (for backward compatibility)
    'remove_missing',
    'standardize_climate_2sd',
    'standardize_column_names',
    'add_elevation_squared',
    'log_transform_biodiversity',
    'add_fungi_richness',
    'create_temperate_categories',
    'create_biodiversity_categories',
    'add_categorical_interactions',
    'preprocess_for_regression',
    
    # Processors (still available)
    'MESFProcessor',
    'get_fungi_richness',
    'ColumnStandardizer'
]