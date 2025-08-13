"""
Machine learning preprocessing module.

Provides H2O-optimized preprocessing functions for the ML pipeline.
All functions work directly with H2O frames to avoid conversions.
"""

# H2O-optimized preprocessing (primary interface)
from .h2o_preprocessing import (
    # Data cleaning
    remove_missing,
    
    # Standardization
    standardize_climate_2sd,
    standardize_column_names,
    
    # Feature engineering
    add_elevation_squared,
    log_transform_biodiversity,
    add_fungi_richness,
    
    # Categorical features for Method 2
    create_temperate_categories,
    create_biodiversity_categories,
    add_categorical_interactions,
    
    # Complete pipeline
    preprocess_for_regression
)

# Specialized processors
from .mesf_processor import MESFProcessor
from .fungi_processor import get_fungi_richness

# PyArrow-based utilities (for caching/parquet operations)
from .column_standardizer import ColumnStandardizer

__all__ = [
    # H2O preprocessing functions
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
    
    # Processors
    'MESFProcessor',
    'get_fungi_richness',
    'ColumnStandardizer'
]