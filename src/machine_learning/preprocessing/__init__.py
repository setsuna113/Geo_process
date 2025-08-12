"""
Machine learning preprocessing module.

Provides modular preprocessing functions for the ML pipeline.
Each preprocessing step is a separate function that can be called independently.
"""

# Column standardization (includes fungi_richness addition)
from .column_standardizer import ColumnStandardizer

# Spatial sampling
from .spatial_sampler import (
    sample_spatial_grid,
    sample_by_distance
)

# Missing data handling
from .missing_handler import (
    remove_missing,
    get_required_columns,
    check_missing_patterns
)

# Climate standardization
from .climate_standardizer import (
    standardize_climate_2sd,
    standardize_climate_zscore,
    get_standardization_stats
)

# Polynomial features
from .polynomial_features import (
    add_elevation_squared,
    add_polynomial_terms,
    add_spatial_polynomial
)

# Log transformations
from .log_transformer import (
    log_transform_biodiversity,
    sqrt_transform,
    box_cox_transform,
    inverse_transform,
    check_skewness
)

# Interaction features
from .interaction_features import (
    add_continuous_interaction,
    add_plants_bio4_interaction,
    add_three_way_interaction,
    add_all_pairwise_interactions
)

# Categorical features
from .categorical_features import (
    create_temperate_categories,
    create_biodiversity_categories,
    add_categorical_interactions,
    create_climate_zones,
    create_quantile_bins
)

__all__ = [
    # Column standardization
    'ColumnStandardizer',
    
    # Spatial sampling
    'sample_spatial_grid',
    'sample_by_distance',
    
    # Missing data
    'remove_missing',
    'get_required_columns',
    'check_missing_patterns',
    
    # Climate standardization
    'standardize_climate_2sd',
    'standardize_climate_zscore',
    'get_standardization_stats',
    
    # Polynomial features
    'add_elevation_squared',
    'add_polynomial_terms',
    'add_spatial_polynomial',
    
    # Log transformations
    'log_transform_biodiversity',
    'sqrt_transform',
    'box_cox_transform',
    'inverse_transform',
    'check_skewness',
    
    # Interaction features
    'add_continuous_interaction',
    'add_plants_bio4_interaction',
    'add_three_way_interaction',
    'add_all_pairwise_interactions',
    
    # Categorical features
    'create_temperate_categories',
    'create_biodiversity_categories',
    'add_categorical_interactions',
    'create_climate_zones',
    'create_quantile_bins'
]