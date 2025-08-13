"""H2O-optimized preprocessing functions.

All functions work directly with H2O frames to avoid conversions.
"""

import logging
import h2o
from typing import List, Optional

logger = logging.getLogger(__name__)


def remove_missing(data: h2o.H2OFrame) -> h2o.H2OFrame:
    """Remove rows with missing data from H2O frame.
    
    Args:
        data: H2O frame
        
    Returns:
        H2O frame with missing rows removed
    """
    initial_rows = data.nrows
    data_clean = data.na_omit()
    removed = initial_rows - data_clean.nrows
    
    if removed > 0:
        logger.info(f"Removed {removed:,} rows with missing data ({removed/initial_rows*100:.2f}%)")
    
    return data_clean


def standardize_climate_2sd(
    data: h2o.H2OFrame,
    climate_vars: Optional[List[str]] = None
) -> h2o.H2OFrame:
    """Standardize climate variables using 2SD method (Gelman standardization).
    
    Divides by 2 standard deviations for better interpretability with binary predictors.
    
    Args:
        data: H2O frame
        climate_vars: List of climate variables to standardize
                     (default: bio01, bio04, bio12, bio15, elevation)
        
    Returns:
        H2O frame with standardized climate variables added as *_std columns
    """
    if climate_vars is None:
        climate_vars = ['bio01', 'bio04', 'bio12', 'bio15', 'elevation']
    
    for var in climate_vars:
        if var in data.columns:
            mean_val = data[var].mean()
            std_val = data[var].sd()
            
            # Gelman standardization: divide by 2 SDs
            data[f'{var}_std'] = (data[var] - mean_val) / (2 * std_val)
            
            logger.debug(f"Standardized {var}: mean={mean_val:.2f}, std={std_val:.2f}")
        else:
            logger.warning(f"Variable {var} not found in data")
    
    logger.info(f"Standardized {len(climate_vars)} climate variables using 2SD method")
    return data


def add_elevation_squared(data: h2o.H2OFrame) -> h2o.H2OFrame:
    """Add elevation squared term for non-linear effects.
    
    Args:
        data: H2O frame with elevation_std column
        
    Returns:
        H2O frame with elevation_std_sq column added
    """
    if 'elevation_std' in data.columns:
        data['elevation_std_sq'] = data['elevation_std'] ** 2
        logger.info("Added elevation_std_sq polynomial term")
    else:
        logger.warning("elevation_std not found, skipping squared term")
    
    return data


def log_transform_biodiversity(
    data: h2o.H2OFrame,
    biodiversity_vars: Optional[List[str]] = None
) -> h2o.H2OFrame:
    """Log-transform biodiversity predictors.
    
    Uses log(x + 1) transformation to handle zeros.
    
    Args:
        data: H2O frame
        biodiversity_vars: List of biodiversity variables to transform
                          (default: plants_richness, vertebrates_richness)
        
    Returns:
        H2O frame with log-transformed variables added as *_log columns
    """
    if biodiversity_vars is None:
        biodiversity_vars = ['plants_richness', 'vertebrates_richness']
    
    for var in biodiversity_vars:
        if var in data.columns:
            # log(x + 1) transformation
            data[f'{var}_log'] = (data[var] + 1).log()
            logger.debug(f"Log-transformed {var}")
        else:
            logger.warning(f"Variable {var} not found in data")
    
    logger.info(f"Log-transformed {len(biodiversity_vars)} biodiversity predictors")
    return data


def standardize_column_names(data: h2o.H2OFrame) -> h2o.H2OFrame:
    """Standardize column names for consistency.
    
    Maps common variations to standard names.
    
    Args:
        data: H2O frame
        
    Returns:
        H2O frame with standardized column names
    """
    # Create mapping of old to new names
    mappings = {
        'plant_richness': 'plants_richness',
        'animal_richness': 'vertebrates_richness',
        'bio1': 'bio01',
        'bio4': 'bio04'
    }
    
    for old_name, new_name in mappings.items():
        if old_name in data.columns and new_name not in data.columns:
            data[new_name] = data[old_name]
            logger.debug(f"Renamed {old_name} to {new_name}")
    
    return data


def add_fungi_richness(data: h2o.H2OFrame) -> h2o.H2OFrame:
    """Add fungi_richness as sum of AM and ECM fungi.
    
    Args:
        data: H2O frame with am_fungi_richness and ecm_fungi_richness
        
    Returns:
        H2O frame with fungi_richness column added
    """
    if 'fungi_richness' not in data.columns:
        if 'am_fungi_richness' in data.columns and 'ecm_fungi_richness' in data.columns:
            # Use the fungi_processor for consistency
            from src.machine_learning.preprocessing.fungi_processor import get_fungi_richness
            
            fungi_col = get_fungi_richness(
                data['am_fungi_richness'],
                data['ecm_fungi_richness']
            )
            data = data.cbind(fungi_col)
            logger.info("Added fungi_richness column")
        else:
            logger.warning("Missing AM or ECM fungi columns, skipping fungi_richness")
    else:
        logger.debug("fungi_richness already exists")
    
    return data


def create_temperate_categories(
    data: h2o.H2OFrame,
    bio4_col: str = 'bio4_std',
    output_col: str = 'temperate',
    method: str = 'quantile',
    quantile: float = 0.75
) -> h2o.H2OFrame:
    """Create categorical variable for temperate vs non-temperate regions.
    
    Based on bio4 (temperature seasonality) - high seasonality = temperate.
    
    Args:
        data: H2O frame
        bio4_col: Name of temperature seasonality column
        output_col: Name for categorical output
        method: 'median', 'quantile', or 'threshold'
        quantile: Quantile to use if method='quantile'
        
    Returns:
        H2O frame with temperate category added (1 = temperate, 0 = non-temperate)
    """
    if bio4_col not in data.columns:
        logger.error(f"Column {bio4_col} not found")
        return data
    
    if method == 'median':
        threshold = data[bio4_col].median()
        if isinstance(threshold, list):
            threshold = threshold[0]
    elif method == 'quantile':
        threshold = data[bio4_col].quantile([quantile])[0]
    else:
        threshold = 0  # For standardized data
    
    # High seasonality = temperate (bio4 > threshold)
    data[output_col] = (data[bio4_col] > threshold).ifelse(1, 0)
    
    logger.info(f"Created temperate category using {bio4_col} (threshold={threshold:.3f}, method={method})")
    return data


def create_biodiversity_categories(
    data: h2o.H2OFrame,
    richness_col: str = 'plants_richness',
    output_col: str = 'plant_high',
    method: str = 'quantile',
    quantile: float = 0.75
) -> h2o.H2OFrame:
    """Create categorical variable for high vs low biodiversity.
    
    Args:
        data: H2O frame
        richness_col: Name of richness column
        output_col: Name for categorical output
        method: 'median', 'quantile', etc.
        quantile: Quantile to use if method='quantile'
        
    Returns:
        H2O frame with biodiversity category added
    """
    if richness_col not in data.columns:
        logger.error(f"Column {richness_col} not found")
        return data
    
    if method == 'median':
        threshold = data[richness_col].median()
        if isinstance(threshold, list):
            threshold = threshold[0]
    elif method == 'quantile':
        threshold = data[richness_col].quantile([quantile])[0]
    else:
        threshold = 100  # Default threshold
    
    data[output_col] = (data[richness_col] > threshold).ifelse(1, 0)
    
    logger.info(f"Created biodiversity category for {richness_col} (threshold={threshold:.0f}, method={method})")
    return data


def add_categorical_interactions(
    data: h2o.H2OFrame,
    cat1: str = 'temperate',
    cat2: str = 'plant_high',
    output_col: str = 'temperate_plant_high'
) -> h2o.H2OFrame:
    """Add interaction between two categorical variables.
    
    Args:
        data: H2O frame
        cat1: First categorical variable
        cat2: Second categorical variable
        output_col: Name for interaction column
        
    Returns:
        H2O frame with interaction column added
    """
    if cat1 not in data.columns or cat2 not in data.columns:
        logger.error(f"Required columns {cat1} or {cat2} not found")
        return data
    
    # Create interaction: 1 only when both are 1
    data[output_col] = data[cat1] * data[cat2]
    
    logger.info(f"Created interaction {output_col} = {cat1} × {cat2}")
    return data


def preprocess_for_regression(
    data: h2o.H2OFrame,
    add_polynomial: bool = True,
    log_transform: bool = True
) -> h2o.H2OFrame:
    """Complete preprocessing pipeline for regression analysis.
    
    Applies all standard preprocessing steps in order:
    1. Remove missing data
    2. Standardize column names
    3. Add fungi richness
    4. Standardize climate variables (2SD)
    5. Add polynomial terms (optional)
    6. Log-transform biodiversity predictors (optional)
    
    Args:
        data: H2O frame to preprocess
        add_polynomial: Whether to add polynomial terms
        log_transform: Whether to log-transform biodiversity predictors
        
    Returns:
        Preprocessed H2O frame ready for modeling
    """
    logger.info("Starting H2O preprocessing pipeline")
    initial_shape = (data.nrows, data.ncols)
    
    # Step 1: Remove missing data
    data = remove_missing(data)
    
    # Step 2: Standardize column names
    data = standardize_column_names(data)
    
    # Step 3: Add fungi richness
    data = add_fungi_richness(data)
    
    # Step 4: Standardize climate variables
    data = standardize_climate_2sd(data)
    
    # Step 5: Add polynomial terms
    if add_polynomial:
        data = add_elevation_squared(data)
    
    # Step 6: Log-transform biodiversity predictors
    if log_transform:
        data = log_transform_biodiversity(data)
    
    final_shape = (data.nrows, data.ncols)
    logger.info(f"Preprocessing complete: {initial_shape} → {final_shape}")
    
    return data