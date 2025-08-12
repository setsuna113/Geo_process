"""
Polynomial feature engineering for ML preprocessing.

Creates polynomial terms like squared elevation.
"""

import logging
import h2o
from typing import Union, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def add_elevation_squared(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    elevation_col: str = 'elevation_std',
    output_col: str = 'elevation_std_sq'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Add squared elevation term for non-linear relationships.
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        elevation_col: Name of elevation column to square
        output_col: Name for squared elevation column
        
    Returns:
        Data with squared elevation added
    """
    if elevation_col not in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
        logger.warning(f"Column {elevation_col} not found, skipping elevation squared")
        return data
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        result[output_col] = data[elevation_col] ** 2
    else:  # H2O Frame
        data[output_col] = data[elevation_col] ** 2
        result = data
    
    logger.info(f"Created quadratic elevation term: {output_col}")
    return result


def add_polynomial_terms(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    variables: List[str],
    degree: int = 2,
    interaction_only: bool = False
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Add polynomial features for specified variables.
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        variables: Variables to create polynomial features for
        degree: Maximum polynomial degree
        interaction_only: If True, only create interaction terms
        
    Returns:
        Data with polynomial features added
    """
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        
        # Add squared terms if degree >= 2 and not interaction_only
        if degree >= 2 and not interaction_only:
            for var in variables:
                if var in data.columns:
                    result[f'{var}_sq'] = data[var] ** 2
                    logger.debug(f"Added {var}_sq")
        
        # Add interaction terms if degree >= 2
        if degree >= 2:
            for i, var1 in enumerate(variables):
                for var2 in variables[i+1:]:
                    if var1 in data.columns and var2 in data.columns:
                        result[f'{var1}_x_{var2}'] = data[var1] * data[var2]
                        logger.debug(f"Added interaction: {var1}_x_{var2}")
        
        # Add cubic terms if degree >= 3 and not interaction_only
        if degree >= 3 and not interaction_only:
            for var in variables:
                if var in data.columns:
                    result[f'{var}_cu'] = data[var] ** 3
                    logger.debug(f"Added {var}_cu")
                    
    else:  # H2O Frame
        result = data
        
        # Add squared terms
        if degree >= 2 and not interaction_only:
            for var in variables:
                if var in data.columns:
                    result[f'{var}_sq'] = data[var] ** 2
        
        # Add interaction terms
        if degree >= 2:
            for i, var1 in enumerate(variables):
                for var2 in variables[i+1:]:
                    if var1 in data.columns and var2 in data.columns:
                        result[f'{var1}_x_{var2}'] = data[var1] * data[var2]
        
        # Add cubic terms
        if degree >= 3 and not interaction_only:
            for var in variables:
                if var in data.columns:
                    result[f'{var}_cu'] = data[var] ** 3
    
    logger.info(f"Added polynomial features (degree={degree}) for {len(variables)} variables")
    return result


def add_spatial_polynomial(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Add polynomial terms for spatial coordinates.
    
    Creates lat^2, lon^2, lat*lon for capturing spatial patterns.
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        
    Returns:
        Data with spatial polynomial features
    """
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        if lat_col in data.columns and lon_col in data.columns:
            result[f'{lat_col}_sq'] = data[lat_col] ** 2
            result[f'{lon_col}_sq'] = data[lon_col] ** 2
            result[f'{lat_col}_{lon_col}'] = data[lat_col] * data[lon_col]
    else:  # H2O Frame
        result = data
        if lat_col in data.columns and lon_col in data.columns:
            result[f'{lat_col}_sq'] = data[lat_col] ** 2
            result[f'{lon_col}_sq'] = data[lon_col] ** 2
            result[f'{lat_col}_{lon_col}'] = data[lat_col] * data[lon_col]
    
    logger.info("Added spatial polynomial features")
    return result