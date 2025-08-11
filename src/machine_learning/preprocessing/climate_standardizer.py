"""
Climate variable standardization for ML preprocessing.

Standardizes climate variables using 2SD method (divide by 2 standard deviations).
"""

import logging
import h2o
from typing import Union, List, Optional, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def standardize_climate_2sd(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    climate_vars: Optional[List[str]] = None,
    suffix: str = '_std'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Standardize climate variables using 2SD method.
    
    The 2SD method (Gelman, 2008) divides by 2 standard deviations to make
    coefficients comparable to binary predictors.
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        climate_vars: Variables to standardize (None = use defaults)
        suffix: Suffix to add to standardized column names
        
    Returns:
        Data with standardized climate variables added
    """
    if climate_vars is None:
        climate_vars = ['bio1', 'bio4', 'bio12', 'bio15', 'elevation']
    
    logger.info(f"Standardizing climate variables (2SD method): {climate_vars}")
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for var in climate_vars:
            if var in data.columns:
                mean_val = data[var].mean()
                std_val = data[var].std()
                result[f'{var}{suffix}'] = (data[var] - mean_val) / (2 * std_val)
                logger.debug(f"  {var}: mean={mean_val:.2f}, std={std_val:.2f}")
    else:  # H2O Frame
        result = data
        for var in climate_vars:
            if var in data.columns:
                mean_val = data[var].mean()
                std_val = data[var].std()
                # H2O returns lists, extract scalar
                if isinstance(mean_val, list):
                    mean_val = mean_val[0]
                if isinstance(std_val, list):
                    std_val = std_val[0]
                
                result[f'{var}{suffix}'] = (data[var] - mean_val) / (2 * std_val)
                logger.debug(f"  {var}: mean={mean_val:.2f}, std={std_val:.2f}")
    
    return result


def standardize_climate_zscore(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    climate_vars: Optional[List[str]] = None,
    suffix: str = '_z'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Standardize climate variables using z-score (mean=0, std=1).
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        climate_vars: Variables to standardize (None = use defaults)
        suffix: Suffix to add to standardized column names
        
    Returns:
        Data with z-score standardized climate variables
    """
    if climate_vars is None:
        climate_vars = ['bio1', 'bio4', 'bio12', 'bio15', 'elevation']
    
    logger.info(f"Standardizing climate variables (z-score): {climate_vars}")
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for var in climate_vars:
            if var in data.columns:
                result[f'{var}{suffix}'] = (data[var] - data[var].mean()) / data[var].std()
    else:  # H2O Frame
        result = data
        for var in climate_vars:
            if var in data.columns:
                mean_val = data[var].mean()
                std_val = data[var].std()
                # H2O returns lists, extract scalar
                if isinstance(mean_val, list):
                    mean_val = mean_val[0]
                if isinstance(std_val, list):
                    std_val = std_val[0]
                    
                result[f'{var}{suffix}'] = (data[var] - mean_val) / std_val
    
    return result


def get_standardization_stats(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate standardization statistics for variables.
    
    Args:
        data: Input DataFrame
        variables: Variables to analyze (None = climate defaults)
        
    Returns:
        Dictionary with mean, std, min, max for each variable
    """
    if variables is None:
        variables = ['bio1', 'bio4', 'bio12', 'bio15', 'elevation']
    
    stats = {}
    for var in variables:
        if var in data.columns:
            stats[var] = {
                'mean': data[var].mean(),
                'std': data[var].std(),
                'min': data[var].min(),
                'max': data[var].max(),
                '2sd': 2 * data[var].std()
            }
    
    return stats