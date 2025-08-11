"""
Log transformation for biodiversity variables.

Applies log transformations to handle skewed distributions in count data.
"""

import logging
import h2o
from typing import Union, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def log_transform_biodiversity(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    variables: Optional[List[str]] = None,
    offset: float = 1.0,
    suffix: str = '_log'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Apply log transformation to biodiversity count variables.
    
    Uses log(x + offset) to handle zeros in count data.
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        variables: Variables to transform (None = use defaults)
        offset: Offset to add before log (handles zeros)
        suffix: Suffix to add to transformed column names
        
    Returns:
        Data with log-transformed variables added
    """
    if variables is None:
        variables = ['plants_richness', 'vertebrates_richness']
    
    logger.info(f"Log-transforming biodiversity variables: {variables}")
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for var in variables:
            if var in data.columns:
                result[f'{var}{suffix}'] = np.log(data[var] + offset)
                logger.debug(f"  {var}: log({var} + {offset})")
    else:  # H2O Frame
        result = data
        for var in variables:
            if var in data.columns:
                result[f'{var}{suffix}'] = (data[var] + offset).log()
                logger.debug(f"  {var}: log({var} + {offset})")
    
    logger.info(f"Created log-transformed predictors with offset={offset}")
    return result


def sqrt_transform(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    variables: List[str],
    suffix: str = '_sqrt'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Apply square root transformation to variables.
    
    Useful for count data with moderate skewness.
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        variables: Variables to transform
        suffix: Suffix to add to transformed column names
        
    Returns:
        Data with sqrt-transformed variables
    """
    logger.info(f"Square root transforming: {variables}")
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for var in variables:
            if var in data.columns:
                result[f'{var}{suffix}'] = np.sqrt(data[var])
    else:  # H2O Frame
        result = data
        for var in variables:
            if var in data.columns:
                result[f'{var}{suffix}'] = data[var].sqrt()
    
    return result


def box_cox_transform(
    data: pd.DataFrame,
    variables: List[str],
    lambda_param: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply Box-Cox transformation (pandas only).
    
    Args:
        data: Input DataFrame
        variables: Variables to transform
        lambda_param: Lambda parameter (None = estimate from data)
        
    Returns:
        DataFrame with Box-Cox transformed variables
    """
    from scipy import stats
    
    result = data.copy()
    
    for var in variables:
        if var in data.columns:
            # Remove any non-positive values
            positive_data = data[var][data[var] > 0]
            
            if len(positive_data) > 0:
                if lambda_param is None:
                    # Estimate optimal lambda
                    transformed, fitted_lambda = stats.boxcox(positive_data)
                    logger.info(f"Box-Cox for {var}: fitted λ={fitted_lambda:.3f}")
                else:
                    # Use specified lambda
                    if lambda_param == 0:
                        transformed = np.log(positive_data)
                    else:
                        transformed = (positive_data ** lambda_param - 1) / lambda_param
                
                # Apply transformation to full column
                result[f'{var}_boxcox'] = data[var].apply(
                    lambda x: transformed[0] if x > 0 else np.nan
                )
    
    return result


def inverse_transform(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    variables: List[str],
    suffix: str = '_inv'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Apply inverse transformation (1/x) to variables.
    
    Args:
        data: Input data
        variables: Variables to transform
        suffix: Suffix for transformed columns
        
    Returns:
        Data with inverse-transformed variables
    """
    logger.info(f"Inverse transforming: {variables}")
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for var in variables:
            if var in data.columns:
                # Avoid division by zero
                result[f'{var}{suffix}'] = 1.0 / (data[var] + 1e-10)
    else:  # H2O Frame
        result = data
        for var in variables:
            if var in data.columns:
                result[f'{var}{suffix}'] = 1.0 / (data[var] + 1e-10)
    
    return result


def check_skewness(data: pd.DataFrame, variables: Optional[List[str]] = None) -> dict:
    """
    Check skewness of variables to determine if transformation is needed.
    
    Args:
        data: Input DataFrame
        variables: Variables to check (None = all numeric)
        
    Returns:
        Dictionary with skewness statistics
    """
    from scipy import stats
    
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    
    skewness_stats = {}
    for var in variables:
        if var in data.columns:
            skew_val = stats.skew(data[var].dropna())
            skewness_stats[var] = {
                'skewness': skew_val,
                'interpretation': (
                    'highly left-skewed' if skew_val < -1 else
                    'moderately left-skewed' if skew_val < -0.5 else
                    'approximately symmetric' if abs(skew_val) < 0.5 else
                    'moderately right-skewed' if skew_val < 1 else
                    'highly right-skewed'
                ),
                'transform_suggested': 'log' if skew_val > 1 else 'none' if abs(skew_val) < 0.5 else 'sqrt'
            }
    
    return skewness_stats