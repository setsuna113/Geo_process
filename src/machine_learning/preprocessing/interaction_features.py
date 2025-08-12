"""
Interaction feature engineering for hypothesis testing.

Creates interaction terms for testing biodiversity-climate relationships.
"""

import logging
import h2o
from typing import Union, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def add_continuous_interaction(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    var1: str,
    var2: str,
    output_col: Optional[str] = None
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Create continuous interaction term between two variables.
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        var1: First variable name
        var2: Second variable name
        output_col: Name for interaction column (None = auto-generate)
        
    Returns:
        Data with interaction term added
    """
    if output_col is None:
        output_col = f'{var1.replace("_log", "").replace("_std", "")}_{var2.replace("_std", "")}_interaction'
    
    if var1 not in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
        logger.error(f"Variable {var1} not found in data")
        return data
    
    if var2 not in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
        logger.error(f"Variable {var2} not found in data")
        return data
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        result[output_col] = data[var1] * data[var2]
    else:  # H2O Frame
        data[output_col] = data[var1] * data[var2]
        result = data
    
    logger.info(f"Created interaction term: {var1} × {var2} → {output_col}")
    return result


def add_plants_bio4_interaction(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    plants_col: str = 'plants_richness_log',
    bio4_col: str = 'bio4_std',
    output_col: str = 'plants_bio4_interaction'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Create plants × temperature seasonality interaction for H2 hypothesis.
    
    This tests whether the plant-fungi relationship varies with temperature seasonality.
    
    Args:
        data: Input data
        plants_col: Name of plants richness column (usually log-transformed)
        bio4_col: Name of bio4 (temperature seasonality) column (usually standardized)
        output_col: Name for interaction term
        
    Returns:
        Data with plants_bio4_interaction added
    """
    # Try log version first, fall back to raw if not available
    if plants_col not in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
        if 'plants_richness' in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
            plants_col = 'plants_richness'
            logger.warning(f"Using raw plants_richness instead of {plants_col}")
        else:
            logger.error("No plants richness column found")
            return data
    
    return add_continuous_interaction(data, plants_col, bio4_col, output_col)


def add_three_way_interaction(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    var1: str,
    var2: str,
    var3: str,
    output_col: Optional[str] = None
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Create three-way interaction term.
    
    Args:
        data: Input data
        var1, var2, var3: Variable names
        output_col: Name for interaction (None = auto-generate)
        
    Returns:
        Data with three-way interaction added
    """
    if output_col is None:
        output_col = f'{var1}_{var2}_{var3}_interaction'
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        result[output_col] = data[var1] * data[var2] * data[var3]
    else:  # H2O Frame
        data[output_col] = data[var1] * data[var2] * data[var3]
        result = data
    
    logger.info(f"Created three-way interaction: {var1} × {var2} × {var3}")
    return result


def add_all_pairwise_interactions(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    variables: List[str],
    include_squared: bool = False
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Create all pairwise interactions between specified variables.
    
    Args:
        data: Input data
        variables: List of variables to create interactions for
        include_squared: If True, also include squared terms
        
    Returns:
        Data with all pairwise interactions
    """
    count = 0
    result = data
    
    # Add squared terms if requested
    if include_squared:
        for var in variables:
            if var in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
                if isinstance(result, pd.DataFrame):
                    result = result.copy() if count == 0 else result
                    result[f'{var}_sq'] = result[var] ** 2
                else:
                    result[f'{var}_sq'] = result[var] ** 2
                count += 1
    
    # Add pairwise interactions
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1:]:
            if var1 in (data.columns if isinstance(data, pd.DataFrame) else data.columns) and \
               var2 in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
                interaction_name = f'{var1}_{var2}_interact'
                if isinstance(result, pd.DataFrame):
                    result = result.copy() if count == 0 else result
                    result[interaction_name] = result[var1] * result[var2]
                else:
                    result[interaction_name] = result[var1] * result[var2]
                count += 1
    
    logger.info(f"Created {count} interaction terms")
    return result