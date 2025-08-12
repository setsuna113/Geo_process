"""
Categorical feature engineering for biodiversity analysis.

Creates categorical variables for testing discrete hypotheses.
"""

import logging
import h2o
from typing import Union, List, Optional, Tuple, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def create_temperate_categories(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    bio4_col: str = 'bio4_std',  # Changed to bio4 (temperature seasonality)
    output_col: str = 'temperate',
    method: str = 'median',
    quantile: float = 0.75
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Create categorical variable for temperate vs non-temperate regions.
    Based on bio4 (temperature seasonality) - high seasonality = temperate.
    
    Args:
        data: Input data
        bio4_col: Name of temperature seasonality column (bio4_std)
        output_col: Name for categorical output
        method: 'median', 'quantile', or 'threshold'
        quantile: Quantile to use if method='quantile' (default 0.75)
        
    Returns:
        Data with temperate category added (1 = temperate, 0 = non-temperate)
    """
    if bio4_col not in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
        logger.error(f"Column {bio4_col} not found")
        return data
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        if method == 'median':
            threshold = data[bio4_col].median()
        elif method == 'quantile':
            threshold = data[bio4_col].quantile(quantile)
        else:
            threshold = 500  # Default for bio4 (temperature seasonality)
        
        # High seasonality = temperate (bio4 > threshold)
        result[output_col] = (data[bio4_col] > threshold).astype(int)
        
    else:  # H2O Frame
        if method == 'median':
            threshold = data[bio4_col].median()
            if isinstance(threshold, list):
                threshold = threshold[0]
        elif method == 'quantile':
            threshold = data[bio4_col].quantile([quantile])[0]
        else:
            threshold = 500
        
        # High seasonality = temperate (bio4 > threshold)
        data[output_col] = (data[bio4_col] > threshold).ifelse(1, 0)
        result = data
    
    logger.info(f"Created temperate category using {bio4_col} (threshold={threshold:.3f}, method={method})")
    return result


def create_biodiversity_categories(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    richness_col: str = 'plants_richness',
    output_col: str = 'plant_high',
    method: str = 'median',
    quantile: float = 0.75
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Create categorical variable for high vs low biodiversity.
    
    Args:
        data: Input data
        richness_col: Name of richness column
        output_col: Name for categorical output  
        method: 'median', 'quantile', 'quartile', or 'tertile'
        quantile: Quantile to use if method='quantile' (default 0.75)
        
    Returns:
        Data with biodiversity category added
    """
    if richness_col not in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
        logger.error(f"Column {richness_col} not found")
        return data
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        
        if method == 'median':
            threshold = data[richness_col].median()
            result[output_col] = (data[richness_col] > threshold).astype(int)
        
        elif method == 'quantile':
            threshold = data[richness_col].quantile(quantile)
            result[output_col] = (data[richness_col] > threshold).astype(int)
            
        elif method == 'quartile':
            q1 = data[richness_col].quantile(0.25)
            q3 = data[richness_col].quantile(0.75)
            result[f'{output_col}_low'] = (data[richness_col] <= q1).astype(int)
            result[f'{output_col}_mid'] = ((data[richness_col] > q1) & 
                                           (data[richness_col] <= q3)).astype(int)
            result[f'{output_col}_high'] = (data[richness_col] > q3).astype(int)
            
        elif method == 'tertile':
            t1 = data[richness_col].quantile(0.33)
            t2 = data[richness_col].quantile(0.67)
            result[f'{output_col}_low'] = (data[richness_col] <= t1).astype(int)
            result[f'{output_col}_mid'] = ((data[richness_col] > t1) & 
                                          (data[richness_col] <= t2)).astype(int)
            result[f'{output_col}_high'] = (data[richness_col] > t2).astype(int)
            
    else:  # H2O Frame
        if method == 'median':
            threshold = data[richness_col].median()
            if isinstance(threshold, list):
                threshold = threshold[0]
            data[output_col] = (data[richness_col] > threshold).ifelse(1, 0)
        
        elif method == 'quantile':
            threshold = data[richness_col].quantile([quantile])[0]
            data[output_col] = (data[richness_col] > threshold).ifelse(1, 0)
            
        elif method == 'quartile':
            quantiles = data[richness_col].quantile([0.25, 0.75])
            q1 = quantiles[0] if isinstance(quantiles[0], (int, float)) else quantiles[0][0]
            q3 = quantiles[1] if isinstance(quantiles[1], (int, float)) else quantiles[1][0]
            
            data[f'{output_col}_low'] = (data[richness_col] <= q1).ifelse(1, 0)
            data[f'{output_col}_mid'] = ((data[richness_col] > q1) & 
                                        (data[richness_col] <= q3)).ifelse(1, 0)
            data[f'{output_col}_high'] = (data[richness_col] > q3).ifelse(1, 0)
            
        result = data
    
    logger.info(f"Created biodiversity categories using {method} method")
    return result


def add_categorical_interactions(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    cat1: str = 'temperate',
    cat2: str = 'plant_high',
    output_col: str = 'temperate_x_plant_high'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Create interaction between two categorical variables.
    
    Args:
        data: Input data
        cat1: First categorical variable
        cat2: Second categorical variable
        output_col: Name for interaction term
        
    Returns:
        Data with categorical interaction added
    """
    if cat1 not in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
        logger.error(f"Column {cat1} not found")
        return data
    
    if cat2 not in (data.columns if isinstance(data, pd.DataFrame) else data.columns):
        logger.error(f"Column {cat2} not found")
        return data
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        result[output_col] = data[cat1] * data[cat2]
    else:  # H2O Frame
        data[output_col] = data[cat1] * data[cat2]
        result = data
    
    logger.info(f"Created categorical interaction: {cat1} × {cat2} → {output_col}")
    return result


def create_climate_zones(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    bio1_col: str = 'bio1',
    bio12_col: str = 'bio12',
    output_col: str = 'climate_zone'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Create climate zone categories based on temperature and precipitation.
    
    Zones:
    - 0: Cold & Dry
    - 1: Cold & Wet
    - 2: Warm & Dry
    - 3: Warm & Wet
    
    Args:
        data: Input data
        bio1_col: Mean annual temperature column
        bio12_col: Annual precipitation column
        output_col: Name for climate zone column
        
    Returns:
        Data with climate zone categories
    """
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        
        temp_median = data[bio1_col].median()
        precip_median = data[bio12_col].median()
        
        result[output_col] = 0  # Default: Cold & Dry
        result.loc[(data[bio1_col] <= temp_median) & 
                  (data[bio12_col] > precip_median), output_col] = 1  # Cold & Wet
        result.loc[(data[bio1_col] > temp_median) & 
                  (data[bio12_col] <= precip_median), output_col] = 2  # Warm & Dry
        result.loc[(data[bio1_col] > temp_median) & 
                  (data[bio12_col] > precip_median), output_col] = 3  # Warm & Wet
                  
    else:  # H2O Frame
        temp_median = data[bio1_col].median()
        precip_median = data[bio12_col].median()
        
        if isinstance(temp_median, list):
            temp_median = temp_median[0]
        if isinstance(precip_median, list):
            precip_median = precip_median[0]
        
        # Create zone indicators
        cold = (data[bio1_col] <= temp_median).ifelse(1, 0)
        wet = (data[bio12_col] > precip_median).ifelse(1, 0)
        
        # Combine into zones
        data[output_col] = cold * 0 + (1 - cold) * 2 + wet * 1
        result = data
    
    logger.info("Created climate zone categories")
    return result


def create_quantile_bins(
    data: pd.DataFrame,
    variable: str,
    n_bins: int = 4,
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create quantile-based bins for a continuous variable (pandas only).
    
    Args:
        data: Input DataFrame
        variable: Variable to bin
        n_bins: Number of bins
        labels: Labels for bins (None = auto-generate)
        
    Returns:
        DataFrame with binned variable
    """
    result = data.copy()
    
    if labels is None:
        labels = [f'{variable}_q{i+1}' for i in range(n_bins)]
    
    result[f'{variable}_binned'] = pd.qcut(
        data[variable],
        q=n_bins,
        labels=labels,
        duplicates='drop'
    )
    
    # Also create dummy variables
    dummies = pd.get_dummies(result[f'{variable}_binned'], prefix=f'{variable}_bin')
    result = pd.concat([result, dummies], axis=1)
    
    logger.info(f"Created {n_bins} quantile bins for {variable}")
    return result