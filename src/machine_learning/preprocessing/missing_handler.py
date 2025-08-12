"""
Missing data handler for ML preprocessing.

Handles complete case analysis and other missing data strategies.
"""

import logging
import h2o
from typing import Union, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def remove_missing(
    data: Union[pd.DataFrame, h2o.H2OFrame],
    columns: Optional[List[str]] = None,
    how: str = 'any'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Remove rows with missing values (complete case analysis).
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        columns: Columns to check for missing values (None = all columns)
        how: 'any' = remove if any column is missing, 'all' = remove if all columns are missing
        
    Returns:
        Data with missing values removed
    """
    initial_rows = len(data) if isinstance(data, pd.DataFrame) else data.nrows
    
    if isinstance(data, pd.DataFrame):
        if columns:
            result = data.dropna(subset=columns, how=how).copy()
        else:
            result = data.dropna(how=how).copy()
    else:  # H2O Frame
        if columns is None:
            columns = data.columns
        
        # H2O approach: filter out rows with NAs
        result = data
        for col in columns:
            if col in data.columns:
                result = result[result[col].isna() == False]
    
    final_rows = len(result) if isinstance(result, pd.DataFrame) else result.nrows
    rows_dropped = initial_rows - final_rows
    
    logger.info(f"Complete case analysis: dropped {rows_dropped:,} rows "
                f"({rows_dropped/initial_rows*100:.2f}%)")
    
    return result


def get_required_columns() -> List[str]:
    """
    Get list of required columns for biodiversity analysis.
    
    Returns:
        List of column names that must be non-missing
    """
    return [
        'fungi_richness', 'plants_richness', 'vertebrates_richness',
        'bio1', 'bio4', 'bio12', 'bio15', 'elevation',
        'longitude', 'latitude'
    ]


def check_missing_patterns(data: pd.DataFrame) -> dict:
    """
    Analyze missing data patterns.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary with missing data statistics
    """
    missing_stats = {}
    
    # Overall missing
    total_missing = data.isnull().sum().sum()
    total_cells = data.shape[0] * data.shape[1]
    missing_stats['overall_missing_pct'] = (total_missing / total_cells) * 100
    
    # Per column missing
    missing_by_col = data.isnull().sum()
    missing_pct_by_col = (missing_by_col / len(data)) * 100
    missing_stats['columns'] = missing_pct_by_col.to_dict()
    
    # Rows with any missing
    rows_with_missing = data.isnull().any(axis=1).sum()
    missing_stats['rows_with_missing'] = rows_with_missing
    missing_stats['rows_with_missing_pct'] = (rows_with_missing / len(data)) * 100
    
    # Complete rows
    complete_rows = (~data.isnull().any(axis=1)).sum()
    missing_stats['complete_rows'] = complete_rows
    missing_stats['complete_rows_pct'] = (complete_rows / len(data)) * 100
    
    logger.info(f"Missing data analysis: {missing_stats['overall_missing_pct']:.2f}% missing overall, "
                f"{missing_stats['complete_rows_pct']:.2f}% complete rows")
    
    return missing_stats