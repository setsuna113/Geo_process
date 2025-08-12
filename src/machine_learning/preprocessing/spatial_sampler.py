"""
Spatial grid sampling for large datasets.

Samples every Nth point in spatial grid to reduce data size while maintaining spatial coverage.
"""

import logging
import h2o
from typing import Union, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def sample_spatial_grid(
    data: Union[pd.DataFrame, h2o.H2OFrame], 
    step: int = 1,
    lon_col: str = 'longitude',
    lat_col: str = 'latitude'
) -> Union[pd.DataFrame, h2o.H2OFrame]:
    """
    Sample every Nth point in spatial grid.
    
    Args:
        data: Input data (pandas DataFrame or H2O Frame)
        step: Take every Nth point (1 = no sampling)
        lon_col: Name of longitude column
        lat_col: Name of latitude column
        
    Returns:
        Sampled data in same format as input
    """
    if step <= 1:
        logger.info("No spatial sampling applied (step=1)")
        return data
    
    initial_rows = len(data) if isinstance(data, pd.DataFrame) else data.nrows
    
    if isinstance(data, pd.DataFrame):
        # Create spatial index
        data['spatial_idx'] = (
            data[lon_col].rank(method='dense').astype(int) * 1000 + 
            data[lat_col].rank(method='dense').astype(int)
        )
        
        # Sample every Nth point
        sampled = data[data['spatial_idx'] % step == 0].copy()
        sampled.drop('spatial_idx', axis=1, inplace=True)
        
    else:  # H2O Frame
        # Create spatial index in H2O
        data['spatial_idx'] = (
            data[lon_col].rank() * 1000 + 
            data[lat_col].rank()
        )
        
        # Sample every Nth point
        sampled = data[data['spatial_idx'] % step == 0]
        sampled = sampled.drop('spatial_idx')
    
    final_rows = len(sampled) if isinstance(sampled, pd.DataFrame) else sampled.nrows
    logger.info(f"Spatial sampling: {initial_rows:,} → {final_rows:,} rows (step={step})")
    
    return sampled


def sample_by_distance(
    data: pd.DataFrame,
    min_distance_km: float = 10,
    lon_col: str = 'longitude', 
    lat_col: str = 'latitude',
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Sample points ensuring minimum distance between samples.
    
    Args:
        data: Input DataFrame
        min_distance_km: Minimum distance between points in km
        lon_col: Name of longitude column
        lat_col: Name of latitude column
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    from scipy.spatial import cKDTree
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert lat/lon to approximate km
    coords_km = data[[lon_col, lat_col]].values.copy()
    coords_km[:, 0] *= 111  # lon to km (approximate)
    coords_km[:, 1] *= 111  # lat to km (approximate)
    
    # Build spatial tree
    tree = cKDTree(coords_km)
    
    # Greedy sampling
    selected = np.zeros(len(data), dtype=bool)
    indices = np.random.permutation(len(data))
    
    for idx in indices:
        if not selected[idx]:
            # Mark this point as selected
            selected[idx] = True
            
            # Find all points within min_distance
            nearby = tree.query_ball_point(coords_km[idx], min_distance_km)
            
            # Mark nearby points as excluded
            for near_idx in nearby:
                if near_idx != idx:
                    selected[near_idx] = False
    
    sampled = data[selected].copy()
    
    logger.info(f"Distance-based sampling: {len(data):,} → {len(sampled):,} rows "
                f"(min_distance={min_distance_km}km)")
    
    return sampled