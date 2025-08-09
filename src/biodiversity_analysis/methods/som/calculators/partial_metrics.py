"""Validation utilities for missing data handling in SOM.

This module provides validation functions for assessing data quality
with respect to missing values. The actual distance calculations are
handled by StreamlinedDistanceCalculator.
"""

import numpy as np
from typing import Optional
from ..utils.constants import INVALID_DISTANCE, INVALID_INDEX




def validate_missing_data_handling(data: np.ndarray, threshold: float = 0.7) -> dict:
    """Validate data for missing value handling.
    
    Args:
        data: Input data array
        threshold: Maximum allowed proportion of missing values
        
    Returns:
        Dictionary with validation results
    """
    n_samples, n_features = data.shape
    
    # Overall missing proportion
    missing_mask = np.isnan(data)
    overall_missing = missing_mask.sum() / data.size
    
    # Per-feature missing
    feature_missing = missing_mask.sum(axis=0) / n_samples
    
    # Per-sample missing
    sample_missing = missing_mask.sum(axis=1) / n_features
    
    # Find problematic features/samples
    problematic_features = np.where(feature_missing > threshold)[0]
    problematic_samples = np.where(sample_missing > threshold)[0]
    
    # Estimate average valid pairs without computing full matrix
    # Sample a small subset to estimate statistics
    sample_size = min(100, n_samples)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    avg_valid_pairs = 0
    
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            valid = ~(missing_mask[sample_indices[i]] | missing_mask[sample_indices[j]])
            avg_valid_pairs += valid.sum()
    
    if sample_size > 1:
        avg_valid_pairs = avg_valid_pairs / (sample_size * (sample_size - 1) / 2)
    else:
        avg_valid_pairs = n_features * (1 - overall_missing)
    
    return {
        'overall_missing_proportion': overall_missing,
        'feature_missing_proportions': feature_missing,
        'sample_missing_proportions': sample_missing,
        'problematic_features': problematic_features,
        'problematic_samples': problematic_samples,
        'estimated_avg_valid_pairs': avg_valid_pairs,
        'min_features': n_features,
        'passed_validation': overall_missing <= threshold
    }