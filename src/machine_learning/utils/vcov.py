"""
Main variance-covariance matrix interface.

LINUS: Clean delegation. Each method in its own file.
Default is sandwich - the industry standard.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_vcov(h2o_model, data, x_names, y_name, method='sandwich', **kwargs):
    """
    Extract variance-covariance matrix from H2O GLM model.
    
    Clean interface that delegates to specific implementations.
    
    Args:
        h2o_model: Fitted H2O GLM model
        data: H2O DataFrame with training data
        x_names: List of predictor column names
        y_name: Target column name
        method: One of 'sandwich', 'bootstrap', 'hac', 'clustered'
        **kwargs: Method-specific parameters
        
    Returns:
        Tuple of (coefficients, vcov_matrix, coefficient_names)
    """
    # Extract coefficients from model
    coef_dict = h2o_model.coef()
    coef_names = list(coef_dict.keys())
    coef_values = np.array(list(coef_dict.values()))
    
    # Validate inputs
    if not x_names or not y_name:
        raise ValueError("x_names and y_name are required for vcov computation")
    
    # Delegate to specific implementation
    if method == 'sandwich':
        # Use JAX-accelerated implementation for full dataset processing
        from .vcov_sandwich import compute_sandwich_vcov
        vcov = compute_sandwich_vcov(h2o_model, data, x_names, y_name)
        
    elif method == 'bootstrap':
        from .vcov_bootstrap import compute_bootstrap_vcov
        n_bootstrap = kwargs.get('n_bootstrap', 100)
        seed = kwargs.get('seed', 42)
        vcov = compute_bootstrap_vcov(h2o_model, data, x_names, y_name, n_bootstrap, seed)
        
    elif method == 'hac':
        from .vcov_sandwich import compute_hac_vcov
        max_lag = kwargs.get('max_lag', 10)
        vcov = compute_hac_vcov(h2o_model, data, x_names, y_name, max_lag)
        
    elif method == 'clustered':
        from .vcov_sandwich import compute_clustered_vcov
        cluster_var = kwargs.get('cluster_var')
        if not cluster_var:
            # Fall back to sandwich if no cluster variable
            logger.warning("No cluster_var provided, using sandwich estimator")
            from .vcov_sandwich import compute_sandwich_vcov
            vcov = compute_sandwich_vcov(h2o_model, data, x_names, y_name)
        else:
            vcov = compute_clustered_vcov(h2o_model, data, x_names, y_name, cluster_var)
            
    else:
        raise ValueError(
            f"Unknown vcov method: {method}. "
            f"Choose from: sandwich, bootstrap, hac, clustered"
        )
    
    logger.info(f"Extracted vcov using {method} method: shape {vcov.shape}")
    
    return coef_values, vcov, coef_names