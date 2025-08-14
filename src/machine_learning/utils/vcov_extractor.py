"""
Variance-Covariance Matrix Extractor for H2O GLM Models.

This module provides methods to extract or compute the full variance-covariance matrix
from H2O GLM models, which only provide diagonal elements (standard errors) by default.

The primary method is bootstrap, which empirically estimates the full vcov matrix
including all covariances between parameters.
"""

import numpy as np
import h2o
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_vcov_from_h2o(
    h2o_model,
    data: h2o.H2OFrame,
    method: str = 'bootstrap',
    n_bootstrap: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract or compute full variance-covariance matrix from H2O GLM model.
    
    Args:
        h2o_model: Fitted H2O GLM model
        data: Original training data (required for bootstrap)
        method: 'diagonal' (from H2O) or 'bootstrap' (full matrix)
        n_bootstrap: Number of bootstrap iterations if method='bootstrap'
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (coefficients, vcov_matrix, coefficient_names)
    """
    # Get coefficient names and values
    coef_dict = h2o_model.coef()
    coef_names = list(coef_dict.keys())
    coef_values = np.array(list(coef_dict.values()))
    
    if method == 'diagonal':
        vcov = _get_diagonal_vcov(h2o_model, len(coef_values))
    elif method == 'bootstrap':
        vcov = _bootstrap_vcov(h2o_model, data, coef_names, n_bootstrap, seed)
    else:
        raise ValueError(f"Method must be 'diagonal' or 'bootstrap', got {method}")
    
    return coef_values, vcov, coef_names


def _get_diagonal_vcov(h2o_model, n_coef: int) -> np.ndarray:
    """
    Get diagonal-only vcov matrix from H2O model.
    
    Args:
        h2o_model: Fitted H2O GLM model
        n_coef: Number of coefficients
        
    Returns:
        Diagonal vcov matrix (off-diagonal elements are zero)
    """
    try:
        # Try to get standard errors if model was fitted with compute_p_values=True
        coef_table = h2o_model.coef_with_p_values()
        if 'std_error' in coef_table.columns:
            std_errors = coef_table['std_error'].values
            vcov = np.diag(std_errors ** 2)
            logger.info("Using standard errors from H2O model (diagonal vcov only)")
        else:
            raise ValueError("No standard errors available")
    except:
        logger.warning("No standard errors available from H2O model. "
                      "Model must be fitted with compute_p_values=True and lambda=0. "
                      "Returning identity matrix scaled by residual deviance.")
        # Fallback: scaled identity matrix
        residual_deviance = h2o_model.residual_deviance()
        df_residual = h2o_model.residual_degrees_of_freedom()
        if df_residual > 0:
            scale = residual_deviance / df_residual
        else:
            scale = 1.0
        vcov = np.eye(n_coef) * scale
    
    return vcov


def _bootstrap_vcov(
    h2o_model,
    data: h2o.H2OFrame,
    coef_names: list,
    n_bootstrap: int,
    seed: int
) -> np.ndarray:
    """
    Compute full vcov matrix via bootstrap.
    
    Args:
        h2o_model: Fitted H2O GLM model
        data: Training data
        coef_names: List of coefficient names
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed
        
    Returns:
        Full variance-covariance matrix with all covariances
    """
    logger.info(f"Starting bootstrap vcov computation with {n_bootstrap} iterations")
    
    n_coef = len(coef_names)
    bootstrap_coefs = np.zeros((n_bootstrap, n_coef))
    
    # Extract model parameters for refitting
    model_params = _extract_minimal_model_params(h2o_model)
    
    # Get predictor and response names
    # Check different ways H2O might store these
    x_names = None
    y_name = None
    
    # Try actual_params first
    if hasattr(h2o_model, 'actual_params'):
        x_names = h2o_model.actual_params.get('x')
        y_name = h2o_model.actual_params.get('y')
    
    # Try _parms if actual_params didn't work
    if x_names is None and hasattr(h2o_model, '_parms'):
        x_names = h2o_model._parms.get('x')
        y_name = h2o_model._parms.get('y')
    
    # Try model JSON
    if x_names is None and hasattr(h2o_model, '_model_json'):
        try:
            model_json = h2o_model._model_json
            if 'output' in model_json:
                # Get column names from the model
                if 'coefficients_table' in model_json['output']:
                    coef_table = model_json['output']['coefficients_table']
                    if hasattr(coef_table, 'cell_values'):
                        # Extract non-intercept coefficient names
                        x_names = [row[0] for row in coef_table.cell_values if row[0] != 'Intercept']
                if 'response_column_name' in model_json['output']:
                    y_name = model_json['output']['response_column_name']
        except Exception as e:
            logger.debug(f"Could not extract from model JSON: {e}")
    
    # Final fallback: use coefficient names
    if x_names is None:
        x_names = [name for name in coef_names if name != 'Intercept']
        y_name = 'fungi_richness'  # Default assumption
        logger.warning(f"Could not find x/y in model params, using coefficient names: {len(x_names)} predictors")
    
    for i in range(n_bootstrap):
        # Bootstrap sample - H2O doesn't have sample(), use split_frame
        # Create bootstrap sample by random sampling with replacement
        n_rows = data.nrows
        # Create random indices with replacement
        import random
        random.seed(seed + i)
        indices = [random.randint(0, n_rows - 1) for _ in range(n_rows)]
        boot_data = data[indices, :]
        
        # Refit model with same parameters
        boot_model = h2o.estimators.H2OGeneralizedLinearEstimator(**model_params)
        
        # Train on bootstrap sample
        boot_model.train(
            x=x_names,
            y=y_name,
            training_frame=boot_data
        )
        
        # Extract coefficients in same order
        boot_coef_dict = boot_model.coef()
        bootstrap_coefs[i, :] = [boot_coef_dict.get(name, 0.0) for name in coef_names]
        
        # Clean up to save memory
        h2o.remove(boot_model)
        h2o.remove(boot_data)
        
        if (i + 1) % 20 == 0:
            logger.info(f"Bootstrap iteration {i + 1}/{n_bootstrap}")
    
    # Compute empirical covariance matrix
    vcov_full = np.cov(bootstrap_coefs.T)
    
    logger.info(f"Bootstrap complete. Full vcov matrix shape: {vcov_full.shape}")
    
    return vcov_full


def _extract_minimal_model_params(h2o_model) -> Dict[str, Any]:
    """
    Extract minimal parameters needed to refit model.
    
    Args:
        h2o_model: H2O GLM model
        
    Returns:
        Dictionary of model parameters
    """
    # Get essential parameters only - handle missing actual_params
    if hasattr(h2o_model, 'actual_params'):
        actual_params = h2o_model.actual_params
    elif hasattr(h2o_model, 'params'):
        actual_params = h2o_model.params  
    elif hasattr(h2o_model, '_parms'):
        actual_params = h2o_model._parms
    else:
        # Default to negative binomial
        actual_params = {}
        logger.warning("Could not find model parameters, using defaults")
    
    params = {
        'family': actual_params.get('family', 'negativebinomial'),
        'link': actual_params.get('link', 'log'),
        'lambda_': actual_params.get('lambda', 0),
        'alpha': actual_params.get('alpha', 0),
        'solver': actual_params.get('solver', 'IRLSM'),
        'standardize': actual_params.get('standardize', False),
        'max_iterations': 20,  # Limit iterations for speed
        'compute_p_values': False,  # Not needed for bootstrap
        'remove_collinear_columns': True
    }
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    return params


def get_vcov_for_emmeans(
    h2o_model,
    data: Optional[h2o.H2OFrame] = None,
    method: str = 'auto'
) -> Dict[str, Any]:
    """
    Get vcov matrix formatted for use with emmeans calculations.
    
    Args:
        h2o_model: Fitted H2O GLM model
        data: Training data (required if method='bootstrap' or 'auto' with no std errors)
        method: 'auto', 'diagonal', or 'bootstrap'
                'auto' uses bootstrap if data provided and no std errors available
        
    Returns:
        Dictionary with coefficients, vcov, and names ready for emmeans
    """
    # Determine method
    if method == 'auto':
        try:
            coef_table = h2o_model.coef_with_p_values()
            has_std_errors = 'std_error' in coef_table.columns
        except:
            has_std_errors = False
        
        if not has_std_errors and data is not None:
            method = 'bootstrap'
            logger.info("Auto-selected bootstrap method (no std errors, data available)")
        else:
            method = 'diagonal'
            logger.info("Auto-selected diagonal method")
    
    # Extract vcov
    if method == 'bootstrap':
        if data is None:
            raise ValueError("Bootstrap method requires training data")
        coef, vcov, names = extract_vcov_from_h2o(h2o_model, data, 'bootstrap')
    else:
        coef, vcov, names = extract_vcov_from_h2o(h2o_model, data, 'diagonal')
    
    return {
        'coefficients': coef,
        'vcov_matrix': vcov,
        'coefficient_names': names,
        'method_used': method,
        'has_full_covariances': method == 'bootstrap'
    }


def validate_vcov_matrix(vcov: np.ndarray) -> Dict[str, Any]:
    """
    Validate and diagnose vcov matrix properties.
    
    Args:
        vcov: Variance-covariance matrix
        
    Returns:
        Dictionary with validation results and diagnostics
    """
    n = vcov.shape[0]
    
    # Check symmetry
    is_symmetric = np.allclose(vcov, vcov.T)
    
    # Check positive semi-definite
    eigenvalues = np.linalg.eigvals(vcov)
    is_positive_semidefinite = np.all(eigenvalues >= -1e-10)
    
    # Check if diagonal (no covariances)
    is_diagonal = np.allclose(vcov, np.diag(np.diag(vcov)))
    
    # Compute condition number
    condition_number = np.linalg.cond(vcov)
    
    # Extract correlation matrix
    D = np.sqrt(np.diag(vcov))
    D_inv = np.diag(1.0 / D)
    correlation_matrix = D_inv @ vcov @ D_inv
    
    return {
        'shape': vcov.shape,
        'is_symmetric': is_symmetric,
        'is_positive_semidefinite': is_positive_semidefinite,
        'is_diagonal_only': is_diagonal,
        'condition_number': condition_number,
        'min_eigenvalue': np.min(eigenvalues),
        'max_eigenvalue': np.max(eigenvalues),
        'has_full_covariances': not is_diagonal,
        'max_correlation': np.max(np.abs(correlation_matrix[np.triu_indices(n, k=1)])) if not is_diagonal else 0.0
    }