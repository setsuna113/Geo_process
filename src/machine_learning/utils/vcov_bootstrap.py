"""
Bootstrap variance-covariance matrix estimation.

LINUS: One thing, done right. No fallbacks, no compromises.
"""

import numpy as np
import h2o
from h2o.grid import H2OGridSearch
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import logging

logger = logging.getLogger(__name__)


def compute_bootstrap_vcov(h2o_model, data, x_names, y_name, n_bootstrap=100, seed=42):
    """
    Compute variance-covariance matrix via bootstrap.
    
    Clean H2O Grid Search implementation for parallel bootstrap.
    No fallbacks, no diagonal approximations - pure bootstrap.
    
    Args:
        h2o_model: Fitted H2O GLM model
        data: H2O DataFrame with training data
        x_names: List of predictor column names
        y_name: Target column name
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility
        
    Returns:
        Full variance-covariance matrix from bootstrap
    """
    logger.info(f"Bootstrap: {data.nrows:,} rows, {n_bootstrap} iterations")
    
    # Extract model parameters for refitting
    model_params = {
        'family': h2o_model.actual_params.get('family', 'negativebinomial'),
        'link': h2o_model.actual_params.get('link', 'log'),
        'lambda_': h2o_model.actual_params.get('lambda', 0),
        'alpha': h2o_model.actual_params.get('alpha', 0),
        'standardize': h2o_model.actual_params.get('standardize', False),
        'compute_p_values': False,
        'max_iterations': h2o_model.actual_params.get('max_iterations', 50)
    }
    
    # Get coefficient names from original model
    coef_dict = h2o_model.coef()
    coef_names = list(coef_dict.keys())
    n_coef = len(coef_names)
    
    # For large data, subsample for bootstrap
    n_rows = data.nrows
    if n_rows > 500000:
        logger.info(f"Subsampling 500k from {n_rows:,} rows for bootstrap")
        sample_frac = 500000 / n_rows
        splits = data.split_frame(ratios=[sample_frac], seed=seed)
        work_data = splits[0]
        n_rows = work_data.nrows
    else:
        work_data = data
    
    logger.info("Generating bootstrap weight columns...")
    np.random.seed(seed)
    
    # Generate all bootstrap weights efficiently
    weight_cols = []
    for i in range(n_bootstrap):
        # Poisson(1) weights = bootstrap sampling
        weights = np.random.poisson(1, n_rows).astype(np.float32)
        weight_frame = h2o.H2OFrame(weights.reshape(-1, 1).tolist())
        col_name = f'boot_w_{i}'
        weight_frame.set_names([col_name])
        work_data = work_data.cbind(weight_frame)
        weight_cols.append(col_name)
        
        if (i + 1) % 25 == 0:
            logger.info(f"Generated {i + 1}/{n_bootstrap} weight columns")
    
    # Set up Grid Search for parallel bootstrap
    logger.info(f"Running H2O Grid Search with {n_bootstrap} models in parallel...")
    
    hyper_params = {
        'weights_column': weight_cols
    }
    
    grid_model = H2OGeneralizedLinearEstimator(**model_params)
    
    grid = H2OGridSearch(
        model=grid_model,
        hyper_params=hyper_params
    )
    
    # Train all bootstrap models in parallel
    grid.train(
        x=x_names,
        y=y_name,
        training_frame=work_data
    )
    
    logger.info(f"Grid complete: {len(grid.models)} models fitted")
    
    # Extract coefficients from all models
    bootstrap_coefs = np.zeros((n_bootstrap, n_coef))
    
    for i, model in enumerate(grid.models):
        model_coef_dict = model.coef()
        bootstrap_coefs[i, :] = [model_coef_dict.get(name, 0.0) for name in coef_names]
        
        # Clean up model immediately
        h2o.remove(model)
        
        if (i + 1) % 25 == 0:
            logger.info(f"Extracted {i + 1}/{n_bootstrap} coefficient sets")
    
    # Compute empirical covariance matrix
    vcov = np.cov(bootstrap_coefs.T)
    
    logger.info(f"Bootstrap vcov computed: shape {vcov.shape}")
    
    return vcov