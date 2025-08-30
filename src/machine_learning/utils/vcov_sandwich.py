from src.abstractions import get_logger
"""
JAX-accelerated sandwich estimator for robust variance-covariance matrix.

Uses full dataset (no subsampling) for proper statistical inference.
Leverages JAX's JIT compilation and optimized matrix operations.
"""

import numpy as np
import jax
import jax.numpy as jnp
import h2o
import logging
from typing import Tuple

logger = get_logger(__name__)

# Configure JAX for maximum performance
jax.config.update('jax_enable_x64', True)  # Use float64 for precision
jax.config.update('jax_platform_name', 'cpu')  # Explicit CPU backend

# Set JAX CPU threading based on config
import os
# Read from config or use default
n_threads = int(os.environ.get('H2O_NTHREADS', '150'))
os.environ['XLA_FLAGS'] = f'--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n_threads}'
logger.info(f"JAX configured with {n_threads} threads")


def compute_sandwich_vcov_jax(h2o_model, data, x_names, y_name) -> np.ndarray:
    """
    Compute sandwich (robust) variance-covariance matrix using JAX.
    
    Uses FULL dataset for proper inference - no subsampling.
    
    The sandwich estimator is:
    V = (X'X)^{-1} * X'diag(e^2)X * (X'X)^{-1}
    
    Args:
        h2o_model: Fitted H2O GLM model
        data: H2O DataFrame with training data
        x_names: List of predictor column names
        y_name: Target column name
        
    Returns:
        Robust variance-covariance matrix as numpy array
    """
    logger.info("Computing JAX-accelerated sandwich estimator using FULL dataset...")
    
    # Get predictions and residuals in H2O
    preds = h2o_model.predict(data)
    
    # For GLM models, predict returns multiple columns - we need just the 'predict' column
    # Check if preds has multiple columns
    if preds.ncols > 1:
        # H2O GLM returns 'predict' as the first column for regression
        preds = preds[:, 0]  # Select first column which contains predictions
    
    y_actual = data[y_name]
    residuals = y_actual - preds
    residuals_sq = residuals ** 2
    
    # Get the design matrix X
    X = data[x_names]
    n = X.nrows
    p = len(x_names) + 1  # +1 for intercept
    
    logger.info(f"Processing {n:,} rows × {p} features for full sandwich computation")
    
    # Add intercept column
    ones = h2o.H2OFrame([[1.0]] * n)
    ones.set_names(['intercept'])
    X_with_intercept = ones.cbind(X)
    
    # Convert to numpy with multi-threading
    logger.info("Converting H2O frames to numpy arrays with multi-threading...")
    X_np = X_with_intercept.as_data_frame(use_multi_thread=True).values
    r2_np = residuals_sq.as_data_frame(use_multi_thread=True).values.ravel()
    
    logger.info(f"Data converted: X shape {X_np.shape}, residuals shape {r2_np.shape}")
    
    # Convert to JAX arrays
    logger.info("Converting to JAX arrays...")
    X_jax = jnp.array(X_np)
    r2_jax = jnp.array(r2_np)
    
    # JIT compile the sandwich computation for maximum performance
    @jax.jit
    def _compute_sandwich_jax(X: jnp.ndarray, r2: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled sandwich computation."""
        # Compute (X'X)
        XtX = X.T @ X
        
        # Add ridge regularization for numerical stability
        XtX_reg = XtX + jnp.eye(p) * 1e-8
        
        # Compute (X'X)^{-1}
        XtX_inv = jnp.linalg.inv(XtX_reg)
        
        # Compute the "meat": X'diag(e^2)X
        # Efficient computation: X.T @ (X * r2[:, None])
        meat = X.T @ (X * r2[:, jnp.newaxis])
        
        # Sandwich: (X'X)^{-1} * meat * (X'X)^{-1}
        vcov_sandwich = XtX_inv @ meat @ XtX_inv
        
        # Scale by n/(n-p) for small sample correction
        scale = n / (n - p)
        vcov_sandwich = vcov_sandwich * scale
        
        return vcov_sandwich
    
    # Compute with JAX
    logger.info("Computing sandwich vcov with JAX (JIT compiled)...")
    vcov_jax = _compute_sandwich_jax(X_jax, r2_jax)
    
    # Block until computation is complete
    vcov_jax = vcov_jax.block_until_ready()
    
    # Convert back to numpy
    vcov_np = np.array(vcov_jax)
    
    logger.info(f"JAX sandwich vcov computed: shape {vcov_np.shape}")
    logger.info(f"Using FULL {n:,} observations - no subsampling!")
    
    return vcov_np


def compute_sandwich_vcov_chunked_jax(h2o_model, data, x_names, y_name, 
                                      chunk_size: int = 1_000_000) -> np.ndarray:
    """
    Compute sandwich estimator using chunked processing for memory efficiency.
    
    Processes data in chunks but uses ALL data - no subsampling.
    Useful when full dataset doesn't fit in memory at once.
    
    Args:
        h2o_model: Fitted H2O GLM model
        data: H2O DataFrame with training data
        x_names: List of predictor column names
        y_name: Target column name
        chunk_size: Number of rows to process at once
        
    Returns:
        Robust variance-covariance matrix as numpy array
    """
    logger.info(f"Computing JAX sandwich estimator in chunks of {chunk_size:,} rows...")
    
    # Get predictions and residuals in H2O
    preds = h2o_model.predict(data)
    
    # For GLM models, predict returns multiple columns - we need just the 'predict' column
    # Check if preds has multiple columns
    if preds.ncols > 1:
        # H2O GLM returns 'predict' as the first column for regression
        preds = preds[:, 0]  # Select first column which contains predictions
    
    y_actual = data[y_name]
    residuals = y_actual - preds
    residuals_sq = residuals ** 2
    
    # Get the design matrix X
    X = data[x_names]
    n = X.nrows
    p = len(x_names) + 1  # +1 for intercept
    
    logger.info(f"Processing {n:,} rows × {p} features in chunks")
    
    # Add intercept column
    ones = h2o.H2OFrame([[1.0]] * n)
    ones.set_names(['intercept'])
    X_with_intercept = ones.cbind(X)
    
    # Initialize accumulators using JAX
    XtX_acc = jnp.zeros((p, p))
    meat_acc = jnp.zeros((p, p))
    
    # Process in chunks
    n_chunks = (n + chunk_size - 1) // chunk_size
    
    @jax.jit
    def _update_accumulators(XtX_acc, meat_acc, X_chunk, r2_chunk):
        """JIT-compiled chunk processing."""
        # Update X'X
        XtX_acc = XtX_acc + X_chunk.T @ X_chunk
        
        # Update meat matrix
        meat_chunk = X_chunk.T @ (X_chunk * r2_chunk[:, jnp.newaxis])
        meat_acc = meat_acc + meat_chunk
        
        return XtX_acc, meat_acc
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n)
        chunk_rows = end_idx - start_idx
        
        logger.info(f"Processing chunk {i+1}/{n_chunks} (rows {start_idx:,}-{end_idx:,}, size: {chunk_rows:,})")
        
        # Get chunk from H2O - fix the slicing issue
        X_chunk_h2o = X_with_intercept[start_idx:end_idx, :]
        r2_chunk_h2o = residuals_sq[start_idx:end_idx, 0]  # Explicitly get first column
        
        # Convert to numpy then JAX
        X_chunk = jnp.array(X_chunk_h2o.as_data_frame(use_multi_thread=True).values)
        r2_chunk_values = r2_chunk_h2o.as_data_frame(use_multi_thread=True).values
        
        # Ensure r2_chunk is the right shape
        if len(r2_chunk_values.shape) > 1:
            r2_chunk_values = r2_chunk_values.ravel()
        r2_chunk = jnp.array(r2_chunk_values)
        
        # Verify shapes match
        if X_chunk.shape[0] != r2_chunk.shape[0]:
            logger.error(f"Shape mismatch: X_chunk {X_chunk.shape}, r2_chunk {r2_chunk.shape}")
            raise ValueError(f"Chunk size mismatch: X has {X_chunk.shape[0]} rows, residuals has {r2_chunk.shape[0]} rows")
        
        # Update accumulators
        XtX_acc, meat_acc = _update_accumulators(XtX_acc, meat_acc, X_chunk, r2_chunk)
    
    # Compute final sandwich estimator
    # Remove JIT here since p needs to be concrete for jnp.eye()
    def _finalize_sandwich(XtX, meat, n, p):
        """Compute final sandwich from accumulated matrices."""
        # Add regularization
        XtX_reg = XtX + jnp.eye(p) * 1e-8
        
        # Compute inverse
        XtX_inv = jnp.linalg.inv(XtX_reg)
        
        # Sandwich
        vcov = XtX_inv @ meat @ XtX_inv
        
        # Small sample correction
        scale = n / (n - p)
        return vcov * scale
    
    logger.info("Finalizing sandwich computation...")
    vcov_jax = _finalize_sandwich(XtX_acc, meat_acc, n, p)
    vcov_jax = vcov_jax.block_until_ready()
    
    # Convert to numpy
    vcov_np = np.array(vcov_jax)
    
    logger.info(f"JAX chunked sandwich vcov computed: shape {vcov_np.shape}")
    logger.info(f"Used ALL {n:,} observations across {n_chunks} chunks")
    
    return vcov_np


# Backward compatibility wrapper
def compute_sandwich_vcov(h2o_model, data, x_names, y_name) -> np.ndarray:
    """
    Wrapper that automatically selects best method based on data size.
    
    For datasets > 10M rows, uses chunked processing for memory efficiency.
    For smaller datasets, processes all at once for speed.
    """
    n_rows = data.nrows
    
    if n_rows > 10_000_000:  # 10M rows
        logger.info(f"Large dataset ({n_rows:,} rows) - using chunked JAX processing")
        return compute_sandwich_vcov_chunked_jax(h2o_model, data, x_names, y_name)
    else:
        logger.info(f"Dataset size ({n_rows:,} rows) - using direct JAX processing")
        return compute_sandwich_vcov_jax(h2o_model, data, x_names, y_name)