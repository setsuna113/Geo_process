"""Utility functions for tensor operations in biodiversity analysis.

This module provides helper functions for tensor manipulation, validation,
and common operations used across tensor factorization methods.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import logging
from scipy.linalg import khatri_rao
from numpy.linalg import norm

logger = logging.getLogger(__name__)


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Unfold (matricize) a tensor along a specified mode.
    
    Args:
        tensor: Input tensor
        mode: Mode along which to unfold (0, 1, or 2 for 3D tensor)
        
    Returns:
        Unfolded matrix of shape (tensor.shape[mode], prod(other dimensions))
    """
    if mode < 0 or mode >= tensor.ndim:
        raise ValueError(f"Mode {mode} out of range for {tensor.ndim}D tensor")
    
    # Move the mode to the front
    n = tensor.shape[mode]
    perm = list(range(tensor.ndim))
    perm[mode] = 0
    perm[0] = mode
    
    # Permute and reshape
    tensor_perm = np.transpose(tensor, perm)
    return tensor_perm.reshape(n, -1)


def fold(matrix: np.ndarray, mode: int, shape: Tuple[int, ...]) -> np.ndarray:
    """Fold a matrix back into tensor form.
    
    Args:
        matrix: Unfolded matrix
        mode: Mode that was unfolded
        shape: Original tensor shape
        
    Returns:
        Folded tensor
    """
    # Determine the permutation needed
    n_modes = len(shape)
    perm = list(range(n_modes))
    perm[mode] = 0
    perm[0] = mode
    
    # Reshape to permuted shape
    shape_perm = [shape[i] for i in perm]
    tensor_perm = matrix.reshape(shape_perm)
    
    # Permute back to original order
    inv_perm = np.argsort(perm)
    return np.transpose(tensor_perm, inv_perm)


def tensor_multiply(tensor: np.ndarray, factors: List[np.ndarray], 
                   skip_mode: Optional[int] = None) -> np.ndarray:
    """Multiply a tensor by factor matrices along all modes except one.
    
    Used in ALS updates: Y = X ×₁ A ×₂ B ×₃ C ... (skipping one mode)
    
    Args:
        tensor: Input tensor
        factors: List of factor matrices
        skip_mode: Mode to skip (for ALS updates)
        
    Returns:
        Result tensor
    """
    result = tensor.copy()
    
    for mode, factor in enumerate(factors):
        if mode == skip_mode:
            continue
            
        # Unfold along mode
        unfolded = unfold(result, mode)
        
        # Multiply
        result_unfolded = factor.T @ unfolded
        
        # Update shape
        new_shape = list(result.shape)
        new_shape[mode] = factor.shape[1]
        
        # Fold back
        result = fold(result_unfolded, mode, tuple(new_shape))
    
    return result


def cp_reconstruct(factors: List[np.ndarray], 
                  weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Reconstruct tensor from CP factors.
    
    Args:
        factors: List of factor matrices [A, B, C, ...]
        weights: Optional component weights
        
    Returns:
        Reconstructed tensor
    """
    n_modes = len(factors)
    rank = factors[0].shape[1]
    
    # Get tensor shape
    shape = [f.shape[0] for f in factors]
    
    # Initialize with zeros
    tensor = np.zeros(shape)
    
    # Sum up rank-1 tensors
    for r in range(rank):
        # Extract r-th column from each factor
        rank1_tensor = factors[0][:, r]
        for mode in range(1, n_modes):
            rank1_tensor = np.outer(rank1_tensor, factors[mode][:, r]).reshape(-1)
        
        # Reshape to tensor
        rank1_tensor = rank1_tensor.reshape(shape)
        
        # Add with weight if provided
        if weights is not None:
            tensor += weights[r] * rank1_tensor
        else:
            tensor += rank1_tensor
    
    return tensor


def weighted_norm(tensor: np.ndarray, weights: np.ndarray) -> float:
    """Calculate weighted Frobenius norm.
    
    Args:
        tensor: Input tensor
        weights: Weight mask (same shape as tensor)
        
    Returns:
        Weighted norm: sqrt(sum(weights * tensor^2))
    """
    return np.sqrt(np.sum(weights * tensor * tensor))


def calculate_fit(original: np.ndarray, reconstructed: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
    """Calculate goodness of fit for tensor decomposition.
    
    Args:
        original: Original tensor
        reconstructed: Reconstructed tensor
        weights: Optional weight mask
        
    Returns:
        Fit value between 0 and 1 (1 = perfect reconstruction)
    """
    if weights is None:
        weights = np.ones_like(original)
    
    # Calculate residual
    residual = original - reconstructed
    
    # Calculate norms
    residual_norm = weighted_norm(residual, weights)
    original_norm = weighted_norm(original, weights)
    
    # Calculate fit
    if original_norm == 0:
        return 1.0 if residual_norm == 0 else 0.0
    
    fit = 1.0 - (residual_norm / original_norm)
    return max(0.0, fit)  # Ensure non-negative


def initialize_factors(shape: Tuple[int, ...], rank: int, 
                      method: str = 'random',
                      random_state: Optional[int] = None) -> List[np.ndarray]:
    """Initialize factor matrices for CP decomposition.
    
    Args:
        shape: Tensor shape
        rank: Number of components
        method: Initialization method ('random', 'svd', 'nndsvd')
        random_state: Random seed
        
    Returns:
        List of initialized factor matrices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    factors = []
    
    if method == 'random':
        # Random initialization with small positive values
        for mode_size in shape:
            factors.append(np.random.rand(mode_size, rank) * 0.1)
            
    elif method == 'svd':
        # SVD-based initialization (requires unfolding)
        # For now, fall back to random with different distribution
        for mode_size in shape:
            factors.append(np.random.randn(mode_size, rank) * 0.1 + 0.5)
            
    elif method == 'nndsvd':
        # Non-negative double SVD - ensure non-negative factors
        for mode_size in shape:
            factor = np.random.rand(mode_size, rank)
            # Make more structured
            for r in range(rank):
                # Create some structure in initialization
                pattern = np.linspace(0, 1, mode_size)
                noise = np.random.rand(mode_size) * 0.1
                factor[:, r] = pattern + noise
            factors.append(factor)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    # Normalize factors
    for factor in factors:
        for r in range(rank):
            factor[:, r] /= norm(factor[:, r]) + 1e-10
    
    return factors


def compute_gram_matrices(factors: List[np.ndarray], 
                         skip_mode: Optional[int] = None) -> List[np.ndarray]:
    """Compute Gram matrices (A^T A) for all factor matrices.
    
    Used to speed up ALS computations.
    
    Args:
        factors: List of factor matrices
        skip_mode: Mode to skip
        
    Returns:
        List of Gram matrices
    """
    grams = []
    for mode, factor in enumerate(factors):
        if mode == skip_mode:
            grams.append(None)
        else:
            grams.append(factor.T @ factor)
    return grams


def compute_mttkrp(tensor: np.ndarray, factors: List[np.ndarray], 
                  mode: int) -> np.ndarray:
    """Compute Matricized Tensor Times Khatri-Rao Product (MTTKRP).
    
    Core operation in CP-ALS: X_(n) (C ⊙ B ⊙ A)
    
    Args:
        tensor: Input tensor
        factors: List of factor matrices
        mode: Mode for which to compute MTTKRP
        
    Returns:
        MTTKRP result
    """
    # Get factors for other modes
    other_factors = [factors[m] for m in range(len(factors)) if m != mode]
    
    # Compute Khatri-Rao product of other factors
    kr = other_factors[0]
    for factor in other_factors[1:]:
        kr = khatri_rao(kr, factor)
    
    # Unfold tensor and multiply
    unfolded = unfold(tensor, mode)
    return unfolded @ kr


def apply_weight_mask(tensor: np.ndarray, weights: np.ndarray,
                     fill_value: float = 0.0) -> np.ndarray:
    """Apply weight mask to tensor, setting missing values.
    
    Args:
        tensor: Input tensor
        weights: Binary weight mask
        fill_value: Value for missing entries
        
    Returns:
        Masked tensor
    """
    result = tensor.copy()
    result[weights == 0] = fill_value
    return result


def tensor_stats(tensor: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Calculate statistics for a tensor.
    
    Args:
        tensor: Input tensor
        weights: Optional weight mask
        
    Returns:
        Dictionary of statistics
    """
    if weights is None:
        weights = ~np.isnan(tensor)
    
    valid_data = tensor[weights > 0]
    
    stats = {
        'shape': tensor.shape,
        'size': tensor.size,
        'n_observed': len(valid_data),
        'sparsity': 1.0 - (len(valid_data) / tensor.size),
        'min': np.min(valid_data) if len(valid_data) > 0 else np.nan,
        'max': np.max(valid_data) if len(valid_data) > 0 else np.nan,
        'mean': np.mean(valid_data) if len(valid_data) > 0 else np.nan,
        'std': np.std(valid_data) if len(valid_data) > 0 else np.nan,
        'n_zeros': np.sum(valid_data == 0) if len(valid_data) > 0 else 0
    }
    
    return stats


def check_convergence(errors: List[float], threshold: float = 1e-6,
                     patience: int = 10) -> bool:
    """Check if algorithm has converged based on error history.
    
    Args:
        errors: List of reconstruction errors
        threshold: Relative change threshold
        patience: Number of iterations to check
        
    Returns:
        True if converged
    """
    if len(errors) < patience:
        return False
    
    # Check relative change over last 'patience' iterations
    recent_errors = errors[-patience:]
    
    # Calculate relative changes
    rel_changes = []
    for i in range(1, len(recent_errors)):
        if recent_errors[i-1] > 0:
            rel_change = abs(recent_errors[i] - recent_errors[i-1]) / recent_errors[i-1]
            rel_changes.append(rel_change)
    
    # Check if all recent changes are below threshold
    if rel_changes and all(change < threshold for change in rel_changes):
        logger.info(f"Converged: all relative changes < {threshold}")
        return True
    
    return False


def validate_cp_factors(factors: List[np.ndarray]) -> Tuple[bool, List[str]]:
    """Validate CP factor matrices.
    
    Args:
        factors: List of factor matrices
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not factors:
        issues.append("No factors provided")
        return False, issues
    
    # Check rank consistency
    ranks = [f.shape[1] for f in factors]
    if len(set(ranks)) > 1:
        issues.append(f"Inconsistent ranks: {ranks}")
    
    # Check for NaN/Inf
    for i, factor in enumerate(factors):
        if np.any(np.isnan(factor)):
            issues.append(f"Factor {i} contains NaN")
        if np.any(np.isinf(factor)):
            issues.append(f"Factor {i} contains Inf")
        if factor.size == 0:
            issues.append(f"Factor {i} is empty")
    
    return len(issues) == 0, issues