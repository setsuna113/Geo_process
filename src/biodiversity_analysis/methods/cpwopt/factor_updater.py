"""Advanced factor update strategies for CP-WOPT.

This module implements various optimization strategies for updating factor matrices,
including gradient-based methods and constrained optimization.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple, Callable
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import solve, lstsq
import warnings

from src.abstractions.types.tensor_types import CPFactors
from src.biodiversity_analysis.shared.tensor_utils import (
    unfold, compute_mttkrp, weighted_norm, compute_gram_matrices
)
from .constants import EPSILON, MIN_COMPONENT_NORM, MIN_IMPROVEMENT

logger = logging.getLogger(__name__)


class FactorUpdater:
    """Advanced factor update strategies beyond basic ALS."""
    
    def __init__(self, method: str = 'als', backend=None):
        """Initialize factor updater.
        
        Args:
            method: Update method ('als', 'gradient', 'nonnegative', 'sparse')
            backend: CPWOPTBackend instance for GPU/CPU computation (required)
        """
        if backend is None:
            from src.biodiversity_analysis.methods.cpwopt.backend import CPWOPTBackend
            backend = CPWOPTBackend('auto')
        self.method = method
        self.backend = backend
        self.update_history = []
        
    def update_factor(self, tensor: np.ndarray, weights: np.ndarray,
                     factors: List[np.ndarray], mode: int,
                     **kwargs) -> np.ndarray:
        """Update factor matrix for specified mode.
        
        Args:
            tensor: Input tensor (NaN replaced with 0)
            weights: Weight mask
            factors: Current factor matrices
            mode: Mode to update
            **kwargs: Method-specific parameters
            
        Returns:
            Updated factor matrix
        """
        if self.method == 'als':
            return self._update_als_optimized(tensor, weights, factors, mode)
        elif self.method == 'nonnegative':
            return self._update_nonnegative(tensor, weights, factors, mode, **kwargs)
        elif self.method == 'sparse':
            return self._update_sparse(tensor, weights, factors, mode, **kwargs)
        elif self.method == 'gradient':
            return self._update_gradient(tensor, weights, factors, mode, **kwargs)
        else:
            raise ValueError(f"Unknown update method: {self.method}")
    
    def _update_als_optimized(self, tensor: np.ndarray, weights: np.ndarray,
                             factors: List[np.ndarray], mode: int) -> np.ndarray:
        """Optimized ALS update with better numerical stability.
        
        This is an improved version of the basic ALS update with:
        - Better handling of missing values
        - Numerical stability improvements
        - Efficient computation of weighted operations
        """
        n_modes = len(factors)
        rank = factors[0].shape[1]
        mode_size = tensor.shape[mode]
        
        # Precompute Gram matrices
        grams = compute_gram_matrices(factors, skip_mode=mode)
        
        # Compute Hadamard product of all Gram matrices
        gram_prod = grams[0]
        for g in grams[1:]:
            gram_prod *= g  # Element-wise
        
        # Add regularization
        gram_prod += np.eye(rank) * EPSILON
        
        # Precompute inverse
        try:
            gram_inv = np.linalg.inv(gram_prod)
        except np.linalg.LinAlgError:
            logger.warning("Gram matrix singular, using pseudo-inverse")
            gram_inv = np.linalg.pinv(gram_prod)
        
        # Unfold tensor and weights
        tensor_unfold = unfold(tensor, mode)
        weights_unfold = unfold(weights, mode)
        
        # Initialize new factor
        new_factor = np.zeros((mode_size, rank))
        
        # Compute MTTKRP efficiently
        # First, get Khatri-Rao product of all other factors
        kr_product = self._compute_khatri_rao_product(factors, skip_mode=mode)
        
        # Update each row
        for i in range(mode_size):
            w_i = weights_unfold[i, :]
            
            # Skip rows with no observations
            if np.sum(w_i) == 0:
                continue
            
            # Weighted MTTKRP for this row
            weighted_data = tensor_unfold[i, :] * w_i
            mttkrp_row = weighted_data @ kr_product
            
            # Weighted Gram matrix for this row
            # This accounts for missing values pattern
            W_diag = np.diag(w_i)
            weighted_gram = kr_product.T @ W_diag @ kr_product
            weighted_gram += np.eye(rank) * EPSILON
            
            # Solve the system
            try:
                new_factor[i, :] = np.linalg.solve(weighted_gram, mttkrp_row)
            except np.linalg.LinAlgError:
                new_factor[i, :] = mttkrp_row @ np.linalg.pinv(weighted_gram)
        
        # Normalize columns
        new_factor = self._normalize_factor(new_factor)
        
        return new_factor
    
    def _compute_khatri_rao_product(self, factors: List[np.ndarray], 
                                   skip_mode: int) -> np.ndarray:
        """Compute Khatri-Rao product of all factors except one mode.
        
        Uses broadcasting and chunking to avoid memory overflow for large tensors.
        
        Args:
            factors: List of factor matrices
            skip_mode: Mode to skip
            
        Returns:
            Khatri-Rao product matrix
        """
        # Get factors to multiply (all except skip_mode)
        other_factors = [f for i, f in enumerate(factors) if i != skip_mode]
        
        if len(other_factors) == 0:
            raise ValueError("Need at least one other factor")
        
        # Start with first factor
        result = other_factors[0]
        
        # Memory-efficient successive Khatri-Rao products
        for factor in other_factors[1:]:
            n1, r = result.shape
            n2, _ = factor.shape
            
            # Check if result will be too large
            result_size = n1 * n2 * r * 8  # 8 bytes per float64
            if result_size > 1e9:  # More than 1GB
                # Use chunked computation
                result = self._compute_khatri_rao_chunked(result, factor)
            else:
                # Use efficient broadcasting method
                # This avoids the loop and uses NumPy's optimized operations
                result = (result[:, np.newaxis, :] * factor[np.newaxis, :, :]).reshape(n1 * n2, r)
        
        return result
    
    def _compute_khatri_rao_chunked(self, A: np.ndarray, B: np.ndarray, 
                                   chunk_size: int = 10000) -> np.ndarray:
        """Compute Khatri-Rao product in chunks to avoid memory overflow.
        
        Args:
            A: First factor matrix (n1 x r)
            B: Second factor matrix (n2 x r)
            chunk_size: Number of rows to process at once
            
        Returns:
            Khatri-Rao product (n1*n2 x r)
        """
        n1, r = A.shape
        n2 = B.shape[0]
        
        # Pre-allocate result
        result = np.empty((n1 * n2, r), dtype=A.dtype)
        
        # Process in chunks
        for i in range(0, n1, chunk_size):
            end_i = min(i + chunk_size, n1)
            chunk_A = A[i:end_i]
            
            # Compute Khatri-Rao for this chunk
            chunk_result = (chunk_A[:, np.newaxis, :] * B[np.newaxis, :, :])
            
            # Store in result
            start_idx = i * n2
            end_idx = end_i * n2
            result[start_idx:end_idx] = chunk_result.reshape(-1, r)
        
        return result
    
    def _update_nonnegative(self, tensor: np.ndarray, weights: np.ndarray,
                           factors: List[np.ndarray], mode: int,
                           alpha: float = 0.0) -> np.ndarray:
        """Update factor with non-negativity constraints.
        
        Uses multiplicative updates or projected gradient descent.
        
        Args:
            tensor: Input tensor
            weights: Weight mask
            factors: Current factors
            mode: Mode to update
            alpha: L2 regularization parameter
            
        Returns:
            Non-negative factor matrix
        """
        # Get standard ALS solution first
        factor_als = self._update_als_optimized(tensor, weights, factors, mode)
        
        # Project to non-negative orthant
        factor_nn = np.maximum(factor_als, 0)
        
        # Multiplicative refinement
        for _ in range(10):  # Few iterations of multiplicative updates
            # Compute gradients
            reconstructed = self._partial_reconstruct(factors, mode, factor_nn)
            residual = (tensor - reconstructed) * weights
            
            # Multiplicative update rule
            numerator = self._compute_positive_gradient(
                residual, weights, factors, mode
            )
            denominator = self._compute_negative_gradient(
                factor_nn, factors, mode
            ) + alpha * factor_nn + EPSILON
            
            # Update
            factor_nn *= (numerator / denominator) ** 0.5
            
            # Normalize
            factor_nn = self._normalize_factor(factor_nn)
        
        return factor_nn
    
    def _update_sparse(self, tensor: np.ndarray, weights: np.ndarray,
                      factors: List[np.ndarray], mode: int,
                      sparsity: float = 0.1) -> np.ndarray:
        """Update factor with sparsity constraints.
        
        Uses L1 regularization or hard thresholding.
        
        Args:
            tensor: Input tensor
            weights: Weight mask
            factors: Current factors
            mode: Mode to update
            sparsity: Target sparsity level (0-1)
            
        Returns:
            Sparse factor matrix
        """
        # Get standard ALS solution
        factor_dense = self._update_als_optimized(tensor, weights, factors, mode)
        
        # Apply soft thresholding for sparsity
        rank = factor_dense.shape[1]
        factor_sparse = np.zeros_like(factor_dense)
        
        for r in range(rank):
            # Get column
            col = factor_dense[:, r]
            
            # Determine threshold for target sparsity
            n_zeros = int(len(col) * sparsity)
            if n_zeros > 0:
                threshold = np.sort(np.abs(col))[n_zeros]
                
                # Soft threshold
                col_sparse = np.sign(col) * np.maximum(np.abs(col) - threshold, 0)
            else:
                col_sparse = col
            
            factor_sparse[:, r] = col_sparse
        
        # Normalize
        factor_sparse = self._normalize_factor(factor_sparse)
        
        return factor_sparse
    
    def _update_gradient(self, tensor: np.ndarray, weights: np.ndarray,
                        factors: List[np.ndarray], mode: int,
                        learning_rate: float = 0.01,
                        momentum: float = 0.9,
                        n_iterations: int = 10) -> np.ndarray:
        """Update factor using gradient descent.
        
        Useful for non-convex constraints or custom loss functions.
        
        Args:
            tensor: Input tensor
            weights: Weight mask
            factors: Current factors
            mode: Mode to update
            learning_rate: Step size
            momentum: Momentum parameter
            n_iterations: Number of gradient steps
            
        Returns:
            Updated factor matrix
        """
        mode_size = tensor.shape[mode]
        rank = factors[0].shape[1]
        
        # Initialize with current factor
        factor = factors[mode].copy()
        velocity = np.zeros_like(factor)
        
        # Gradient descent iterations
        for _ in range(n_iterations):
            # Compute gradient
            gradient = self._compute_gradient(
                tensor, weights, factors, mode, factor
            )
            
            # Momentum update
            velocity = momentum * velocity - learning_rate * gradient
            factor += velocity
            
            # Project to feasible set (if needed)
            factor = self._project_factor(factor)
        
        # Final normalization
        factor = self._normalize_factor(factor)
        
        return factor
    
    def _compute_gradient(self, tensor: np.ndarray, weights: np.ndarray,
                         factors: List[np.ndarray], mode: int,
                         current_factor: np.ndarray) -> np.ndarray:
        """Compute gradient of weighted loss w.r.t. factor matrix.
        
        Gradient of ||W ⊙ (X - [[A,B,C]])||^2 w.r.t. A
        """
        # Update factors list with current factor
        factors_temp = factors.copy()
        factors_temp[mode] = current_factor
        
        # Compute residual
        from src.biodiversity_analysis.shared.tensor_utils import cp_reconstruct
        reconstructed = cp_reconstruct(factors_temp)
        residual = (tensor - reconstructed) * weights
        
        # Compute gradient as -2 * MTTKRP(residual)
        gradient = -2 * compute_mttkrp(residual, factors_temp, mode, self.backend)
        
        return gradient
    
    def _partial_reconstruct(self, factors: List[np.ndarray], mode: int,
                           new_factor: np.ndarray) -> np.ndarray:
        """Reconstruct tensor with updated factor for one mode."""
        factors_temp = factors.copy()
        factors_temp[mode] = new_factor
        
        from src.biodiversity_analysis.shared.tensor_utils import cp_reconstruct
        return cp_reconstruct(factors_temp)
    
    def _compute_positive_gradient(self, residual: np.ndarray, 
                                 weights: np.ndarray,
                                 factors: List[np.ndarray], 
                                 mode: int) -> np.ndarray:
        """Compute positive part of gradient for multiplicative updates."""
        positive_residual = np.maximum(residual, 0)
        return compute_mttkrp(positive_residual, factors, mode, self.backend)
    
    def _compute_negative_gradient(self, factor: np.ndarray,
                                 factors: List[np.ndarray], 
                                 mode: int) -> np.ndarray:
        """Compute negative part of gradient for multiplicative updates."""
        # This is simplified - full implementation would compute
        # the exact negative gradient term
        rank = factor.shape[1]
        return factor @ np.eye(rank)
    
    def _normalize_factor(self, factor: np.ndarray) -> np.ndarray:
        """Normalize factor columns to unit norm."""
        rank = factor.shape[1]
        
        for r in range(rank):
            norm = np.linalg.norm(factor[:, r])
            if norm > MIN_COMPONENT_NORM:
                factor[:, r] /= norm
            else:
                # Reinitialize near-zero components
                factor[:, r] = np.random.rand(factor.shape[0]) * 0.01
                
        return factor
    
    def _project_factor(self, factor: np.ndarray) -> np.ndarray:
        """Project factor to feasible set (e.g., simplex, box constraints)."""
        # Default: no projection
        return factor


class AdaptiveFactorUpdater(FactorUpdater):
    """Factor updater that adapts strategy based on convergence."""
    
    def __init__(self, base_method: str = 'als', backend=None):
        """Initialize adaptive updater."""
        super().__init__(base_method, backend)
        self.convergence_history = []
        self.method_history = []
        
    def update_factor_adaptive(self, tensor: np.ndarray, weights: np.ndarray,
                             factors: List[np.ndarray], mode: int,
                             iteration: int) -> np.ndarray:
        """Adaptively choose update method based on convergence.
        
        Args:
            tensor: Input tensor
            weights: Weight mask
            factors: Current factors
            mode: Mode to update
            iteration: Current iteration number
            
        Returns:
            Updated factor matrix
        """
        # Start with base method
        if iteration < 50:
            method = self.method
        else:
            # Check recent convergence
            if len(self.convergence_history) > 10:
                recent_improvement = self._compute_recent_improvement()
                
                if recent_improvement < MIN_IMPROVEMENT:
                    # Switch to gradient method for fine-tuning
                    method = 'gradient'
                else:
                    method = self.method
            else:
                method = self.method
        
        # Log method switch
        if len(self.method_history) == 0 or method != self.method_history[-1]:
            logger.info(f"Switching to {method} method at iteration {iteration}")
        
        self.method_history.append(method)
        
        # Update with chosen method
        return self.update_factor(tensor, weights, factors, mode)
    
    def _compute_recent_improvement(self) -> float:
        """Compute recent relative improvement in convergence."""
        if len(self.convergence_history) < 10:
            return 1.0
        
        recent = self.convergence_history[-10:]
        old_error = recent[0]
        new_error = recent[-1]
        
        if old_error == 0:
            return 0.0
            
        return abs(old_error - new_error) / old_error


def create_factor_updater(method: str = 'als', 
                         adaptive: bool = False,
                         backend=None) -> FactorUpdater:
    """Factory function to create appropriate factor updater.
    
    Args:
        method: Base update method
        adaptive: Whether to use adaptive switching
        
    Returns:
        FactorUpdater instance
    """
    if adaptive:
        return AdaptiveFactorUpdater(method, backend)
    else:
        return FactorUpdater(method, backend)