"""Core CP-WOPT implementation for sparse biodiversity tensors.

This module implements the CANDECOMP/PARAFAC decomposition with weighted
optimization, specifically designed for sparse biodiversity data.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import time

from src.abstractions.types.tensor_types import (
    TensorData, CPFactors, TensorMetrics, TensorDecompositionResult,
    TensorOptimizer, FactorInitMethod
)
from src.biodiversity_analysis.shared.tensor_utils import (
    initialize_factors, compute_mttkrp, cp_reconstruct,
    weighted_norm, calculate_fit, check_convergence,
    validate_cp_factors, unfold, compute_gram_matrices
)
from .constants import *

logger = logging.getLogger(__name__)


class CPWOPT:
    """CP decomposition with weighted optimization for sparse tensors.
    
    This implementation uses Alternating Least Squares (ALS) with support
    for missing values through weight masks. Optimized for biodiversity
    data with >95% sparsity.
    """
    
    def __init__(self,
                 rank: int = DEFAULT_RANK,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
                 init_method: str = DEFAULT_INIT_METHOD,
                 optimizer: str = DEFAULT_OPTIMIZER,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """Initialize CP-WOPT algorithm.
        
        Args:
            rank: Number of components to extract
            max_iterations: Maximum number of iterations
            convergence_threshold: Relative error threshold for convergence
            init_method: Factor initialization method
            optimizer: Optimization algorithm ('als' or 'lbfgs')
            random_state: Random seed for reproducibility
            verbose: Whether to log progress
        """
        if rank <= 0:
            raise ValueError(ERROR_INVALID_RANK.format(rank))
            
        self.rank = rank
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.init_method = init_method
        self.optimizer = optimizer
        self.random_state = random_state
        self.verbose = verbose
        
        # State variables
        self.factors_: Optional[CPFactors] = None
        self.metrics_: Optional[TensorMetrics] = None
        self.convergence_history_: List[float] = []
        self.training_time_: float = 0.0
        self.n_iterations_: int = 0
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        
    def fit(self, tensor_data: TensorData) -> 'CPWOPT':
        """Fit CP decomposition to tensor data.
        
        Args:
            tensor_data: Input tensor with weight mask
            
        Returns:
            Self (fitted model)
        """
        start_time = time.time()
        
        # Validate input
        self._validate_input(tensor_data)
        
        # Initialize factors
        if self.verbose:
            logger.info(f"Initializing CP-WOPT with rank={self.rank}")
            logger.info(f"Tensor shape: {tensor_data.shape}, sparsity: {tensor_data.sparsity:.2%}")
            
        factors = self._initialize_factors(tensor_data)
        
        # Optimize
        if self.optimizer == 'als':
            factors, metrics = self._fit_als(tensor_data, factors)
        else:
            raise NotImplementedError(f"Optimizer '{self.optimizer}' not implemented")
        
        # Store results
        self.factors_ = factors
        self.metrics_ = metrics
        self.training_time_ = time.time() - start_time
        
        if self.verbose:
            logger.info(f"CP-WOPT completed in {self.training_time_:.2f}s")
            logger.info(f"Final fit: {metrics.fit:.4f}, iterations: {self.n_iterations_}")
            
        return self
    
    def _validate_input(self, tensor_data: TensorData) -> None:
        """Validate input tensor data."""
        if tensor_data.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {tensor_data.ndim}D")
            
        if tensor_data.sparsity > MAX_MISSING_RATIO:
            logger.warning(WARNING_HIGH_SPARSITY.format(tensor_data.sparsity))
            
        if self.rank > min(tensor_data.shape):
            logger.warning(WARNING_RANK_TOO_HIGH.format(self.rank, tensor_data.shape))
    
    def _initialize_factors(self, tensor_data: TensorData) -> CPFactors:
        """Initialize factor matrices."""
        factors = initialize_factors(
            tensor_data.shape,
            self.rank,
            method=self.init_method,
            random_state=self.random_state
        )
        
        mode_names = ['latitude', 'longitude', 'features']
        
        return CPFactors(
            factors=factors,
            rank=self.rank,
            mode_names=mode_names
        )
    
    def _fit_als(self, tensor_data: TensorData, 
                 factors: CPFactors) -> Tuple[CPFactors, TensorMetrics]:
        """Fit using Alternating Least Squares.
        
        This is the core optimization loop for CP-WOPT.
        """
        tensor = tensor_data.tensor
        weights = tensor_data.weights
        
        # Replace NaN with 0 for computation
        tensor_clean = np.nan_to_num(tensor, nan=0.0)
        
        # Track convergence
        errors = []
        prev_error = np.inf
        
        # Main ALS loop
        for iteration in range(self.max_iterations):
            
            # Update each mode
            for mode in range(tensor.ndim):
                # Update factor for this mode
                factors.factors[mode] = self._update_factor_als(
                    tensor_clean, weights, factors.factors, mode
                )
            
            # Calculate reconstruction error
            reconstructed = cp_reconstruct(factors.factors)
            residual = (tensor_clean - reconstructed) * weights
            error = weighted_norm(residual, weights)
            
            # Calculate relative error and fit
            tensor_norm = weighted_norm(tensor_clean, weights)
            rel_error = error / (tensor_norm + EPSILON)
            fit = 1.0 - rel_error
            
            errors.append(error)
            
            # Log progress
            if self.verbose and iteration % PROGRESS_INTERVAL == 0:
                logger.info(f"Iteration {iteration}: fit={fit:.4f}, error={error:.4e}")
                
            # Progress callback
            if self.progress_callback:
                self.progress_callback(iteration, self.max_iterations, 
                                     f"CP-WOPT iteration {iteration}, fit={fit:.4f}")
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(error - prev_error) / (prev_error + EPSILON)
                if rel_change < self.convergence_threshold:
                    if self.verbose:
                        logger.info(f"Converged at iteration {iteration}")
                    break
                    
            prev_error = error
            
        self.n_iterations_ = iteration + 1
        self.convergence_history_ = errors
        
        # Create metrics
        metrics = TensorMetrics(
            reconstruction_error=error,
            relative_error=rel_error,
            fit=fit,
            convergence_history=errors
        )
        
        # Calculate component weights (norm of each component)
        component_weights = np.zeros(self.rank)
        for r in range(self.rank):
            weight = 1.0
            for mode in range(tensor.ndim):
                weight *= np.linalg.norm(factors.factors[mode][:, r])
            component_weights[r] = weight
            
        # Sort components by weight
        sort_idx = np.argsort(component_weights)[::-1]
        for mode in range(tensor.ndim):
            factors.factors[mode] = factors.factors[mode][:, sort_idx]
        component_weights = component_weights[sort_idx]
        
        metrics.component_weights = component_weights
        
        return factors, metrics
    
    def _update_factor_als(self, tensor: np.ndarray, weights: np.ndarray,
                          factors: List[np.ndarray], mode: int) -> np.ndarray:
        """Update one factor matrix using weighted ALS.
        
        This solves: A = X_(n) (C ⊙ B)^T [(C^T C) * (B^T B)]^{-1}
        with modifications for missing values.
        """
        n_modes = len(factors)
        rank = factors[0].shape[1]
        mode_size = tensor.shape[mode]
        
        # Compute Gram matrices for other modes
        grams = []
        for m in range(n_modes):
            if m != mode:
                grams.append(factors[m].T @ factors[m])
        
        # Initialize new factor
        new_factor = np.zeros((mode_size, rank))
        
        # Unfold tensor and weights along mode
        tensor_unfold = unfold(tensor, mode)
        weights_unfold = unfold(weights, mode)
        
        # Update each row of the factor matrix
        for i in range(mode_size):
            # Get weights for this fiber
            w_i = weights_unfold[i, :]
            
            # Skip if no observed values
            if np.sum(w_i) == 0:
                continue
            
            # Build the system for this row
            # Left-hand side: weighted MTTKRP
            # Right-hand side: weighted Hadamard product of Gram matrices
            
            # Compute V = (C ⊙ B ⊙ ...) for other modes
            V = factors[0] if mode != 0 else factors[1]
            for m in range(n_modes):
                if m != mode and m != (0 if mode != 0 else 1):
                    # Khatri-Rao product
                    V_new = np.zeros((V.shape[0] * factors[m].shape[0], rank))
                    for r in range(rank):
                        V_new[:, r] = np.kron(V[:, r], factors[m][:, r])
                    V = V_new
            
            # Apply weights
            V_weighted = V * w_i[:, np.newaxis]
            
            # Left-hand side
            lhs = tensor_unfold[i, :] @ V_weighted
            
            # Right-hand side (Hadamard product of Gram matrices)
            rhs = grams[0]
            for g in grams[1:]:
                rhs = rhs * g  # Element-wise product
                
            # Add regularization for numerical stability
            rhs += np.eye(rank) * EPSILON
            
            # Solve the system
            try:
                new_factor[i, :] = np.linalg.solve(rhs, lhs)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                new_factor[i, :] = lhs @ np.linalg.pinv(rhs)
        
        # Normalize to prevent numerical issues
        for r in range(rank):
            norm = np.linalg.norm(new_factor[:, r])
            if norm > MIN_COMPONENT_NORM:
                new_factor[:, r] /= norm
            else:
                # Reinitialize near-zero components
                new_factor[:, r] = np.random.rand(mode_size) * 0.01
                
        return new_factor
    
    def transform(self, tensor_data: TensorData) -> np.ndarray:
        """Transform tensor data to component space.
        
        Args:
            tensor_data: Input tensor
            
        Returns:
            Component activations for each data point
        """
        if self.factors_ is None:
            raise ValueError("Model must be fitted before transform")
            
        # This would project the tensor onto the learned components
        # For now, return a placeholder
        raise NotImplementedError("Transform not yet implemented")
    
    def predict(self) -> np.ndarray:
        """Reconstruct the tensor from learned factors.
        
        Returns:
            Reconstructed tensor
        """
        if self.factors_ is None:
            raise ValueError("Model must be fitted before predict")
            
        return cp_reconstruct(self.factors_.factors)
    
    def reconstruct(self) -> np.ndarray:
        """Alias for predict() - reconstruct the tensor from learned factors.
        
        Returns:
            Reconstructed tensor
        """
        return self.predict()
    
    def get_component_matrices(self) -> Dict[str, np.ndarray]:
        """Get factor matrices organized by mode name.
        
        Returns:
            Dictionary mapping mode name to factor matrix
        """
        if self.factors_ is None:
            raise ValueError("Model must be fitted first")
            
        return {
            name: factor
            for name, factor in zip(self.factors_.mode_names, self.factors_.factors)
        }
    
    def get_top_features(self, component: int, n_top: int = TOP_N_FEATURES) -> Dict[str, np.ndarray]:
        """Get top features for a specific component.
        
        Args:
            component: Component index
            n_top: Number of top features to return
            
        Returns:
            Dictionary with top feature indices and weights for each mode
        """
        if self.factors_ is None:
            raise ValueError("Model must be fitted first")
            
        if component >= self.rank:
            raise ValueError(f"Component {component} out of range (rank={self.rank})")
            
        top_features = {}
        
        for mode_name, factor in zip(self.factors_.mode_names, self.factors_.factors):
            # Get loadings for this component
            loadings = np.abs(factor[:, component])
            
            # Get top indices
            top_idx = np.argsort(loadings)[-n_top:][::-1]
            
            top_features[mode_name] = {
                'indices': top_idx,
                'loadings': loadings[top_idx]
            }
            
        return top_features
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """Set progress callback function.
        
        Args:
            callback: Function called with (current, total, message)
        """
        self.progress_callback = callback