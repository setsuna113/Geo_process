"""Core CP-WOPT implementation for sparse biodiversity tensors.

This module implements the CANDECOMP/PARAFAC decomposition with weighted
optimization, specifically designed for sparse biodiversity data.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple, Callable
import time

from src.abstractions import (
    TensorData, CPFactors, TensorMetrics
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
                 verbose: bool = True,
                 use_gpu: str = 'auto',
                 learning_rate: float = 0.01,
                 momentum: float = 0.9):
        """Initialize CP-WOPT algorithm.
        
        Args:
            rank: Number of components to extract
            max_iterations: Maximum number of iterations
            convergence_threshold: Relative error threshold for convergence
            init_method: Factor initialization method
            optimizer: Optimization algorithm ('als', 'gradient', 'sgd_momentum')
            random_state: Random seed for reproducibility
            verbose: Whether to log progress
            use_gpu: Device selection ('auto', 'cpu', 'gpu')
            learning_rate: Learning rate for SGD methods
            momentum: Momentum parameter for SGD with momentum
            
        Attributes:
            sparse_config: Configuration for HiCOO sparse tensor format
                - block_size: Size of blocks for HiCOO format (default: 128)
                - memory_factor: Memory usage factor (default: 0.9)
                - format: Sparse format to use ('hicoo')
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
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize backend
        from .backend import CPWOPTBackend
        self.backend = CPWOPTBackend(use_gpu)
        
        # Initialize sparse configuration
        self.sparse_config = {
            'block_size': HICOO_BLOCK_SIZE,
            'memory_factor': HICOO_MEMORY_FACTOR,
            'format': 'hicoo'
        }
        
        # Initialize factor updater
        from .factor_updater import FactorUpdater
        self.updater = FactorUpdater(method=optimizer, backend=self.backend)
        
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
        
        # Transfer tensor to device if using GPU
        if self.backend.device == 'gpu':
            tensor_data = self.backend.transfer_tensor_data(tensor_data)
        
        # Optimize based on selected optimizer
        if self.optimizer == 'als':
            factors, metrics = self._fit_als(tensor_data, factors)
        elif self.optimizer in ['gradient', 'sgd']:
            factors, metrics = self._fit_sgd(tensor_data, factors)
        elif self.optimizer == 'sgd_momentum':
            factors, metrics = self._fit_sgd_momentum(tensor_data, factors)
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
        """Initialize factor matrices with validation.
        
        Args:
            tensor_data: Input tensor data
            
        Returns:
            Initialized CPFactors with validated dimensions
            
        Raises:
            ValueError: If factor dimensions don't match tensor shape
        """
        factors = initialize_factors(
            tensor_data.shape,
            self.rank,
            method=self.init_method,
            random_state=self.random_state
        )
        
        # Validate factor dimensions match tensor shape
        for i, (factor, dim) in enumerate(zip(factors, tensor_data.shape)):
            if factor.shape[0] != dim:
                raise ValueError(
                    f"Factor {i} dimension mismatch: "
                    f"got {factor.shape[0]}, expected {dim}"
                )
            if factor.shape[1] != self.rank:
                raise ValueError(
                    f"Factor {i} rank mismatch: "
                    f"got {factor.shape[1]}, expected {self.rank}"
                )
        
        mode_names = ['latitude', 'longitude', 'features']
        
        return CPFactors(
            factors=factors,
            rank=self.rank,
            mode_names=mode_names
        )
    
    def _fit_als(self, tensor_data: TensorData, 
                 factors: CPFactors) -> Tuple[CPFactors, TensorMetrics]:
        """Fit using Alternating Least Squares with optimized GPU memory management.
        
        Keeps data on device throughout iterations to avoid transfer overhead.
        """
        # Get backend array module (numpy or cupy)
        xp = self.backend.get_array_module()
        
        # Transfer tensor data to device ONCE at the beginning
        try:
            tensor = self.backend.to_device(tensor_data.tensor)
            weights = self.backend.to_device(tensor_data.weights)
            tensor_clean = xp.nan_to_num(tensor, nan=NAN_REPLACEMENT_VALUE)
        except Exception as e:
            if 'out of memory' in str(e).lower():
                logger.error(ERROR_CUDA_OOM)
                raise RuntimeError(ERROR_CUDA_OOM) from e
            else:
                logger.error(ERROR_CUDA_DEVICE.format(str(e)))
                raise RuntimeError(ERROR_CUDA_DEVICE.format(str(e))) from e
        
        # CRITICAL FIX: Convert to HiCOO ONCE before the loop!
        # This was being done inside _update_factor_als_device on EVERY iteration
        from .sparse_handler import SparseHandler
        handler = SparseHandler()  # No config needed - HiCOO only
        
        # Create TensorData for HiCOO conversion (done ONCE)
        # Use empty coordinates for temporary tensor (validation allows this)
        feature_names = [f"feature_{i}" for i in range(tensor.shape[2])]
        tensor_data_hicoo = TensorData(
            tensor=tensor_clean,
            weights=weights,
            coordinates=[],  # Empty is allowed for grid-based tensors
            feature_names=feature_names,
            metadata={'temporary': True, 'original_shape': tensor.shape},
            is_sparse=True
        )
        hicoo_tensor = handler.to_sparse(tensor_data_hicoo)  # Converts to HiCOO
        
        # Transfer factors to device ONCE
        for mode in range(len(factors.factors)):
            factors.factors[mode] = self.backend.to_device(factors.factors[mode])
        
        # Pre-compute tensor norm ONCE (doesn't change)
        tensor_norm = float(xp.sqrt(xp.sum(tensor_clean * tensor_clean * weights)))
        
        # Track convergence
        errors = []
        prev_error = np.inf
        
        # Main ALS loop - ALL operations stay on device
        for iteration in range(self.max_iterations):
            
            # Update each mode (on device)
            for mode in range(tensor.ndim):
                # Perform ALS update directly on device with pre-converted HiCOO
                updated_factor = self._update_factor_als_device(
                    hicoo_tensor, handler, factors.factors, mode, xp
                )
                factors.factors[mode] = updated_factor
            
            # Calculate error on device (only transfer final scalar)
            error = self._compute_error_device(
                tensor_clean, weights, factors.factors, xp
            )
            error = float(error)  # Only transfer this scalar
            
            
            # Calculate relative error and fit (on device)
            rel_error = error / (tensor_norm + EPSILON)
            fit = 1.0 - rel_error
            
            errors.append(error)
            
            # Log progress
            if self.verbose and iteration % PROGRESS_INTERVAL == 0:
                logger.info(f"Iteration {iteration}: fit={fit:.4f}, error={error:.4e}")
                
                # Monitor GPU memory less frequently (every 50 iterations)
                # to avoid overhead in the hot path
                if self.backend.device == 'gpu' and iteration % 50 == 0:
                    mem_info = self.backend.get_memory_info()
                    if mem_info.get('memory_pressure') == 'high':
                        logger.warning(f"High GPU memory usage: {mem_info.get('usage_percent', 0):.1f}%")
                
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
        
        # PERFORMANCE: Batch transfer factors back to CPU
        # Transfer all factors at once to minimize overhead
        factors.factors = [self.backend.to_cpu(f) for f in factors.factors]
        
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
    
    def _fit_sgd(self, tensor_data: TensorData, 
                 initial_factors: CPFactors) -> Tuple[CPFactors, TensorMetrics]:
        """Fit CP decomposition using stochastic gradient descent.
        
        Uses the gradient update from factor_updater.py.
        
        Args:
            tensor_data: Input tensor with weight mask
            initial_factors: Initial factor matrices
            
        Returns:
            Tuple of (optimized factors, metrics)
        """
        if self.verbose:
            logger.info(f"Starting SGD optimization (lr={self.learning_rate})")
        
        factors = initial_factors
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            # Update each factor using gradient descent
            for mode in range(len(factors.factors)):
                factors.factors[mode] = self.updater._update_gradient(
                    tensor_data.values,
                    tensor_data.weights,
                    factors.factors,
                    mode,
                    learning_rate=self.learning_rate,
                    momentum=0.0,  # No momentum for basic SGD
                    n_iterations=1  # Single gradient step per mode
                )
            
            # Check convergence periodically
            if iteration % 10 == 0:
                error = weighted_norm(
                    tensor_data.values - cp_reconstruct(factors),
                    tensor_data.weights
                )
                convergence_history.append(error)
                
                if self.verbose and iteration % 50 == 0:
                    logger.info(f"Iteration {iteration}: error={error:.6f}")
                
                if len(convergence_history) > 1:
                    if check_convergence(convergence_history, self.convergence_threshold):
                        if self.verbose:
                            logger.info(f"Converged at iteration {iteration}")
                        break
            
            # Update iteration count
            self.n_iterations_ = iteration + 1
        
        # Compute final metrics
        metrics = self._compute_metrics(tensor_data, factors)
        self.convergence_history_ = convergence_history
        
        return factors, metrics
    
    def _fit_sgd_momentum(self, tensor_data: TensorData,
                         initial_factors: CPFactors) -> Tuple[CPFactors, TensorMetrics]:
        """Fit CP decomposition using SGD with momentum.
        
        Uses the existing _update_gradient from factor_updater.py with momentum.
        
        Args:
            tensor_data: Input tensor with weight mask
            initial_factors: Initial factor matrices
            
        Returns:
            Tuple of (optimized factors, metrics)
        """
        if self.verbose:
            logger.info(f"Starting SGD with momentum (lr={self.learning_rate}, "
                       f"momentum={self.momentum})")
        
        factors = initial_factors
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            # Update each factor using gradient descent with momentum
            for mode in range(len(factors.factors)):
                factors.factors[mode] = self.updater._update_gradient(
                    tensor_data.values,
                    tensor_data.weights,
                    factors.factors,
                    mode,
                    learning_rate=self.learning_rate,
                    momentum=self.momentum,
                    n_iterations=1  # Single gradient step per mode
                )
            
            # Check convergence periodically
            if iteration % 10 == 0:
                error = weighted_norm(
                    tensor_data.values - cp_reconstruct(factors),
                    tensor_data.weights
                )
                convergence_history.append(error)
                
                if self.verbose and iteration % 50 == 0:
                    logger.info(f"Iteration {iteration}: error={error:.6f}")
                
                if len(convergence_history) > 1:
                    if check_convergence(convergence_history, self.convergence_threshold):
                        if self.verbose:
                            logger.info(f"Converged at iteration {iteration}")
                        break
            
            # Callback for progress tracking
            if self.progress_callback:
                self.progress_callback(iteration, error)
            
            # Update iteration count
            self.n_iterations_ = iteration + 1
        
        # Compute final metrics
        metrics = self._compute_metrics(tensor_data, factors)
        self.convergence_history_ = convergence_history
        
        return factors, metrics
    
    def _update_factor_als_device(self, hicoo_tensor, handler, factors: List, 
                                  mode: int, xp):
        """Update factor on device using pre-converted HiCOO MTTKRP.
        
        PERFORMANCE FIX: HiCOO tensor is now pre-converted ONCE before the loop
        instead of being converted on EVERY iteration!
        
        Args:
            hicoo_tensor: Pre-converted HiCOO tensor (converted ONCE)
            handler: SparseHandler instance for operations
            factors: Factor matrices on device
            mode: Mode to update
            xp: Array module (numpy or cupy)
            
        Returns:
            Updated factor matrix on device
        """
        # CRITICAL FIX: No more HiCOO conversion here!
        # The tensor is already in HiCOO format
        
        # Directly use the pre-converted HiCOO tensor for MTTKRP
        # The handler wraps HiCOOHandler, so use its hicoo instance
        mttkrp_result = handler.hicoo.hicoo_mttkrp(
            hicoo_tensor, factors, mode, self.backend
        )
        
        # Compute Gram matrices for other modes
        gram_matrices = []
        for m in range(len(factors)):
            if m != mode:
                gram = xp.dot(factors[m].T, factors[m])
                gram_matrices.append(gram)
        
        # Hadamard product of Gram matrices
        gram_product = gram_matrices[0]
        for g in gram_matrices[1:]:
            gram_product = gram_product * g
        
        # Solve for updated factor (with regularization for stability)
        reg = REGULARIZATION_LAMBDA * xp.eye(gram_product.shape[0])
        updated_factor = xp.linalg.solve(gram_product + reg, mttkrp_result.T).T
        
        return updated_factor
    
    def _compute_error_device(self, tensor, weights, factors: List, xp):
        """Compute reconstruction error on device.
        
        Args:
            tensor: Tensor on device
            weights: Weights on device  
            factors: Factor matrices on device
            xp: Array module (numpy or cupy)
            
        Returns:
            Scalar error value on device
        """
        # Efficient error computation using Frobenius norm
        # ||W * (X - [[factors]])||_F^2
        
        # Compute factor Gram matrices
        gram_matrices = [xp.dot(f.T, f) for f in factors]
        
        # Compute ||[[factors]]||_F^2 efficiently
        factor_norm_sq = xp.ones((factors[0].shape[1],))
        for gram in gram_matrices:
            factor_norm_sq = factor_norm_sq * xp.diag(gram)
        factor_norm_sq = xp.sum(factor_norm_sq)
        
        # Compute <X, [[factors]]> using MTTKRP for all modes
        # This is more efficient than reconstructing full tensor
        inner_product = 0
        for mode in range(len(factors)):
            mttkrp = xp.zeros_like(factors[mode])
            
            # Compute MTTKRP for each mode
            if len(factors) == 3:  # 3D tensor case
                for r in range(factors[0].shape[1]):
                    if mode == 0:
                        slice_prod = xp.outer(factors[1][:, r], factors[2][:, r])
                        mttkrp[:, r] = xp.einsum('ijk,jk->i', tensor * weights, slice_prod)
                    elif mode == 1:
                        slice_prod = xp.outer(factors[0][:, r], factors[2][:, r])
                        mttkrp[:, r] = xp.einsum('ijk,ik->j', tensor * weights, slice_prod)
                    else:  # mode == 2
                        slice_prod = xp.outer(factors[0][:, r], factors[1][:, r])
                        mttkrp[:, r] = xp.einsum('ijk,ij->k', tensor * weights, slice_prod)
            else:
                # General n-dimensional case
                raise NotImplementedError("Error computation for n-D tensors not yet implemented")
            
            inner_product += xp.sum(factors[mode] * mttkrp)
        
        # Compute error: ||X||^2 - 2<X, [[factors]]> + ||[[factors]]||^2
        tensor_norm_sq = xp.sum(tensor * tensor * weights)
        error = xp.sqrt(tensor_norm_sq - 2*inner_product + factor_norm_sq)
        
        return error
    
    def _update_factor_als(self, tensor, weights,
                          factors: List, mode: int):
        """Update one factor matrix using weighted ALS.
        
        This solves: A = X_(n) (C ⊙ B)^T [(C^T C) * (B^T B)]^{-1}
        with modifications for missing values.
        
        Args:
            tensor: Input tensor
            weights: Weight mask
            factors: List of factor matrices
            mode: Mode to update
            
        Returns:
            Updated factor matrix
            
        Raises:
            ValueError: If input shapes are invalid
        """
        # Input validation
        if not isinstance(factors, list) or len(factors) != tensor.ndim:
            raise ValueError(f"Expected {tensor.ndim} factors, got {len(factors)}")
        if tensor.shape != weights.shape:
            raise ValueError(f"Tensor shape {tensor.shape} != weights shape {weights.shape}")
        if mode >= tensor.ndim:
            raise ValueError(f"Mode {mode} out of range for {tensor.ndim}D tensor")
            
        n_modes = len(factors)
        rank = factors[0].shape[1]
        mode_size = tensor.shape[mode]
        
        # Validate factor dimensions
        for i, factor in enumerate(factors):
            if factor.shape != (tensor.shape[i], rank):
                raise ValueError(f"Factor {i} shape {factor.shape} incompatible with tensor")
        
        # Compute Gram matrices for other modes
        grams = []
        for m in range(n_modes):
            if m != mode:
                grams.append(factors[m].T @ factors[m])
        
        # Compute Hadamard product of Gram matrices once
        gram_product = grams[0].copy()
        for g in grams[1:]:
            gram_product *= g  # Element-wise product
        gram_product += np.eye(rank) * EPSILON  # Regularization
        
        # Pre-compute the Khatri-Rao product of other factors
        # This is the most expensive operation - do it once outside the loop
        other_factors = [factors[m] for m in range(n_modes) if m != mode]
        V = self._compute_khatri_rao_product(other_factors)
        
        # Unfold tensor and weights along mode
        tensor_unfold = unfold(tensor, mode)
        weights_unfold = unfold(weights, mode)
        
        # Vectorized update for all rows at once
        # Find rows with non-zero weights
        valid_rows = np.sum(weights_unfold, axis=1) > 0
        n_valid = int(np.sum(valid_rows))
        
        if n_valid == 0:
            # No valid data, return random initialization
            logger.warning(f"No valid data for mode {mode}, using random initialization")
            return np.random.rand(mode_size, rank) * 0.01
        
        # Initialize new factor
        new_factor = np.zeros((mode_size, rank))
        
        # Batch process valid rows for better performance
        batch_size = min(1000, n_valid)  # Process in chunks to balance memory/speed
        valid_indices = np.where(valid_rows)[0]
        
        for start_idx in range(0, n_valid, batch_size):
            end_idx = min(start_idx + batch_size, n_valid)
            batch_indices = valid_indices[start_idx:end_idx]
            
            # Get batch data
            batch_tensor = tensor_unfold[batch_indices, :]
            batch_weights = weights_unfold[batch_indices, :]
            
            # MEMORY OPTIMIZATION: Avoid creating large intermediate arrays
            # Process in smaller sub-batches if needed
            lhs_batch = np.zeros((len(batch_indices), rank))
            
            # Check estimated memory usage
            estimated_memory_mb = (batch_weights.shape[0] * batch_weights.shape[1] * rank * 8) / (1024**2)
            if estimated_memory_mb > 100:  # If over 100MB, process in sub-batches
                sub_batch_size = max(1, int(100 * 1024 * 1024 / (batch_weights.shape[1] * rank * 8)))
                for sub_start in range(0, len(batch_indices), sub_batch_size):
                    sub_end = min(sub_start + sub_batch_size, len(batch_indices))
                    
                    # Process sub-batch without creating full V_weighted
                    for i in range(sub_start, sub_end):
                        weighted_row = batch_tensor[i, :] * batch_weights[i, :]
                        lhs_batch[i, :] = weighted_row @ V
            else:
                # Original approach for smaller batches
                for i in range(len(batch_indices)):
                    # Element-wise multiply data by weights, then multiply by KR product
                    weighted_row = batch_tensor[i, :] * batch_weights[i, :]
                    lhs_batch[i, :] = weighted_row @ V
            
            # Solve for all rows in batch
            try:
                # Use Cholesky decomposition for speed (gram_product is positive definite)
                L = np.linalg.cholesky(gram_product)
                y = np.linalg.solve(L, lhs_batch.T)
                new_factor[batch_indices, :] = np.linalg.solve(L.T, y).T
            except np.linalg.LinAlgError:
                # Fall back to standard solve
                new_factor[batch_indices, :] = np.linalg.solve(gram_product, lhs_batch.T).T
        
        # Normalize columns to prevent numerical issues
        norms = np.linalg.norm(new_factor, axis=0)
        valid_norms = norms > MIN_COMPONENT_NORM
        
        # BUGFIX: Check if there are any valid norms before indexing
        if np.any(valid_norms):
            # Vectorized normalization - only for valid components
            new_factor[:, valid_norms] /= norms[valid_norms]
        else:
            logger.warning(f"No valid norms found for mode {mode}, all components near zero")
        
        # Reinitialize near-zero components
        if not np.all(valid_norms):
            new_factor[:, ~valid_norms] = np.random.rand(mode_size, np.sum(~valid_norms)) * 0.01
                
        return new_factor
    
    def _compute_khatri_rao_product(self, factors: List[np.ndarray]) -> np.ndarray:
        """Compute Khatri-Rao product of factors efficiently.
        
        Args:
            factors: List of factor matrices
            
        Returns:
            Khatri-Rao product
        """
        if not factors:
            raise ValueError("No factors provided")
            
        result = factors[0]
        for factor in factors[1:]:
            # Efficient Khatri-Rao using broadcasting
            n1, r = result.shape
            n2 = factor.shape[0]
            kr_product = (result[:, np.newaxis, :] * factor[np.newaxis, :, :]).reshape(n1 * n2, r)
            result = kr_product
            
        return result
    
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
            Reconstructed tensor (as numpy array)
        """
        if self.factors_ is None:
            raise ValueError("Model must be fitted before predict")
            
        # Convert factors to CPU/numpy for reconstruction
        factors_cpu = [self.backend.to_cpu(f) for f in self.factors_.factors]
        return cp_reconstruct(factors_cpu)
    
    def reconstruct(self) -> np.ndarray:
        """Alias for predict() - reconstruct the tensor from learned factors.
        
        Returns:
            Reconstructed tensor
        """
        return self.predict()
    
    def get_component_matrices(self) -> Dict[str, np.ndarray]:
        """Get factor matrices organized by mode name.
        
        Returns:
            Dictionary mapping mode name to factor matrix (as numpy arrays)
        """
        if self.factors_ is None:
            raise ValueError("Model must be fitted first")
            
        # Convert to CPU/numpy arrays if on GPU
        return {
            name: self.backend.to_cpu(factor)
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
        
        # Get array module (numpy or cupy)
        xp = self.backend.get_array_module()
        
        for mode_name, factor in zip(self.factors_.mode_names, self.factors_.factors):
            # Get loadings for this component (using appropriate array module)
            loadings = xp.abs(factor[:, component])
            
            # Get top indices
            top_idx = xp.argsort(loadings)[-n_top:][::-1]
            
            # Convert to CPU/numpy for output
            top_features[mode_name] = {
                'indices': self.backend.to_cpu(top_idx),
                'loadings': self.backend.to_cpu(loadings[top_idx])
            }
            
        return top_features
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """Set progress callback function.
        
        Args:
            callback: Function called with (current, total, message)
        """
        self.progress_callback = callback