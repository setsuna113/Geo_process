"""Sparse tensor handling for extreme sparsity in biodiversity data.

This module provides specialized handling for tensors with >95% sparsity,
including sparse storage formats and efficient operations.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
from dataclasses import dataclass
import warnings

from src.abstractions.types.tensor_types import TensorData, CPFactors
from .constants import SPARSITY_THRESHOLD, MIN_OBSERVATIONS_PER_CELL, EPSILON

logger = logging.getLogger(__name__)


@dataclass
class SparseTensorData:
    """Sparse representation of tensor data.
    
    Stores only non-zero entries with their indices and values.
    """
    indices: np.ndarray  # Shape: (nnz, ndims)
    values: np.ndarray   # Shape: (nnz,)
    weights: np.ndarray  # Shape: (nnz,)
    shape: Tuple[int, ...]
    
    @property
    def nnz(self) -> int:
        """Number of non-zero entries."""
        return len(self.values)
    
    @property
    def sparsity(self) -> float:
        """Proportion of zero entries."""
        total_elements = np.prod(self.shape)
        return 1.0 - (self.nnz / total_elements)
    
    @property
    def density(self) -> float:
        """Proportion of non-zero entries."""
        return 1.0 - self.sparsity


class SparseHandler:
    """Handles operations on extremely sparse tensors."""
    
    def __init__(self, force_sparse: bool = False):
        """Initialize sparse handler.
        
        Args:
            force_sparse: Force sparse format even below threshold
        """
        self.force_sparse = force_sparse
        self._cache = {}
        
    def to_sparse(self, tensor_data: TensorData) -> SparseTensorData:
        """Convert dense tensor to sparse format.
        
        Args:
            tensor_data: Dense tensor data
            
        Returns:
            Sparse tensor representation
        """
        tensor = tensor_data.tensor
        weights = tensor_data.weights
        
        # Find non-zero or non-NaN entries
        if np.isnan(tensor).any():
            valid_mask = ~np.isnan(tensor) & (weights > 0)
        else:
            valid_mask = (tensor != 0) & (weights > 0)
        
        # Get indices of valid entries
        indices = np.array(np.where(valid_mask)).T
        
        # Extract values and weights
        values = tensor[valid_mask]
        sparse_weights = weights[valid_mask]
        
        # Replace any remaining NaN with 0
        values = np.nan_to_num(values, nan=0.0)
        
        sparse_data = SparseTensorData(
            indices=indices,
            values=values,
            weights=sparse_weights,
            shape=tensor.shape
        )
        
        logger.info(f"Converted to sparse: {sparse_data.nnz:,} non-zero entries "
                   f"({sparse_data.density:.2%} density)")
        
        return sparse_data
    
    def to_dense(self, sparse_data: SparseTensorData) -> TensorData:
        """Convert sparse tensor back to dense format.
        
        Args:
            sparse_data: Sparse tensor data
            
        Returns:
            Dense tensor representation
        """
        # Initialize dense arrays
        tensor = np.zeros(sparse_data.shape)
        weights = np.zeros(sparse_data.shape)
        
        # Fill in non-zero values
        for i, (idx, val, w) in enumerate(zip(sparse_data.indices, 
                                             sparse_data.values,
                                             sparse_data.weights)):
            tensor[tuple(idx)] = val
            weights[tuple(idx)] = w
        
        # Create TensorData (would need additional fields in practice)
        return TensorData(
            tensor=tensor,
            weights=weights,
            coordinates=[],  # Placeholder
            feature_names=[],  # Placeholder
            metadata={},
            is_sparse=False
        )
    
    def sparse_mttkrp(self, sparse_data: SparseTensorData,
                     factors: List[np.ndarray], mode: int) -> np.ndarray:
        """Compute MTTKRP efficiently for sparse tensor.
        
        Matricized Tensor Times Khatri-Rao Product without forming full tensor.
        
        Args:
            sparse_data: Sparse tensor
            factors: Factor matrices
            mode: Mode for MTTKRP
            
        Returns:
            MTTKRP result matrix
        """
        mode_size = sparse_data.shape[mode]
        rank = factors[0].shape[1]
        
        # Initialize result
        result = np.zeros((mode_size, rank))
        
        # Process each non-zero entry
        for idx, val, weight in zip(sparse_data.indices, 
                                  sparse_data.values,
                                  sparse_data.weights):
            # Get the row index for this mode
            row_idx = idx[mode]
            
            # Compute the contribution from other modes
            contribution = val * weight
            
            for r in range(rank):
                # Product of factor values for other modes
                factor_prod = 1.0
                for m in range(len(factors)):
                    if m != mode:
                        factor_prod *= factors[m][idx[m], r]
                
                # Add to result
                result[row_idx, r] += contribution * factor_prod
        
        return result
    
    def sparse_reconstruction_error(self, sparse_data: SparseTensorData,
                                  factors: CPFactors) -> float:
        """Compute reconstruction error for sparse tensor.
        
        Only computes error at observed entries.
        
        Args:
            sparse_data: Sparse tensor
            factors: CP factors
            
        Returns:
            Weighted reconstruction error
        """
        total_error = 0.0
        total_weight = 0.0
        
        # Process each non-zero entry
        for idx, true_val, weight in zip(sparse_data.indices,
                                        sparse_data.values,
                                        sparse_data.weights):
            # Reconstruct value at this index
            reconstructed = 0.0
            for r in range(factors.rank):
                factor_prod = 1.0
                for m in range(len(factors.factors)):
                    factor_prod *= factors.factors[m][idx[m], r]
                reconstructed += factor_prod
            
            # Weighted squared error
            error = (true_val - reconstructed) ** 2
            total_error += error * weight
            total_weight += weight
        
        # Return RMSE
        if total_weight > 0:
            return np.sqrt(total_error / total_weight)
        else:
            return np.inf
    
    def sparse_als_update(self, sparse_data: SparseTensorData,
                         factors: List[np.ndarray], mode: int) -> np.ndarray:
        """Update factor matrix using sparse ALS.
        
        Efficient ALS update that only uses non-zero entries.
        
        Args:
            sparse_data: Sparse tensor
            factors: Current factor matrices
            mode: Mode to update
            
        Returns:
            Updated factor matrix
        """
        mode_size = sparse_data.shape[mode]
        rank = factors[0].shape[1]
        
        # Initialize new factor
        new_factor = np.zeros((mode_size, rank))
        
        # Build index mapping for efficient access
        row_indices = {}
        for i, idx in enumerate(sparse_data.indices):
            row = idx[mode]
            if row not in row_indices:
                row_indices[row] = []
            row_indices[row].append(i)
        
        # Update each row
        for row in range(mode_size):
            if row not in row_indices:
                # No data for this row
                continue
            
            # Get entries for this row
            entry_indices = row_indices[row]
            
            # Build matrices for this row's system
            V_list = []  # Khatri-Rao product entries
            w_list = []  # Weights
            y_list = []  # Target values
            
            for entry_idx in entry_indices:
                idx = sparse_data.indices[entry_idx]
                val = sparse_data.values[entry_idx]
                weight = sparse_data.weights[entry_idx]
                
                # Compute Khatri-Rao product for this entry
                kr_vec = np.ones(rank)
                for m in range(len(factors)):
                    if m != mode:
                        kr_vec *= factors[m][idx[m], :]
                
                V_list.append(kr_vec)
                w_list.append(weight)
                y_list.append(val)
            
            # Convert to arrays
            V = np.array(V_list)  # Shape: (n_entries, rank)
            w = np.array(w_list)  # Shape: (n_entries,)
            y = np.array(y_list)  # Shape: (n_entries,)
            
            # Weighted least squares for this row
            # min ||sqrt(W) * (y - V @ a)||^2
            W_sqrt = np.diag(np.sqrt(w))
            V_weighted = W_sqrt @ V
            y_weighted = W_sqrt @ y
            
            # Solve normal equations
            gram = V_weighted.T @ V_weighted + EPSILON * np.eye(rank)
            rhs = V_weighted.T @ y_weighted
            
            try:
                new_factor[row, :] = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                new_factor[row, :] = rhs @ np.linalg.pinv(gram)
        
        # Normalize
        for r in range(rank):
            norm = np.linalg.norm(new_factor[:, r])
            if norm > EPSILON:
                new_factor[:, r] /= norm
        
        return new_factor
    
    def estimate_memory_usage(self, shape: Tuple[int, ...], 
                            sparsity: float) -> Dict[str, float]:
        """Estimate memory usage for sparse vs dense storage.
        
        Args:
            shape: Tensor shape
            sparsity: Expected sparsity level
            
        Returns:
            Dictionary with memory estimates in GB
        """
        total_elements = np.prod(shape)
        nnz = int(total_elements * (1 - sparsity))
        ndims = len(shape)
        
        # Dense storage: tensor + weights
        dense_memory = total_elements * 8 * 2 / 1e9  # 8 bytes per float64
        
        # Sparse storage: indices + values + weights
        sparse_memory = (
            nnz * ndims * 8 +  # indices (int64)
            nnz * 8 +          # values (float64)
            nnz * 8            # weights (float64)
        ) / 1e9
        
        return {
            'dense_gb': dense_memory,
            'sparse_gb': sparse_memory,
            'compression_ratio': dense_memory / sparse_memory if sparse_memory > 0 else np.inf,
            'recommended': 'sparse' if sparse_memory < dense_memory * 0.5 else 'dense'
        }
    
    def create_sparse_unfolding(self, sparse_data: SparseTensorData,
                              mode: int) -> csr_matrix:
        """Create sparse unfolding along specified mode.
        
        Args:
            sparse_data: Sparse tensor
            mode: Mode to unfold along
            
        Returns:
            Sparse matrix in CSR format
        """
        # Compute unfolded dimensions
        mode_size = sparse_data.shape[mode]
        other_sizes = [s for i, s in enumerate(sparse_data.shape) if i != mode]
        other_size = np.prod(other_sizes)
        
        # Build row and column indices for unfolding
        rows = []
        cols = []
        vals = []
        
        for idx, val, weight in zip(sparse_data.indices,
                                   sparse_data.values, 
                                   sparse_data.weights):
            # Row is the mode index
            row = idx[mode]
            
            # Column is linearized index of other modes
            col_idx = 0
            stride = 1
            for i in range(len(sparse_data.shape)):
                if i != mode:
                    if i < mode:
                        col_idx += idx[i] * stride
                        stride *= sparse_data.shape[i]
                    else:
                        col_idx += idx[i] * stride
                        stride *= sparse_data.shape[i]
            
            rows.append(row)
            cols.append(col_idx)
            vals.append(val * weight)  # Apply weight
        
        # Create sparse matrix
        sparse_unfold = csr_matrix(
            (vals, (rows, cols)),
            shape=(mode_size, other_size)
        )
        
        return sparse_unfold


class AdaptiveSparseStrategy:
    """Adaptively switch between sparse and dense operations."""
    
    def __init__(self, threshold: float = SPARSITY_THRESHOLD):
        """Initialize adaptive strategy.
        
        Args:
            threshold: Sparsity threshold for switching to sparse
        """
        self.threshold = threshold
        self.sparse_handler = SparseHandler()
        
    def should_use_sparse(self, tensor_data: TensorData) -> bool:
        """Determine if sparse operations should be used.
        
        Args:
            tensor_data: Tensor data to check
            
        Returns:
            True if sparse operations recommended
        """
        # Check sparsity
        if hasattr(tensor_data, 'sparsity'):
            sparsity = tensor_data.sparsity
        else:
            # Calculate sparsity
            valid_mask = ~np.isnan(tensor_data.tensor) & (tensor_data.weights > 0)
            density = np.sum(valid_mask) / tensor_data.tensor.size
            sparsity = 1.0 - density
        
        # Check memory usage
        estimates = self.sparse_handler.estimate_memory_usage(
            tensor_data.shape, sparsity
        )
        
        # Use sparse if:
        # 1. Above sparsity threshold AND
        # 2. Sparse storage is more efficient
        use_sparse = (
            sparsity >= self.threshold and
            estimates['recommended'] == 'sparse'
        )
        
        if use_sparse:
            logger.info(f"Using sparse operations (sparsity: {sparsity:.1%}, "
                       f"compression ratio: {estimates['compression_ratio']:.1f}x)")
        
        return use_sparse
    
    def compute_mttkrp_adaptive(self, tensor_data: TensorData,
                              factors: List[np.ndarray],
                              mode: int) -> np.ndarray:
        """Compute MTTKRP using appropriate method.
        
        Args:
            tensor_data: Tensor data
            factors: Factor matrices
            mode: Mode for MTTKRP
            
        Returns:
            MTTKRP result
        """
        if self.should_use_sparse(tensor_data):
            # Convert to sparse and compute
            sparse_data = self.sparse_handler.to_sparse(tensor_data)
            return self.sparse_handler.sparse_mttkrp(sparse_data, factors, mode)
        else:
            # Use dense computation
            from src.biodiversity_analysis.shared.tensor_utils import compute_mttkrp
            weighted_tensor = tensor_data.tensor * tensor_data.weights
            return compute_mttkrp(weighted_tensor, factors, mode)
    
    def update_factor_adaptive(self, tensor_data: TensorData,
                             factors: List[np.ndarray],
                             mode: int) -> np.ndarray:
        """Update factor using appropriate method.
        
        Args:
            tensor_data: Tensor data
            factors: Current factors
            mode: Mode to update
            
        Returns:
            Updated factor matrix
        """
        if self.should_use_sparse(tensor_data):
            # Use sparse ALS
            sparse_data = self.sparse_handler.to_sparse(tensor_data)
            return self.sparse_handler.sparse_als_update(sparse_data, factors, mode)
        else:
            # Use dense ALS (would import from factor_updater)
            from .factor_updater import FactorUpdater
            updater = FactorUpdater('als')
            return updater.update_factor(
                tensor_data.tensor, 
                tensor_data.weights,
                factors, 
                mode
            )


def optimize_for_sparsity(tensor_data: TensorData) -> Union[TensorData, SparseTensorData]:
    """Optimize tensor representation based on sparsity.
    
    Args:
        tensor_data: Input tensor data
        
    Returns:
        Optimized representation (dense or sparse)
    """
    handler = SparseHandler()
    adaptive = AdaptiveSparseStrategy()
    
    if adaptive.should_use_sparse(tensor_data):
        return handler.to_sparse(tensor_data)
    else:
        return tensor_data