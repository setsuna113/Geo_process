"""Mini-batch processing for memory-efficient CP-WOPT.

This module provides utilities for processing large tensors in mini-batches,
enabling CP-WOPT to handle datasets that exceed available memory.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Generator, Callable
from dataclasses import dataclass
import psutil
from joblib import Parallel, delayed

from src.abstractions.types.tensor_types import TensorData, MiniBatchConfig
from src.biodiversity_analysis.shared.tensor_utils import (
    unfold, fold, compute_mttkrp, weighted_norm
)
from .constants import *

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Results from processing a single batch."""
    batch_idx: int
    mttkrp: Optional[np.ndarray] = None
    gram_contribution: Optional[np.ndarray] = None
    reconstruction_error: Optional[float] = None
    n_samples: int = 0


class TensorMiniBatchProcessor:
    """Handles mini-batch processing for tensor factorization.
    
    Key features:
    - Processes tensor slices to avoid memory overflow
    - Supports parallel batch processing
    - Accumulates results for factor updates
    - Memory-aware batch size adjustment
    """
    
    def __init__(self, config: MiniBatchConfig):
        """Initialize mini-batch processor.
        
        Args:
            config: Mini-batch configuration
        """
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
    def generate_batches(self, tensor_shape: Tuple[int, ...], 
                        mode: int = 0) -> Generator[Tuple[int, int], None, None]:
        """Generate batch indices along specified mode.
        
        Args:
            tensor_shape: Shape of the tensor
            mode: Mode to batch along (default: 0)
            
        Yields:
            Tuples of (start_idx, end_idx) for each batch
        """
        n_slices = tensor_shape[mode]
        batch_size = min(self.config.batch_size, n_slices)
        
        # Adjust batch size based on available memory
        if self.memory_monitor.is_memory_constrained():
            batch_size = max(batch_size // 2, MIN_MINIBATCH_SIZE)
            logger.warning(f"Memory constrained, reducing batch size to {batch_size}")
        
        indices = list(range(n_slices))
        
        # Shuffle if requested
        if self.config.shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        for start in range(0, n_slices, batch_size):
            end = min(start + batch_size, n_slices)
            
            # Skip last batch if too small and drop_last is True
            if self.config.drop_last and end - start < batch_size:
                continue
                
            yield start, end
    
    def extract_batch(self, tensor: np.ndarray, weights: np.ndarray,
                     batch_indices: Tuple[int, int], 
                     mode: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a batch slice from tensor and weights.
        
        Args:
            tensor: Full tensor
            weights: Weight mask
            batch_indices: (start, end) indices
            mode: Mode to slice along
            
        Returns:
            Tuple of (batch_tensor, batch_weights)
        """
        start, end = batch_indices
        
        # Create slice indices for all dimensions
        slices = [slice(None)] * tensor.ndim
        slices[mode] = slice(start, end)
        
        batch_tensor = tensor[tuple(slices)]
        batch_weights = weights[tuple(slices)]
        
        return batch_tensor, batch_weights
    
    def process_batch_als(self, batch_tensor: np.ndarray, 
                         batch_weights: np.ndarray,
                         factors: List[np.ndarray],
                         mode: int,
                         batch_idx: int) -> BatchResult:
        """Process a single batch for ALS update.
        
        Args:
            batch_tensor: Batch slice of tensor
            batch_weights: Batch slice of weights
            factors: Current factor matrices
            mode: Mode being updated
            batch_idx: Batch index
            
        Returns:
            BatchResult with MTTKRP and Gram contributions
        """
        # Replace NaN with 0
        batch_clean = np.nan_to_num(batch_tensor, nan=0.0)
        
        # Compute weighted MTTKRP for this batch
        mttkrp_batch = self._compute_weighted_mttkrp(
            batch_clean, batch_weights, factors, mode
        )
        
        # Compute Gram matrix contribution
        gram_contrib = self._compute_gram_contribution(
            batch_weights, factors, mode
        )
        
        # Count valid samples
        n_samples = np.sum(np.any(batch_weights > 0, axis=tuple(
            i for i in range(batch_weights.ndim) if i != mode
        )))
        
        return BatchResult(
            batch_idx=batch_idx,
            mttkrp=mttkrp_batch,
            gram_contribution=gram_contrib,
            n_samples=int(n_samples)
        )
    
    def _compute_weighted_mttkrp(self, tensor: np.ndarray, 
                               weights: np.ndarray,
                               factors: List[np.ndarray],
                               mode: int) -> np.ndarray:
        """Compute weighted MTTKRP for a batch.
        
        This is the key operation: X_(n) ⊙ W_(n) (C ⊙ B ⊙ A)
        """
        # Apply weights to tensor
        weighted_tensor = tensor * weights
        
        # Compute standard MTTKRP
        mttkrp = compute_mttkrp(weighted_tensor, factors, mode)
        
        return mttkrp
    
    def _compute_gram_contribution(self, weights: np.ndarray,
                                 factors: List[np.ndarray],
                                 mode: int) -> np.ndarray:
        """Compute contribution to Gram matrix from batch.
        
        For weighted ALS, we need to accumulate weighted Gram matrices.
        """
        rank = factors[0].shape[1]
        gram = np.zeros((rank, rank))
        
        # This is a simplified version - full implementation would
        # compute the exact weighted Hadamard product of Gram matrices
        # based on the weight pattern in this batch
        
        # For now, return identity scaled by batch weight
        batch_weight = np.mean(weights)
        gram = np.eye(rank) * batch_weight
        
        return gram
    
    def parallel_process_batches(self, tensor_data: TensorData,
                               factors: List[np.ndarray],
                               mode: int,
                               n_workers: int = -1) -> Dict[str, np.ndarray]:
        """Process all batches in parallel for one mode update.
        
        Args:
            tensor_data: Full tensor data
            factors: Current factor matrices
            mode: Mode to update
            n_workers: Number of parallel workers (-1 for all CPUs)
            
        Returns:
            Dictionary with accumulated results
        """
        tensor = tensor_data.tensor
        weights = tensor_data.weights
        
        # Generate batches
        batches = list(self.generate_batches(tensor.shape, mode))
        n_batches = len(batches)
        
        logger.info(f"Processing {n_batches} batches for mode {mode} "
                   f"with {n_workers} workers")
        
        # Process batches in parallel
        results = Parallel(n_jobs=n_workers, backend='threading')(
            delayed(self._process_single_batch_wrapper)(
                tensor, weights, factors, mode, i, batch_indices
            )
            for i, batch_indices in enumerate(batches)
        )
        
        # Accumulate results
        mode_size = tensor.shape[mode]
        rank = factors[0].shape[1]
        
        accumulated_mttkrp = np.zeros((mode_size, rank))
        accumulated_gram = np.zeros((rank, rank))
        total_samples = 0
        
        for result in results:
            if result.mttkrp is not None:
                start = batches[result.batch_idx][0]
                end = batches[result.batch_idx][1]
                accumulated_mttkrp[start:end] = result.mttkrp
                
            if result.gram_contribution is not None:
                accumulated_gram += result.gram_contribution
                
            total_samples += result.n_samples
        
        return {
            'mttkrp': accumulated_mttkrp,
            'gram': accumulated_gram,
            'n_samples': total_samples
        }
    
    def _process_single_batch_wrapper(self, tensor: np.ndarray,
                                    weights: np.ndarray,
                                    factors: List[np.ndarray],
                                    mode: int,
                                    batch_idx: int,
                                    batch_indices: Tuple[int, int]) -> BatchResult:
        """Wrapper for parallel batch processing."""
        batch_tensor, batch_weights = self.extract_batch(
            tensor, weights, batch_indices, mode
        )
        
        return self.process_batch_als(
            batch_tensor, batch_weights, factors, mode, batch_idx
        )
    
    def compute_reconstruction_error_batched(self, tensor_data: TensorData,
                                           factors: List[np.ndarray],
                                           mode: int = 0) -> float:
        """Compute reconstruction error using mini-batches.
        
        Args:
            tensor_data: Original tensor data
            factors: CP factors
            mode: Mode to batch along
            
        Returns:
            Weighted reconstruction error
        """
        tensor = tensor_data.tensor
        weights = tensor_data.weights
        
        total_error = 0.0
        total_weight = 0.0
        
        # Process in batches
        for start, end in self.generate_batches(tensor.shape, mode):
            # Extract batch
            batch_tensor, batch_weights = self.extract_batch(
                tensor, weights, (start, end), mode
            )
            
            # Reconstruct this batch
            # This is simplified - full implementation would reconstruct
            # only the necessary slice
            batch_reconstructed = self._reconstruct_batch(
                factors, (start, end), mode, tensor.shape
            )
            
            # Compute weighted error for batch
            batch_clean = np.nan_to_num(batch_tensor, nan=0.0)
            batch_residual = (batch_clean - batch_reconstructed) * batch_weights
            
            batch_error = np.sum(batch_residual ** 2)
            batch_weight_sum = np.sum(batch_weights)
            
            total_error += batch_error
            total_weight += batch_weight_sum
        
        # Return weighted RMSE
        if total_weight > 0:
            return np.sqrt(total_error / total_weight)
        else:
            return np.inf
    
    def _reconstruct_batch(self, factors: List[np.ndarray],
                         batch_indices: Tuple[int, int],
                         mode: int,
                         full_shape: Tuple[int, ...]) -> np.ndarray:
        """Reconstruct a batch slice from factors.
        
        This is a placeholder - full implementation would efficiently
        reconstruct only the needed slice without materializing the full tensor.
        """
        # For now, return zeros with correct shape
        batch_shape = list(full_shape)
        batch_shape[mode] = batch_indices[1] - batch_indices[0]
        
        return np.zeros(tuple(batch_shape))


class MemoryMonitor:
    """Monitor system memory to adjust batch sizes."""
    
    def __init__(self, threshold: float = 0.8):
        """Initialize memory monitor.
        
        Args:
            threshold: Memory usage threshold (0-1)
        """
        self.threshold = threshold
        
    def get_memory_usage(self) -> float:
        """Get current memory usage as fraction."""
        try:
            return psutil.virtual_memory().percent / 100.0
        except:
            # If psutil fails, assume we have memory
            return 0.5
    
    def is_memory_constrained(self) -> bool:
        """Check if memory usage is above threshold."""
        return self.get_memory_usage() > self.threshold
    
    def estimate_batch_size(self, tensor_shape: Tuple[int, ...],
                          rank: int,
                          mode: int = 0) -> int:
        """Estimate appropriate batch size based on available memory.
        
        Args:
            tensor_shape: Shape of full tensor
            rank: Number of components
            mode: Mode to batch along
            
        Returns:
            Recommended batch size
        """
        # Get available memory in bytes
        try:
            available = psutil.virtual_memory().available
        except:
            # Default to 1GB if psutil fails
            available = 1e9
        
        # Estimate memory per slice
        slice_shape = list(tensor_shape)
        slice_shape[mode] = 1
        
        # Memory for tensor slice + weights + factor updates
        bytes_per_float = 8
        slice_memory = (
            np.prod(slice_shape) * bytes_per_float * 2 +  # tensor + weights
            tensor_shape[mode] * rank * bytes_per_float * 3  # factors
        )
        
        # Use 50% of available memory for safety
        safe_memory = available * 0.5
        
        # Calculate batch size
        batch_size = int(safe_memory / slice_memory)
        
        # Clamp to reasonable range
        batch_size = max(MIN_MINIBATCH_SIZE, 
                        min(batch_size, MAX_MINIBATCH_SIZE))
        
        logger.info(f"Estimated batch size: {batch_size} "
                   f"(available memory: {available/1e9:.1f}GB)")
        
        return batch_size


def create_minibatch_processor(tensor_data: TensorData,
                             rank: int,
                             auto_adjust: bool = True) -> TensorMiniBatchProcessor:
    """Create mini-batch processor with appropriate configuration.
    
    Args:
        tensor_data: Tensor data to process
        rank: Number of components
        auto_adjust: Whether to auto-adjust batch size
        
    Returns:
        Configured TensorMiniBatchProcessor
    """
    # Create base config
    config = MiniBatchConfig()
    
    if auto_adjust:
        # Estimate appropriate batch size
        monitor = MemoryMonitor()
        estimated_size = monitor.estimate_batch_size(
            tensor_data.shape, rank
        )
        config.batch_size = estimated_size
        
    # Adjust based on sparsity
    if tensor_data.sparsity > SPARSITY_THRESHOLD:
        # Use smaller batches for very sparse data
        config.batch_size = max(config.batch_size // 2, MIN_MINIBATCH_SIZE)
        logger.info(f"High sparsity detected, using batch size: {config.batch_size}")
    
    return TensorMiniBatchProcessor(config)