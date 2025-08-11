"""Parallel batch processing for memory-efficient SOM training.

PURPOSE: This module handles parallel processing of data chunks for efficient
batch SOM updates on large datasets. It splits data into manageable chunks
that can be processed in parallel across CPU cores.

IMPLEMENTATION:
- Processes data in chunks for memory efficiency and parallelization
- Each chunk contributes to the batch update using the standard formula: w_k = Σ(h_k * x) / Σ(h_k)
- Accumulates numerator and denominator across all chunks
- Applies single weight update after processing all chunks

NOTE: This implements batch SOM with parallel processing. For sequential
mini-batch training with explicit epoch control, see minibatch_trainer.py.
"""

import os

# CRITICAL: Set thread limits BEFORE importing NumPy!
# With threading backend, all workers share the same process.
# Setting this to >1 causes thread explosion (30 workers × 2 = 60 NumPy threads)
# This leads to massive oversubscription and poor performance.
# MUST be set before NumPy/OpenBLAS initialization!
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # For macOS
os.environ['BLIS_NUM_THREADS'] = '1'        # For BLIS backend

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
import logging
from joblib import Parallel, delayed
from multiprocessing import cpu_count
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from ..calculators.optimized_distance_calculator import OptimizedDistanceCalculator
from ..utils.constants import (
    INVALID_DISTANCE, DEFAULT_GEO_DISTANCE_NORM, DEFAULT_INTERNAL_CHUNK_SIZE
)

# Singleton distance calculator to avoid repeated initialization
_DISTANCE_CALCULATOR = None

def get_distance_calculator():
    """Get or create singleton distance calculator."""
    global _DISTANCE_CALCULATOR
    if _DISTANCE_CALCULATOR is None:
        _DISTANCE_CALCULATOR = OptimizedDistanceCalculator()
    return _DISTANCE_CALCULATOR

# NOTE: StreamlinedDistanceCalculator provides memory-efficient distance calculations.
# Real mini-batch training is implemented in minibatch_trainer.py with MiniBatchTrainer.

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Results from processing a single chunk."""
    numerator: np.ndarray
    denominator: np.ndarray
    chunk_idx: int
    n_valid_samples: int


def process_single_chunk(chunk_idx: int,
                        chunk_data: np.ndarray,
                        chunk_coords: Optional[np.ndarray],
                        weights: np.ndarray,
                        neuron_coords: Optional[np.ndarray],
                        config: Any,
                        grid_positions: np.ndarray,
                        current_radius: float,
                        use_chunking: bool = True,
                        chunk_size_internal: int = 1000) -> ChunkResult:
    """Process a single chunk of data - standalone function for parallel execution.
    
    This function is defined at module level to be pickleable for multiprocessing.
    """
    n_neurons = len(weights)
    min_valid_features = config.min_valid_features
    spatial_weight = config.spatial_weight
    
    # Check if chunk has any valid data
    valid_features_per_sample = (~np.isnan(chunk_data)).sum(axis=1)
    valid_samples = valid_features_per_sample >= min_valid_features
    
    if not valid_samples.any():
        return ChunkResult(
            numerator=np.zeros_like(weights),
            denominator=np.zeros(n_neurons),
            chunk_idx=chunk_idx,
            n_valid_samples=0
        )
    
    # Calculate distances for this chunk using optimized calculator
    # Note: OptimizedDistanceCalculator handles memory-efficient processing
    calculator = get_distance_calculator()
    
    if chunk_coords is not None and neuron_coords is not None:
        chunk_distances = calculator.combined_distance_chunked(
            chunk_data, weights, chunk_coords, neuron_coords,
            spatial_weight, min_valid_features
        )
    else:
        chunk_distances = calculator.partial_bray_curtis_chunked(
            chunk_data, weights, min_valid_features
        )
    
    # Find BMUs for chunk
    chunk_bmu_indices = np.argmin(chunk_distances, axis=1)
    min_distances = np.take_along_axis(
        chunk_distances, chunk_bmu_indices[:, np.newaxis], axis=1
    ).squeeze()
    
    # Mark invalid BMUs
    invalid_mask = np.isinf(min_distances)
    chunk_bmu_indices[invalid_mask] = -1
    n_valid = (~invalid_mask).sum()
    
    # Skip if all samples are invalid
    if invalid_mask.all():
        return ChunkResult(
            numerator=np.zeros_like(weights),
            denominator=np.zeros(n_neurons),
            chunk_idx=chunk_idx,
            n_valid_samples=0
        )
    
    # Calculate neighborhood influences for chunk
    chunk_influences = _vectorized_neighborhood(
        chunk_bmu_indices, grid_positions, current_radius
    )
    
    # Accumulate updates from this chunk
    chunk_data_clean = np.nan_to_num(chunk_data, nan=0.0)
    chunk_valid_mask = ~np.isnan(chunk_data)
    
    # Accumulate for BATCH SOM update: w_k = Σ(h_k * x) / Σ(h_k)
    
    # Expand dimensions for broadcasting: (samples, neurons, features)
    chunk_data_expanded = chunk_data_clean[:, np.newaxis, :]  # (samples, 1, features)
    influences_expanded = chunk_influences[:, :, np.newaxis]  # (samples, neurons, 1)
    
    # Accumulate weighted samples: Σ(h_k * x)
    weighted_samples = influences_expanded * chunk_data_expanded  # (samples, neurons, features)
    
    # Apply validity mask to handle missing values
    valid_mask_expanded = chunk_valid_mask[:, np.newaxis, :]  # (samples, 1, features)
    masked_weighted_samples = weighted_samples * valid_mask_expanded
    
    # Sum across samples for numerator: Σ(h_k * x)
    numerator = np.sum(masked_weighted_samples, axis=0)       # (neurons, features)
    
    # Sum influences for denominator: Σ(h_k)
    denominator = np.sum(chunk_influences, axis=0)            # (neurons,)
    
    return ChunkResult(
        numerator=numerator,
        denominator=denominator,
        chunk_idx=chunk_idx,
        n_valid_samples=n_valid
    )


def _vectorized_neighborhood(bmu_indices: np.ndarray, 
                           grid_positions: np.ndarray,
                           current_radius: float) -> np.ndarray:
    """Calculate neighborhood influences for all BMUs at once."""
    n_samples = len(bmu_indices)
    n_neurons = len(grid_positions)
    influences = np.zeros((n_samples, n_neurons))
    
    valid_mask = bmu_indices >= 0
    valid_bmus = bmu_indices[valid_mask]
    
    if len(valid_bmus) > 0:
        bmu_positions = grid_positions[valid_bmus]
        pos_diff = bmu_positions[:, np.newaxis, :] - grid_positions[np.newaxis, :, :]
        distances_sq = np.sum(pos_diff ** 2, axis=2)
        neighborhood = np.exp(-distances_sq / (2 * current_radius ** 2))
        influences[valid_mask] = neighborhood
    
    return influences


class ParallelBatchProcessor:
    """Handles parallel processing of data chunks for SOM batch updates."""
    
    def __init__(self, n_workers: Optional[int] = None, config=None):
        """Initialize with config for optimizations.
        
        Args:
            n_workers: Number of parallel workers
            config: Full config object for optimization settings
        """
        self.config = config
        self.n_workers = self._determine_worker_count(n_workers)
        self._parallel_context = None  # Renamed from _executor
        self._shared_weights = None
        self._shared_grid_positions = None
        
    def _determine_worker_count(self, requested=None) -> int:
        """Determine number of workers for threading backend.
        
        Priority order:
        1. Environment variable SOM_MAX_WORKERS
        2. Config optimal_workers setting  
        3. Requested parameter
        4. Default of 100 (optimal for threading)
        """
        available_cpus = cpu_count()
        
        # Start with optimal default for threading backend
        optimal = 100
        
        # Check config for override
        if self.config and hasattr(self.config, 'get'):
            training_config = self.config.get('training', {})
            if isinstance(training_config, dict):
                parallel_opt = training_config.get('parallel_optimization', {})
                if parallel_opt and isinstance(parallel_opt, dict):
                    optimal = parallel_opt.get('optimal_workers', optimal)
        
        # Environment variable takes precedence
        if 'SOM_MAX_WORKERS' in os.environ:
            optimal = int(os.environ['SOM_MAX_WORKERS'])
        
        # Apply requested override if provided
        if requested:
            optimal = requested
        
        # Cap at available CPUs (threading can efficiently use all cores)
        suggested = min(optimal, available_cpus)
        
        logger.info(f"Threading backend with {suggested} workers (system has {available_cpus} CPUs)")
        return suggested
    
    def prepare_shared_data(self, weights, grid_positions):
        """Prepare shared data as memory-mapped arrays.
        
        Called once before epoch loop to avoid repeated serialization.
        """
        # Store as contiguous arrays for better performance
        self._shared_weights = np.ascontiguousarray(weights)
        self._shared_grid_positions = np.ascontiguousarray(grid_positions)
        logger.info("Prepared shared data for parallel processing")
    
    def process_chunks_parallel(self, 
                              data: np.ndarray,
                              coordinates: Optional[np.ndarray],
                              weights: np.ndarray,
                              neuron_coords: Optional[np.ndarray],
                              config: Any,
                              grid_positions: np.ndarray,
                              current_radius: float,
                              chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process data chunks in parallel using joblib.
        
        Returns:
            Tuple of (numerator, denominator) for weight updates
        """
        n_samples = len(data)
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        if n_chunks < 2:
            # Not enough chunks to benefit from parallelization
            logger.info(f"Only {n_chunks} chunks, using sequential processing")
            return self._process_chunks_sequential(
                data, coordinates, weights, neuron_coords, 
                config, grid_positions, current_radius, chunk_size
            )
        
        # Log initial resource usage
        if HAS_PSUTIL:
            initial_memory = psutil.virtual_memory().percent
            initial_cpu = psutil.cpu_percent(interval=0.1)
            logger.info(f"Starting parallel processing - Memory: {initial_memory:.1f}%, CPU: {initial_cpu:.1f}%")
        
        # Adjust workers based on chunks
        actual_workers = min(self.n_workers, n_chunks)
        logger.info(f"Processing {n_chunks} chunks in parallel with {actual_workers} workers")
        
        # Prepare chunk parameters
        chunk_params = []
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
            
            if start_idx >= n_samples:
                continue
                
            chunk_data = data[start_idx:end_idx]
            chunk_coords = coordinates[start_idx:end_idx] if coordinates is not None else None
            
            # Memory chunking is always enabled for consistency
            use_chunking = True  # Maintained for API compatibility
            chunk_size_internal = DEFAULT_INTERNAL_CHUNK_SIZE
            
            # Use shared data references if available to avoid serialization
            shared_weights = self._shared_weights if self._shared_weights is not None else weights
            shared_grid = self._shared_grid_positions if self._shared_grid_positions is not None else grid_positions
            
            chunk_params.append((
                chunk_idx, chunk_data, chunk_coords,
                shared_weights, neuron_coords, config, shared_grid, current_radius,
                use_chunking, chunk_size_internal
            ))
        
        # Monitor memory usage and adjust workers if needed
        if HAS_PSUTIL:
            current_memory = psutil.virtual_memory().percent
            if current_memory > 85:
                actual_workers = max(1, actual_workers // 2)
                logger.warning(f"High memory usage ({current_memory:.1f}%), reducing workers to {actual_workers}")
            elif current_memory > 95:
                actual_workers = max(1, actual_workers // 4)
                logger.warning(f"Critical memory usage ({current_memory:.1f}%), reducing workers to {actual_workers}")
        
        # Always use threading backend - it's optimal for NumPy operations
        backend = 'threading'
        
        logger.info(f"Starting joblib Parallel with threading backend, n_jobs={actual_workers}")
        logger.info(f"Number of chunk_params: {len(chunk_params)}")
        
        # Debug: Log first chunk details
        if chunk_params:
            first_chunk = chunk_params[0]
            logger.info(f"First chunk - idx: {first_chunk[0]}, data shape: {first_chunk[1].shape}")
        
        # Create context manager for proper cleanup
        try:
            # Increased timeout: 60 seconds per chunk is reasonable for large datasets
            timeout_seconds = max(3600, len(chunk_params) * 60)  # At least 1 hour, or 60s per chunk
            logger.info(f"Using timeout of {timeout_seconds} seconds for {len(chunk_params)} chunks")
            
            # Build parallel parameters optimized for threading
            parallel_params = {
                'n_jobs': actual_workers,
                'backend': backend,
                'verbose': 0,
                'timeout': timeout_seconds,
                'batch_size': 50,  # Fixed optimal batch size for threading
                'pre_dispatch': '1.5*n_jobs'  # Moderate pre-dispatch for threading
            }
            
            with Parallel(**parallel_params) as parallel:
                logger.info("Parallel context created, starting processing...")
                results = parallel(
                    delayed(process_single_chunk)(*params) for params in chunk_params
                )
                logger.info(f"Parallel processing returned {len(results)} results")
        except Exception as e:
            import traceback
            logger.error(f"Parallel processing failed with {type(e).__name__}: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Parallel processing failed after {timeout_seconds}s timeout: {str(e)}") from e
        
        # Aggregate results for batch SOM update
        accumulated_numerator = np.zeros_like(weights)  # Σ(h_k * x)
        accumulated_denominator = np.zeros(len(weights))  # Σ(h_k)
        
        for result in results:
            accumulated_numerator += result.numerator    # Sum weighted samples
            accumulated_denominator += result.denominator  # Sum influences
        
        total_valid = sum(r.n_valid_samples for r in results)
        logger.info(f"Parallel processing complete. Processed {total_valid} valid samples")
        
        return accumulated_numerator, accumulated_denominator
    
    def _process_chunks_sequential(self,
                                 data: np.ndarray,
                                 coordinates: Optional[np.ndarray],
                                 weights: np.ndarray,
                                 neuron_coords: Optional[np.ndarray],
                                 config: Any,
                                 grid_positions: np.ndarray,
                                 current_radius: float,
                                 chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fall back to sequential processing for small datasets."""
        n_samples = len(data)
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        accumulated_numerator = np.zeros_like(weights)
        accumulated_denominator = np.zeros(len(weights))
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
            
            if start_idx >= n_samples:
                continue
                
            chunk_data = data[start_idx:end_idx]
            chunk_coords = coordinates[start_idx:end_idx] if coordinates is not None else None
            
            # Memory chunking is always enabled for consistency
            use_chunking = True  # Maintained for API compatibility
            chunk_size_internal = DEFAULT_INTERNAL_CHUNK_SIZE
            
            result = process_single_chunk(
                chunk_idx, chunk_data, chunk_coords,
                weights, neuron_coords, config, grid_positions, current_radius,
                use_chunking, chunk_size_internal
            )
            accumulated_numerator += result.numerator
            accumulated_denominator += result.denominator
        
        return accumulated_numerator, accumulated_denominator