"""Parallel batch processing utilities for SOM training.

This module provides parallel chunk processing capabilities to utilize
multiple CPU cores for SOM batch updates.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
import logging
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import os

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
                        current_radius: float) -> ChunkResult:
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
    
    # Calculate distances for this chunk
    if chunk_coords is not None and neuron_coords is not None:
        # Combined distance calculation (feature + geographic)
        chunk_distances = _vectorized_combined_distance(
            chunk_data, weights, chunk_coords, neuron_coords,
            min_valid_features, spatial_weight
        )
    else:
        # Feature distance only
        chunk_distances = _vectorized_partial_bray_curtis_chunk(
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
    
    # Calculate numerator and denominator for this chunk
    numerator = np.einsum('ij,ik,ik->jk', chunk_influences, chunk_data_clean, chunk_valid_mask)
    denominator = np.einsum('ij,ik->j', chunk_influences, chunk_valid_mask)
    
    return ChunkResult(
        numerator=numerator,
        denominator=denominator,
        chunk_idx=chunk_idx,
        n_valid_samples=n_valid
    )


def _vectorized_partial_bray_curtis_chunk(data_chunk: np.ndarray, 
                                        weights: np.ndarray,
                                        min_valid_features: int) -> np.ndarray:
    """Vectorized Bray-Curtis distance calculation for a chunk of data."""
    # Expand dimensions for broadcasting
    data_exp = data_chunk[:, np.newaxis, :]
    weights_exp = weights[np.newaxis, :, :]
    
    # Find valid pairs
    valid_mask = ~(np.isnan(data_exp) | np.isnan(weights_exp))
    n_valid = valid_mask.sum(axis=2)
    
    # Clean data for calculation
    data_clean = np.nan_to_num(data_exp, nan=0.0)
    weights_clean = np.nan_to_num(weights_exp, nan=0.0)
    
    # Bray-Curtis calculation
    diff = np.abs(data_clean - weights_clean)
    sum_vals = data_clean + weights_clean
    
    numerator = (diff * valid_mask).sum(axis=2)
    denominator = (sum_vals * valid_mask).sum(axis=2)
    
    # Handle division
    with np.errstate(divide='ignore', invalid='ignore'):
        distances = numerator / denominator
    
    # Set invalid distances
    invalid = (n_valid < min_valid_features) | (denominator == 0)
    distances[invalid] = np.inf
    
    return distances


def _vectorized_combined_distance(data_chunk: np.ndarray, 
                                weights: np.ndarray,
                                coords_chunk: np.ndarray, 
                                neuron_coords: np.ndarray,
                                min_valid_features: int,
                                spatial_weight: float) -> np.ndarray:
    """Calculate combined feature and geographic distances for a chunk."""
    # Feature distances
    feature_distances = _vectorized_partial_bray_curtis_chunk(
        data_chunk, weights, min_valid_features
    )
    
    # Geographic distances (vectorized haversine)
    geo_distances = _vectorized_haversine_distances(coords_chunk, neuron_coords)
    
    # Normalize geographic distances (rough earth circumference / 2)
    geo_distances_norm = geo_distances / 20000.0
    
    # Combined distance with spatial weighting
    valid_feature = ~np.isinf(feature_distances)
    combined = np.full_like(feature_distances, np.inf)
    combined[valid_feature] = (
        (1 - spatial_weight) * feature_distances[valid_feature] +
        spatial_weight * geo_distances_norm[valid_feature]
    )
    
    return combined


def _vectorized_haversine_distances(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance calculation between coordinate sets."""
    # Convert to radians
    coords1_rad = np.radians(coords1)
    coords2_rad = np.radians(coords2)
    
    # Extract lat/lon
    lat1 = coords1_rad[:, 1:2]  # Keep as (n, 1) for broadcasting
    lon1 = coords1_rad[:, 0:1]
    lat2 = coords2_rad[:, 1].reshape(1, -1)  # Shape (1, m)
    lon2 = coords2_rad[:, 0].reshape(1, -1)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # clip for numerical stability
    
    # Earth radius in km
    return 6371 * c


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
    
    def __init__(self, n_workers: Optional[int] = None):
        """Initialize with specified number of workers.
        
        Args:
            n_workers: Number of parallel workers. If None, auto-detect.
        """
        self.n_workers = n_workers or self._determine_worker_count()
        
    def _determine_worker_count(self) -> int:
        """Determine optimal number of workers based on system and data."""
        # Get available CPUs
        available_cpus = cpu_count()
        
        # Use environment variable if set
        if 'SOM_MAX_WORKERS' in os.environ:
            max_workers = int(os.environ['SOM_MAX_WORKERS'])
            return min(max_workers, available_cpus)
        
        # For very large systems, don't use all cores
        if available_cpus > 64:
            # Use 1/4 of cores but at least 16
            suggested = max(16, available_cpus // 4)
        elif available_cpus > 32:
            # Use 1/2 of cores
            suggested = available_cpus // 2
        else:
            # Use all but 2 cores (leave some for system)
            suggested = max(1, available_cpus - 2)
            
        logger.info(f"System has {available_cpus} CPUs, using {suggested} workers for SOM")
        return suggested
    
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
            
            chunk_params.append((
                chunk_idx, chunk_data, chunk_coords,
                weights, neuron_coords, config, grid_positions, current_radius
            ))
        
        # Process chunks in parallel using joblib
        results = Parallel(n_jobs=actual_workers, backend='loky', verbose=10)(
            delayed(process_single_chunk)(*params) for params in chunk_params
        )
        
        # Aggregate results
        numerator = np.zeros_like(weights)
        denominator = np.zeros(len(weights))
        
        for result in results:
            numerator += result.numerator
            denominator += result.denominator
        
        total_valid = sum(r.n_valid_samples for r in results)
        logger.info(f"Parallel processing complete. Processed {total_valid} valid samples")
        
        return numerator, denominator
    
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
        
        numerator = np.zeros_like(weights)
        denominator = np.zeros(len(weights))
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
            
            if start_idx >= n_samples:
                continue
                
            chunk_data = data[start_idx:end_idx]
            chunk_coords = coordinates[start_idx:end_idx] if coordinates is not None else None
            
            result = process_single_chunk(
                chunk_idx, chunk_data, chunk_coords,
                weights, neuron_coords, config, grid_positions, current_radius
            )
            numerator += result.numerator
            denominator += result.denominator
        
        return numerator, denominator