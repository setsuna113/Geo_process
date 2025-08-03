"""Mini-batch distance calculator for memory-efficient SOM training.

This module implements mini-batching within chunks to reduce memory usage
during distance calculations. Instead of creating massive 3D arrays, it
processes data in smaller mini-batches.

Error Handling Convention:
    All invalid distances (insufficient features, division by zero, etc.) 
    are set to INVALID_DISTANCE (np.inf). This ensures consistent behavior
    across all distance calculation methods.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Standardized invalid distance value across all calculations
INVALID_DISTANCE = np.inf


class MiniBatchDistanceCalculator:
    """Handles distance calculations using mini-batches to reduce memory usage."""
    
    def __init__(self, mini_batch_size: int = 1000):
        """Initialize the calculator.
        
        Args:
            mini_batch_size: Number of samples to process at once within a chunk.
                            Default 1000 reduces memory from GB to MB scale.
        """
        if mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}")
        self.mini_batch_size = mini_batch_size
        
    def partial_bray_curtis_minibatch(self, 
                                     data_chunk: np.ndarray, 
                                     weights: np.ndarray,
                                     min_valid_features: int = 1) -> np.ndarray:
        """Calculate Bray-Curtis distances using mini-batches.
        
        Instead of creating a (chunk_size, n_neurons, n_features) array all at once,
        process in mini-batches to reduce memory usage.
        
        Args:
            data_chunk: Shape (n_samples, n_features)
            weights: Shape (n_neurons, n_features)
            min_valid_features: Minimum valid features for distance calculation
            
        Returns:
            distances: Shape (n_samples, n_neurons)
        """
        # Input validation
        if data_chunk.ndim != 2 or weights.ndim != 2:
            raise ValueError("Input arrays must be 2D")
        if data_chunk.shape[1] != weights.shape[1]:
            raise ValueError(f"Feature dimensions must match: data {data_chunk.shape[1]} != weights {weights.shape[1]}")
        
        n_samples = len(data_chunk)
        n_neurons = len(weights)
        
        # Pre-allocate output array - use empty for performance, will fill all values
        distances = np.empty((n_samples, n_neurons), dtype=np.float32)
        distances.fill(INVALID_DISTANCE)  # Use standardized invalid value
        
        # Process in mini-batches
        for start_idx in range(0, n_samples, self.mini_batch_size):
            end_idx = min(start_idx + self.mini_batch_size, n_samples)
            batch_data = data_chunk[start_idx:end_idx]
            
            # Calculate distances for this mini-batch
            batch_distances = self._compute_bray_curtis_batch(
                batch_data, weights, min_valid_features
            )
            
            # Store results
            distances[start_idx:end_idx] = batch_distances
            
        return distances
    
    def _compute_bray_curtis_batch(self,
                                  batch_data: np.ndarray,
                                  weights: np.ndarray,
                                  min_valid_features: int) -> np.ndarray:
        """Compute Bray-Curtis distances for a single mini-batch.
        
        Args:
            batch_data: Shape (batch_size, n_features)
            weights: Shape (n_neurons, n_features)
            min_valid_features: Minimum valid features
            
        Returns:
            distances: Shape (batch_size, n_neurons)
        """
        # Expand dimensions for broadcasting
        data_exp = batch_data[:, np.newaxis, :]  # (batch, 1, features)
        weights_exp = weights[np.newaxis, :, :]  # (1, neurons, features)
        
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
        distances[invalid] = INVALID_DISTANCE
        
        return distances
    
    def haversine_distances_minibatch(self,
                                     coords1: np.ndarray,
                                     coords2: np.ndarray) -> np.ndarray:
        """Calculate haversine distances using mini-batches.
        
        Args:
            coords1: Shape (n_samples, 2) - lat, lon in degrees
            coords2: Shape (n_neurons, 2) - lat, lon in degrees
            
        Returns:
            distances: Shape (n_samples, n_neurons) - distances in km
        """
        n_samples = len(coords1)
        n_neurons = len(coords2)
        
        # Pre-allocate output
        distances = np.empty((n_samples, n_neurons), dtype=np.float32)
        
        # Validate coordinate ranges
        if np.any(np.abs(coords1[:, 0]) > 90) or np.any(np.abs(coords2[:, 0]) > 90):
            logger.warning("Latitude values outside expected range [-90, 90]")
        if np.any(np.abs(coords1[:, 1]) > 180) or np.any(np.abs(coords2[:, 1]) > 180):
            logger.warning("Longitude values outside expected range [-180, 180]")
        
        # Pre-compute coords2 in radians (reused for all batches)
        lat2_rad = np.radians(coords2[:, 0])
        lon2_rad = np.radians(coords2[:, 1])
        cos_lat2 = np.cos(lat2_rad)
        
        # Process in mini-batches
        for start_idx in range(0, n_samples, self.mini_batch_size):
            end_idx = min(start_idx + self.mini_batch_size, n_samples)
            batch_coords = coords1[start_idx:end_idx]
            
            # Convert batch to radians
            lat1_rad = np.radians(batch_coords[:, 0:1])
            lon1_rad = np.radians(batch_coords[:, 1:2])
            
            # Vectorized haversine
            dlat = lat1_rad - lat2_rad[np.newaxis, :]
            dlon = lon1_rad - lon2_rad[np.newaxis, :]
            
            a = (np.sin(dlat/2)**2 + 
                 np.cos(lat1_rad) * cos_lat2[np.newaxis, :] * 
                 np.sin(dlon/2)**2)
            # More robust clipping with validation
            if np.any(a < 0) or np.any(a > 1):
                logger.debug(f"Haversine 'a' values outside [0,1]: min={np.min(a):.6f}, max={np.max(a):.6f}")
            
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            
            distances[start_idx:end_idx] = 6371 * c  # Earth radius in km
            
        return distances
    
    def combined_distance_minibatch(self,
                                   data_chunk: np.ndarray,
                                   weights: np.ndarray,
                                   coords_chunk: np.ndarray,
                                   neuron_coords: np.ndarray,
                                   spatial_weight: float,
                                   min_valid_features: int = 1) -> np.ndarray:
        """Calculate combined feature and geographic distances using mini-batches.
        
        Args:
            data_chunk: Shape (n_samples, n_features)
            weights: Shape (n_neurons, n_features)
            coords_chunk: Shape (n_samples, 2) - lat, lon
            neuron_coords: Shape (n_neurons, 2) - lat, lon
            spatial_weight: Weight for geographic distance (0-1)
            min_valid_features: Minimum valid features
            
        Returns:
            distances: Shape (n_samples, n_neurons)
        """
        # Feature distances with mini-batching
        feature_distances = self.partial_bray_curtis_minibatch(
            data_chunk, weights, min_valid_features
        )
        
        # Geographic distances with mini-batching
        geo_distances = self.haversine_distances_minibatch(
            coords_chunk, neuron_coords
        )
        
        # Normalize geographic distances
        geo_distances_norm = geo_distances / 20000.0
        
        # Combine distances
        valid_feature = ~np.isinf(feature_distances)
        combined = np.full_like(feature_distances, INVALID_DISTANCE)
        combined[valid_feature] = (
            (1 - spatial_weight) * feature_distances[valid_feature] +
            spatial_weight * geo_distances_norm[valid_feature]
        )
        
        return combined