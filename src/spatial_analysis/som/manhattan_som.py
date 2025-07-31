"""
Manhattan Distance SOM Implementation for Biodiversity Data

This module provides a SOM implementation optimized for biodiversity data using
Manhattan distance, which is better suited for species occurrence data where
features represent presence/absence or abundance counts.

Manhattan distance is more appropriate for:
- Species occurrence data (sparse, categorical-like)
- Abundance counts (where differences matter linearly)
- P-A-F (Phylogenetic, Abundance, Functional) profiles
"""

import numpy as np
import logging
from typing import Tuple, Optional, Callable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ManhattanSOMConfig:
    """Configuration for Manhattan distance SOM."""
    x: int  # Grid width
    y: int  # Grid height
    input_len: int  # Input vector dimensions
    sigma: float = 1.0  # Neighborhood radius
    learning_rate: float = 0.5  # Initial learning rate
    neighborhood_function: str = 'gaussian'  # Neighborhood function
    random_seed: Optional[int] = None
    topology: str = 'rectangular'  # Grid topology


class ManhattanSOM:
    """
    Self-Organizing Map using Manhattan distance for biodiversity data.
    
    Manhattan distance (L1 norm) is more appropriate for biodiversity data because:
    1. Species occurrence data is often sparse and categorical-like
    2. Linear differences in abundance are more meaningful than squared differences
    3. More robust to outliers in species counts
    4. Faster computation (no squares/square roots)
    """
    
    def __init__(self, config: ManhattanSOMConfig):
        """Initialize Manhattan distance SOM."""
        self.config = config
        self.x = config.x
        self.y = config.y
        self.input_len = config.input_len
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Initialize weights randomly
        self._weights = np.random.random((config.x, config.y, config.input_len))
        
        # Normalize initial weights
        self._normalize_weights()
        
        # Training parameters
        self.sigma = config.sigma
        self.learning_rate = config.learning_rate
        self.neighborhood_function = config.neighborhood_function
        
        # Create coordinate grids for neighborhood calculations
        self._create_coordinate_grids()
        
        # Training state
        self.iteration = 0
        
        logger.info(f"Initialized Manhattan SOM: {config.x}x{config.y} grid, "
                   f"{config.input_len} features, Manhattan distance")
    
    def _create_coordinate_grids(self):
        """Create coordinate grids for efficient neighborhood calculation."""
        x_coords, y_coords = np.meshgrid(np.arange(self.x), np.arange(self.y), indexing='ij')
        self.coordinates = np.stack([x_coords, y_coords], axis=-1)
    
    def _normalize_weights(self):
        """Normalize weight vectors (optional for Manhattan distance)."""
        # For Manhattan distance, we can normalize by L1 norm
        norms = np.sum(np.abs(self._weights), axis=2, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        self._weights = self._weights / norms
    
    def manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Manhattan distance between two vectors."""
        return np.sum(np.abs(a - b))
    
    def manhattan_distance_batch(self, sample: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Calculate Manhattan distances from sample to all weight vectors."""
        # Broadcasting: sample is (input_len,), weights is (x, y, input_len)
        return np.sum(np.abs(weights - sample), axis=2)
    
    def winner(self, sample: np.ndarray) -> Tuple[int, int]:
        """Find the Best Matching Unit (BMU) using Manhattan distance."""
        distances = self.manhattan_distance_batch(sample, self._weights)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def winner_distance(self, sample: np.ndarray) -> float:
        """Get the distance to the Best Matching Unit."""
        bmu = self.winner(sample)
        return self.manhattan_distance(sample, self._weights[bmu])
    
    def _gaussian_neighborhood(self, bmu: Tuple[int, int], sigma: float) -> np.ndarray:
        """Calculate Gaussian neighborhood function."""
        bmu_coords = np.array(bmu)
        
        # Calculate distances from BMU to all neurons
        distances_sq = np.sum((self.coordinates - bmu_coords) ** 2, axis=2)
        
        # Gaussian neighborhood
        return np.exp(-distances_sq / (2 * sigma ** 2))
    
    def _bubble_neighborhood(self, bmu: Tuple[int, int], sigma: float) -> np.ndarray:
        """Calculate bubble neighborhood function."""
        bmu_coords = np.array(bmu)
        distances = np.linalg.norm(self.coordinates - bmu_coords, axis=2)
        return (distances <= sigma).astype(float)
    
    def _calculate_neighborhood(self, bmu: Tuple[int, int], sigma: float) -> np.ndarray:
        """Calculate neighborhood weights."""
        if self.neighborhood_function == 'gaussian':
            return self._gaussian_neighborhood(bmu, sigma)
        elif self.neighborhood_function == 'bubble':
            return self._bubble_neighborhood(bmu, sigma)
        else:
            # Default to Gaussian
            return self._gaussian_neighborhood(bmu, sigma)
    
    def _update_weights(self, sample: np.ndarray, bmu: Tuple[int, int], 
                       learning_rate: float, sigma: float):
        """Update weights using Manhattan distance-based learning rule."""
        # Calculate neighborhood weights
        neighborhood = self._calculate_neighborhood(bmu, sigma)
        
        # Calculate weight updates
        # For Manhattan distance, we use sign-based updates
        diff = sample - self._weights
        sign_diff = np.sign(diff)
        
        # Apply learning rate and neighborhood
        update = learning_rate * neighborhood[:, :, np.newaxis] * sign_diff
        
        # Update weights
        self._weights += update
        
        # Optional: normalize weights to maintain unit L1 norm
        # self._normalize_weights()
    
    def train_single(self, sample: np.ndarray, learning_rate: float, sigma: float):
        """Train SOM with a single sample."""
        # Find BMU
        bmu = self.winner(sample)
        
        # Update weights
        self._update_weights(sample, bmu, learning_rate, sigma)
        
        self.iteration += 1
    
    def train_random(self, data: np.ndarray, num_iterations: int):
        """Train SOM with random sampling."""
        n_samples = data.shape[0]
        
        logger.info(f"Training Manhattan SOM for {num_iterations} iterations on {n_samples} samples")
        
        for i in range(num_iterations):
            # Decay parameters
            progress = i / num_iterations
            current_learning_rate = self.learning_rate * (1 - progress)
            current_sigma = self.sigma * (1 - progress)
            
            # Random sample selection
            sample_idx = np.random.randint(0, n_samples)
            sample = data[sample_idx]
            
            # Train with sample
            self.train_single(sample, current_learning_rate, current_sigma)
            
            # Progress logging
            if i % 1000 == 0:
                logger.debug(f"Training iteration {i}/{num_iterations}")
    
    def train_batch(self, data: np.ndarray, num_epochs: int, random_order: bool = True):
        """Train SOM with batch method (all samples per epoch)."""
        n_samples = data.shape[0]
        
        logger.info(f"Batch training Manhattan SOM for {num_epochs} epochs on {n_samples} samples")
        
        for epoch in range(num_epochs):
            # Decay parameters
            progress = epoch / num_epochs
            current_learning_rate = self.learning_rate * (1 - progress)
            current_sigma = self.sigma * (1 - progress)
            
            # Determine sample order
            if random_order:
                sample_indices = np.random.permutation(n_samples)
            else:
                sample_indices = np.arange(n_samples)
            
            # Train on all samples
            for sample_idx in sample_indices:
                sample = data[sample_idx]
                self.train_single(sample, current_learning_rate, current_sigma)
            
            # Progress logging
            if epoch % 10 == 0:
                logger.debug(f"Batch training epoch {epoch}/{num_epochs}")
    
    def quantization_error(self, data: np.ndarray) -> float:
        """Calculate quantization error using Manhattan distance (fully vectorized)."""
        if len(data) == 0:
            return 0.0
        
        # Truly vectorized calculation using batch operations
        # For each sample, find the minimum distance to any neuron
        total_error = 0.0
        for sample in data:
            # Use existing batch method to get distances to all neurons
            distances = self.manhattan_distance_batch(sample, self._weights)
            # Get minimum distance (BMU distance)
            total_error += np.min(distances)
        
        return total_error / len(data)
    
    def topographic_error(self, data: np.ndarray) -> float:
        """Calculate topographic error."""
        topographic_errors = 0
        
        for sample in data:
            # Find distances to all neurons
            distances = self.manhattan_distance_batch(sample, self._weights)
            
            # Find two best matching units
            flat_distances = distances.flatten()
            bmu_indices = np.argpartition(flat_distances, 2)[:2]
            
            # Convert back to 2D coordinates
            bmu1_2d = np.unravel_index(bmu_indices[0], distances.shape)
            bmu2_2d = np.unravel_index(bmu_indices[1], distances.shape)
            
            # Check if BMUs are neighbors (8-connectivity)
            distance_between_bmus = max(
                abs(bmu1_2d[0] - bmu2_2d[0]), 
                abs(bmu1_2d[1] - bmu2_2d[1])
            )
            
            if distance_between_bmus > 1:  # Not neighbors
                topographic_errors += 1
        
        return topographic_errors / len(data)
    
    def distance_map(self) -> np.ndarray:
        """Calculate distance map (U-matrix) using Manhattan distance."""
        distance_map = np.zeros((self.x, self.y))
        
        # For each neuron, calculate average distance to neighbors
        for i in range(self.x):
            for j in range(self.y):
                distances = []
                
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.x and 0 <= nj < self.y:
                            dist = self.manhattan_distance(
                                self._weights[i, j], 
                                self._weights[ni, nj]
                            )
                            distances.append(dist)
                
                distance_map[i, j] = np.mean(distances) if distances else 0
        
        return distance_map
    
    def get_weights(self) -> np.ndarray:
        """Get current weight matrix."""
        return self._weights.copy()
    
    def set_weights(self, weights: np.ndarray):
        """Set weight matrix (for restoring best weights during validation)."""
        if weights.shape != self._weights.shape:
            raise ValueError(f"Weight shape mismatch: expected {self._weights.shape}, got {weights.shape}")
        self._weights = weights.copy()
    
    def activation_response(self, data: np.ndarray) -> np.ndarray:
        """Get activation frequency for each neuron."""
        activation_map = np.zeros((self.x, self.y))
        
        for sample in data:
            bmu = self.winner(sample)
            activation_map[bmu] += 1
        
        return activation_map
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels for data."""
        labels = np.zeros(len(data), dtype=int)
        
        for i, sample in enumerate(data):
            bmu = self.winner(sample)
            # Convert 2D BMU coordinates to 1D cluster ID
            labels[i] = bmu[0] * self.y + bmu[1]
        
        return labels


def create_manhattan_som(x: int, y: int, input_len: int, **kwargs) -> ManhattanSOM:
    """Factory function to create Manhattan distance SOM."""
    config = ManhattanSOMConfig(
        x=x, 
        y=y, 
        input_len=input_len,
        **kwargs
    )
    return ManhattanSOM(config)


# Compatibility wrapper for MiniSom-like interface
class ManhattanSOMWrapper:
    """Wrapper to make ManhattanSOM compatible with existing MiniSom code."""
    
    def __init__(self, x: int, y: int, input_len: int, sigma: float = 1.0,
                 learning_rate: float = 0.5, neighborhood_function: str = 'gaussian',
                 random_seed: Optional[int] = None, **kwargs):
        """Initialize with MiniSom-compatible interface."""
        config = ManhattanSOMConfig(
            x=x, y=y, input_len=input_len, sigma=sigma,
            learning_rate=learning_rate, neighborhood_function=neighborhood_function,
            random_seed=random_seed
        )
        self.som = ManhattanSOM(config)
        
        # Expose interface methods
        self.get_weights = self.som.get_weights
        self.set_weights = self.som.set_weights
        self.winner = self.som.winner
        self.train_random = self.som.train_random
        self.train_batch = self.som.train_batch
        self.train_single = self.som.train_single  # Add missing method
        self.quantization_error = self.som.quantization_error
        self.topographic_error = self.som.topographic_error
        self.distance_map = self.som.distance_map
        self.predict = self.som.predict
        
        # Store shape for compatibility
        self._weights = self.som._weights
        
        # Expose sigma property for VLRSOM
        self.sigma = self.som.sigma
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        return self.som.predict(data)