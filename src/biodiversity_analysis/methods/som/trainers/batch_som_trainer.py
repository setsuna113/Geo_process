"""Optimized Batch SOM Trainer with momentum and advanced optimizations.

This is the production trainer with all optimizations built-in:
1. Numba-optimized Bray-Curtis distance calculation
2. Momentum-based weight updates with adaptive decay
3. All Numba flags optimized (cache=True, fastmath=True)
4. No redundant code paths or fallbacks
"""

import numpy as np
from numba import jit, prange
from typing import Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass

from ..calculators.optimized_distance_calculator import OptimizedDistanceCalculator

logger = logging.getLogger(__name__)


@dataclass
class OptimizedBatchConfig:
    """Configuration for optimized batch SOM training."""
    # Batch processing
    batch_size: int = 5000
    
    # Learning rate
    initial_learning_rate: float = 0.5
    final_learning_rate: float = 0.01
    learning_schedule: str = "exponential"
    
    # Neighborhood
    initial_radius: Optional[float] = None
    final_radius: float = 1.0
    radius_schedule: str = "exponential"
    
    # Momentum settings (NEW)
    initial_momentum: float = 0.9  # High initial momentum for fast convergence
    final_momentum: float = 0.1    # Low final momentum for fine-tuning
    momentum_schedule: str = "linear"  # How momentum decays
    
    # Distance metric
    distance_metric: str = "bray_curtis"
    min_valid_features: int = 2
    
    # QE calculation
    qe_sample_size: int = 10000
    calculate_qe_frequency: int = 1


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def accumulate_updates_numba(batch_data: np.ndarray, neighborhoods: np.ndarray,
                            n_neurons: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-optimized accumulation of batch updates.
    
    Args:
        batch_data: Input batch (n_samples, n_features)
        neighborhoods: Neighborhood influences (n_samples, n_neurons)
        n_neurons: Number of neurons
        n_features: Number of features
        
    Returns:
        Tuple of (numerator, denominator) for weight update
    """
    numerator = np.zeros((n_neurons, n_features), dtype=np.float32)
    denominator = np.zeros(n_neurons, dtype=np.float32)
    
    n_samples = batch_data.shape[0]
    
    # Parallel accumulation over neurons
    for n in prange(n_neurons):
        for s in range(n_samples):
            h = neighborhoods[s, n]
            denominator[n] += h
            
            for f in range(n_features):
                if not np.isnan(batch_data[s, f]):
                    numerator[n, f] += h * batch_data[s, f]
    
    return numerator, denominator


@jit(nopython=True, cache=True, fastmath=True)
def apply_momentum_update(weights: np.ndarray, velocity: np.ndarray,
                         gradient: np.ndarray, learning_rate: float,
                         momentum: float) -> None:
    """Apply momentum-based weight update (in-place).
    
    Args:
        weights: Current weights (modified in-place)
        velocity: Velocity from previous update (modified in-place)
        gradient: Current gradient
        learning_rate: Current learning rate
        momentum: Current momentum value
    """
    n_neurons = weights.shape[0]
    n_features = weights.shape[1]
    
    for i in range(n_neurons):
        for j in range(n_features):
            # Update velocity with momentum
            velocity[i, j] = momentum * velocity[i, j] + learning_rate * gradient[i, j]
            # Update weights
            weights[i, j] += velocity[i, j]


class OptimizedBatchTrainer:
    """Optimized batch SOM trainer with momentum and Numba acceleration."""
    
    def __init__(self, grid_width: int, grid_height: int, n_features: int,
                 config: Optional[OptimizedBatchConfig] = None):
        """Initialize optimized trainer.
        
        Args:
            grid_width: Width of SOM grid
            grid_height: Height of SOM grid
            n_features: Number of input features
            config: Training configuration
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.n_neurons = grid_width * grid_height
        self.n_features = n_features
        
        self.config = config or OptimizedBatchConfig()
        
        # Set default radius if not specified
        if self.config.initial_radius is None:
            self.config.initial_radius = max(grid_width, grid_height) / 2.0
        
        # Pre-compute grid positions (always contiguous)
        self.grid_positions = self._create_grid_positions()
        
        # Pre-compute neuron-to-neuron distances (always enabled)
        self._precompute_neuron_distances()
        
        # Initialize optimized distance calculator
        self.distance_calculator = OptimizedDistanceCalculator(
            distance_metric=self.config.distance_metric,
            min_valid_features=self.config.min_valid_features
        )
        
        # Initialize momentum velocity (NEW)
        self.velocity = None  # Will be initialized on first use
        
        # Tracking
        self.current_epoch = 0
        self.qe_history = []
        
        logger.info(f"OptimizedBatchTrainer initialized: {grid_width}x{grid_height} grid, "
                   f"momentum {self.config.initial_momentum}->{self.config.final_momentum}")
    
    def _create_grid_positions(self) -> np.ndarray:
        """Create grid positions for neurons (always contiguous)."""
        positions = np.zeros((self.n_neurons, 2), dtype=np.float32)
        idx = 0
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                positions[idx, 0] = i
                positions[idx, 1] = j
                idx += 1
        
        return np.ascontiguousarray(positions)
    
    def _precompute_neuron_distances(self):
        """Precompute squared distances between all neurons."""
        # Use Numba-optimized distance calculation
        from .optimized_distance_calculator import euclidean_distances_numba
        
        # For neuron positions, we don't have NaN values
        self.neuron_distances_sq = euclidean_distances_numba(
            self.grid_positions, self.grid_positions
        )
        
        logger.info(f"Precomputed {self.n_neurons}x{self.n_neurons} neuron distance matrix")
    
    def get_learning_rate(self, epoch: int, max_epochs: int) -> float:
        """Get learning rate for current epoch."""
        if self.config.learning_schedule == "linear":
            progress = epoch / max(max_epochs - 1, 1)
            return (self.config.initial_learning_rate * (1 - progress) + 
                   self.config.final_learning_rate * progress)
        else:  # exponential
            if max_epochs <= 1:
                return self.config.initial_learning_rate
            decay_rate = np.log(self.config.final_learning_rate / 
                               self.config.initial_learning_rate) / (max_epochs - 1)
            return self.config.initial_learning_rate * np.exp(decay_rate * epoch)
    
    def get_radius(self, epoch: int, max_epochs: int) -> float:
        """Get neighborhood radius for current epoch."""
        if self.config.radius_schedule == "linear":
            progress = epoch / max(max_epochs - 1, 1)
            return (self.config.initial_radius * (1 - progress) + 
                   self.config.final_radius * progress)
        else:  # exponential
            if max_epochs <= 1:
                return self.config.initial_radius
            decay_rate = np.log(self.config.final_radius / 
                               self.config.initial_radius) / (max_epochs - 1)
            return self.config.initial_radius * np.exp(decay_rate * epoch)
    
    def get_momentum(self, epoch: int, max_epochs: int) -> float:
        """Get momentum value for current epoch (NEW).
        
        Momentum decreases over training for better fine-tuning.
        """
        if self.config.momentum_schedule == "exponential":
            if max_epochs <= 1:
                return self.config.initial_momentum
            decay_rate = np.log(self.config.final_momentum / 
                               self.config.initial_momentum) / (max_epochs - 1)
            return self.config.initial_momentum * np.exp(decay_rate * epoch)
        else:  # linear
            progress = epoch / max(max_epochs - 1, 1)
            return (self.config.initial_momentum * (1 - progress) + 
                   self.config.final_momentum * progress)
    
    def find_bmus_batch(self, batch_data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Find BMUs using optimized distance calculator.
        
        Args:
            batch_data: Input batch (n_samples, n_features)
            weights: SOM weights (n_neurons, n_features)
            
        Returns:
            BMU indices for each sample (n_samples,)
        """
        return self.distance_calculator.find_bmus(batch_data, weights)
    
    def calculate_neighborhoods_batch(self, bmu_indices: np.ndarray, 
                                     radius: float) -> np.ndarray:
        """Calculate neighborhoods using precomputed distances.
        
        Args:
            bmu_indices: BMU indices for each sample (n_samples,)
            radius: Current neighborhood radius
            
        Returns:
            Neighborhood influences (n_samples, n_neurons)
        """
        n_samples = len(bmu_indices)
        neighborhoods = np.zeros((n_samples, self.n_neurons), dtype=np.float32)
        
        # Vectorized neighborhood calculation
        valid_mask = bmu_indices >= 0
        if valid_mask.any():
            valid_bmus = bmu_indices[valid_mask]
            # Use precomputed distances
            bmu_distances_sq = self.neuron_distances_sq[valid_bmus]
            # Gaussian neighborhood
            valid_neighborhoods = np.exp(-bmu_distances_sq / (2 * radius ** 2))
            neighborhoods[valid_mask] = valid_neighborhoods
        
        return neighborhoods
    
    def process_batch(self, batch_data: np.ndarray, weights: np.ndarray,
                     radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch with all optimizations.
        
        Args:
            batch_data: Batch of samples (n_samples, n_features)
            weights: Current SOM weights (n_neurons, n_features)
            radius: Current neighborhood radius
            
        Returns:
            Tuple of (numerator, denominator) for weight update
        """
        # Ensure contiguous memory (always enabled)
        batch_data = np.ascontiguousarray(batch_data, dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        
        # Find BMUs using optimized distance calculator
        bmu_indices = self.find_bmus_batch(batch_data, weights)
        
        # Calculate neighborhoods using precomputed distances
        neighborhoods = self.calculate_neighborhoods_batch(bmu_indices, radius)
        
        # Accumulate updates using Numba
        numerator, denominator = accumulate_updates_numba(
            batch_data, neighborhoods, self.n_neurons, self.n_features
        )
        
        return numerator, denominator
    
    def apply_weight_update_momentum(self, weights: np.ndarray,
                                    numerator: np.ndarray,
                                    denominator: np.ndarray,
                                    learning_rate: float,
                                    momentum: float) -> int:
        """Apply weight update with momentum (NEW).
        
        Args:
            weights: Current weights (updated in-place)
            numerator: Accumulated weighted sum of inputs
            denominator: Accumulated neighborhood influences
            learning_rate: Current learning rate
            momentum: Current momentum value
            
        Returns:
            Number of neurons updated
        """
        # Initialize velocity on first use
        if self.velocity is None:
            self.velocity = np.zeros_like(weights, dtype=np.float32)
        
        # Find neurons that received updates
        updated_mask = denominator > 1e-10
        
        if not updated_mask.any():
            return 0
        
        # Calculate gradient (target - current)
        gradient = np.zeros_like(weights, dtype=np.float32)
        gradient[updated_mask] = (numerator[updated_mask] / 
                                 denominator[updated_mask, np.newaxis] - 
                                 weights[updated_mask])
        
        # Apply momentum update using Numba
        apply_momentum_update(weights, self.velocity, gradient, 
                            learning_rate, momentum)
        
        return np.sum(updated_mask)
    
    def train_epoch(self, data: np.ndarray, weights: np.ndarray,
                   epoch: int, max_epochs: int) -> Dict[str, Any]:
        """Train one epoch with all optimizations.
        
        Args:
            data: Training data (n_samples, n_features)
            weights: Current weights (updated in-place)
            epoch: Current epoch number
            max_epochs: Total number of epochs
            
        Returns:
            Training statistics
        """
        # Get epoch parameters
        learning_rate = self.get_learning_rate(epoch, max_epochs)
        radius = self.get_radius(epoch, max_epochs)
        momentum = self.get_momentum(epoch, max_epochs)  # NEW
        
        # Shuffle data for this epoch
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        
        # Process mini-batches
        batch_count = 0
        total_updates = 0
        
        for i in range(0, n_samples, self.config.batch_size):
            end_idx = min(i + self.config.batch_size, n_samples)
            batch_indices = indices[i:end_idx]
            batch_data = data[batch_indices]
            
            # Process batch with all optimizations
            numerator, denominator = self.process_batch(batch_data, weights, radius)
            
            # Apply weight update with momentum
            neurons_updated = self.apply_weight_update_momentum(
                weights, numerator, denominator, learning_rate, momentum
            )
            
            total_updates += neurons_updated
            batch_count += 1
        
        # Calculate QE if needed
        qe = None
        if epoch % self.config.calculate_qe_frequency == 0:
            qe = self.calculate_qe_optimized(data, weights)
            self.qe_history.append(qe)
        
        self.current_epoch = epoch
        
        return {
            'epoch': epoch,
            'qe': qe,
            'learning_rate': learning_rate,
            'radius': radius,
            'momentum': momentum,  # NEW
            'n_batches': batch_count,
            'neurons_updated': total_updates
        }
    
    def calculate_qe_optimized(self, data: np.ndarray, weights: np.ndarray) -> float:
        """Calculate quantization error using optimized distance calculator."""
        n_samples = len(data)
        sample_size = min(self.config.qe_sample_size, n_samples)
        
        # Sample data if needed
        if sample_size < n_samples:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        # Find BMUs using optimized calculator
        bmu_indices = self.distance_calculator.find_bmus(sample_data, weights)
        
        # Calculate errors
        valid_mask = bmu_indices >= 0
        if not valid_mask.any():
            return np.inf
        
        valid_samples = sample_data[valid_mask]
        valid_bmus = bmu_indices[valid_mask]
        bmu_weights = weights[valid_bmus]
        
        # Compute errors (handling NaN)
        diff = bmu_weights - valid_samples
        nan_mask = ~np.isnan(valid_samples)
        diff = np.where(nan_mask, diff, 0)
        errors = np.sqrt(np.sum(diff ** 2, axis=1))
        
        return np.mean(errors)