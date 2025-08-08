"""Real Mini-batch SOM Trainer.

This module implements true mini-batch training for SOM, providing:
1. Stochastic weight updates with partial data
2. Proper VLRSOM integration with mini-batch QE
3. Clean abstraction for batch generation and weight accumulation
4. Memory-efficient processing without redundancy

Key distinction from fake mini-batch:
- Processes subsets of data for weight updates (stochastic)
- Updates weights after small batches, not full chunks
- Provides regularization through mini-batch noise
"""

import numpy as np
from typing import Iterator, Tuple, Optional, Protocol, Dict, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MiniBatchConfig:
    """Configuration for real mini-batch training."""
    batch_size: int = 1000                    # True mini-batch size
    update_frequency: int = 5                 # Update weights every N mini-batches
    shuffle_batches: bool = True              # Shuffle mini-batches each epoch
    accumulation_strategy: str = "weighted"   # "weighted" or "simple"


class BatchGenerator:
    """Generates mini-batches for stochastic training."""
    
    def __init__(self, config: MiniBatchConfig):
        self.config = config
        
    def create_mini_batches(self, 
                           data: np.ndarray, 
                           coordinates: Optional[np.ndarray] = None,
                           shuffle: bool = None) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], int]]:
        """Generate mini-batches from data.
        
        Args:
            data: Full dataset (n_samples, n_features)
            coordinates: Optional coordinates (n_samples, 2)
            shuffle: Override config shuffle setting
            
        Yields:
            Tuple of (batch_data, batch_coordinates, batch_start_idx)
        """
        n_samples = len(data)
        shuffle = shuffle if shuffle is not None else self.config.shuffle_batches
        
        # Generate indices
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
            
        # Generate batches
        for start_idx in range(0, n_samples, self.config.batch_size):
            end_idx = min(start_idx + self.config.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_data = data[batch_indices]
            batch_coords = coordinates[batch_indices] if coordinates is not None else None
            
            yield batch_data, batch_coords, start_idx


class WeightUpdateAccumulator:
    """Accumulates weight updates from mini-batches for stochastic application."""
    
    def __init__(self, n_neurons: int, n_features: int, strategy: str = "weighted"):
        self.n_neurons = n_neurons
        self.n_features = n_features
        self.strategy = strategy
        self.reset()
        
    def reset(self):
        """Reset accumulation buffers."""
        if self.strategy == "weighted":
            self.numerator = np.zeros((self.n_neurons, self.n_features), dtype=np.float64)
            self.denominator = np.zeros(self.n_neurons, dtype=np.float64)
        else:  # simple
            self.updates = []
            self.weights = []
            
    def accumulate_update(self, 
                         neuron_idx: int,
                         sample: np.ndarray,
                         influence: float,
                         learning_rate: float):
        """Accumulate single neuron update.
        
        Args:
            neuron_idx: Index of neuron to update
            sample: Sample data for update
            influence: Neighborhood influence (0-1)
            learning_rate: Current learning rate
        """
        if self.strategy == "weighted":
            weight = influence * learning_rate
            self.numerator[neuron_idx] += weight * sample
            self.denominator[neuron_idx] += weight
        else:  # simple
            self.updates.append((neuron_idx, sample, influence * learning_rate))
            
    def apply_updates(self, weights: np.ndarray) -> int:
        """Apply accumulated updates to weights.
        
        Args:
            weights: Current SOM weights to update
            
        Returns:
            Number of neurons updated
        """
        if self.strategy == "weighted":
            return self._apply_weighted_updates(weights)
        else:
            return self._apply_simple_updates(weights)
            
    def _apply_weighted_updates(self, weights: np.ndarray) -> int:
        """Apply weighted accumulated updates."""
        valid_updates = self.denominator > 0
        n_updated = np.sum(valid_updates)
        
        if n_updated > 0:
            # Apply weighted average updates
            weights[valid_updates] = (
                self.numerator[valid_updates] / 
                self.denominator[valid_updates, np.newaxis]
            )
            
        return n_updated
        
    def _apply_simple_updates(self, weights: np.ndarray) -> int:
        """Apply simple accumulated updates."""
        neurons_updated = set()
        
        for neuron_idx, sample, weight in self.updates:
            # Simple weighted update
            weights[neuron_idx] += weight * (sample - weights[neuron_idx])
            neurons_updated.add(neuron_idx)
            
        return len(neurons_updated)
        
    def is_empty(self) -> bool:
        """Check if accumulator has pending updates."""
        if self.strategy == "weighted":
            return np.sum(self.denominator) == 0
        else:
            return len(self.updates) == 0


class DistanceCalculatorProtocol(Protocol):
    """Protocol for distance calculation in mini-batch training."""
    
    def combined_distance_minibatch(self,
                                   data_chunk: np.ndarray,
                                   weights: np.ndarray,
                                   coords_chunk: Optional[np.ndarray],
                                   neuron_coords: np.ndarray,
                                   spatial_weight: float,
                                   min_valid_features: int = 1) -> np.ndarray:
        """Calculate combined distances for a mini-batch."""
        ...


class MiniBatchTrainer:
    """Real mini-batch trainer for SOM with proper stochastic updates."""
    
    def __init__(self, 
                 config: MiniBatchConfig,
                 distance_calculator: DistanceCalculatorProtocol,
                 spatial_weight: float = 0.3,
                 min_valid_features: int = 1):
        self.config = config
        self.distance_calculator = distance_calculator
        self.spatial_weight = spatial_weight
        self.min_valid_features = min_valid_features
        self.batch_generator = BatchGenerator(config)
        
    def train_epoch(self,
                   data: np.ndarray,
                   coordinates: Optional[np.ndarray],
                   weights: np.ndarray,
                   neuron_coords: np.ndarray,
                   learning_rate: float,
                   radius: float,
                   neighborhood_func: callable) -> Dict[str, Any]:
        """Train one epoch using real mini-batch processing.
        
        Args:
            data: Training data (n_samples, n_features)
            coordinates: Sample coordinates (n_samples, 2)
            weights: Current SOM weights (n_neurons, n_features)
            neuron_coords: Neuron coordinates (n_neurons, 2)
            learning_rate: Current learning rate
            radius: Current neighborhood radius
            neighborhood_func: Function to calculate neighborhood influences
            
        Returns:
            Training statistics and mini-batch QE values
        """
        n_neurons, n_features = weights.shape
        accumulator = WeightUpdateAccumulator(n_neurons, n_features, 
                                            self.config.accumulation_strategy)
        
        mini_batch_qe_values = []
        total_samples_processed = 0
        total_updates_applied = 0
        
        # Process mini-batches
        for batch_idx, (batch_data, batch_coords, _) in enumerate(
            self.batch_generator.create_mini_batches(data, coordinates)
        ):
            # Calculate BMUs for mini-batch
            batch_distances = self.distance_calculator.combined_distance_minibatch(
                batch_data, weights, batch_coords, neuron_coords,
                self.spatial_weight, self.min_valid_features
            )
            
            # Find Best Matching Units
            bmu_indices = np.argmin(batch_distances, axis=1)
            
            # Calculate mini-batch QE for VLRSOM
            valid_distances = batch_distances[~np.isinf(batch_distances).all(axis=1)]
            if len(valid_distances) > 0:
                batch_qe = np.mean(np.min(valid_distances, axis=1))
                mini_batch_qe_values.append(batch_qe)
            
            # Accumulate weight updates for this mini-batch
            self._accumulate_mini_batch_updates(
                batch_data, batch_coords, bmu_indices, batch_distances,
                weights, neuron_coords, learning_rate, radius, 
                neighborhood_func, accumulator
            )
            
            total_samples_processed += len(batch_data)
            
            # Apply updates periodically (key difference from fake mini-batch)
            if (batch_idx + 1) % self.config.update_frequency == 0:
                n_updated = accumulator.apply_updates(weights)
                total_updates_applied += n_updated
                accumulator.reset()
        
        # Apply any remaining updates at end of epoch
        if not accumulator.is_empty():
            n_updated = accumulator.apply_updates(weights)
            total_updates_applied += n_updated
            
        return {
            'mini_batch_qe_values': mini_batch_qe_values,
            'samples_processed': total_samples_processed,
            'neurons_updated': total_updates_applied,
            'n_mini_batches': batch_idx + 1 if 'batch_idx' in locals() else 0,
            'mean_mini_batch_qe': np.mean(mini_batch_qe_values) if mini_batch_qe_values else np.inf
        }
    
    def _accumulate_mini_batch_updates(self,
                                     batch_data: np.ndarray,
                                     batch_coords: Optional[np.ndarray],
                                     bmu_indices: np.ndarray,
                                     batch_distances: np.ndarray,
                                     weights: np.ndarray,
                                     neuron_coords: np.ndarray,
                                     learning_rate: float,
                                     radius: float,
                                     neighborhood_func: callable,
                                     accumulator: WeightUpdateAccumulator):
        """Accumulate weight updates from mini-batch."""
        
        for sample_idx, (sample, bmu_idx) in enumerate(zip(batch_data, bmu_indices)):
            # Skip if no valid BMU found
            if np.isinf(batch_distances[sample_idx, bmu_idx]):
                continue
                
            # Calculate neighborhood influences for all neurons
            influences = neighborhood_func(bmu_idx, radius, weights.shape[0])
            
            # Accumulate updates for neurons with non-zero influence
            for neuron_idx, influence in enumerate(influences):
                if influence > 1e-10:  # Skip negligible influences
                    accumulator.accumulate_update(
                        neuron_idx, sample, influence, learning_rate
                    )


class MiniBatchVLRSOM:
    """VLRSOM implementation with real mini-batch support."""
    
    def __init__(self, 
                 vlrsom_config: Dict[str, Any],
                 minibatch_trainer: MiniBatchTrainer):
        self.vlrsom_config = vlrsom_config
        self.trainer = minibatch_trainer
        self.current_lr = vlrsom_config.get('initial_learning_rate', 0.5)
        self.qe_history = []
        
    def adapt_learning_rate_from_minibatch(self, mini_batch_qe_values: list) -> float:
        """Adapt learning rate using mini-batch QE values.
        
        This provides more responsive VLRSOM adaptation compared to epoch-level QE.
        
        Args:
            mini_batch_qe_values: QE values from mini-batches in current epoch
            
        Returns:
            Updated learning rate
        """
        if not mini_batch_qe_values:
            return self.current_lr
            
        # Use mean of mini-batch QE values for epoch-level adaptation
        current_qe = np.mean(mini_batch_qe_values)
        
        if len(self.qe_history) > 0:
            previous_qe = self.qe_history[-1]
            qe_improvement = (previous_qe - current_qe) / previous_qe
            
            # Apply VLRSOM adaptation logic
            if qe_improvement > 0.01:  # Significant improvement
                self.current_lr *= self.vlrsom_config.get('lr_increase_factor', 1.1)
            else:  # No improvement or degradation
                self.current_lr *= self.vlrsom_config.get('lr_decrease_factor', 0.95)
                
            # Apply bounds
            min_lr = self.vlrsom_config.get('min_learning_rate', 0.01)
            max_lr = self.vlrsom_config.get('max_learning_rate', 0.8)
            self.current_lr = np.clip(self.current_lr, min_lr, max_lr)
            
        self.qe_history.append(current_qe)
        return self.current_lr
        
    def train_epoch_with_vlrsom(self, *args, **kwargs) -> Tuple[Dict[str, Any], float]:
        """Train epoch and adapt learning rate.
        
        Returns:
            Tuple of (training_stats, updated_learning_rate)
        """
        # Update learning rate in kwargs
        kwargs['learning_rate'] = self.current_lr
        
        # Train epoch
        stats = self.trainer.train_epoch(*args, **kwargs)
        
        # Adapt learning rate based on mini-batch QE
        new_lr = self.adapt_learning_rate_from_minibatch(
            stats.get('mini_batch_qe_values', [])
        )
        
        # Update stats with learning rate info
        stats['learning_rate'] = new_lr
        stats['qe_improvement'] = (
            (self.qe_history[-2] - self.qe_history[-1]) / self.qe_history[-2] 
            if len(self.qe_history) >= 2 else 0.0
        )
        
        return stats, new_lr