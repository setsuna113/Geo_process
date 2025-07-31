"""Core SOM implementation optimized for biodiversity data."""

import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from scipy.spatial.distance import cdist
from src.base.som_base import BaseSOM
from src.abstractions.types.som_types import (
    SOMConfig, SOMTrainingResult, InitializationMethod, 
    NeighborhoodFunction, DistanceMetric
)
from src.biodiversity_analysis.shared.data.som_preprocessor import SOMPreprocessor
import logging
import time

logger = logging.getLogger(__name__)


class BiodiversitySOM(BaseSOM):
    """SOM implementation optimized for biodiversity data.
    
    Features:
    - Multiple distance metrics including Bray-Curtis for species data
    - Batch and online training modes
    - Adaptive learning rate and neighborhood radius
    - Convergence detection
    - Memory-efficient processing for large datasets
    - Support for zero-inflated species data
    """
    
    def __init__(self, config: SOMConfig):
        """Initialize BiodiversitySOM.
        
        Args:
            config: SOM configuration parameters
        """
        super().__init__(config)
        self.preprocessor = SOMPreprocessor()
        self._training_history = {
            'quantization_errors': [],
            'topographic_errors': [],
            'learning_rates': [],
            'neighborhood_radii': []
        }
    
    def initialize_weights(self, data: np.ndarray) -> None:
        """Initialize SOM weights using the specified method.
        
        Args:
            data: Training data of shape (n_samples, n_features)
        """
        self.input_dim = data.shape[1]
        
        if self.config.initialization_method == InitializationMethod.PCA:
            # PCA-based initialization
            weights_grid = self.preprocessor.initialize_weights_pca(
                data, self.config.grid_size
            )
            self.weights = weights_grid.reshape(self.n_neurons, self.input_dim)
            
        elif self.config.initialization_method == InitializationMethod.SAMPLE:
            # Initialize from random samples
            weights_grid = self.preprocessor.initialize_weights_sample(
                data, self.config.grid_size, self.config.random_seed
            )
            self.weights = weights_grid.reshape(self.n_neurons, self.input_dim)
            
        else:
            # Random initialization
            self.weights = np.random.randn(self.n_neurons, self.input_dim)
            # Scale to match data range
            data_std = np.std(data, axis=0)
            data_mean = np.mean(data, axis=0)
            self.weights = self.weights * data_std + data_mean
        
        logger.info(f"Initialized SOM weights using {self.config.initialization_method.value} method")
    
    def train(self, data: np.ndarray, 
              progress_callback: Optional[Callable] = None) -> SOMTrainingResult:
        """Train the SOM with adaptive learning and convergence detection.
        
        Args:
            data: Training data of shape (n_samples, n_features)
            progress_callback: Optional callback for progress updates
            
        Returns:
            SOMTrainingResult with training metrics and final weights
        """
        start_time = time.time()
        
        # Initialize weights if not already done
        if self.weights is None:
            self.initialize_weights(data)
        
        # Initialize training parameters
        initial_lr = self.config.learning_rate
        initial_radius = self.config.neighborhood_radius
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Decay learning rate and neighborhood radius
            lr = self._decay_parameter(
                initial_lr, epoch, self.config.min_learning_rate
            )
            radius = self._decay_parameter(
                initial_radius, epoch, self.config.min_neighborhood_radius
            )
            
            # Store parameters
            self._training_history['learning_rates'].append(lr)
            self._training_history['neighborhood_radii'].append(radius)
            
            # Train one epoch
            epoch_metrics = self.train_epoch(data, epoch, lr, radius)
            
            # Calculate metrics every few epochs
            if epoch % max(1, self.config.epochs // 20) == 0 or epoch == self.config.epochs - 1:
                qe = self.calculate_quantization_error(data)
                te = self.calculate_topographic_error(data)
                
                self._training_history['quantization_errors'].append(qe)
                self._training_history['topographic_errors'].append(te)
                
                logger.debug(f"Epoch {epoch}: QE={qe:.4f}, TE={te:.4f}, "
                           f"LR={lr:.4f}, Radius={radius:.2f}")
                
                # Check convergence
                if self.check_convergence(self._training_history['quantization_errors']):
                    logger.info(f"Training converged at epoch {epoch}")
                    convergence_epoch = epoch
                    break
            
            # Progress callback
            if progress_callback:
                progress_callback((epoch + 1) / self.config.epochs)
        else:
            convergence_epoch = None
        
        # Final metrics
        final_qe = self.calculate_quantization_error(data)
        final_te = self.calculate_topographic_error(data)
        
        if final_qe not in self._training_history['quantization_errors']:
            self._training_history['quantization_errors'].append(final_qe)
            self._training_history['topographic_errors'].append(final_te)
        
        training_time = time.time() - start_time
        
        # Create result
        result = SOMTrainingResult(
            weights=self.weights.reshape(self.grid_rows, self.grid_cols, self.input_dim),
            quantization_errors=self._training_history['quantization_errors'],
            topographic_errors=self._training_history['topographic_errors'],
            training_time=training_time,
            convergence_epoch=convergence_epoch,
            final_learning_rate=lr,
            final_neighborhood_radius=radius,
            n_samples_trained=len(data)
        )
        
        logger.info(f"Training completed in {training_time:.2f}s. "
                   f"Final QE: {final_qe:.4f}, Final TE: {final_te:.4f}")
        
        return result
    
    def train_epoch(self, data: np.ndarray, epoch: int, 
                   learning_rate: float, neighborhood_radius: float) -> Dict[str, float]:
        """Train one epoch using batch or online mode.
        
        Args:
            data: Training data
            epoch: Current epoch number
            learning_rate: Current learning rate
            neighborhood_radius: Current neighborhood radius
            
        Returns:
            Dictionary with epoch metrics
        """
        if self.config.batch_size:
            return self._train_batch(data, learning_rate, neighborhood_radius)
        else:
            return self._train_online(data, learning_rate, neighborhood_radius)
    
    def _train_online(self, data: np.ndarray, lr: float, radius: float) -> Dict[str, float]:
        """Online training - process one sample at a time.
        
        Args:
            data: Training data
            lr: Learning rate
            radius: Neighborhood radius
            
        Returns:
            Epoch metrics
        """
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        
        for idx in indices:
            sample = data[idx]
            
            # Find BMU
            bmu_idx = self._find_bmu_vectorized(sample)
            
            # Update weights
            self._update_weights(sample, bmu_idx, lr, radius)
        
        return {'samples_processed': n_samples}
    
    def _train_batch(self, data: np.ndarray, lr: float, radius: float) -> Dict[str, float]:
        """Batch training - process multiple samples together.
        
        Args:
            data: Training data
            lr: Learning rate
            radius: Neighborhood radius
            
        Returns:
            Epoch metrics
        """
        n_samples = len(data)
        n_batches = 0
        
        # Process in batches
        for i in range(0, n_samples, self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            
            # Find BMUs for entire batch
            bmu_indices = np.array([self._find_bmu_vectorized(s) for s in batch])
            
            # Batch update weights
            self._batch_update_weights(batch, bmu_indices, lr, radius)
            n_batches += 1
        
        return {'samples_processed': n_samples, 'n_batches': n_batches}
    
    def _find_bmu_vectorized(self, sample: np.ndarray) -> int:
        """Find Best Matching Unit using vectorized operations.
        
        Args:
            sample: Data sample
            
        Returns:
            Index of BMU
        """
        distance_func = self.get_distance_function()
        
        # Calculate distances to all neurons
        if self.config.distance_metric == DistanceMetric.BRAY_CURTIS:
            # Special handling for Bray-Curtis
            distances = distance_func(sample.reshape(1, -1), self.weights).flatten()
        else:
            # Standard distance calculation
            distances = distance_func(sample, self.weights)
        
        return np.argmin(distances)
    
    def _update_weights(self, sample: np.ndarray, bmu_idx: int, 
                       lr: float, radius: float):
        """Update weights based on neighborhood function.
        
        Args:
            sample: Training sample
            bmu_idx: Index of BMU
            lr: Learning rate
            radius: Neighborhood radius
        """
        # Calculate neighborhood influence
        influences = self._calculate_neighborhood(bmu_idx, radius)
        
        # Update weights
        for i in range(self.n_neurons):
            if influences[i] > 0:
                self.weights[i] += lr * influences[i] * (sample - self.weights[i])
    
    def _batch_update_weights(self, batch: np.ndarray, bmu_indices: np.ndarray,
                             lr: float, radius: float):
        """Update weights for a batch of samples.
        
        Args:
            batch: Batch of samples
            bmu_indices: BMU index for each sample
            lr: Learning rate
            radius: Neighborhood radius
        """
        # Accumulate updates
        weight_updates = np.zeros_like(self.weights)
        update_counts = np.zeros(self.n_neurons)
        
        for sample, bmu_idx in zip(batch, bmu_indices):
            # Calculate neighborhood influence
            influences = self._calculate_neighborhood(bmu_idx, radius)
            
            # Accumulate updates
            for i in range(self.n_neurons):
                if influences[i] > 0:
                    weight_updates[i] += influences[i] * (sample - self.weights[i])
                    update_counts[i] += influences[i]
        
        # Apply averaged updates
        for i in range(self.n_neurons):
            if update_counts[i] > 0:
                self.weights[i] += lr * weight_updates[i] / update_counts[i]
    
    def _calculate_neighborhood(self, bmu_idx: int, radius: float) -> np.ndarray:
        """Calculate neighborhood function values for all neurons.
        
        Args:
            bmu_idx: Index of BMU
            radius: Neighborhood radius
            
        Returns:
            Array of neighborhood influence values
        """
        bmu_pos = self._grid_positions[bmu_idx]
        
        if self.config.neighborhood_function == NeighborhoodFunction.GAUSSIAN:
            # Gaussian neighborhood
            distances_squared = np.sum((self._grid_positions - bmu_pos)**2, axis=1)
            influences = np.exp(-distances_squared / (2 * radius**2))
            
        elif self.config.neighborhood_function == NeighborhoodFunction.BUBBLE:
            # Bubble neighborhood (binary)
            distances = np.sqrt(np.sum((self._grid_positions - bmu_pos)**2, axis=1))
            influences = (distances <= radius).astype(float)
            
        elif self.config.neighborhood_function == NeighborhoodFunction.MEXICAN_HAT:
            # Mexican hat neighborhood
            distances = np.sqrt(np.sum((self._grid_positions - bmu_pos)**2, axis=1))
            normalized_dist = distances / radius
            influences = 2 * np.exp(-0.5 * normalized_dist**2) - np.exp(-0.125 * normalized_dist**2)
            
        else:
            raise ValueError(f"Unknown neighborhood function: {self.config.neighborhood_function}")
        
        return influences
    
    def calculate_quantization_error(self, data: np.ndarray) -> float:
        """Calculate average quantization error.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Average quantization error
        """
        if self.weights is None:
            raise ValueError("SOM must be trained before calculating quantization error")
        
        # Handle empty data
        if len(data) == 0:
            return 0.0
        
        distance_func = self.get_distance_function()
        total_error = 0.0
        
        for sample in data:
            bmu_idx = self._find_bmu_vectorized(sample)
            bmu_weights = self.weights[bmu_idx]
            
            # Calculate distance to BMU
            if self.config.distance_metric == DistanceMetric.BRAY_CURTIS:
                error = distance_func(sample.reshape(1, -1), bmu_weights.reshape(1, -1))[0]
            else:
                error = distance_func(sample, bmu_weights)
            
            total_error += error
        
        return total_error / len(data)
    
    def calculate_topographic_error(self, data: np.ndarray) -> float:
        """Calculate topographic error.
        
        Topographic error is the proportion of samples for which
        the first and second BMUs are not adjacent.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Topographic error (0 to 1)
        """
        if self.weights is None:
            raise ValueError("SOM must be trained before calculating topographic error")
        
        # Handle empty data
        if len(data) == 0:
            return 0.0
        
        distance_func = self.get_distance_function()
        n_errors = 0
        
        for sample in data:
            # Find distances to all neurons
            if self.config.distance_metric == DistanceMetric.BRAY_CURTIS:
                distances = distance_func(sample.reshape(1, -1), self.weights).flatten()
            else:
                distances = distance_func(sample, self.weights)
            
            # Find first and second BMUs
            sorted_indices = np.argsort(distances)
            bmu1_idx = sorted_indices[0]
            bmu2_idx = sorted_indices[1]
            
            # Check if they are adjacent
            pos1 = self._grid_positions[bmu1_idx]
            pos2 = self._grid_positions[bmu2_idx]
            
            # Adjacent if distance <= sqrt(2) (diagonal neighbors)
            if np.sqrt(np.sum((pos1 - pos2)**2)) > np.sqrt(2):
                n_errors += 1
        
        return n_errors / len(data)
    
    def find_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        """Find Best Matching Unit for a sample.
        
        Args:
            sample: Data sample
            
        Returns:
            Tuple of (row, col) position of BMU
        """
        bmu_idx = self._find_bmu_vectorized(sample)
        return self.get_grid_position(bmu_idx)
    
    def get_weights(self) -> np.ndarray:
        """Get the current weight matrix.
        
        Returns:
            Weight matrix of shape (n_rows, n_cols, n_features)
        """
        if self.weights is None:
            raise ValueError("SOM has not been initialized")
        
        return self.weights.reshape(self.grid_rows, self.grid_cols, self.input_dim)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict BMU indices for data samples.
        
        Args:
            data: Data samples
            
        Returns:
            Array of BMU indices
        """
        return np.array([self._find_bmu_vectorized(sample) for sample in data])
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to BMU grid coordinates.
        
        Args:
            data: Data samples
            
        Returns:
            Array of (row, col) coordinates
        """
        bmu_indices = self.predict(data)
        return np.array([self.get_grid_position(idx) for idx in bmu_indices])