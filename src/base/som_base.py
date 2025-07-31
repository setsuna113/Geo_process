"""Base class for Self-Organizing Map implementations."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Callable, Dict, List
from src.abstractions.types.som_types import SOMConfig, DistanceMetric
import logging

logger = logging.getLogger(__name__)


class BaseSOM(ABC):
    """Base class for SOM implementations.
    
    Provides common functionality for distance calculations,
    parameter decay, and basic operations.
    """
    
    def __init__(self, config: SOMConfig):
        """Initialize base SOM.
        
        Args:
            config: SOM configuration parameters
        """
        self.config = config
        self.weights: Optional[np.ndarray] = None
        self.input_dim: Optional[int] = None
        self.grid_rows, self.grid_cols = config.grid_size
        self.n_neurons = self.grid_rows * self.grid_cols
        
        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Initialize distance function mapping
        self._distance_functions = {
            DistanceMetric.EUCLIDEAN: self._euclidean_distance,
            DistanceMetric.MANHATTAN: self._manhattan_distance,
            DistanceMetric.COSINE: self._cosine_distance,
            DistanceMetric.BRAY_CURTIS: self._bray_curtis_distance
        }
        
        # Cache for grid positions
        self._grid_positions = self._create_grid_positions()
    
    def _create_grid_positions(self) -> np.ndarray:
        """Create array of grid positions for each neuron.
        
        Returns:
            Array of shape (n_neurons, 2) with (row, col) positions
        """
        positions = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                positions.append([i, j])
        return np.array(positions)
    
    @abstractmethod
    def initialize_weights(self, data: np.ndarray) -> None:
        """Initialize SOM weights based on data.
        
        Args:
            data: Training data of shape (n_samples, n_features)
        """
        pass
    
    @abstractmethod
    def train_epoch(self, data: np.ndarray, epoch: int, 
                   learning_rate: float, neighborhood_radius: float) -> Dict[str, float]:
        """Train one epoch.
        
        Args:
            data: Training data
            epoch: Current epoch number
            learning_rate: Current learning rate
            neighborhood_radius: Current neighborhood radius
            
        Returns:
            Dictionary with epoch metrics
        """
        pass
    
    def _decay_parameter(self, initial_value: float, current_epoch: int, 
                        min_value: float, total_epochs: Optional[int] = None) -> float:
        """Decay a parameter over epochs using exponential decay.
        
        Args:
            initial_value: Starting value
            current_epoch: Current epoch number
            min_value: Minimum allowed value
            total_epochs: Total number of epochs (defaults to config.epochs)
            
        Returns:
            Decayed parameter value
        """
        if total_epochs is None:
            total_epochs = self.config.epochs
        
        # Exponential decay
        decay_rate = -np.log(min_value / initial_value) / total_epochs
        value = initial_value * np.exp(-decay_rate * current_epoch)
        
        return max(value, min_value)
    
    def _euclidean_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized Euclidean distance.
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Distance array
        """
        return np.sqrt(np.sum((x - y) ** 2, axis=-1))
    
    def _manhattan_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized Manhattan distance.
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Distance array
        """
        return np.sum(np.abs(x - y), axis=-1)
    
    def _cosine_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized Cosine distance.
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Distance array (1 - cosine similarity)
        """
        # Normalize vectors
        x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        y_norm = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        cos_sim = np.sum(x_norm * y_norm, axis=-1)
        
        # Convert to distance
        return 1 - cos_sim
    
    def _bray_curtis_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bray-Curtis distance for biodiversity data.
        
        Bray-Curtis distance is particularly suitable for species abundance data
        as it accounts for the total abundance and is not affected by joint absences.
        
        Args:
            x: First array (species abundances)
            y: Second array (species abundances)
            
        Returns:
            Distance array (values between 0 and 1)
        """
        # Ensure non-negative values (abundances should be non-negative)
        x = np.maximum(x, 0)
        y = np.maximum(y, 0)
        
        # Calculate numerator and denominator
        numerator = np.sum(np.abs(x - y), axis=-1)
        denominator = np.sum(x + y, axis=-1)
        
        # Handle division by zero (when both samples have zero abundance)
        denominator = np.maximum(denominator, 1e-8)
        
        return numerator / denominator
    
    def get_distance_function(self) -> Callable:
        """Get the configured distance function.
        
        Returns:
            Distance function callable
        """
        return self._distance_functions[self.config.distance_metric]
    
    def grid_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two grid positions.
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
            
        Returns:
            Euclidean distance on the grid
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_grid_position(self, neuron_idx: int) -> Tuple[int, int]:
        """Convert neuron index to grid position.
        
        Args:
            neuron_idx: Flattened neuron index
            
        Returns:
            Tuple of (row, col) position
        """
        row = neuron_idx // self.grid_cols
        col = neuron_idx % self.grid_cols
        return (row, col)
    
    def get_neuron_index(self, row: int, col: int) -> int:
        """Convert grid position to neuron index.
        
        Args:
            row: Grid row
            col: Grid column
            
        Returns:
            Flattened neuron index
        """
        return row * self.grid_cols + col
    
    def get_neighbors(self, neuron_idx: int, radius: float) -> np.ndarray:
        """Get indices of neurons within radius of given neuron.
        
        Args:
            neuron_idx: Center neuron index
            radius: Neighborhood radius
            
        Returns:
            Array of neighbor indices
        """
        center_pos = self._grid_positions[neuron_idx]
        distances = np.sqrt(np.sum((self._grid_positions - center_pos)**2, axis=1))
        return np.where(distances <= radius)[0]
    
    def check_convergence(self, errors: List[float]) -> bool:
        """Check if training has converged based on error history.
        
        Args:
            errors: List of recent errors
            
        Returns:
            True if converged, False otherwise
        """
        if len(errors) < self.config.convergence_window:
            return False
        
        recent_errors = errors[-self.config.convergence_window:]
        
        # Check if error reduction is below threshold
        if recent_errors[0] > 0:
            improvement = (recent_errors[0] - recent_errors[-1]) / recent_errors[0]
            return improvement < self.config.convergence_threshold
        
        return False