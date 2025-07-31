"""Self-Organizing Map analyzer interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from .analyzer import IAnalyzer, AnalysisResult


class ISOMAnalyzer(IAnalyzer, ABC):
    """Interface for Self-Organizing Map analyzers."""
    
    @abstractmethod
    def train(self, data: np.ndarray, epochs: int, 
              learning_rate: float, neighborhood_radius: float) -> None:
        """Train the SOM on data.
        
        Args:
            data: Training data array of shape (n_samples, n_features)
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            neighborhood_radius: Initial neighborhood radius
        """
        pass
    
    @abstractmethod
    def find_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        """Find Best Matching Unit for a sample.
        
        Args:
            sample: Data sample of shape (n_features,)
            
        Returns:
            Tuple of (row, col) indices of the BMU in the SOM grid
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """Get the current weight matrix.
        
        Returns:
            Weight matrix of shape (n_rows, n_cols, n_features)
        """
        pass
    
    @abstractmethod
    def calculate_quantization_error(self, data: np.ndarray) -> float:
        """Calculate quantization error for the given data.
        
        Quantization error is the average distance between each data point
        and its best matching unit.
        
        Args:
            data: Data array of shape (n_samples, n_features)
            
        Returns:
            Average quantization error
        """
        pass
    
    @abstractmethod
    def calculate_topographic_error(self, data: np.ndarray) -> float:
        """Calculate topographic error for the given data.
        
        Topographic error is the proportion of data points for which the
        first and second BMUs are not adjacent in the SOM grid.
        
        Args:
            data: Data array of shape (n_samples, n_features)
            
        Returns:
            Topographic error (proportion between 0 and 1)
        """
        pass