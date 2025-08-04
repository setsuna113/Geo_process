"""Interface definitions for tensor-based analyzers.

This module defines the abstract interface that tensor factorization
methods must implement to integrate with the biodiversity analysis pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np

from .analyzer import IAnalyzer, AnalysisResult
from ..types.tensor_types import (
    TensorData, CPFactors, TensorMetrics, TensorDecompositionResult,
    TensorOptimizer, FactorInitMethod
)


class ITensorAnalyzer(IAnalyzer, ABC):
    """Extended interface for tensor-based analysis methods.
    
    This interface extends IAnalyzer with tensor-specific functionality
    while maintaining compatibility with the existing pipeline.
    """
    
    @abstractmethod
    def create_tensor(self, data_path: str, **kwargs) -> TensorData:
        """Create tensor representation from input data.
        
        Args:
            data_path: Path to input parquet file
            **kwargs: Additional parameters for tensor construction
                - lat_bins: Number of latitude bins
                - lon_bins: Number of longitude bins
                - feature_cols: List of feature columns to include
                
        Returns:
            TensorData object ready for factorization
        """
        pass
    
    @abstractmethod
    def factorize(self, tensor_data: TensorData, rank: int, 
                  **parameters) -> TensorDecompositionResult:
        """Perform tensor factorization.
        
        Args:
            tensor_data: Input tensor data
            rank: Number of components
            **parameters: Method-specific parameters
            
        Returns:
            TensorDecompositionResult with factors and metrics
        """
        pass
    
    @abstractmethod
    def evaluate_rank(self, tensor_data: TensorData, 
                     rank_range: Tuple[int, int]) -> Dict[int, float]:
        """Evaluate different rank values for model selection.
        
        Args:
            tensor_data: Input tensor data
            rank_range: Range of ranks to test (min, max)
            
        Returns:
            Dictionary mapping rank to quality metric (e.g., reconstruction error)
        """
        pass
    
    @abstractmethod
    def interpret_factors(self, factors: CPFactors, 
                         tensor_data: TensorData) -> Dict[str, Any]:
        """Interpret the meaning of extracted factors.
        
        Args:
            factors: Decomposed factor matrices
            tensor_data: Original tensor data for context
            
        Returns:
            Dictionary with interpretations:
                - 'spatial_patterns': Dominant spatial patterns
                - 'feature_loadings': Feature importance per component
                - 'component_descriptions': Text descriptions
        """
        pass


class ITensorPreprocessor(ABC):
    """Interface for tensor-specific preprocessing."""
    
    @abstractmethod
    def grid_spatial_data(self, coordinates: np.ndarray, values: np.ndarray,
                         lat_bins: int, lon_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Grid irregular spatial data into regular tensor format.
        
        Args:
            coordinates: (N, 2) array of lat/lon coordinates
            values: (N, M) array of feature values
            lat_bins: Number of latitude bins
            lon_bins: Number of longitude bins
            
        Returns:
            Tuple of (gridded_tensor, grid_coordinates)
        """
        pass
    
    @abstractmethod
    def create_weight_mask(self, tensor: np.ndarray) -> np.ndarray:
        """Create weight mask for missing values.
        
        Args:
            tensor: Input tensor (may contain NaN)
            
        Returns:
            Binary weight mask (1 for observed, 0 for missing)
        """
        pass
    
    @abstractmethod
    def transform_features(self, tensor: np.ndarray, 
                          method: str = 'log1p') -> np.ndarray:
        """Apply feature transformation suitable for tensor factorization.
        
        Args:
            tensor: Input tensor
            method: Transformation method ('log1p', 'sqrt', 'none')
            
        Returns:
            Transformed tensor
        """
        pass
    
    @abstractmethod
    def normalize_modes(self, tensor: np.ndarray, 
                       center: bool = True,
                       scale: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize tensor modes (dimensions).
        
        Args:
            tensor: Input tensor
            center: Whether to center each mode
            scale: Whether to scale each mode
            
        Returns:
            Tuple of (normalized_tensor, normalization_params)
        """
        pass


class ITensorMetricsCalculator(ABC):
    """Interface for calculating tensor-specific metrics."""
    
    @abstractmethod
    def reconstruction_error(self, original: np.ndarray, 
                           reconstructed: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> float:
        """Calculate weighted reconstruction error.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
            weights: Optional weight mask
            
        Returns:
            Weighted Frobenius norm of residual
        """
        pass
    
    @abstractmethod
    def tucker_congruence(self, factors1: CPFactors, 
                         factors2: CPFactors) -> np.ndarray:
        """Calculate Tucker congruence between two sets of factors.
        
        Args:
            factors1: First set of factors
            factors2: Second set of factors
            
        Returns:
            Congruence coefficients per component
        """
        pass
    
    @abstractmethod
    def core_consistency(self, tensor: np.ndarray, 
                        factors: CPFactors) -> float:
        """Calculate core consistency diagnostic.
        
        Args:
            tensor: Original tensor
            factors: CP factors
            
        Returns:
            Core consistency value (100 = perfect, <0 = poor model)
        """
        pass
    
    @abstractmethod
    def factor_match_score(self, factors: CPFactors, 
                          reference: CPFactors) -> float:
        """Calculate how well factors match a reference decomposition.
        
        Args:
            factors: Factors to evaluate
            reference: Reference factors
            
        Returns:
            Match score between 0 and 1
        """
        pass


class ITensorReporter(ABC):
    """Interface for tensor factorization result reporting."""
    
    @abstractmethod
    def generate_report(self, result: TensorDecompositionResult,
                       output_dir: str) -> Dict[str, str]:
        """Generate comprehensive report of tensor factorization results.
        
        Args:
            result: Tensor decomposition result
            output_dir: Directory to save report files
            
        Returns:
            Dictionary mapping report type to file path
        """
        pass
    
    @abstractmethod
    def visualize_factors(self, factors: CPFactors, 
                         mode: int,
                         output_path: str) -> None:
        """Visualize factor matrices for a specific mode.
        
        Args:
            factors: CP factors to visualize
            mode: Which mode to visualize
            output_path: Path to save visualization
        """
        pass
    
    @abstractmethod
    def plot_convergence(self, metrics: TensorMetrics,
                        output_path: str) -> None:
        """Plot convergence history.
        
        Args:
            metrics: Tensor metrics with convergence history
            output_path: Path to save plot
        """
        pass
    
    @abstractmethod
    def export_factors(self, factors: CPFactors,
                      format: str = 'csv',
                      output_dir: str = '.') -> List[str]:
        """Export factor matrices to files.
        
        Args:
            factors: CP factors to export
            format: Export format ('csv', 'npy', 'mat')
            output_dir: Directory to save files
            
        Returns:
            List of created file paths
        """
        pass