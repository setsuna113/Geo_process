# src/spatial_analysis/som/som_trainer.py
"""Self-Organizing Map trainer for biodiversity pattern analysis."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False
    MiniSom = None  # type: ignore
    import warnings
    warnings.warn("minisom not available. Install with: pip install minisom")

from src.spatial_analysis.base_analyzer import BaseAnalyzer, AnalysisResult, AnalysisMetadata
from src.config.config import Config
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class SOMAnalyzer(BaseAnalyzer):
    """
    Self-Organizing Map analyzer for clustering biodiversity patterns.
    
    Groups pixels with similar P, A, F profiles into clusters while
    preserving topological relationships.
    """
    
    def __init__(self, config: Config, db_connection: Optional[DatabaseManager] = None):
        super().__init__(config, db_connection)

        if MiniSom is None:
            raise ImportError("minisom is required for SOM analysis. Install with: pip install minisom")
        
        # SOM-specific config
        som_config = config.get('spatial_analysis', {}).get('som', {})
        self.default_grid_size = som_config.get('grid_size', [10, 10])
        self.default_iterations = som_config.get('iterations', 1000)
        self.default_sigma = som_config.get('sigma', 1.0)
        self.default_learning_rate = som_config.get('learning_rate', 0.5)
        self.default_neighborhood = som_config.get('neighborhood_function', 'gaussian')
        
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default SOM parameters."""
        return {
            'grid_size': self.default_grid_size,
            'iterations': self.default_iterations,
            'sigma': self.default_sigma,
            'learning_rate': self.default_learning_rate,
            'neighborhood_function': self.default_neighborhood,
            'random_seed': 42,
            'topology': 'rectangular'
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate SOM parameters."""
        issues = []
        
        # Check grid size
        grid_size = parameters.get('grid_size', self.default_grid_size)
        if not isinstance(grid_size, (list, tuple)) or len(grid_size) != 2:
            issues.append("grid_size must be a list/tuple of 2 integers")
        elif not all(isinstance(x, int) and x > 0 for x in grid_size):
            issues.append("grid_size values must be positive integers")
        
        # Check iterations
        iterations = parameters.get('iterations', self.default_iterations)
        if not isinstance(iterations, int) or iterations <= 0:
            issues.append("iterations must be a positive integer")
        
        # Check sigma
        sigma = parameters.get('sigma', self.default_sigma)
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            issues.append("sigma must be a positive number")
        
        # Check learning rate
        lr = parameters.get('learning_rate', self.default_learning_rate)
        if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
            issues.append("learning_rate must be between 0 and 1")
        
        # Check neighborhood function
        valid_neighborhoods = ['gaussian', 'mexican_hat', 'bubble', 'triangle']
        neighborhood = parameters.get('neighborhood_function', self.default_neighborhood)
        if neighborhood not in valid_neighborhoods:
            issues.append(f"neighborhood_function must be one of {valid_neighborhoods}")
        
        return len(issues) == 0, issues
    
    def analyze(self, 
                data,
                grid_size: Optional[List[int]] = None,
                iterations: Optional[int] = None,
                sigma: Optional[float] = None,
                learning_rate: Optional[float] = None,
                neighborhood_function: Optional[str] = None,
                random_seed: Optional[int] = None,
                **kwargs) -> AnalysisResult:
        """
        Perform SOM analysis on biodiversity data.
        
        Args:
            data: Input data (P, A, F values)
            grid_size: SOM grid dimensions [rows, cols]
            iterations: Number of training iterations
            sigma: Spread of neighborhood function
            learning_rate: Initial learning rate
            neighborhood_function: Type of neighborhood function
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            AnalysisResult with cluster assignments and SOM properties
        """
        logger.info("Starting SOM analysis")
        start_time = time.time()
        
        # Prepare parameters
        params = self.get_default_parameters()
        params.update({
            'grid_size': grid_size or params['grid_size'],
            'iterations': iterations or params['iterations'],
            'sigma': sigma or params['sigma'],
            'learning_rate': learning_rate or params['learning_rate'],
            'neighborhood_function': neighborhood_function or params['neighborhood_function'],
            'random_seed': random_seed or params['random_seed']
        })
        
        # Validate parameters
        valid, issues = self.validate_parameters(params)
        if not valid:
            raise ValueError(f"Invalid parameters: {'; '.join(issues)}")
        
        # Validate input
        valid, issues = self.validate_input_data(data)
        if not valid:
            raise ValueError(f"Invalid input data: {'; '.join(issues)}")
        
        # Prepare data
        self._update_progress(1, 5, "Preparing data")
        prepared_data, metadata = self.prepare_data(data, flatten=True)
        
        # Ensure 2D shape (n_samples, n_features)
        if prepared_data.ndim == 1:
            prepared_data = prepared_data.reshape(-1, 1)
        
        n_samples, n_features = prepared_data.shape
        logger.info(f"Training SOM on {n_samples} samples with {n_features} features")
        
        # Initialize SOM
        self._update_progress(2, 5, "Initializing SOM")
        if MiniSom is None:

            raise ImportError("minisom is required")

        if MiniSom is None:


            raise ImportError("minisom is required for SOM analysis")


        som = MiniSom(
            x=params['grid_size'][0],
            y=params['grid_size'][1],
            input_len=n_features,
            sigma=params['sigma'],
            learning_rate=params['learning_rate'],
            neighborhood_function=params['neighborhood_function'],
            random_seed=params['random_seed']
        )
        
        # Train SOM
        self._update_progress(3, 5, "Training SOM")
        som.train_random(prepared_data, params['iterations'])  # type: ignore[attr-defined]
        
        # Get cluster assignments
        self._update_progress(4, 5, "Assigning clusters")
        labels = np.zeros(n_samples, dtype=int)
        
        for i, sample in enumerate(prepared_data):
            winner = som.winner(sample)  # type: ignore  # type: ignore[attr-defined]
            # Convert 2D grid position to 1D cluster ID
            labels[i] = winner[0] * params['grid_size'][1] + winner[1]
        
        # Calculate statistics
        self._update_progress(5, 5, "Calculating statistics")
        statistics = self._calculate_statistics(prepared_data, labels, som, params)
        
        # Restore spatial structure
        spatial_output = self.restore_spatial_structure(labels, metadata)
        
        # Create metadata
        analysis_metadata = AnalysisMetadata(
            analysis_type='SOM',
            input_shape=metadata['original_shape'],
            input_bands=metadata.get('bands', ['unknown']),
            parameters=params,
            processing_time=time.time() - start_time,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            normalization_applied=metadata.get('normalized', False)
        )
        
        # Store additional outputs
        additional_outputs = {
            'som_weights': som.get_weights(),
            'distance_map': som.distance_map(),
            'quantization_error': som.quantization_error(prepared_data),
            'topographic_error': som.topographic_error(prepared_data),
            'activation_map': self._get_activation_map(som, prepared_data),
            'component_planes': self._get_component_planes(som, metadata)
        }
        
        result = AnalysisResult(
            labels=labels,
            metadata=analysis_metadata,
            statistics=statistics,
            spatial_output=spatial_output,
            additional_outputs=additional_outputs
        )
        
        # Store in database if configured
        if self.save_results:
            self.store_in_database(result)
        
        return result
    
    def _calculate_statistics(self, data: np.ndarray, labels: np.ndarray, 
                            som: MiniSom, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SOM statistics."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Cluster statistics
        cluster_stats = {}
        for label in unique_labels:
            mask = labels == label
            cluster_data = data[mask]
            
            cluster_stats[int(label)] = {
                'count': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(labels) * 100),
                'mean': cluster_data.mean(axis=0).tolist(),
                'std': cluster_data.std(axis=0).tolist(),
                'min': cluster_data.min(axis=0).tolist(),
                'max': cluster_data.max(axis=0).tolist()
            }
        
        # Global statistics
        return {
            'n_clusters': n_clusters,
            'grid_size': params['grid_size'],
            "quantization_error": float(som.quantization_error(data)),  # type: ignore[attr-defined]
            "topographic_error": float(som.topographic_error(data)),  # type: ignore[attr-defined]
            'cluster_statistics': cluster_stats,
            'empty_neurons': self._count_empty_neurons(labels, params['grid_size']),
            'cluster_balance': self._calculate_cluster_balance(labels)
        }
    
    def _get_activation_map(self, som: Any, data: np.ndarray) -> np.ndarray:
        """Get activation frequency for each neuron."""
        activation_map = np.zeros(som.get_weights().shape[:2])  # type: ignore[attr-defined]
        
        for sample in data:
            winner = som.winner(sample)  # type: ignore  # type: ignore[attr-defined]
            activation_map[winner] += 1
        
        return activation_map
    
    def _get_component_planes(self, som: Any, metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract component planes for each feature."""
        weights = som.get_weights()  # type: ignore  # type: ignore[attr-defined]
        component_planes = {}
        
        band_names = metadata.get('bands', [f'feature_{i}' for i in range(weights.shape[2])])
        
        for i, band in enumerate(band_names[:weights.shape[2]]):
            component_planes[band] = weights[:, :, i]
        
        return component_planes
    
    def _count_empty_neurons(self, labels: np.ndarray, grid_size: List[int]) -> int:
        """Count neurons with no assigned samples."""
        total_neurons = grid_size[0] * grid_size[1]
        used_neurons = len(np.unique(labels))
        return total_neurons - used_neurons
    
    def _calculate_cluster_balance(self, labels: np.ndarray) -> float:
        """Calculate how evenly distributed samples are across clusters."""
        _, counts = np.unique(labels, return_counts=True)
        # Use coefficient of variation
        return float(np.std(counts) / np.mean(counts))