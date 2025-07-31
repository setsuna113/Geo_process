"""SOM Analyzer for biodiversity data analysis."""

from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
from src.biodiversity_analysis.base_analyzer import BaseBiodiversityAnalyzer
from src.abstractions.interfaces.som_analyzer import ISOMAnalyzer
from src.abstractions.interfaces.analyzer import AnalysisResult
from src.abstractions.types.som_types import (
    SOMConfig, DistanceMetric, NeighborhoodFunction, 
    InitializationMethod, BiodiversityMetrics
)
from .som_core import BiodiversitySOM
from src.biodiversity_analysis.shared.metrics.som_metrics import SOMMetrics
from src.biodiversity_analysis.shared.spatial.spatial_splitter import SpatialSplitter, SpatialSplitStrategy
import logging
import json
import os

logger = logging.getLogger(__name__)


class SOMAnalyzer(BaseBiodiversityAnalyzer, ISOMAnalyzer):
    """Biodiversity-specific SOM analyzer.
    
    Integrates with the biodiversity analysis framework and provides:
    - Automated preprocessing for biodiversity data
    - Spatial validation support
    - Biodiversity-specific metrics
    - Integration with pipeline system
    - Result persistence and visualization
    """
    
    def __init__(self, config: Dict[str, Any], db=None):
        """Initialize SOM analyzer.
        
        Args:
            config: Configuration dictionary
            db: Optional database connection
        """
        super().__init__(method_name='som', version='2.0.0')
        self.config = config
        self.db = db
        self.som = None
        self.metrics_calculator = SOMMetrics()
        self.training_result = None
        self._som_config = None
    
    def analyze(self, dataset, **kwargs) -> AnalysisResult:
        """Main analysis method following IAnalyzer interface.
        
        Args:
            dataset: Dataset object with file_path attribute
            **kwargs: Additional parameters including:
                - spatial_validation: Whether to use spatial validation
                - spatial_strategy: Spatial split strategy
                - save_results: Whether to save results
                - output_dir: Directory for saving results
                
        Returns:
            AnalysisResult with SOM analysis outputs
        """
        logger.info("Starting SOM analysis for biodiversity data")
        
        # Load biodiversity data
        biodiv_data = self.load_data(dataset.file_path)
        logger.info(f"Loaded data: {biodiv_data.features.shape[0]} samples, "
                   f"{biodiv_data.features.shape[1]} features")
        
        # Apply preprocessing
        processed_data = self.preprocess_data(biodiv_data)
        
        # Prepare data splits
        if kwargs.get('spatial_validation', True):
            data_splits = self._perform_spatial_split(
                processed_data.features,
                processed_data.coordinates,
                strategy=kwargs.get('spatial_strategy', 'random_blocks')
            )
        else:
            # Simple random split
            data_splits = self._random_split(
                processed_data.features,
                train_ratio=0.7,
                val_ratio=0.15
            )
        
        # Create SOM configuration
        self._som_config = self._create_som_config(**kwargs)
        self.som = BiodiversitySOM(self._som_config)
        
        # Train SOM on training data
        logger.info("Training SOM...")
        self.training_result = self.som.train(
            data_splits['train'],
            progress_callback=self.progress_callback
        )
        
        # Evaluate on all data splits
        evaluation_results = self._evaluate_som(
            data_splits,
            processed_data,
            self.training_result
        )
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance()
        
        # Prepare data for create_result
        result_data = {
            'labels': evaluation_results['cluster_assignments'],
            'statistics': {
                'training_metrics': {
                    'quantization_errors': self.training_result.quantization_errors,
                    'topographic_errors': self.training_result.topographic_errors,
                    'convergence_epoch': self.training_result.convergence_epoch,
                    'training_time': self.training_result.training_time
                },
                'validation_metrics': evaluation_results['validation_metrics'],
                'biodiversity_metrics': evaluation_results['biodiversity_metrics'].to_dict()
            }
        }
        
        # Create analysis result using base class method
        result = self.create_result(
            success=True,
            data=result_data,
            runtime_seconds=self.training_result.training_time,
            n_samples=len(processed_data.features),
            n_features=processed_data.features.shape[1],
            parameters=self._som_config.__dict__
        )
        
        # Add additional metadata
        result.additional_outputs['spatial_info'] = {
            'bounds': processed_data.get_bounds() if hasattr(processed_data, 'get_bounds') else None,
            'crs': getattr(processed_data, 'crs', None)
        }
        result.additional_outputs['data_info'] = {
            'n_samples': len(processed_data.features),
            'n_features': processed_data.features.shape[1],
            'feature_names': processed_data.feature_names,
            'zero_inflated': getattr(processed_data, 'zero_inflated', False)
        }
        result.additional_outputs['feature_importance'] = feature_importance
        
        # Save results if requested
        if kwargs.get('save_results', False):
            self._save_results(result, kwargs.get('output_dir', './results'))
        
        return result
    
    def _perform_spatial_split(self, features: np.ndarray, 
                             coordinates: np.ndarray,
                             strategy: str = 'random_blocks') -> Dict[str, np.ndarray]:
        """Perform spatial data splitting.
        
        Args:
            features: Feature data
            coordinates: Spatial coordinates
            strategy: Splitting strategy name
            
        Returns:
            Dictionary with train, validation, and test data
        """
        # Convert strategy string to enum
        strategy_map = {
            'random_blocks': SpatialSplitStrategy.RANDOM_BLOCKS,
            'systematic_blocks': SpatialSplitStrategy.SYSTEMATIC_BLOCKS,
            'latitudinal': SpatialSplitStrategy.LATITUDINAL,
            'environmental_blocks': SpatialSplitStrategy.ENVIRONMENTAL_BLOCKS
        }
        
        split_strategy = strategy_map.get(strategy, SpatialSplitStrategy.RANDOM_BLOCKS)
        
        # Create spatial splitter
        splitter = SpatialSplitter(
            strategy=split_strategy,
            train_ratio=0.7,
            val_ratio=0.15,
            buffer_distance=0.1,  # 10% buffer between regions
            random_state=self.config.get('random_seed', 42)
        )
        
        # Perform split
        train_idx, val_idx, test_idx = splitter.split(coordinates, features)
        
        logger.info(f"Spatial split: {len(train_idx)} train, "
                   f"{len(val_idx)} validation, {len(test_idx)} test samples")
        
        return {
            'train': features[train_idx],
            'validation': features[val_idx],
            'test': features[test_idx],
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
    
    def _random_split(self, features: np.ndarray, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Dict[str, np.ndarray]:
        """Perform random data splitting.
        
        Args:
            features: Feature data
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            Dictionary with data splits
        """
        n_samples = len(features)
        indices = np.random.permutation(n_samples)
        
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        return {
            'train': features[train_idx],
            'validation': features[val_idx],
            'test': features[test_idx],
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
    
    def _create_som_config(self, **kwargs) -> SOMConfig:
        """Create SOM configuration from config dict and kwargs.
        
        Args:
            **kwargs: Override parameters
            
        Returns:
            SOMConfig object
        """
        som_params = self.config.get('som_analysis', {})
        
        # Merge with kwargs
        params = {**som_params, **kwargs}
        
        return SOMConfig(
            grid_size=tuple(params.get('grid_size', params.get('default_grid_size', [10, 10]))),
            distance_metric=DistanceMetric(params.get('distance_metric', 'euclidean')),
            neighborhood_function=NeighborhoodFunction(params.get('neighborhood_function', 'gaussian')),
            initialization_method=InitializationMethod(params.get('initialization_method', 'pca')),
            learning_rate=params.get('learning_rate', 0.5),
            min_learning_rate=params.get('min_learning_rate', 0.01),
            neighborhood_radius=params.get('sigma', 1.0),
            min_neighborhood_radius=params.get('min_neighborhood_radius', 0.1),
            epochs=params.get('iterations', 100),
            batch_size=params.get('batch_size', None),
            random_seed=params.get('random_seed', self.config.get('random_seed', None))
        )
    
    def _evaluate_som(self, data_splits: Dict, 
                     biodiv_data, 
                     training_result) -> Dict[str, Any]:
        """Comprehensive evaluation of trained SOM.
        
        Args:
            data_splits: Dictionary with data splits
            biodiv_data: Original biodiversity data
            training_result: SOM training result
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        # Get all data for cluster assignments
        all_indices = np.concatenate([
            data_splits.get('train_idx', []),
            data_splits.get('val_idx', []),
            data_splits.get('test_idx', [])
        ])
        
        # Sort to restore original order
        sorted_idx = np.argsort(all_indices)
        all_data = np.vstack([
            data_splits['train'],
            data_splits['validation'],
            data_splits['test']
        ])[sorted_idx]
        
        # Get cluster assignments
        bmu_indices = self.som.predict(all_data)
        results['cluster_assignments'] = self.som.transform(all_data)
        
        # Standard SOM metrics
        results['validation_metrics'] = {
            'train_qe': self.som.calculate_quantization_error(data_splits['train']),
            'val_qe': self.som.calculate_quantization_error(data_splits['validation']),
            'test_qe': self.som.calculate_quantization_error(data_splits['test']),
            'train_te': self.som.calculate_topographic_error(data_splits['train']),
            'val_te': self.som.calculate_topographic_error(data_splits['validation']),
            'test_te': self.som.calculate_topographic_error(data_splits['test'])
        }
        
        # Biodiversity-specific metrics
        biodiv_metrics = BiodiversityMetrics(
            species_coherence=self.metrics_calculator.calculate_species_coherence(
                self.som.weights,
                biodiv_data.features,
                bmu_indices,
                biodiv_data.feature_names
            ),
            beta_diversity_preservation=self.metrics_calculator.calculate_beta_diversity_preservation(
                biodiv_data.features,
                bmu_indices,
                distance_metric='bray_curtis' if self._is_abundance_data(biodiv_data) else 'jaccard'
            ),
            environmental_gradient_detection={},  # Would need environmental data
            rare_species_representation=self.metrics_calculator.calculate_rare_species_representation(
                biodiv_data.features,
                bmu_indices,
                prevalence_threshold=0.1
            ),
            spatial_autocorrelation_preserved=self.metrics_calculator.calculate_spatial_autocorrelation_preserved(
                biodiv_data.coordinates,
                biodiv_data.features,
                bmu_indices
            ) if biodiv_data.coordinates is not None else 0.0
        )
        
        results['biodiversity_metrics'] = biodiv_metrics
        
        return results
    
    def _is_abundance_data(self, biodiv_data) -> bool:
        """Check if data contains abundance values or just presence/absence.
        
        Args:
            biodiv_data: Biodiversity data object
            
        Returns:
            True if abundance data, False if binary
        """
        # Check if data has non-binary values
        unique_values = np.unique(biodiv_data.features)
        return len(unique_values) > 2 or not np.array_equal(unique_values, [0, 1])
    
    def _calculate_feature_importance(self) -> Optional[np.ndarray]:
        """Calculate feature importance based on weight variance.
        
        Returns:
            Array of feature importance scores
        """
        if self.som is None or self.som.weights is None:
            return None
        
        # Calculate variance of each feature across SOM weights
        weights_reshaped = self.som.get_weights()
        feature_variance = np.var(weights_reshaped, axis=(0, 1))
        
        # Normalize to sum to 1
        if np.sum(feature_variance) > 0:
            feature_importance = feature_variance / np.sum(feature_variance)
        else:
            feature_importance = np.ones(len(feature_variance)) / len(feature_variance)
        
        return feature_importance
    
    def _save_results(self, result: AnalysisResult, output_dir: str):
        """Save analysis results to disk.
        
        Args:
            result: Analysis result
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'som_analysis_metadata.json')
        # Convert dataclass to dict
        metadata_dict = {
            'analysis_type': result.metadata.analysis_type,
            'input_shape': result.metadata.input_shape,
            'input_bands': result.metadata.input_bands,
            'parameters': result.metadata.parameters,
            'processing_time': result.metadata.processing_time,
            'timestamp': result.metadata.timestamp,
            'data_source': result.metadata.data_source,
            'normalization_applied': result.metadata.normalization_applied,
            'coordinate_system': result.metadata.coordinate_system
        }
        # Add statistics
        metadata_dict['statistics'] = result.statistics
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Save labels
        labels_path = os.path.join(output_dir, 'som_cluster_assignments.npy')
        np.save(labels_path, result.labels)
        
        # Save weights
        if self.som is not None:
            weights_path = os.path.join(output_dir, 'som_weights.npy')
            np.save(weights_path, self.som.get_weights())
        
        # Save feature importance
        if result.additional_outputs and 'feature_importance' in result.additional_outputs:
            importance_path = os.path.join(output_dir, 'feature_importance.npy')
            np.save(importance_path, result.additional_outputs['feature_importance'])
        
        logger.info(f"Results saved to {output_dir}")
    
    def train(self, data: np.ndarray, epochs: int, 
              learning_rate: float, neighborhood_radius: float) -> None:
        """Train the SOM (ISOMAnalyzer interface).
        
        Args:
            data: Training data
            epochs: Number of epochs
            learning_rate: Learning rate
            neighborhood_radius: Neighborhood radius
        """
        # Create config if not already created
        if self._som_config is None:
            self._som_config = self._create_som_config()
        
        # Update config parameters
        self._som_config.epochs = epochs
        self._som_config.learning_rate = learning_rate
        self._som_config.neighborhood_radius = neighborhood_radius
        
        self.som = BiodiversitySOM(self._som_config)
        self.training_result = self.som.train(data)
    
    def find_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        """Find Best Matching Unit for a sample.
        
        Args:
            sample: Data sample
            
        Returns:
            Tuple of (row, col) position
        """
        if self.som is None:
            raise ValueError("SOM must be trained first")
        return self.som.find_bmu(sample)
    
    def get_weights(self) -> np.ndarray:
        """Get the current weight matrix.
        
        Returns:
            Weight matrix
        """
        if self.som is None:
            raise ValueError("SOM must be trained first")
        return self.som.get_weights()
    
    def calculate_quantization_error(self, data: np.ndarray) -> float:
        """Calculate quantization error.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Quantization error
        """
        if self.som is None:
            raise ValueError("SOM must be trained first")
        return self.som.calculate_quantization_error(data)
    
    def calculate_topographic_error(self, data: np.ndarray) -> float:
        """Calculate topographic error.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Topographic error
        """
        if self.som is None:
            raise ValueError("SOM must be trained first")
        return self.som.calculate_topographic_error(data)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate SOM parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        required = ['grid_size']
        
        for param in required:
            if param not in parameters:
                issues.append(f"Missing required parameter: {param}")
        
        # Validate grid_size format
        if 'grid_size' in parameters:
            grid_size = parameters['grid_size']
            if not (isinstance(grid_size, (list, tuple)) and len(grid_size) == 2):
                issues.append("grid_size must be a list or tuple of 2 integers")
            elif not all(isinstance(x, int) and x > 0 for x in grid_size):
                issues.append("grid_size dimensions must be positive integers")
        
        # Validate optional parameters
        if 'distance_metric' in parameters:
            valid_metrics = ['euclidean', 'manhattan', 'cosine', 'bray_curtis']
            if parameters['distance_metric'] not in valid_metrics:
                issues.append(f"Invalid distance_metric. Must be one of: {valid_metrics}")
        
        if 'epochs' in parameters:
            if not isinstance(parameters['epochs'], int) or parameters['epochs'] <= 0:
                issues.append("epochs must be a positive integer")
        
        if 'learning_rate' in parameters:
            if not isinstance(parameters['learning_rate'], (int, float)) or parameters['learning_rate'] <= 0:
                issues.append("learning_rate must be a positive number")
        
        return len(issues) == 0, issues
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for SOM analysis.
        
        Returns:
            Dictionary of default parameter values
        """
        return {
            'grid_size': [10, 10],
            'distance_metric': 'euclidean',
            'epochs': 100,
            'learning_rate': 0.5,
            'min_learning_rate': 0.01,
            'neighborhood_function': 'gaussian',
            'neighborhood_radius': 1.0,
            'min_neighborhood_radius': 0.1,
            'initialization_method': 'pca',
            'batch_size': None,
            'spatial_validation': True,
            'spatial_strategy': 'random_blocks'
        }
    
    def save_results(self, result: AnalysisResult, output_path: str) -> None:
        """Save analysis results to specified path.
        
        Args:
            result: Analysis result to save
            output_path: Path to save results
        """
        self._save_results(result, output_path)
    
    def set_progress_callback(self, callback: Optional[Callable[[str, float], None]]) -> None:
        """Set progress callback function.
        
        Args:
            callback: Progress callback function
        """
        self.progress_callback = callback
    
    def validate_input_data(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate input data for SOM analysis.
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if it's a dataset object with file_path
        if not hasattr(data, 'file_path'):
            issues.append("Input must be a dataset object with 'file_path' attribute")
            return False, issues
        
        # Check if file exists
        import os
        if not os.path.exists(data.file_path):
            issues.append(f"File not found: {data.file_path}")
        
        # Check file extension
        if not data.file_path.endswith('.parquet'):
            issues.append("Only .parquet files are supported")
        
        return len(issues) == 0, issues