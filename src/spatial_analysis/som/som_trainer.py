# src/spatial_analysis/som/som_trainer.py
"""Self-Organizing Map trainer for biodiversity pattern analysis."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import tempfile
import os
import gc
from src.base.analyzer import BaseAnalyzer
from .manhattan_som import ManhattanSOMWrapper
from src.abstractions.interfaces.analyzer import AnalysisResult, AnalysisMetadata
from src.spatial_analysis.memory_aware_processor import (
    SubsamplingStrategy, 
    MemoryAwareProcessor, 
    check_memory_usage
)
from src.config.config import Config
from src.database.connection import DatabaseManager
from .dynamic_convergence import DynamicConvergenceDetector
from .advanced_convergence import AdvancedConvergenceDetector, create_advanced_convergence_detector
from .batch_unified_convergence import BatchUnifiedConvergenceDetector, create_batch_convergence_detector
from .biodiversity_som_validator import DataSplit

logger = logging.getLogger(__name__)

class SOMAnalyzer(BaseAnalyzer):
    """
    Self-Organizing Map analyzer for clustering biodiversity patterns.
    
    Groups pixels with similar P, A, F profiles into clusters while
    preserving topological relationships.
    """
    
    def __init__(self, config: Config, db_connection: Optional[DatabaseManager] = None):
        """
        Initialize SOM analyzer with standardized constructor.
        
        Args:
            config: Configuration object
            db_connection: Optional database connection
        """
        # Initialize base analyzer with config
        super().__init__(config, db_connection)

        # Always use Manhattan distance for biodiversity data (objectively better)
        logger.info("Using Manhattan distance SOM (optimized for biodiversity data)")
        
        # SOM-specific config using safe_get_config from base
        som_config = self.safe_get_config('spatial_analysis.som', {})
        self.default_grid_size = som_config.get('grid_size', [10, 10])
        self.default_iterations = som_config.get('iterations', 1000)
        self.default_sigma = som_config.get('sigma', 1.0)
        self.default_learning_rate = som_config.get('learning_rate', 0.5)
        self.default_neighborhood = som_config.get('neighborhood_function', 'gaussian')
        
        # Dynamic convergence configuration
        self.enable_dynamic_convergence = som_config.get('enable_dynamic_convergence', True)
        self.convergence_method = som_config.get('convergence_method', 'unified')  # 'unified', 'multi_criteria', 'batch_unified'
        self.convergence_config = som_config.get('convergence', {})
        
        # Progress callback for convergence tracking
        self.progress_callback = None
        
        # Memory-aware processing config using safe_get_config
        self.subsampling_config = self.safe_get_config('processing.subsampling', {})
        self.som_analysis_config = self.safe_get_config('som_analysis', {})
        self.max_pixels_in_memory = self.som_analysis_config.get('max_pixels_in_memory', 1000000)
        self.memory_overhead_factor = self.som_analysis_config.get('memory_overhead_factor', 3.0)
        self.use_memory_mapping = self.som_analysis_config.get('use_memory_mapping', True)
        
        # Initialize subsampling strategy
        self.subsampler = SubsamplingStrategy(self.subsampling_config)
        self.memory_processor = MemoryAwareProcessor(
            memory_limit_gb=self.subsampling_config.get('memory_limit_gb', 8.0)
        )
        
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default SOM parameters."""
        return {
            'grid_size': self.default_grid_size,
            'iterations': self.default_iterations,
            'sigma': self.default_sigma,
            'learning_rate': self.default_learning_rate,
            'neighborhood_function': self.default_neighborhood,
            'random_seed': 42,
            'topology': 'rectangular',
            'enable_dynamic_convergence': self.enable_dynamic_convergence,
            'convergence_method': self.convergence_method
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
        
        # Check convergence method
        valid_convergence_methods = ['unified', 'multi_criteria', 'batch_unified']
        convergence_method = parameters.get('convergence_method', 'unified')
        if convergence_method not in valid_convergence_methods:
            issues.append(f"convergence_method must be one of {valid_convergence_methods}")
        
        return len(issues) == 0, issues
    
    def estimate_memory_requirements(self, data_shape: Tuple[int, ...], 
                                   dtype: np.dtype = np.dtype(np.float64)) -> Dict[str, Any]:
        """Estimate memory requirements for SOM analysis."""
        memory_info = check_memory_usage(data_shape, dtype)
        
        # Add SOM-specific overhead with proper type conversion
        som_overhead_gb = float(memory_info['data_size_gb'] * self.memory_overhead_factor)
        memory_info['som_overhead_gb'] = som_overhead_gb
        memory_info['total_required_gb'] = som_overhead_gb
        memory_info['fits_in_memory'] = bool(som_overhead_gb < memory_info['available_gb'] * 0.6)
        
        return memory_info
    
    def set_progress_callback(self, callback):
        """Set progress callback for convergence tracking."""
        self.progress_callback = callback
    
    def train_with_dynamic_convergence(self,
                                     data: np.ndarray,
                                     som: Any,
                                     params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Train SOM with dynamic convergence detection."""
        # Create convergence detector based on method
        convergence_method = params.get('convergence_method', 'unified')
        
        if convergence_method == 'batch_unified':
            convergence_detector = create_batch_convergence_detector(
                **self.convergence_config
            )
        elif convergence_method == 'unified':
            convergence_detector = create_advanced_convergence_detector(
                method='unified',
                **self.convergence_config
            )
        else:
            # Multi-criteria fallback
            convergence_detector = DynamicConvergenceDetector(
                **self.convergence_config
            )
        
        logger.info(f"Training SOM with {convergence_method} convergence detection")
        
        max_iterations = params['iterations']
        actual_iterations = 0
        converged = False
        convergence_info = {}
        
        # Training loop with convergence checking
        for iteration in range(max_iterations):
            # Perform one iteration of training
            if convergence_method == 'batch_unified':
                # Batch training mode
                som.train_batch(data, 1, random_order=False)
                
                # Check convergence every 10 iterations
                if iteration > 0 and iteration % 10 == 0:
                    result = convergence_detector.check_batch_convergence(
                        som, data, iteration, max_iterations
                    )
                    converged = result.is_converged
                    convergence_info = {
                        'method': 'batch_unified',
                        'weight_stabilized': result.weight_stabilized,
                        'quality_converged': result.quality_converged,
                        'convergence_reason': result.convergence_reason,
                        'weight_change': result.weight_change,
                        'unified_index': result.unified_result.convergence_index
                    }
            else:
                # Sequential training mode
                som.train_random(data, 1)
                
                # Check convergence
                if convergence_detector.should_check_convergence(iteration):
                    if hasattr(convergence_detector, 'check_advanced_convergence'):
                        converged, conv_info = convergence_detector.check_advanced_convergence(
                            som, data, iteration, max_iterations
                        )
                        convergence_info = conv_info
                    else:
                        converged, metrics = convergence_detector.check_convergence(
                            som, data, iteration
                        )
                        convergence_info = {
                            'method': 'multi_criteria',
                            'weight_change': metrics.weight_change,
                            'quantization_error': metrics.quantization_error,
                            'topographic_error': metrics.topographic_error,
                            'convergence_score': metrics.convergence_score
                        }
            
            actual_iterations = iteration + 1
            
            # Progress callback
            if self.progress_callback and iteration % 50 == 0:
                self.progress_callback(
                    iteration, max_iterations, 
                    f"Training iteration {iteration}/{max_iterations}"
                )
            
            # Check if converged
            if converged:
                logger.info(f"SOM converged after {actual_iterations} iterations")
                break
        
        # Final convergence summary
        if hasattr(convergence_detector, 'get_batch_summary'):
            summary = convergence_detector.get_batch_summary()
        elif hasattr(convergence_detector, 'get_advanced_summary'):
            summary = convergence_detector.get_advanced_summary()
        else:
            summary = convergence_detector.get_convergence_summary()
        
        training_info = {
            'converged': converged,
            'actual_iterations': actual_iterations,
            'max_iterations': max_iterations,
            'convergence_method': convergence_method,
            'convergence_info': convergence_info,
            'convergence_summary': summary
        }
        
        return som, training_info
    
    def train_with_subsampling(self, 
                              data: np.ndarray, 
                              som: Any,
                              params: Dict[str, Any],
                              coordinates: Optional[np.ndarray] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Train SOM with automatic subsampling for large datasets.
        
        Args:
            data: Input data (n_samples, n_features)
            som: Initialized SOM object
            params: Training parameters
            coordinates: Spatial coordinates for stratified sampling
            
        Returns:
            Tuple of (trained_som, sampling_info)
        """
        n_samples, n_features = data.shape
        
        # Check memory requirements
        memory_info = self.estimate_memory_requirements(data.shape, data.dtype)
        logger.info(f"Data size: {memory_info['data_size_gb']:.2f} GB, "
                   f"SOM overhead: {memory_info['som_overhead_gb']:.2f} GB, "
                   f"Available: {memory_info['available_gb']:.2f} GB")
        
        # Decide if subsampling is needed
        needs_subsampling = (
            not memory_info['fits_in_memory'] or 
            n_samples > self.max_pixels_in_memory or
            self.subsampling_config.get('enabled', True)
        )
        
        if needs_subsampling:
            logger.warning(f"Dataset too large ({n_samples:,} samples), using subsampling")
            
            # First subsample without coordinates to avoid memory spike
            if coordinates is None:
                logger.info("Using random subsampling (no spatial coordinates provided)")
                # Use random sampling to avoid creating huge coordinate arrays
                train_data, sample_indices = self.subsampler.subsample_data(data, coordinates=None)
            else:
                # Use spatial coordinates if provided
                train_data, sample_indices = self.subsampler.subsample_data(data, coordinates)
            
            # Memory mapping with proper scope management (not needed for subsampled data)
            try:
                # Train with dynamic convergence if enabled
                if params.get('enable_dynamic_convergence', True):
                    logger.info(f"Training SOM on {train_data.shape[0]:,} samples (subsampled from {n_samples:,}) with dynamic convergence")
                    som, training_info = self.train_with_dynamic_convergence(train_data, som, params)
                else:
                    # Traditional fixed iteration training
                    logger.info(f"Training SOM on {train_data.shape[0]:,} samples (subsampled from {n_samples:,}) with fixed iterations")
                    som.train_random(train_data, params['iterations'])
                    training_info = {'converged': False, 'actual_iterations': params['iterations']}
                
                # Store sampling info with proper type conversion
                sampling_info = {
                    'used_subsampling': True,
                    'sample_indices': sample_indices,
                    'total_samples': int(n_samples),  # Ensure Python int
                    'train_samples': int(train_data.shape[0]),  # Ensure Python int
                    'sampling_ratio': float(train_data.shape[0] / n_samples),  # Ensure Python float
                    'training_info': training_info
                }
                
                # Force garbage collection
                gc.collect()
                
            finally:
                # Cleanup - minimal since we're using subsampled data directly
                pass
        else:
            # Train with dynamic convergence if enabled
            if params.get('enable_dynamic_convergence', True):
                logger.info(f"Training SOM on full dataset ({n_samples:,} samples) with dynamic convergence")
                som, training_info = self.train_with_dynamic_convergence(data, som, params)
            else:
                # Traditional fixed iteration training
                logger.info(f"Training SOM on full dataset ({n_samples:,} samples) with fixed iterations")
                som.train_random(data, params['iterations'])
                training_info = {'converged': False, 'actual_iterations': params['iterations']}
            
            sampling_info = {
                'used_subsampling': False,
                'total_samples': int(n_samples),  # Ensure Python int
                'train_samples': int(n_samples),  # Ensure Python int
                'sampling_ratio': float(1.0),  # Ensure Python float
                'training_info': training_info
            }
        
        return som, sampling_info
    
    def classify_in_batches(self, 
                           data: np.ndarray, 
                           som: Any, 
                           batch_size: Optional[int] = None) -> np.ndarray:
        """
        Classify large dataset in batches to avoid memory issues.
        
        Args:
            data: Full dataset to classify
            som: Trained SOM model
            batch_size: Size of processing batches
            
        Returns:
            Cluster labels for all samples
        """
        n_samples = data.shape[0]
        
        if batch_size is None:
            # Calculate optimal batch size with proper type conversion
            memory_available_mb = float(self.memory_processor.memory_limit_gb * 1024)
            sample_size_mb = float((data.nbytes / n_samples) / (1024 * 1024))
            batch_size = max(1000, int(memory_available_mb / (sample_size_mb * 2)))
        
        logger.info(f"Classifying {n_samples:,} samples in batches of {batch_size:,}")
        
        labels = np.zeros(n_samples, dtype=int)
        last_log_time = time.time()
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = data[start_idx:end_idx]
            
            # Process batch
            for i, sample in enumerate(batch_data):
                winner = som.winner(sample)
                # Convert 2D grid position to 1D cluster ID
                labels[start_idx + i] = winner[0] * som._weights.shape[1] + winner[1]
            
            # Throttled progress logging - update at most every 5 seconds
            progress = int((end_idx / n_samples) * 100)
            current_time = time.time()
            if current_time - last_log_time >= 5.0 or end_idx >= n_samples:
                batch_num = start_idx // batch_size + 1
                total_batches = int(np.ceil(n_samples / batch_size))
                logger.info(f"Classification progress: {progress}% - Batch {batch_num}/{total_batches}")
                last_log_time = current_time
            
            # Force garbage collection every few batches
            if (start_idx // batch_size) % 5 == 0:
                gc.collect()
        
        return labels
    
    def _safe_quantization_error(self, som: Any, data: np.ndarray, sampling_info: Dict[str, Any]) -> float:
        """Calculate quantization error safely for large datasets."""
        if sampling_info['used_subsampling']:
            # Use a small sample for error calculation
            sample_size = min(10000, data.shape[0])
            indices = np.random.choice(data.shape[0], sample_size, replace=False)
            sample_data = data[indices]
            error = som.quantization_error(sample_data)
            return float(error)  # Ensure Python float
        else:
            error = som.quantization_error(data)
            return float(error)  # Ensure Python float
    
    def _safe_topographic_error(self, som: Any, data: np.ndarray, sampling_info: Dict[str, Any]) -> float:
        """Calculate topographic error safely for large datasets."""
        if sampling_info['used_subsampling']:
            # Use a small sample for error calculation
            sample_size = min(10000, data.shape[0])
            indices = np.random.choice(data.shape[0], sample_size, replace=False)
            sample_data = data[indices]
            error = som.topographic_error(sample_data)
            return float(error)  # Ensure Python float
        else:
            error = som.topographic_error(data)
            return float(error)  # Ensure Python float
    
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
        
        # Add dynamic convergence parameters from kwargs
        params['enable_dynamic_convergence'] = kwargs.get('enable_dynamic_convergence', self.enable_dynamic_convergence)
        params['convergence_method'] = kwargs.get('convergence_method', self.convergence_method)
        params['convergence_threshold'] = kwargs.get('convergence_threshold', 1e-6)
        
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
        
        # Initialize Manhattan distance SOM (objectively better for biodiversity data)
        self._update_progress(2, 5, "Initializing Manhattan SOM")
        
        som = ManhattanSOMWrapper(
            x=params['grid_size'][0],
            y=params['grid_size'][1],
            input_len=n_features,
            sigma=params['sigma'],
            learning_rate=params['learning_rate'],
            neighborhood_function=params['neighborhood_function'],
            random_seed=params['random_seed']
        )
        logger.info("Initialized Manhattan distance SOM (L1 norm) - optimized for species data")
        
        # Memory-aware training
        self._update_progress(3, 5, "Training SOM")
        
        # Extract coordinates if available in metadata
        coordinates = kwargs.get('coordinates', None)
        if coordinates is None and 'coordinates' in metadata:
            coordinates = metadata['coordinates']
        
        # Use memory-aware training
        trained_som, sampling_info = self.train_with_subsampling(
            prepared_data, som, params, coordinates
        )
        
        # Memory-aware cluster assignment
        self._update_progress(4, 5, "Assigning clusters")
        
        if sampling_info['used_subsampling'] and n_samples > self.max_pixels_in_memory:
            # Use batch classification for large datasets
            labels = self.classify_in_batches(prepared_data, trained_som)
        else:
            # Standard classification for smaller datasets
            labels = np.zeros(n_samples, dtype=int)
            for i, sample in enumerate(prepared_data):
                winner = trained_som.winner(sample)
                labels[i] = winner[0] * params['grid_size'][1] + winner[1]
        
        # Calculate statistics
        self._update_progress(5, 5, "Calculating statistics")
        statistics = self._calculate_statistics(prepared_data, labels, trained_som, params)
        
        # Add sampling information to statistics
        statistics.update({
            'sampling_info': sampling_info,
            'memory_usage': self.estimate_memory_requirements(prepared_data.shape, prepared_data.dtype)
        })
        
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
            'som_weights': trained_som.get_weights(),
            'distance_map': trained_som.distance_map(),
            # Only calculate these on subsampled data to avoid memory issues
            'quantization_error': self._safe_quantization_error(trained_som, prepared_data, sampling_info),
            'topographic_error': self._safe_topographic_error(trained_som, prepared_data, sampling_info),
            'activation_map': self._get_activation_map(trained_som, prepared_data),
            'component_planes': self._get_component_planes(trained_som, metadata)
        }
        
        result = AnalysisResult(
            labels=labels,
            metadata=analysis_metadata,
            statistics=statistics,
            spatial_output=spatial_output,
            additional_outputs=additional_outputs
        )
        
        # Store in database if configured
        if self.save_results_enabled:
            self.store_in_database(result)
            # Also save to disk
            self.save_results(result)
        
        return result
    
    def _calculate_statistics(self, data: np.ndarray, labels: np.ndarray, 
                            som: Any, params: Dict[str, Any]) -> Dict[str, Any]:
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
        
        # Global statistics - use safe methods to avoid memory issues
        return {
            'n_clusters': n_clusters,
            'grid_size': params['grid_size'],
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
    
    def _train_with_spatial_validation(self, 
                                      data: np.ndarray,
                                      som,
                                      params: Dict[str, Any],
                                      coordinates: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Train SOM with spatial validation framework."""
        logger.info("Starting SOM training with spatial validation framework")
        
        # Create spatial data splitter
        splitter = create_spatial_splitter(
            strategy=params['spatial_split_strategy'],
            train_ratio=self.train_ratio,
            validation_ratio=self.validation_ratio,
            test_ratio=self.test_ratio
        )
        
        # Split data spatially
        data_split = splitter.split_data(data, coordinates)
        
        # Create SOM validator
        validator = create_som_validator(
            patience=self.validation_patience,
            min_improvement=self.validation_min_improvement,
            validation_interval=10,
            max_epochs=params.get('iterations', 1000)
        )
        
        # Train with validation-based early stopping using advanced convergence
        training_results = validator.train_with_validation(
            som=som, 
            data_split=data_split,
            convergence_method=params.get('convergence_method', 'unified'),
            use_batch_training=(params.get('convergence_method') == 'batch_unified'),
            progress_callback=self.progress_callback
        )
        
        # Return training info with all details
        training_info = {
            'converged': training_results['converged'],
            'best_epoch': training_results['best_epoch'],
            'total_epochs': training_results['total_epochs'],
            'training_time': training_results['training_time'],
            'best_validation_qe': training_results['best_validation_qe'],
            'overfitting_detected': training_results['overfitting_detected'],
            'data_split': data_split,
            'validation_history': training_results['validation_history'],
            'spatial_split_metadata': data_split.split_metadata
        }
        
        logger.info(f"Spatial validation training completed: "
                   f"converged={training_results['converged']}, "
                   f"epochs={training_results['total_epochs']}, "
                   f"overfitting={training_results['overfitting_detected']}")
        
        return som, training_info
    
    def _perform_final_evaluation(self, 
                                 som,
                                 data_split: DataSplit,
                                 full_data: np.ndarray,
                                 coordinates: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform comprehensive biodiversity evaluation on test set."""
        logger.info("Performing final biodiversity evaluation on test set")
        
        # Create biodiversity evaluator
        evaluator = create_biodiversity_evaluator(
            coordinates=coordinates,
            species_data=full_data  # Use full data as proxy for species data
        )
        
        # Evaluate on test set
        test_metrics = evaluator.evaluate_som(
            som, data_split.test_data, data_split.test_coords
        )
        
        # Also evaluate on validation set for comparison
        validation_metrics = evaluator.evaluate_som(
            som, data_split.validation_data, data_split.validation_coords
        )
        
        evaluation_results = {
            'test_metrics': test_metrics,
            'validation_metrics': validation_metrics,
            'test_sample_count': len(data_split.test_data),
            'validation_sample_count': len(data_split.validation_data),
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("Final biodiversity evaluation completed")
        logger.info(f"Test set performance:")
        logger.info(f"  Quantization Error: {test_metrics.quantization_error:.4f}")
        logger.info(f"  Biogeographic Coherence: {test_metrics.biogeographic_coherence:.3f}")
        logger.info(f"  Species Association Accuracy: {test_metrics.species_association_accuracy:.3f}")
        
        return evaluation_results