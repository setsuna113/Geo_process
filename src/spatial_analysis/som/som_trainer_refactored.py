# src/spatial_analysis/som/som_trainer_refactored.py
"""Refactored Self-Organizing Map trainer using proper abstractions."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import xarray as xr
import tempfile
import os
import gc
from minisom import MiniSom

from src.spatial_analysis.base_analyzer import BaseAnalyzer, AnalysisResult, AnalysisMetadata
from src.processors.spatial_analysis.data_processor import SpatialDataProcessor
from src.database.analysis_store import AnalysisStore
from src.processors.spatial_analysis.result_store import ResultStore
from src.database.connection import DatabaseManager
from src.spatial_analysis.memory_aware_processor import (
    SubsamplingStrategy, 
    MemoryAwareProcessor, 
    check_memory_usage
)

logger = logging.getLogger(__name__)

class SOMAnalyzer(BaseAnalyzer):
    """
    Self-Organizing Map analyzer with proper abstraction.
    
    Uses:
    - SpatialDataProcessor for data preparation
    - AnalysisStore for database operations  
    - ResultStore for file I/O
    """
    
    def __init__(self, config: Dict[str, Any], 
                 db_manager: Optional[DatabaseManager] = None,
                 output_dir: Optional[str] = None):
        super().__init__(config)
        
        if MiniSom is None:
            raise ImportError("minisom is required for SOM analysis. Install with: pip install minisom")
        
        # Initialize components using proper abstractions
        self.data_processor = SpatialDataProcessor(config)
        
        self.db_store = AnalysisStore(db_manager) if db_manager else None
        
        output_path = output_dir or config.get('spatial_analysis', {}).get('output_dir', 'outputs/spatial_analysis')
        self.file_store = ResultStore(output_path)
        
        # SOM-specific config
        som_config = config.get('spatial_analysis', {}).get('som', {})
        self.default_grid_size = som_config.get('grid_size', [10, 10])
        self.default_iterations = som_config.get('iterations', 1000)
        self.default_sigma = som_config.get('sigma', 1.0)
        self.default_learning_rate = som_config.get('learning_rate', 0.5)
        self.default_neighborhood = som_config.get('neighborhood_function', 'gaussian')
        
        # Memory-aware processing config
        self.subsampling_config = config.get('processing', {}).get('subsampling', {})
        self.som_analysis_config = config.get('som_analysis', {})
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
    
    def estimate_memory_requirements(self, data_shape: Tuple[int, ...], 
                                   dtype: np.dtype = np.dtype(np.float64)) -> Dict[str, Any]:
        """Estimate memory requirements for SOM analysis."""
        memory_info = check_memory_usage(data_shape, dtype)
        
        # Add SOM-specific overhead
        som_overhead_gb = float(memory_info['data_size_gb'] * self.memory_overhead_factor)
        memory_info['som_overhead_gb'] = som_overhead_gb
        memory_info['total_required_gb'] = som_overhead_gb
        memory_info['fits_in_memory'] = bool(som_overhead_gb < memory_info['available_gb'] * 0.6)
        
        return memory_info
    
    def analyze(self, 
                data: Union[xr.Dataset, xr.DataArray, np.ndarray],
                grid_size: Optional[List[int]] = None,
                iterations: Optional[int] = None,
                sigma: Optional[float] = None,
                learning_rate: Optional[float] = None,
                neighborhood_function: Optional[str] = None,
                random_seed: Optional[int] = None,
                experiment_name: Optional[str] = None,
                **kwargs) -> AnalysisResult:
        """
        Perform SOM analysis using proper abstractions.
        
        Args:
            data: Input data (P, A, F values)
            grid_size: SOM grid dimensions [rows, cols]
            iterations: Number of training iterations
            sigma: Spread of neighborhood function
            learning_rate: Initial learning rate
            neighborhood_function: Type of neighborhood function
            random_seed: Random seed for reproducibility
            experiment_name: Name for storing results
            **kwargs: Additional parameters
            
        Returns:
            AnalysisResult with cluster assignments and SOM properties
        """
        logger.info("Starting SOM analysis with refactored architecture")
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
        
        # Validate input using data processor
        valid, issues = self.data_processor.validate_input_data(data)
        if not valid:
            raise ValueError(f"Invalid input data: {'; '.join(issues)}")
        
        # Prepare data using data processor
        self.update_progress(1, 5, "Preparing data")
        prepared_data, metadata = self.data_processor.prepare_data(
            data, flatten=True, memory_aware=True
        )
        
        # Ensure 2D shape (n_samples, n_features)
        if prepared_data.ndim == 1:
            prepared_data = prepared_data.reshape(-1, 1)
        
        n_samples, n_features = prepared_data.shape
        logger.info(f"Training SOM on {n_samples} samples with {n_features} features")
        
        # Initialize SOM
        self.update_progress(2, 5, "Initializing SOM")
        som = MiniSom(
            x=params['grid_size'][0],
            y=params['grid_size'][1],
            input_len=n_features,
            sigma=params['sigma'],
            learning_rate=params['learning_rate'],
            neighborhood_function=params['neighborhood_function'],
            random_seed=params['random_seed']
        )
        
        # Memory-aware training
        self.update_progress(3, 5, "Training SOM")
        
        # Extract coordinates if available in metadata
        coordinates = kwargs.get('coordinates', None)
        if coordinates is None and 'coordinates' in metadata:
            coordinates = metadata['coordinates']
        
        # Use memory-aware training
        trained_som, sampling_info = self._train_with_subsampling(
            prepared_data, som, params, coordinates
        )
        
        # Memory-aware cluster assignment
        self.update_progress(4, 5, "Assigning clusters")
        
        if sampling_info['used_subsampling'] and n_samples > self.max_pixels_in_memory:
            # Use batch classification for large datasets
            labels = self._classify_in_batches(prepared_data, trained_som)
        else:
            # Standard classification for smaller datasets
            labels = np.zeros(n_samples, dtype=int)
            for i, sample in enumerate(prepared_data):
                winner = trained_som.winner(sample)
                labels[i] = winner[0] * params['grid_size'][1] + winner[1]
        
        # Calculate statistics
        self.update_progress(5, 5, "Calculating statistics")
        statistics = self._calculate_statistics(prepared_data, labels, trained_som, params, sampling_info)
        
        # Restore spatial structure using data processor
        spatial_output = self.data_processor.restore_spatial_structure(labels, metadata)
        
        # Create metadata
        analysis_metadata = AnalysisMetadata(
            analysis_type=self.analysis_type,
            input_shape=metadata['original_shape'],
            input_bands=metadata.get('bands', ['unknown']),
            parameters=params,
            processing_time=time.time() - start_time,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            normalization_applied=metadata.get('normalized', False)
        )
        
        # Store additional outputs
        additional_outputs = {
            'som_weights': trained_som.get_weights(),
            'distance_map': trained_som.distance_map(),
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
        
        # Store results using proper abstractions
        if experiment_name:
            # Save to files
            self.file_store.save_results(result, experiment_name)
            
            # Store in database if available
            if self.db_store:
                try:
                    self.db_store.store_result(result, experiment_name)
                except Exception as e:
                    logger.warning(f"Could not store results in database: {e}")
        
        return result
    
    def _train_with_subsampling(self, 
                              data: np.ndarray, 
                              som: Any,
                              params: Dict[str, Any],
                              coordinates: Optional[np.ndarray] = None) -> Tuple[Any, Dict[str, Any]]:
        """Train SOM with automatic subsampling for large datasets."""
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
            
            # Create spatial coordinates if not provided
            if coordinates is None and data.ndim >= 2:
                logger.info("Generating synthetic spatial coordinates for stratified sampling")
                n_rows = int(np.sqrt(n_samples))
                n_cols = int(np.ceil(n_samples / n_rows))
                row_indices = np.repeat(np.arange(n_rows), n_cols)[:n_samples]
                col_indices = np.tile(np.arange(n_cols), n_rows)[:n_samples]
                coordinates = np.column_stack([col_indices, row_indices])
            
            # Subsample for training
            train_data, sample_indices = self.subsampler.subsample_data(data, coordinates)
            
            # Train on subsample
            logger.info(f"Training SOM on {train_data.shape[0]:,} samples (subsampled from {n_samples:,})")
            som.train_random(train_data, params['iterations'])
            
            # Store sampling info
            sampling_info = {
                'used_subsampling': True,
                'sample_indices': sample_indices,
                'total_samples': int(n_samples),
                'train_samples': int(train_data.shape[0]),
                'sampling_ratio': float(train_data.shape[0] / n_samples)
            }
            
            # Force garbage collection
            gc.collect()
            
        else:
            # Train on full dataset
            logger.info(f"Training SOM on full dataset ({n_samples:,} samples)")
            som.train_random(data, params['iterations'])
            sampling_info = {
                'used_subsampling': False,
                'total_samples': int(n_samples),
                'train_samples': int(n_samples),
                'sampling_ratio': float(1.0)
            }
        
        return som, sampling_info
    
    def _classify_in_batches(self, data: np.ndarray, som: Any, 
                           batch_size: Optional[int] = None) -> np.ndarray:
        """Classify large dataset in batches to avoid memory issues."""
        n_samples = data.shape[0]
        
        if batch_size is None:
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
                labels[start_idx + i] = winner[0] * som._weights.shape[1] + winner[1]
            
            # Throttled progress logging
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
    
    def _calculate_statistics(self, data: np.ndarray, labels: np.ndarray, 
                            som: Any, params: Dict[str, Any], 
                            sampling_info: Dict[str, Any]) -> Dict[str, Any]:
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
            'cluster_statistics': cluster_stats,
            'empty_neurons': self._count_empty_neurons(labels, params['grid_size']),
            'cluster_balance': self._calculate_cluster_balance(labels),
            'sampling_info': sampling_info,
            'memory_usage': self.estimate_memory_requirements(data.shape, data.dtype)
        }
    
    def _safe_quantization_error(self, som: Any, data: np.ndarray, sampling_info: Dict[str, Any]) -> float:
        """Calculate quantization error safely for large datasets."""
        if sampling_info['used_subsampling']:
            sample_size = min(10000, data.shape[0])
            indices = np.random.choice(data.shape[0], sample_size, replace=False)
            sample_data = data[indices]
            error = som.quantization_error(sample_data)
            return float(error)
        else:
            error = som.quantization_error(data)
            return float(error)
    
    def _safe_topographic_error(self, som: Any, data: np.ndarray, sampling_info: Dict[str, Any]) -> float:
        """Calculate topographic error safely for large datasets."""
        if sampling_info['used_subsampling']:
            sample_size = min(10000, data.shape[0])
            indices = np.random.choice(data.shape[0], sample_size, replace=False)
            sample_data = data[indices]
            error = som.topographic_error(sample_data)
            return float(error)
        else:
            error = som.topographic_error(data)
            return float(error)
    
    def _get_activation_map(self, som: Any, data: np.ndarray) -> np.ndarray:
        """Get activation frequency for each neuron."""
        activation_map = np.zeros(som.get_weights().shape[:2])
        
        # Sample for large datasets
        if data.shape[0] > 50000:
            indices = np.random.choice(data.shape[0], 50000, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
            
        for sample in sample_data:
            winner = som.winner(sample)
            activation_map[winner] += 1
        
        return activation_map
    
    def _get_component_planes(self, som: Any, metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract component planes for each feature."""
        weights = som.get_weights()
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
        return float(np.std(counts) / np.mean(counts))