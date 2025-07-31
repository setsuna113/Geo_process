"""
Biodiversity SOM Analyzer - Complete VLRSOM System

Integrates:
1. Simple VLRSOM training (following real research)
2. Spatial validation (handles autocorrelation)
3. Manhattan distance SOM (optimal for species data)
4. Parquet input support
5. Clean analysis pipeline
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

from src.base.analyzer import BaseAnalyzer
from src.abstractions.interfaces.analyzer import AnalysisResult, AnalysisMetadata
from .manhattan_som import ManhattanSOMWrapper
from .simple_vlrsom import create_simple_vlrsom, VLRSOMResult
from .spatial_validation import create_spatial_splitter, SpatialDataSplit

logger = logging.getLogger(__name__)


class BiodiversitySOMAnalyzer(BaseAnalyzer):
    """
    Complete SOM analyzer for biodiversity data following real VLRSOM research.
    
    Features:
    - Simple VLRSOM training (real research pattern)
    - Spatial validation (prevents autocorrelation leakage)
    - Manhattan distance (optimal for species data)
    - Parquet input support
    - Clean, understandable results
    """
    
    def __init__(self, config: Union[Dict[str, Any], Any], db_connection: Optional[Any] = None):
        """Initialize biodiversity SOM analyzer."""
        super().__init__(config, db_connection)
        
        # SOM configuration
        self.som_config = self.safe_get_config('som_analysis', {})
        self.default_grid_size = self.som_config.get('default_grid_size', [10, 10])
        self.default_sigma = self.som_config.get('sigma', 1.0)
        self.default_learning_rate = self.som_config.get('learning_rate', 0.5)
        self.max_iterations = self.som_config.get('iterations', 2000)
        
        # VLRSOM configuration
        self.vlrsom_config = self.som_config.get('vlrsom', {})
        self.qe_threshold = self.vlrsom_config.get('qe_threshold', 1e-6)
        self.te_threshold = self.vlrsom_config.get('te_threshold', 0.05)
        self.patience = self.vlrsom_config.get('patience', 50)
        
        # Spatial validation configuration
        self.validation_config = self.som_config.get('validation', {})
        self.enable_spatial_validation = self.validation_config.get('enabled', True)
        self.spatial_split_strategy = self.validation_config.get('spatial_split_strategy', 'latitudinal')
        self.train_ratio = self.validation_config.get('train_ratio', 0.7)
        self.validation_ratio = self.validation_config.get('validation_ratio', 0.15)
        self.test_ratio = self.validation_config.get('test_ratio', 0.15)
        
        logger.info("Initialized BiodiversitySOMAnalyzer")
        logger.info(f"  Grid size: {self.default_grid_size}")
        logger.info(f"  Max iterations: {self.max_iterations}")
        logger.info(f"  Spatial validation: {self.enable_spatial_validation}")
        logger.info(f"  Split strategy: {self.spatial_split_strategy}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default SOM parameters."""
        return {
            'grid_size': self.default_grid_size,
            'sigma': self.default_sigma,
            'learning_rate': self.default_learning_rate,
            'max_iterations': self.max_iterations,
            'qe_threshold': self.qe_threshold,
            'te_threshold': self.te_threshold,
            'patience': self.patience,
            'random_seed': 42,
            'enable_spatial_validation': self.enable_spatial_validation,
            'spatial_split_strategy': self.spatial_split_strategy
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
        
        # Check thresholds
        qe_threshold = parameters.get('qe_threshold', self.qe_threshold)
        if not isinstance(qe_threshold, (int, float)) or qe_threshold <= 0:
            issues.append("qe_threshold must be a positive number")
        
        te_threshold = parameters.get('te_threshold', self.te_threshold) 
        if not isinstance(te_threshold, (int, float)) or te_threshold <= 0 or te_threshold >= 1:
            issues.append("te_threshold must be between 0 and 1")
        
        # Check iterations
        max_iterations = parameters.get('max_iterations', self.max_iterations)
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            issues.append("max_iterations must be a positive integer")
        
        return len(issues) == 0, issues
    
    def load_parquet_data(self, parquet_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load biodiversity data from parquet file.
        
        Args:
            parquet_path: Path to parquet file
            
        Returns:
            Tuple of (features, coordinates, feature_names)
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        logger.info(f"Loading parquet data from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Extract coordinates (assume longitude, latitude columns exist)
        coord_cols = ['longitude', 'latitude']
        if not all(col in df.columns for col in coord_cols):
            # Try alternative names
            alt_coord_cols = [
                ['lon', 'lat'],
                ['x', 'y'],
                ['long', 'lat'],
                ['lng', 'lat']
            ]
            for alt_cols in alt_coord_cols:
                if all(col in df.columns for col in alt_cols):
                    coord_cols = alt_cols
                    break
            else:
                raise ValueError(f"Could not find coordinate columns in: {list(df.columns)}")
        
        coordinates = df[coord_cols].values
        logger.info(f"Found coordinates: {coord_cols}")
        
        # Extract feature columns (exclude coordinates and any ID columns)
        exclude_cols = set(coord_cols + ['id', 'cell_id', 'index'])
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No feature columns found after excluding coordinates")
        
        features = df[feature_cols].values
        logger.info(f"Using {len(feature_cols)} feature columns: {feature_cols[:5]}...")
        
        # Handle missing values with more appropriate strategy
        if np.any(np.isnan(features)):
            n_missing = np.sum(np.isnan(features))
            missing_percent = (n_missing / features.size) * 100
            logger.warning(f"Found {n_missing:,} missing values ({missing_percent:.2f}% of data)")
            
            # Use sophisticated imputation strategies appropriate for biodiversity metrics
            for col_idx in range(features.shape[1]):
                col_data = features[:, col_idx]
                nan_mask = np.isnan(col_data)
                if np.any(nan_mask):
                    col_name = feature_cols[col_idx] if col_idx < len(feature_cols) else f"feature_{col_idx}"
                    
                    if np.all(nan_mask):
                        # All values are NaN - this column should be excluded
                        logger.warning(f"Column '{col_name}' has all NaN values - excluding from analysis")
                        features = np.delete(features, col_idx, axis=1)
                        feature_cols.pop(col_idx) if col_idx < len(feature_cols) else None
                        continue
                    
                    # Choose imputation strategy based on column characteristics
                    valid_data = col_data[~nan_mask]
                    if 'richness' in col_name.lower() or 'count' in col_name.lower():
                        # For species counts/richness: use median (better than mean for counts)
                        fill_value = np.median(valid_data)
                    elif 'diversity' in col_name.lower() or 'shannon' in col_name.lower():
                        # For diversity indices: use mean (continuous measures)
                        fill_value = np.mean(valid_data)
                    elif np.all(np.isin(valid_data, [0, 1])):
                        # Binary presence/absence: use mode (most common value)
                        fill_value = np.bincount(valid_data.astype(int)).argmax()
                    else:
                        # Default: median for robustness
                        fill_value = np.median(valid_data)
                    
                    features[nan_mask, col_idx] = fill_value
                    logger.debug(f"Filled {np.sum(nan_mask)} NaN values in '{col_name}' with {fill_value:.4f}")
        
        return features, coordinates, feature_cols
    
    def _calculate_streaming_statistics(self, 
                                      features: np.ndarray,
                                      som,
                                      vlrsom_result,
                                      data_split,
                                      params: Dict[str, Any],
                                      coordinates: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate statistics using streaming approach to minimize memory usage."""
        logger.info("Calculating statistics using streaming approach")
        
        # Initialize accumulators for streaming calculations
        total_samples = len(features)
        chunk_size = 10000
        
        # Streaming statistics
        cluster_counts = {}
        total_quantization_error = 0.0
        total_topographic_error = 0.0
        processed_samples = 0
        
        # Process in chunks without storing all labels
        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            chunk_features = features[i:end_idx]
            chunk_labels = som.predict(chunk_features)
            
            # Update cluster counts
            unique, counts = np.unique(chunk_labels, return_counts=True)
            for cluster_id, count in zip(unique, counts):
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + count
            
            # Accumulate errors (streaming calculation)
            chunk_qe = som.quantization_error(chunk_features)
            chunk_te = som.topographic_error(chunk_features) if hasattr(som, 'topographic_error') else 0.0
            
            total_quantization_error += chunk_qe * len(chunk_features)
            total_topographic_error += chunk_te * len(chunk_features)
            processed_samples += len(chunk_features)
            
            # Log progress
            if i % (chunk_size * 10) == 0:
                logger.debug(f"Streaming statistics: processed {processed_samples:,}/{total_samples:,} samples")
        
        # Calculate final statistics
        n_clusters = len(cluster_counts)
        avg_quantization_error = total_quantization_error / processed_samples
        avg_topographic_error = total_topographic_error / processed_samples
        
        statistics = {
            'quantization_error': avg_quantization_error,
            'topographic_error': avg_topographic_error,
            'n_clusters_used': n_clusters,
            'cluster_distribution': cluster_counts,
            'converged': vlrsom_result.converged,
            'final_learning_rate': vlrsom_result.final_learning_rate,
            'total_iterations': vlrsom_result.total_iterations,
            'streaming_mode': True,  # Flag to indicate streaming was used
            'processed_samples': processed_samples
        }
        
        # Add spatial autocorrelation if coordinates available (but use sampling for memory efficiency)
        if coordinates is not None:
            # Sample subset for spatial autocorrelation calculation
            sample_size = min(50000, len(coordinates))  # Max 50k samples for spatial calc
            sample_indices = np.random.choice(len(coordinates), sample_size, replace=False)
            sample_coords = coordinates[sample_indices]
            sample_labels = som.predict(features[sample_indices])
            
            from .biodiversity_evaluation import calculate_spatial_autocorrelation
            spatial_autocorr = calculate_spatial_autocorrelation(sample_labels, sample_coords)
            statistics['spatial_autocorrelation'] = spatial_autocorr
            statistics['spatial_autocorr_sample_size'] = sample_size
        
        return statistics
    
    def analyze(self, 
                data: Union[str, Path, np.ndarray],
                coordinates: Optional[np.ndarray] = None,
                **kwargs) -> AnalysisResult:
        """
        Perform biodiversity SOM analysis.
        
        Args:
            data: Input data (parquet path or numpy array)
            coordinates: Spatial coordinates (if data is numpy array)
            **kwargs: Analysis parameters
            
        Returns:
            AnalysisResult with SOM clustering results
        """
        logger.info("Starting biodiversity SOM analysis")
        start_time = time.time()
        
        # Prepare parameters
        params = self.get_default_parameters()
        params.update(kwargs)
        
        # Validate parameters
        valid, issues = self.validate_parameters(params)
        if not valid:
            raise ValueError(f"Invalid parameters: {'; '.join(issues)}")
        
        # Load data
        self._update_progress(1, 6, "Loading data")
        if isinstance(data, (str, Path)):
            features, coordinates, feature_names = self.load_parquet_data(data)
        else:
            if coordinates is None:
                raise ValueError("Coordinates required when data is numpy array")
            features = data
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Validate input data
        valid, issues = self.validate_input_data(features)
        if not valid:
            raise ValueError(f"Invalid input data: {'; '.join(issues)}")
        
        n_samples, n_features = features.shape
        logger.info(f"Analyzing {n_samples:,} samples with {n_features} features")
        
        # Spatial data splitting
        self._update_progress(2, 6, "Spatial data splitting")
        if self.enable_spatial_validation and coordinates is not None:
            splitter = create_spatial_splitter(
                strategy=params['spatial_split_strategy'],
                train_ratio=self.train_ratio,
                validation_ratio=self.validation_ratio,
                test_ratio=self.test_ratio
            )
            data_split = splitter.split_data(features, coordinates)
            train_data = data_split.train_data
            val_data = data_split.validation_data
            test_data = data_split.test_data
            
            logger.info(f"Spatial split completed: {len(train_data):,}/{len(val_data):,}/{len(test_data):,}")
        else:
            # No spatial validation - use all data for training
            train_data = features
            val_data = None
            test_data = None
            data_split = None
            logger.info("Spatial validation disabled, using all data for training")
        
        # Initialize Manhattan SOM
        self._update_progress(3, 6, "Initializing Manhattan SOM")
        som = ManhattanSOMWrapper(
            x=params['grid_size'][0],
            y=params['grid_size'][1],
            input_len=n_features,
            sigma=params['sigma'],
            learning_rate=params['learning_rate'],
            random_seed=params['random_seed']
        )
        logger.info(f"Initialized {params['grid_size'][0]}x{params['grid_size'][1]} Manhattan SOM")
        
        # VLRSOM training
        self._update_progress(4, 6, "VLRSOM training")
        vlrsom_trainer = create_simple_vlrsom(
            som=som,
            initial_learning_rate=params['learning_rate'],
            qe_threshold=params['qe_threshold'],
            te_threshold=params['te_threshold'],
            max_iterations=params['max_iterations'],
            patience=params['patience']
        )
        
        vlrsom_result = vlrsom_trainer.train(
            train_data=train_data,
            validation_data=val_data,
            progress_callback=self._progress_callback
        )
        
        logger.info(f"VLRSOM training completed: converged={vlrsom_result.converged}")
        
        # Generate cluster labels with memory-conscious approach
        self._update_progress(5, 6, "Generating cluster labels")
        
        # For extremely large datasets (>1M samples), use streaming approach
        if n_samples > 1000000:
            logger.info(f"Using streaming approach for {n_samples:,} samples to minimize memory usage")
            
            # Calculate statistics directly without storing all labels
            statistics = self._calculate_streaming_statistics(
                features, som, vlrsom_result, data_split, params, coordinates
            )
            
        else:
            # For manageable datasets, use chunked approach
            if data_split is not None:
                if n_samples > 100000:
                    logger.info(f"Processing {n_samples:,} samples in chunks to optimize memory usage")
                    all_labels = np.zeros(n_samples, dtype=np.int32)
                    
                    # Process train data in chunks
                    chunk_size = 10000
                    for i in range(0, len(data_split.train_indices), chunk_size):
                        end_idx = min(i + chunk_size, len(data_split.train_indices))
                        indices_chunk = data_split.train_indices[i:end_idx]
                        data_chunk = data_split.train_data[i:end_idx]
                        all_labels[indices_chunk] = som.predict(data_chunk)
                    
                    # Process validation data in chunks
                    for i in range(0, len(data_split.validation_indices), chunk_size):
                        end_idx = min(i + chunk_size, len(data_split.validation_indices))
                        indices_chunk = data_split.validation_indices[i:end_idx]
                        data_chunk = data_split.validation_data[i:end_idx]
                        all_labels[indices_chunk] = som.predict(data_chunk)
                    
                    # Process test data in chunks
                    for i in range(0, len(data_split.test_indices), chunk_size):
                        end_idx = min(i + chunk_size, len(data_split.test_indices))
                        indices_chunk = data_split.test_indices[i:end_idx]
                        data_chunk = data_split.test_data[i:end_idx]
                        all_labels[indices_chunk] = som.predict(data_chunk)
                else:
                    # Original method for smaller datasets
                    all_labels = np.zeros(n_samples, dtype=np.int32)
                    all_labels[data_split.train_indices] = som.predict(data_split.train_data)
                    all_labels[data_split.validation_indices] = som.predict(data_split.validation_data)
                    all_labels[data_split.test_indices] = som.predict(data_split.test_data)
            else:
                # Process features in chunks if dataset is large
                if len(features) > 100000:
                    logger.info(f"Processing {len(features):,} features in chunks to optimize memory usage")
                    all_labels = np.zeros(len(features), dtype=np.int32)
                    chunk_size = 10000
                    for i in range(0, len(features), chunk_size):
                        end_idx = min(i + chunk_size, len(features))
                        chunk_labels = som.predict(features[i:end_idx])
                        all_labels[i:end_idx] = chunk_labels
                else:
                    all_labels = som.predict(features)
            
            # Calculate statistics for non-streaming approach
            self._update_progress(6, 6, "Calculating statistics")
            statistics = self._calculate_statistics(
                features, all_labels, som, vlrsom_result, data_split, params
            )
        
        # Create metadata
        processing_time = time.time() - start_time
        metadata = self.create_metadata(
            data_shape=features.shape,
            parameters=params,
            processing_time=processing_time,
            data_source=str(data) if isinstance(data, (str, Path)) else "numpy_array",
            normalization_applied=False
        )
        
        # Create result
        result = AnalysisResult(
            labels=all_labels,
            metadata=metadata,
            statistics=statistics,
            spatial_output=None,  # Could add spatial output later
            additional_outputs={
                'som_weights': som.get_weights(),
                'vlrsom_result': vlrsom_result,
                'data_split': data_split,
                'feature_names': feature_names,
                'coordinates': coordinates
            }
        )
        
        logger.info(f"SOM analysis completed in {processing_time:.2f}s")
        
        return result
    
    def _calculate_statistics(self,
                            features: np.ndarray,
                            labels: np.ndarray,
                            som,
                            vlrsom_result: VLRSOMResult,
                            data_split: Optional[SpatialDataSplit],
                            params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for SOM results."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Cluster statistics
        cluster_stats = {}
        for label in unique_labels:
            mask = labels == label
            cluster_data = features[mask]
            
            cluster_stats[int(label)] = {
                'count': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(labels) * 100),
                'mean': cluster_data.mean(axis=0).tolist(),
                'std': cluster_data.std(axis=0).tolist()
            }
        
        # VLRSOM training statistics
        training_stats = {
            'converged': vlrsom_result.converged,
            'total_iterations': vlrsom_result.total_iterations,
            'training_time': vlrsom_result.training_time,
            'final_quantization_error': vlrsom_result.final_quantization_error,
            'final_topographic_error': vlrsom_result.final_topographic_error,
            'learning_rate_final': vlrsom_result.learning_rate_history[-1] if vlrsom_result.learning_rate_history else None
        }
        
        # Spatial validation statistics
        spatial_stats = {}
        if data_split is not None:
            spatial_stats = {
                'split_strategy': data_split.split_strategy,
                'split_metadata': data_split.split_metadata,
                'train_qe': som.quantization_error(data_split.train_data),
                'validation_qe': som.quantization_error(data_split.validation_data),
                'test_qe': som.quantization_error(data_split.test_data)
            }
        
        return {
            'n_clusters': n_clusters,
            'grid_size': params['grid_size'],
            'cluster_statistics': cluster_stats,
            'training_statistics': training_stats,
            'spatial_validation': spatial_stats,
            'som_metrics': {
                'quantization_error': som.quantization_error(features),
                'topographic_error': som.topographic_error(features)
            }
        }


def create_biodiversity_som_analyzer(config: Union[Dict[str, Any], Any], **kwargs) -> BiodiversitySOMAnalyzer:
    """Factory function for creating biodiversity SOM analyzer."""
    return BiodiversitySOMAnalyzer(config, **kwargs)