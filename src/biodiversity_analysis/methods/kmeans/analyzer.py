"""K-means analyzer for biodiversity data."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import json
from datetime import datetime

from src.biodiversity_analysis.base_analyzer import BaseBiodiversityAnalyzer
from src.abstractions.interfaces.analyzer import AnalysisResult
from .core import BiodiversityKMeans
from .kmeans_config import KMeansConfig
from .visualization import KMeansVisualizer

logger = logging.getLogger(__name__)


class KMeansAnalyzer(BaseBiodiversityAnalyzer):
    """K-means clustering analyzer for biodiversity patterns.
    
    This analyzer implements optimized k-means clustering for biodiversity data
    with high missing values and fine spatial resolution.
    """
    
    def __init__(self):
        """Initialize k-means analyzer."""
        super().__init__(method_name='kmeans', version='1.0.0')
        self.kmeans = None
        self.visualizer = KMeansVisualizer()
        
    def analyze(self, data_path: str, **kwargs) -> AnalysisResult:
        """Perform k-means clustering analysis on biodiversity data.
        
        Args:
            data_path: Path to parquet file with biodiversity data
            **kwargs: Additional parameters including:
                - n_clusters: Number of clusters (default from config)
                - determine_k: Whether to automatically determine optimal k
                - k_range: Range of k values to test if determine_k=True
                - save_results: Whether to save results
                - output_dir: Directory for saving results
                
        Returns:
            AnalysisResult: Analysis results with cluster assignments
        """
        start_time = datetime.now()
        
        # Load biodiversity data
        logger.info(f"Loading biodiversity data from {data_path}")
        biodiv_data = self.load_data(data_path)
        
        # Log data statistics
        logger.info(f"Loaded data: {biodiv_data.n_samples} samples, "
                   f"{biodiv_data.n_features} features")
        
        # Preprocess features
        logger.info("Preprocessing features...")
        # Apply standard preprocessing from base class
        biodiv_data = self.preprocess_data(biodiv_data)
        features_processed = biodiv_data.features
        
        # Create k-means configuration
        config = self._create_kmeans_config(**kwargs)
        
        # Determine optimal k if requested
        if kwargs.get('determine_k', False):
            optimal_k = self._determine_optimal_k(
                features_processed,
                biodiv_data.coordinates,
                config,
                kwargs.get('k_range', range(5, 31))
            )
            config.n_clusters = optimal_k
            logger.info(f"Determined optimal k: {optimal_k}")
        
        # Initialize and fit k-means
        logger.info(f"Fitting k-means with {config.n_clusters} clusters...")
        self.kmeans = BiodiversityKMeans(config)
        
        # Add progress callback if available
        if hasattr(self, 'progress_callback') and self.progress_callback:
            # Create a simple progress wrapper
            def progress_wrapper(msg):
                self.progress_callback(f"K-means clustering: {msg}", None)
            
            # Fit with progress updates
            progress_wrapper("Calculating feature weights...")
            self.kmeans.fit(features_processed, biodiv_data.coordinates)
            progress_wrapper("Clustering complete")
        else:
            self.kmeans.fit(features_processed, biodiv_data.coordinates)
        
        # Calculate quality metrics
        logger.info("Calculating quality metrics...")
        quality_metrics = self._calculate_quality_metrics(
            features_processed,
            self.kmeans.labels_,
            biodiv_data.coordinates
        )
        
        # Create additional outputs
        additional_outputs = self._create_additional_outputs(
            biodiv_data,
            features_processed,
            quality_metrics
        )
        
        # Create result
        result = self._create_result(
            biodiv_data,
            features_processed,
            quality_metrics,
            additional_outputs,
            start_time
        )
        
        # Save results if requested
        if kwargs.get('save_results', False):
            output_dir = kwargs.get('output_dir', './outputs/kmeans_results')
            self._save_results(result, output_dir)
        
        return result
    
    def _create_kmeans_config(self, **kwargs) -> KMeansConfig:
        """Create k-means configuration from kwargs and defaults."""
        # Get method-specific config from base class config
        kmeans_defaults = self.method_params
        
        # Override with kwargs
        config_dict = {**kmeans_defaults, **kwargs}
        
        # Remove non-config parameters
        for key in ['determine_k', 'k_range', 'save_results', 'output_dir']:
            config_dict.pop(key, None)
        
        # Create config object
        return KMeansConfig(**config_dict)
    
    def _determine_optimal_k(self, features: np.ndarray, coordinates: np.ndarray,
                           base_config: KMeansConfig, k_range) -> int:
        """Determine optimal number of clusters."""
        # Convert to list if needed
        if hasattr(k_range, 'start'):
            k_values = list(k_range)
            logger.info(f"Testing k values in range {k_range.start}-{k_range.stop-1}")
        else:
            k_values = list(k_range)
            logger.info(f"Testing k values: {k_values}")
        
        if base_config.determine_k_method == 'silhouette':
            scores = []
            
            for k in k_values:
                logger.debug(f"Testing k={k}")
                config = KMeansConfig(**{**base_config.__dict__, 'n_clusters': k})
                km = BiodiversityKMeans(config)
                km.fit(features, coordinates)
                
                # Calculate silhouette score
                score = self._calculate_silhouette_score(features, km.labels_, sample_size=1000)
                scores.append(score)
                logger.debug(f"k={k}, silhouette={score:.3f}")
            
            # Find best k
            best_idx = np.argmax(scores)
            best_k = k_values[best_idx]
            logger.info(f"Best k={best_k} with silhouette score={scores[best_idx]:.3f}")
            
            return best_k
            
        elif base_config.determine_k_method == 'elbow':
            inertias = []
            
            for k in k_values:
                logger.debug(f"Testing k={k}")
                config = KMeansConfig(**{**base_config.__dict__, 'n_clusters': k})
                km = BiodiversityKMeans(config)
                km.fit(features, coordinates)
                inertias.append(km.inertia_)
            
            # Find elbow point
            best_k = self._find_elbow_point(k_values, inertias)
            logger.info(f"Elbow point at k={best_k}")
            
            return best_k
        
        else:
            # Default to middle of range
            return k_values[len(k_values) // 2]
    
    def _calculate_quality_metrics(self, features: np.ndarray, labels: np.ndarray,
                                 coordinates: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics."""
        metrics = {}
        
        # Basic statistics
        unique_labels = np.unique(labels[labels >= 0])
        n_clusters = len(unique_labels)
        
        # Cluster sizes
        cluster_sizes = {}
        for k in unique_labels:
            cluster_sizes[int(k)] = int((labels == k).sum())
        metrics['cluster_sizes'] = cluster_sizes
        metrics['n_clusters'] = n_clusters
        
        # Calculate silhouette score if not too large
        n_samples = len(features)
        if n_samples < 10000:
            score = self._calculate_silhouette_score(features, labels)
            metrics['silhouette_score'] = float(score)
        else:
            # Use sampling for large datasets
            score = self._calculate_silhouette_score(
                features, labels, sample_size=self.kmeans.config.silhouette_sample_size
            )
            metrics['silhouette_score_sampled'] = float(score)
        
        # Missing data statistics by cluster
        missing_by_cluster = {}
        for k in unique_labels:
            mask = labels == k
            if mask.sum() > 0:
                cluster_missing = np.isnan(features[mask]).mean()
                missing_by_cluster[int(k)] = float(cluster_missing)
        metrics['missing_data_by_cluster'] = missing_by_cluster
        
        # Geographic coherence (skip for now - could be added later)
        # if coordinates is not None:
        #     coherence = self._calculate_geographic_coherence(coordinates, labels)
        #     metrics['geographic_coherence'] = float(coherence)
        
        # Feature weights used
        if self.kmeans.feature_weights_ is not None:
            metrics['feature_weights'] = self.kmeans.feature_weights_.tolist()
        
        return metrics
    
    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray,
                                   sample_size: Optional[int] = None) -> float:
        """Calculate silhouette score with partial distance support."""
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import pairwise_distances
        
        # Handle sampling for large datasets
        if sample_size is not None and len(features) > sample_size:
            indices = np.random.choice(len(features), sample_size, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]
        else:
            features_sample = features
            labels_sample = labels
        
        # Filter out outliers
        valid_mask = labels_sample >= 0
        if valid_mask.sum() < 2:
            return 0.0
        
        features_valid = features_sample[valid_mask]
        labels_valid = labels_sample[valid_mask]
        
        # Custom distance matrix for partial distances
        if self.kmeans.config.distance_metric == 'bray_curtis':
            # Compute pairwise distances using our adaptive calculator
            n = len(features_valid)
            distances = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i+1, n):
                    d = self.kmeans.distance_calculator.calculate_distance(
                        features_valid[i], features_valid[j]
                    )
                    if np.isfinite(d) and d >= 0:
                        distances[i, j] = distances[j, i] = d
                    else:
                        distances[i, j] = distances[j, i] = 1.0  # Max distance for Bray-Curtis
            
            return silhouette_score(distances, labels_valid, metric='precomputed')
        else:
            # Use sklearn's implementation for Euclidean
            return silhouette_score(features_valid, labels_valid, metric='euclidean')
    
    def _calculate_geographic_coherence(self, coordinates: np.ndarray, 
                                      labels: np.ndarray) -> float:
        """Calculate geographic coherence of clusters."""
        from ..som.spatial_utils import calculate_morans_i, construct_spatial_weights_matrix
        
        # Create binary cluster membership matrices
        unique_labels = np.unique(labels[labels >= 0])
        coherence_scores = []
        
        for k in unique_labels:
            # Binary vector: 1 if in cluster k, 0 otherwise
            membership = (labels == k).astype(float)
            
            # Calculate Moran's I for this cluster
            weights_matrix = construct_spatial_weights_matrix(
                coordinates, 
                k_neighbors=min(10, len(coordinates) // 10)
            )
            
            moran_i = calculate_morans_i(membership, weights_matrix)
            coherence_scores.append(moran_i)
        
        # Return average coherence
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in inertia curve."""
        if len(k_values) < 3:
            return k_values[0]
        
        # Calculate curvature
        deltas = np.diff(inertias)
        delta_deltas = np.diff(deltas)
        
        # Find point with maximum curvature
        if len(delta_deltas) > 0:
            elbow_idx = np.argmax(np.abs(delta_deltas)) + 2
            if elbow_idx < len(k_values):
                return k_values[elbow_idx]
        
        # Fallback to middle
        return k_values[len(k_values) // 2]
    
    def _create_additional_outputs(self, biodiv_data, features_processed,
                                 quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create additional outputs for results."""
        outputs = {}
        
        # Cluster centers
        if self.kmeans.cluster_centers_ is not None:
            outputs['cluster_centers'] = self.kmeans.cluster_centers_.tolist()
        
        # Create visualizations
        try:
            # Feature profiles
            fig = self.visualizer.plot_cluster_profiles(
                features_processed, 
                self.kmeans.labels_,
                feature_names=biodiv_data.metadata.get('feature_names', None)
            )
            outputs['cluster_profiles_plot'] = fig
            
            # Geographic distribution
            fig = self.visualizer.plot_geographic_clusters(
                biodiv_data.coordinates,
                self.kmeans.labels_
            )
            outputs['geographic_clusters_plot'] = fig
            
            # Quality metrics
            fig = self.visualizer.plot_cluster_quality(quality_metrics)
            outputs['quality_metrics_plot'] = fig
            
            # Missing data patterns
            fig = self.visualizer.plot_missing_data_patterns(
                features_processed,
                self.kmeans.labels_
            )
            outputs['missing_patterns_plot'] = fig
            
        except Exception as e:
            logger.warning(f"Failed to create some visualizations: {e}")
        
        return outputs
    
    def _create_result(self, biodiv_data, features_processed, quality_metrics,
                      additional_outputs, start_time) -> AnalysisResult:
        """Create analysis result object."""
        from src.abstractions.interfaces.analyzer import AnalysisMetadata
        
        # Create metadata
        metadata = AnalysisMetadata(
            analysis_type='kmeans',
            input_shape=(int(biodiv_data.n_samples), int(biodiv_data.n_features)),
            input_bands=[str(name) for name in biodiv_data.feature_names] if biodiv_data.feature_names else [],
            parameters=self.kmeans.config.__dict__,
            processing_time=(datetime.now() - start_time).total_seconds(),
            timestamp=datetime.now().isoformat(),
            data_source=str(biodiv_data.metadata.get('source_file', 'unknown')) if biodiv_data.metadata else None,
            normalization_applied=self.kmeans.config.normalize is not None
        )
        
        # Create statistics
        statistics = {
            'n_clusters': int(self.kmeans.config.n_clusters),
            'cluster_sizes': quality_metrics.get('cluster_sizes', {}),
            'silhouette_score': quality_metrics.get('silhouette_score', 
                                quality_metrics.get('silhouette_score_sampled', None)),
            'missing_data_by_cluster': quality_metrics.get('missing_data_by_cluster', {}),
            'feature_weights': quality_metrics.get('feature_weights', None)
        }
        
        # Add coordinate information to additional outputs
        if additional_outputs is None:
            additional_outputs = {}
        
        additional_outputs['coordinates'] = biodiv_data.coordinates.tolist()
        additional_outputs['data_path'] = str(biodiv_data.metadata.get('source_file', 'unknown'))
        
        return AnalysisResult(
            labels=self.kmeans.labels_,
            metadata=metadata,
            statistics=statistics,
            additional_outputs=additional_outputs
        )
    
    def _save_results(self, result: AnalysisResult, output_dir: str):
        """Save analysis results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON
        results_file = output_path / 'kmeans_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_data = {
                'labels': result.labels.tolist() if isinstance(result.labels, np.ndarray) else result.labels,
                'metadata': {
                    'analysis_type': result.metadata.analysis_type,
                    'input_shape': result.metadata.input_shape,
                    'input_bands': result.metadata.input_bands,
                    'parameters': result.metadata.parameters,
                    'processing_time': result.metadata.processing_time,
                    'timestamp': result.metadata.timestamp,
                    'data_source': result.metadata.data_source,
                    'normalization_applied': result.metadata.normalization_applied
                },
                'statistics': result.statistics
            }
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        
        # Save visualizations
        for key, value in result.additional_outputs.items():
            if key.endswith('_plot') and hasattr(value, 'savefig'):
                plot_file = output_path / f'{key}.png'
                value.savefig(plot_file, dpi=150, bbox_inches='tight')
                logger.info(f"Saved {key} to {plot_file}")
        
        # Save cluster assignments as CSV for easy access
        if 'coordinates' in result.additional_outputs:
            coords = result.additional_outputs['coordinates']
            cluster_df = pd.DataFrame({
                'latitude': [coord[0] for coord in coords],
                'longitude': [coord[1] for coord in coords],
                'cluster': result.labels.tolist() if isinstance(result.labels, np.ndarray) else result.labels
            })
            csv_file = output_path / 'cluster_assignments.csv'
            cluster_df.to_csv(csv_file, index=False)
            logger.info(f"Saved cluster assignments to {csv_file}")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate analysis parameters.
        
        Args:
            parameters: Dictionary of parameters to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate n_clusters
        if 'n_clusters' in parameters:
            n_clusters = parameters['n_clusters']
            if not isinstance(n_clusters, int) or n_clusters < 2:
                errors.append("n_clusters must be an integer >= 2")
        
        # Validate k_range if determine_k is True
        if parameters.get('determine_k', False):
            k_range = parameters.get('k_range', None)
            if k_range is None:
                errors.append("k_range must be provided when determine_k=True")
            elif not hasattr(k_range, '__iter__'):
                errors.append("k_range must be iterable (list, range, etc.)")
        
        # Validate distance metric
        if 'distance_metric' in parameters:
            valid_metrics = ['bray_curtis', 'euclidean']
            if parameters['distance_metric'] not in valid_metrics:
                errors.append(f"distance_metric must be one of {valid_metrics}")
        
        # Validate weight method
        if 'weight_method' in parameters:
            valid_methods = ['auto', 'completeness', 'variance', 'fixed']
            if parameters['weight_method'] not in valid_methods:
                errors.append(f"weight_method must be one of {valid_methods}")
        
        return len(errors) == 0, errors
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for k-means analysis.
        
        Returns:
            Dictionary of default parameters
        """
        # Get from base class config
        kmeans_defaults = self.method_params
        
        # Add analysis-specific defaults
        defaults = {
            'n_clusters': 20,
            'determine_k': False,
            'k_range': range(5, 31),
            'save_results': True,
            'output_dir': './outputs/kmeans_results',
            **kmeans_defaults
        }
        
        return defaults
    
    def set_progress_callback(self, callback: Callable) -> None:
        """Set progress callback function.
        
        Args:
            callback: Function that accepts (message, progress)
        """
        self.progress_callback = callback
    
    def validate_input_data(self, data_path: str) -> Tuple[bool, List[str]]:
        """Validate input data file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check file exists
        path = Path(data_path)
        if not path.exists():
            errors.append(f"Input file does not exist: {data_path}")
            return False, errors
        
        # Check file format
        if not path.suffix.lower() in ['.parquet', '.csv']:
            errors.append(f"Unsupported file format: {path.suffix}. Use .parquet or .csv")
        
        # Try to load and validate structure
        try:
            if path.suffix.lower() == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            
            # Check required columns
            required_cols = ['latitude', 'longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
            
            # Check for at least some feature columns
            coord_cols = ['latitude', 'longitude', 'x', 'y']
            feature_cols = [col for col in df.columns if col not in coord_cols]
            if len(feature_cols) < 2:
                errors.append("Need at least 2 feature columns for clustering")
                
        except Exception as e:
            errors.append(f"Failed to read data file: {str(e)}")
        
        return len(errors) == 0, errors
    
    def save_results(self, result: AnalysisResult, output_name: str, 
                     output_dir: Path = None) -> Path:
        """Save analysis results to disk.
        
        Args:
            result: Analysis result to save
            output_name: Base name for output files
            output_dir: Directory to save results
            
        Returns:
            Path to saved results
        """
        if output_dir is None:
            output_dir = Path("./results/kmeans")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use internal save method
        self._save_results(result, str(output_dir))
        
        return output_dir