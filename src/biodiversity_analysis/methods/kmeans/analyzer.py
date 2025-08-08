"""K-means analyzer for biodiversity data."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import json
from datetime import datetime

from src.biodiversity_analysis.shared.base_analyzer import BaseBiodiversityAnalyzer
from src.abstractions.interfaces.analyzer import AnalysisResult
from src.biodiversity_analysis.preprocessing.pipeline import SharedPreprocessor
from .core_simplified import BiodiversityKMeans
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
        
        # Validate and sanitize input path
        import os
        data_path = os.path.abspath(data_path)
        
        # Security check: prevent directory traversal
        if ".." in data_path or data_path.count("/") > 50:
            raise ValueError("Invalid file path")
        
        # Validate file exists and is readable
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if not os.access(data_path, os.R_OK):
            raise PermissionError(f"Cannot read data file: {data_path}")
        
        # Validate file extension
        valid_extensions = ['.parquet', '.csv']
        if not any(data_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported file format. Supported: {valid_extensions}")
        
        # Validate parameters
        self._validate_analysis_parameters(kwargs)
        
        # Load biodiversity data
        logger.info(f"Loading biodiversity data from {data_path}")
        biodiv_data = self.load_data(data_path)
        
        # Log data statistics
        logger.info(f"Loaded data: {biodiv_data.n_samples} samples, "
                   f"{biodiv_data.n_features} features")
        
        # Preprocess features using shared preprocessor for consistency
        logger.info("Preprocessing features...")
        
        # Use SharedPreprocessor with robust settings
        preprocessor = SharedPreprocessor(
            method_name='kmeans',
            normalize=True,
            normalization_method='robust',
            handle_missing=True,
            missing_strategy='mean',
            remove_constant_features=True
        )
        
        # Transform the data
        features_processed = preprocessor.fit_transform(biodiv_data.features)
        
        # Update feature names if any were removed
        feature_names = biodiv_data.feature_names
        if hasattr(preprocessor, 'selected_features_'):
            feature_names = [feature_names[i] for i in preprocessor.selected_features_]
        
        logger.info(f"Preprocessing complete: {features_processed.shape} "
                   f"(original: {biodiv_data.features.shape})")
        
        # Update biodiversity data with preprocessed features
        biodiv_data.features = features_processed
        biodiv_data.feature_names = feature_names
        
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
        """Determine optimal number of clusters using silhouette score."""
        # Convert to list if needed
        if hasattr(k_range, 'start'):
            k_values = list(k_range)
            logger.info(f"Testing k values in range {k_range.start}-{k_range.stop-1}")
        else:
            k_values = list(k_range)
            logger.info(f"Testing k values: {k_values}")
        
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
        
        return metrics
    
    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray,
                                   sample_size: Optional[int] = None) -> float:
        """Calculate silhouette score with partial distance support."""
        from sklearn.metrics import silhouette_score
        
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
        
        # Simplified implementation - always use Euclidean distance
        # The complex distance calculations are removed in the simplified version
        return silhouette_score(features_valid, labels_valid, metric='euclidean')
    
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
    
    def _validate_analysis_parameters(self, params: Dict[str, Any]) -> None:
        """Validate analysis parameters for security and correctness.
        
        Args:
            params: Parameters dictionary
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate n_clusters
        if 'n_clusters' in params:
            n_clusters = params['n_clusters']
            if not isinstance(n_clusters, int):
                raise TypeError(f"n_clusters must be int, got {type(n_clusters)}")
            if n_clusters < 2 or n_clusters > 1000:
                raise ValueError(f"n_clusters must be between 2 and 1000, got {n_clusters}")
        
        # Validate k_range
        if 'k_range' in params:
            k_range = params['k_range']
            if not hasattr(k_range, '__iter__'):
                raise TypeError("k_range must be iterable")
            k_list = list(k_range)
            if len(k_list) < 2:
                raise ValueError("k_range must contain at least 2 values")
            if not all(isinstance(k, int) and 2 <= k <= 100 for k in k_list):
                raise ValueError("k_range values must be integers between 2 and 100")
        
        # Validate output_dir if save_results is True
        if params.get('save_results', False):
            output_dir = params.get('output_dir', './outputs/kmeans_results')
            
            # Security: prevent directory traversal
            if '..' in str(output_dir):
                raise ValueError("Invalid output directory path")
            
            # Convert to Path for validation
            from pathlib import Path
            output_path = Path(output_dir)
            
            # Check if parent directory exists
            if not output_path.parent.exists():
                raise ValueError(f"Parent directory does not exist: {output_path.parent}")
    
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
            'missing_data_by_cluster': quality_metrics.get('missing_data_by_cluster', {})
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
        # Sanitize output path
        output_path = Path(output_dir).resolve()
        
        # Security check
        if '..' in str(output_path):
            raise ValueError("Invalid output directory")
        
        # Create directory safely
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise IOError(f"Cannot create output directory: {e}")
        
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