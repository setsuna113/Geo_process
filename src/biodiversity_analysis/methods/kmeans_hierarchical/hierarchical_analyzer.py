        from src.abstractions import AnalysisMetadata
        from src.abstractions import validate_parameters as validate_params
from src.abstractions import AnalysisResult
from src.abstractions import get_logger, validate_parameters
            from dataclasses import dataclass
from .hierarchical_config import KMeansHierarchicalConfig
from .hierarchical_core import (
from .hierarchical_visualization import HierarchicalVisualizer
from datetime import datetime
from pathlib import Path
from src.biodiversity_analysis.preprocessing.pipeline import SharedPreprocessor
from src.biodiversity_analysis.shared.base_analyzer import BaseBiodiversityAnalyzer
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np



    TwoStageClusterer,
    ElbowMethodKMeans,
    BootstrapStability,
    calculate_clustering_metrics
)

logger = get_logger(__name__)


class KMeansHierarchicalAnalyzer(BaseBiodiversityAnalyzer):
    """Two-stage clustering analyzer for large biodiversity datasets.
    
    Combines k-means pre-clustering with hierarchical clustering to handle
    millions of data points efficiently while preserving hierarchical structure.
    """
    
    def __init__(self):
        """Initialize two-stage analyzer."""
        super().__init__(method_name='kmeans_hierarchical', version='2.0.0')
        self.clusterer = None
        self.visualizer = HierarchicalVisualizer()
        
    def analyze(self, data_path: str, **kwargs) -> AnalysisResult:
        """Perform two-stage clustering analysis.
        
        Args:
            data_path: Path to biodiversity data (parquet format)
            **kwargs: Configuration parameters
            
        Returns:
            AnalysisResult with cluster assignments and visualizations
        """
        start_time = datetime.now()
        
        # Load and validate data with spatial grid sampling
        logger.info(f"Loading biodiversity data from {data_path}")
        
        # Get spatial sampling parameter from config or kwargs
        spatial_grid_step = kwargs.get('spatial_grid_step', self.data_config.get('spatial_grid_step', 1))
        
        if spatial_grid_step > 1:
            logger.info(f"Using systematic spatial sampling with grid step: {spatial_grid_step}")
            logger.info("This provides better geographic coverage than random sampling")
        else:
            logger.info("Using full dataset (no spatial sampling)")
        
        # Get selected features from config
        selected_features = kwargs.get('selected_features', self.data_config.get('selected_features', None))
        if selected_features:
            logger.info(f"Using selected features for clustering: {selected_features}")
        else:
            logger.info("Using all available features for clustering")
        
        biodiv_data = self.load_data(
            data_path, 
            spatial_grid_step=spatial_grid_step,
            feature_cols=selected_features
        )
        
        logger.info(f"Loaded {biodiv_data.n_samples:,} samples with "
                   f"{biodiv_data.n_features} features")
        
        # Store original data shape
        original_n_samples = biodiv_data.n_samples
        
        # Preprocess features (returns only complete cases)
        features_processed, feature_names = self._preprocess_features(
            biodiv_data.features,
            biodiv_data.coordinates,
            biodiv_data.feature_names
        )
        
        # Create configuration
        config = self._create_config(**kwargs)
        
        # Perform two-stage clustering on complete cases
        self.clusterer, labels_complete = self._perform_clustering(
            features_processed,
            config
        )
        
        # Map labels back to full dataset with memory monitoring
        # Check memory requirements
        memory_required_mb = (original_n_samples * 4) / (1024 * 1024)  # int32 array
        if memory_required_mb > 1000:  # Warn if > 1GB
            logger.warning(f"Allocating {memory_required_mb:.1f}MB for final labels array. "
                         f"Consider increasing spatial sampling for very large datasets.")
        
        final_labels = np.full(original_n_samples, -1, dtype=int)
        final_labels[self._complete_mask] = labels_complete
        
        # Calculate quality metrics on complete cases
        quality_metrics = calculate_clustering_metrics(
            features_processed,
            labels_complete,
            sample_size=config.silhouette_sample_size,
            method_name='kmeans_hierarchical'
        )
        
        # Add hierarchical-specific metrics
        quality_metrics['cophenetic_correlation'] = self.clusterer.cophenetic_correlation_
        quality_metrics['variance_preserved'] = self.clusterer.variance_preserved_
        
        # Generate visualizations if requested
        visualizations = {}
        if any([config.generate_dendrogram, config.generate_geographic_maps,
                config.generate_3d_visualization, config.generate_validation_plots]):
            # Create clean data object for visualization (complete cases only)
            @dataclass
            class BiodivDataComplete:
                features: np.ndarray
                coordinates: np.ndarray
                feature_names: list
                n_samples: int
                n_features: int
                metadata: dict
            
            biodiv_data_complete = BiodivDataComplete(
                features=features_processed,
                coordinates=biodiv_data.coordinates[self._complete_mask],
                feature_names=feature_names,
                n_samples=len(features_processed),
                n_features=features_processed.shape[1],
                metadata=biodiv_data.metadata
            )
            
            visualizations = self._generate_visualizations(
                biodiv_data_complete,
                features_processed,
                labels_complete,
                quality_metrics,
                config
            )
        
        # Create result with full labels
        return self._create_result(
            biodiv_data,
            final_labels,
            quality_metrics,
            visualizations,
            config,
            start_time
        )
    
    def _preprocess_features(self, features, coordinates, feature_names):
        """Preprocess features using shared preprocessor."""
        logger.info("Preprocessing features...")
        
        # Use SharedPreprocessor which has proper fit/transform separation
        preprocessor = SharedPreprocessor(
            method_name='kmeans_hierarchical',
            normalize=True,
            normalization_method='robust',
            handle_missing=True,
            missing_strategy='keep',  # We'll handle complete cases in clustering
            remove_constant_features=False
        )
        
        features_processed = preprocessor.fit_transform(features)
        
        # Create mask for complete cases
        valid_mask = ~np.isnan(features_processed).any(axis=1)
        features_clean = features_processed[valid_mask]
        
        logger.info(f"Preprocessing complete: {features_clean.shape}")
        logger.info(f"Complete cases: {valid_mask.sum()} / {len(valid_mask)} ({valid_mask.mean():.1%})")
        
        # Store mask for later use
        self._complete_mask = valid_mask
        
        return features_clean, feature_names
    
    def _create_config(self, **kwargs):
        """Create configuration from kwargs and defaults."""
        # Get method defaults
        defaults = self.method_params.copy()
        
        # Override with user parameters
        config_dict = {**defaults, **kwargs}
        
        # Handle clustering approach - ensure valid combinations
        approach = config_dict.get('clustering_approach', 'ecological')
        if approach == 'geometric':
            # Geometric approach: Ward + Euclidean
            config_dict['linkage_method'] = 'ward'
            config_dict['distance_metric'] = 'euclidean'
            logger.info("Using GEOMETRIC clustering: Ward linkage + Euclidean distance")
        elif approach == 'ecological':
            # Ecological approach: Average + Bray-Curtis
            config_dict['linkage_method'] = 'average'
            config_dict['distance_metric'] = 'braycurtis'
            logger.info("Using ECOLOGICAL clustering: Average linkage + Bray-Curtis distance")
        else:
            raise ValueError(f"Invalid clustering_approach: {approach}. Use 'geometric' or 'ecological'")
        
        # Filter out non-config parameters and legacy parameters
        non_config_keys = ['save_results', 'output_dir', 'experiment_name', 'resume', 'method',
                          'weight_method', 'adaptive_mode', 'grid_size_km',
                          'handle_zeros', 'prefilter_empty', 'min_features_prefilter',
                          'use_sparse_distances', 'determine_k_method', 'clustering_approach',
                          'n_jobs', 'chunk_size', 'calculate_silhouette']
        for key in non_config_keys:
            config_dict.pop(key, None)
        
        # Create config object
        return KMeansHierarchicalConfig(**config_dict)
    
    def _perform_clustering(self, features, config):
        """Perform two-stage clustering.
        
        Note: features are already preprocessed complete cases only.
        """
        
        # Determine optimal k for k-means using elbow method
        if config.use_elbow_method:
            logger.info("Determining optimal k using elbow method...")
            elbow_kmeans = ElbowMethodKMeans(
                k_range=config.k_range,
                batch_size=config.batch_size,
                elbow_threshold=config.elbow_threshold
            )
            n_pre_clusters = elbow_kmeans.determine_k(features)
        else:
            n_pre_clusters = config.n_pre_clusters
        
        # Initialize two-stage clusterer
        clusterer = TwoStageClusterer(
            n_pre_clusters=n_pre_clusters,
            linkage_method=config.linkage_method,
            distance_metric=config.distance_metric,
            final_k=config.final_k,
            batch_size=config.batch_size,
            random_state=config.random_state
        )
        
        # Fit clustering
        clusterer.fit(features)
        
        # Determine optimal final k if not fixed
        if config.final_k is None:
            if config.selection_method == 'bootstrap':
                logger.info("Determining optimal k using bootstrap stability...")
                bootstrap = BootstrapStability(
                    n_bootstrap=config.n_bootstrap,
                    sample_ratio=config.bootstrap_sample_ratio,
                    linkage_method=config.linkage_method
                )
                # Use filtered centroids if available
                centroids_for_bootstrap = (
                    clusterer.filtered_centroids_ 
                    if hasattr(clusterer, 'filtered_centroids_') and clusterer.filtered_centroids_ is not None
                    else clusterer.kmeans.cluster_centers_
                )
                optimal_k = bootstrap.evaluate(
                    centroids_for_bootstrap,
                    clusterer.linkage_matrix,
                    config.final_k_range
                )
                
                # Re-cluster with optimal k
                clusterer.final_k = optimal_k
                clusterer._fit_hierarchical()
                clusterer._map_labels(features)
        
        logger.info(f"Clustering complete: {clusterer.final_k} final clusters")
        
        return clusterer, clusterer.final_labels_
    
    def _generate_visualizations(self, biodiv_data, features, labels, 
                                metrics, config):
        """Generate comprehensive visualizations."""
        logger.info("Generating visualizations...")
        viz = {}
        
        try:
            if config.generate_dendrogram and self.clusterer:
                viz['dendrogram'] = self.visualizer.plot_dendrogram(
                    self.clusterer.linkage_matrix,
                    self.clusterer.final_k
                )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Failed to generate dendrogram: {e}")
        
        try:
            if config.generate_geographic_maps:
                viz['geographic_clusters'] = self.visualizer.plot_geographic_clusters(
                    biodiv_data.coordinates,
                    labels,
                    sample_size=config.n_samples_for_viz
                )
                
                viz['cluster_density_maps'] = self.visualizer.plot_cluster_densities(
                    biodiv_data.coordinates,
                    labels
                )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Failed to generate geographic maps: {e}")
        
        try:
            if config.generate_3d_visualization:
                viz['3d_clusters'] = self.visualizer.plot_3d_clusters(
                    features[:, :3],
                    labels,
                    feature_names=biodiv_data.feature_names[:3],
                    sample_size=config.n_samples_for_viz
                )
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Failed to generate 3D visualization: {e}")
        
        try:
            if config.generate_validation_plots:
                viz['validation_metrics'] = self.visualizer.plot_validation_metrics(
                    metrics,
                    getattr(self.clusterer, 'variance_preserved_', None)
                )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Failed to generate validation plots: {e}")
        
        try:
            # Reuse existing visualizations
            viz['cluster_profiles'] = self.visualizer.plot_cluster_profiles(
                features,
                labels,
                biodiv_data.feature_names
            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Failed to generate cluster profiles: {e}")
        
        try:
            viz['parallel_coordinates'] = self.visualizer.plot_parallel_coordinates(
                features,
                labels,
                biodiv_data.feature_names,
                sample_size=min(10000, len(features))
            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Failed to generate parallel coordinates: {e}")
        
        return viz
    
    def _create_result(self, biodiv_data, labels, metrics, visualizations,
                      config, start_time):
        """Create analysis result."""
        
        # Create metadata
        metadata = AnalysisMetadata(
            analysis_type='kmeans_hierarchical',
            input_shape=(int(biodiv_data.n_samples), int(biodiv_data.n_features)),
            input_bands=[str(n) for n in biodiv_data.feature_names] if biodiv_data.feature_names else [],
            parameters={
                'n_pre_clusters': self.clusterer.n_pre_clusters,
                'linkage_method': config.linkage_method,
                'final_k': self.clusterer.final_k,
                'variance_preserved': getattr(self.clusterer, 'variance_preserved_', None)
            },
            processing_time=(datetime.now() - start_time).total_seconds(),
            timestamp=datetime.now().isoformat(),
            data_source=str(biodiv_data.metadata.get('source_file', 'unknown')) if biodiv_data.metadata else None,
            normalization_applied=True
        )
        
        # Add cluster centers if available
        additional_outputs = visualizations.copy()
        if self.clusterer and hasattr(self.clusterer, 'cluster_centers_'):
            additional_outputs['cluster_centers'] = self.clusterer.cluster_centers_.tolist()
        
        additional_outputs['coordinates'] = biodiv_data.coordinates.tolist()
        
        return AnalysisResult(
            labels=labels,
            metadata=metadata,
            statistics=metrics,
            additional_outputs=additional_outputs
        )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:

    
        """Validate parameters using utility."""

    

    
        # Define required parameters for this analyzer

    
        required = self.get_required_parameters() if hasattr(self, 'get_required_parameters') else []

    
        return validate_params(parameters, required)
