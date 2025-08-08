"""Two-stage K-means → Hierarchical analyzer for biodiversity data.

Clean implementation that reuses existing infrastructure without redundancy.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

from src.biodiversity_analysis.shared.base_analyzer import BaseBiodiversityAnalyzer
from src.abstractions.interfaces.analyzer import AnalysisResult
from src.biodiversity_analysis.preprocessing.pipeline import BiodiversityPreprocessingPipeline

from .hierarchical_core import (
    TwoStageClusterer,
    VariancePreservingKMeans,
    BootstrapStability,
    calculate_clustering_metrics
)
from .hierarchical_config import KMeansHierarchicalConfig
from .hierarchical_visualization import HierarchicalVisualizer

logger = logging.getLogger(__name__)


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
        
        # Load and validate data
        logger.info(f"Loading biodiversity data from {data_path}")
        biodiv_data = self.load_data(data_path)
        
        logger.info(f"Loaded {biodiv_data.n_samples:,} samples with "
                   f"{biodiv_data.n_features} features")
        
        # Preprocess features
        features_processed, feature_names = self._preprocess_features(
            biodiv_data.features,
            biodiv_data.coordinates,
            biodiv_data.feature_names
        )
        
        # Create configuration
        config = self._create_config(**kwargs)
        
        # Perform two-stage clustering
        self.clusterer, final_labels = self._perform_clustering(
            features_processed,
            config
        )
        
        # Calculate quality metrics
        quality_metrics = calculate_clustering_metrics(
            features_processed,
            final_labels,
            sample_size=config.silhouette_sample_size
        )
        
        # Generate visualizations if requested
        visualizations = {}
        if any([config.generate_dendrogram, config.generate_geographic_maps,
                config.generate_3d_visualization, config.generate_validation_plots]):
            visualizations = self._generate_visualizations(
                biodiv_data,
                features_processed,
                final_labels,
                quality_metrics,
                config
            )
        
        # Create result
        return self._create_result(
            biodiv_data,
            final_labels,
            quality_metrics,
            visualizations,
            config,
            start_time
        )
    
    def _preprocess_features(self, features, coordinates, feature_names):
        """Preprocess features using shared pipeline."""
        logger.info("Preprocessing features...")
        
        # Use shared preprocessing pipeline
        preprocessing_pipeline = BiodiversityPreprocessingPipeline(
            compositional_transform='hurdle',
            structural_zero_threshold=0.95,
            use_pmem=False,  # Disable spatial features for clustering
            calculate_diversity_indices=False,
            calculate_interactions=False,
            standardize=True,
            standardization_method='robust'
        )
        
        features_processed, new_feature_names, _ = preprocessing_pipeline.fit_transform(
            species_data=features,
            coordinates=coordinates,
            species_names=feature_names
        )
        
        logger.info(f"Preprocessing complete: {features_processed.shape}")
        
        return features_processed, new_feature_names
    
    def _create_config(self, **kwargs):
        """Create configuration from kwargs and defaults."""
        # Get method defaults
        defaults = self.method_params.copy()
        
        # Override with user parameters
        config_dict = {**defaults, **kwargs}
        
        # Create config object
        return KMeansHierarchicalConfig(**config_dict)
    
    def _perform_clustering(self, features, config):
        """Perform two-stage clustering."""
        
        # Handle complete cases only
        complete_mask = ~np.isnan(features).any(axis=1)
        features_complete = features[complete_mask]
        
        logger.info(f"Using {complete_mask.sum():,} complete cases "
                   f"({complete_mask.mean():.1%} of data)")
        
        # Determine optimal k for k-means if needed
        if config.use_variance_preservation:
            logger.info("Determining optimal k using variance preservation...")
            vp_kmeans = VariancePreservingKMeans(
                variance_threshold=config.variance_threshold,
                k_range=config.variance_k_range,
                batch_size=config.batch_size
            )
            n_pre_clusters = vp_kmeans.determine_k(features_complete)
        else:
            n_pre_clusters = config.n_pre_clusters
        
        # Initialize two-stage clusterer
        clusterer = TwoStageClusterer(
            n_pre_clusters=n_pre_clusters,
            linkage_method=config.linkage_method,
            final_k=config.final_k,
            batch_size=config.batch_size,
            random_state=config.random_state
        )
        
        # Fit clustering
        clusterer.fit(features_complete)
        
        # Determine optimal final k if not fixed
        if config.final_k is None:
            if config.selection_method == 'bootstrap':
                logger.info("Determining optimal k using bootstrap stability...")
                bootstrap = BootstrapStability(
                    n_bootstrap=config.n_bootstrap,
                    sample_ratio=config.bootstrap_sample_ratio
                )
                optimal_k = bootstrap.evaluate(
                    clusterer.kmeans.cluster_centers_,
                    clusterer.linkage_matrix,
                    config.final_k_range
                )
                
                # Re-cluster with optimal k
                clusterer.final_k = optimal_k
                clusterer._fit_hierarchical()
                clusterer._map_labels(features_complete)
        
        # Map labels back to full dataset
        full_labels = np.full(len(features), -1, dtype=int)
        full_labels[complete_mask] = clusterer.final_labels_
        
        logger.info(f"Clustering complete: {clusterer.final_k} final clusters")
        
        return clusterer, full_labels
    
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
            
            if config.generate_3d_visualization:
                viz['3d_clusters'] = self.visualizer.plot_3d_clusters(
                    features[:, :3],
                    labels,
                    feature_names=biodiv_data.feature_names[:3],
                    sample_size=config.n_samples_for_viz
                )
            
            if config.generate_validation_plots:
                viz['validation_metrics'] = self.visualizer.plot_validation_metrics(
                    metrics,
                    getattr(self.clusterer, 'variance_preserved_', None)
                )
            
            # Reuse existing visualizations
            viz['cluster_profiles'] = self.visualizer.plot_cluster_profiles(
                features,
                labels,
                biodiv_data.feature_names
            )
            
            viz['parallel_coordinates'] = self.visualizer.plot_parallel_coordinates(
                features,
                labels,
                biodiv_data.feature_names,
                sample_size=min(10000, len(features))
            )
            
        except Exception as e:
            logger.warning(f"Some visualizations failed: {e}")
        
        return viz
    
    def _create_result(self, biodiv_data, labels, metrics, visualizations,
                      config, start_time):
        """Create analysis result."""
        from src.abstractions.interfaces.analyzer import AnalysisMetadata
        
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
        """Validate analysis parameters."""
        errors = []
        
        # Validate n_pre_clusters
        if 'n_pre_clusters' in parameters:
            n = parameters['n_pre_clusters']
            if not isinstance(n, int) or n < 10:
                errors.append("n_pre_clusters must be an integer >= 10")
        
        # Validate linkage method
        if 'linkage_method' in parameters:
            valid = ['ward', 'complete', 'average', 'single']
            if parameters['linkage_method'] not in valid:
                errors.append(f"linkage_method must be one of {valid}")
        
        # Validate final_k_range
        if 'final_k_range' in parameters:
            r = parameters['final_k_range']
            if not isinstance(r, (tuple, list)) or len(r) != 2:
                errors.append("final_k_range must be a tuple of (min, max)")
            elif r[0] >= r[1]:
                errors.append("final_k_range must have min < max")
        
        return len(errors) == 0, errors
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters."""
        return {
            'n_pre_clusters': 750,
            'use_variance_preservation': True,
            'variance_threshold': 0.95,
            'linkage_method': 'ward',
            'final_k_range': (4, 8),
            'selection_method': 'bootstrap',
            'n_bootstrap': 30,
            'generate_dendrogram': True,
            'generate_geographic_maps': True,
            'save_results': True,
            'output_dir': './outputs/kmeans_hierarchical_results'
        }
    
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
            output_dir = Path("./outputs/kmeans_hierarchical_results")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save labels
        labels_file = output_dir / f"{output_name}_labels.npy"
        np.save(labels_file, result.labels)
        
        # Save metadata as JSON
        import json
        metadata_file = output_dir / f"{output_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'analysis_type': result.metadata.analysis_type,
                'input_shape': result.metadata.input_shape,
                'parameters': result.metadata.parameters,
                'processing_time': result.metadata.processing_time,
                'timestamp': result.metadata.timestamp
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
        return output_dir
    
    def set_progress_callback(self, callback):
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
        path = Path(data_path)
        
        if not path.exists():
            errors.append(f"Input file does not exist: {data_path}")
            return False, errors
        
        if not path.suffix.lower() in ['.parquet', '.csv']:
            errors.append(f"Unsupported format: {path.suffix}. Use .parquet or .csv")
        
        return len(errors) == 0, errors