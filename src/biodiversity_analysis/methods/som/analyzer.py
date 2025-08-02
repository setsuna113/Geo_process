"""GeoSOM + VLRSOM Analyzer for biodiversity data.

This analyzer implements the specifications from final_som_configuration_decisions.md.
It replaces the old analyzer.py with the new architecture.
"""

import numpy as np
import time
import json
import os
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
from datetime import datetime

from src.biodiversity_analysis.base_analyzer import BaseBiodiversityAnalyzer
from src.abstractions.interfaces.analyzer import AnalysisResult
from .geo_som_core import GeoSOMVLRSOM, GeoSOMConfig
from .preprocessing import BiodiversityPreprocessor
from .spatial_utils import SpatialBlockGenerator, GeographicCoherence
from .partial_metrics import validate_missing_data_handling
from .progress_tracker import SOMProgressTracker, create_progress_callback

logger = logging.getLogger(__name__)


class GeoSOMAnalyzer(BaseBiodiversityAnalyzer):
    """Geographic Self-Organizing Map analyzer for biodiversity patterns.
    
    Implements:
    - GeoSOM + VLRSOM hybrid architecture
    - Partial Bray-Curtis for 70% missing data
    - Spatial weighting (30% spatial, 70% features)
    - Adaptive learning rate
    - Geographic coherence convergence
    - 750km spatial block cross-validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize GeoSOM analyzer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(method_name='geosom', version='1.0.0')
        
        # Load configuration from file if not provided
        if config is None:
            config = self._load_default_config()
        
        self.config = config
        self.som = None
        self.preprocessor = BiodiversityPreprocessor()
        self.training_result = None
        self._geosom_config = None
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load configuration from SOM config module."""
        from src.config.som import get_som_config
        som_config = get_som_config()
        
        # Return the loaded configuration as a dictionary
        return som_config._config
    
    def analyze(self, data_path: str, **kwargs) -> AnalysisResult:
        """Main analysis method for GeoSOM.
        
        Args:
            data_path: Path to parquet file with biodiversity data
            **kwargs: Additional parameters including:
                - observed_columns: List of observed feature column indices
                - predicted_columns: List of predicted feature column indices
                - coordinate_columns: Names of coordinate columns
                - grid_size: SOM grid size (tuple or 'auto')
                - save_results: Whether to save results
                - output_dir: Directory for saving results
                
        Returns:
            AnalysisResult with GeoSOM analysis outputs
        """
        start_time = time.time()
        logger.info("Starting GeoSOM analysis for biodiversity data")
        
        # Load data
        biodiv_data = self.load_data(
            data_path,
            coordinate_cols=kwargs.get('coordinate_columns', ['longitude', 'latitude'])
        )
        
        # Validate missing data handling
        missing_stats = validate_missing_data_handling(biodiv_data.features)
        logger.info(f"Data has {missing_stats['overall_missing_proportion']:.1%} missing values")
        
        # Get column indices
        observed_cols = kwargs.get('observed_columns', [0, 1])
        predicted_cols = kwargs.get('predicted_columns', [2, 3])
        
        # Preprocess data
        features_processed = self.preprocessor.fit_transform(
            biodiv_data.features,
            observed_cols,
            predicted_cols
        )
        
        # Determine grid size
        grid_size = self._determine_grid_size(
            len(features_processed),
            kwargs.get('grid_size', 'auto')
        )
        
        # Create GeoSOM configuration
        # Remove grid_size from kwargs if present to avoid duplicate argument
        config_kwargs = {k: v for k, v in kwargs.items() if k != 'grid_size'}
        self._geosom_config = self._create_geosom_config(grid_size, **config_kwargs)
        
        # Initialize progress tracker
        output_dir = kwargs.get('output_dir', './outputs/analysis_results/som')
        experiment_name = kwargs.get('experiment_name', f'som_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.progress_tracker = SOMProgressTracker(output_dir, experiment_name)
        
        # Perform spatial block cross-validation
        cv_results = self._spatial_cross_validation(
            features_processed,
            biodiv_data.coordinates,
            self._geosom_config,
            **kwargs
        )
        
        # Train final model on all data
        self.som = GeoSOMVLRSOM(self._geosom_config)
        
        # Create a simple progress callback if we have one
        progress_cb = None
        if hasattr(self, 'progress_callback') and self.progress_callback:
            def progress_cb(progress):
                msg = f"Training SOM: {int(progress * 100)}% complete"
                self.progress_callback(msg, progress)
        
        self.training_result = self.som.train_batch(
            features_processed,
            biodiv_data.coordinates,
            progress_callback=progress_cb
        )
        
        # Calculate comprehensive metrics
        final_metrics = self._calculate_final_metrics(
            features_processed,
            biodiv_data.coordinates,
            biodiv_data
        )
        
        # Generate visualizations and reports
        reports = self._generate_reports(
            features_processed,
            biodiv_data.coordinates,
            biodiv_data.feature_names
        )
        
        # Create result
        runtime = time.time() - start_time
        result = self._create_result(
            features_processed,
            biodiv_data,
            cv_results,
            final_metrics,
            reports,
            runtime
        )
        
        # Save if requested
        if kwargs.get('save_results', False):
            self._save_results(result, kwargs.get('output_dir', './results'))
        
        return result
    
    def _determine_grid_size(self, n_samples: int, grid_size_param: Any) -> Tuple[int, int]:
        """Determine appropriate grid size based on data."""
        # Handle both list and tuple
        if isinstance(grid_size_param, (list, tuple)) and len(grid_size_param) == 2:
            return tuple(grid_size_param)
        
        # Auto-determine based on sample size
        # Rule of thumb: sqrt(n_samples) neurons, arranged in square grid
        n_neurons = int(np.sqrt(n_samples))
        grid_dim = int(np.sqrt(n_neurons))
        
        # Ensure reasonable bounds
        grid_dim = max(5, min(grid_dim, 50))
        
        logger.info(f"Auto-determined grid size: {grid_dim}x{grid_dim} for {n_samples} samples")
        return (grid_dim, grid_dim)
    
    def _create_geosom_config(self, grid_size: Tuple[int, int], **kwargs) -> GeoSOMConfig:
        """Create GeoSOM configuration from settings."""
        # Use the default config we loaded, not the biodiversity config
        default_config = self._load_default_config()
        arch_config = default_config['architecture_config']
        conv_config = arch_config['convergence']
        
        # Use iterations from kwargs if provided, otherwise fall back to default
        max_epochs = kwargs.get('iterations', conv_config['max_epochs'])
        
        # Import SOM config to get performance settings
        from src.config.som import get_som_config
        som_config = get_som_config()
        
        # Get performance settings
        chunk_size = som_config.get_chunk_size()
        qe_sample_size = som_config.get_qe_sample_size()
        
        return GeoSOMConfig(
            grid_size=grid_size,
            topology=arch_config['topology'],
            spatial_weight=arch_config['spatial_weight'],
            geographic_distance=arch_config['geographic_distance'],
            initial_learning_rate=arch_config['initial_learning_rate'],
            min_learning_rate=arch_config['min_learning_rate'],
            max_learning_rate=arch_config['max_learning_rate'],
            lr_increase_factor=arch_config['lr_increase_factor'],
            lr_decrease_factor=arch_config['lr_decrease_factor'],
            high_qe_lr_min=arch_config['high_qe_lr_range'][0],
            high_qe_lr_max=arch_config['high_qe_lr_range'][1],
            low_qe_lr_min=arch_config['low_qe_lr_range'][0],
            low_qe_lr_max=arch_config['low_qe_lr_range'][1],
            neighborhood_function=arch_config['neighborhood_function'],
            initial_radius=arch_config['initial_radius'],
            final_radius=arch_config['final_radius'],
            radius_decay=arch_config['radius_decay'],
            geographic_coherence_threshold=conv_config['geographic_coherence_threshold'],
            lr_stability_threshold=conv_config['lr_stability_threshold'],
            qe_improvement_threshold=conv_config['qe_improvement_threshold'],
            patience=conv_config['patience'],
            max_epochs=max_epochs,
            min_valid_features=default_config['distance_config']['min_valid_features'],
            random_seed=kwargs.get('random_seed', 42),
            # Add performance settings
            chunk_size=kwargs.get('chunk_size', chunk_size),
            qe_sample_size=kwargs.get('qe_sample_size', qe_sample_size)
        )
    
    def _spatial_cross_validation(self, features: np.ndarray, 
                                coordinates: np.ndarray,
                                config: GeoSOMConfig,
                                **kwargs) -> Dict[str, Any]:
        """Perform spatial block cross-validation."""
        # Create spatial blocks
        validation_config = self.config.get('validation_config', {})
        n_folds = kwargs.get('cv_folds', validation_config.get('n_folds', 5))
        block_size = validation_config.get('block_size', '750km')
        block_size_km = float(block_size.replace('km', ''))
        block_gen = SpatialBlockGenerator(block_size_km=block_size_km, random_state=config.random_seed)
        cv_splits = block_gen.create_cv_splits(coordinates, n_folds=n_folds)
        
        cv_results = {
            'fold_results': [],
            'mean_qe': 0.0,
            'std_qe': 0.0,
            'mean_geographic_coherence': 0.0,
            'std_geographic_coherence': 0.0
        }
        
        qe_scores = []
        coherence_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            logger.info(f"Processing CV fold {fold_idx + 1}/{n_folds}")
            self.progress_tracker.update_phase(f'cv_fold_{fold_idx + 1}', cv_fold=fold_idx + 1)
            
            # Create fold SOM
            fold_som = GeoSOMVLRSOM(config)
            
            # Create progress callback for this fold
            fold_callback = create_progress_callback(
                self.progress_tracker, 
                phase=f'cv_fold_{fold_idx + 1}'
            )
            
            # Train on fold
            fold_result = fold_som.train_batch(
                features[train_idx],
                coordinates[train_idx],
                progress_callback=fold_callback
            )
            
            # Evaluate on test set
            test_qe = fold_som.calculate_quantization_error(
                features[test_idx],
                coordinates[test_idx]
            )
            
            # Calculate geographic coherence
            test_predictions = fold_som.predict(features[test_idx], coordinates[test_idx])
            geo_coherence = GeographicCoherence.morans_i(
                test_predictions,
                coordinates[test_idx]
            )
            
            qe_scores.append(test_qe)
            coherence_scores.append(geo_coherence)
            
            cv_results['fold_results'].append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'test_qe': test_qe,
                'geographic_coherence': geo_coherence,
                'convergence_epoch': fold_result.convergence_epoch
            })
        
        # Aggregate results
        cv_results['mean_qe'] = np.mean(qe_scores)
        cv_results['std_qe'] = np.std(qe_scores)
        cv_results['mean_geographic_coherence'] = np.mean(coherence_scores)
        cv_results['std_geographic_coherence'] = np.std(coherence_scores)
        
        logger.info(f"CV Results: QE={cv_results['mean_qe']:.3f}±{cv_results['std_qe']:.3f}, "
                   f"Geo Coherence={cv_results['mean_geographic_coherence']:.3f}±{cv_results['std_geographic_coherence']:.3f}")
        
        return cv_results
    
    def _calculate_final_metrics(self, features: np.ndarray,
                               coordinates: np.ndarray,
                               biodiv_data) -> Dict[str, Any]:
        """Calculate comprehensive final metrics."""
        # Get predictions
        predictions = self.som.predict(features, coordinates)
        grid_coords = self.som.transform(features, coordinates)
        
        # Basic SOM metrics
        final_qe = self.som.calculate_quantization_error(features, coordinates)
        
        # Geographic metrics
        geo_metrics = GeographicCoherence.geographic_cluster_quality(
            predictions, coordinates
        )
        
        # Feature importance (variance across map)
        weights = self.som.get_weights()
        feature_variance = np.var(weights, axis=(0, 1))
        feature_importance = feature_variance / np.sum(feature_variance)
        
        return {
            'final_quantization_error': final_qe,
            'geographic_metrics': geo_metrics,
            'feature_importance': feature_importance,
            'convergence_info': {
                'converged': self.training_result.convergence_epoch is not None,
                'convergence_epoch': self.training_result.convergence_epoch,
                'final_learning_rate': self.training_result.final_learning_rate,
                'final_radius': self.training_result.final_neighborhood_radius
            },
            'cluster_assignments': predictions,
            'grid_coordinates': grid_coords
        }
    
    def _generate_reports(self, features: np.ndarray,
                        coordinates: np.ndarray,
                        feature_names: List[str]) -> Dict[str, Any]:
        """Generate comprehensive reports and visualizations."""
        reports = {
            'preprocessing_info': self.preprocessor.get_preprocessing_info(),
            'training_history': self.som.training_history,
            'grid_info': {
                'size': self._geosom_config.grid_size,
                'topology': self._geosom_config.topology,
                'n_neurons': self.som.n_neurons
            },
            'data_info': {
                'n_samples': len(features),
                'n_features': features.shape[1],
                'missing_proportion': np.isnan(features).mean(),
                'geographic_bounds': {
                    'lon_min': coordinates[:, 0].min(),  # coordinates are [longitude, latitude]
                    'lon_max': coordinates[:, 0].max(),
                    'lat_min': coordinates[:, 1].min(),
                    'lat_max': coordinates[:, 1].max()
                }
            }
        }
        
        # Note: Actual visualization generation would be implemented here
        # For now, we just prepare the data structures
        
        return reports
    
    def _create_result(self, features: np.ndarray, biodiv_data,
                      cv_results: Dict, final_metrics: Dict,
                      reports: Dict, runtime: float) -> AnalysisResult:
        """Create analysis result object."""
        # Prepare result data
        result_data = {
            'labels': final_metrics['grid_coordinates'],
            'statistics': {
                'cv_results': cv_results,
                'final_metrics': final_metrics,
                'preprocessing': reports['preprocessing_info']
            }
        }
        
        # Create result using base class method
        result = self.create_result(
            success=True,
            data=result_data,
            runtime_seconds=runtime,
            n_samples=len(features),
            n_features=features.shape[1],
            parameters={
                'method': 'GeoSOM+VLRSOM',
                'grid_size': self._geosom_config.grid_size,
                'spatial_weight': self._geosom_config.spatial_weight,
                'distance_metric': 'partial_bray_curtis',
                'preprocessing': 'log1p+separate_zscore'
            }
        )
        
        # Add additional outputs
        result.additional_outputs.update({
            'weights': self.som.get_weights(),
            'training_history': reports['training_history'],
            'feature_importance': final_metrics['feature_importance'],
            'cluster_assignments': final_metrics['cluster_assignments'],
            'geographic_quality': final_metrics['geographic_metrics']
        })
        
        return result
    
    def _save_results(self, result: AnalysisResult, output_dir: str):
        """Save analysis results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'geosom_metadata.json')
        metadata = {
            'analysis_type': result.metadata.analysis_type,
            'input_shape': result.metadata.input_shape,
            'parameters': result.metadata.parameters,
            'processing_time': result.metadata.processing_time,
            'timestamp': result.metadata.timestamp,
            'statistics': result.statistics
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save numpy arrays
        np.save(os.path.join(output_dir, 'cluster_assignments.npy'), 
                result.additional_outputs['cluster_assignments'])
        np.save(os.path.join(output_dir, 'grid_coordinates.npy'), 
                result.labels)
        np.save(os.path.join(output_dir, 'som_weights.npy'), 
                result.additional_outputs['weights'])
        np.save(os.path.join(output_dir, 'feature_importance.npy'), 
                result.additional_outputs['feature_importance'])
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(result.additional_outputs['training_history'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for GeoSOM analysis."""
        return {
            'grid_size': 'auto',
            'spatial_weight': 0.3,
            'observed_columns': [0, 1],
            'predicted_columns': [2, 3],
            'coordinate_columns': ['longitude', 'latitude'],
            'max_epochs': 1000,
            'random_seed': 42
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate analysis parameters."""
        issues = []
        
        # Check required parameters
        if 'observed_columns' not in parameters:
            issues.append("Missing required parameter: observed_columns")
        if 'predicted_columns' not in parameters:
            issues.append("Missing required parameter: predicted_columns")
        
        # Validate grid size
        if 'grid_size' in parameters:
            grid_size = parameters['grid_size']
            if grid_size != 'auto':
                if not isinstance(grid_size, (list, tuple)) or len(grid_size) != 2:
                    issues.append("grid_size must be 'auto' or tuple of 2 integers")
        
        # Validate spatial weight
        if 'spatial_weight' in parameters:
            sw = parameters['spatial_weight']
            if not isinstance(sw, (int, float)) or sw < 0 or sw > 1:
                issues.append("spatial_weight must be between 0 and 1")
        
        return len(issues) == 0, issues
    
    def save_results(self, result: AnalysisResult, output_path: str) -> None:
        """Save analysis results to specified path."""
        self._save_results(result, output_path)
    
    def set_progress_callback(self, callback: Optional[Callable[[str, float], None]]) -> None:
        """Set progress callback function."""
        self.progress_callback = callback
    
    def validate_input_data(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate input data for GeoSOM analysis."""
        issues = []
        
        if not isinstance(data, str):
            issues.append("Input must be a path to parquet file")
        elif not data.endswith('.parquet'):
            issues.append("Only .parquet files are supported")
        elif not os.path.exists(data):
            issues.append(f"File not found: {data}")
        
        return len(issues) == 0, issues