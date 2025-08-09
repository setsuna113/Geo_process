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

from src.biodiversity_analysis.shared.base_analyzer import BaseBiodiversityAnalyzer
from src.abstractions.interfaces.analyzer import AnalysisResult
from .core.geo_som_core import GeoSOMVLRSOM, GeoSOMConfig
from src.biodiversity_analysis.preprocessing import SharedPreprocessor
from .utils.spatial_utils import SpatialBlockGenerator, GeographicCoherence
from .calculators.partial_metrics import validate_missing_data_handling
from .utils.progress_tracker import SOMProgressTracker, create_progress_callback

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
        
        # Use shared preprocessor configured for SOM
        self.preprocessor = SharedPreprocessor(method_name='som')
        logger.info("Using shared preprocessor for SOM (minmax normalization, zero imputation)")
            
        self.training_result = None
        self._geosom_config = None
    
    def __del__(self):
        """Cleanup resources on deletion."""
        if hasattr(self, 'som') and self.som is not None:
            try:
                self.som.cleanup()
            except:
                pass
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load configuration from SOM config module."""
        import os
        from src.config.som import get_som_config
        
        # Check for config override
        config_path = os.environ.get('SOM_CONFIG_PATH')
        if config_path:
            try:
                if not os.path.exists(config_path):
                    logger.warning(f"Config path does not exist: {config_path}")
                else:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    if not isinstance(config, dict):
                        raise ValueError(f"Config must be dict, got {type(config)}")
                    logger.info(f"Loaded SOM config from: {config_path}")
                    self._validate_config_memory_requirements(config)
                    return config
            except (yaml.YAMLError, IOError, ValueError) as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Falling back to default configuration")
        
        # Otherwise use default
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
        
        # Load data with flexible coordinate column names and spatial sampling
        # First check kwargs, then fall back to config
        logger.info(f"self.config keys: {self.config.keys() if self.config else 'None'}")
        # Support both old ('data_config') and new ('data') config structure
        data_config = self.config.get('data', self.config.get('data_config', {}))
        logger.info(f"data_config from self.config: {data_config}")
        # Use standardized naming: coordinate_columns and feature_cols
        coord_cols = kwargs.get('coordinate_columns', data_config.get('coordinate_columns', ['longitude', 'latitude']))
        feature_cols = kwargs.get('feature_cols', data_config.get('feature_cols', None))
        sample_size = kwargs.get('sample_size', None)  # Optional random sampling
        random_seed = kwargs.get('random_seed', None)
        spatial_grid_step = kwargs.get('spatial_grid_step', data_config.get('spatial_grid_step', 1))
        
        logger.info(f"Data config: spatial_grid_step={spatial_grid_step}, feature_cols={feature_cols}, coord_cols={coord_cols}")
        
        biodiv_data = self.load_data(
            data_path,
            coordinate_cols=coord_cols,
            feature_cols=feature_cols,
            sample_size=sample_size,
            random_seed=random_seed,
            spatial_grid_step=spatial_grid_step
        )
        
        # Handle coordinate column renaming if needed (x,y -> longitude,latitude)
        if coord_cols == ['x', 'y']:
            logger.info("Renaming coordinates: x -> longitude, y -> latitude")
            # The data loader should handle this, but we'll ensure it's done
        
        # Validate missing data handling
        missing_stats = validate_missing_data_handling(biodiv_data.features)
        logger.info(f"Data has {missing_stats['overall_missing_proportion']:.1%} missing values")
        
        # Get column indices from kwargs first, then data_config, then defaults
        observed_cols = kwargs.get('observed_columns', data_config.get('observed_columns', [0, 1]))
        predicted_cols = kwargs.get('predicted_columns', data_config.get('predicted_columns', [2, 3]))
        
        # For biodiversity-only analysis with no predicted columns, ensure it's empty
        if not predicted_cols or predicted_cols == []:
            predicted_cols = []
            logger.info(f"No predicted columns for biodiversity-only analysis")
        
        logger.info(f"Column indices - observed: {observed_cols}, predicted: {predicted_cols}")
        
        # Preprocess data using shared preprocessor
        logger.info("Preprocessing data with SharedPreprocessor (minmax normalization, zero imputation)")
        features_processed = self.preprocessor.fit_transform(biodiv_data.features)
        
        # Note: We no longer need to track observed/predicted columns separately for preprocessing
        # The shared preprocessor handles all features uniformly
        
        # Determine grid size - first check config file, then kwargs, then auto
        # Check both new 'grid' section and old 'architecture_config' for compatibility
        grid_config = self.config.get('grid', {})
        config_grid_size = grid_config.get('size', None)
        
        if config_grid_size is None:
            # Fallback to old architecture_config for compatibility
            arch_config = self.config.get('architecture_config', {})
            config_grid_size = arch_config.get('grid_size', 'auto')
        
        # Ensure it's in the right format
        if isinstance(config_grid_size, list):
            config_grid_size = tuple(config_grid_size)
        
        # Use kwargs grid_size if provided, otherwise use config, otherwise auto
        grid_size_param = kwargs.get('grid_size', config_grid_size)
        
        logger.info(f"Grid size source - kwargs: {kwargs.get('grid_size')}, config: {config_grid_size}, using: {grid_size_param}")
        
        grid_size = self._determine_grid_size(
            len(features_processed),
            grid_size_param
        )
        
        # Create GeoSOM configuration
        # Remove grid_size from kwargs if present to avoid duplicate argument
        config_kwargs = {k: v for k, v in kwargs.items() if k != 'grid_size'}
        self._geosom_config = self._create_geosom_config(grid_size, **config_kwargs)
        
        # Initialize progress tracker
        output_dir = kwargs.get('output_dir', './outputs/analysis_results/som')
        experiment_name = kwargs.get('experiment_name', f'som_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.progress_tracker = SOMProgressTracker(output_dir, experiment_name)
        
        # Check if we should skip CV for exploratory analysis
        skip_cv = kwargs.get('skip_cv', False)
        cv_folds = kwargs.get('cv_folds', self.config.get('validation_config', {}).get('cv_folds', 5))
        
        # For exploratory analysis, set cv_folds=0 or skip_cv=True to skip cross-validation
        if cv_folds == 0:
            skip_cv = True
            logger.info("CV folds set to 0, skipping cross-validation for exploratory analysis")
        
        # Also skip CV if resuming from checkpoint
        if kwargs.get('resume', False):
            checkpoint_dir = 'checkpoints/som'
            latest_checkpoint = os.path.join(checkpoint_dir, 'som_checkpoint_latest.npz')
            if os.path.exists(latest_checkpoint):
                skip_cv = True
                logger.info("Resume requested with existing checkpoint - skipping cross-validation")
        
        cv_results = None
        if not skip_cv:
            logger.info(f"Performing {cv_folds}-fold spatial block cross-validation")
            # Perform spatial block cross-validation
            cv_results = self._spatial_cross_validation(
                features_processed,
                biodiv_data.coordinates,
                self._geosom_config,
                **kwargs
            )
        else:
            logger.info("Skipping cross-validation for exploratory analysis - training on full dataset")
        
        # Train final model on all data
        self.som = GeoSOMVLRSOM(self._geosom_config)
        
        # Check if we should resume from a SOM checkpoint
        # Look for latest checkpoint in the standard location
        checkpoint_dir = 'checkpoints/som'
        latest_checkpoint = os.path.join(checkpoint_dir, 'som_checkpoint_latest.npz')
        logger.info(f"Resume check: resume={kwargs.get('resume', False)}, checkpoint exists={os.path.exists(latest_checkpoint)}, path={latest_checkpoint}")
        if kwargs.get('resume', False) and os.path.exists(latest_checkpoint):
            logger.info(f"Found SOM checkpoint, loading from: {latest_checkpoint}")
            # Set resume_checkpoint in config so geo_som_core.py will load it
            self._geosom_config.resume_checkpoint = latest_checkpoint
            # Recreate SOM with resume config
            self.som = GeoSOMVLRSOM(self._geosom_config)
        else:
            logger.info(f"Not resuming: resume flag={kwargs.get('resume', False)}, checkpoint exists={os.path.exists(latest_checkpoint)}")
        
        # Create a simple progress callback if we have one
        progress_cb = None
        if hasattr(self, 'progress_callback') and self.progress_callback:
            def progress_cb(progress):
                # Handle both dict and float progress values
                if isinstance(progress, dict):
                    # Extract progress value from dict (e.g., from mini-batch training)
                    progress_val = progress.get('progress', progress.get('epoch', 0) / self._geosom_config.max_epochs)
                else:
                    progress_val = progress
                msg = f"Training SOM: {int(progress_val * 100)}% complete"
                self.progress_callback(msg, progress_val)
        
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
    
    def _validate_config_memory_requirements(self, config: Dict[str, Any]) -> None:
        """Validate that configured grid size fits in available memory."""
        import psutil
        
        # Extract grid size from config
        arch_config = config.get('architecture_config', {})
        grid_size = arch_config.get('grid_size', [10, 10])
        if not isinstance(grid_size, (list, tuple)) or len(grid_size) != 2:
            return  # Skip validation for auto-determined sizes
        
        # Estimate requirements
        n_neurons = grid_size[0] * grid_size[1]
        
        # Get feature count from config
        data_config = config.get('data_config', {})
        feature_cols = data_config.get('feature_cols', ['animal_richness', 'plant_richness', 'am_fungi_richness', 'ecm_fungi_richness'])
        n_features = len(feature_cols)
        
        # Conservative estimate (assuming 1M samples)
        estimated_samples = 1000000
        weights_memory = n_neurons * n_features * 8
        distance_memory = estimated_samples * n_neurons * 8
        total_mb = (weights_memory + distance_memory) / (1024 * 1024)
        
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        if total_mb > available_mb * 0.5:
            logger.warning(f"Config specifies {grid_size[0]}x{grid_size[1]} grid "
                         f"which may require up to {total_mb:.1f} MB. "
                         f"Only {available_mb:.1f} MB available. "
                         f"Grid size will be adjusted at runtime if needed.")
    
    def _determine_grid_size(self, n_samples: int, grid_size_param: Any) -> Tuple[int, int]:
        """Determine appropriate grid size based on data with memory validation."""
        import psutil
        
        # Handle both list and tuple
        if isinstance(grid_size_param, (list, tuple)) and len(grid_size_param) == 2:
            grid_size = tuple(grid_size_param)
        else:
            # Auto-determine based on sample size
            # Rule of thumb: sqrt(n_samples) neurons, arranged in square grid
            n_neurons = int(np.sqrt(n_samples))
            grid_dim = int(np.sqrt(n_neurons))
            
            # Ensure reasonable bounds
            grid_dim = max(5, min(grid_dim, 50))
            grid_size = (grid_dim, grid_dim)
        
        # Memory validation
        # Get feature count from config
        data_config = self.config.get('data_config', {})
        feature_cols = data_config.get('feature_cols', ['animal_richness', 'plant_richness', 'am_fungi_richness', 'ecm_fungi_richness'])
        n_features = len(feature_cols)
        n_neurons = grid_size[0] * grid_size[1]
        
        # Estimate memory requirements (in bytes)
        # Weights: n_neurons * n_features * 8 bytes (float64)
        weights_memory = n_neurons * n_features * 8
        # Distance matrix (temporary): n_samples * n_neurons * 8 bytes
        distance_memory = n_samples * n_neurons * 8
        # Additional buffers and overhead: ~2x the weights
        overhead_memory = weights_memory * 2
        
        total_required = weights_memory + distance_memory + overhead_memory
        total_required_mb = total_required / (1024 * 1024)
        
        # Get available memory
        available_memory = psutil.virtual_memory().available
        available_mb = available_memory / (1024 * 1024)
        
        # Check if we have enough memory (use 50% of available as safe threshold)
        safe_available = available_mb * 0.5
        
        # Allow forcing grid size via environment variable
        import os
        force_grid_value = os.environ.get('SOM_FORCE_GRID_SIZE', '')
        force_grid = force_grid_value.lower() in ['1', 'true', 'yes']
        logger.info(f"SOM_FORCE_GRID_SIZE env var: '{force_grid_value}', force_grid: {force_grid}")
        
        if total_required_mb > safe_available and not force_grid:
            # Reduce grid size to fit memory
            max_neurons = int((safe_available * 1024 * 1024 - overhead_memory) / 
                            (n_features * 8 + n_samples * 8))
            max_grid_dim = int(np.sqrt(max_neurons))
            max_grid_dim = max(5, max_grid_dim)  # Minimum 5x5
            
            logger.warning(f"Grid size {grid_size} requires {total_required_mb:.1f} MB, "
                         f"but only {safe_available:.1f} MB safely available. "
                         f"Reducing to {max_grid_dim}x{max_grid_dim}")
            grid_size = (max_grid_dim, max_grid_dim)
        elif force_grid and total_required_mb > safe_available:
            logger.warning(f"FORCING grid size {grid_size[0]}x{grid_size[1]} despite memory constraints "
                         f"({total_required_mb:.1f} MB required, {safe_available:.1f} MB safe threshold). "
                         f"SOM_FORCE_GRID_SIZE is set.")
        else:
            logger.info(f"Grid size {grid_size[0]}x{grid_size[1]} will use "
                       f"{total_required_mb:.1f} MB of {available_mb:.1f} MB available")
        
        return grid_size
    
    def _create_geosom_config(self, grid_size: Tuple[int, int], **kwargs) -> GeoSOMConfig:
        """Create GeoSOM configuration from settings."""
        # Use the config we already loaded in __init__, not load it again
        default_config = self.config
        
        # Get SOM config instance for compatibility layer
        from src.config.som import get_som_config
        som_config = get_som_config()
        
        # Use compatibility layer methods to get config in old format
        arch_config = som_config.get_architecture_config()
        training_config = som_config.get_training_config()
        
        # Use iterations from kwargs if provided, otherwise from training_config
        max_epochs = kwargs.get('iterations', training_config['max_epochs'])  # FIXED: Now reads from training_config!
        
        # Get performance settings - prefer training_config values
        chunk_size = training_config.get('chunk_size', som_config.get_chunk_size())
        qe_sample_size = som_config.get_qe_sample_size()
        use_parallel = som_config.is_parallel_enabled()
        n_workers = training_config.get('n_workers', som_config.get_n_workers())
        
        # Get batch training settings from training config
        # All optimizations are now built-in and always enabled
        batch_config = training_config.get('batch_training', {
            'batch_size': 5000,
            'shuffle_batches': True
        })
        
        # Get memory and performance config
        memory_config = default_config.get('memory_config', {})
        
        # Create config object reading from correct sections
        config = GeoSOMConfig(
            grid_size=grid_size,
            topology=arch_config['topology'],
            spatial_weight=arch_config['spatial_weight'],
            geographic_distance=arch_config['geographic_distance'],
            _config=som_config._config,  # Pass raw config dict for _get_config access
            # Learning rate settings now from training_config
            initial_learning_rate=training_config['initial_learning_rate'],
            min_learning_rate=training_config['min_learning_rate'],
            max_learning_rate=training_config['max_learning_rate'],
            lr_increase_factor=training_config['lr_increase_factor'],
            lr_decrease_factor=training_config['lr_decrease_factor'],
            high_qe_lr_min=training_config['high_qe_lr_range'][0],
            high_qe_lr_max=training_config['high_qe_lr_range'][1],
            low_qe_lr_min=training_config['low_qe_lr_range'][0],
            low_qe_lr_max=training_config['low_qe_lr_range'][1],
            # Neighborhood from architecture (structural)
            neighborhood_function=arch_config['neighborhood_function'],
            initial_radius=arch_config['initial_radius'],
            final_radius=arch_config['final_radius'],
            radius_decay=arch_config['radius_decay'],
            # Convergence criteria from training_config
            geographic_coherence_threshold=training_config['geographic_coherence_threshold'],
            lr_stability_threshold=training_config['lr_stability_threshold'],
            qe_improvement_threshold=training_config['qe_improvement_threshold'],
            patience=training_config['patience'],
            max_epochs=max_epochs,
            min_valid_features=default_config.get('distance', {}).get('min_valid_features', 
                                                  default_config.get('distance_config', {}).get('min_valid_features', 2)),
            random_seed=kwargs.get('random_seed', 42),
            # Add performance settings
            chunk_size=kwargs.get('chunk_size', chunk_size),
            qe_sample_size=kwargs.get('qe_sample_size', qe_sample_size),
            use_parallel=kwargs.get('use_parallel', use_parallel),
            n_workers=kwargs.get('n_workers', n_workers),
            # Add batch training settings as a config dict
            batch_training=batch_config
        )
        
        # Add the new config parameters from training_config and memory_config
        for attr in ['lr_recovery_threshold', 'qe_poor_threshold', 'lr_recovery_target',
                    'significant_improvement', 'small_improvement', 'small_degradation',
                    'moderate_degradation', 'small_increase_factor', 'gentle_decrease_factor',
                    'high_qe_threshold', 'low_qe_threshold']:
            if attr in training_config:
                setattr(config, attr, training_config[attr])
        
        # Add memory config parameters
        for attr in ['qe_calc_chunk_size', 'prediction_chunk_size', 'vectorized_threshold']:
            if attr in memory_config:
                setattr(config, attr, memory_config[attr])
        
        # Add spatial metrics parameters
        spatial_metrics = default_config.get('spatial_metrics', {})
        for attr in ['coherence_sample_threshold', 'coherence_sample_size']:
            if attr in spatial_metrics:
                setattr(config, attr, spatial_metrics[attr])
        
        return config
    
    def _spatial_cross_validation(self, features: np.ndarray, 
                                coordinates: np.ndarray,
                                config: GeoSOMConfig,
                                **kwargs) -> Dict[str, Any]:
        """Perform spatial block cross-validation."""
        # Create spatial blocks
        validation_config = self.config.get('validation_config', {})
        n_folds = kwargs.get('cv_folds', validation_config.get('n_folds', 5))
        
        # Special case: n_folds=1 means no CV, use all data
        if n_folds == 1:
            logger.info("n_folds=1 specified, skipping cross-validation and using all data")
            # Create dummy CV results that indicate we used all data
            return {
                'fold_results': [{
                    'fold': 0,
                    'train_size': len(features),
                    'test_size': 0,
                    'test_qe': np.nan,  # No test set
                    'geographic_coherence': np.nan,  # No test set
                    'convergence_epoch': None,
                    'note': 'No CV performed - used all data for training'
                }],
                'mean_qe': np.nan,
                'std_qe': 0.0,
                'mean_geographic_coherence': np.nan,
                'std_geographic_coherence': 0.0,
                'cv_performed': False
            }
        
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
            try:
                self.progress_tracker.update_phase(f'cv_fold_{fold_idx + 1}', cv_fold=fold_idx + 1)
            except Exception as e:
                logger.warning(f"Failed to update progress tracker: {e}")
            
            # Create fold SOM
            fold_som = GeoSOMVLRSOM(config)
            
            # Create progress callback for this fold
            try:
                fold_callback = create_progress_callback(
                    self.progress_tracker, 
                    phase=f'cv_fold_{fold_idx + 1}'
                )
            except Exception as e:
                logger.warning(f"Failed to create progress callback: {e}")
                fold_callback = None
            
            # Train on fold
            logger.info(f"Starting training for CV fold {fold_idx + 1}/{n_folds}")
            fold_result = fold_som.train_batch(
                features[train_idx],
                coordinates[train_idx],
                progress_callback=fold_callback
            )
            logger.info(f"Completed training for CV fold {fold_idx + 1}/{n_folds}, converged at epoch {fold_result.convergence_epoch}")
            
            # Evaluate on test set
            logger.info(f"Calculating test QE for fold {fold_idx + 1}")
            test_qe = fold_som.calculate_quantization_error(
                features[test_idx],
                coordinates[test_idx]
            )
            logger.info(f"Test QE for fold {fold_idx + 1}: {test_qe:.6f}")
            
            # Calculate geographic coherence
            logger.info(f"Calculating predictions for fold {fold_idx + 1}")
            test_predictions = fold_som.predict(features[test_idx], coordinates[test_idx])
            logger.info(f"Predictions calculated, computing geographic coherence")
            
            # Sample for large test sets to avoid memory issues
            test_size = len(test_idx)
            if test_size > 30000:
                # Use 20,000 samples for large datasets - good balance of accuracy and computation
                sample_size = 20000
                logger.info(f"Test set has {test_size} samples, sampling {sample_size} for geographic coherence")
                sample_indices = np.random.choice(test_size, size=sample_size, replace=False)
                sampled_predictions = test_predictions[sample_indices]
                sampled_coords = coordinates[test_idx][sample_indices]
                geo_coherence = GeographicCoherence.morans_i(sampled_predictions, sampled_coords)
                logger.info(f"Geographic coherence (sampled) for fold {fold_idx + 1}: {geo_coherence:.4f}")
            elif test_size > 10000:
                # Use 10,000 samples for medium datasets
                sample_size = 10000
                logger.info(f"Test set has {test_size} samples, sampling {sample_size} for geographic coherence")
                sample_indices = np.random.choice(test_size, size=sample_size, replace=False)
                sampled_predictions = test_predictions[sample_indices]
                sampled_coords = coordinates[test_idx][sample_indices]
                geo_coherence = GeographicCoherence.morans_i(sampled_predictions, sampled_coords)
                logger.info(f"Geographic coherence (sampled) for fold {fold_idx + 1}: {geo_coherence:.4f}")
            else:
                geo_coherence = GeographicCoherence.morans_i(
                    test_predictions,
                    coordinates[test_idx]
                )
                logger.info(f"Geographic coherence for fold {fold_idx + 1}: {geo_coherence:.4f}")
            
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
            
            # Clean up resources after each fold
            try:
                fold_som.cleanup()
                logger.info(f"Completed cleanup for CV fold {fold_idx + 1}/{n_folds}")
            except Exception as e:
                logger.warning(f"Failed to cleanup fold {fold_idx + 1}: {e}")
            
            logger.info(f"Completed CV fold {fold_idx + 1}/{n_folds}, moving to next fold")
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
        
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
        
        # Geographic metrics with stratified sampling for large datasets
        n_samples = len(predictions)
        
        if n_samples > 100000:
            # Use stratified spatial sampling for very large datasets
            logger.info(f"Dataset has {n_samples} samples, using stratified spatial sampling (100k samples)")
            
            # Create spatial grid for stratification
            lon_bins = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), 50)
            lat_bins = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), 50)
            
            # Assign each point to a grid cell
            lon_idx = np.digitize(coordinates[:, 0], lon_bins) - 1
            lat_idx = np.digitize(coordinates[:, 1], lat_bins) - 1
            cell_idx = lon_idx * 50 + lat_idx
            
            # Sample proportionally from each cell
            sample_indices = []
            unique_cells = np.unique(cell_idx)
            samples_per_cell = max(1, 100000 // len(unique_cells))
            
            for cell in unique_cells:
                cell_points = np.where(cell_idx == cell)[0]
                if len(cell_points) > 0:
                    n_cell_samples = min(len(cell_points), samples_per_cell)
                    cell_samples = np.random.choice(cell_points, size=n_cell_samples, replace=False)
                    sample_indices.extend(cell_samples)
            
            # Ensure we have exactly 100k samples (or close to it)
            sample_indices = np.array(sample_indices)
            if len(sample_indices) > 100000:
                sample_indices = np.random.choice(sample_indices, size=100000, replace=False)
            
            sampled_predictions = predictions[sample_indices]
            sampled_coords = coordinates[sample_indices]
            geo_metrics = GeographicCoherence.geographic_cluster_quality(
                sampled_predictions, sampled_coords
            )
            logger.info(f"Geographic metrics calculated on {len(sample_indices)} stratified samples")
            
        elif n_samples > 50000:
            # Use 50k samples for medium-large datasets
            logger.info(f"Dataset has {n_samples} samples, sampling 50,000 for geographic metrics")
            sample_indices = np.random.choice(n_samples, size=50000, replace=False)
            sampled_predictions = predictions[sample_indices]
            sampled_coords = coordinates[sample_indices]
            geo_metrics = GeographicCoherence.geographic_cluster_quality(
                sampled_predictions, sampled_coords
            )
            
        else:
            # Use all data for smaller datasets
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
            'preprocessing_info': {
                'method': 'SharedPreprocessor',
                'normalization': self.preprocessor.normalization_method,
                'missing_strategy': self.preprocessor.missing_strategy
            },
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
                      cv_results: Optional[Dict], final_metrics: Dict,
                      reports: Dict, runtime: float) -> AnalysisResult:
        """Create analysis result object."""
        # Prepare result data
        result_data = {
            'labels': final_metrics['grid_coordinates'],
            'statistics': {
                'cv_results': cv_results if cv_results else {'skipped': True, 'reason': 'exploratory_analysis'},
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
                'preprocessing': f'{self.preprocessor.normalization_method}+{self.preprocessor.missing_strategy}'
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
    
    # NOTE: Removed misleading save_checkpoint_state() method
    # SOM checkpointing is handled directly in geo_som_core.py
    # It saves .npz files with SOM-specific data (weights, learning rate, radius)
    # This data is not useful for other methods (CP-WOPT needs factors, KMeans needs centroids)
    
    # NOTE: Removed misleading load_checkpoint_state() method
    # SOM checkpoint loading is handled directly in geo_som_core.py
    # The .npz checkpoint files are loaded when resume_checkpoint is set in config
    # This happens automatically when resume=True is passed to analyze()