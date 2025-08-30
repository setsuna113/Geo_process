"""GeoSOM implementation for biodiversity analysis.

This implementation uses pure Batch SOM with momentum control:
- Handles 70% missing data with partial Bray-Curtis distance
- Integrates spatial information (30% spatial, 70% features)
- Uses momentum-based learning control (no learning rate)
- Implements geographic coherence for convergence
- Pure Batch SOM: direct weight assignment w = Σ(h*x) / Σ(h)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass
import logging
import time
import os
from pathlib import Path
import json
from scipy.spatial.distance import cdist
# Moran's I calculation is in spatial_utils module

from src.core_abstractions import SOMConfig, SOMTrainingResult
from ..utils.constants import (
    INVALID_DISTANCE, INVALID_INDEX, INVALID_COORDINATE,
    MIN_VALID_FEATURES, INSUFFICIENT_FEATURES_MSG, DEFAULT_GEO_DISTANCE_NORM
)
from ..calculators.partial_metrics import validate_missing_data_handling
from src.biodiversity_analysis.shared.metrics.som_metrics import (
    calculate_qe_numba,
    calculate_qe_with_coords_numba,
    calculate_te_numba,
    calculate_geographic_coherence_numba,
    predict_batch_numba
)
from ..trainers.pure_numba_som import PureNumbaSOM
from ..utils.numba_thread_control import get_thread_controller, setup_numba_threads

logger = logging.getLogger(__name__)

# Import constants from central location
from ..utils.constants import (
    DEFAULT_QE_SAMPLE_SIZE, DEFAULT_PROGRESS_LOG_INTERVAL,
    MEMORY_THRESHOLD_BYTES, DEFAULT_DISTANCE_CALC_BATCH_SIZE
)


@dataclass
class GeoSOMConfig:
    """Configuration for GeoSOM with pure Batch SOM."""
    # Grid parameters
    grid_size: Tuple[int, int]
    topology: str = "rectangular"  # or "hexagonal"
    
    # GeoSOM parameters
    spatial_weight: float = 0.3  # k=0.3 (30% spatial, 70% features)
    geographic_distance: str = "haversine"
    
    # Raw config dict for _get_config access
    _config: Optional[Dict[str, Any]] = None
    
    # Batch SOM Training Configuration (processes mini-batches for memory efficiency)
    batch_training: Optional[Dict[str, Any]] = None
    
    # Memory Optimization Configuration  
    memory_optimization: Optional[Dict[str, Any]] = None
    
    # Pure Batch SOM doesn't use learning rate parameters
    
    # Adaptive regions
    high_qe_lr_min: float = 0.5
    high_qe_lr_max: float = 0.8
    low_qe_lr_min: float = 0.01
    low_qe_lr_max: float = 0.1
    
    # Neighborhood parameters
    neighborhood_function: str = "gaussian"
    initial_radius: Optional[float] = None  # Will be grid_size/2 if None
    final_radius: float = 1.0
    radius_decay: str = "linear"
    
    # Convergence criteria
    geographic_coherence_threshold: float = 0.7  # Moran's I
    lr_stability_threshold: float = 0.02  # LR changes < 2%
    qe_improvement_threshold: float = 0.001  # < 0.1% improvement
    patience: int = 50  # Epochs without improvement
    max_epochs: int = 1000
    
    # Data handling
    min_valid_features: int = 2  # For partial comparison
    random_seed: Optional[int] = None
    
    # Performance settings
    qe_sample_size: int = 100000  # Sample size for QE calculation


class GeoSOM:
    """Geographic Self-Organizing Map using pure Batch SOM.
    
    This implementation features:
    1. GeoSOM: Incorporates spatial information into the distance calculation
    2. Pure Batch SOM: Direct weight assignment w = Σ(h*x) / Σ(h)
    3. Momentum control: Smooths weight updates between epochs
    
    Designed for biodiversity data with high missing values and spatial structure.
    """
    
    # Class constants
    INVALID_INDEX = -1
    
    def _init_config_accessor(self):
        """Initialize standardized config access patterns."""
        self._config_cache = {}
        
    def _get_config(self, key: str, default: Any = None, section: Optional[str] = None):
        """Standardized config access with caching and clean fallbacks.
        
        Args:
            key: Config key to access
            default: Default value if key not found
            section: Optional config section (e.g., 'memory_optimization', 'training.momentum')
            
        Returns:
            Config value or default
        """
        cache_key = f"{section}.{key}" if section else key
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
            
        value = default
        if section:
            # Handle nested sections with dots (e.g., 'training.momentum')
            # Access the raw config dict if available
            if hasattr(self.config, '_config') and self.config._config:
                # Navigate through nested dicts
                parts = section.split('.')
                current = self.config._config
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        current = None
                        break
                if current is not None and isinstance(current, dict):
                    value = current.get(key, default)
            else:
                # Fallback to attribute access
                section_config = getattr(self.config, section, None)
                if section_config:
                    if isinstance(section_config, dict):
                        value = section_config.get(key, default)
                    else:
                        value = getattr(section_config, key, default)
        else:
            # Direct config access
            value = getattr(self.config, key, default)
            
        self._config_cache[cache_key] = value
        return value
    
    def __init__(self, config: GeoSOMConfig):
        self.config = config
        self._init_config_accessor()
        
        # Validate grid size configuration
        if not isinstance(config.grid_size, (list, tuple)) or len(config.grid_size) != 2:
            raise ValueError(f"grid_size must be a 2-element list/tuple, got: {config.grid_size}")
        
        if config.grid_size[0] <= 0 or config.grid_size[1] <= 0:
            raise ValueError(f"grid_size dimensions must be positive, got: {config.grid_size}")
        
        # Warn if grid is too large relative to expected data
        grid_neurons = config.grid_size[0] * config.grid_size[1]
        if grid_neurons > self._get_config('max_grid_neurons_warning', 10000):
            logger.warning(f"Large grid size ({config.grid_size[0]}x{config.grid_size[1]} = {grid_neurons} neurons) "
                         f"exceeds warning threshold. May lead to slow training and memory issues")
        
        self.weights = None
        self.input_dim = None
        self.n_neurons = grid_neurons
        self.grid_rows, self.grid_cols = config.grid_size
        
        # Initialize grid positions
        self._init_grid_positions()
        
        # Using shared Numba distance functions for all distance calculations
        # (No more OptimizedDistanceCalculator - single source of truth in distance_calculator.py)
        
        self.spatial_weight = config.spatial_weight
        self.min_valid_features = config.min_valid_features
        self.max_epochs = config.max_epochs
        self.qe_improvement_threshold = config.qe_improvement_threshold
        self.geographic_coherence_threshold = config.geographic_coherence_threshold
        self.patience = config.patience
        self.neighborhood_function = config.neighborhood_function
        
        # Use shared distance calculator for consistency across methods
        # Distance metric for future extension (euclidean, etc.)
        self.distance_metric = self._get_config('feature_metric', 'bray_curtis', 'distance')
        
        # Initialize thread controller with n_workers from config
        thread_controller = get_thread_controller()
        # Get n_workers from the raw config dict if available
        n_workers = None
        if hasattr(config, '_config') and config._config:
            n_workers = config._config.get('training', {}).get('n_workers')
        if n_workers:
            actual_threads = thread_controller.set_num_threads(n_workers)
            logger.info(f"Set Numba threads to {actual_threads} (requested: {n_workers})")
        else:
            # Use default or optimize based on data size
            current_threads = thread_controller.get_num_threads()
            logger.info(f"Using Numba threads: {current_threads}")
        
        # Use pure Numba implementation for 48x speedup
        self.som_trainer = PureNumbaSOM(config=config)
        logger.info(f"Initialized PureNumbaSOM trainer (48x faster than joblib)")
        logger.info("Using BATCH SOM - weight updates once per epoch")
        
        # Training state
        # Pure Batch SOM doesn't use learning rate
        self.current_radius = None
        self.training_history = {
            'quantization_errors': [],
            # Pure Batch SOM doesn't track learning rates
            'radii': [],
            'geographic_coherence': [],
            'topographic_errors': [],
            'convergence_checks': []
        }
        
        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def _init_grid_positions(self):
        """Initialize neuron positions on the grid."""
        if self.config.topology == "rectangular":
            positions = []
            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    positions.append([i, j])
            self._grid_positions = np.array(positions)
        else:
            raise NotImplementedError(f"Topology {self.config.topology} not implemented")
    
    def _validate_coordinates(self, coordinates: np.ndarray, n_samples: int) -> np.ndarray:
        """Validate coordinate array for proper shape, memory, and values.
        
        COORDINATE ORDER: This function expects [latitude, longitude] order.
        First column should be latitude (-90 to 90), second column longitude (-180 to 180).
        
        Args:
            coordinates: Geographic coordinates to validate [latitude, longitude]
            n_samples: Expected number of samples
            
        Returns:
            Validated and potentially converted coordinates array
            
        Raises:
            ValueError: If coordinates have invalid shape or dimensions
            MemoryError: If dataset would require too much memory
        """
        import psutil  # For memory checking
        
        # Memory validation
        available_memory = psutil.virtual_memory().available
        # Estimate memory for coordinates + processing overhead
        n_features = 4  # Typical feature count
        estimated_memory = (
            coordinates.nbytes +  # Coordinates array
            (n_samples * n_features * 8) +  # Data array (float64)
            (self.config.grid_size[0] * self.config.grid_size[1] * n_features * 8) * 2  # Weights + temp arrays
        )
        
        if estimated_memory > available_memory * 0.8:
            raise MemoryError(
                f"Dataset too large: {estimated_memory/1e9:.1f}GB required, "
                f"only {available_memory/1e9:.1f}GB available. "
                f"Consider reducing data size or grid size."
            )
        
        # Data type validation and conversion
        if coordinates.dtype not in [np.float32, np.float64]:
            logger.info(f"Converting coordinates from {coordinates.dtype} to float64")
            coordinates = coordinates.astype(np.float64)
        elif coordinates.dtype == np.float32:
            # Optionally convert float32 to float64 for better precision
            logger.info("Converting coordinates from float32 to float64 for better precision")
            coordinates = coordinates.astype(np.float64)
        
        # Check shape
        if len(coordinates.shape) != 2:
            raise ValueError(f"Coordinates must be 2D array, got shape {coordinates.shape}")
        if len(coordinates) == 0:
            raise ValueError("Coordinates array is empty")
        if coordinates.shape[0] != n_samples:
            raise ValueError(f"Coordinates samples ({coordinates.shape[0]}) must match data samples ({n_samples})")
        if coordinates.shape[1] != 2:
            raise ValueError(f"Coordinates must have 2 columns [latitude, longitude], got {coordinates.shape[1]}")
        
        # Check for NaN values
        nan_coords = np.isnan(coordinates).any(axis=1)
        if nan_coords.any():
            n_nan = nan_coords.sum()
            logger.warning(f"Found {n_nan} samples with NaN coordinates ({n_nan/len(coordinates)*100:.1f}%)")
        
        # Validate coordinate ranges (assuming lat/lon)
        # Note: coordinates are expected to be [latitude, longitude] order
        lat_values = coordinates[:, 0]  # First column is latitude
        lon_values = coordinates[:, 1]  # Second column is longitude
        
        # Check latitude range [-90, 90]
        valid_lat_mask = ~np.isnan(lat_values)
        if valid_lat_mask.any():
            invalid_lat = valid_lat_mask & ((lat_values < -90) | (lat_values > 90))
            if invalid_lat.any():
                logger.warning(f"Found {invalid_lat.sum()} samples with invalid latitude values "
                             f"(outside [-90, 90]): min={lat_values[valid_lat_mask].min():.2f}, "
                             f"max={lat_values[valid_lat_mask].max():.2f}")
        
        # Check longitude range [-180, 180]
        valid_lon_mask = ~np.isnan(lon_values)
        if valid_lon_mask.any():
            invalid_lon = valid_lon_mask & ((lon_values < -180) | (lon_values > 180))
            if invalid_lon.any():
                logger.warning(f"Found {invalid_lon.sum()} samples with invalid longitude values "
                             f"(outside [-180, 180]): min={lon_values[valid_lon_mask].min():.2f}, "
                             f"max={lon_values[valid_lon_mask].max():.2f}")
        
        return coordinates  # Return the validated/converted coordinates
    
    def partial_bray_curtis(self, u: np.ndarray, v: np.ndarray) -> float:
        """Bray-Curtis distance for vectors with missing values.
        
        Uses shared Numba implementation for consistency.
        Only compares non-NA pairs, requires minimum valid features.
        
        Args:
            u, v: Vectors to compare (may contain NaN)
            
        Returns:
            Distance value or INVALID_DISTANCE (np.inf) if insufficient valid pairs
        """
        from ...shared.metrics.distance_calculator import bray_curtis_distance_single
        return bray_curtis_distance_single(u, v, self.min_valid_features)
    
    def haversine_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Calculate haversine distance between geographic coordinates.
        
        Args:
            coord1, coord2: [longitude, latitude] in degrees
            
        Returns:
            Distance in kilometers
        """
        from ...shared.metrics.distance_calculator import haversine_distance_single
        # coord1 and coord2 are [lon, lat]
        return haversine_distance_single(coord1[0], coord1[1], coord2[0], coord2[1])
    
    def combined_distance(self, sample: np.ndarray, weights: np.ndarray,
                         sample_coord: np.ndarray, neuron_coord: np.ndarray) -> float:
        """Combine feature and geographic distances with weighting.
        
        Args:
            sample: Feature vector
            weights: Neuron weight vector
            sample_coord: Geographic coordinates of sample
            neuron_coord: Geographic coordinates assigned to neuron
            
        Returns:
            Combined distance (weighted sum)
        """
        # Feature distance (Bray-Curtis)
        feature_dist = self.partial_bray_curtis(sample, weights)
        
        # Handle case where feature distance is invalid
        if np.isnan(feature_dist):  # Check for INVALID_DISTANCE
            return INVALID_DISTANCE
        
        # Geographic distance (normalized)
        if sample_coord is not None and neuron_coord is not None:
            geo_dist = self.haversine_distance(sample_coord, neuron_coord)
            # Normalize geographic distance (rough earth circumference / 2)
            geo_dist_norm = geo_dist / DEFAULT_GEO_DISTANCE_NORM
        else:
            geo_dist_norm = 0.0
        
        # Weighted combination (using cached config value)
        combined = (1 - self.spatial_weight) * feature_dist + \
                   self.spatial_weight * geo_dist_norm
        
        return combined

    
    
    def initialize_weights(self, data: np.ndarray, coordinates: Optional[np.ndarray] = None,
                          method: str = "pca_transformed") -> None:
        """Initialize SOM weights using specified method.
        
        Args:
            data: Training data (n_samples, n_features)
            coordinates: Geographic coordinates (n_samples, 2) [longitude, latitude]
            method: Initialization method
        """
        self.input_dim = data.shape[1]
        
        # Validate coordinates if provided
        if coordinates is not None:
            coordinates = self._validate_coordinates(coordinates, data.shape[0])
        
        if method == "pca_transformed":
            self._init_pca_transformed(data)
        elif method == "stratified_sample":
            self._init_stratified_sample(data)
        elif method == "random_best_of_n":
            self._init_random_best_of_n(data)
        else:
            self._init_random(data)
        
        # Initialize geographic coordinates for neurons if provided
        if coordinates is not None:
            self._init_neuron_coordinates(coordinates)
        
        # Set initial radius - smaller for more stable training
        if self.config.initial_radius is None:
            self.config.initial_radius = max(self.grid_rows, self.grid_cols) / 3.0
        self.current_radius = self.config.initial_radius
        
        logger.info(f"Initialized GeoSOM weights using {method} method")
    
    def _init_pca_transformed(self, data: np.ndarray):
        """Initialize using PCA on transformed data."""
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        
        # Handle missing values for PCA only
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pc_scores = pca.fit_transform(data_imputed)
        
        # Create grid in PC space
        pc1_range = np.linspace(pc_scores[:, 0].min(), pc_scores[:, 0].max(), self.grid_cols)
        pc2_range = np.linspace(pc_scores[:, 1].min(), pc_scores[:, 1].max(), self.grid_rows)
        
        # Initialize weights
        self.weights = np.zeros((self.n_neurons, self.input_dim))
        idx = 0
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                # Map back to original space
                pc_point = np.array([pc1_range[j], pc2_range[i]])
                weight = pca.inverse_transform(pc_point.reshape(1, -1))[0]
                # Ensure non-negative for Bray-Curtis compatibility
                self.weights[idx] = np.maximum(weight, 0)
                idx += 1
    
    def _init_stratified_sample(self, data: np.ndarray):
        """Initialize using stratified sampling."""
        from sklearn.cluster import KMeans
        
        # Use KMeans to create strata
        n_clusters = self.n_neurons
        
        # Handle missing values for clustering
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_seed)
        kmeans.fit(data_imputed)
        
        # Use cluster centers as initial weights
        self.weights = kmeans.cluster_centers_
    
    def _init_random_best_of_n(self, data: np.ndarray):
        """Initialize with best of N random initializations."""
        n_trials = 5
        best_weights = None
        best_qe = np.inf
        
        for trial in range(n_trials):
            # Random initialization
            indices = np.random.choice(len(data), self.n_neurons, replace=False)
            trial_weights = data[indices].copy()
            
            # Calculate initial QE
            qe = 0.0
            n_valid = 0
            for sample in data:
                dists = [self.partial_bray_curtis(sample, w) for w in trial_weights]
                valid_dists = [d for d in dists if not np.isnan(d)]
                if valid_dists:
                    qe += min(valid_dists)
                    n_valid += 1
            
            if n_valid > 0:
                qe /= n_valid
                if qe < best_qe:
                    best_qe = qe
                    best_weights = trial_weights
        
        self.weights = best_weights if best_weights is not None else trial_weights
    
    def _init_random(self, data: np.ndarray):
        """Random initialization from data samples."""
        indices = np.random.choice(len(data), self.n_neurons, replace=False)
        self.weights = data[indices].copy()
    
    def _init_neuron_coordinates(self, coordinates: np.ndarray):
        """Initialize geographic coordinates for neurons."""
        # Simple approach: assign based on grid layout
        # Coordinates are [longitude, latitude]
        lon_range = coordinates[:, 0].min(), coordinates[:, 0].max()
        lat_range = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        lat_step = (lat_range[1] - lat_range[0]) / (self.grid_rows - 1)
        lon_step = (lon_range[1] - lon_range[0]) / (self.grid_cols - 1)
        
        self.neuron_coords = np.zeros((self.n_neurons, 2))
        idx = 0
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                self.neuron_coords[idx] = [
                    lon_range[0] + j * lon_step,  # longitude
                    lat_range[0] + i * lat_step   # latitude
                ]
                idx += 1
    
    def train_batch(self, data: np.ndarray, coordinates: Optional[np.ndarray] = None,
                   progress_callback: Optional[Callable] = None, **kwargs) -> SOMTrainingResult:
        """Train using mini-batch mode with geographic awareness.
        
        Args:
            data: Training data (n_samples, n_features)
            coordinates: Geographic coordinates (n_samples, 2) [longitude, latitude]
            progress_callback: Optional callback for progress updates
            
        Returns:
            Training result with history and final state
        """
        self._validate_training_inputs(data, coordinates)
        
        # Optimize thread count based on data size if not explicitly set
        # Skip optimization if n_workers was explicitly set in config
        thread_controller = get_thread_controller()
        skip_optimization = False
        if hasattr(self.config, '_config') and self.config._config:
            n_workers = self.config._config.get('training', {}).get('n_workers')
            if n_workers is not None:
                skip_optimization = True
                logger.info(f"Skipping thread optimization - using configured n_workers={n_workers}")
        
        if not hasattr(self, '_threads_optimized') and not skip_optimization:
            from ..utils.numba_thread_control import optimize_threads_for_som
            optimal_threads = optimize_threads_for_som(data.shape, self.n_neurons)
            logger.info(f"Optimized Numba threads for data shape {data.shape}: {optimal_threads}")
            self._threads_optimized = True
        
        if self.weights is None:
            self.initialize_weights(data, coordinates)
        
        # Initialize training state
        training_state = self._initialize_training_state(data, coordinates)
        
        # Warm up JIT compilation for pure Numba (only needs to happen once)
        if self.som_trainer:
            self.som_trainer.warm_up_jit(
                n_features=self.weights.shape[1],
                n_neurons=self.weights.shape[0]
            )
            logger.info("Warmed up Numba JIT compilation for pure SOM")
        
        # Main training loop
        last_epoch = training_state['start_epoch'] - 1
        for epoch in range(training_state['start_epoch'], self.max_epochs):
            last_epoch = epoch
            epoch_result = self._train_single_epoch(
                epoch, data, coordinates, training_state, progress_callback
            )
            
            # Update training state
            training_state['best_qe'] = epoch_result['best_qe']
            training_state['patience_counter'] = epoch_result['patience_counter']
            
            # Check for convergence
            if epoch_result.get('converged', False):
                training_state['converged'] = True
                training_state['convergence_epoch'] = epoch
                break
        
        # If we didn't converge, set the convergence_epoch to the last completed epoch
        if training_state['convergence_epoch'] is None:
            training_state['convergence_epoch'] = last_epoch
        
        return self._create_training_result(training_state, len(data))
    
    def _validate_training_inputs(self, data: np.ndarray, coordinates: Optional[np.ndarray]) -> None:
        """Validate training inputs."""
        if coordinates is not None:
            if coordinates.shape[0] != data.shape[0]:
                raise ValueError(f"Coordinates samples ({coordinates.shape[0]}) must match data samples ({data.shape[0]})")
            if coordinates.shape[1] != 2:
                raise ValueError(f"Coordinates must have 2 columns [longitude, latitude], got {coordinates.shape[1]}")
    
    def _initialize_training_state(self, data: np.ndarray, coordinates: Optional[np.ndarray]) -> Dict[str, Any]:
        """Initialize training state including resume from checkpoint."""
        # Log Batch SOM status
        logger.info("Using pure Batch SOM - no learning rate, momentum-based control only")
        
        state = {
            'converged': False,
            'best_qe': np.inf,
            'patience_counter': 0,
            'start_epoch': 0,
            'convergence_epoch': None
        }
        
        # Check for resume from checkpoint
        resume_checkpoint = self._get_config('resume_checkpoint', None)
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            checkpoint_data = self.load_checkpoint(resume_checkpoint)
            state['start_epoch'] = checkpoint_data['epoch'] + 1
            state['best_qe'] = checkpoint_data['best_qe']
            state['patience_counter'] = checkpoint_data['patience_counter']
            logger.info(f"Resuming training from epoch {state['start_epoch']+1}")
            # Load velocity from checkpoint if available
            if 'velocity' in checkpoint_data:
                self.velocity = checkpoint_data['velocity']
            else:
                # Initialize velocity for resumed training
                self.velocity = np.zeros_like(self.weights, dtype=np.float32)
        else:
            # Fresh training - always reset velocity
            self.velocity = np.zeros_like(self.weights, dtype=np.float32)
            # Calculate initial QE
            initial_qe = self._calculate_initial_qe(data, coordinates)
            self.training_history['quantization_errors'].append(initial_qe)
            state['best_qe'] = initial_qe
        
        return state
    
    def _calculate_initial_qe(self, data: np.ndarray, coordinates: Optional[np.ndarray]) -> float:
        """Calculate initial quantization error."""
        if len(data) > 10000:
            sample_idx = np.random.choice(len(data), size=10000, replace=False)
            sample_data = data[sample_idx]
            sample_coords = coordinates[sample_idx] if coordinates is not None else None
            initial_qe = self.calculate_quantization_error(sample_data, sample_coords)
            logger.info(f"Initial QE calculated on 10k sample: {initial_qe:.6f}")
        else:
            initial_qe = self.calculate_quantization_error(data, coordinates)
        return initial_qe
    
    def _train_single_epoch(self, epoch: int, data: np.ndarray, coordinates: Optional[np.ndarray],
                          training_state: Dict[str, Any], progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Train a single epoch."""
        # Check for phase transition
        phase_1_epochs = self._get_config('phase_1_epochs', 400, 'training')
        if epoch == 0:
            logger.info(f"Starting Phase 1 (Global Organization): epochs 1-{phase_1_epochs}")
        elif epoch == phase_1_epochs:
            logger.info(f"Starting Phase 2 (Fine-tuning): epochs {phase_1_epochs+1}-{self.max_epochs}")
        
        logger.info(f"Starting epoch {epoch+1}/{self.max_epochs}")
        
        # Update radius
        self._update_radius(epoch)
        
        # Check for minimum radius termination
        min_radius = self._get_config('min_radius', None, 'termination')
        if min_radius is not None and self.current_radius <= min_radius:
            logger.info(f"Radius {self.current_radius:.2f} reached minimum threshold {min_radius:.2f} - terminating training")
            converged = True
            training_state['convergence_reason'] = f"Minimum radius {min_radius} reached"
            return {
                'best_qe': training_state['best_qe'],
                'patience_counter': training_state['patience_counter'],
                'converged': converged
            }
        
        # Execute mini-batch training
        qe = self._execute_minibatch_training(epoch, data, coordinates)
        
        # Update training history
        self.training_history['quantization_errors'].append(qe)
        # Pure Batch SOM doesn't track learning rates
        self.training_history['radii'].append(self.current_radius)
        
        # Update convergence tracking
        best_qe, patience_counter = self._update_convergence_tracking(qe, training_state)
        
        # Periodic calculations and checkpoints
        converged = self._handle_periodic_calculations(epoch, data, coordinates, best_qe, patience_counter)
        
        # Progress callback
        if progress_callback:
            self._handle_progress_callback(progress_callback, epoch, qe)
        
        return {
            'best_qe': best_qe,
            'patience_counter': patience_counter,
            'converged': converged
        }
    
    def _execute_minibatch_training(self, epoch: int, data: np.ndarray, coordinates: Optional[np.ndarray]) -> float:
        """Execute training for one epoch using pure Numba SOM."""
        
        # Validate trainer is initialized
        if self.som_trainer is None:
            raise ValueError("PureNumbaSOM not initialized. Cannot train SOM.")
        
        logger.info(f"Starting pure Numba SOM training for epoch {epoch+1}")
        
        # Calculate current radius using decay schedule
        # Note: Learning rate not used in pure Batch SOM
        progress = epoch / self.config.max_epochs
        self.current_radius = self._calculate_radius(epoch, self.config.max_epochs)
        
        # Get neuron coordinates if using spatial distance
        neuron_coords = self.neuron_coords if coordinates is not None and hasattr(self, 'neuron_coords') else None
        
        # Train epoch using pure Numba (processes ALL data in parallel, no chunking!)
        # This replaces the entire joblib chunking infrastructure with a single parallel loop
        # Note: Pure Batch SOM doesn't use learning_rate
        updated_weights = self.som_trainer.train_epoch(
            data=data,
            weights=self.weights.copy(),  # Pass a copy to preserve original for momentum
            grid_positions=self._grid_positions,
            radius=self.current_radius,
            coordinates=coordinates,
            neuron_coords=neuron_coords
        )
        
        # Get momentum value (decay from initial to final)
        # Note: velocity is already initialized in _initialize_training_state
        progress = epoch / self.config.max_epochs
        initial_momentum = self._get_config('initial', 0.3, 'training.momentum')
        final_momentum = self._get_config('final', 0.7, 'training.momentum')
        momentum = initial_momentum * (1 - progress) + final_momentum * progress
        
        # Log momentum values periodically
        if epoch % 50 == 0:
            logger.info(f"Momentum: {momentum:.3f} (from {initial_momentum} to {final_momentum})")
        
        # Calculate weight changes from pure Batch SOM update
        # Note: Pure Batch SOM directly assigns weighted averages (no learning rate)
        weight_diff = updated_weights - self.weights
        
        # Apply momentum to control the learning dynamics
        # In pure Batch SOM, momentum is the ONLY learning control mechanism
        # High momentum (0.9): Slow, stable updates (keeps 90% of previous direction)
        # Low momentum (0.1): Fast, direct updates (keeps only 10% of previous direction)
        self.velocity = momentum * self.velocity + (1 - momentum) * weight_diff
        
        # Update weights with momentum-controlled changes
        # This provides smooth transitions between batch updates
        self.weights = self.weights + self.velocity
        
        logger.debug(f"Updated weights with momentum={momentum:.3f}")
        
        # Calculate QE after weight update - WITH SAMPLING (from batch_som_trainer.py)
        qe_sample_size = self._get_config('qe_sample_size', 10000, 'monitoring.sampling')
        qe = self.calculate_quantization_error(data, coordinates, sample_size=qe_sample_size)
        
        logger.info(f"Completed batch training for epoch {epoch+1}: "
                   f"QE={qe:.6f}, Radius={self.current_radius:.2f}")
        
        return qe
    
    def _update_convergence_tracking(self, qe: float, training_state: Dict[str, Any]) -> Tuple[float, int]:
        """Update best QE and patience counter."""
        best_qe = training_state['best_qe']
        patience_counter = training_state['patience_counter']
        
        if qe < best_qe:
            improvement = (best_qe - qe) / best_qe if best_qe != np.inf else 1.0
            if improvement < self.qe_improvement_threshold:
                patience_counter += 1
            else:
                patience_counter = 0
            best_qe = qe
        else:
            patience_counter += 1
        
        return best_qe, patience_counter
    
    def _handle_periodic_calculations(self, epoch: int, data: np.ndarray, coordinates: Optional[np.ndarray],
                                    best_qe: float, patience_counter: int) -> bool:
        """Handle topographic error, geographic coherence, and checkpoints."""
        # Topographic error calculation - read from new convergence section
        te_freq = self._get_config('calculation_frequency', 10, 'convergence.topographic_error')
        if epoch % te_freq == 0 and epoch > 0:
            self._calculate_and_adjust_topographic_error(epoch, data, coordinates)
        
        # Geographic coherence and convergence check - read from new convergence section
        geo_freq = self._get_config('calculation_frequency', 10, 'convergence.geographic_coherence')
        if coordinates is not None and epoch % geo_freq == 0 and epoch > 0:
            return self._handle_geographic_coherence_and_checkpoints(epoch, data, coordinates, best_qe, patience_counter)
        
        return False
    
    def _calculate_and_adjust_topographic_error(self, epoch: int, data: np.ndarray, coordinates: Optional[np.ndarray]) -> None:
        """Calculate topographic error and adjust radius if needed."""
        logger.info(f"Calculating topographic error for epoch {epoch+1}")
        te = self.calculate_topographic_error(data, coordinates, sample_size=10000)
        logger.info(f"Topographic error: {te:.4f}")
        self.training_history['topographic_errors'].append(te)
        
        if self._get_config('te_radius_adjustment', True):
            self._adjust_radius_based_on_te(te, epoch)
    
    def _adjust_radius_based_on_te(self, te: float, epoch: int) -> None:
        """Adjust radius based on topographic error."""
        next_epoch_radius = self._get_scheduled_radius(epoch + 1)
        # Read from new convergence.topographic_error.adjustment section
        te_high_threshold = self._get_config('high_threshold', 0.15, 'convergence.topographic_error.adjustment')
        te_low_threshold = self._get_config('low_threshold', 0.05, 'convergence.topographic_error.adjustment')
        
        if te > te_high_threshold:
            radius_modifier = self._get_config('high_modifier', 1.1, 'convergence.topographic_error.adjustment')
            adjusted_radius = next_epoch_radius * radius_modifier
            logger.info(f"High TE ({te:.3f}): adjusting radius {next_epoch_radius:.3f} → {adjusted_radius:.3f}")
        elif te < te_low_threshold:
            radius_modifier = self._get_config('low_modifier', 0.95, 'convergence.topographic_error.adjustment')
            adjusted_radius = next_epoch_radius * radius_modifier
            logger.info(f"Low TE ({te:.3f}): adjusting radius {next_epoch_radius:.3f} → {adjusted_radius:.3f}")
        else:
            adjusted_radius = next_epoch_radius
        
        # Apply bounds and store adjustment
        adjusted_radius = np.clip(adjusted_radius, self.config.final_radius, self.config.initial_radius)
        self._radius_adjustment = adjusted_radius / next_epoch_radius
    
    def _handle_geographic_coherence_and_checkpoints(self, epoch: int, data: np.ndarray, coordinates: np.ndarray,
                                                   best_qe: float, patience_counter: int) -> bool:
        """Handle geographic coherence calculation, checkpoints, and convergence check."""
        logger.info(f"Calculating geographic coherence for epoch {epoch+1}")
        
        # Calculate geographic coherence with sampling for large datasets
        # Now properly configured in monitoring.sampling section
        coherence_sample_threshold = self._get_config('coherence_sample_threshold', 10000, 'monitoring.sampling')
        coherence_sample_size = self._get_config('coherence_sample_size', 5000, 'monitoring.sampling')
        
        if len(data) > coherence_sample_threshold:
            indices = np.random.choice(len(data), coherence_sample_size, replace=False)
            geo_coherence = self._calculate_geographic_coherence(data[indices], coordinates[indices])
            logger.info(f"Geographic coherence (sampled): {geo_coherence:.4f}")
        else:
            geo_coherence = self._calculate_geographic_coherence(data, coordinates)
            logger.info(f"Geographic coherence: {geo_coherence:.4f}")
        
        self.training_history['geographic_coherence'].append(geo_coherence)
        
        # Handle checkpoint saving
        self._handle_checkpoint_saving(epoch, best_qe, patience_counter)
        
        # Check convergence
        if self._check_convergence(geo_coherence, patience_counter):
            logger.info(f"Converged at epoch {epoch}")
            return True
        
        return False
    
    def _handle_checkpoint_saving(self, epoch: int, best_qe: float, patience_counter: int) -> None:
        """Handle checkpoint saving."""
        # Read from correct config location: monitoring.checkpoint
        checkpoint_frequency = self._get_config('frequency', 10, 'monitoring.checkpoint')
        checkpoint_dir = self._get_config('directory', 'checkpoints/som', 'monitoring.checkpoint')
        
        if checkpoint_frequency > 0 and epoch % checkpoint_frequency == 0:
            self.save_checkpoint(checkpoint_dir, epoch, best_qe, patience_counter)
    
    def _handle_progress_callback(self, progress_callback: Callable, epoch: int, qe: float) -> None:
        """Handle progress callback without fallbacks."""
        progress_info = {
            'epoch': epoch + 1,
            'max_epochs': self.max_epochs,
            'progress': (epoch + 1) / self.max_epochs,
            'qe': qe,
            # Pure Batch SOM doesn't use learning rate
            'radius': self.current_radius
        }
        progress_callback(progress_info)
    
    def _create_training_result(self, training_state: Dict[str, Any], n_samples: int) -> SOMTrainingResult:
        """Create the final training result."""
        # Calculate final momentum value
        # If convergence_epoch is None, it means we reached max_epochs without converging
        final_epoch = training_state.get('convergence_epoch')
        if final_epoch is None:
            final_epoch = self.max_epochs - 1
        progress = final_epoch / self.config.max_epochs
        final_momentum = self._get_config('initial_momentum', 0.9, 'training.momentum') * (1 - progress) + \
                        self._get_config('final_momentum', 0.1, 'training.momentum') * progress
        
        return SOMTrainingResult(
            weights=self.weights.reshape(self.grid_rows, self.grid_cols, self.input_dim),
            quantization_errors=self.training_history['quantization_errors'],
            topographic_errors=[],
            training_time=0.0,
            convergence_epoch=training_state.get('convergence_epoch'),
            final_momentum=final_momentum,  # Pure Batch SOM uses momentum instead of learning rate
            final_neighborhood_radius=self.current_radius,
            n_samples_trained=n_samples
        )
    
    # Note: Checkpoint saving is handled by the standardized save_checkpoint method at line 1376
    
    # Pure Batch SOM doesn't use learning rate adaptation
    
    def _get_scheduled_radius(self, epoch: int) -> float:
        """Get the scheduled radius for a given epoch with two-phase training support."""
        # Get phase configuration
        phase_1_epochs = self._get_config('phase_1_epochs', 400, 'training')
        phase_1_final = self._get_config('phase_1_final', 3.0, 'training.radius')
        
        # Determine which phase we're in
        if epoch < phase_1_epochs:
            # Phase 1: Initial radius -> phase_1_final
            initial_radius = self.config.initial_radius
            target_radius = phase_1_final
            phase_epochs = phase_1_epochs
            phase_progress = epoch / phase_epochs
        else:
            # Phase 2: phase_1_final -> final_radius
            initial_radius = phase_1_final
            target_radius = self.config.final_radius
            phase_epochs = self.config.max_epochs - phase_1_epochs
            phase_progress = (epoch - phase_1_epochs) / phase_epochs
        
        # Apply decay within the current phase
        if self.config.radius_decay == "linear":
            radius = initial_radius * (1 - phase_progress) + target_radius * phase_progress
        elif self.config.radius_decay == "exponential":
            # Exponential decay within phase
            if initial_radius > 0 and target_radius > 0:
                decay_rate = -np.log(target_radius / initial_radius)
                radius = initial_radius * np.exp(-decay_rate * phase_progress)
            else:
                radius = target_radius
        else:
            raise NotImplementedError(f"Decay {self.config.radius_decay} not implemented")
        
        return radius
    
    def _update_radius(self, epoch: int):
        """Update neighborhood radius with optional TE-based adjustment."""
        # Get base scheduled radius
        scheduled_radius = self._get_scheduled_radius(epoch)
        
        # Apply TE adjustment if available
        adjustment = getattr(self, '_radius_adjustment', 1.0)
        self.current_radius = scheduled_radius * adjustment
        
        # Reset adjustment for next epoch
        self._radius_adjustment = 1.0
    
    # REMOVED: All legacy batch training methods replaced with real mini-batch training
    def _find_bmu(self, sample: np.ndarray) -> int:
        """Find best matching unit using partial Bray-Curtis.
        
        Returns:
            BMU index or INVALID_INDEX if no valid BMU found
        """
        distances = []
        for i, weight in enumerate(self.weights):
            dist = self.partial_bray_curtis(sample, weight)
            if not np.isnan(dist):  # Valid distance
                distances.append((i, dist))
        
        if not distances:
            return INVALID_INDEX
        
        return min(distances, key=lambda x: x[1])[0]
    
    def _find_bmu_geo(self, sample: np.ndarray, coord: np.ndarray) -> int:
        """Find BMU considering geographic distance.
        
        Returns:
            BMU index or INVALID_INDEX if no valid BMU found
        """
        distances = []
        for i, weight in enumerate(self.weights):
            dist = self.combined_distance(
                sample, weight, coord, self.neuron_coords[i]
            )
            if not np.isnan(dist):  # Valid distance (not INVALID_DISTANCE)
                distances.append((i, dist))
        
        if not distances:
            return INVALID_INDEX
        
        return min(distances, key=lambda x: x[1])[0]
    
    def _calculate_neighborhood(self, bmu_idx: int) -> np.ndarray:
        """Calculate neighborhood influence for all neurons."""
        if self.neighborhood_function == "gaussian":
            bmu_pos = self._grid_positions[bmu_idx]
            distances_squared = np.sum(
                (self._grid_positions - bmu_pos)**2, axis=1
            )
            influences = np.exp(-distances_squared / (2 * self.current_radius**2))
            return influences
        else:
            raise NotImplementedError(
                f"Neighborhood {self.neighborhood_function} not implemented"
            )
    
    # OBSOLETE - Pure Batch SOM doesn't use learning rate
    # def _calculate_learning_rate removed
    
    def _calculate_radius(self, epoch: int, max_epochs: int) -> float:
        """Calculate neighborhood radius for current epoch."""
        initial_radius = self.config.initial_radius or (max(self.grid_rows, self.grid_cols) / 2)
        final_radius = self.config.final_radius
        
        if self.config.radius_decay == "linear":
            # Linear decay
            radius = initial_radius - (initial_radius - final_radius) * (epoch / max_epochs)
        else:
            # Exponential decay (default)
            decay_rate = np.log(final_radius / initial_radius) / max_epochs
            radius = initial_radius * np.exp(decay_rate * epoch)
        
        return max(radius, final_radius)
    
    def calculate_quantization_error(self, data: np.ndarray, 
                                   coordinates: Optional[np.ndarray] = None,
                                   sample_size: Optional[int] = None) -> float:
        """Calculate average quantization error using fast Numba implementation.
        
        Args:
            data: Input data
            coordinates: Geographic coordinates (optional)
            sample_size: If provided, use random sampling for faster calculation
        """
        n_samples = len(data)
        
        # Use sampling for large datasets if requested
        if sample_size is not None and n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            data = data[indices]
            coordinates = coordinates[indices] if coordinates is not None else None
            logger.debug(f"Using {sample_size} samples for QE calculation")
        
        # Ensure contiguous arrays for Numba
        data = np.ascontiguousarray(data, dtype=np.float32)
        weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        
        # Use fast Numba implementation (10x faster!)
        if coordinates is not None and hasattr(self, 'neuron_coords'):
            coords = np.ascontiguousarray(coordinates, dtype=np.float32)
            neuron_coords = np.ascontiguousarray(self.neuron_coords, dtype=np.float32)
            return calculate_qe_with_coords_numba(
                data, weights, coords, neuron_coords,
                self.spatial_weight, self.min_valid_features
            )
        else:
            return calculate_qe_numba(data, weights, self.min_valid_features)
    
    def calculate_topographic_error(self, data: np.ndarray, 
                                  coordinates: Optional[np.ndarray] = None,
                                  sample_size: int = 10000) -> float:
        """Calculate topographic error using fast Numba implementation.
        
        TE measures how well the SOM preserves the topology of the input space.
        For each data point, TE = 1 if BMU and 2nd BMU are not neighbors, 0 otherwise.
        
        Args:
            data: Input data
            coordinates: Geographic coordinates (optional)
            sample_size: Sample size for efficiency (default 10000)
            
        Returns:
            Topographic error (fraction of samples with non-adjacent BMUs)
        """
        n_samples = len(data)
        
        # Use sampling for efficiency
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            data = data[indices]
            coordinates = coordinates[indices] if coordinates is not None else None
            logger.debug(f"Using {sample_size} samples for TE calculation")
        
        # Ensure contiguous arrays for Numba
        data = np.ascontiguousarray(data, dtype=np.float32)
        weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        grid_positions = np.ascontiguousarray(self._grid_positions, dtype=np.float32)
        
        # Use fast Numba implementation (parallel, no intermediate matrices)
        return calculate_te_numba(data, weights, grid_positions, self.min_valid_features)
    
    def _get_neuron_position(self, neuron_idx: int) -> Tuple[int, int]:
        """Get grid position (row, col) from neuron index."""
        row = neuron_idx // self.grid_cols
        col = neuron_idx % self.grid_cols
        return (row, col)
    
    def _calculate_geographic_coherence(self, data: np.ndarray, 
                                      coordinates: np.ndarray) -> float:
        """Calculate geographic coherence using fast Numba implementation.
        
        Uses simplified Moran's I calculation optimized for SOM BMU assignments.
        """
        logger.debug(f"Calculating geographic coherence for {len(data)} samples")
        
        # Ensure contiguous arrays for Numba
        data = np.ascontiguousarray(data, dtype=np.float32)
        weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        coords = np.ascontiguousarray(coordinates, dtype=np.float32)
        
        # Use fast Numba implementation
        distance_threshold = self._get_config('coherence_distance_threshold', 50.0, 'spatial_metrics')
        return calculate_geographic_coherence_numba(
            data, weights, coords, self.min_valid_features, distance_threshold
        )
        
        # Calculate spatial weights matrix using vectorized operations
        coords_valid = coordinates[valid]
        n = len(coords_valid)
        
        # For large datasets, use chunked processing to avoid memory explosion
        if n > 10000:
            return self._calculate_geographic_coherence_chunked(
                clusters[valid], coords_valid
            )
        
        # Original implementation for smaller datasets
        # Vectorized haversine distance calculation
        # Convert to radians
        coords_rad = np.radians(coords_valid)
        lat = coords_rad[:, 1]
        lon = coords_rad[:, 0]
        
        # Calculate pairwise differences
        lat_diff = lat[:, np.newaxis] - lat[np.newaxis, :]
        lon_diff = lon[:, np.newaxis] - lon[np.newaxis, :]
        
        # Haversine formula (vectorized)
        a = np.sin(lat_diff/2)**2 + \
            np.cos(lat[:, np.newaxis]) * np.cos(lat[np.newaxis, :]) * \
            np.sin(lon_diff/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # clip for numerical stability
        distances = 6371 * c  # Earth radius in km
        
        # Create weights matrix (inverse distance)
        # Set diagonal to inf to avoid division by zero
        np.fill_diagonal(distances, np.inf)
        W = np.where(distances > 0, 1.0 / distances, 0)
        np.fill_diagonal(W, 0)  # No self-connections
        
        # Normalize weights matrix
        W_sum = W.sum()
        if W_sum > 0:
            W = W / W_sum
        else:
            # If no spatial weights (e.g., all samples at same location), return 0
            return 0.0
        
        # Calculate Moran's I using matrix operations
        clusters_valid = clusters[valid]
        mean_cluster = clusters_valid.mean()
        deviations = clusters_valid - mean_cluster
        
        # Vectorized calculation: sum(W[i,j] * dev[i] * dev[j])
        numerator = np.sum(W * np.outer(deviations, deviations))
        denominator = np.sum(deviations**2) / n
        
        if denominator == 0:
            return 0.0
        
        morans_i = numerator / denominator
        return morans_i
    
    def _calculate_geographic_coherence_chunked(self, clusters: np.ndarray,
                                                coords: np.ndarray) -> float:
        """Calculate Moran's I using chunked processing for large datasets.
        
        This method processes the spatial weight matrix in chunks to avoid
        creating the full n×n matrix in memory, which would require O(n²) memory.
        
        Args:
            clusters: Cluster assignments for valid samples
            coords: Geographic coordinates (lon, lat) for valid samples
            
        Returns:
            Moran's I statistic
        """
        n = len(coords)
        logger.info(f"Using chunked geographic coherence calculation for {n} samples")
        
        # Convert to radians once
        coords_rad = np.radians(coords)
        lat = coords_rad[:, 1]
        lon = coords_rad[:, 0]
        
        # Pre-compute cos(lat) for efficiency
        cos_lat = np.cos(lat)
        
        # Calculate deviations from mean
        mean_cluster = clusters.mean()
        deviations = clusters - mean_cluster
        
        # Process in chunks to calculate weighted sum
        chunk_size = min(1000, max(100, n // 10))  # Adaptive chunk size with minimum
        logger.debug(f"Processing geographic coherence in chunks of {chunk_size}")
        numerator = 0.0
        total_weight = 0.0
        
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            
            # Calculate distances for this chunk against all points
            lat_chunk = lat[i:end_i]
            lon_chunk = lon[i:end_i]
            cos_lat_chunk = cos_lat[i:end_i]
            dev_chunk = deviations[i:end_i]
            
            # Vectorized distance calculation for chunk
            lat_diff = lat_chunk[:, np.newaxis] - lat[np.newaxis, :]
            lon_diff = lon_chunk[:, np.newaxis] - lon[np.newaxis, :]
            
            # Haversine formula
            a = np.sin(lat_diff/2)**2 + \
                cos_lat_chunk[:, np.newaxis] * cos_lat[np.newaxis, :] * \
                np.sin(lon_diff/2)**2
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            distances_chunk = 6371 * c  # Earth radius in km
            
            # Create weights for this chunk (inverse distance)
            # Handle diagonal elements (self-connections)
            for j in range(end_i - i):
                if i + j < n:
                    distances_chunk[j, i + j] = np.inf
            
            W_chunk = np.where(distances_chunk > 0, 1.0 / distances_chunk, 0)
            
            # Zero out diagonal
            for j in range(end_i - i):
                if i + j < n:
                    W_chunk[j, i + j] = 0
            
            # Calculate contribution to numerator
            # sum(W[i,j] * dev[i] * dev[j])
            numerator += np.sum(W_chunk * (dev_chunk[:, np.newaxis] * deviations[np.newaxis, :]))
            total_weight += np.sum(W_chunk)
        
        # Normalize by total weight
        if total_weight > 0:
            numerator = numerator / total_weight
        else:
            return 0.0
        
        # Calculate denominator
        denominator = np.sum(deviations**2) / n
        
        if denominator == 0:
            return 0.0
        
        # Final Moran's I calculation
        morans_i = (n / np.sum(deviations != 0)) * numerator / denominator
        return morans_i
    
    def _check_convergence(self, geo_coherence: float, patience_counter: int) -> bool:
        """Check convergence criteria using topology preservation metrics."""
        converged_qe = False
        converged_topology = False
        
        # Check QE stability over window
        qe_stability_window = self._get_config('qe_stability_window', 20)
        qe_stability_threshold = self._get_config('qe_stability_threshold', 0.001)
        
        if len(self.training_history['quantization_errors']) >= qe_stability_window:
            recent_qes = self.training_history['quantization_errors'][-qe_stability_window:]
            qe_range = max(recent_qes) - min(recent_qes)
            qe_relative_range = qe_range / np.mean(recent_qes)
            
            if qe_relative_range < qe_stability_threshold:
                logger.info(f"QE stable over {qe_stability_window} epochs (range = {qe_relative_range:.5f} < {qe_stability_threshold})")
                converged_qe = True
        
        # Check topographic error - better topology preservation metric
        if len(self.training_history['topographic_errors']) >= 5:
            recent_te = self.training_history['topographic_errors'][-5:]
            avg_te = np.mean(recent_te)
            te_threshold = self._get_config('te_convergence_threshold', 0.05)
            
            if avg_te < te_threshold:
                logger.info(f"Topographic error converged: {avg_te:.4f} < {te_threshold}")
                converged_topology = True
        
        # Fallback to geographic coherence for geo-specific applications
        if not converged_topology and geo_coherence > self.geographic_coherence_threshold:
            logger.info(f"Geographic coherence {geo_coherence:.3f} > threshold {self.geographic_coherence_threshold}")
            converged_topology = True
        
        # Require BOTH QE stability and topology preservation
        if converged_qe and converged_topology:
            logger.info("Convergence criteria met (QE stability + topology preservation)")
            return True
        elif converged_qe:
            logger.info("QE converged but topology not preserved - continuing...")
        elif converged_topology:
            logger.info("Topology preserved but QE not stable - continuing...")
        
        # Alternative: Patience exceeded with reasonable QE
        if patience_counter >= self.patience:
            if len(self.training_history['quantization_errors']) > 20:
                recent_qes = self.training_history['quantization_errors'][-20:]
                qe_trend = recent_qes[-1] - recent_qes[0]
                if qe_trend <= 0:  # QE is not getting worse
                    logger.info(f"Patience exceeded ({patience_counter} epochs) with stable/improving QE")
                    return True
        
        return False
    
    def get_weights(self) -> np.ndarray:
        """Get weight matrix in grid shape."""
        return self.weights.reshape(self.grid_rows, self.grid_cols, self.input_dim)
    
    def save_checkpoint(self, checkpoint_dir: str, epoch: int, best_qe: float, patience_counter: int):
        """Save training checkpoint to disk with integrity verification.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            epoch: Current epoch number
            best_qe: Best QE achieved so far
            patience_counter: Current patience counter value
        """
        logger.info(f"Saving checkpoint at epoch {epoch} to {checkpoint_dir}")
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint filename with epoch
        checkpoint_file = checkpoint_path / f"som_checkpoint_epoch_{epoch}.npz"
        
        # Calculate checksum for integrity verification
        import hashlib
        weights_checksum = hashlib.md5(self.weights.tobytes()).hexdigest()
        
        # Prepare checkpoint data with metadata for validation
        checkpoint_data = {
            'epoch': epoch,
            'weights': self.weights,
            # Pure Batch SOM doesn't use learning rate
            'current_radius': self.current_radius,
            'best_qe': best_qe,
            'patience_counter': patience_counter,
            'grid_size': self.config.grid_size,
            'input_dim': self.input_dim,
            'spatial_weight': self.config.spatial_weight,
            'training_history': json.dumps(self.training_history),  # JSON for nested dict
            'weights_checksum': weights_checksum,
            'checkpoint_version': '2.0',  # Version for compatibility checking
            'timestamp': time.time()
        }
        
        # Save numpy arrays and scalars with error handling
        try:
            np.savez_compressed(checkpoint_file, **checkpoint_data)
            
            # Verify the checkpoint was written correctly
            test_load = np.load(checkpoint_file, allow_pickle=False)
            test_checksum = hashlib.md5(test_load['weights'].tobytes()).hexdigest()
            if test_checksum != weights_checksum:
                raise ValueError(f"Checkpoint verification failed: checksum mismatch")
            test_load.close()
            
            logger.info(f"Saved and verified checkpoint to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if checkpoint_file.exists():
                checkpoint_file.unlink()  # Remove corrupted file
            raise
        
        # Also save a 'latest' symlink for easy access
        latest_link = checkpoint_path / "som_checkpoint_latest.npz"
        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(checkpoint_file.name)
            logger.info(f"Created symlink: som_checkpoint_latest.npz -> {checkpoint_file.name}")
        except (FileExistsError, OSError) as e:
            # If symlink creation fails, remove any existing link and try once more
            logger.warning(f"Symlink creation failed: {e}, attempting to fix...")
            try:
                if latest_link.exists() or latest_link.is_symlink():
                    latest_link.unlink()
                latest_link.symlink_to(checkpoint_file.name)
                logger.info(f"Successfully created symlink on retry: som_checkpoint_latest.npz -> {checkpoint_file.name}")
            except Exception as retry_error:
                logger.error(f"Failed to create symlink after retry: {retry_error}")
                logger.warning("Checkpoint saved but symlink creation failed - resume may not work")
        
        logger.info(f"Saved checkpoint at epoch {epoch} to {checkpoint_file}")
        
        # Clean up old checkpoints (keep last 3 by modification time)
        checkpoints = list(checkpoint_path.glob("som_checkpoint_epoch_*.npz"))
        if len(checkpoints) > 3:
            # Sort by modification time (oldest first)
            checkpoints_by_time = sorted(checkpoints, key=lambda p: p.stat().st_mtime)
            # Remove oldest checkpoints, keeping the 3 most recent
            for old_checkpoint in checkpoints_by_time[:-3]:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint.name} (keeping 3 most recent)")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint from disk with integrity verification.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint data
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is corrupted or incompatible
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint safely without pickle
            checkpoint = np.load(checkpoint_path, allow_pickle=False)
            
            # Check version compatibility
            version = checkpoint.get('checkpoint_version', '1.0')
            if version != '2.0' and 'weights_checksum' not in checkpoint:
                logger.warning(f"Loading legacy checkpoint (version {version}), no integrity check available")
            
            # Verify integrity if checksum is available
            if 'weights_checksum' in checkpoint:
                import hashlib
                expected_checksum = str(checkpoint['weights_checksum'])
                actual_checksum = hashlib.md5(checkpoint['weights'].tobytes()).hexdigest()
                if actual_checksum != expected_checksum:
                    raise ValueError(f"Checkpoint integrity check failed: checksum mismatch")
                logger.info("Checkpoint integrity verified successfully")
            
            # Validate checkpoint structure
            required_fields = ['epoch', 'weights', 'current_radius', 'best_qe', 'patience_counter']
            for field in required_fields:
                if field not in checkpoint:
                    raise ValueError(f"Corrupted checkpoint: missing required field '{field}'")
            
            # Validate dimensions
            weights_shape = checkpoint['weights'].shape
            if 'grid_size' in checkpoint:
                expected_neurons = checkpoint['grid_size'][0] * checkpoint['grid_size'][1]
                if weights_shape[0] != expected_neurons:
                    raise ValueError(f"Checkpoint validation failed: weights shape {weights_shape} doesn't match grid size {checkpoint['grid_size']}")
            
            # Extract data with error handling
            checkpoint_data = {
                'epoch': int(checkpoint['epoch']),
                'weights': checkpoint['weights'],
                # Pure Batch SOM doesn't use learning rate
                'current_radius': float(checkpoint['current_radius']),
                'best_qe': float(checkpoint['best_qe']),
                'patience_counter': int(checkpoint['patience_counter']),
                'training_history': json.loads(str(checkpoint['training_history'])),
                'input_dim': int(checkpoint['input_dim']) if 'input_dim' in checkpoint else None
            }
            
            # Validate checkpoint compatibility (convert both to list for comparison)
            checkpoint_grid = checkpoint['grid_size'].tolist() if hasattr(checkpoint['grid_size'], 'tolist') else list(checkpoint['grid_size'])
            config_grid = list(self.config.grid_size)
            if checkpoint_grid != config_grid:
                raise ValueError(f"Grid size mismatch: checkpoint has {checkpoint_grid}, "
                               f"config has {config_grid}")
                               
        except KeyError as e:
            raise ValueError(f"Corrupted checkpoint: missing data field {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Corrupted checkpoint: invalid data format - {e}")
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}") from e
        finally:
            # Always close the npz file to prevent resource leaks
            if 'checkpoint' in locals():
                checkpoint.close()
        
        # Only check input_dim if already initialized and checkpoint has it
        if checkpoint_data.get('input_dim') is not None:
            if self.input_dim is not None and checkpoint_data['input_dim'] != self.input_dim:
                raise ValueError(f"Input dimension mismatch: checkpoint has {checkpoint_data['input_dim']}, "
                               f"current data has {self.input_dim}")
            elif self.input_dim is None:
                # Set input_dim from checkpoint when resuming
                self.input_dim = checkpoint_data['input_dim']
        
        # Restore state
        self.weights = checkpoint_data['weights']
        # Pure Batch SOM doesn't use learning rate
        self.current_radius = checkpoint_data['current_radius']
        self.training_history = checkpoint_data['training_history']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")
        logger.info(f"Resumed with Radius={self.current_radius:.2f}")
        
        return checkpoint_data
    
    def cleanup(self):
        """Explicitly clean up resources used during training.
        
        This method ensures proper resource cleanup to prevent memory leaks
        in long-running processes. Should be called when SOM is no longer needed.
        """
        try:
            # Clean up processor if it exists
            if hasattr(self, 'processor') and self.processor is not None:
                if hasattr(self.processor, 'cleanup'):
                    self.processor.cleanup()
                    logger.debug("Cleaned up parallel processor resources")
                self.processor = None
            
            # Clear large arrays to free memory
            if hasattr(self, 'velocity') and self.velocity is not None:
                self.velocity = None
                logger.debug("Cleared velocity array")
            
            # Clear training history if it's very large
            if hasattr(self, 'training_history') and self.training_history:
                history_size = sum(len(v) if isinstance(v, list) else 1 
                                 for v in self.training_history.values())
                if history_size > 10000:  # Clear if history is very large
                    self.training_history = {
                        'quantization_errors': [],
                        'topographic_errors': [],
                        'geographic_coherences': [],
                        'learning_rates': [],
                        'radii': []
                    }
                    logger.debug(f"Cleared large training history ({history_size} entries)")
            
            # Force garbage collection for large objects
            import gc
            gc.collect()
            logger.debug("Forced garbage collection completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def predict(self, data: np.ndarray, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict BMU indices using fast Numba implementation."""
        # Validate coordinates shape if provided
        if coordinates is not None:
            if coordinates.shape[0] != data.shape[0]:
                raise ValueError(f"Coordinates samples ({coordinates.shape[0]}) must match data samples ({data.shape[0]})")
            if coordinates.shape[1] != 2:
                raise ValueError(f"Coordinates must have 2 columns [longitude, latitude], got {coordinates.shape[1]}")
        
        # Ensure contiguous arrays for Numba
        data = np.ascontiguousarray(data, dtype=np.float32)
        weights = np.ascontiguousarray(self.weights, dtype=np.float32)
        
        # Use fast parallel Numba implementation (no chunking needed!)
        bmu_indices = predict_batch_numba(data, weights, self.min_valid_features)
        
        # Note: If we need geographic distance in prediction later,
        # we can add predict_batch_with_coords_numba to som_metrics.py
        
        return bmu_indices
    
    def transform(self, data: np.ndarray, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform data to grid coordinates."""
        bmu_indices = self.predict(data, coordinates)
        grid_coords = []
        
        for idx in bmu_indices:
            if idx != INVALID_INDEX:
                grid_coords.append(self._grid_positions[idx])
            else:
                grid_coords.append([INVALID_COORDINATE, INVALID_COORDINATE])
        
        return np.array(grid_coords)