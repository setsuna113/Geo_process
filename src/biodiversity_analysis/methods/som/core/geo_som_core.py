"""GeoSOM + VLRSOM hybrid implementation for biodiversity analysis.

This implementation follows the specifications in final_som_configuration_decisions.md:
- Handles 70% missing data with partial Bray-Curtis distance
- Integrates spatial information (30% spatial, 70% features)
- Uses adaptive learning rate based on QE improvement
- Implements geographic coherence for convergence
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

from src.abstractions.types.som_types import SOMConfig, SOMTrainingResult
from ..utils.constants import (
    INVALID_DISTANCE, INVALID_INDEX, INVALID_COORDINATE,
    MIN_VALID_FEATURES, INSUFFICIENT_FEATURES_MSG, DEFAULT_GEO_DISTANCE_NORM
)
from ..calculators.partial_metrics import validate_missing_data_handling
from src.biodiversity_analysis.shared.metrics.distance_calculator import SharedDistanceCalculator
from ..trainers.batch_som_trainer import OptimizedBatchTrainer, OptimizedBatchConfig

logger = logging.getLogger(__name__)

# Import constants from central location
from ..utils.constants import (
    DEFAULT_CHUNK_SIZE, DEFAULT_QE_SAMPLE_SIZE, DEFAULT_PROGRESS_LOG_INTERVAL,
    MEMORY_THRESHOLD_BYTES, DEFAULT_DISTANCE_CALC_BATCH_SIZE
)


@dataclass
class GeoSOMConfig:
    """Configuration for GeoSOM + VLRSOM hybrid."""
    # Grid parameters
    grid_size: Tuple[int, int]
    topology: str = "rectangular"  # or "hexagonal"
    
    # GeoSOM parameters
    spatial_weight: float = 0.3  # k=0.3 (30% spatial, 70% features)
    geographic_distance: str = "haversine"
    
    # Batch SOM Training Configuration (processes mini-batches for memory efficiency)
    batch_training: Optional[Dict[str, Any]] = None
    
    # Memory Optimization Configuration  
    memory_optimization: Optional[Dict[str, Any]] = None
    
    # VLRSOM parameters
    initial_learning_rate: float = 0.5
    min_learning_rate: float = 0.01
    max_learning_rate: float = 0.8
    lr_increase_factor: float = 1.1  # When QE improves
    lr_decrease_factor: float = 0.85  # When QE worsens
    
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
    chunk_size: int = 50000  # Default chunk size for batch processing (override with config)
    qe_sample_size: int = 100000  # Sample size for QE calculation
    use_parallel: bool = True  # Enable parallel chunk processing
    n_workers: Optional[int] = None  # Number of parallel workers (None=auto)


class GeoSOMVLRSOM:
    """Geographic Self-Organizing Map with Variable Learning Rate SOM.
    
    This hybrid architecture combines:
    1. GeoSOM: Incorporates spatial information into the distance calculation
    2. VLRSOM: Adapts learning rate based on quantization error improvement
    
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
            section: Optional config section (e.g., 'memory_optimization', 'minibatch_training')
            
        Returns:
            Config value or default
        """
        cache_key = f"{section}.{key}" if section else key
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
            
        value = default
        if section:
            # Access nested config section
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
        
        # FIXED: Streamlined distance calculator without counterproductive caching
        from ..calculators.optimized_distance_calculator import OptimizedDistanceCalculator
        
        self.spatial_weight = config.spatial_weight
        self.min_valid_features = config.min_valid_features
        self.max_epochs = config.max_epochs
        self.qe_improvement_threshold = config.qe_improvement_threshold
        self.geographic_coherence_threshold = config.geographic_coherence_threshold
        self.patience = config.patience
        self.neighborhood_function = config.neighborhood_function
        
        # Use shared distance calculator for consistency across methods
        self.distance_calculator = SharedDistanceCalculator(
            distance_metric='bray_curtis',
            min_valid_features=self.min_valid_features
        )
        
        # Initialize fully optimized batch SOM trainer with momentum
        # All optimizations are built-in with no configuration redundancy
        optimized_config = OptimizedBatchConfig(
            batch_size=self._get_config('batch_size', 5000, 'batch_training'),
            learning_schedule=self._get_config('learning_schedule', 'exponential', 'batch_training'),
            initial_learning_rate=config.initial_learning_rate,
            final_learning_rate=config.min_learning_rate,
            # Momentum settings (NEW)
            initial_momentum=self._get_config('initial_momentum', 0.9, 'batch_training'),
            final_momentum=self._get_config('final_momentum', 0.1, 'batch_training'),
            momentum_schedule=self._get_config('momentum_schedule', 'linear', 'batch_training'),
            # Distance metric settings
            distance_metric=self._get_config('distance_metric', 'bray_curtis', 'batch_training'),
            min_valid_features=self._get_config('min_valid_features', 2, 'batch_training'),
        )
        
        self.batch_trainer = OptimizedBatchTrainer(
            grid_width=self.grid_cols,
            grid_height=self.grid_rows,
            n_features=1,  # Placeholder, will be updated when weights are initialized
            config=optimized_config
        )
        
        # All optimizations are now always enabled in the trainer:
        # - Numba-optimized Bray-Curtis distance calculation
        # - Momentum-based weight updates with adaptive decay
        # - Precomputed neuron-to-neuron distances
        # - All Numba JIT with cache=True, fastmath=True
        # - Contiguous memory layout for cache efficiency
        logger.info(f"Initialized fully optimized batch SOM trainer with batch_size={optimized_config.batch_size}, "
                   f"momentum={optimized_config.initial_momentum}->{optimized_config.final_momentum}, "
                   f"learning_schedule={optimized_config.learning_schedule}")
        
        # Training state
        self.current_lr = config.initial_learning_rate
        self.current_radius = None
        self.training_history = {
            'quantization_errors': [],
            'learning_rates': [],
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
    
    def _validate_coordinates(self, coordinates: np.ndarray, n_samples: int) -> None:
        """Validate coordinate array for proper shape and values.
        
        Args:
            coordinates: Geographic coordinates to validate
            n_samples: Expected number of samples
            
        Raises:
            ValueError: If coordinates have invalid shape or dimensions
        """
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
    
    def partial_bray_curtis(self, u: np.ndarray, v: np.ndarray) -> float:
        """Bray-Curtis distance for vectors with missing values.
        
        Uses SharedDistanceCalculator for consistent implementation.
        Only compares non-NA pairs, requires minimum valid features.
        
        Args:
            u, v: Vectors to compare (may contain NaN)
            
        Returns:
            Distance value or INVALID_DISTANCE (np.inf) if insufficient valid pairs
        """
        # Use the shared distance calculator for consistency
        distances = self.distance_calculator.compute_distances(
            u.reshape(1, -1), v.reshape(1, -1)
        )
        return distances[0, 0]
    
    def haversine_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Calculate haversine distance between geographic coordinates.
        
        Args:
            coord1, coord2: [longitude, latitude] in degrees
            
        Returns:
            Distance in kilometers
        """
        # Use shared distance calculator for consistency
        # Convert from [lon, lat] to [lat, lon] for SharedDistanceCalculator
        coords1 = np.array([[coord1[1], coord1[0]]])  # [lat, lon]
        coords2 = np.array([[coord2[1], coord2[0]]])  # [lat, lon]
        distances = self.distance_calculator.compute_pairwise_distances(
            coords1, coords2, metric='haversine'
        )
        return distances[0, 0]
    
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

    def _vectorized_partial_bray_curtis_chunk(self, data_chunk: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Vectorized Bray-Curtis distance calculation for a data chunk.
        
        Uses SharedDistanceCalculator to maintain abstraction and avoid code duplication.
        
        Args:
            data_chunk: Shape (n_samples, n_features)
            weights: Shape (n_neurons, n_features)
            
        Returns:
            distances: Shape (n_samples, n_neurons)
        """
        # Use shared distance calculator - fixes abstraction violation
        return self.distance_calculator.compute_distances(data_chunk, weights)
    
    def _vectorized_combined_distance(self, data_chunk: np.ndarray, weights: np.ndarray,
                                    coords_chunk: np.ndarray, neuron_coords: np.ndarray) -> np.ndarray:
        """Vectorized combined distance calculation (feature + geographic).
        
        Combines Bray-Curtis feature distances with Haversine geographic distances
        using SharedDistanceCalculator to maintain proper abstraction.
        
        Args:
            data_chunk: Shape (n_samples, n_features)
            weights: Shape (n_neurons, n_features)  
            coords_chunk: Shape (n_samples, 2) - [lat, lon]
            neuron_coords: Shape (n_neurons, 2) - [lat, lon]
            
        Returns:
            distances: Shape (n_samples, n_neurons)
        """
        # Calculate feature distances using shared calculator
        feature_distances = self.distance_calculator.compute_distances(data_chunk, weights)
        
        # Calculate geographic distances using shared calculator  
        geo_distances = self.distance_calculator.compute_pairwise_distances(
            coords_chunk, neuron_coords, metric='haversine'
        )
        
        # Normalize geographic distances (max ~20,000 km)
        geo_distances_norm = geo_distances / DEFAULT_GEO_DISTANCE_NORM
        
        # Combine distances with spatial weighting
        valid_feature = ~np.isinf(feature_distances)
        combined = np.full_like(feature_distances, INVALID_DISTANCE)
        combined[valid_feature] = (
            (1 - self.spatial_weight) * feature_distances[valid_feature] +
            self.spatial_weight * geo_distances_norm[valid_feature]
        )
        
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
        
        # Update mini-batch trainer with correct n_features
        self.batch_trainer.n_features = self.input_dim
        
        # WeightUpdater is only needed for standard trainer (fast trainer has internal accumulation)
        if hasattr(self.batch_trainer, 'weight_updater'):
            self.batch_trainer.weight_updater = WeightUpdater(
                self.batch_trainer.n_neurons, self.input_dim
            )
        
        # Validate coordinates if provided
        if coordinates is not None:
            self._validate_coordinates(coordinates, data.shape[0])
        
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
        
        if self.weights is None:
            self.initialize_weights(data, coordinates)
        
        # Initialize training state
        training_state = self._initialize_training_state(data, coordinates)
        
        # Main training loop
        for epoch in range(training_state['start_epoch'], self.max_epochs):
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
        else:
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
        logger.info(f"Starting epoch {epoch+1}/{self.max_epochs}")
        
        # Update radius
        self._update_radius(epoch)
        
        # Execute mini-batch training
        qe = self._execute_minibatch_training(epoch, data, coordinates)
        
        # Update training history
        self.training_history['quantization_errors'].append(qe)
        self.training_history['learning_rates'].append(self.current_lr)
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
        """Execute mini-batch training for one epoch."""
        logger.info(f"Starting mini-batch training for epoch {epoch+1}")
        
        # Use the minibatch trainer's train_epoch method
        training_stats = self.batch_trainer.train_epoch(
            data=data, 
            weights=self.weights,
            epoch=epoch,
            max_epochs=self.config.max_epochs
        )
        
        # Update learning rate from trainer (uses standard schedule)
        self.current_lr = training_stats['learning_rate']
        self.current_radius = training_stats['radius']
        
        # Get QE from training stats (calculated after weight updates)
        qe = training_stats.get('qe', 0.0)
        if qe == 0.0 or qe is None:
            # If QE wasn't calculated this epoch, calculate it now
            qe = self.calculate_quantization_error(data, coordinates)
        
        converged = training_stats.get('converged', False)
        logger.info(f"Completed mini-batch training for epoch {epoch+1}: "
                   f"QE={qe:.6f}, LR={self.current_lr:.4f}, converged={converged}")
        
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
        # Topographic error calculation
        te_freq = self._get_config('te_calculation_frequency', 10, 'progress_tracking')
        if epoch % te_freq == 0 and epoch > 0:
            self._calculate_and_adjust_topographic_error(epoch, data, coordinates)
        
        # Geographic coherence and convergence check
        geo_freq = self._get_config('geographic_coherence_frequency', 10, 'progress_tracking')
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
        te_high_threshold = self._get_config('te_high_threshold', 0.15)
        te_low_threshold = self._get_config('te_low_threshold', 0.05)
        
        if te > te_high_threshold:
            radius_modifier = self._get_config('te_high_radius_modifier', 1.1)
            adjusted_radius = next_epoch_radius * radius_modifier
            logger.info(f"High TE ({te:.3f}): adjusting radius {next_epoch_radius:.3f} → {adjusted_radius:.3f}")
        elif te < te_low_threshold:
            radius_modifier = self._get_config('te_low_radius_modifier', 0.95)
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
        coherence_sample_threshold = self._get_config('coherence_sample_threshold', 10000)
        coherence_sample_size = self._get_config('coherence_sample_size', 5000)
        
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
        checkpoint_frequency = self._get_config('checkpoint_frequency', 10, 'progress_tracking')
        checkpoint_dir = self._get_config('checkpoint_dir', 'checkpoints/som', 'progress_tracking')
        
        if checkpoint_frequency > 0 and epoch % checkpoint_frequency == 0:
            self.save_checkpoint(checkpoint_dir, epoch, best_qe, patience_counter)
    
    def _handle_progress_callback(self, progress_callback: Callable, epoch: int, qe: float) -> None:
        """Handle progress callback without fallbacks."""
        progress_info = {
            'epoch': epoch + 1,
            'max_epochs': self.max_epochs,
            'progress': (epoch + 1) / self.max_epochs,
            'qe': qe,
            'learning_rate': self.current_lr,
            'radius': self.current_radius
        }
        progress_callback(progress_info)
    
    def _create_training_result(self, training_state: Dict[str, Any], n_samples: int) -> SOMTrainingResult:
        """Create the final training result."""
        return SOMTrainingResult(
            weights=self.weights.reshape(self.grid_rows, self.grid_cols, self.input_dim),
            quantization_errors=self.training_history['quantization_errors'],
            topographic_errors=[],
            training_time=0.0,
            convergence_epoch=training_state.get('convergence_epoch'),
            final_learning_rate=self.current_lr,
            final_neighborhood_radius=self.current_radius,
            n_samples_trained=n_samples
        )
    
    # Note: Checkpoint saving is handled by the standardized save_checkpoint method at line 1376
    
    def _calculate_vlrsom_error_squared(self, data: np.ndarray, 
                                       coordinates: Optional[np.ndarray] = None,
                                       sample_size: int = 10000) -> float:
        """Calculate J²(t) error function for VLRSOM.
        
        From paper Equation (18): J(t) = (1/n) Σ ||wc(t) - xi(t)||²
        
        Args:
            data: Input data samples
            coordinates: Geographic coordinates (optional)
            sample_size: Number of samples to use for approximation
            
        Returns:
            J²(t): Squared error function value
        """
        n_samples = len(data)
        
        # Use sampling for efficiency (leverages existing infrastructure)
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            data_sample = data[indices]
            coords_sample = coordinates[indices] if coordinates is not None else None
        else:
            data_sample = data
            coords_sample = coordinates
        
        # Pre-filter valid samples
        valid_features = (~np.isnan(data_sample)).sum(axis=1)
        valid_mask = valid_features >= self.min_valid_features
        
        if not valid_mask.any():
            logger.warning("No valid samples for VLRSOM error calculation")
            return 1.0  # High error value
        
        data_valid = data_sample[valid_mask]
        coords_valid = coords_sample[valid_mask] if coords_sample is not None else None
        n_valid = len(data_valid)
        
        # Calculate distances to all neurons for each sample
        if coordinates is not None and hasattr(self, 'neuron_coords'):
            # Combined distance (feature + geographic)
            distances = self._vectorized_combined_distance(
                data_valid, self.weights, coords_valid, self.neuron_coords
            )
        else:
            # Feature distance only
            distances = self._vectorized_partial_bray_curtis_chunk(data_valid, self.weights)
        
        # Find BMU for each sample
        bmu_indices = np.argmin(distances, axis=1)
        min_distances = np.take_along_axis(
            distances, bmu_indices[:, np.newaxis], axis=1
        ).squeeze()
        
        # Handle invalid distances
        valid_distances = min_distances[~np.isinf(min_distances)]
        if len(valid_distances) == 0:
            logger.warning("All samples have infinite distances in VLRSOM error calculation")
            return 1.0
        
        # Calculate mean squared error: J(t) = (1/n) Σ ||error||²
        mean_squared_error = np.mean(valid_distances ** 2)
        
        # Return J²(t) as required by VLRSOM formula
        return mean_squared_error
    
    def _adapt_learning_rate_vlrsom(self, data: np.ndarray,
                                   coordinates: Optional[np.ndarray] = None):
        """VLRSOM: Adapt learning rate using proper algorithm from paper.
        
        From paper Equation (20): μ'(t+1) = αμ(t) + γJ²(t)
        From paper Equation (21): Bounds checking
        
        Args:
            data: Current training data (already sampled by caller)
            coordinates: Geographic coordinates (if available)
        """
        # Get VLRSOM parameters from config
        alpha = self._get_config('vlrsom_alpha', 0.95)  # Decay factor (0 < α < 1)
        gamma = self._get_config('vlrsom_gamma', 0.001)  # Adaptation parameter (γ > 0)
        
        # Calculate J²(t) using existing sampling infrastructure
        error_sample_size = self._get_config('vlrsom_error_sample_size', 10000)
        j_squared = self._calculate_vlrsom_error_squared(data, coordinates, error_sample_size)
        
        # Apply VLRSOM formula: μ'(t+1) = αμ(t) + γJ²(t)
        new_lr = alpha * self.current_lr + gamma * j_squared
        
        # Apply bounds from paper Equation (21)
        min_lr = self._get_config('vlrsom_min_lr', self.config.min_learning_rate)
        max_lr = self._get_config('vlrsom_max_lr', self.config.max_learning_rate)
        
        self.current_lr = np.clip(new_lr, min_lr, max_lr)
        
        # Log VLRSOM adaptation for debugging
        if hasattr(self, '_vlrsom_log_frequency'):
            epoch = len(self.training_history.get('learning_rates', [])) 
            if epoch % 10 == 0:
                logger.debug(f"VLRSOM: J²(t)={j_squared:.6f}, α={alpha}, γ={gamma}, "
                           f"LR: {self.current_lr:.6f} (bounds: [{min_lr:.6f}, {max_lr:.6f}])")
        
    def _adapt_learning_rate(self, data: np.ndarray, 
                           coordinates: Optional[np.ndarray] = None):
        """Adapt learning rate using VLRSOM algorithm.
        
        This is the main entry point called by the training loop.
        
        Args:
            data: Current training data (already sampled by caller)
            coordinates: Geographic coordinates (if available)
        """
        self._adapt_learning_rate_vlrsom(data, coordinates)
    
    def _get_scheduled_radius(self, epoch: int) -> float:
        """Get the scheduled radius for a given epoch without adjustment."""
        if self.config.radius_decay == "linear":
            progress = epoch / self.config.max_epochs
            return self.config.initial_radius * (1 - progress) + \
                   self.config.final_radius * progress
        elif self.config.radius_decay == "exponential":
            # Exponential decay: r(t) = r_init * exp(-lambda * t)
            # where lambda is chosen so that r(max_epochs) ≈ r_final
            decay_rate = -np.log(self.config.final_radius / self.config.initial_radius) / self.config.max_epochs
            return self.config.initial_radius * np.exp(-decay_rate * epoch)
        else:
            raise NotImplementedError(f"Decay {self.config.radius_decay} not implemented")
    
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
    
    def calculate_quantization_error(self, data: np.ndarray, 
                                   coordinates: Optional[np.ndarray] = None,
                                   sample_size: Optional[int] = None) -> float:
        """Calculate average quantization error with optional sampling.
        
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
            n_samples = sample_size
            logger.debug(f"Using {sample_size} samples for QE calculation")
        
        # Pre-filter valid samples to avoid unnecessary computation
        valid_features = (~np.isnan(data)).sum(axis=1)
        valid_mask = valid_features >= self.min_valid_features
        
        if not valid_mask.any():
            return np.inf
        
        data_valid = data[valid_mask]
        coords_valid = coordinates[valid_mask] if coordinates is not None else None
        
        # Process in chunks for memory efficiency
        chunk_size = self._get_config('qe_calc_chunk_size', 1000)
        total_error = 0.0
        n_valid = 0
        
        for start_idx in range(0, len(data_valid), chunk_size):
            end_idx = min(start_idx + chunk_size, len(data_valid))
            chunk_data = data_valid[start_idx:end_idx]
            chunk_coords = coords_valid[start_idx:end_idx] if coords_valid is not None else None
            
            # Vectorized distance calculation
            if coordinates is not None and hasattr(self, 'neuron_coords'):
                distances = self._vectorized_combined_distance(
                    chunk_data, self.weights, chunk_coords, self.neuron_coords
                )
            else:
                distances = self._vectorized_partial_bray_curtis_chunk(chunk_data, self.weights)
            
            # Find BMUs and calculate errors
            bmu_indices = np.argmin(distances, axis=1)
            min_errors = np.take_along_axis(distances, bmu_indices[:, np.newaxis], axis=1).squeeze()
            
            # Accumulate valid errors
            valid_errors = ~np.isinf(min_errors)
            total_error += min_errors[valid_errors].sum()
            n_valid += valid_errors.sum()
        
        return total_error / n_valid if n_valid > 0 else np.inf
    
    def calculate_topographic_error(self, data: np.ndarray, 
                                  coordinates: Optional[np.ndarray] = None,
                                  sample_size: int = 10000) -> float:
        """Calculate topographic error with sampling for efficiency.
        
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
            n_samples = sample_size
            logger.debug(f"Using {sample_size} samples for TE calculation")
        
        # Pre-filter valid samples
        valid_features = (~np.isnan(data)).sum(axis=1)
        valid_mask = valid_features >= self.min_valid_features
        
        if not valid_mask.any():
            return 1.0  # Maximum error if no valid samples
        
        data_valid = data[valid_mask]
        coords_valid = coordinates[valid_mask] if coordinates is not None else None
        
        # Calculate distances to all neurons
        if coordinates is not None and hasattr(self, 'neuron_coords'):
            distances = self._vectorized_combined_distance(
                data_valid, self.weights, coords_valid, self.neuron_coords
            )
        else:
            distances = self._vectorized_partial_bray_curtis_chunk(data_valid, self.weights)
        
        # Find BMU and 2nd BMU for each sample
        te_errors = 0
        n_valid = 0
        
        for i in range(len(distances)):
            dist_row = distances[i]
            if np.all(np.isinf(dist_row)):
                continue
                
            # Find BMU
            bmu_idx = np.argmin(dist_row)
            
            # Find 2nd BMU by masking out the BMU
            dist_row_2nd = dist_row.copy()
            dist_row_2nd[bmu_idx] = np.inf
            second_bmu_idx = np.argmin(dist_row_2nd)
            
            # Check if they are neighbors on the grid
            bmu_pos = self._get_neuron_position(bmu_idx)
            second_bmu_pos = self._get_neuron_position(second_bmu_idx)
            
            # Calculate grid distance (Manhattan distance for rectangular grid)
            grid_distance = abs(bmu_pos[0] - second_bmu_pos[0]) + abs(bmu_pos[1] - second_bmu_pos[1])
            
            # Neighbors are adjacent if grid distance is 1
            if grid_distance > 1:
                te_errors += 1
            
            n_valid += 1
        
        return te_errors / n_valid if n_valid > 0 else 1.0
    
    def _get_neuron_position(self, neuron_idx: int) -> Tuple[int, int]:
        """Get grid position (row, col) from neuron index."""
        row = neuron_idx // self.grid_cols
        col = neuron_idx % self.grid_cols
        return (row, col)
    
    def _calculate_geographic_coherence(self, data: np.ndarray, 
                                      coordinates: np.ndarray) -> float:
        """Calculate geographic coherence using Moran's I.
        
        Uses chunked processing for large datasets to avoid O(n²) memory usage.
        For datasets > 10k samples, uses a chunked approach that processes
        the distance matrix in blocks.
        """
        # Get cluster assignments using vectorized approach
        logger.debug(f"Finding BMUs for {len(data)} samples for geographic coherence")
        
        # Use vectorized distance calculation
        if hasattr(self, 'neuron_coords'):
            distances = self._vectorized_combined_distance(
                data, self.weights, coordinates, self.neuron_coords
            )
        else:
            distances = self._vectorized_partial_bray_curtis_chunk(data, self.weights)
        
        # Find BMUs
        clusters = np.argmin(distances, axis=1)
        
        # Mark invalid distances as INVALID_INDEX
        invalid_mask = np.all(np.isinf(distances), axis=1)
        clusters[invalid_mask] = self.INVALID_INDEX
        valid = clusters >= 0
        
        if valid.sum() < 10:  # Need minimum samples
            return 0.0
        
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
        """Check convergence criteria using topology preservation metrics from VLRSOM paper."""
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
        
        # Check topographic error (from VLRSOM paper) - better topology preservation metric
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
        """Save training checkpoint to disk.
        
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
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'weights': self.weights,
            'current_lr': self.current_lr,
            'current_radius': self.current_radius,
            'best_qe': best_qe,
            'patience_counter': patience_counter,
            'grid_size': self.config.grid_size,
            'input_dim': self.input_dim,
            'spatial_weight': self.config.spatial_weight,
            'training_history': json.dumps(self.training_history)  # JSON for nested dict
        }
        
        # Save numpy arrays and scalars
        np.savez_compressed(checkpoint_file, **checkpoint_data)
        logger.info(f"Saved checkpoint to {checkpoint_file}")
        
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
        
        # Clean up old checkpoints (keep last 3)
        checkpoints = sorted(checkpoint_path.glob("som_checkpoint_epoch_*.npz"))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint from disk.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint safely without pickle
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        
        # Extract data
        checkpoint_data = {
            'epoch': int(checkpoint['epoch']),
            'weights': checkpoint['weights'],
            'current_lr': float(checkpoint['current_lr']),
            'current_radius': float(checkpoint['current_radius']),
            'best_qe': float(checkpoint['best_qe']),
            'patience_counter': int(checkpoint['patience_counter']),
            'training_history': json.loads(str(checkpoint['training_history']))
        }
        
        # Validate checkpoint compatibility (convert both to list for comparison)
        checkpoint_grid = checkpoint['grid_size'].tolist() if hasattr(checkpoint['grid_size'], 'tolist') else list(checkpoint['grid_size'])
        config_grid = list(self.config.grid_size)
        if checkpoint_grid != config_grid:
            raise ValueError(f"Grid size mismatch: checkpoint has {checkpoint_grid}, "
                           f"config has {config_grid}")
        
        # Only check input_dim if already initialized
        if self.input_dim is not None and int(checkpoint['input_dim']) != self.input_dim:
            raise ValueError(f"Input dimension mismatch: checkpoint has {checkpoint['input_dim']}, "
                           f"current data has {self.input_dim}")
        elif self.input_dim is None:
            # Set input_dim from checkpoint when resuming
            self.input_dim = int(checkpoint['input_dim'])
        
        # Restore state
        self.weights = checkpoint_data['weights']
        self.current_lr = checkpoint_data['current_lr']
        self.current_radius = checkpoint_data['current_radius']
        self.training_history = checkpoint_data['training_history']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")
        logger.info(f"Resumed with LR={self.current_lr:.4f}, Radius={self.current_radius:.2f}")
        
        return checkpoint_data
    
    def cleanup(self):
        """Clean up resources used during training."""
        # Note: Processor cleanup is now handled automatically
        # Since we create a new processor per training run (not stored as instance variable),
        # Python's garbage collector will clean it up when it goes out of scope.
        # This is safer than manual cleanup and prevents memory leaks.
    
    def predict(self, data: np.ndarray, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict BMU indices for data using vectorized operations."""
        # Validate coordinates shape if provided
        if coordinates is not None:
            if coordinates.shape[0] != data.shape[0]:
                raise ValueError(f"Coordinates samples ({coordinates.shape[0]}) must match data samples ({data.shape[0]})")
            if coordinates.shape[1] != 2:
                raise ValueError(f"Coordinates must have 2 columns [longitude, latitude], got {coordinates.shape[1]}")
        
        n_samples = len(data)
        bmu_indices = np.full(n_samples, INVALID_INDEX, dtype=np.int32)
        
        # Process in chunks for memory efficiency
        prediction_chunk_size = self._get_config('prediction_chunk_size', 5000)
        chunk_size = min(prediction_chunk_size, n_samples)
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_data = data[start_idx:end_idx]
            chunk_coords = coordinates[start_idx:end_idx] if coordinates is not None else None
            
            # Vectorized distance calculation
            if coordinates is not None and hasattr(self, 'neuron_coords'):
                distances = self._vectorized_combined_distance(
                    chunk_data, self.weights, chunk_coords, self.neuron_coords
                )
            else:
                distances = self._vectorized_partial_bray_curtis_chunk(chunk_data, self.weights)
            
            # Find BMUs for valid samples
            valid_distances = ~np.all(np.isinf(distances), axis=1)
            if valid_distances.any():
                chunk_bmus = np.argmin(distances[valid_distances], axis=1)
                valid_indices = np.where(valid_distances)[0]
                bmu_indices[start_idx + valid_indices] = chunk_bmus
            
            # Log progress for large datasets
            if n_samples > DEFAULT_PROGRESS_LOG_INTERVAL and (end_idx % DEFAULT_PROGRESS_LOG_INTERVAL == 0 or end_idx == n_samples):
                logger.debug(f"Predicted {end_idx}/{n_samples} samples")
        
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