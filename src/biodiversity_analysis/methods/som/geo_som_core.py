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
from .constants import (
    INVALID_DISTANCE, INVALID_INDEX, INVALID_COORDINATE,
    MIN_VALID_FEATURES, INSUFFICIENT_FEATURES_MSG
)
from .minibatch_distance_calculator import MiniBatchDistanceCalculator
from .partial_metrics import partial_bray_curtis_numba, partial_euclidean_numba

logger = logging.getLogger(__name__)

# Configuration constants (can be overridden via config or environment variables)
DEFAULT_CHUNK_SIZE = int(os.environ.get('SOM_CHUNK_SIZE', '50000'))
DEFAULT_GEO_DISTANCE_NORM = float(os.environ.get('SOM_GEO_DISTANCE_NORM', '20000.0'))
DEFAULT_QE_SAMPLE_SIZE = int(os.environ.get('SOM_QE_SAMPLE_SIZE', '50000'))
DEFAULT_PROGRESS_LOG_INTERVAL = int(os.environ.get('SOM_PROGRESS_LOG_INTERVAL', '50000'))
MEMORY_THRESHOLD_BYTES = int(os.environ.get('SOM_MEMORY_THRESHOLD', '100000000'))


@dataclass
class GeoSOMConfig:
    """Configuration for GeoSOM + VLRSOM hybrid."""
    # Grid parameters
    grid_size: Tuple[int, int]
    topology: str = "rectangular"  # or "hexagonal"
    
    # GeoSOM parameters
    spatial_weight: float = 0.3  # k=0.3 (30% spatial, 70% features)
    geographic_distance: str = "haversine"
    
    # Mini-batch parameters for memory efficiency
    use_minibatch: bool = True
    minibatch_size: int = 1000  # Process 1000 samples at a time within chunks
    
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
    
    def __init__(self, config: GeoSOMConfig):
        self.config = config
        
        # Validate grid size configuration
        if not isinstance(config.grid_size, (list, tuple)) or len(config.grid_size) != 2:
            raise ValueError(f"grid_size must be a 2-element list/tuple, got: {config.grid_size}")
        
        if config.grid_size[0] <= 0 or config.grid_size[1] <= 0:
            raise ValueError(f"grid_size dimensions must be positive, got: {config.grid_size}")
        
        # Warn if grid is too large relative to expected data
        grid_neurons = config.grid_size[0] * config.grid_size[1]
        max_neurons_warning = getattr(config, 'max_grid_neurons_warning', 10000)
        if grid_neurons > max_neurons_warning:
            logger.warning(f"Large grid size ({config.grid_size[0]}x{config.grid_size[1]} = {grid_neurons} neurons) "
                         f"exceeds warning threshold ({max_neurons_warning}). "
                         f"May lead to slow training and memory issues")
        
        self.weights = None
        self.input_dim = None
        self.n_neurons = grid_neurons
        self.grid_rows, self.grid_cols = config.grid_size
        
        # Initialize grid positions
        self._init_grid_positions()
        
        # Initialize mini-batch distance calculator if enabled
        if config.use_minibatch:
            # Validate mini-batch size relative to chunk size
            if config.minibatch_size > config.chunk_size:
                logger.warning(f"Mini-batch size ({config.minibatch_size}) exceeds chunk size ({config.chunk_size}), adjusting...")
                config.minibatch_size = min(config.minibatch_size, config.chunk_size // 2)
            
            self.distance_calculator = MiniBatchDistanceCalculator(
                mini_batch_size=config.minibatch_size
            )
        else:
            self.distance_calculator = None
        
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
        
        Uses Numba-optimized implementation for better performance.
        Only compares non-NA pairs, requires minimum valid features.
        
        Args:
            u, v: Vectors to compare (may contain NaN)
            
        Returns:
            Distance value or INVALID_DISTANCE (np.nan) if insufficient valid pairs
        """
        # Delegate to Numba-optimized implementation from partial_metrics
        return partial_bray_curtis_numba(u, v, self.config.min_valid_features)
    
    def haversine_distance(self, coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Calculate haversine distance between geographic coordinates.
        
        Args:
            coord1, coord2: [longitude, latitude] in degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lon1, lat1 = np.radians(coord1)
        lon2, lat2 = np.radians(coord2)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        r = 6371
        return c * r
    
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
        
        # Weighted combination
        combined = (1 - self.config.spatial_weight) * feature_dist + \
                   self.config.spatial_weight * geo_dist_norm
        
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
        """Train using batch mode with geographic awareness.
        
        Args:
            data: Training data (n_samples, n_features)
            coordinates: Geographic coordinates (n_samples, 2) [longitude, latitude]
            progress_callback: Optional callback for progress updates
            
        Returns:
            Training result with history and final state
        """
        # Validate coordinates shape if provided
        if coordinates is not None:
            if coordinates.shape[0] != data.shape[0]:
                raise ValueError(f"Coordinates samples ({coordinates.shape[0]}) must match data samples ({data.shape[0]})")
            if coordinates.shape[1] != 2:
                raise ValueError(f"Coordinates must have 2 columns [longitude, latitude], got {coordinates.shape[1]}")
        
        if self.weights is None:
            self.initialize_weights(data, coordinates)
        
        converged = False
        best_qe = np.inf
        patience_counter = 0
        start_epoch = 0
        
        # Check for resume from checkpoint
        resume_checkpoint = getattr(self.config, 'resume_checkpoint', None)
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            checkpoint_data = self.load_checkpoint(resume_checkpoint)
            # Resume from NEXT epoch after checkpoint
            start_epoch = checkpoint_data['epoch'] + 1
            best_qe = checkpoint_data['best_qe']
            patience_counter = checkpoint_data['patience_counter']
            logger.info(f"Resuming training from epoch {start_epoch+1} (checkpoint was at epoch {checkpoint_data['epoch']+1})")
            # Training history already restored in load_checkpoint
        
        # Calculate initial QE on a sample for large datasets
        if len(data) > 10000:
            # Sample 10k points for initial QE calculation
            sample_idx = np.random.choice(len(data), size=10000, replace=False)
            sample_data = data[sample_idx]
            sample_coords = coordinates[sample_idx] if coordinates is not None else None
            initial_qe = self.calculate_quantization_error(sample_data, sample_coords)
            logger.info(f"Initial QE calculated on 10k sample: {initial_qe:.6f}")
        else:
            initial_qe = self.calculate_quantization_error(data, coordinates)
        
        # Only calculate initial QE if not resuming
        if start_epoch == 0:
            self.training_history['quantization_errors'].append(initial_qe)
            best_qe = initial_qe
        
        for epoch in range(start_epoch, self.config.max_epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.config.max_epochs}")
            
            # Decay radius
            self._update_radius(epoch)
            logger.debug(f"Updated radius to {self.current_radius:.2f}")
            
            # Batch update
            logger.info(f"Starting batch update for epoch {epoch+1}")
            self._batch_update(data, coordinates)
            logger.info(f"Completed batch update for epoch {epoch+1}")
            
            # Calculate QE after update with smart sampling
            logger.info(f"Calculating QE for epoch {epoch+1}")
            if len(data) > 100000:
                # Large datasets: use aggressive sampling
                qe_sample_freq = getattr(self.config.progress_tracking, 'qe_sample_frequency', 10) if hasattr(self.config, 'progress_tracking') else 10
                if epoch % qe_sample_freq == 0:
                    # Every N epochs: larger sample for accuracy
                    logger.debug("Using 50k sample for QE calculation (validation epoch)")
                    qe = self.calculate_quantization_error(data, coordinates, sample_size=DEFAULT_QE_SAMPLE_SIZE)
                else:
                    # Other epochs: smaller sample for speed
                    logger.debug("Using 10k sample for QE calculation")
                    qe = self.calculate_quantization_error(data, coordinates, sample_size=10000)
            elif len(data) > 10000:
                # Medium datasets: moderate sampling
                # Use configurable sample size
                sample_divisor = getattr(self.config.progress_tracking, 'quick_sample_divisor', 10) if hasattr(self.config, 'progress_tracking') else 10
                max_sample = getattr(self.config.progress_tracking, 'large_sample_size', 50000) if hasattr(self.config, 'progress_tracking') else 50000
                sample_size = min(len(data) // sample_divisor, max_sample)
                logger.debug(f"Using {sample_size} sample for QE calculation")
                qe = self.calculate_quantization_error(data, coordinates, sample_size=sample_size)
            else:
                # Small datasets: use all data
                logger.debug("Using full dataset for QE calculation")
                qe = self.calculate_quantization_error(data, coordinates)
            logger.info(f"QE calculated for epoch {epoch+1}: {qe:.6f}")
            
            self.training_history['quantization_errors'].append(qe)
            
            # Log progress every 10 epochs or on significant events
            if epoch % 10 == 0 or epoch < 5:
                logger.info(f"Epoch {epoch+1}/{self.config.max_epochs}: QE={qe:.6f}, "
                           f"LR={self.current_lr:.4f}, Radius={self.current_radius:.2f}")
            
            # VLRSOM: Adapt learning rate using proper error function
            self._adapt_learning_rate(data, coordinates)
            
            # Update best QE
            if qe < best_qe:
                improvement = (best_qe - qe) / best_qe if best_qe != np.inf else 1.0
                if improvement < self.config.qe_improvement_threshold:
                    patience_counter += 1
                else:
                    patience_counter = 0
                best_qe = qe
            else:
                patience_counter += 1
            
            # Track learning rate and radius
            self.training_history['learning_rates'].append(self.current_lr)
            self.training_history['radii'].append(self.current_radius)
            
            # Calculate topographic error every N epochs
            te_freq = getattr(self.config.progress_tracking, 'te_calculation_frequency', 10) if hasattr(self.config, 'progress_tracking') else 10
            if epoch % te_freq == 0 and epoch > 0:
                logger.info(f"Calculating topographic error for epoch {epoch+1}")
                te = self.calculate_topographic_error(data, coordinates, sample_size=10000)
                logger.info(f"Topographic error: {te:.4f}")
                self.training_history['topographic_errors'].append(te)
                
                # Adjust radius based on TE if enabled
                te_radius_adjustment = getattr(self.config, 'te_radius_adjustment', True)
                if te_radius_adjustment:
                    # Get scheduled radius for next epoch
                    next_epoch_radius = self._get_scheduled_radius(epoch + 1)
                    
                    # Adjust based on TE thresholds
                    te_high_threshold = getattr(self.config, 'te_high_threshold', 0.15)
                    te_low_threshold = getattr(self.config, 'te_low_threshold', 0.05)
                    
                    if te > te_high_threshold:
                        # High TE: slow down radius decay
                        radius_modifier = getattr(self.config, 'te_high_radius_modifier', 1.1)
                        adjusted_radius = next_epoch_radius * radius_modifier
                        logger.info(f"High TE ({te:.3f} > {te_high_threshold}): adjusting radius {next_epoch_radius:.3f} → {adjusted_radius:.3f}")
                    elif te < te_low_threshold:
                        # Low TE: can decay faster
                        radius_modifier = getattr(self.config, 'te_low_radius_modifier', 0.95)
                        adjusted_radius = next_epoch_radius * radius_modifier
                        logger.info(f"Low TE ({te:.3f} < {te_low_threshold}): adjusting radius {next_epoch_radius:.3f} → {adjusted_radius:.3f}")
                    else:
                        # Normal TE: use scheduled radius
                        adjusted_radius = next_epoch_radius
                    
                    # Apply bounds
                    adjusted_radius = np.clip(adjusted_radius, self.config.final_radius, self.config.initial_radius)
                    
                    # Store adjustment for next update
                    self._radius_adjustment = adjusted_radius / next_epoch_radius
                else:
                    self._radius_adjustment = 1.0
            
            # Calculate geographic coherence
            # Calculate geographic coherence every N epochs
            geo_freq = getattr(self.config.progress_tracking, 'geographic_coherence_frequency', 10) if hasattr(self.config, 'progress_tracking') else 10
            if coordinates is not None and epoch % geo_freq == 0 and epoch > 0:
                logger.info(f"Calculating geographic coherence for epoch {epoch+1}")
                # Sample for large datasets to avoid O(n²) computation
                coherence_sample_threshold = getattr(self.config, 'coherence_sample_threshold', 10000)
                coherence_sample_size = getattr(self.config, 'coherence_sample_size', 5000)
                if len(data) > coherence_sample_threshold:
                    sample_size = coherence_sample_size
                    indices = np.random.choice(len(data), sample_size, replace=False)
                    geo_coherence = self._calculate_geographic_coherence(
                        data[indices], coordinates[indices]
                    )
                    logger.info(f"Geographic coherence (sampled): {geo_coherence:.4f}")
                else:
                    geo_coherence = self._calculate_geographic_coherence(data, coordinates)
                    logger.info(f"Geographic coherence: {geo_coherence:.4f}")
                self.training_history['geographic_coherence'].append(geo_coherence)
                
                # Save checkpoint if enabled
                # Access nested config properly
                progress_config = getattr(self.config, 'progress_tracking', {})
                if isinstance(progress_config, dict):
                    checkpoint_frequency = progress_config.get('checkpoint_frequency', 10)
                    checkpoint_dir = progress_config.get('checkpoint_dir', 'checkpoints/som')
                else:
                    checkpoint_frequency = getattr(progress_config, 'checkpoint_frequency', 10)
                    checkpoint_dir = getattr(progress_config, 'checkpoint_dir', 'checkpoints/som')
                
                logger.info(f"Checkpoint check: frequency={checkpoint_frequency}, epoch={epoch}, "
                           f"should_save={(epoch) % checkpoint_frequency == 0}")
                if checkpoint_frequency > 0 and epoch % checkpoint_frequency == 0:
                    self.save_checkpoint(checkpoint_dir, epoch, best_qe, patience_counter)
                
                # Check convergence
                if self._check_convergence(geo_coherence, patience_counter):
                    converged = True
                    logger.info(f"Converged at epoch {epoch}")
                    break
            
            # Progress callback with detailed info
            if progress_callback:
                logger.debug(f"Calling progress callback for epoch {epoch+1}")
                progress_info = {
                    'epoch': epoch + 1,
                    'max_epochs': self.config.max_epochs,
                    'progress': (epoch + 1) / self.config.max_epochs,
                    'qe': qe,  # Use the calculated qe from line 375
                    'learning_rate': self.current_lr,
                    'radius': self.current_radius
                }
                # Support both simple and detailed callbacks
                try:
                    progress_callback(progress_info)
                    logger.debug(f"Progress callback completed for epoch {epoch+1}")
                except TypeError:
                    # Fallback to simple progress
                    logger.debug(f"Using fallback progress callback for epoch {epoch+1}")
                    progress_callback((epoch + 1) / self.config.max_epochs)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")
                    raise
            
            # Checkpoint saving is now handled by the unified checkpoint system
            # The analyzer's save_checkpoint_state() method will be called externally
            # checkpoint_freq = getattr(self.config, 'checkpoint_frequency', 10)
            # if (epoch + 1) % checkpoint_freq == 0:
            #     checkpoint_dir = kwargs.get('checkpoint_dir', 
            #         f'outputs/analysis_results/som/{kwargs.get("experiment_name", "biodiv_fine_som_grid10")}/checkpoints')
            #     # Use standardized save_checkpoint method with all required parameters
            #     self.save_checkpoint(
            #         checkpoint_dir=checkpoint_dir,
            #         epoch=epoch + 1,
            #         best_qe=best_qe,
            #         patience_counter=patience_counter
            #     )
            
            logger.info(f"Completed epoch {epoch+1}/{self.config.max_epochs}")
        
        # Create result
        return SOMTrainingResult(
            weights=self.weights.reshape(self.grid_rows, self.grid_cols, self.input_dim),
            quantization_errors=self.training_history['quantization_errors'],
            topographic_errors=[],  # Not used in this implementation
            training_time=0.0,  # Would need timing
            convergence_epoch=epoch if converged else None,
            final_learning_rate=self.current_lr,
            final_neighborhood_radius=self.current_radius,
            n_samples_trained=len(data)
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
        valid_mask = valid_features >= self.config.min_valid_features
        
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
        alpha = getattr(self.config, 'vlrsom_alpha', 0.95)  # Decay factor (0 < α < 1)
        gamma = getattr(self.config, 'vlrsom_gamma', 0.001)  # Adaptation parameter (γ > 0)
        
        # Calculate J²(t) using existing sampling infrastructure
        error_sample_size = getattr(self.config, 'vlrsom_error_sample_size', 10000)
        j_squared = self._calculate_vlrsom_error_squared(data, coordinates, error_sample_size)
        
        # Apply VLRSOM formula: μ'(t+1) = αμ(t) + γJ²(t)
        new_lr = alpha * self.current_lr + gamma * j_squared
        
        # Apply bounds from paper Equation (21)
        min_lr = getattr(self.config, 'vlrsom_min_lr', self.config.min_learning_rate)
        max_lr = getattr(self.config, 'vlrsom_max_lr', self.config.max_learning_rate)
        
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
    
    def _batch_update(self, data: np.ndarray, coordinates: Optional[np.ndarray]):
        """Perform batch update of weights using vectorized operations."""
        # Use vectorized implementation for large datasets
        vectorized_threshold = getattr(self.config, 'vectorized_threshold', 1000)
        if len(data) > vectorized_threshold:
            self._batch_update_vectorized(data, coordinates)
        else:
            self._batch_update_sequential(data, coordinates)
    
    def _batch_update_vectorized(self, data: np.ndarray, coordinates: Optional[np.ndarray]):
        """Vectorized batch update with chunking for memory efficiency.
        
        This implementation includes geographic distance calculations as required
        by the GeoSOM specification (30% spatial, 70% features).
        
        If use_parallel is enabled in config, chunks will be processed in parallel.
        """
        n_samples, n_features = data.shape
        
        # Check if parallel processing is enabled and beneficial
        min_samples_for_parallel = getattr(self.config, 'min_samples_for_parallel', 100000)
        if self.config.use_parallel and n_samples > min_samples_for_parallel:
            logger.info(f"Starting PARALLEL batch update for {n_samples} samples")
            # No try-catch, no fallback - parallel must work or fail completely
            self._batch_update_parallel(data, coordinates)
            return
        
        # This should only be reached for small datasets
        if n_samples > 1000000:  # 1M samples
            raise RuntimeError(f"Dataset too large ({n_samples} samples) for sequential processing. "
                             f"Parallel processing is required but not enabled or failed.")
        
        logger.info(f"Starting chunked vectorized batch update for {n_samples} samples")
        
        # Initialize accumulators for batch updates
        numerator = np.zeros_like(self.weights)
        denominator = np.zeros(self.n_neurons)
        
        # Process in chunks to manage memory
        # Use chunk size from config if available
        chunk_size = getattr(self.config, 'chunk_size', None)
        if chunk_size is None:
            # Try to get from training config
            chunk_size = 50000  # Increased default from 10k to 50k for better performance
        chunk_size = min(chunk_size, n_samples)  # Don't exceed data size
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
            
            # Skip empty chunks
            if start_idx >= n_samples:
                logger.debug(f"Skipping chunk {chunk_idx+1}/{n_chunks} - beyond data size")
                continue
                
            chunk_data = data[start_idx:end_idx]
            chunk_coords = coordinates[start_idx:end_idx] if coordinates is not None else None
            
            # Check if chunk has any valid data
            valid_features_per_sample = (~np.isnan(chunk_data)).sum(axis=1)
            if (valid_features_per_sample >= self.config.min_valid_features).sum() == 0:
                logger.debug(f"Skipping chunk {chunk_idx+1}/{n_chunks} - no samples with enough valid features")
                continue
            
            # Calculate distances for this chunk
            if coordinates is not None and hasattr(self, 'neuron_coords'):
                # Combined distance calculation (feature + geographic)
                chunk_distances = self._vectorized_combined_distance(
                    chunk_data, self.weights, chunk_coords, self.neuron_coords
                )
            else:
                # Feature distance only
                chunk_distances = self._vectorized_partial_bray_curtis_chunk(chunk_data, self.weights)
            
            # Find BMUs for chunk
            chunk_bmu_indices = np.argmin(chunk_distances, axis=1)
            min_distances = np.take_along_axis(
                chunk_distances, chunk_bmu_indices[:, np.newaxis], axis=1
            ).squeeze()
            
            # Mark invalid BMUs (samples with no valid distances to any neuron)
            invalid_mask = np.isinf(min_distances)
            chunk_bmu_indices[invalid_mask] = -1
            
            # Skip this chunk if all samples are invalid
            if invalid_mask.all():
                logger.debug(f"Skipping chunk {chunk_idx+1}/{n_chunks} - all samples have invalid distances")
                continue
            
            # Calculate neighborhood influences for chunk
            chunk_influences = self._vectorized_neighborhood(chunk_bmu_indices)
            
            # Accumulate updates from this chunk
            chunk_data_clean = np.nan_to_num(chunk_data, nan=0.0)
            chunk_valid_mask = ~np.isnan(chunk_data)
            
            # Accumulate numerator and denominator
            numerator += np.einsum('ij,ik,ik->jk', chunk_influences, chunk_data_clean, chunk_valid_mask)
            denominator += np.einsum('ij,ik->j', chunk_influences, chunk_valid_mask)
            
            # Log progress for large datasets
            if n_chunks > 10 and (chunk_idx + 1) % max(1, n_chunks // 20) == 0:
                progress = (chunk_idx + 1) / n_chunks
                logger.info(f"Batch update progress: {progress:.0%} ({chunk_idx + 1}/{n_chunks} chunks)")
            
            # Extra logging for debugging last chunks
            if chunk_idx >= n_chunks - 10:
                logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks}: {end_idx - start_idx} samples, "
                          f"valid samples: {(~invalid_mask).sum()}")
        
        # Apply accumulated updates
        update_mask = denominator > n_features
        
        if np.any(update_mask):
            targets = numerator[update_mask] / denominator[update_mask, np.newaxis]
            momentum = np.minimum(denominator[update_mask] / (n_samples * n_features), 1.0)
            effective_lr = momentum * self.current_lr
            
            # Vectorized weight update
            for j, neuron_idx in enumerate(np.where(update_mask)[0]):
                valid = ~(np.isnan(targets[j]) | np.isnan(self.weights[neuron_idx]))
                self.weights[neuron_idx, valid] = (
                    (1 - effective_lr[j]) * self.weights[neuron_idx, valid] +
                    effective_lr[j] * targets[j, valid]
                )
    
    def _vectorized_partial_bray_curtis_chunk(self, data_chunk: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Vectorized Bray-Curtis distance calculation for a chunk of data."""
        # Use mini-batch calculator if enabled
        if self.config.use_minibatch and self.distance_calculator is not None:
            return self.distance_calculator.partial_bray_curtis_minibatch(
                data_chunk, weights, self.config.min_valid_features
            )
        
        # Original implementation (memory-intensive)
        # Expand dimensions for broadcasting
        data_exp = data_chunk[:, np.newaxis, :]
        weights_exp = weights[np.newaxis, :, :]
        
        # Find valid pairs
        valid_mask = ~(np.isnan(data_exp) | np.isnan(weights_exp))
        n_valid = valid_mask.sum(axis=2)
        
        # Clean data for calculation
        data_clean = np.nan_to_num(data_exp, nan=0.0)
        weights_clean = np.nan_to_num(weights_exp, nan=0.0)
        
        # Bray-Curtis calculation
        diff = np.abs(data_clean - weights_clean)
        sum_vals = data_clean + weights_clean
        
        numerator = (diff * valid_mask).sum(axis=2)
        denominator = (sum_vals * valid_mask).sum(axis=2)
        
        # Handle division
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = numerator / denominator
        
        # Set invalid distances
        invalid = (n_valid < self.config.min_valid_features) | (denominator == 0)
        distances[invalid] = np.inf
        
        return distances
    
    def _vectorized_combined_distance(self, data_chunk: np.ndarray, weights: np.ndarray,
                                    coords_chunk: np.ndarray, neuron_coords: np.ndarray) -> np.ndarray:
        """Calculate combined feature and geographic distances for a chunk.
        
        Implements the GeoSOM specification: 30% spatial + 70% feature distance.
        """
        # Use mini-batch calculator if enabled
        if self.config.use_minibatch and self.distance_calculator is not None:
            return self.distance_calculator.combined_distance_minibatch(
                data_chunk, weights, coords_chunk, neuron_coords,
                self.config.spatial_weight, self.config.min_valid_features
            )
        
        # Original implementation
        # Feature distances
        feature_distances = self._vectorized_partial_bray_curtis_chunk(data_chunk, weights)
        
        # Geographic distances (vectorized haversine)
        geo_distances = self._vectorized_haversine_distances(coords_chunk, neuron_coords)
        
        # Normalize geographic distances (rough earth circumference / 2)
        geo_distances_norm = geo_distances / DEFAULT_GEO_DISTANCE_NORM
        
        # Combined distance with spatial weighting
        # Handle invalid feature distances
        valid_feature = ~np.isinf(feature_distances)
        combined = np.full_like(feature_distances, np.inf)
        combined[valid_feature] = (
            (1 - self.config.spatial_weight) * feature_distances[valid_feature] +
            self.config.spatial_weight * geo_distances_norm[valid_feature]
        )
        
        return combined
    
    def _vectorized_haversine_distances(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Vectorized haversine distance calculation between coordinate sets.
        
        Args:
            coords1: (n_samples, 2) array of [longitude, latitude] in degrees
            coords2: (n_neurons, 2) array of [longitude, latitude] in degrees
            
        Returns:
            (n_samples, n_neurons) array of distances in kilometers
        """
        # Convert to radians
        coords1_rad = np.radians(coords1)
        coords2_rad = np.radians(coords2)
        
        # Extract lat/lon
        lat1 = coords1_rad[:, 1:2]  # Keep as (n, 1) for broadcasting
        lon1 = coords1_rad[:, 0:1]
        lat2 = coords2_rad[:, 1].reshape(1, -1)  # Shape (1, m)
        lon2 = coords2_rad[:, 0].reshape(1, -1)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # clip for numerical stability
        
        # Earth radius in km
        return 6371 * c
    
    def _vectorized_neighborhood(self, bmu_indices: np.ndarray) -> np.ndarray:
        """Calculate neighborhood influences for all BMUs at once."""
        n_samples = len(bmu_indices)
        influences = np.zeros((n_samples, self.n_neurons))
        
        valid_mask = bmu_indices >= 0
        valid_bmus = bmu_indices[valid_mask]
        
        if len(valid_bmus) > 0:
            bmu_positions = self._grid_positions[valid_bmus]
            pos_diff = bmu_positions[:, np.newaxis, :] - self._grid_positions[np.newaxis, :, :]
            distances_sq = np.sum(pos_diff ** 2, axis=2)
            neighborhood = np.exp(-distances_sq / (2 * self.current_radius ** 2))
            influences[valid_mask] = neighborhood
        
        return influences
    
    def _batch_update_parallel(self, data: np.ndarray, coordinates: Optional[np.ndarray]):
        """Parallel batch update using multiple CPU cores.
        
        This method distributes chunks across multiple workers for faster processing
        on systems with many CPU cores.
        """
        from .parallel_batch_processor import ParallelBatchProcessor
        
        n_samples, n_features = data.shape
        
        # Get chunk size
        chunk_size = getattr(self.config, 'chunk_size', DEFAULT_CHUNK_SIZE)
        chunk_size = min(chunk_size, n_samples)
        
        # Create a new processor instance for each training run to ensure isolation
        # This prevents thread safety issues and ensures clean state
        processor = ParallelBatchProcessor(n_workers=self.config.n_workers)
        
        # Note: We create a new processor per call rather than reusing to avoid:
        # 1. Thread safety issues in multi-threaded environments
        # 2. State contamination between training runs
        # 3. Memory leaks from accumulated state
        # The overhead is minimal as processor creation is lightweight
        
        # Process chunks in parallel
        numerator, denominator = processor.process_chunks_parallel(
            data=data,
            coordinates=coordinates,
            weights=self.weights,
            neuron_coords=getattr(self, 'neuron_coords', None),
            config=self.config,
            grid_positions=self._grid_positions,
            current_radius=self.current_radius,
            chunk_size=chunk_size
        )
        
        # Apply accumulated updates (same as vectorized version)
        update_mask = denominator > n_features
        
        if np.any(update_mask):
            targets = numerator[update_mask] / denominator[update_mask, np.newaxis]
            momentum = np.minimum(denominator[update_mask] / (n_samples * n_features), 1.0)
            effective_lr = momentum * self.current_lr
            
            # Vectorized weight update
            for j, neuron_idx in enumerate(np.where(update_mask)[0]):
                valid = ~(np.isnan(targets[j]) | np.isnan(self.weights[neuron_idx]))
                self.weights[neuron_idx, valid] = (
                    (1 - effective_lr[j]) * self.weights[neuron_idx, valid] +
                    effective_lr[j] * targets[j, valid]
                )
    
    def _batch_update_sequential(self, data: np.ndarray, coordinates: Optional[np.ndarray]):
        """Original sequential batch update for small datasets."""
        # Initialize accumulators
        numerator = np.zeros_like(self.weights)
        denominator = np.zeros(self.n_neurons)
        
        # Process each sample
        for idx, sample in enumerate(data):
            # Find BMU considering geographic distance
            if coordinates is not None:
                bmu_idx = self._find_bmu_geo(sample, coordinates[idx])
            else:
                bmu_idx = self._find_bmu(sample)
            
            # Skip if no valid BMU found
            if bmu_idx == INVALID_INDEX:
                continue
            
            # Calculate neighborhood influence
            influences = self._calculate_neighborhood(bmu_idx)
            
            # Accumulate updates
            for j in range(self.n_neurons):
                if influences[j] > 0:
                    # Only update with valid values
                    valid = ~np.isnan(sample)
                    numerator[j, valid] += influences[j] * sample[valid]
                    denominator[j] += influences[j]
        
        # Apply updates with momentum based on sample count
        for j in range(self.n_neurons):
            if denominator[j] > 1:  # Only update if at least 2 samples assigned
                # Calculate target position
                target = numerator[j] / denominator[j]
                
                # Adaptive momentum based on how many samples assigned
                # More samples = more confidence in update
                momentum = min(denominator[j] / len(data), 1.0)
                effective_lr = momentum * self.current_lr  # Scale down LR
                
                # Only update non-NaN components
                for k in range(self.input_dim):
                    if not np.isnan(target[k]) and not np.isnan(self.weights[j, k]):
                        self.weights[j, k] = (1 - effective_lr) * self.weights[j, k] + \
                                           effective_lr * target[k]
    
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
        if self.config.neighborhood_function == "gaussian":
            bmu_pos = self._grid_positions[bmu_idx]
            distances_squared = np.sum(
                (self._grid_positions - bmu_pos)**2, axis=1
            )
            influences = np.exp(-distances_squared / (2 * self.current_radius**2))
            return influences
        else:
            raise NotImplementedError(
                f"Neighborhood {self.config.neighborhood_function} not implemented"
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
        valid_mask = valid_features >= self.config.min_valid_features
        
        if not valid_mask.any():
            return np.inf
        
        data_valid = data[valid_mask]
        coords_valid = coordinates[valid_mask] if coordinates is not None else None
        
        # Process in chunks for memory efficiency
        chunk_size = getattr(self.config, 'qe_calc_chunk_size', 1000)
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
        valid_mask = valid_features >= self.config.min_valid_features
        
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
        
        Uses vectorized operations for O(n²) distance calculations to improve
        performance on large datasets.
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
    
    def _check_convergence(self, geo_coherence: float, patience_counter: int) -> bool:
        """Check convergence criteria - requires BOTH QE stability AND geographic coherence."""
        converged_qe = False
        converged_geo = False
        
        # Check QE stability over window (new criterion)
        qe_stability_window = getattr(self.config, 'qe_stability_window', 20)
        qe_stability_threshold = getattr(self.config, 'qe_stability_threshold', 0.001)
        
        if len(self.training_history['quantization_errors']) >= qe_stability_window:
            recent_qes = self.training_history['quantization_errors'][-qe_stability_window:]
            qe_range = max(recent_qes) - min(recent_qes)
            qe_relative_range = qe_range / np.mean(recent_qes)
            
            if qe_relative_range < qe_stability_threshold:
                logger.info(f"QE stable over {qe_stability_window} epochs (range = {qe_relative_range:.5f} < {qe_stability_threshold})")
                converged_qe = True
        
        # Alternative: Check QE improvement threshold (original criterion)
        if not converged_qe and len(self.training_history['quantization_errors']) > 10:
            recent_qes = self.training_history['quantization_errors'][-10:]
            qe_improvement = abs(recent_qes[-1] - recent_qes[0]) / recent_qes[0]
            if qe_improvement < self.config.qe_improvement_threshold:
                logger.info(f"QE stabilized (improvement = {qe_improvement:.5f} < {self.config.qe_improvement_threshold})")
                converged_qe = True
        
        # Geographic coherence threshold
        if geo_coherence > self.config.geographic_coherence_threshold:
            logger.info(f"Geographic coherence {geo_coherence:.3f} > threshold {self.config.geographic_coherence_threshold}")
            converged_geo = True
        
        # Require BOTH criteria to be met
        if converged_qe and converged_geo:
            logger.info("Both QE stability and geographic coherence criteria met - converged!")
            return True
        elif converged_qe:
            logger.info("QE converged but geographic coherence not met - continuing...")
        elif converged_geo:
            logger.info("Geographic coherence met but QE not stable - continuing...")
        
        # Alternative: Patience exceeded with reasonable QE
        if patience_counter >= self.config.patience:
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
        
        # Load checkpoint
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        
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
        prediction_chunk_size = getattr(self.config, 'prediction_chunk_size', 5000)
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