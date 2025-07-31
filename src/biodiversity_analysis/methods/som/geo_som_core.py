"""GeoSOM + VLRSOM hybrid implementation for biodiversity analysis.

This implementation follows the specifications in final_som_configuration_decisions.md:
- Handles 70% missing data with partial Bray-Curtis distance
- Integrates spatial information (30% spatial, 70% features)
- Uses adaptive learning rate based on QE improvement
- Implements geographic coherence for convergence
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist
# Moran's I calculation is in spatial_utils module

from src.abstractions.types.som_types import SOMConfig, SOMTrainingResult

logger = logging.getLogger(__name__)


@dataclass
class GeoSOMConfig:
    """Configuration for GeoSOM + VLRSOM hybrid."""
    # Grid parameters
    grid_size: Tuple[int, int]
    topology: str = "rectangular"  # or "hexagonal"
    
    # GeoSOM parameters
    spatial_weight: float = 0.3  # k=0.3 (30% spatial, 70% features)
    geographic_distance: str = "haversine"
    
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


class GeoSOMVLRSOM:
    """Geographic Self-Organizing Map with Variable Learning Rate SOM.
    
    This hybrid architecture combines:
    1. GeoSOM: Incorporates spatial information into the distance calculation
    2. VLRSOM: Adapts learning rate based on quantization error improvement
    
    Designed for biodiversity data with high missing values and spatial structure.
    """
    
    def __init__(self, config: GeoSOMConfig):
        self.config = config
        self.weights = None
        self.input_dim = None
        self.n_neurons = config.grid_size[0] * config.grid_size[1]
        self.grid_rows, self.grid_cols = config.grid_size
        
        # Initialize grid positions
        self._init_grid_positions()
        
        # Training state
        self.current_lr = config.initial_learning_rate
        self.current_radius = None
        self.training_history = {
            'quantization_errors': [],
            'learning_rates': [],
            'radii': [],
            'geographic_coherence': [],
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
    
    def partial_bray_curtis(self, u: np.ndarray, v: np.ndarray) -> float:
        """Bray-Curtis distance for vectors with missing values.
        
        Only compares non-NA pairs, requires minimum valid features.
        
        Args:
            u, v: Vectors to compare (may contain NaN)
            
        Returns:
            Distance value or NaN if insufficient valid pairs
        """
        valid = ~(np.isnan(u) | np.isnan(v))
        
        if valid.sum() < self.config.min_valid_features:
            return np.nan
        
        u_valid = u[valid]
        v_valid = v[valid]
        
        # Avoid division by zero
        denominator = np.sum(u_valid + v_valid)
        if denominator == 0:
            return 0.0 if np.allclose(u_valid, v_valid) else 1.0
        
        return np.sum(np.abs(u_valid - v_valid)) / denominator
    
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
        
        # Handle case where feature distance is NaN
        if np.isnan(feature_dist):
            return np.inf
        
        # Geographic distance (normalized)
        if sample_coord is not None and neuron_coord is not None:
            geo_dist = self.haversine_distance(sample_coord, neuron_coord)
            # Normalize geographic distance (rough earth circumference / 2)
            geo_dist_norm = geo_dist / 20000.0
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
            coordinates: Geographic coordinates (n_samples, 2) [lat, lon]
            method: Initialization method
        """
        self.input_dim = data.shape[1]
        
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
                   progress_callback: Optional[Callable] = None) -> SOMTrainingResult:
        """Train using batch mode with geographic awareness.
        
        Args:
            data: Training data (n_samples, n_features)
            coordinates: Geographic coordinates (n_samples, 2)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Training result with history and final state
        """
        if self.weights is None:
            self.initialize_weights(data, coordinates)
        
        converged = False
        best_qe = np.inf
        patience_counter = 0
        
        # Calculate initial QE
        initial_qe = self.calculate_quantization_error(data, coordinates)
        self.training_history['quantization_errors'].append(initial_qe)
        best_qe = initial_qe
        
        for epoch in range(self.config.max_epochs):
            # Decay radius
            self._update_radius(epoch)
            
            # Batch update
            self._batch_update(data, coordinates)
            
            # Calculate QE after update
            qe = self.calculate_quantization_error(data, coordinates)
            self.training_history['quantization_errors'].append(qe)
            
            # VLRSOM: Adapt learning rate based on QE improvement
            self._adapt_learning_rate(qe, best_qe)
            
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
            
            # Calculate geographic coherence
            if coordinates is not None and epoch % 10 == 0:
                geo_coherence = self._calculate_geographic_coherence(data, coordinates)
                self.training_history['geographic_coherence'].append(geo_coherence)
                
                # Check convergence
                if self._check_convergence(geo_coherence, patience_counter):
                    converged = True
                    logger.info(f"Converged at epoch {epoch}")
                    break
            
            # Progress callback
            if progress_callback:
                progress_callback((epoch + 1) / self.config.max_epochs)
        
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
    
    def _adapt_learning_rate(self, current_qe: float, best_qe: float):
        """VLRSOM: Adapt learning rate based on QE improvement."""
        # Calculate improvement ratio
        if best_qe > 0:
            improvement = (best_qe - current_qe) / best_qe
        else:
            improvement = 0
        
        # More nuanced adaptation based on improvement magnitude
        if improvement > 0.05:  # Significant improvement (>5%)
            # Increase learning rate
            self.current_lr = min(
                self.current_lr * 1.05,  # Gentle increase
                self.config.max_learning_rate
            )
        elif improvement > 0:  # Small improvement (0-5%)
            # Keep learning rate stable
            pass
        elif improvement > -0.05:  # Small degradation (0-5%)
            # Slightly decrease learning rate
            self.current_lr = max(
                self.current_lr * 0.98,  # Very gentle decrease
                self.config.min_learning_rate
            )
        else:  # Significant degradation (>5%)
            # Decrease learning rate more
            self.current_lr = max(
                self.current_lr * 0.95,
                self.config.min_learning_rate
            )
        
        # Adaptive regions based on QE level
        if current_qe > 0.5:  # High QE region
            self.current_lr = np.clip(
                self.current_lr,
                self.config.high_qe_lr_min,
                self.config.high_qe_lr_max
            )
        elif current_qe < 0.1:  # Low QE region
            self.current_lr = np.clip(
                self.current_lr,
                self.config.low_qe_lr_min,
                self.config.low_qe_lr_max
            )
    
    def _update_radius(self, epoch: int):
        """Update neighborhood radius."""
        if self.config.radius_decay == "linear":
            progress = epoch / self.config.max_epochs
            self.current_radius = self.config.initial_radius * (1 - progress) + \
                                 self.config.final_radius * progress
        else:
            raise NotImplementedError(f"Decay {self.config.radius_decay} not implemented")
    
    def _batch_update(self, data: np.ndarray, coordinates: Optional[np.ndarray]):
        """Perform batch update of weights."""
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
            if bmu_idx is None:
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
                effective_lr = momentum * self.current_lr * 0.5  # Scale down LR
                
                # Only update non-NaN components
                for k in range(self.input_dim):
                    if not np.isnan(target[k]) and not np.isnan(self.weights[j, k]):
                        self.weights[j, k] = (1 - effective_lr) * self.weights[j, k] + \
                                           effective_lr * target[k]
    
    def _find_bmu(self, sample: np.ndarray) -> Optional[int]:
        """Find best matching unit using partial Bray-Curtis."""
        distances = []
        for i, weight in enumerate(self.weights):
            dist = self.partial_bray_curtis(sample, weight)
            if not np.isnan(dist):
                distances.append((i, dist))
        
        if not distances:
            return None
        
        return min(distances, key=lambda x: x[1])[0]
    
    def _find_bmu_geo(self, sample: np.ndarray, coord: np.ndarray) -> Optional[int]:
        """Find BMU considering geographic distance."""
        distances = []
        for i, weight in enumerate(self.weights):
            dist = self.combined_distance(
                sample, weight, coord, self.neuron_coords[i]
            )
            if not np.isinf(dist):
                distances.append((i, dist))
        
        if not distances:
            return None
        
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
                                   coordinates: Optional[np.ndarray] = None) -> float:
        """Calculate average quantization error."""
        total_error = 0.0
        n_valid = 0
        
        for idx, sample in enumerate(data):
            if coordinates is not None:
                bmu_idx = self._find_bmu_geo(sample, coordinates[idx])
                if bmu_idx is not None:
                    error = self.combined_distance(
                        sample, self.weights[bmu_idx],
                        coordinates[idx], self.neuron_coords[bmu_idx]
                    )
                    if not np.isinf(error):
                        total_error += error
                        n_valid += 1
            else:
                bmu_idx = self._find_bmu(sample)
                if bmu_idx is not None:
                    error = self.partial_bray_curtis(sample, self.weights[bmu_idx])
                    if not np.isnan(error):
                        total_error += error
                        n_valid += 1
        
        return total_error / n_valid if n_valid > 0 else np.inf
    
    def _calculate_geographic_coherence(self, data: np.ndarray, 
                                      coordinates: np.ndarray) -> float:
        """Calculate geographic coherence using Moran's I."""
        # Get cluster assignments
        clusters = []
        for idx, sample in enumerate(data):
            bmu_idx = self._find_bmu_geo(sample, coordinates[idx])
            if bmu_idx is not None:
                clusters.append(bmu_idx)
            else:
                clusters.append(-1)
        
        clusters = np.array(clusters)
        valid = clusters >= 0
        
        if valid.sum() < 10:  # Need minimum samples
            return 0.0
        
        # Calculate spatial weights matrix (inverse distance)
        coords_valid = coordinates[valid]
        n = len(coords_valid)
        W = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.haversine_distance(coords_valid[i], coords_valid[j])
                if dist > 0:
                    W[i, j] = W[j, i] = 1.0 / dist
        
        # Normalize weights matrix
        W_sum = W.sum()
        if W_sum > 0:
            W = W / W_sum
        else:
            # If no spatial weights (e.g., all samples at same location), return 0
            return 0.0
        
        # Calculate Moran's I
        clusters_valid = clusters[valid]
        mean_cluster = clusters_valid.mean()
        
        numerator = 0.0
        for i in range(n):
            for j in range(n):
                numerator += W[i, j] * (clusters_valid[i] - mean_cluster) * \
                            (clusters_valid[j] - mean_cluster)
        
        denominator = np.sum((clusters_valid - mean_cluster)**2) / n
        
        if denominator == 0:
            return 0.0
        
        morans_i = numerator / denominator
        return morans_i
    
    def _check_convergence(self, geo_coherence: float, patience_counter: int) -> bool:
        """Check convergence criteria."""
        # Geographic coherence threshold
        if geo_coherence > self.config.geographic_coherence_threshold:
            logger.info(f"Geographic coherence {geo_coherence:.3f} > threshold")
            return True
        
        # Learning rate stability
        if len(self.training_history['learning_rates']) > 10:
            recent_lrs = self.training_history['learning_rates'][-10:]
            lr_std = np.std(recent_lrs) / np.mean(recent_lrs)
            if lr_std < self.config.lr_stability_threshold:
                logger.info(f"Learning rate stabilized (std/mean = {lr_std:.3f})")
                return True
        
        # Patience exceeded
        if patience_counter >= self.config.patience:
            logger.info(f"Patience exceeded ({patience_counter} epochs)")
            return True
        
        return False
    
    def get_weights(self) -> np.ndarray:
        """Get weight matrix in grid shape."""
        return self.weights.reshape(self.grid_rows, self.grid_cols, self.input_dim)
    
    def predict(self, data: np.ndarray, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict BMU indices for data."""
        bmu_indices = []
        
        for idx, sample in enumerate(data):
            if coordinates is not None:
                bmu_idx = self._find_bmu_geo(sample, coordinates[idx])
            else:
                bmu_idx = self._find_bmu(sample)
            
            bmu_indices.append(bmu_idx if bmu_idx is not None else -1)
        
        return np.array(bmu_indices)
    
    def transform(self, data: np.ndarray, coordinates: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform data to grid coordinates."""
        bmu_indices = self.predict(data, coordinates)
        grid_coords = []
        
        for idx in bmu_indices:
            if idx >= 0:
                grid_coords.append(self._grid_positions[idx])
            else:
                grid_coords.append([-1, -1])
        
        return np.array(grid_coords)