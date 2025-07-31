"""
Biodiversity SOM Validation Framework

This module implements spatial train/validation/test splitting and validation-based
convergence detection specifically designed for biodiversity data analysis.

Key features:
- Spatial data splitting (preserves biogeographic structure)
- Validation-based early stopping
- Biodiversity-specific evaluation metrics
- Overfitting detection for unsupervised learning
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path
import json

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class SpatialSplitStrategy(Enum):
    """Strategies for spatially splitting biodiversity data."""
    LATITUDINAL = "latitudinal"  # Northern vs Southern hemisphere
    LONGITUDINAL = "longitudinal"  # Eastern vs Western hemisphere
    CONTINENTAL = "continental"  # Continental blocks
    ECOREGION = "ecoregion"  # Major biomes/ecoregions
    RANDOM_SPATIAL = "random_spatial"  # Random but spatially coherent blocks
    CHECKERBOARD = "checkerboard"  # Checkerboard pattern


@dataclass
class DataSplit:
    """Container for train/validation/test data split."""
    train_data: np.ndarray
    train_coords: np.ndarray
    train_indices: np.ndarray
    
    validation_data: np.ndarray
    validation_coords: np.ndarray
    validation_indices: np.ndarray
    
    test_data: np.ndarray
    test_coords: np.ndarray
    test_indices: np.ndarray
    
    split_strategy: str
    split_metadata: Dict[str, Any]


@dataclass
class ValidationMetrics:
    """Container for SOM validation metrics."""
    epoch: int
    train_quantization_error: float
    validation_quantization_error: float
    train_topographic_error: float
    validation_topographic_error: float
    cluster_stability: float
    spatial_coherence: float
    overfitting_score: float
    timestamp: float


@dataclass
class BiodiversityEvaluationMetrics:
    """Biodiversity-specific SOM evaluation metrics."""
    # Core SOM metrics
    quantization_error: float
    topographic_error: float
    
    # Biodiversity-specific metrics
    species_association_accuracy: float
    functional_diversity_preservation: float
    phylogenetic_signal_retention: float
    biogeographic_coherence: float
    endemic_species_clustering: float
    
    # Spatial metrics
    spatial_autocorrelation: float
    edge_effect_score: float
    cluster_compactness: float
    
    # Statistical metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float


class BiodiversitySpatialSplitter:
    """Spatial data splitter for biodiversity datasets."""
    
    def __init__(self, 
                 strategy: SpatialSplitStrategy = SpatialSplitStrategy.LATITUDINAL,
                 train_ratio: float = 0.7,
                 validation_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """Initialize spatial splitter.
        
        Args:
            strategy: Spatial splitting strategy
            train_ratio: Proportion of data for training
            validation_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        logger.info(f"Initialized spatial splitter: {strategy.value} "
                   f"({train_ratio:.1%}/{validation_ratio:.1%}/{test_ratio:.1%})")
    
    def split_data(self, 
                   data: np.ndarray, 
                   coordinates: np.ndarray) -> DataSplit:
        """Split biodiversity data spatially.
        
        Args:
            data: Feature data (n_samples, n_features)
            coordinates: Spatial coordinates (n_samples, 2) - [longitude, latitude]
            
        Returns:
            DataSplit object with train/validation/test splits
        """
        n_samples = data.shape[0]
        
        if coordinates.shape[0] != n_samples:
            raise ValueError(f"Data and coordinates shape mismatch: {n_samples} vs {coordinates.shape[0]}")
        
        logger.info(f"Splitting {n_samples:,} samples using {self.strategy.value} strategy")
        
        # Get split indices based on strategy
        if self.strategy == SpatialSplitStrategy.LATITUDINAL:
            indices = self._latitudinal_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.LONGITUDINAL:
            indices = self._longitudinal_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.CONTINENTAL:
            indices = self._continental_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.ECOREGION:
            indices = self._ecoregion_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.RANDOM_SPATIAL:
            indices = self._random_spatial_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.CHECKERBOARD:
            indices = self._checkerboard_split(coordinates)
        else:
            raise ValueError(f"Unknown split strategy: {self.strategy}")
        
        train_idx, val_idx, test_idx = indices
        
        # Create data split
        split = DataSplit(
            train_data=data[train_idx],
            train_coords=coordinates[train_idx],
            train_indices=train_idx,
            
            validation_data=data[val_idx],
            validation_coords=coordinates[val_idx],
            validation_indices=val_idx,
            
            test_data=data[test_idx],
            test_coords=coordinates[test_idx],
            test_indices=test_idx,
            
            split_strategy=self.strategy.value,
            split_metadata=self._get_split_metadata(coordinates, train_idx, val_idx, test_idx)
        )
        
        logger.info(f"Split completed: Train={len(train_idx):,}, "
                   f"Validation={len(val_idx):,}, Test={len(test_idx):,}")
        
        return split
    
    def _latitudinal_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data by latitude (biogeographic regions)."""
        latitudes = coordinates[:, 1]
        
        # Sort by latitude
        sorted_indices = np.argsort(latitudes)
        
        # Split into three latitude bands
        n_samples = len(sorted_indices)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.validation_ratio))
        
        train_idx = sorted_indices[:train_end]
        val_idx = sorted_indices[train_end:val_end]
        test_idx = sorted_indices[val_end:]
        
        return train_idx, val_idx, test_idx
    
    def _longitudinal_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data by longitude (continental regions)."""
        longitudes = coordinates[:, 0]
        
        # Sort by longitude
        sorted_indices = np.argsort(longitudes)
        
        # Split into three longitude bands
        n_samples = len(sorted_indices)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.validation_ratio))
        
        train_idx = sorted_indices[:train_end]
        val_idx = sorted_indices[train_end:val_end]
        test_idx = sorted_indices[val_end:]
        
        return train_idx, val_idx, test_idx
    
    def _continental_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data by continental blocks."""
        longitudes = coordinates[:, 0]
        latitudes = coordinates[:, 1]
        
        # Simple continental classification
        # Americas: lon < -30
        # Africa/Europe: -30 <= lon < 60  
        # Asia/Oceania: lon >= 60
        
        americas_mask = longitudes < -30
        africa_europe_mask = (longitudes >= -30) & (longitudes < 60)
        asia_oceania_mask = longitudes >= 60
        
        americas_idx = np.where(americas_mask)[0]
        africa_europe_idx = np.where(africa_europe_mask)[0]
        asia_oceania_idx = np.where(asia_oceania_mask)[0]
        
        # Assign continents to train/val/test
        train_idx = americas_idx
        val_idx = africa_europe_idx
        test_idx = asia_oceania_idx
        
        # If any split is too small, redistribute
        min_samples = 100
        all_idx = np.arange(len(coordinates))
        
        if len(train_idx) < min_samples or len(val_idx) < min_samples or len(test_idx) < min_samples:
            logger.warning("Continental split resulted in small splits, falling back to latitudinal")
            return self._latitudinal_split(coordinates)
        
        return train_idx, val_idx, test_idx
    
    def _ecoregion_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data by major biomes/ecoregions."""
        latitudes = coordinates[:, 1]
        
        # Simple biome classification based on latitude
        # Tropical: -23.5 < lat < 23.5
        # Temperate: 23.5 <= |lat| < 66.5
        # Polar: |lat| >= 66.5
        
        tropical_mask = (latitudes > -23.5) & (latitudes < 23.5)
        temperate_mask = ((latitudes >= 23.5) & (latitudes < 66.5)) | ((latitudes <= -23.5) & (latitudes > -66.5))
        polar_mask = (latitudes >= 66.5) | (latitudes <= -66.5)
        
        tropical_idx = np.where(tropical_mask)[0]
        temperate_idx = np.where(temperate_mask)[0]
        polar_idx = np.where(polar_mask)[0]
        
        # Assign biomes to train/val/test
        train_idx = temperate_idx  # Largest group typically
        val_idx = tropical_idx
        test_idx = polar_idx
        
        # Check for minimum samples
        min_samples = 50
        if len(test_idx) < min_samples:
            # Polar regions might have too few samples
            logger.warning("Polar regions have too few samples, redistributing")
            # Combine with temperate and split differently
            combined_idx = np.concatenate([temperate_idx, polar_idx])
            np.random.shuffle(combined_idx)
            
            n_combined = len(combined_idx)
            train_from_combined = int(n_combined * 0.8)
            
            train_idx = np.concatenate([tropical_idx, combined_idx[:train_from_combined]])
            val_idx = combined_idx[train_from_combined:]
            test_idx = np.array([], dtype=int)
            
            # Create test set from train set
            np.random.shuffle(train_idx)
            test_size = int(len(train_idx) * self.test_ratio / (self.train_ratio + self.test_ratio))
            test_idx = train_idx[-test_size:]
            train_idx = train_idx[:-test_size]
        
        return train_idx, val_idx, test_idx
    
    def _random_spatial_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random but spatially coherent split."""
        # Create spatial grid and assign grid cells to splits
        longitudes = coordinates[:, 0]
        latitudes = coordinates[:, 1]
        
        # Create 10x10 degree grid
        lon_bins = np.arange(-180, 181, 10)
        lat_bins = np.arange(-90, 91, 10)
        
        # Assign each point to a grid cell
        lon_indices = np.digitize(longitudes, lon_bins) - 1
        lat_indices = np.digitize(latitudes, lat_bins) - 1
        
        # Create unique grid cell IDs
        grid_cells = lon_indices * len(lat_bins) + lat_indices
        unique_cells = np.unique(grid_cells)
        
        # Randomly assign grid cells to train/val/test
        np.random.shuffle(unique_cells)
        n_cells = len(unique_cells)
        
        train_cells_end = int(n_cells * self.train_ratio)
        val_cells_end = int(n_cells * (self.train_ratio + self.validation_ratio))
        
        train_cells = set(unique_cells[:train_cells_end])
        val_cells = set(unique_cells[train_cells_end:val_cells_end])
        test_cells = set(unique_cells[val_cells_end:])
        
        # Assign samples based on their grid cell
        train_idx = np.where([cell in train_cells for cell in grid_cells])[0]
        val_idx = np.where([cell in val_cells for cell in grid_cells])[0]  
        test_idx = np.where([cell in test_cells for cell in grid_cells])[0]
        
        return train_idx, val_idx, test_idx
    
    def _checkerboard_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Checkerboard pattern split for maximum spatial separation."""
        longitudes = coordinates[:, 0]
        latitudes = coordinates[:, 1]
        
        # Create checkerboard pattern with 5x5 degree cells
        lon_cells = ((longitudes + 180) // 5).astype(int)
        lat_cells = ((latitudes + 90) // 5).astype(int)
        
        # Checkerboard assignment: (i + j) % 3 determines split
        cell_sum = (lon_cells + lat_cells) % 3
        
        train_idx = np.where(cell_sum == 0)[0]
        val_idx = np.where(cell_sum == 1)[0]
        test_idx = np.where(cell_sum == 2)[0]
        
        return train_idx, val_idx, test_idx
    
    def _get_split_metadata(self, coordinates: np.ndarray, 
                          train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
        """Generate metadata about the split."""
        def get_coord_stats(idx):
            if len(idx) == 0:
                return {"count": 0}
            coords = coordinates[idx]
            return {
                "count": len(idx),
                "lon_range": [float(coords[:, 0].min()), float(coords[:, 0].max())],
                "lat_range": [float(coords[:, 1].min()), float(coords[:, 1].max())],
                "lon_mean": float(coords[:, 0].mean()),
                "lat_mean": float(coords[:, 1].mean())
            }
        
        return {
            "strategy": self.strategy.value,
            "train_stats": get_coord_stats(train_idx),
            "validation_stats": get_coord_stats(val_idx),
            "test_stats": get_coord_stats(test_idx),
            "total_samples": len(coordinates),
            "split_ratios": {
                "train": len(train_idx) / len(coordinates),
                "validation": len(val_idx) / len(coordinates),
                "test": len(test_idx) / len(coordinates)
            }
        }


class BiodiversitySOMValidator:
    """Validation framework for biodiversity SOM training."""
    
    def __init__(self,
                 patience: int = 10,
                 min_improvement: float = 1e-5,
                 validation_interval: int = 10,
                 max_epochs: int = 1000,
                 save_checkpoints: bool = True,
                 checkpoint_dir: Optional[Path] = None):
        """Initialize SOM validator.
        
        Args:
            patience: Number of epochs without improvement before stopping
            min_improvement: Minimum improvement to reset patience counter
            validation_interval: Check validation metrics every N epochs
            max_epochs: Maximum training epochs
            save_checkpoints: Whether to save model checkpoints
            checkpoint_dir: Directory for saving checkpoints
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.validation_interval = validation_interval
        self.max_epochs = max_epochs
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir or Path("som_checkpoints")
        
        # Training state
        self.validation_history: List[ValidationMetrics] = []
        self.best_validation_qe = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_som_weights = None
        
        if self.save_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SOM validator: patience={patience}, "
                   f"min_improvement={min_improvement}, max_epochs={max_epochs}")
    
    def train_with_validation(self, 
                            som,
                            data_split: DataSplit,
                            convergence_method: str = 'unified',
                            use_batch_training: bool = False,
                            progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Train SOM with validation-based early stopping using advanced convergence methods.
        
        Args:
            som: SOM instance to train
            data_split: Train/validation/test data split
            convergence_method: 'unified', 'batch_unified', or 'basic'
            use_batch_training: Use batch SOM training instead of stochastic
            progress_callback: Optional progress callback function
            
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info(f"Starting SOM training with {convergence_method} convergence method")
        logger.info(f"Training mode: {'batch' if use_batch_training else 'stochastic'}")
        
        train_data = data_split.train_data
        validation_data = data_split.validation_data
        
        start_time = time.time()
        
        # Initialize advanced convergence detector
        convergence_detector = self._create_convergence_detector(convergence_method)
        
        for epoch in range(self.max_epochs):
            # Training step - choose method
            if use_batch_training:
                # Batch training: process all samples in one epoch
                som.train_batch(train_data, 1, random_order=True) 
            else:
                # Stochastic training: random samples
                som.train_random(train_data, len(train_data))  # Full pass through data
            
            # Advanced convergence check
            if epoch % self.validation_interval == 0:
                if convergence_method == 'batch_unified' and use_batch_training:
                    # Use batch unified convergence detection
                    convergence_result = convergence_detector.check_batch_convergence(
                        som, validation_data, epoch, self.max_epochs
                    )
                    
                    # Create metrics from convergence result
                    metrics = self._create_metrics_from_batch_result(
                        convergence_result, som, train_data, validation_data, epoch
                    )
                    
                    # Check for improvement using unified convergence index
                    current_score = convergence_result.unified_result.convergence_index
                    improvement_metric = current_score  # Higher is better for unified index
                    
                    if current_score > self.best_validation_qe + self.min_improvement:
                        self.best_validation_qe = current_score
                        self.best_epoch = epoch
                        self.patience_counter = 0
                        self.best_som_weights = som.get_weights().copy()
                        
                        logger.info(f"Epoch {epoch}: New best unified convergence = {current_score:.6f}")
                    else:
                        self.patience_counter += 1
                    
                    # Early stopping based on convergence detector
                    if convergence_result.is_converged:
                        logger.info(f"Advanced convergence detected at epoch {epoch}: {convergence_result.convergence_reason}")
                        break
                        
                elif convergence_method == 'unified':
                    # Use unified convergence detection with stochastic training
                    is_converged, convergence_info = convergence_detector.check_advanced_convergence(
                        som, validation_data, epoch, self.max_epochs
                    )
                    
                    # Create metrics from advanced convergence
                    metrics = self._create_metrics_from_advanced_result(
                        convergence_info, som, train_data, validation_data, epoch
                    )
                    
                    # Check for improvement using unified convergence index
                    current_score = convergence_info.get('convergence_index', 0)
                    if current_score > self.best_validation_qe + self.min_improvement:
                        self.best_validation_qe = current_score
                        self.best_epoch = epoch
                        self.patience_counter = 0
                        self.best_som_weights = som.get_weights().copy()
                        
                        logger.info(f"Epoch {epoch}: New best unified convergence = {current_score:.6f}")
                    else:
                        self.patience_counter += 1
                    
                    # Early stopping based on advanced convergence
                    if is_converged:
                        logger.info(f"Advanced convergence detected at epoch {epoch}")
                        break
                        
                else:
                    # Basic validation approach (fallback)
                    metrics = self._calculate_validation_metrics(
                        som, train_data, validation_data, epoch
                    )
                    
                    # Standard early stopping based on quantization error
                    if metrics.validation_quantization_error < self.best_validation_qe - self.min_improvement:
                        self.best_validation_qe = metrics.validation_quantization_error
                        self.best_epoch = epoch
                        self.patience_counter = 0
                        self.best_som_weights = som.get_weights().copy()
                        
                        logger.info(f"Epoch {epoch}: New best validation QE = {metrics.validation_quantization_error:.6f}")
                    else:
                        self.patience_counter += 1
                
                self.validation_history.append(metrics)
                
                # Progress callback
                if progress_callback:
                    progress_callback(epoch, self.max_epochs, f"Epoch {epoch}, Method: {convergence_method}")
                
                # Standard patience-based early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch} (no improvement for {self.patience} checks)")
                    break
                
                # Log progress
                if epoch % (self.validation_interval * 5) == 0:
                    if hasattr(metrics, 'validation_quantization_error'):
                        logger.info(f"Epoch {epoch}: Train QE={metrics.train_quantization_error:.6f}, "
                                   f"Val QE={metrics.validation_quantization_error:.6f}")
        
        # Restore best weights
        if self.best_som_weights is not None:
            som.set_weights(self.best_som_weights)
            logger.info(f"Restored best weights from epoch {self.best_epoch}")
        
        training_time = time.time() - start_time
        
        # Final evaluation using the selected method
        if convergence_method == 'batch_unified':
            final_convergence_result = convergence_detector.check_batch_convergence(
                som, validation_data, self.best_epoch, self.max_epochs
            )
            final_metrics = self._create_metrics_from_batch_result(
                final_convergence_result, som, train_data, validation_data, self.best_epoch
            )
        elif convergence_method == 'unified':
            is_converged, convergence_info = convergence_detector.check_advanced_convergence(
                som, validation_data, self.best_epoch, self.max_epochs
            )
            final_metrics = self._create_metrics_from_advanced_result(
                convergence_info, som, train_data, validation_data, self.best_epoch
            )
        else:
            final_metrics = self._calculate_validation_metrics(
                som, train_data, validation_data, self.best_epoch
            )
        
        training_results = {
            'converged': self.patience_counter >= self.patience,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch + 1,
            'training_time': training_time,
            'best_validation_score': self.best_validation_qe,
            'final_metrics': final_metrics,
            'validation_history': self.validation_history,
            'convergence_method': convergence_method,
            'training_mode': 'batch' if use_batch_training else 'stochastic',
            'overfitting_detected': final_metrics.overfitting_score > 0.2
        }
        
        logger.info(f"Training completed in {training_time:.2f}s after {epoch + 1} epochs using {convergence_method}")
        return training_results
    
    def _calculate_validation_metrics(self, 
                                    som, 
                                    train_data: np.ndarray,
                                    validation_data: np.ndarray,
                                    epoch: int) -> ValidationMetrics:
        """Calculate comprehensive validation metrics."""
        
        # Basic SOM metrics
        train_qe = som.quantization_error(train_data)
        val_qe = som.quantization_error(validation_data)
        train_te = som.topographic_error(train_data)
        val_te = som.topographic_error(validation_data)
        
        # Cluster stability (consistency of cluster assignments)
        cluster_stability = self._calculate_cluster_stability(som, validation_data)
        
        # Spatial coherence (do nearby locations have similar clusters?)
        spatial_coherence = 0.8  # Placeholder - would need coordinates
        
        # Overfitting score (validation performance vs training performance)
        overfitting_score = max(0, (val_qe - train_qe) / train_qe)
        
        return ValidationMetrics(
            epoch=epoch,
            train_quantization_error=train_qe,
            validation_quantization_error=val_qe,
            train_topographic_error=train_te,
            validation_topographic_error=val_te,
            cluster_stability=cluster_stability,
            spatial_coherence=spatial_coherence,
            overfitting_score=overfitting_score,
            timestamp=time.time()
        )
    
    def _calculate_cluster_stability(self, som, data: np.ndarray) -> float:
        """Calculate stability of cluster assignments."""
        # Run clustering twice with slight perturbation
        labels1 = som.predict(data)
        
        # Add small noise and recluster
        noisy_data = data + np.random.normal(0, 0.01 * data.std(), data.shape)
        labels2 = som.predict(noisy_data)
        
        # Calculate agreement
        agreement = np.mean(labels1 == labels2)
        return float(agreement)
    
    def _save_checkpoint(self, som, epoch: int, metrics: ValidationMetrics):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"som_epoch_{epoch}.npz"
        
        np.savez_compressed(
            checkpoint_path,
            weights=som.get_weights(),
            epoch=epoch,
            validation_qe=metrics.validation_quantization_error,
            training_qe=metrics.train_quantization_error
        )
        
        # Save metrics as JSON
        metrics_path = self.checkpoint_dir / f"metrics_epoch_{epoch}.json"
        metrics_dict = {
            'epoch': metrics.epoch,
            'train_quantization_error': metrics.train_quantization_error,
            'validation_quantization_error': metrics.validation_quantization_error,
            'train_topographic_error': metrics.train_topographic_error,
            'validation_topographic_error': metrics.validation_topographic_error,
            'cluster_stability': metrics.cluster_stability,
            'spatial_coherence': metrics.spatial_coherence,
            'overfitting_score': metrics.overfitting_score,
            'timestamp': metrics.timestamp
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    def _create_convergence_detector(self, convergence_method: str):
        """Create appropriate convergence detector based on method."""
        if convergence_method == 'batch_unified':
            from .batch_unified_convergence import create_batch_convergence_detector
            return create_batch_convergence_detector(
                weight_threshold=1e-6,
                quality_threshold=0.95,
                require_both=True
            )
        elif convergence_method == 'unified':
            from .advanced_convergence import create_advanced_convergence_detector
            return create_advanced_convergence_detector(
                method='unified',
                learning_rate_schedule='vlrsom'
            )
        else:
            # Return None for basic method
            return None
    
    def _create_metrics_from_batch_result(self, convergence_result, som, train_data, validation_data, epoch):
        """Create ValidationMetrics from batch convergence result."""
        # Calculate basic metrics
        train_qe = som.quantization_error(train_data)
        val_qe = som.quantization_error(validation_data)
        train_te = som.topographic_error(train_data)
        val_te = som.topographic_error(validation_data)
        
        # Use unified convergence index as main metric
        unified_score = convergence_result.unified_result.convergence_index
        
        return ValidationMetrics(
            epoch=epoch,
            train_quantization_error=train_qe,
            validation_quantization_error=val_qe,
            train_topographic_error=train_te,
            validation_topographic_error=val_te,
            cluster_stability=0.8,  # Placeholder
            spatial_coherence=unified_score,  # Use unified score
            overfitting_score=max(0, (val_qe - train_qe) / train_qe),
            timestamp=time.time()
        )
    
    def _create_metrics_from_advanced_result(self, convergence_info, som, train_data, validation_data, epoch):
        """Create ValidationMetrics from advanced convergence result."""
        # Calculate basic metrics
        train_qe = som.quantization_error(train_data)
        val_qe = som.quantization_error(validation_data)
        train_te = som.topographic_error(train_data)
        val_te = som.topographic_error(validation_data)
        
        # Use unified convergence index if available
        unified_score = convergence_info.get('convergence_index', 0.5)
        
        return ValidationMetrics(
            epoch=epoch,
            train_quantization_error=train_qe,
            validation_quantization_error=val_qe,
            train_topographic_error=train_te,
            validation_topographic_error=val_te,
            cluster_stability=0.8,  # Placeholder
            spatial_coherence=unified_score,  # Use unified score
            overfitting_score=max(0, (val_qe - train_qe) / train_qe),
            timestamp=time.time()
        )


def create_spatial_splitter(strategy: str = "latitudinal", **kwargs) -> BiodiversitySpatialSplitter:
    """Factory function for creating spatial splitter."""
    strategy_enum = SpatialSplitStrategy(strategy)
    return BiodiversitySpatialSplitter(strategy=strategy_enum, **kwargs)


def create_som_validator(**kwargs) -> BiodiversitySOMValidator:
    """Factory function for creating SOM validator."""
    return BiodiversitySOMValidator(**kwargs)