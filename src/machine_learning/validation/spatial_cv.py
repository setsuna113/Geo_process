"""Spatial cross-validation strategies for biodiversity data."""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, List, Union
from sklearn.model_selection import BaseCrossValidator
import logging

from ...core.registry import cv_strategy
from ...abstractions.types.ml_types import SpatialCVStrategy

logger = logging.getLogger(__name__)


@cv_strategy(
    spatial_aware=True,
    min_samples_per_fold=10,
    description="Spatial block cross-validation with configurable block size"
)
class SpatialBlockCV(BaseCrossValidator):
    """
    Spatial block cross-validation.
    
    Creates spatially contiguous blocks for train/test splits to
    reduce spatial autocorrelation between training and test sets.
    Uses a checkerboard pattern for fold assignment.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 block_size: float = 100.0,  # km
                 buffer_size: float = 0.0,    # km
                 random_state: Optional[int] = None,
                 shuffle: bool = True):
        """
        Initialize spatial block CV.
        
        Args:
            n_splits: Number of folds
            block_size: Size of spatial blocks in km
            buffer_size: Buffer zone around test blocks in km
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle block assignments
        """
        self.n_splits = n_splits
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.random_state = random_state
        self.shuffle = shuffle
        
        # Will be set during splitting
        self.block_assignments_ = None
        self.fold_indices_ = None
    
    def split(self, 
              X: Union[np.ndarray, pd.DataFrame],
              y: Optional[Union[np.ndarray, pd.Series]] = None,
              groups: Optional[np.ndarray] = None,
              lat_lon: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Features (must include latitude/longitude if lat_lon not provided)
            y: Target values (not used, for sklearn compatibility)
            groups: Group labels (not used)
            lat_lon: Latitude/longitude data if not in X
            
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        # Extract coordinates
        if lat_lon is not None:
            if isinstance(lat_lon, pd.DataFrame):
                coords = lat_lon[['latitude', 'longitude']].values
            else:
                coords = lat_lon
        elif isinstance(X, pd.DataFrame) and all(col in X.columns for col in ['latitude', 'longitude']):
            coords = X[['latitude', 'longitude']].values
        else:
            raise ValueError("Latitude and longitude must be provided either in X or as lat_lon parameter")
        
        n_samples = len(coords)
        
        # Convert block size from km to degrees (approximate)
        # 1 degree latitude ≈ 111 km
        block_size_deg = self.block_size / 111.0
        buffer_size_deg = self.buffer_size / 111.0
        
        # Find spatial extent
        lat_min, lon_min = coords.min(axis=0)
        lat_max, lon_max = coords.max(axis=0)
        
        # Create spatial blocks
        lat_blocks = np.arange(lat_min, lat_max + block_size_deg, block_size_deg)
        lon_blocks = np.arange(lon_min, lon_max + block_size_deg, block_size_deg)
        
        n_lat_blocks = len(lat_blocks) - 1
        n_lon_blocks = len(lon_blocks) - 1
        
        # Assign each point to a block
        lat_indices = np.digitize(coords[:, 0], lat_blocks) - 1
        lon_indices = np.digitize(coords[:, 1], lon_blocks) - 1
        
        # Clip to valid range
        lat_indices = np.clip(lat_indices, 0, n_lat_blocks - 1)
        lon_indices = np.clip(lon_indices, 0, n_lon_blocks - 1)
        
        # Create block IDs
        block_ids = lat_indices * n_lon_blocks + lon_indices
        
        # Assign blocks to folds using checkerboard pattern
        block_to_fold = {}
        fold_id = 0
        
        for i in range(n_lat_blocks):
            for j in range(n_lon_blocks):
                block_id = i * n_lon_blocks + j
                # Checkerboard pattern
                if self.n_splits == 2:
                    fold_id = (i + j) % 2
                else:
                    # For more folds, use modulo to distribute blocks
                    fold_id = (i * 2 + j) % self.n_splits
                block_to_fold[block_id] = fold_id
        
        # Shuffle block assignments if requested
        if self.shuffle and self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            fold_values = list(block_to_fold.values())
            rng.shuffle(fold_values)
            block_to_fold = dict(zip(block_to_fold.keys(), fold_values))
        
        # Assign samples to folds based on their blocks
        sample_folds = np.array([block_to_fold[bid] for bid in block_ids])
        
        # Store for later use
        self.block_assignments_ = block_ids
        self.fold_indices_ = sample_folds
        
        # Generate train/test splits
        for fold in range(self.n_splits):
            # Test indices are samples in this fold
            test_mask = sample_folds == fold
            test_indices = np.where(test_mask)[0]
            
            if self.buffer_size > 0:
                # Create buffer zone around test points
                train_mask = np.ones(n_samples, dtype=bool)
                train_mask[test_mask] = False
                
                # Remove points within buffer distance of test points
                test_coords = coords[test_mask]
                for i in np.where(train_mask)[0]:
                    # Check distance to all test points
                    distances = np.sqrt(
                        ((coords[i, 0] - test_coords[:, 0]) * 111)**2 +  # Convert to km
                        ((coords[i, 1] - test_coords[:, 1]) * 111 * np.cos(np.radians(coords[i, 0])))**2
                    )
                    if np.min(distances) < self.buffer_size:
                        train_mask[i] = False
                
                train_indices = np.where(train_mask)[0]
            else:
                # No buffer - all non-test samples are training
                train_indices = np.where(~test_mask)[0]
            
            # Log fold information
            logger.debug(f"Fold {fold}: {len(train_indices)} train, {len(test_indices)} test samples")
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits
    
    def get_fold_statistics(self) -> Optional[pd.DataFrame]:
        """
        Get statistics about the spatial distribution of folds.
        
        Returns:
            DataFrame with fold statistics if available
        """
        if self.fold_indices_ is None:
            return None
        
        stats = []
        for fold in range(self.n_splits):
            fold_mask = self.fold_indices_ == fold
            stats.append({
                'fold': fold,
                'n_samples': np.sum(fold_mask),
                'n_blocks': len(np.unique(self.block_assignments_[fold_mask])),
                'percentage': np.sum(fold_mask) / len(self.fold_indices_) * 100
            })
        
        return pd.DataFrame(stats)


@cv_strategy(
    spatial_aware=True,
    min_samples_per_fold=10,
    description="Spatial buffer cross-validation with distance-based buffers"
)
class SpatialBufferCV(BaseCrossValidator):
    """
    Spatial buffer cross-validation.
    
    Creates buffer zones around test points to ensure spatial independence
    between training and test sets.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 buffer_distance: float = 50.0,  # km
                 test_size: float = 0.2,
                 random_state: Optional[int] = None):
        """
        Initialize spatial buffer CV.
        
        Args:
            n_splits: Number of folds
            buffer_distance: Buffer distance around test points in km
            test_size: Proportion of samples in each test fold
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.buffer_distance = buffer_distance
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self,
              X: Union[np.ndarray, pd.DataFrame],
              y: Optional[Union[np.ndarray, pd.Series]] = None,
              groups: Optional[np.ndarray] = None,
              lat_lon: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data with spatial buffers.
        
        Args:
            X: Features
            y: Target values
            groups: Group labels (not used)
            lat_lon: Coordinates if not in X
            
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        # Extract coordinates
        if lat_lon is not None:
            if isinstance(lat_lon, pd.DataFrame):
                coords = lat_lon[['latitude', 'longitude']].values
            else:
                coords = lat_lon
        elif isinstance(X, pd.DataFrame) and all(col in X.columns for col in ['latitude', 'longitude']):
            coords = X[['latitude', 'longitude']].values
        else:
            raise ValueError("Coordinates required for spatial buffer CV")
        
        n_samples = len(coords)
        n_test = int(n_samples * self.test_size)
        
        # Random state for reproducibility
        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)
        
        for fold in range(self.n_splits):
            # Randomly select test points
            test_indices = rng.choice(indices, size=n_test, replace=False)
            test_coords = coords[test_indices]
            
            # Find training points outside buffer
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False
            
            # Remove points within buffer distance
            for i in np.where(train_mask)[0]:
                # Calculate distance to all test points
                distances = self._haversine_distance(
                    coords[i, 0], coords[i, 1],
                    test_coords[:, 0], test_coords[:, 1]
                )
                
                if np.min(distances) < self.buffer_distance:
                    train_mask[i] = False
            
            train_indices = np.where(train_mask)[0]
            
            logger.debug(f"Fold {fold}: {len(train_indices)} train, {len(test_indices)} test, "
                        f"{n_samples - len(train_indices) - len(test_indices)} in buffer")
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate haversine distance between points in km.
        
        Args:
            lat1, lon1: Coordinates of first point(s)
            lat2, lon2: Coordinates of second point(s)
            
        Returns:
            Distance(s) in km
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        r = 6371
        
        return c * r


@cv_strategy(
    spatial_aware=True,
    min_samples_per_fold=10,
    description="Environmental block CV stratified by latitude"
)
class EnvironmentalBlockCV(BaseCrossValidator):
    """
    Environmental block cross-validation.
    
    Creates folds based on environmental gradients (e.g., latitude bands)
    to ensure models are tested on different environmental conditions.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 stratify_by: str = 'latitude',
                 buffer_size: float = 0.0):
        """
        Initialize environmental block CV.
        
        Args:
            n_splits: Number of folds
            stratify_by: Environmental variable to stratify by
            buffer_size: Buffer between environmental blocks
        """
        self.n_splits = n_splits
        self.stratify_by = stratify_by
        self.buffer_size = buffer_size
    
    def split(self,
              X: Union[np.ndarray, pd.DataFrame],
              y: Optional[Union[np.ndarray, pd.Series]] = None,
              groups: Optional[np.ndarray] = None,
              env_data: Optional[pd.DataFrame] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for environmental block CV.
        
        Args:
            X: Features
            y: Target values
            groups: Group labels (not used)
            env_data: Environmental data if not in X
            
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        # Get stratification variable
        if self.stratify_by == 'latitude':
            if isinstance(X, pd.DataFrame) and 'latitude' in X.columns:
                strat_values = X['latitude'].values
            elif env_data is not None and 'latitude' in env_data.columns:
                strat_values = env_data['latitude'].values
            else:
                raise ValueError("Latitude not found for stratification")
        else:
            # Support for other environmental variables
            if isinstance(X, pd.DataFrame) and self.stratify_by in X.columns:
                strat_values = X[self.stratify_by].values
            elif env_data is not None and self.stratify_by in env_data.columns:
                strat_values = env_data[self.stratify_by].values
            else:
                raise ValueError(f"{self.stratify_by} not found for stratification")
        
        n_samples = len(strat_values)
        
        # Create environmental blocks
        percentiles = np.linspace(0, 100, self.n_splits + 1)
        thresholds = np.percentile(strat_values, percentiles)
        
        # Assign samples to blocks
        block_assignments = np.digitize(strat_values, thresholds[1:-1])
        
        # Generate splits
        for fold in range(self.n_splits):
            test_mask = block_assignments == fold
            test_indices = np.where(test_mask)[0]
            
            if self.buffer_size > 0:
                # Create buffer zone
                test_min = strat_values[test_mask].min() - self.buffer_size
                test_max = strat_values[test_mask].max() + self.buffer_size
                
                train_mask = ~test_mask & ((strat_values < test_min) | (strat_values > test_max))
                train_indices = np.where(train_mask)[0]
            else:
                train_indices = np.where(~test_mask)[0]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits