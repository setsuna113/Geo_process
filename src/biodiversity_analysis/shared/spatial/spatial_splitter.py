"""
Spatial data splitting to prevent autocorrelation leakage.

Implements blockCV and other spatial validation strategies.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SpatialSplitStrategy(Enum):
    """Available spatial splitting strategies."""
    RANDOM_BLOCKS = "random_blocks"
    SYSTEMATIC_BLOCKS = "systematic_blocks"
    LATITUDINAL = "latitudinal"
    ENVIRONMENTAL_BLOCKS = "environmental_blocks"


@dataclass
class SpatialSplit:
    """Container for spatial train/validation/test split."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    split_info: Dict[str, any]
    
    @property
    def n_train(self) -> int:
        return len(self.train_idx)
    
    @property
    def n_val(self) -> int:
        return len(self.val_idx)
    
    @property
    def n_test(self) -> int:
        return len(self.test_idx)


class SpatialSplitter:
    """Split spatial data to avoid autocorrelation in validation."""
    
    def __init__(self, 
                 strategy: SpatialSplitStrategy = SpatialSplitStrategy.RANDOM_BLOCKS,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 buffer_distance: float = 0.0,
                 random_state: Optional[int] = None):
        """Initialize spatial splitter.
        
        Args:
            strategy: Splitting strategy to use
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            buffer_distance: Buffer between regions
            random_state: Random seed
        """
        self.strategy = strategy
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.buffer_distance = buffer_distance
        self.random_state = random_state
    
    def split(self, coordinates: np.ndarray, 
              features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform spatial split.
        
        Args:
            coordinates: Spatial coordinates
            features: Optional features for environmental blocking
            
        Returns:
            Tuple of (train_idx, val_idx, test_idx)
        """
        if self.strategy == SpatialSplitStrategy.RANDOM_BLOCKS:
            return self._random_block_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.LATITUDINAL:
            return self._latitudinal_split(coordinates)
        else:
            # Default to random blocks for now
            return self._random_block_split(coordinates)
    
    def _random_block_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random spatial blocks split."""
        n_samples = len(coordinates)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Simple implementation - divide into grid blocks
        # This is a placeholder - a real implementation would create actual spatial blocks
        indices = np.random.permutation(n_samples)
        
        train_size = int(n_samples * self.train_ratio)
        val_size = int(n_samples * self.val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        return train_idx, val_idx, test_idx
    
    def _latitudinal_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split by latitude bands."""
        n_samples = len(coordinates)
        
        # Sort by latitude
        lat_order = np.argsort(coordinates[:, 1])
        
        train_size = int(n_samples * self.train_ratio)
        val_size = int(n_samples * self.val_ratio)
        
        train_idx = lat_order[:train_size]
        val_idx = lat_order[train_size:train_size + val_size]
        test_idx = lat_order[train_size + val_size:]
        
        return train_idx, val_idx, test_idx
    
    @staticmethod
    def block_cv_split(
        coordinates: np.ndarray,
        n_folds: int = 5,
        buffer_size: Optional[float] = None,
        random_state: int = 42
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create spatial block cross-validation folds.
        
        Args:
            coordinates: Array of coordinates (n_samples, 2)
            n_folds: Number of CV folds
            buffer_size: Buffer between train/test blocks (in coordinate units)
            random_state: Random seed
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        np.random.seed(random_state)
        n_samples = len(coordinates)
        
        # Get spatial extent
        lon_min, lat_min = coordinates.min(axis=0)
        lon_max, lat_max = coordinates.max(axis=0)
        
        # Create spatial blocks
        n_blocks_per_dim = int(np.ceil(np.sqrt(n_folds)))
        lon_blocks = np.linspace(lon_min, lon_max, n_blocks_per_dim + 1)
        lat_blocks = np.linspace(lat_min, lat_max, n_blocks_per_dim + 1)
        
        # Assign samples to blocks
        block_assignments = np.zeros(n_samples, dtype=int)
        block_id = 0
        
        for i in range(n_blocks_per_dim):
            for j in range(n_blocks_per_dim):
                mask = (
                    (coordinates[:, 0] >= lon_blocks[i]) &
                    (coordinates[:, 0] < lon_blocks[i + 1]) &
                    (coordinates[:, 1] >= lat_blocks[j]) &
                    (coordinates[:, 1] < lat_blocks[j + 1])
                )
                block_assignments[mask] = block_id
                block_id += 1
        
        # Create folds by grouping blocks
        unique_blocks = np.unique(block_assignments)
        np.random.shuffle(unique_blocks)
        
        blocks_per_fold = len(unique_blocks) // n_folds
        folds = []
        
        for fold in range(n_folds):
            if fold == n_folds - 1:
                test_blocks = unique_blocks[fold * blocks_per_fold:]
            else:
                test_blocks = unique_blocks[fold * blocks_per_fold:(fold + 1) * blocks_per_fold]
            
            test_mask = np.isin(block_assignments, test_blocks)
            test_idx = np.where(test_mask)[0]
            
            if buffer_size:
                # Remove buffer zone from training data
                train_mask = ~test_mask
                for idx in test_idx:
                    distances = np.sqrt(np.sum((coordinates - coordinates[idx])**2, axis=1))
                    train_mask &= distances > buffer_size
                train_idx = np.where(train_mask)[0]
            else:
                train_idx = np.where(~test_mask)[0]
            
            folds.append((train_idx, test_idx))
            
        logger.info(f"Created {n_folds} spatial CV folds with "
                   f"{'buffer' if buffer_size else 'no buffer'}")
        
        return folds
    
    @staticmethod
    def train_val_test_split(
        coordinates: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        strategy: str = "random_blocks",
        block_size: Optional[float] = None,
        random_state: int = 42
    ) -> SpatialSplit:
        """
        Create spatial train/validation/test split.
        
        Args:
            coordinates: Array of coordinates (n_samples, 2)
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            strategy: 'random_blocks', 'systematic_blocks', or 'latitudinal'
            block_size: Size of spatial blocks (auto if None)
            random_state: Random seed
            
        Returns:
            SpatialSplit object
        """
        np.random.seed(random_state)
        n_samples = len(coordinates)
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        if strategy == "latitudinal":
            # Split by latitude bands
            lat_sorted_idx = np.argsort(coordinates[:, 1])
            
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            train_idx = lat_sorted_idx[:n_train]
            val_idx = lat_sorted_idx[n_train:n_train + n_val]
            test_idx = lat_sorted_idx[n_train + n_val:]
            
            split_info = {'strategy': 'latitudinal', 'direction': 'south_to_north'}
            
        elif strategy in ["random_blocks", "systematic_blocks"]:
            # Create spatial blocks
            if block_size is None:
                # Auto-determine block size
                extent = coordinates.max(axis=0) - coordinates.min(axis=0)
                n_blocks_target = 20  # Target number of blocks
                block_size = np.mean(extent) / np.sqrt(n_blocks_target)
            
            # Assign to blocks
            block_coords = ((coordinates - coordinates.min(axis=0)) / block_size).astype(int)
            block_ids = block_coords[:, 0] * 1000 + block_coords[:, 1]  # Unique block ID
            unique_blocks = np.unique(block_ids)
            
            if strategy == "random_blocks":
                np.random.shuffle(unique_blocks)
            else:  # systematic_blocks
                # Sort blocks spatially
                pass
            
            # Assign blocks to splits
            n_blocks = len(unique_blocks)
            n_train_blocks = int(n_blocks * train_ratio)
            n_val_blocks = int(n_blocks * val_ratio)
            
            train_blocks = unique_blocks[:n_train_blocks]
            val_blocks = unique_blocks[n_train_blocks:n_train_blocks + n_val_blocks]
            test_blocks = unique_blocks[n_train_blocks + n_val_blocks:]
            
            train_idx = np.where(np.isin(block_ids, train_blocks))[0]
            val_idx = np.where(np.isin(block_ids, val_blocks))[0]
            test_idx = np.where(np.isin(block_ids, test_blocks))[0]
            
            split_info = {
                'strategy': strategy,
                'block_size': block_size,
                'n_blocks': n_blocks
            }
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Spatial split - Train: {len(train_idx)}, "
                   f"Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return SpatialSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            split_info=split_info
        )
    
    @staticmethod
    def environmental_block_split(
        coordinates: np.ndarray,
        environmental_data: np.ndarray,
        n_blocks: int = 10,
        train_ratio: float = 0.7,
        random_state: int = 42
    ) -> SpatialSplit:
        """
        Split based on environmental blocks (for better generalization).
        
        Args:
            coordinates: Spatial coordinates
            environmental_data: Environmental variables at each location
            n_blocks: Number of environmental blocks
            train_ratio: Proportion for training
            random_state: Random seed
            
        Returns:
            SpatialSplit object
        """
        from sklearn.cluster import KMeans
        
        # Cluster based on environmental variables
        kmeans = KMeans(n_clusters=n_blocks, random_state=random_state)
        env_blocks = kmeans.fit_predict(environmental_data)
        
        # Randomly assign blocks to train/val/test
        unique_blocks = np.arange(n_blocks)
        np.random.seed(random_state)
        np.random.shuffle(unique_blocks)
        
        n_train_blocks = int(n_blocks * train_ratio)
        remaining = n_blocks - n_train_blocks
        n_val_blocks = remaining // 2
        
        train_blocks = unique_blocks[:n_train_blocks]
        val_blocks = unique_blocks[n_train_blocks:n_train_blocks + n_val_blocks]
        test_blocks = unique_blocks[n_train_blocks + n_val_blocks:]
        
        train_idx = np.where(np.isin(env_blocks, train_blocks))[0]
        val_idx = np.where(np.isin(env_blocks, val_blocks))[0]
        test_idx = np.where(np.isin(env_blocks, test_blocks))[0]
        
        split_info = {
            'strategy': 'environmental_blocks',
            'n_blocks': n_blocks,
            'cluster_centers': kmeans.cluster_centers_
        }
        
        logger.info(f"Environmental block split created with {n_blocks} blocks")
        
        return SpatialSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            split_info=split_info
        )