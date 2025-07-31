"""
Simple Spatial Validation for Biodiversity SOM

Handles spatial autocorrelation in biodiversity data through geographic splitting.
Based on blockCV and spatial cross-validation research for species distribution modeling.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SpatialSplitStrategy(Enum):
    """Spatial splitting strategies for biodiversity data."""
    LATITUDINAL = "latitudinal"      # North/South bands
    LONGITUDINAL = "longitudinal"    # East/West bands  
    CONTINENTAL = "continental"      # Continental blocks
    CHECKERBOARD = "checkerboard"    # Checkerboard pattern
    RANDOM_BLOCKS = "random_blocks"  # Random spatial blocks


@dataclass 
class SpatialDataSplit:
    """Container for spatially split biodiversity data."""
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


class BiodiversitySpatialSplitter:
    """
    Simple spatial data splitter for biodiversity datasets.
    
    Prevents data leakage from spatial autocorrelation by ensuring
    training, validation, and test sets are geographically separated.
    """
    
    def __init__(self,
                 strategy: SpatialSplitStrategy = SpatialSplitStrategy.LATITUDINAL,
                 train_ratio: float = 0.7,
                 validation_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        Initialize spatial splitter.
        
        Args:
            strategy: Geographic splitting strategy
            train_ratio: Fraction for training (0.7 = 70%)
            validation_ratio: Fraction for validation (0.15 = 15%)
            test_ratio: Fraction for testing (0.15 = 15%)
            random_seed: Random seed for reproducibility
        """
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + validation_ratio + test_ratio}")
        
        self.strategy = strategy
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        logger.info(f"Initialized spatial splitter: {strategy.value}")
        logger.info(f"Split ratios: {train_ratio:.1%}/{validation_ratio:.1%}/{test_ratio:.1%}")
    
    def split_data(self, 
                   data: np.ndarray,
                   coordinates: np.ndarray) -> SpatialDataSplit:
        """
        Split biodiversity data spatially to prevent autocorrelation leakage.
        
        Args:
            data: Feature data (n_samples, n_features)
            coordinates: Spatial coordinates (n_samples, 2) - [longitude, latitude]
            
        Returns:
            SpatialDataSplit with geographically separated train/val/test sets
        """
        n_samples = data.shape[0]
        
        if coordinates.shape[0] != n_samples:
            raise ValueError(f"Data and coordinates shape mismatch: {n_samples} vs {coordinates.shape[0]}")
        
        logger.info(f"Splitting {n_samples:,} samples using {self.strategy.value} strategy")
        
        # Get split indices based on strategy
        if self.strategy == SpatialSplitStrategy.LATITUDINAL:
            train_idx, val_idx, test_idx = self._latitudinal_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.LONGITUDINAL:
            train_idx, val_idx, test_idx = self._longitudinal_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.CONTINENTAL:
            train_idx, val_idx, test_idx = self._continental_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.CHECKERBOARD:
            train_idx, val_idx, test_idx = self._checkerboard_split(coordinates)
        elif self.strategy == SpatialSplitStrategy.RANDOM_BLOCKS:
            train_idx, val_idx, test_idx = self._random_blocks_split(coordinates)
        else:
            raise ValueError(f"Unknown split strategy: {self.strategy}")
        
        # Create spatial data split
        split = SpatialDataSplit(
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
        
        logger.info(f"Split completed: Train={len(train_idx):,}, Val={len(val_idx):,}, Test={len(test_idx):,}")
        
        return split
    
    def _latitudinal_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split by latitude bands (biogeographic regions)."""
        latitudes = coordinates[:, 1]
        sorted_indices = np.argsort(latitudes)
        
        n_samples = len(sorted_indices)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.validation_ratio))
        
        train_idx = sorted_indices[:train_end]
        val_idx = sorted_indices[train_end:val_end]  
        test_idx = sorted_indices[val_end:]
        
        return train_idx, val_idx, test_idx
    
    def _longitudinal_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split by longitude bands (continental regions)."""
        longitudes = coordinates[:, 0]
        sorted_indices = np.argsort(longitudes)
        
        n_samples = len(sorted_indices)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.validation_ratio))
        
        train_idx = sorted_indices[:train_end]
        val_idx = sorted_indices[train_end:val_end]
        test_idx = sorted_indices[val_end:]
        
        return train_idx, val_idx, test_idx
    
    def _continental_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split by continental blocks."""
        longitudes = coordinates[:, 0]
        
        # Simple continental boundaries
        americas_mask = longitudes < -30
        africa_europe_mask = (longitudes >= -30) & (longitudes < 60)
        asia_oceania_mask = longitudes >= 60
        
        americas_idx = np.where(americas_mask)[0]
        africa_europe_idx = np.where(africa_europe_mask)[0]
        asia_oceania_idx = np.where(asia_oceania_mask)[0]
        
        # Check minimum sample sizes
        min_samples = 50
        if any(len(idx) < min_samples for idx in [americas_idx, africa_europe_idx, asia_oceania_idx]):
            logger.warning("Continental split has small regions, falling back to latitudinal")
            return self._latitudinal_split(coordinates)
        
        # Assign continents to splits
        train_idx = americas_idx
        val_idx = africa_europe_idx
        test_idx = asia_oceania_idx
        
        return train_idx, val_idx, test_idx
    
    def _checkerboard_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Checkerboard pattern for maximum spatial separation."""
        longitudes = coordinates[:, 0]
        latitudes = coordinates[:, 1]
        
        # Create 5-degree grid cells
        lon_cells = ((longitudes + 180) // 5).astype(int)
        lat_cells = ((latitudes + 90) // 5).astype(int)
        
        # Checkerboard assignment
        cell_sum = (lon_cells + lat_cells) % 3
        
        train_idx = np.where(cell_sum == 0)[0]
        val_idx = np.where(cell_sum == 1)[0]
        test_idx = np.where(cell_sum == 2)[0]
        
        return train_idx, val_idx, test_idx
    
    def _random_blocks_split(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random spatial blocks while maintaining spatial coherence."""
        longitudes = coordinates[:, 0]
        latitudes = coordinates[:, 1]
        
        # Create 10-degree spatial grid
        lon_bins = np.arange(-180, 181, 10)
        lat_bins = np.arange(-90, 91, 10)
        
        lon_indices = np.digitize(longitudes, lon_bins) - 1
        lat_indices = np.digitize(latitudes, lat_bins) - 1
        
        # Create unique grid cell IDs
        grid_cells = lon_indices * len(lat_bins) + lat_indices
        unique_cells = np.unique(grid_cells)
        
        # Randomly assign grid cells to splits
        np.random.shuffle(unique_cells)
        n_cells = len(unique_cells)
        
        train_cells_end = int(n_cells * self.train_ratio)
        val_cells_end = int(n_cells * (self.train_ratio + self.validation_ratio))
        
        train_cells = set(unique_cells[:train_cells_end])
        val_cells = set(unique_cells[train_cells_end:val_cells_end])
        test_cells = set(unique_cells[val_cells_end:])
        
        # Assign samples based on grid cell membership
        train_idx = np.where([cell in train_cells for cell in grid_cells])[0]
        val_idx = np.where([cell in val_cells for cell in grid_cells])[0]
        test_idx = np.where([cell in test_cells for cell in grid_cells])[0]
        
        return train_idx, val_idx, test_idx
    
    def _get_split_metadata(self, coordinates: np.ndarray,
                          train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
        """Generate metadata about the spatial split."""
        def get_coord_stats(idx):
            if len(idx) == 0:
                return {"count": 0}
            coords = coordinates[idx]
            return {
                "count": len(idx),
                "lon_range": [float(coords[:, 0].min()), float(coords[:, 0].max())],
                "lat_range": [float(coords[:, 1].min()), float(coords[:, 1].max())],
                "lon_center": float(coords[:, 0].mean()),
                "lat_center": float(coords[:, 1].mean())
            }
        
        return {
            "strategy": self.strategy.value,
            "train_stats": get_coord_stats(train_idx),
            "validation_stats": get_coord_stats(val_idx),
            "test_stats": get_coord_stats(test_idx),
            "total_samples": len(coordinates),
            "actual_ratios": {
                "train": len(train_idx) / len(coordinates),
                "validation": len(val_idx) / len(coordinates),
                "test": len(test_idx) / len(coordinates)  
            }
        }


def create_spatial_splitter(strategy: str = "latitudinal", **kwargs) -> BiodiversitySpatialSplitter:
    """Factory function for creating spatial splitter."""
    strategy_enum = SpatialSplitStrategy(strategy)
    return BiodiversitySpatialSplitter(strategy=strategy_enum, **kwargs)