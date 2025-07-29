"""Data iterators for analysis-specific access patterns."""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional, List
import numpy as np
import logging
from pathlib import Path

from src.base.dataset import BaseDataset

logger = logging.getLogger(__name__)


class AnalysisDataIterator(ABC):
    """Base class for analysis-specific data iteration."""
    
    @abstractmethod
    def iterate(self, dataset: BaseDataset, **kwargs) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over data in analysis-appropriate way.
        
        Args:
            dataset: Dataset to iterate over
            **kwargs: Method-specific parameters
            
        Yields:
            Tuple of (features, coordinates) as numpy arrays
        """
        pass


class SOMDataIterator(AnalysisDataIterator):
    """Random sampling iterator for SOM analysis."""
    
    def iterate(self, dataset: BaseDataset, max_samples: Optional[int] = None, 
                random_seed: int = 42) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate with optional random sampling for SOM.
        
        Args:
            dataset: Dataset to iterate over
            max_samples: Maximum samples to use (None for all)
            random_seed: Random seed for reproducible sampling
            
        Yields:
            Tuple of (features, coordinates)
        """
        # If we need all data or more than available, just iterate normally
        if max_samples is None:
            logger.info("SOM iterator: Using all available data")
            if hasattr(dataset, 'get_features_and_coords'):
                yield from dataset.get_features_and_coords()
            else:
                # Fallback for datasets without this method
                for chunk in dataset.read_chunks():
                    coords = chunk[['x', 'y']].values
                    feature_cols = [c for c in chunk.columns if c not in ['x', 'y', 'cell_id']]
                    features = chunk[feature_cols].values
                    yield features, coords
        else:
            # Implement reservoir sampling for random selection
            logger.info(f"SOM iterator: Reservoir sampling {max_samples} samples")
            
            np.random.seed(random_seed)
            reservoir_features = []
            reservoir_coords = []
            samples_seen = 0
            
            for features, coords in dataset.get_features_and_coords():
                n_chunk = len(features)
                
                for i in range(n_chunk):
                    samples_seen += 1
                    
                    # First max_samples go directly into reservoir
                    if len(reservoir_features) < max_samples:
                        reservoir_features.append(features[i])
                        reservoir_coords.append(coords[i])
                    else:
                        # Random replacement with decreasing probability
                        j = np.random.randint(0, samples_seen)
                        if j < max_samples:
                            reservoir_features[j] = features[i]
                            reservoir_coords[j] = coords[i]
            
            # Yield the sampled data
            if reservoir_features:
                features_array = np.array(reservoir_features)
                coords_array = np.array(reservoir_coords)
                logger.info(f"Sampled {len(features_array)} from {samples_seen} total samples")
                yield features_array, coords_array


class GWPCADataIterator(AnalysisDataIterator):
    """Spatial tile iterator for GWPCA analysis."""
    
    def iterate(self, dataset: BaseDataset, tile_size: int = 1000, 
                overlap: float = 0.1) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over spatial tiles for GWPCA.
        
        Args:
            dataset: Dataset to iterate over
            tile_size: Number of points per tile
            overlap: Fraction of overlap between tiles (0-1)
            
        Yields:
            Tuple of (features, coordinates) for each tile
        """
        logger.info(f"GWPCA iterator: Creating spatial tiles of size {tile_size}")
        
        # First pass: collect all data to determine spatial extent
        # In production, this could be optimized with spatial indexing
        all_features = []
        all_coords = []
        
        for features, coords in dataset.get_features_and_coords():
            all_features.append(features)
            all_coords.append(coords)
        
        if not all_features:
            logger.warning("No data found for GWPCA")
            return
        
        # Concatenate all data
        all_features = np.vstack(all_features)
        all_coords = np.vstack(all_coords)
        
        # Determine spatial grid
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        
        # Calculate tile dimensions
        n_points = len(all_coords)
        n_tiles = max(1, n_points // tile_size)
        tiles_per_dim = int(np.sqrt(n_tiles))
        
        x_step = (x_max - x_min) / tiles_per_dim
        y_step = (y_max - y_min) / tiles_per_dim
        
        # Add overlap
        x_overlap = x_step * overlap
        y_overlap = y_step * overlap
        
        # Generate tiles
        tiles_generated = 0
        for i in range(tiles_per_dim):
            for j in range(tiles_per_dim):
                # Define tile bounds with overlap
                tile_x_min = x_min + i * x_step - x_overlap
                tile_x_max = x_min + (i + 1) * x_step + x_overlap
                tile_y_min = y_min + j * y_step - y_overlap
                tile_y_max = y_min + (j + 1) * y_step + y_overlap
                
                # Find points in this tile
                mask = (
                    (all_coords[:, 0] >= tile_x_min) & 
                    (all_coords[:, 0] <= tile_x_max) &
                    (all_coords[:, 1] >= tile_y_min) & 
                    (all_coords[:, 1] <= tile_y_max)
                )
                
                tile_features = all_features[mask]
                tile_coords = all_coords[mask]
                
                if len(tile_features) > 0:
                    tiles_generated += 1
                    yield tile_features, tile_coords
        
        logger.info(f"Generated {tiles_generated} spatial tiles")


class MaxPDataIterator(AnalysisDataIterator):
    """Connected region iterator for MaxP regionalization."""
    
    def iterate(self, dataset: BaseDataset, batch_size: int = 5000) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate ensuring spatial connectivity for MaxP.
        
        Args:
            dataset: Dataset to iterate over
            batch_size: Size of connected batches
            
        Yields:
            Tuple of (features, coordinates) for connected regions
        """
        logger.info(f"MaxP iterator: Creating connected batches of size {batch_size}")
        
        # For MaxP, we need to maintain spatial connectivity
        # This simplified version just yields in spatial order
        # A full implementation would build a spatial graph
        
        current_batch_features = []
        current_batch_coords = []
        
        for features, coords in dataset.get_features_and_coords():
            for i in range(len(features)):
                current_batch_features.append(features[i])
                current_batch_coords.append(coords[i])
                
                # Yield batch when size reached
                if len(current_batch_features) >= batch_size:
                    yield (
                        np.array(current_batch_features),
                        np.array(current_batch_coords)
                    )
                    current_batch_features = []
                    current_batch_coords = []
        
        # Yield remaining data
        if current_batch_features:
            yield (
                np.array(current_batch_features),
                np.array(current_batch_coords)
            )