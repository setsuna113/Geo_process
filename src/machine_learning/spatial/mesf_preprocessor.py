"""
MESF Preprocessor - Separate workflow for computing and caching MESF features.

This module handles MESF computation separately from the main pipeline to enable:
1. One-time computation for large datasets
2. Direct H2O loading without pandas conversion
3. Memory-efficient processing for 180M+ points
"""

import os
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Tuple, Union
from sklearn.neighbors import NearestNeighbors

from .mesf import MESF

logger = logging.getLogger(__name__)


class MESFPreprocessor:
    """Handles MESF computation and caching for large-scale datasets."""
    
    def __init__(self, 
                 cache_dir: str = "cache/mesf",
                 max_exact_points: int = 500000,
                 grid_resolution_km: float = 10.0):
        """
        Initialize MESF preprocessor.
        
        Args:
            cache_dir: Directory for caching MESF features
            max_exact_points: Maximum points for exact MESF computation
            grid_resolution_km: Grid resolution for approximate MESF
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_exact_points = max_exact_points
        self.grid_resolution_km = grid_resolution_km
        
    def get_mesf_path(self, input_path: str) -> Path:
        """Get the path for cached MESF features."""
        input_name = Path(input_path).stem
        return self.cache_dir / f"{input_name}_mesf.parquet"
    
    def check_mesf_exists(self, input_path: str) -> bool:
        """Check if MESF features already computed for this dataset."""
        mesf_path = self.get_mesf_path(input_path)
        exists = mesf_path.exists()
        
        if exists:
            logger.info(f"MESF features found in cache: {mesf_path}")
        else:
            logger.info(f"MESF features not found, need to compute: {mesf_path}")
            
        return exists
    
    def compute_and_cache_mesf(self, 
                               input_parquet: str,
                               spatial_config: dict,
                               force_recompute: bool = False) -> str:
        """
        Compute MESF features and cache them separately.
        
        Args:
            input_parquet: Path to input parquet file
            spatial_config: Spatial configuration for MESF
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            Path to MESF features parquet file
        """
        mesf_path = self.get_mesf_path(input_parquet)
        
        # Check if already exists
        if mesf_path.exists() and not force_recompute:
            logger.info(f"Using cached MESF features: {mesf_path}")
            return str(mesf_path)
        
        logger.info("Computing MESF features...")
        
        # Load coordinates and ID
        logger.info(f"Loading coordinates from {input_parquet}")
        coords_df = pd.read_parquet(
            input_parquet, 
            columns=['longitude', 'latitude']
        )
        
        n_points = len(coords_df)
        logger.info(f"Loaded {n_points:,} coordinate pairs")
        
        # Decide computation strategy
        if n_points <= self.max_exact_points:
            logger.info(f"Using exact MESF computation ({n_points:,} points)")
            mesf_features = self._compute_exact_mesf(coords_df, spatial_config)
        else:
            logger.info(f"Using approximate MESF for {n_points:,} points")
            logger.info(f"Grid resolution: {self.grid_resolution_km} km")
            mesf_features = self._compute_approximate_mesf(coords_df, spatial_config)
        
        # Create DataFrame with MESF features
        mesf_df = pd.DataFrame(
            mesf_features,
            columns=[f'ME{i+1}' for i in range(mesf_features.shape[1])],
            index=coords_df.index
        )
        
        # Add index for joining
        mesf_df['row_id'] = np.arange(len(mesf_df))
        
        # Save to parquet
        logger.info(f"Saving MESF features to {mesf_path}")
        mesf_df.to_parquet(mesf_path, compression='snappy')
        
        logger.info(f"MESF computation complete: {mesf_features.shape}")
        return str(mesf_path)
    
    def _compute_exact_mesf(self, 
                           coords_df: pd.DataFrame, 
                           spatial_config: dict) -> np.ndarray:
        """Compute exact MESF for manageable number of points."""
        coords = coords_df[['longitude', 'latitude']].values
        
        # Initialize MESF
        mesf_computer = MESF(spatial_config)
        
        # Compute eigenvectors
        eigenvectors = mesf_computer.compute(coords)
        
        return eigenvectors
    
    def _compute_approximate_mesf(self, 
                                  coords_df: pd.DataFrame,
                                  spatial_config: dict) -> np.ndarray:
        """
        Compute approximate MESF using grid aggregation.
        
        For 180M points:
        1. Aggregate to grid cells (~500K cells)
        2. Compute MESF on grid centers
        3. Interpolate back to original points
        """
        logger.info("Starting approximate MESF computation")
        
        # Step 1: Create grid
        coords = coords_df[['longitude', 'latitude']].values
        grid_coords, point_to_grid = self._create_spatial_grid(coords)
        
        logger.info(f"Created grid with {len(grid_coords):,} cells")
        
        # Step 2: Compute MESF on grid
        logger.info("Computing MESF on grid centers...")
        mesf_computer = MESF(spatial_config)
        grid_mesf = mesf_computer.compute(grid_coords)
        
        logger.info(f"Computed {grid_mesf.shape[1]} eigenvectors on grid")
        
        # Step 3: Interpolate to full points
        logger.info("Interpolating MESF to full dataset...")
        full_mesf = self._interpolate_mesf(
            grid_coords, 
            grid_mesf, 
            coords, 
            point_to_grid
        )
        
        return full_mesf
    
    def _create_spatial_grid(self, 
                            coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial grid for approximate MESF.
        
        Returns:
            grid_coords: Unique grid cell centers
            point_to_grid: Mapping from points to grid cells
        """
        # Convert km to degrees (approximate)
        lat_resolution = self.grid_resolution_km / 111.0  # 1 degree ~ 111km
        lon_resolution = self.grid_resolution_km / 111.0  # Simplified
        
        # Discretize coordinates to grid
        grid_lat = np.round(coords[:, 1] / lat_resolution) * lat_resolution
        grid_lon = np.round(coords[:, 0] / lon_resolution) * lon_resolution
        
        # Get unique grid cells
        grid_points = np.column_stack([grid_lon, grid_lat])
        unique_grids, point_to_grid = np.unique(
            grid_points, 
            axis=0, 
            return_inverse=True
        )
        
        return unique_grids, point_to_grid
    
    def _interpolate_mesf(self,
                         grid_coords: np.ndarray,
                         grid_mesf: np.ndarray,
                         full_coords: np.ndarray,
                         point_to_grid: np.ndarray) -> np.ndarray:
        """
        Interpolate MESF from grid to full points.
        
        Simple approach: Assign grid MESF to all points in that grid cell.
        Advanced approach: Use KNN interpolation for smoother transitions.
        """
        n_points = len(full_coords)
        n_components = grid_mesf.shape[1]
        
        # Simple assignment (fast)
        full_mesf = np.zeros((n_points, n_components))
        for i in range(n_points):
            grid_idx = point_to_grid[i]
            full_mesf[i] = grid_mesf[grid_idx]
        
        # Optional: Smooth with KNN (commented out for speed)
        # logger.info("Smoothing with KNN interpolation...")
        # nbrs = NearestNeighbors(n_neighbors=5, metric='haversine')
        # nbrs.fit(np.radians(grid_coords[:, [1, 0]]))
        # distances, indices = nbrs.kneighbors(np.radians(full_coords[:, [1, 0]]))
        # weights = 1.0 / (distances + 1e-10)
        # weights /= weights.sum(axis=1, keepdims=True)
        # for i in range(n_points):
        #     full_mesf[i] = np.average(grid_mesf[indices[i]], weights=weights[i], axis=0)
        
        return full_mesf
    
    def augment_parquet_with_mesf(self,
                                  input_parquet: str,
                                  output_parquet: str,
                                  mesf_parquet: str) -> str:
        """
        Create new parquet file with MESF features included.
        
        Memory-efficient chunked processing for large files.
        """
        logger.info(f"Augmenting {input_parquet} with MESF features")
        
        # Load MESF features
        mesf_df = pd.read_parquet(mesf_parquet)
        mesf_cols = [col for col in mesf_df.columns if col.startswith('ME')]
        
        # Read original parquet in chunks and append MESF
        original = pq.ParquetFile(input_parquet)
        
        # Create new schema with MESF columns
        mesf_fields = [
            pa.field(col, pa.float32()) 
            for col in mesf_cols
        ]
        
        # Get original schema
        original_schema = original.schema_pandas
        
        # Write augmented parquet
        chunk_size = 100000
        first_chunk = True
        
        for batch in original.iter_batches(batch_size=chunk_size):
            batch_df = batch.to_pandas()
            
            # Add MESF features for this batch
            start_idx = batch_df.index[0]
            end_idx = batch_df.index[-1] + 1
            
            for col in mesf_cols:
                batch_df[col] = mesf_df.loc[start_idx:end_idx-1, col].values
            
            # Write to output
            if first_chunk:
                batch_df.to_parquet(output_parquet, compression='snappy')
                first_chunk = False
            else:
                batch_df.to_parquet(output_parquet, compression='snappy', append=True)
        
        logger.info(f"Created augmented parquet: {output_parquet}")
        return output_parquet