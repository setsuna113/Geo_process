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
    
    def check_mesf_in_parquet(self, parquet_path: str) -> bool:
        """Check if MESF columns already exist in the parquet file.
        
        Args:
            parquet_path: Path to parquet file
            
        Returns:
            True if MESF columns (ME1, ME2, ...) exist
        """
        import pyarrow.parquet as pq
        
        # Read just the schema
        parquet_file = pq.ParquetFile(parquet_path)
        columns = parquet_file.schema.names
        
        # Check for ME columns
        me_columns = [col for col in columns if col.startswith('ME')]
        
        if me_columns:
            logger.info(f"Found {len(me_columns)} MESF columns in parquet: {me_columns[:5]}...")
            return True
        else:
            logger.info("No MESF columns found in parquet")
            return False
    
    def process_and_augment_parquet(self,
                                   input_parquet: str,
                                   spatial_config: dict,
                                   force_recompute: bool = False) -> str:
        """
        Process parquet file and add MESF columns if not present.
        
        This is the main entry point that:
        1. Checks if MESF already exists in parquet
        2. If not, computes MESF
        3. Writes MESF back to the original parquet
        
        Args:
            input_parquet: Path to input parquet file
            spatial_config: Spatial configuration for MESF
            force_recompute: Force recomputation even if MESF exists
            
        Returns:
            Path to augmented parquet file (same as input)
        """
        # Check if MESF already in parquet
        if not force_recompute and self.check_mesf_in_parquet(input_parquet):
            logger.info(f"MESF already present in {input_parquet}, skipping computation")
            return input_parquet
        
        logger.info(f"Computing MESF and augmenting {input_parquet}")
        
        # Compute MESF
        mesf_features = self._compute_mesf_for_parquet(input_parquet, spatial_config)
        
        # Write back to parquet
        self._augment_parquet_with_mesf(input_parquet, mesf_features)
        
        logger.info(f"✅ Successfully augmented {input_parquet} with MESF columns")
        return input_parquet
    
    def _compute_mesf_for_parquet(self,
                                  input_parquet: str,
                                  spatial_config: dict) -> np.ndarray:
        """Compute MESF features for the parquet file.
        
        Args:
            input_parquet: Path to input parquet file
            spatial_config: Spatial configuration for MESF
            
        Returns:
            MESF features as numpy array
        """
        
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
    
    def _augment_parquet_with_mesf(self,
                                   input_parquet: str,
                                   mesf_features: np.ndarray) -> None:
        """
        Augment the original parquet file with MESF columns.
        
        This creates a temporary file and then replaces the original.
        Memory-efficient chunked processing for large files.
        
        Args:
            input_parquet: Path to input parquet file
            mesf_features: MESF features array
        """
        import tempfile
        import shutil
        
        logger.info(f"Augmenting {input_parquet} with {mesf_features.shape[1]} MESF columns")
        
        # Create MESF column names
        mesf_cols = [f'ME{i+1}' for i in range(mesf_features.shape[1])]
        
        # Create temporary output file
        temp_output = input_parquet + '.tmp'
        
        try:
            # Read original parquet in chunks
            original = pq.ParquetFile(input_parquet)
            
            # Create new schema with MESF columns
            original_schema = original.schema
            mesf_fields = [pa.field(col, pa.float32()) for col in mesf_cols]
            
            # Combine schemas
            new_fields = list(original_schema) + mesf_fields
            new_schema = pa.schema(new_fields)
            
            # Write augmented parquet
            chunk_size = 100000
            row_offset = 0
            
            with pq.ParquetWriter(temp_output, new_schema, compression='snappy') as writer:
                for batch in original.iter_batches(batch_size=chunk_size):
                    batch_df = batch.to_pandas()
                    
                    # Add MESF features for this batch
                    batch_size_actual = len(batch_df)
                    for i, col in enumerate(mesf_cols):
                        batch_df[col] = mesf_features[row_offset:row_offset+batch_size_actual, i]
                    
                    # Write augmented batch
                    table = pa.Table.from_pandas(batch_df, schema=new_schema)
                    writer.write_table(table)
                    
                    row_offset += batch_size_actual
            
            # Replace original file with augmented version
            shutil.move(temp_output, input_parquet)
            logger.info(f"✅ Successfully replaced {input_parquet} with MESF-augmented version")
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_output):
                os.remove(temp_output)
            raise e