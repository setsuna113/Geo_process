"""MESF Preprocessor V2 with proper caching that never modifies original files.

This version:
1. Never modifies original data files
2. Caches MESF based on (file, spatial_sampling, params)
3. Returns path to cached augmented file
"""

import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from ..spatial.mesf_cache_manager import MESFCacheManager
from ..spatial.mesf import StandardMESF as MESF, NystromMESF

logger = logging.getLogger(__name__)


class MESFPreprocessor:
    """MESF preprocessor with proper caching support."""
    
    def __init__(self, 
                 max_exact_points: int = 10000,
                 cache_dir: Optional[str] = None):
        """Initialize MESF preprocessor.
        
        Args:
            max_exact_points: Maximum points for exact MESF computation
                            Above this, uses Nystrom approximation
            cache_dir: Directory for caching MESF results
        """
        self.max_exact_points = max_exact_points
        self.cache_manager = MESFCacheManager(cache_dir)
        
        logger.info(f"Initialized MESFPreprocessor with cache at {self.cache_manager.cache_dir}")
    
    def process_with_sampling(self,
                             input_parquet: str,
                             spatial_grid_step: int,
                             spatial_config: Dict[str, Any]) -> str:
        """Process parquet with spatial sampling and MESF computation.
        
        This is the main entry point that:
        1. Checks cache for existing MESF with these parameters
        2. If not cached, loads data with sampling and computes MESF
        3. Returns path to cached file with MESF columns
        
        Args:
            input_parquet: Path to input parquet file
            spatial_grid_step: Spatial sampling rate (1 = no sampling)
            spatial_config: MESF computation parameters
            
        Returns:
            Path to cached parquet file with MESF columns
        """
        # Extract MESF parameters
        mesf_params = spatial_config.get('mesf', {
            'variance_threshold': 0.75,
            'max_eigenvectors': 10,
            'min_eigenvectors': 5,
            'distance_threshold': 100
        })
        
        # Define compute function for cache manager
        def compute_mesf_with_sampling(file_path: str, 
                                      sampling: int, 
                                      params: Dict[str, Any]) -> Tuple[pa.Table, Dict]:
            """Compute MESF on sampled data."""
            # Load and sample data
            if sampling > 1:
                logger.info(f"Loading data with spatial sampling: every {sampling}th point")
                table = self._load_with_spatial_sampling(file_path, sampling)
            else:
                logger.info("Loading full data (no sampling)")
                table = pq.read_table(file_path)
            
            # Extract coordinates
            coords_df = table.select(['longitude', 'latitude']).to_pandas()
            n_points = len(coords_df)
            
            logger.info(f"Computing MESF for {n_points:,} points")
            
            # Compute MESF
            if n_points <= self.max_exact_points:
                mesf_features = self._compute_exact_mesf(coords_df, params)
            else:
                mesf_features = self._compute_approximate_mesf(coords_df, params)
            
            # Add MESF columns to table
            mesf_cols = [f'ME{i+1}' for i in range(mesf_features.shape[1])]
            for i, col_name in enumerate(mesf_cols):
                table = table.append_column(col_name, pa.array(mesf_features[:, i]))
            
            # Create metadata
            metadata = {
                'n_samples': n_points,
                'spatial_grid_step': sampling,
                'n_mesf_components': mesf_features.shape[1],
                'mesf_params': params,
                'original_file': file_path
            }
            
            return table, metadata
        
        # Use cache manager to get or compute
        cached_path = self.cache_manager.get_or_compute_mesf(
            input_parquet,
            spatial_grid_step,
            mesf_params,
            compute_mesf_with_sampling
        )
        
        return cached_path
    
    def _load_with_spatial_sampling(self, 
                                   parquet_path: str, 
                                   spatial_grid_step: int) -> pa.Table:
        """Load parquet with spatial grid sampling.
        
        Args:
            parquet_path: Path to parquet file
            spatial_grid_step: Take every Nth point in spatial grid
            
        Returns:
            PyArrow table with sampled data
        """
        import pyarrow.parquet as pq
        
        # Open parquet file for metadata
        parquet_file = pq.ParquetFile(parquet_path)
        
        # Get coordinate columns
        all_columns = parquet_file.schema.names
        if 'longitude' in all_columns and 'latitude' in all_columns:
            x_col, y_col = 'longitude', 'latitude'
        elif 'x' in all_columns and 'y' in all_columns:
            x_col, y_col = 'x', 'y'
        else:
            raise ValueError(f"No coordinate columns found in {all_columns[:10]}")
        
        # Read coordinates to find unique values
        logger.info("Scanning coordinates for spatial grid...")
        unique_x = set()
        unique_y = set()
        
        for batch in parquet_file.iter_batches(columns=[x_col, y_col], batch_size=1000000):
            batch_df = batch.to_pandas()
            unique_x.update(batch_df[x_col].unique())
            unique_y.update(batch_df[y_col].unique())
        
        # Select every Nth value
        unique_x = sorted(unique_x)
        unique_y = sorted(unique_y)
        selected_x = unique_x[::spatial_grid_step]
        selected_y = unique_y[::spatial_grid_step]
        
        logger.info(f"Spatial grid: {len(unique_x)}x{len(unique_y)} -> "
                   f"{len(selected_x)}x{len(selected_y)} points")
        
        # Use PyArrow predicate pushdown for efficient filtering
        filtered_table = pq.read_table(
            parquet_path,
            filters=[
                (x_col, 'in', selected_x),
                (y_col, 'in', selected_y)
            ]
        )
        
        logger.info(f"Loaded {filtered_table.num_rows:,} rows after sampling")
        
        # Rename x,y to longitude,latitude if needed
        if x_col == 'x' and y_col == 'y':
            filtered_table = filtered_table.rename_columns({
                'x': 'longitude',
                'y': 'latitude'
            })
        
        return filtered_table
    
    def _compute_exact_mesf(self, 
                           coords_df: pd.DataFrame,
                           mesf_params: Dict[str, Any]) -> np.ndarray:
        """Compute exact MESF for manageable number of points.
        
        Args:
            coords_df: DataFrame with longitude, latitude columns
            mesf_params: MESF computation parameters
            
        Returns:
            MESF eigenvectors as numpy array
        """
        coords = coords_df[['longitude', 'latitude']].values
        
        # Initialize MESF
        mesf_computer = MESF(
            variance_threshold=mesf_params.get('variance_threshold', 0.75),
            max_eigenvectors=mesf_params.get('max_eigenvectors', 10),
            min_eigenvectors=mesf_params.get('min_eigenvectors', 5),
            distance_threshold=mesf_params.get('distance_threshold', 100)
        )
        
        # Compute eigenvectors
        eigenvectors = mesf_computer.fit_transform(coords)
        
        logger.info(f"Computed {eigenvectors.shape[1]} MESF eigenvectors")
        
        return eigenvectors
    
    def _compute_approximate_mesf(self,
                                 coords_df: pd.DataFrame,
                                 mesf_params: Dict[str, Any]) -> np.ndarray:
        """Compute approximate MESF using Nystrom approximation.
        
        Uses Nystrom method as specified in regression_analysis_decisions.md.
        Suitable for large datasets (>100k points).
        
        Args:
            coords_df: DataFrame with longitude, latitude columns
            mesf_params: MESF computation parameters
            
        Returns:
            MESF eigenvectors as numpy array
        """
        logger.info("Using Nystrom MESF approximation for large dataset")
        
        coords = coords_df[['longitude', 'latitude']].values
        n_points = len(coords)
        
        # Use Nystrom approximation for large datasets
        mesf_computer = NystromMESF(
            n_landmark=mesf_params.get('n_landmark', 5000),
            n_sample=mesf_params.get('n_sample', 25000),
            variance_threshold=mesf_params.get('variance_threshold', 0.75),
            max_eigenvectors=mesf_params.get('max_eigenvectors', 10),
            min_eigenvectors=mesf_params.get('min_eigenvectors', 5),
            distance_threshold=mesf_params.get('distance_threshold', 100)
        )
        
        logger.info(f"Computing Nystrom MESF for {n_points:,} points")
        eigenvectors = mesf_computer.fit_transform(coords)
        logger.info(f"Computed {eigenvectors.shape[1]} MESF eigenvectors using Nystrom")
        
        return eigenvectors
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about MESF cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache_manager.get_cache_info()
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """Clear MESF cache.
        
        Args:
            older_than_days: Only clear files older than this many days
        """
        self.cache_manager.clear_cache(older_than_days)