# src/processors/data_preparation/coordinate_merger.py
"""Coordinate-based merger for passthrough and resampled data."""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.infrastructure.processors.base_processor import EnhancedBaseProcessor as BaseProcessor
from src.database.connection import DatabaseManager
from src.domain.raster.catalog import RasterCatalog

logger = logging.getLogger(__name__)


class CoordinateMerger(BaseProcessor):
    """Merges datasets from passthrough/resampled tables into ML-ready format."""
    
    def __init__(self, config, db: DatabaseManager):
        super().__init__(
            batch_size=1000,
            config=config,
            enable_progress=True,
            enable_checkpoints=False,
            supports_chunking=True
        )
        self.db = db
        self.catalog = RasterCatalog(db, config)
        
    def create_ml_ready_parquet(self, 
                               resampled_datasets: List[Dict], 
                               output_dir: Path) -> Path:
        """Main entry point - creates parquet from database tables."""
        
        # Step 1: Load coordinate data from each dataset
        all_dfs = []
        for dataset_info in resampled_datasets:
            df = self._load_dataset_coordinates(dataset_info)
            if df is not None:
                all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No coordinate data found in any dataset")
        
        # Step 2: Merge all datasets
        merged_df = self._merge_coordinate_datasets(all_dfs)
        
        # Step 3: Save as parquet
        output_path = output_dir / 'ml_ready_aligned_data.parquet'
        merged_df.to_parquet(output_path, index=False)
        
        logger.info(f"Created ML-ready parquet with {len(merged_df):,} rows and {len(merged_df.columns)} columns")
        logger.info(f"Columns: {list(merged_df.columns)}")
        
        return output_path
    
    def _load_dataset_coordinates(self, dataset_info: Dict) -> Optional[pd.DataFrame]:
        """Load coordinates from passthrough or resampled table."""
        
        # Get actual bounds from the dataset info (now stored with actual bounds)
        bounds = dataset_info.get('bounds')
        if not bounds:
            logger.error(f"No bounds found for dataset {dataset_info['name']}")
            return None
            
        # Ensure bounds is a tuple
        if isinstance(bounds, list):
            bounds = tuple(bounds)
        
        if dataset_info.get('passthrough', False):
            return self._load_passthrough_coordinates(
                dataset_info['name'],
                dataset_info['table_name'],
                bounds,
                dataset_info['resolution']
            )
        else:
            return self._load_resampled_coordinates(
                dataset_info['name'],
                dataset_info['table_name']
            )
    
    def _load_passthrough_coordinates(self, name: str, table_name: str, 
                                    bounds: Tuple, resolution: float) -> pd.DataFrame:
        """Convert row/col indices to coordinates using actual raster bounds."""
        
        min_x, min_y, max_x, max_y = bounds
        
        logger.info(f"Loading passthrough data for {name} with bounds: {bounds}")
        
        # SQL query that does coordinate conversion
        query = f"""
            SELECT 
                {min_x} + (col_idx + 0.5) * {resolution} AS x,
                {max_y} - (row_idx + 0.5) * {resolution} AS y,
                value AS {name.replace('-', '_')}
            FROM {table_name}
            WHERE value IS NOT NULL AND value != 0
        """
        
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)
            
        logger.info(f"Loaded {len(df):,} coordinate points for {name}")
        return df
    
    def _load_resampled_coordinates(self, name: str, table_name: str) -> pd.DataFrame:
        """Load already converted coordinates from resampled table."""
        
        # Check table structure first
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = 'public'
                """, (table_name,))
                columns = [row[0] for row in cur.fetchall()]
        
        if 'x' in columns and 'y' in columns:
            # Standard coordinate structure
            query = f"""
                SELECT x, y, value AS {name.replace('-', '_')}
                FROM {table_name}
                WHERE value IS NOT NULL AND value != 0
            """
        else:
            # This shouldn't happen for resampled data, but handle it anyway
            logger.warning(f"Resampled table {table_name} missing x,y columns")
            return None
        
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)
            
        logger.info(f"Loaded {len(df):,} coordinate points for {name}")
        return df
    
    def _merge_coordinate_datasets(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple coordinate datasets."""
        
        if not dfs:
            raise ValueError("No datasets to merge")
        
        logger.info("Merging coordinate datasets")
        
        # Start with first dataset
        merged = dfs[0]
        
        # Merge others
        for df in dfs[1:]:
            # Round coordinates to avoid floating point issues
            for col in ['x', 'y']:
                merged[col] = merged[col].round(6)
                df[col] = df[col].round(6)
            
            merged = merged.merge(df, on=['x', 'y'], how='outer')
            logger.info(f"After merge: {len(merged):,} rows")
        
        # Sort by coordinates for consistent ordering
        merged = merged.sort_values(['y', 'x']).reset_index(drop=True)
        
        return merged
    
    # BaseProcessor abstract method implementations
    def process_single(self, item: Dict) -> Dict:
        """Process single item - not used for this processor."""
        return item
    
    def validate_input(self, item: Dict) -> Tuple[bool, Optional[str]]:
        """Validate input - not used for this processor."""
        return True, None