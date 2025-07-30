# src/processors/data_preparation/coordinate_merger.py
"""Coordinate-based merger for passthrough and resampled data."""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from src.base.processor import BaseProcessor
from src.database.connection import DatabaseManager
from src.domain.raster.catalog import RasterCatalog
from src.domain.validators.coordinate_integrity import (
    BoundsConsistencyValidator, CoordinateTransformValidator, ParquetValueValidator
)
from src.abstractions.interfaces.validator import ValidationSeverity
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_stage, log_operation

logger = get_logger(__name__)


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
        
        # Initialize validators
        self.bounds_validator = BoundsConsistencyValidator(tolerance=1e-6)
        self.transform_validator = CoordinateTransformValidator(max_error_meters=1.0)
        self.value_validator = ParquetValueValidator(
            max_null_percentage=10.0,
            outlier_std_threshold=3.0
        )
        
        # Validation results tracking
        self.validation_results: List[Dict] = []
        
    def create_merged_dataset(self,
                            resampled_datasets: List[Dict],
                            chunk_size: Optional[int] = None,
                            return_as: str = 'xarray',
                            context: Optional[any] = None) -> any:
        """Create merged dataset and return in-memory without writing to file.
        
        Args:
            resampled_datasets: List of dataset info dicts
            chunk_size: If provided, use chunked processing for memory efficiency
            return_as: 'xarray' or 'dataframe' - format to return
            context: Optional pipeline context with memory monitor
            
        Returns:
            xarray.Dataset or pandas.DataFrame with merged data
        """
        import xarray as xr
        
        logger.info(f"Creating merged dataset in memory (return_as={return_as})")
        self.validation_results.clear()
        
        # Get memory monitor
        memory_monitor = context.memory_monitor if context and hasattr(context, 'memory_monitor') else None
        
        # Adaptive chunk size
        base_chunk_size = chunk_size or self.config.get('merge.chunk_size', 5000)
        self._adaptive_chunk_size = base_chunk_size
        
        def on_memory_warning(usage):
            self._adaptive_chunk_size = max(1000, self._adaptive_chunk_size // 2)
            logger.warning(f"Memory pressure (warning): reducing merge chunk size to {self._adaptive_chunk_size}")
        
        def on_memory_critical(usage):
            self._adaptive_chunk_size = 500  # Minimum viable chunk size
            logger.error(f"Memory pressure (critical): merge chunk size set to minimum {self._adaptive_chunk_size}")
            # Force garbage collection
            import gc
            gc.collect()
        
        # Register callbacks
        if memory_monitor:
            memory_monitor.register_warning_callback(on_memory_warning)
            memory_monitor.register_critical_callback(on_memory_critical)
        
        # Validate all datasets first
        for dataset_info in resampled_datasets:
            self._validate_dataset_bounds(dataset_info)
        
        # Load and merge data
        all_dfs = []
        for dataset_info in resampled_datasets:
            df = self._load_dataset_coordinates(dataset_info)
            if df is not None:
                self._validate_coordinate_data(df, dataset_info['name'])
                all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No coordinate data found in any dataset")
        
        # Validate spatial consistency
        self._validate_spatial_consistency(all_dfs, resampled_datasets)
        
        # Merge all datasets
        merged_df = self._merge_coordinate_datasets(all_dfs)
        
        # Validate merged result
        self._validate_merged_data(merged_df)
        
        # Convert to requested format
        if return_as == 'dataframe':
            return merged_df
        elif return_as == 'xarray':
            # Convert to xarray Dataset
            # Set x and y as dimensions
            ds = xr.Dataset()
            
            # Get unique coordinates
            unique_x = sorted(merged_df['x'].unique())
            unique_y = sorted(merged_df['y'].unique())
            
            # Create coordinate arrays
            ds.coords['x'] = unique_x
            ds.coords['y'] = unique_y
            
            # For each data variable, create a 2D array
            data_vars = [col for col in merged_df.columns if col not in ['x', 'y']]
            
            for var in data_vars:
                # Pivot data to 2D array
                pivoted = merged_df.pivot(index='y', columns='x', values=var)
                # Reindex to ensure all coordinates are present
                pivoted = pivoted.reindex(index=unique_y, columns=unique_x)
                ds[var] = (['y', 'x'], pivoted.values)
            
            return ds
        else:
            raise ValueError(f"Unknown return format: {return_as}")
    
    @log_stage("coordinate_merge")
    def create_ml_ready_parquet(self, 
                               resampled_datasets: List[Dict], 
                               output_dir: Path,
                               chunk_size: Optional[int] = None) -> Path:
        """Main entry point - creates parquet from database tables.
        
        Args:
            resampled_datasets: List of dataset info dicts
            output_dir: Output directory for parquet file
            chunk_size: If provided, use chunked processing for memory efficiency
        """
        
        logger.info(f"Starting coordinate merger with validation (chunk_size={chunk_size})")
        self.validation_results.clear()
        
        # Validate all datasets first
        for dataset_info in resampled_datasets:
            self._validate_dataset_bounds(dataset_info)
        
        if chunk_size or hasattr(self, '_adaptive_chunk_size'):
            # Use chunked processing for large datasets
            effective_chunk_size = getattr(self, '_adaptive_chunk_size', chunk_size) if hasattr(self, '_adaptive_chunk_size') else chunk_size
            return self._create_ml_ready_parquet_chunked(
                resampled_datasets, output_dir, effective_chunk_size
            )
        else:
            # Original in-memory processing
            return self._create_ml_ready_parquet_inmemory(
                resampled_datasets, output_dir
            )
    
    @log_operation("merge_inmemory")
    def _create_ml_ready_parquet_inmemory(self, 
                                         resampled_datasets: List[Dict], 
                                         output_dir: Path) -> Path:
        """Original in-memory processing (for smaller datasets)."""
        
        # Step 1: Load coordinate data
        all_dfs = []
        for dataset_info in resampled_datasets:
            df = self._load_dataset_coordinates(dataset_info)
            if df is not None:
                # Validate loaded coordinates
                self._validate_coordinate_data(df, dataset_info['name'])
                all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No coordinate data found in any dataset")
        
        # Step 2: Validate spatial consistency before merging
        self._validate_spatial_consistency(all_dfs, resampled_datasets)
        
        # Step 3: Merge all datasets
        merged_df = self._merge_coordinate_datasets(all_dfs)
        
        # Step 4: Validate merged result
        self._validate_merged_data(merged_df)
        
        # Step 5: Save as parquet
        output_path = output_dir / 'ml_ready_aligned_data.parquet'
        merged_df.to_parquet(output_path, index=False)
        
        # Step 6: Final validation of saved file
        self._validate_output_file(output_path)
        
        logger.info(f"Created ML-ready parquet with {len(merged_df):,} rows and {len(merged_df.columns)} columns")
        logger.info(f"Columns: {list(merged_df.columns)}")
        
        # Report validation summary
        self._report_validation_summary()
        
        return output_path
    
    @log_operation("merge_chunked")
    def _create_ml_ready_parquet_chunked(self,
                                       resampled_datasets: List[Dict],
                                       output_dir: Path,
                                       chunk_size: int) -> Path:
        """Chunked processing for memory-efficient merging of large datasets."""
        
        logger.info(f"Using chunked processing with chunk_size={chunk_size}")
        
        # Get bounds that encompass all datasets
        overall_bounds = self._get_overall_bounds(resampled_datasets)
        min_x, min_y, max_x, max_y = overall_bounds
        
        # Get the finest resolution among all datasets
        min_resolution = min(d['resolution'] for d in resampled_datasets)
        
        # Calculate grid dimensions
        width = max_x - min_x
        height = max_y - min_y
        chunks_x = max(1, int(width / (chunk_size * min_resolution)))
        chunks_y = max(1, int(height / (chunk_size * min_resolution)))
        
        chunk_width = width / chunks_x
        chunk_height = height / chunks_y
        
        logger.info(f"Processing {chunks_x} x {chunks_y} chunks")
        
        # Initialize output file with first chunk to get schema
        output_path = output_dir / 'ml_ready_aligned_data.parquet'
        first_chunk = True
        total_rows = 0
        
        # Process each chunk
        total_chunks = chunks_x * chunks_y
        chunk_idx = 0
        
        for i in range(chunks_x):
            for j in range(chunks_y):
                chunk_idx += 1
                progress = chunk_idx / total_chunks * 100
                
                # Define chunk bounds
                chunk_min_x = min_x + i * chunk_width
                chunk_max_x = min_x + (i + 1) * chunk_width
                chunk_min_y = min_y + j * chunk_height
                chunk_max_y = min_y + (j + 1) * chunk_height
                chunk_bounds = (chunk_min_x, chunk_min_y, chunk_max_x, chunk_max_y)
                
                logger.info(f"Processing chunk ({i},{j}) [{chunk_idx}/{total_chunks}, {progress:.1f}%]")
                
                # Load data for this chunk from all datasets
                chunk_dfs = []
                for dataset_info in resampled_datasets:
                    df = self._load_dataset_coordinates_bounded(
                        dataset_info, chunk_bounds
                    )
                    if df is not None and not df.empty:
                        chunk_dfs.append(df)
                
                if not chunk_dfs:
                    continue
                
                # Merge chunk data
                merged_chunk = self._merge_coordinate_datasets(chunk_dfs)
                
                if merged_chunk.empty:
                    continue
                
                # Write chunk to parquet
                if first_chunk:
                    # Create new file with first chunk
                    merged_chunk.to_parquet(output_path, index=False)
                    first_chunk = False
                else:
                    # Append to existing file
                    import pyarrow.parquet as pq
                    import pyarrow as pa
                    
                    # Read existing data
                    existing_table = pq.read_table(output_path)
                    new_table = pa.Table.from_pandas(merged_chunk)
                    
                    # Concatenate tables
                    combined_table = pa.concat_tables([existing_table, new_table])
                    
                    # Write back
                    pq.write_table(combined_table, output_path)
                
                total_rows += len(merged_chunk)
                logger.info(f"Chunk ({i},{j}) added {len(merged_chunk)} rows", 
                          extra={
                              'chunk_index': chunk_idx,
                              'chunk_bounds': chunk_bounds,
                              'rows_added': len(merged_chunk),
                              'total_rows': total_rows
                          })
        
        if first_chunk:
            raise ValueError("No data found in any chunks")
        
        # Final validation
        self._validate_output_file(output_path)
        
        logger.info(f"Created ML-ready parquet with {total_rows:,} total rows")
        self._report_validation_summary()
        
        return output_path
    
    def _get_overall_bounds(self, resampled_datasets: List[Dict]) -> Tuple[float, float, float, float]:
        """Get bounds that encompass all datasets."""
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for dataset in resampled_datasets:
            bounds = dataset.get('bounds')
            if bounds:
                if isinstance(bounds, list):
                    bounds = tuple(bounds)
                d_min_x, d_min_y, d_max_x, d_max_y = bounds
                min_x = min(min_x, d_min_x)
                min_y = min(min_y, d_min_y)
                max_x = max(max_x, d_max_x)
                max_y = max(max_y, d_max_y)
        
        return (min_x, min_y, max_x, max_y)
    
    def _load_dataset_coordinates_bounded(self, 
                                        dataset_info: Dict,
                                        bounds: Tuple[float, float, float, float]) -> Optional[pd.DataFrame]:
        """Load coordinates within specified bounds."""
        
        # Get actual bounds from the dataset info
        dataset_bounds = dataset_info.get('bounds')
        if not dataset_bounds:
            logger.error(f"No bounds found for dataset {dataset_info['name']}")
            return None
        
        if isinstance(dataset_bounds, list):
            dataset_bounds = tuple(dataset_bounds)
        
        # Check if bounds overlap
        d_min_x, d_min_y, d_max_x, d_max_y = dataset_bounds
        b_min_x, b_min_y, b_max_x, b_max_y = bounds
        
        if (d_max_x < b_min_x or d_min_x > b_max_x or
            d_max_y < b_min_y or d_min_y > b_max_y):
            # No overlap
            return None
        
        # Load with spatial filter
        if dataset_info.get('passthrough', False):
            return self._load_passthrough_coordinates_bounded(
                dataset_info['name'],
                dataset_info['table_name'],
                dataset_bounds,
                dataset_info['resolution'],
                bounds
            )
        else:
            return self._load_resampled_coordinates_bounded(
                dataset_info['name'],
                dataset_info['table_name'],
                bounds,
                dataset_bounds,
                dataset_info['resolution']
            )
    
    def _load_passthrough_coordinates_bounded(self, name: str, table_name: str,
                                            dataset_bounds: Tuple, resolution: float,
                                            query_bounds: Tuple) -> pd.DataFrame:
        """Load passthrough coordinates within bounds."""
        
        min_x, min_y, max_x, max_y = dataset_bounds
        q_min_x, q_min_y, q_max_x, q_max_y = query_bounds
        
        # Check if table has coordinate columns
        if self._table_has_coordinates(table_name):
            # Use coordinate columns with spatial filter
            query = f"""
                SELECT x_coord as x, y_coord as y, value AS {name.replace('-', '_')}
                FROM {table_name}
                WHERE value IS NOT NULL AND value != 0
                AND x_coord >= {q_min_x} AND x_coord <= {q_max_x}
                AND y_coord >= {q_min_y} AND y_coord <= {q_max_y}
            """
        else:
            # Legacy format - convert indices with spatial filter
            query = f"""
                WITH coords AS (
                    SELECT 
                        {min_x} + (col_idx + 0.5) * {resolution} AS x,
                        {max_y} - (row_idx + 0.5) * {resolution} AS y,
                        value AS {name.replace('-', '_')}
                    FROM {table_name}
                    WHERE value IS NOT NULL AND value != 0
                )
                SELECT * FROM coords
                WHERE x >= {q_min_x} AND x <= {q_max_x}
                AND y >= {q_min_y} AND y <= {q_max_y}
            """
        
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)
        
        return df
    
    def _load_resampled_coordinates_bounded(self, name: str, table_name: str,
                                          query_bounds: Tuple,
                                          dataset_bounds: Optional[Tuple] = None,
                                          resolution: Optional[float] = None) -> pd.DataFrame:
        """Load resampled coordinates within bounds."""
        
        q_min_x, q_min_y, q_max_x, q_max_y = query_bounds
        
        # Check if table has coordinate columns
        if self._table_has_coordinates(table_name):
            # New format with spatial filter
            query = f"""
                SELECT x_coord as x, y_coord as y, value AS {name.replace('-', '_')}
                FROM {table_name}
                WHERE value IS NOT NULL AND value != 0
                AND x_coord >= {q_min_x} AND x_coord <= {q_max_x}
                AND y_coord >= {q_min_y} AND y_coord <= {q_max_y}
            """
        else:
            # Legacy format - need bounds and resolution
            if not dataset_bounds or not resolution:
                logger.error(f"Table {table_name} lacks coordinates and no bounds/resolution provided")
                return pd.DataFrame()
            
            # Use passthrough method with bounds
            return self._load_passthrough_coordinates_bounded(
                name, table_name, dataset_bounds, resolution, query_bounds
            )
        
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)
        
        return df
    
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
        
        # Check if table already has coordinate columns (e.g., from migration)
        if self._table_has_coordinates(table_name):
            # Use coordinate columns directly
            query = f"""
                SELECT x_coord as x, y_coord as y, value AS {name.replace('-', '_')}
                FROM {table_name}
                WHERE value IS NOT NULL AND value != 0
            """
            logger.info(f"Loading passthrough data for {name} using existing coordinate columns")
        else:
            # Legacy format - convert indices to coordinates on the fly
            min_x, min_y, max_x, max_y = bounds
            
            logger.info(f"Loading passthrough data for {name} with bounds: {bounds} (converting indices)")
            
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
    
    def _table_has_coordinates(self, table_name: str) -> bool:
        """Check if table has x_coord and y_coord columns."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s 
                    AND column_name IN ('x_coord', 'y_coord')
                """, (table_name,))
                columns = cursor.fetchall()
                # Should have both x_coord and y_coord columns
                return len(columns) == 2
        except Exception as e:
            logger.warning(f"Error checking table columns for {table_name}: {e}")
            return False
    
    def _load_resampled_coordinates(self, name: str, table_name: str,
                                   bounds: Optional[Tuple] = None,
                                   resolution: Optional[float] = None) -> pd.DataFrame:
        """Load coordinates from resampled table (handles both formats)."""
        
        # Check if table has coordinate columns
        if self._table_has_coordinates(table_name):
            # New format with x_coord, y_coord columns
            query = f"""
                SELECT x_coord as x, y_coord as y, value AS {name.replace('-', '_')}
                FROM {table_name}
                WHERE value IS NOT NULL AND value != 0
            """
        else:
            # Legacy format - need bounds and resolution for conversion
            if not bounds or not resolution:
                logger.error(f"Table {table_name} lacks coordinates and no bounds/resolution provided")
                return None
                
            # Use same conversion as passthrough
            return self._load_passthrough_coordinates(name, table_name, bounds, resolution)
        
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
    
    # Validation methods
    def _validate_dataset_bounds(self, dataset_info: Dict) -> None:
        """Validate bounds consistency for a dataset."""
        bounds = dataset_info.get('bounds')
        if not bounds:
            logger.error(f"No bounds found for dataset {dataset_info['name']}")
            return
        
        validation_data = {
            'bounds': bounds,
            'crs': dataset_info.get('crs', 'EPSG:4326')
        }
        
        result = self.bounds_validator.validate(validation_data)
        self.validation_results.append({
            'stage': 'dataset_bounds',
            'dataset': dataset_info['name'],
            'result': result
        })
        
        if not result.is_valid:
            error_messages = [issue.message for issue in result.issues 
                            if issue.severity == ValidationSeverity.ERROR]
            logger.error(f"Bounds validation failed for {dataset_info['name']}: {error_messages}")
            if result.has_errors:
                raise ValueError(f"Invalid bounds for dataset {dataset_info['name']}: {error_messages}")
        
        # Log warnings
        warnings = [issue.message for issue in result.issues 
                   if issue.severity == ValidationSeverity.WARNING]
        for warning in warnings:
            logger.warning(f"Bounds validation warning for {dataset_info['name']}: {warning}")
    
    def _validate_coordinate_data(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Validate coordinate data after loading."""
        if df.empty:
            return
        
        # Validate coordinate ranges
        if 'x' in df.columns and 'y' in df.columns:
            # Check for reasonable coordinate ranges (adjust based on your domain)
            x_range = (df['x'].min(), df['x'].max())
            y_range = (df['y'].min(), df['y'].max())
            
            # Log coordinate ranges for inspection
            logger.info(f"Dataset {dataset_name} coordinate ranges: "
                       f"X: [{x_range[0]:.6f}, {x_range[1]:.6f}], "
                       f"Y: [{y_range[0]:.6f}, {y_range[1]:.6f}]")
            
            # Check for suspicious coordinate values
            if abs(x_range[0]) > 180 or abs(x_range[1]) > 180:
                if abs(x_range[0]) < 1000000:  # Not projected coordinates
                    logger.warning(f"Dataset {dataset_name} has suspicious X coordinates: {x_range}")
            
            if abs(y_range[0]) > 90 or abs(y_range[1]) > 90:
                if abs(y_range[0]) < 1000000:  # Not projected coordinates
                    logger.warning(f"Dataset {dataset_name} has suspicious Y coordinates: {y_range}")
        
        # Validate using ParquetValueValidator
        result = self.value_validator.validate(df)
        self.validation_results.append({
            'stage': 'coordinate_data',
            'dataset': dataset_name,
            'result': result
        })
        
        if not result.is_valid:
            error_messages = [issue.message for issue in result.issues 
                            if issue.severity == ValidationSeverity.ERROR]
            logger.error(f"Coordinate data validation failed for {dataset_name}: {error_messages}")
        
        # Log warnings
        warnings = [issue.message for issue in result.issues 
                   if issue.severity == ValidationSeverity.WARNING]
        for warning in warnings:
            logger.warning(f"Coordinate data validation warning for {dataset_name}: {warning}")
    
    def _validate_spatial_consistency(self, dfs: List[pd.DataFrame], dataset_infos: List[Dict]) -> None:
        """Validate spatial consistency between datasets before merging."""
        if len(dfs) < 2:
            return
        
        logger.info("Validating spatial consistency between datasets")
        
        # Check coordinate overlaps
        for i, df1 in enumerate(dfs):
            for j, df2 in enumerate(dfs[i+1:], i+1):
                dataset1 = dataset_infos[i]['name']
                dataset2 = dataset_infos[j]['name']
                
                # Find common coordinates
                if 'x' in df1.columns and 'y' in df1.columns and 'x' in df2.columns and 'y' in df2.columns:
                    # Round coordinates for comparison
                    df1_coords = set(zip(df1['x'].round(6), df1['y'].round(6)))
                    df2_coords = set(zip(df2['x'].round(6), df2['y'].round(6)))
                    
                    common_coords = df1_coords & df2_coords
                    overlap_percentage = len(common_coords) / min(len(df1_coords), len(df2_coords)) * 100
                    
                    logger.info(f"Spatial overlap between {dataset1} and {dataset2}: "
                               f"{len(common_coords):,} points ({overlap_percentage:.1f}%)")
                    
                    if overlap_percentage < 10:
                        logger.warning(f"Low spatial overlap between {dataset1} and {dataset2}: "
                                     f"{overlap_percentage:.1f}%")
    
    def _validate_merged_data(self, merged_df: pd.DataFrame) -> None:
        """Validate the merged dataset."""
        logger.info("Validating merged dataset")
        
        result = self.value_validator.validate(merged_df)
        self.validation_results.append({
            'stage': 'merged_data',
            'dataset': 'merged',
            'result': result
        })
        
        if not result.is_valid:
            error_messages = [issue.message for issue in result.issues 
                            if issue.severity == ValidationSeverity.ERROR]
            logger.error(f"Merged data validation failed: {error_messages}")
            if result.has_errors:
                raise ValueError(f"Merged data validation failed: {error_messages}")
        
        # Log data quality metrics
        total_cells = len(merged_df)
        non_null_data_cols = [col for col in merged_df.columns if col not in ['x', 'y']]
        
        for col in non_null_data_cols:
            non_null_count = merged_df[col].notna().sum()
            coverage = non_null_count / total_cells * 100
            logger.info(f"Data coverage for {col}: {non_null_count:,}/{total_cells:,} ({coverage:.1f}%)")
    
    def _validate_output_file(self, output_path: Path) -> None:
        """Validate the saved Parquet file."""
        try:
            # Read back and validate
            saved_df = pd.read_parquet(output_path)
            logger.info(f"Successfully validated saved file: {len(saved_df):,} rows, {len(saved_df.columns)} columns")
            
            # Quick validation check
            if saved_df.empty:
                raise ValueError("Saved Parquet file is empty")
            
            if 'x' not in saved_df.columns or 'y' not in saved_df.columns:
                raise ValueError("Saved Parquet file missing coordinate columns")
                
        except Exception as e:
            logger.error(f"Failed to validate output file {output_path}: {e}")
            raise
    
    def _report_validation_summary(self) -> None:
        """Report summary of all validation results."""
        if not self.validation_results:
            logger.info("No validation results to report")
            return
        
        total_validations = len(self.validation_results)
        failed_validations = sum(1 for v in self.validation_results if not v['result'].is_valid)
        
        total_errors = sum(v['result'].error_count for v in self.validation_results)
        total_warnings = sum(v['result'].warning_count for v in self.validation_results)
        
        logger.info("=== VALIDATION SUMMARY ===")
        logger.info(f"Total validations: {total_validations}")
        logger.info(f"Failed validations: {failed_validations}")
        logger.info(f"Total errors: {total_errors}")
        logger.info(f"Total warnings: {total_warnings}")
        
        # Report by stage
        stages = {}
        for v in self.validation_results:
            stage = v['stage']
            if stage not in stages:
                stages[stage] = {'count': 0, 'errors': 0, 'warnings': 0}
            stages[stage]['count'] += 1
            stages[stage]['errors'] += v['result'].error_count
            stages[stage]['warnings'] += v['result'].warning_count
        
        for stage, metrics in stages.items():
            logger.info(f"Stage '{stage}': {metrics['count']} validations, "
                       f"{metrics['errors']} errors, {metrics['warnings']} warnings")
        
        logger.info("=== END VALIDATION SUMMARY ===")
    
    def get_validation_results(self) -> List[Dict]:
        """Get all validation results for external reporting."""
        return self.validation_results.copy()
    
    # BaseProcessor abstract method implementations
    def process_single(self, item: Dict) -> Dict:
        """Process single item - not used for this processor."""
        return item
    
    def validate_input(self, item: Dict) -> Tuple[bool, Optional[str]]:
        """Validate input - not used for this processor."""
        return True, None