"""Parquet data loader for biodiversity analysis."""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import gc

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from src.abstractions.types.biodiversity_types import BiodiversityData

logger = logging.getLogger(__name__)


class ParquetLoader:
    """Load biodiversity data from parquet files."""
    
    def __init__(self):
        """Initialize parquet loader."""
        self.last_loaded_path = None
        self.last_loaded_data = None
    
    def load(self, 
             data_path: str,
             coordinate_cols: Optional[List[str]] = None,
             feature_cols: Optional[List[str]] = None,
             species_cols: Optional[List[str]] = None,
             validate: bool = True,
             spatial_grid_step: int = 1,
             skip_auto_detection: bool = False,
             skip_coordinate_validation: bool = False,
             skip_zero_inflation_check: bool = False,
             skip_thread_management: bool = False) -> BiodiversityData:
        """Load biodiversity data from parquet file.
        
        Args:
            data_path: Path to parquet file
            coordinate_cols: Names of coordinate columns (auto-detect if None)
            feature_cols: Feature columns to use (all numeric if None)
            species_cols: Species column names if applicable
            validate: Whether to validate the loaded data
            spatial_grid_step: Take every Nth point in spatial grid (1=all points, 10=every 10th)
                             This provides better geographic coverage than random sampling.
            skip_auto_detection: Skip auto-detection of columns (requires explicit cols)
            skip_coordinate_validation: Skip coordinate range validation
            skip_zero_inflation_check: Skip zero inflation detection
            skip_thread_management: Skip PyArrow thread pool management
            
        Returns:
            BiodiversityData object
            
        Raises:
            FileNotFoundError: If parquet file doesn't exist
            ValueError: If data validation fails
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {data_path}")
        
        logger.info(f"Loading parquet file: {data_path}")
        
        # Apply spatial grid sampling if requested
        if spatial_grid_step > 1:
            logger.info(f"Applying spatial grid sampling during load: taking every {spatial_grid_step}th point")
            
            # Open parquet file for metadata inspection
            parquet_file = pq.ParquetFile(data_path)
            
            # Auto-detect coordinate columns if needed
            if coordinate_cols is None:
                # Read just first row to detect columns (minimal memory)
                first_batch = next(parquet_file.iter_batches(batch_size=1))
                # Convert to dict for column detection without pandas
                first_row_dict = {col: first_batch.column(col).to_pylist()[0] 
                                 for col in first_batch.column_names}
                coordinate_cols = self._detect_coordinate_columns_from_dict(first_row_dict, 
                                                                           first_batch.column_names)
            
            x_col, y_col = coordinate_cols[0], coordinate_cols[1]
            
            # EFFICIENT APPROACH: Read coordinates in batches to find unique values
            logger.info(f"Scanning coordinates to determine spatial grid...")
            unique_x = set()
            unique_y = set()
            
            # Read coordinates in batches (memory efficient)
            batch_size = 1000000  # Process 1M rows at a time
            for batch in parquet_file.iter_batches(columns=coordinate_cols, batch_size=batch_size):
                # Get numpy arrays directly without pandas
                x_array = batch.column(x_col).to_numpy()
                y_array = batch.column(y_col).to_numpy()
                unique_x.update(np.unique(x_array))
                unique_y.update(np.unique(y_array))
            
            # Sort and select every Nth value
            unique_x = sorted(unique_x)
            unique_y = sorted(unique_y)
            selected_x = set(unique_x[::spatial_grid_step])
            selected_y = set(unique_y[::spatial_grid_step])
            
            logger.info(f"Spatial grid: {len(unique_x)}x{len(unique_y)} -> "
                       f"{len(selected_x)}x{len(selected_y)} grid points")
            
            # EFFICIENT FILTERING: Use PyArrow expressions for filtering
            # Build filter expression
            x_filter = pc.field(x_col).isin(selected_x)
            y_filter = pc.field(y_col).isin(selected_y)
            combined_filter = pc.and_(x_filter, y_filter)
            
            # Read and filter in one operation (most efficient)
            logger.info(f"Loading filtered data using PyArrow predicate pushdown...")
            filtered_table = pq.read_table(
                data_path,
                filters=[
                    (x_col, 'in', list(selected_x)),
                    (y_col, 'in', list(selected_y))
                ]
            )
            
            # Keep as PyArrow table (no pandas conversion!)
            table = filtered_table
            
            # Get actual row count
            total_rows = parquet_file.metadata.num_rows
            logger.info(f"Loaded {table.num_rows:,} rows after spatial sampling (from {total_rows:,} total)")
        else:
            # Load full parquet file as PyArrow table
            table = pq.read_table(data_path)
            logger.info(f"Loaded {table.num_rows} rows with {len(table.column_names)} columns")
        
        # Auto-detect coordinate columns if not specified
        if coordinate_cols is None:
            if skip_auto_detection:
                raise ValueError("coordinate_cols must be provided when skip_auto_detection=True")
            # Create a dict from first row for detection without pandas
            first_row_dict = {col: table.column(col).to_pylist()[0] 
                            for col in table.column_names}
            coordinate_cols = self._detect_coordinate_columns_from_dict(first_row_dict, 
                                                                       table.column_names)
        
        # Extract coordinates
        if len(coordinate_cols) != 2:
            raise ValueError(f"Expected 2 coordinate columns, got {len(coordinate_cols)}: {coordinate_cols}")
        
        # Extract coordinates and ensure they are in [latitude, longitude] order
        # Define known longitude and latitude column names
        lon_names = ['longitude', 'lon', 'x', 'lng', 'long']
        lat_names = ['latitude', 'lat', 'y']
        
        # Determine which column is latitude and which is longitude
        col0_lower = coordinate_cols[0].lower()
        col1_lower = coordinate_cols[1].lower()
        
        if col0_lower in lon_names and col1_lower in lat_names:
            # First column is longitude, second is latitude - need to swap
            coordinates = np.column_stack([
                table.column(coordinate_cols[1]).to_numpy(),
                table.column(coordinate_cols[0]).to_numpy()
            ])
            logger.info(f"Reordered coordinates from [{coordinate_cols[0]}, {coordinate_cols[1]}] to [latitude, longitude]")
        elif col0_lower in lat_names and col1_lower in lon_names:
            # Already in correct order [latitude, longitude]
            coordinates = np.column_stack([
                table.column(coordinate_cols[0]).to_numpy(),
                table.column(coordinate_cols[1]).to_numpy()
            ])
            logger.info(f"Coordinates already in correct order: [{coordinate_cols[0]}, {coordinate_cols[1]}]")
        else:
            # Unknown column names - warn but proceed
            logger.warning(f"Could not determine coordinate order from column names: {coordinate_cols}. "
                         f"Assuming order is [latitude, longitude]")
            coordinates = np.column_stack([
                table.column(coordinate_cols[0]).to_numpy(),
                table.column(coordinate_cols[1]).to_numpy()
            ])
        
        # Validate coordinate ranges
        if not skip_coordinate_validation:
            lat_values = coordinates[:, 0]
            lon_values = coordinates[:, 1]
            
            if np.any((lat_values < -90) | (lat_values > 90)):
                logger.warning(f"Latitude values outside valid range [-90, 90]: "
                             f"min={lat_values.min():.2f}, max={lat_values.max():.2f}")
            
            if np.any((lon_values < -180) | (lon_values > 180)):
                logger.warning(f"Longitude values outside valid range [-180, 180]: "
                             f"min={lon_values.min():.2f}, max={lon_values.max():.2f}")
        
        # Auto-detect feature columns if not specified
        if feature_cols is None:
            if skip_auto_detection:
                raise ValueError("feature_cols must be provided when skip_auto_detection=True")
            # Use all numeric columns except coordinates
            numeric_cols = []
            for field in table.schema:
                if (pa.types.is_floating(field.type) or pa.types.is_integer(field.type)) \
                   and field.name not in coordinate_cols:
                    numeric_cols.append(field.name)
            feature_cols = numeric_cols
            logger.info(f"Auto-detected {len(feature_cols)} feature columns")
        
        # Extract features directly from PyArrow table
        if feature_cols:
            features = np.column_stack([table.column(col).to_numpy() for col in feature_cols])
        else:
            features = np.empty((table.num_rows, 0))  # Empty array if no features
        feature_names = feature_cols
        
        # Extract species data if specified
        species_data = None
        if species_cols:
            species_data = np.column_stack([table.column(col).to_numpy() for col in species_cols])
        
        # Check for zero inflation
        zero_inflated = False if skip_zero_inflation_check else self._check_zero_inflation(features)
        
        # Create metadata
        metadata = {
            'source_file': str(path),
            'n_rows': table.num_rows,
            'n_cols': len(table.column_names),
            'coordinate_columns': coordinate_cols,
            'feature_columns': feature_cols,
            'species_columns': species_cols,
            'missing_proportion': np.isnan(features).mean() if features.size > 0 else 0,
            'zero_proportion': (features == 0).mean() if features.size > 0 else 0
        }
        
        # Create BiodiversityData object
        biodiv_data = BiodiversityData(
            features=features,
            coordinates=coordinates,
            feature_names=feature_names,
            coordinate_system='WGS84',  # Assume WGS84
            metadata=metadata,
            zero_inflated=zero_inflated,
            species_names=species_cols  # Use species column names if provided
        )
        
        # IMPORTANT: Clean up PyArrow resources and minimize thread pools
        if not skip_thread_management:
            del table  # Delete the PyArrow table
            if 'parquet_file' in locals():
                del parquet_file  # Delete the parquet file handle if it exists
            if 'filtered_table' in locals():
                del filtered_table  # Delete filtered table if it exists
            
            # Minimize PyArrow thread pools after data loading
            # This won't release threads but will prevent them from doing work
            pa.set_cpu_count(1)  # Minimize CPU threads
            pa.set_io_thread_count(1)  # Minimize I/O threads
            
            # Force garbage collection
            gc.collect()
            logger.debug("Minimized PyArrow thread pools (CPU=1, IO=1) after data loading")
        else:
            # Still delete tables but skip thread management
            del table
            if 'parquet_file' in locals():
                del parquet_file
            if 'filtered_table' in locals():
                del filtered_table
        
        # Validate if requested
        if validate:
            self._validate_data(biodiv_data)
        
        # Cache for potential reuse
        self.last_loaded_path = str(path)
        self.last_loaded_data = biodiv_data
        
        return biodiv_data
    
    def load_filtered(self,
                     data_path: str,
                     min_value: float = 0,
                     filter_columns: Optional[List[str]] = None,
                     **kwargs) -> BiodiversityData:
        """Load parquet with filtering for valid values.
        
        Args:
            data_path: Path to parquet file
            min_value: Minimum value to consider valid
            filter_columns: Columns to apply filter to (None = all richness columns)
            **kwargs: Additional arguments for load()
            
        Returns:
            Filtered BiodiversityData
        """
        # Load data first
        biodiv_data = self.load(data_path, **kwargs)
        
        # Determine columns to filter
        if filter_columns is None:
            filter_columns = [name for name in biodiv_data.feature_names 
                            if 'richness' in name.lower()]
        
        # Create filter mask
        valid_mask = np.ones(biodiv_data.n_samples, dtype=bool)
        for col_name in filter_columns:
            if col_name in biodiv_data.feature_names:
                col_idx = biodiv_data.feature_names.index(col_name)
                col_data = biodiv_data.features[:, col_idx]
                valid_mask &= (col_data > min_value) & np.isfinite(col_data)
        
        # Apply filter
        filtered_features = biodiv_data.features[valid_mask]
        filtered_coords = biodiv_data.coordinates[valid_mask]
        
        # Create new BiodiversityData object
        filtered_data = BiodiversityData(
            features=filtered_features,
            coordinates=filtered_coords,
            feature_names=biodiv_data.feature_names,
            coordinate_names=biodiv_data.coordinate_names,
            n_samples=filtered_features.shape[0],
            n_features=biodiv_data.n_features
        )
        
        logger.info(f"Filtered {biodiv_data.n_samples} samples to {filtered_data.n_samples} "
                   f"({filtered_data.n_samples/biodiv_data.n_samples*100:.1f}% retained)")
        
        return filtered_data
    
    def load_chunked(self,
                    data_path: str,
                    chunk_size: int = 1_000_000,
                    columns: Optional[List[str]] = None):
        """Load parquet file in chunks for memory efficiency.
        
        Args:
            data_path: Path to parquet file
            chunk_size: Number of rows per chunk
            columns: Specific columns to load (None = all)
            
        Yields:
            BiodiversityData chunks
        """
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(data_path)
        
        # Get column names if not specified
        if columns is None:
            columns = parquet_file.schema.names
        
        # Determine coordinate and feature columns
        coord_cols = self._detect_coordinate_columns(pd.DataFrame(columns=[c for c in columns]))
        feature_cols = [c for c in columns if c not in coord_cols]
        
        # Process in chunks
        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
            df_chunk = batch.to_pandas()
            
            # Extract coordinates and features
            coordinates = df_chunk[coord_cols].values
            features = df_chunk[feature_cols].values if feature_cols else np.array([])
            
            # Create BiodiversityData for this chunk
            chunk_data = BiodiversityData(
                features=features,
                coordinates=coordinates,
                feature_names=feature_cols,
                coordinate_names=coord_cols,
                n_samples=len(df_chunk),
                n_features=len(feature_cols)
            )
            
            yield chunk_data
    
    def compute_statistics(self,
                          data_path: str,
                          columns: Optional[List[str]] = None,
                          chunk_size: int = 1_000_000) -> Dict[str, Dict]:
        """Compute statistics on large parquet files efficiently.
        
        Args:
            data_path: Path to parquet file
            columns: Columns to compute stats for (None = all richness columns)
            chunk_size: Rows per chunk for processing
            
        Returns:
            Dictionary of statistics per column
        """
        # Initialize accumulators
        stats = {}
        
        # Process in chunks
        for chunk_data in self.load_chunked(data_path, chunk_size=chunk_size):
            for i, col_name in enumerate(chunk_data.feature_names):
                if columns is None or col_name in columns:
                    if col_name not in stats:
                        stats[col_name] = {
                            'count': 0,
                            'valid_count': 0,
                            'sum': 0,
                            'sum_sq': 0,
                            'min': float('inf'),
                            'max': float('-inf')
                        }
                    
                    col_data = chunk_data.features[:, i]
                    valid_mask = np.isfinite(col_data) & (col_data > 0)
                    valid_data = col_data[valid_mask]
                    
                    stats[col_name]['count'] += len(col_data)
                    stats[col_name]['valid_count'] += len(valid_data)
                    
                    if len(valid_data) > 0:
                        stats[col_name]['sum'] += valid_data.sum()
                        stats[col_name]['sum_sq'] += (valid_data ** 2).sum()
                        stats[col_name]['min'] = min(stats[col_name]['min'], valid_data.min())
                        stats[col_name]['max'] = max(stats[col_name]['max'], valid_data.max())
        
        # Compute final statistics
        for col_name in stats:
            s = stats[col_name]
            if s['valid_count'] > 0:
                mean = s['sum'] / s['valid_count']
                variance = (s['sum_sq'] / s['valid_count']) - (mean ** 2)
                std = np.sqrt(variance) if variance > 0 else 0
                
                stats[col_name].update({
                    'mean': mean,
                    'std': std,
                    'coverage': s['valid_count'] / s['count'] * 100
                })
            else:
                stats[col_name].update({
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'coverage': 0
                })
        
        return stats
    
    def _detect_coordinate_columns_from_dict(self, first_row: Dict[str, Any], 
                                            column_names: List[str]) -> List[str]:
        """Auto-detect coordinate columns from first row dict.
        
        Args:
            first_row: Dictionary of column_name -> value from first row
            column_names: List of all column names
            
        Returns:
            List of coordinate column names [longitude, latitude]
        """
        # Common coordinate column names
        lon_names = ['longitude', 'lon', 'x', 'lng', 'long']
        lat_names = ['latitude', 'lat', 'y']
        
        lon_col = None
        lat_col = None
        
        # Case-insensitive search
        cols_lower = {col.lower(): col for col in column_names}
        
        # Find longitude column
        for name in lon_names:
            if name in cols_lower:
                lon_col = cols_lower[name]
                break
        
        # Find latitude column
        for name in lat_names:
            if name in cols_lower:
                lat_col = cols_lower[name]
                break
        
        if lon_col and lat_col:
            return [lon_col, lat_col]
        
        # Fallback: use first two numeric columns
        numeric_cols = []
        for col in column_names:
            val = first_row.get(col)
            if isinstance(val, (int, float)):
                numeric_cols.append(col)
                if len(numeric_cols) == 2:
                    break
        
        if len(numeric_cols) >= 2:
            logger.warning(f"Could not find standard coordinate columns. "
                         f"Using first two numeric columns: {numeric_cols[:2]}")
            return numeric_cols[:2]
        
        raise ValueError("Could not detect coordinate columns")
    
    def _detect_coordinate_columns(self, df) -> List[str]:
        """Auto-detect coordinate columns from dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of coordinate column names [longitude, latitude]
        """
        # Common coordinate column names
        lon_names = ['longitude', 'lon', 'x', 'lng', 'long']
        lat_names = ['latitude', 'lat', 'y']
        
        lon_col = None
        lat_col = None
        
        # Case-insensitive search
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        # Find longitude column
        for name in lon_names:
            if name in df_cols_lower:
                lon_col = df_cols_lower[name]
                break
        
        # Find latitude column
        for name in lat_names:
            if name in df_cols_lower:
                lat_col = df_cols_lower[name]
                break
        
        if lon_col is None or lat_col is None:
            raise ValueError(
                f"Could not auto-detect coordinate columns. "
                f"Please specify coordinate_cols parameter. "
                f"Available columns: {list(df.columns)}"
            )
        
        logger.info(f"Auto-detected coordinate columns: [{lon_col}, {lat_col}]")
        return [lon_col, lat_col]
    
    def _check_zero_inflation(self, features: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if features show zero inflation.
        
        Args:
            features: Feature array
            threshold: Proportion of zeros to consider zero-inflated
            
        Returns:
            True if zero-inflated
        """
        zero_prop = (features == 0).mean()
        return zero_prop > threshold
    
    def _validate_data(self, data: BiodiversityData) -> None:
        """Validate loaded biodiversity data.
        
        Args:
            data: BiodiversityData to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check shapes
        n_samples = data.n_samples
        if data.coordinates.shape[0] != n_samples:
            raise ValueError(
                f"Coordinate samples ({data.coordinates.shape[0]}) "
                f"don't match feature samples ({n_samples})"
            )
        
        # Check coordinate bounds
        # After reordering, coordinates are in [latitude, longitude] order
        lat_min, lat_max = data.coordinates[:, 0].min(), data.coordinates[:, 0].max()
        lon_min, lon_max = data.coordinates[:, 1].min(), data.coordinates[:, 1].max()
        
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            logger.warning(f"Longitude values outside [-180, 180]: [{lon_min}, {lon_max}]")
        
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            logger.warning(f"Latitude values outside [-90, 90]: [{lat_min}, {lat_max}]")
        
        # Check for non-finite values in features
        if not np.isfinite(data.features).any():
            raise ValueError("No finite values found in features")
        
        # Check feature names match feature count
        if len(data.feature_names) != data.n_features:
            raise ValueError(
                f"Feature names count ({len(data.feature_names)}) "
                f"doesn't match feature count ({data.n_features})"
            )
        
        logger.info("Data validation passed")