"""Data source implementations for analysis stage."""

from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple, Optional
import logging
import pandas as pd
import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from src.base.dataset import BaseDataset
from src.abstractions.types import DataType, DatasetInfo
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class ParquetAnalysisDataset(BaseDataset):
    """Dataset for streaming merged parquet files for analysis."""
    
    def __init__(self, parquet_path: Path, **kwargs):
        """
        Initialize parquet dataset.
        
        Args:
            parquet_path: Path to parquet file
            **kwargs: Additional parameters (chunk_size, etc.)
        """
        super().__init__(parquet_path, **kwargs)
        
        if pq is None:
            raise ImportError("pyarrow is required for parquet support. Install with: pip install pyarrow")
        
        self.parquet_file = None
        self._metadata = None
    
    def load_info(self) -> DatasetInfo:
        """Load parquet metadata without reading data."""
        try:
            pf = pq.ParquetFile(self.source)
            metadata = pf.metadata
            schema = pf.schema
            
            # Get column names
            columns = [field.name for field in schema]
            
            # Extract bounds if x,y columns exist
            bounds = None
            if 'x' in columns and 'y' in columns:
                # Read just x,y columns to get bounds
                xy_df = pf.read(['x', 'y']).to_pandas()
                bounds = (
                    xy_df['x'].min(),
                    xy_df['y'].min(),
                    xy_df['x'].max(),
                    xy_df['y'].max()
                )
            
            info = DatasetInfo(
                name=self.source.stem,
                source=str(self.source),
                format='parquet',
                size_mb=self.source.stat().st_size / (1024**2),
                record_count=metadata.num_rows,
                bounds=bounds,
                crs=None,  # Could be stored in metadata
                metadata={
                    'num_columns': len(columns),
                    'columns': columns,
                    'row_groups': metadata.num_row_groups,
                    'created_by': metadata.created_by,
                    'format_version': metadata.format_version
                },
                data_type=DataType.TABULAR
            )
            
            self._info = info
            return info
            
        except Exception as e:
            logger.error(f"Failed to load parquet info: {e}")
            raise
    
    def read_chunks(self) -> Iterator[pd.DataFrame]:
        """Read parquet in chunks."""
        pf = pq.ParquetFile(self.source)
        
        # Use iter_batches for memory-efficient reading
        for batch in pf.iter_batches(batch_size=self.chunk_size):
            yield batch.to_pandas()
    
    def read_records(self) -> Iterator[Dict[str, Any]]:
        """Read individual records."""
        for chunk in self.read_chunks():
            for _, row in chunk.iterrows():
                yield row.to_dict()
    
    def get_features_and_coords(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over features and coordinates for analysis.
        
        Yields:
            Tuple of (features, coordinates) as numpy arrays
        """
        for chunk in self.read_chunks():
            # Separate coordinates from features
            coord_cols = ['x', 'y']
            feature_cols = [col for col in chunk.columns 
                           if col not in ['cell_id', 'x', 'y']]
            
            if not feature_cols:
                logger.warning("No feature columns found in parquet")
                continue
            
            # Extract arrays
            coords = chunk[coord_cols].values
            features = chunk[feature_cols].values
            
            yield features, coords
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        if self._info is None:
            self.load_info()
        
        columns = self._info.metadata.get('columns', [])
        return [col for col in columns if col not in ['cell_id', 'x', 'y']]


class DatabaseAnalysisDataset(BaseDataset):
    """Dataset for streaming from database for analysis."""
    
    def __init__(self, db_manager: DatabaseManager, experiment_id: str, **kwargs):
        """
        Initialize database dataset.
        
        Args:
            db_manager: Database connection manager
            experiment_id: Experiment ID to load data for
            **kwargs: Additional parameters (chunk_size, table_name, etc.)
        """
        # Use experiment_id as source
        super().__init__(source=f"db://{experiment_id}", **kwargs)
        
        self.db = db_manager
        self.experiment_id = experiment_id
        # Validate table name to prevent SQL injection
        self.table_name = self._validate_table_name(kwargs.get('table_name', 'merged_features'))
        self._cursor_name = f"analysis_cursor_{experiment_id[:8]}"
    
    def _validate_table_name(self, table_name: str) -> str:
        """Validate table name to prevent SQL injection."""
        import re
        # Allow only alphanumeric, underscore, and schema.table format
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?$', table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        return table_name
    
    def load_info(self) -> DatasetInfo:
        """Load dataset info from database."""
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get row count
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM {self.table_name}
                        WHERE experiment_id = %s
                    """, [self.experiment_id])
                    record_count = cursor.fetchone()[0]
                    
                    # Get columns
                    cursor.execute(f"""
                        SELECT column_name 
                        FROM information_schema.columns
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                    """, [self.table_name])
                    columns = [row[0] for row in cursor.fetchall()]
                    
                    # Get bounds if spatial columns exist
                    bounds = None
                    if 'x' in columns and 'y' in columns:
                        cursor.execute(f"""
                            SELECT MIN(x), MIN(y), MAX(x), MAX(y)
                            FROM {self.table_name}
                            WHERE experiment_id = %s
                        """, [self.experiment_id])
                        bounds = cursor.fetchone()
                    
                    # Estimate size (rough)
                    cursor.execute(f"""
                        SELECT pg_relation_size(%s) / 1024.0 / 1024.0
                    """, [self.table_name])
                    size_mb = cursor.fetchone()[0] or 0
                    
                    info = DatasetInfo(
                        name=f"experiment_{self.experiment_id}",
                        source=self.source,
                        format='database',
                        size_mb=float(size_mb),
                        record_count=record_count,
                        bounds=bounds,
                        crs='EPSG:4326',  # Assuming standard CRS
                        metadata={
                            'table_name': self.table_name,
                            'columns': columns,
                            'experiment_id': self.experiment_id
                        },
                        data_type=DataType.TABULAR
                    )
                    
                    self._info = info
                    return info
                    
        except Exception as e:
            logger.error(f"Failed to load database info: {e}")
            raise
    
    def read_chunks(self) -> Iterator[pd.DataFrame]:
        """Read from database in chunks using server-side cursor."""
        query = f"""
            SELECT * 
            FROM {self.table_name}
            WHERE experiment_id = %s
            ORDER BY cell_id
        """
        
        with self.db.get_connection() as conn:
            # Use server-side cursor for streaming
            with conn.cursor(name=self._cursor_name) as cursor:
                cursor.itersize = self.chunk_size
                cursor.execute(query, [self.experiment_id])
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Fetch in chunks
                while True:
                    rows = cursor.fetchmany(self.chunk_size)
                    if not rows:
                        break
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rows, columns=columns)
                    yield df
    
    def read_records(self) -> Iterator[Dict[str, Any]]:
        """Read individual records from database."""
        for chunk in self.read_chunks():
            for _, row in chunk.iterrows():
                yield row.to_dict()
    
    def get_features_and_coords(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over features and coordinates for analysis.
        
        Yields:
            Tuple of (features, coordinates) as numpy arrays
        """
        for chunk in self.read_chunks():
            # Separate coordinates from features
            coord_cols = ['x', 'y']
            feature_cols = [col for col in chunk.columns 
                           if col not in ['cell_id', 'x', 'y', 'experiment_id']]
            
            if not feature_cols:
                logger.warning("No feature columns found in database")
                continue
            
            # Extract arrays
            coords = chunk[coord_cols].values
            features = chunk[feature_cols].values
            
            yield features, coords
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        if self._info is None:
            self.load_info()
        
        columns = self._info.metadata.get('columns', [])
        return [col for col in columns 
                if col not in ['cell_id', 'x', 'y', 'experiment_id']]