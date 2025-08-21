"""Standalone analysis dataset classes that work without database dependencies."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
import tempfile
import os

from src.base.dataset import BaseDataset, DatasetInfo
from src.abstractions.types import DataType
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class StandaloneAnalysisDataset(BaseDataset, ABC):
    """Base class for database-free analysis datasets."""
    
    def __init__(self, chunk_size: int = 10000):
        """Initialize standalone analysis dataset.
        
        Args:
            chunk_size: Number of records per chunk for memory-efficient processing
        """
        self.chunk_size = chunk_size
        self._data: Optional[pd.DataFrame] = None
        self._info: Optional[DatasetInfo] = None
    
    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """Load the full dataset. Must be implemented by subclasses."""
        pass
    
    def load_info(self) -> DatasetInfo:
        """Get dataset information without loading full data."""
        if self._info is None:
            # Load data to get info (cached after first load)
            if self._data is None:
                self._data = self._load_data()
            
            # Calculate dataset info
            record_count = len(self._data)
            size_mb = self._data.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Get spatial bounds if coordinates exist
            bounds = None
            if 'latitude' in self._data.columns and 'longitude' in self._data.columns:
                bounds = (
                    self._data['longitude'].min(),
                    self._data['latitude'].min(), 
                    self._data['longitude'].max(),
                    self._data['latitude'].max()
                )
            
            self._info = DatasetInfo(
                name=getattr(self, 'parquet_path', getattr(self, 'csv_path', 'unknown')).name,
                source=str(getattr(self, 'parquet_path', getattr(self, 'csv_path', 'unknown'))),
                format='parquet' if hasattr(self, 'parquet_path') else 'csv',
                size_mb=size_mb,
                record_count=record_count,
                bounds=bounds,
                crs='EPSG:4326',  # Assume geographic coordinates
                metadata={
                    'columns': list(self._data.columns),
                    'data_types': {col: str(dtype) for col, dtype in self._data.dtypes.items()},
                    'column_count': len(self._data.columns)
                },
                data_type=DataType.TABULAR
            )
        
        return self._info
    
    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Iterate over dataset in chunks for memory-efficient processing."""
        if self._data is None:
            self._data = self._load_data()
        
        total_rows = len(self._data)
        logger.info(f"Processing {total_rows:,} records in chunks of {self.chunk_size:,}")
        
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk = self._data.iloc[start_idx:end_idx].copy()
            
            logger.debug(f"Yielding chunk {start_idx//self.chunk_size + 1}: "
                        f"rows {start_idx}-{end_idx} ({len(chunk)} records)")
            
            yield chunk
    
    def get_full_data(self) -> pd.DataFrame:
        """Get the full dataset (use with caution for large datasets)."""
        if self._data is None:
            self._data = self._load_data()
        return self._data.copy()
    
    def get_feature_columns(self, exclude_patterns: Optional[List[str]] = None) -> List[str]:
        """Get feature columns for analysis, excluding specified patterns."""
        if self._data is None:
            self._data = self._load_data()
        
        if exclude_patterns is None:
            exclude_patterns = ['latitude', 'longitude', 'grid_id', 'year', 'id']
        
        # Get numeric columns
        numeric_cols = self._data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out excluded patterns
        feature_cols = []
        for col in numeric_cols:
            if not any(pattern.lower() in col.lower() for pattern in exclude_patterns):
                feature_cols.append(col)
        
        logger.info(f"Identified {len(feature_cols)} feature columns for analysis")
        return feature_cols
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, '_temp_file') and os.path.exists(self._temp_file):
            try:
                os.unlink(self._temp_file)
                logger.debug(f"Cleaned up temporary file: {self._temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


class ParquetAnalysisDataset(StandaloneAnalysisDataset):
    """Analysis dataset that loads from parquet files."""
    
    def __init__(self, parquet_path: Path, chunk_size: int = 10000):
        """Initialize parquet analysis dataset.
        
        Args:
            parquet_path: Path to input parquet file
            chunk_size: Number of records per chunk
        """
        super().__init__(chunk_size=chunk_size)
        self.parquet_path = Path(parquet_path)
        
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from parquet file."""
        logger.info(f"Loading data from parquet: {self.parquet_path}")
        
        try:
            data = pd.read_parquet(self.parquet_path)
            logger.info(f"Loaded {len(data):,} records with {len(data.columns)} columns")
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet file {self.parquet_path}: {e}")
    
    @property
    def data_type(self) -> DataType:
        """Return the data type for this dataset."""
        return DataType.TABULAR
    
    def read_records(self) -> Iterator[Dict[str, Any]]:
        """Read records from parquet file."""
        if self._data is None:
            self._data = self._load_data()
        
        for _, row in self._data.iterrows():
            yield row.to_dict()
    
    def read_chunks(self) -> Iterator[List[Dict[str, Any]]]:
        """Read parquet data in chunks."""
        if self._data is None:
            self._data = self._load_data()
        
        for i in range(0, len(self._data), self.chunk_size):
            chunk = self._data.iloc[i:i + self.chunk_size]
            yield [row.to_dict() for _, row in chunk.iterrows()]
    
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a single record from the parquet data."""
        # Basic validation - check for required coordinate columns
        coord_candidates = [
            ['longitude', 'latitude'],
            ['lon', 'lat'], 
            ['x', 'y'],
            ['long', 'lat'],
            ['lng', 'lat']
        ]
        
        for coord_cols in coord_candidates:
            if all(col in record for col in coord_cols):
                # Check if coordinates are numeric and within valid ranges
                try:
                    lon, lat = record[coord_cols[0]], record[coord_cols[1]]
                    if pd.isna(lon) or pd.isna(lat):
                        return False, f"Missing coordinate values: {coord_cols}"
                    
                    lon, lat = float(lon), float(lat)
                    if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                        return False, f"Invalid coordinate ranges: lon={lon}, lat={lat}"
                    
                    return True, None
                except (ValueError, TypeError) as e:
                    return False, f"Invalid coordinate types: {e}"
        
        return False, "No valid coordinate columns found"


class CSVAnalysisDataset(StandaloneAnalysisDataset):
    """Analysis dataset that loads from CSV files with automatic parquet conversion."""
    
    def __init__(self, csv_path: Path, chunk_size: int = 10000):
        """Initialize CSV analysis dataset.
        
        Args:
            csv_path: Path to input CSV file
            chunk_size: Number of records per chunk
        """
        super().__init__(chunk_size=chunk_size)
        self.csv_path = Path(csv_path)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from CSV: {self.csv_path}")
        
        try:
            # Load CSV
            data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(data):,} records with {len(data.columns)} columns from CSV")
            
            # Convert to parquet temporarily for better performance
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                self._temp_file = tmp.name
                data.to_parquet(tmp.name, index=False)
                logger.debug(f"Created temporary parquet file: {tmp.name}")
            
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file {self.csv_path}: {e}")


class MultiFormatAnalysisDataset(StandaloneAnalysisDataset):
    """Analysis dataset that can load from multiple formats automatically."""
    
    def __init__(self, input_path: Path, chunk_size: int = 10000):
        """Initialize multi-format analysis dataset.
        
        Args:
            input_path: Path to input file (parquet, csv, or other supported format)
            chunk_size: Number of records per chunk
        """
        super().__init__(chunk_size=chunk_size)
        self.input_path = Path(input_path)
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Determine dataset type
        self._dataset_type = self._detect_format()
        logger.info(f"Detected input format: {self._dataset_type}")
    
    def _detect_format(self) -> str:
        """Detect input file format."""
        suffix = self.input_path.suffix.lower()
        
        if suffix == '.parquet':
            return 'parquet'
        elif suffix in ['.csv', '.txt']:
            return 'csv'
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load data using appropriate format handler."""
        if self._dataset_type == 'parquet':
            logger.info(f"Loading parquet file: {self.input_path}")
            return pd.read_parquet(self.input_path)
        
        elif self._dataset_type == 'csv':
            logger.info(f"Loading CSV file: {self.input_path}")
            data = pd.read_csv(self.input_path)
            
            # Create temporary parquet for performance
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                self._temp_file = tmp.name
                data.to_parquet(tmp.name, index=False)
                logger.debug(f"Created temporary parquet file: {tmp.name}")
            
            return data
        
        else:
            raise ValueError(f"Unsupported dataset type: {self._dataset_type}")


def create_analysis_dataset(
    input_path: Path, 
    data_source: str = 'auto',
    chunk_size: int = 10000
) -> StandaloneAnalysisDataset:
    """Factory function to create appropriate analysis dataset.
    
    Args:
        input_path: Path to input data file
        data_source: Data source type ('parquet', 'csv', 'auto')
        chunk_size: Number of records per chunk
        
    Returns:
        Appropriate analysis dataset instance
    """
    input_path = Path(input_path)
    
    if data_source == 'auto':
        return MultiFormatAnalysisDataset(input_path, chunk_size=chunk_size)
    elif data_source == 'parquet':
        return ParquetAnalysisDataset(input_path, chunk_size=chunk_size)
    elif data_source == 'csv':
        return CSVAnalysisDataset(input_path, chunk_size=chunk_size)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")