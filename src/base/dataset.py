# src/base/dataset.py
"""Base dataset class for data loading and management."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass
import logging
import json

from ..core.registry import component_registry
from ..config import config

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Dataset metadata."""
    name: str
    source: str
    format: str
    size_mb: float
    record_count: int
    bounds: Optional[Tuple[float, float, float, float]]
    crs: Optional[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'source': self.source,
            'format': self.format,
            'size_mb': self.size_mb,
            'record_count': self.record_count,
            'bounds': self.bounds,
            'crs': self.crs,
            'metadata': self.metadata
        }


class BaseDataset(ABC):
    """
    Base class for dataset loaders.
    
    Handles:
    - Data loading
    - Format validation
    - Chunked reading for large files
    - Metadata extraction
    """
    
    def __init__(self,
                 source: Union[str, Path],
                 chunk_size: int = 10000,
                 validate: bool = True,
                 **kwargs):
        """
        Initialize dataset.
        
        Args:
            source: Data source (file path, URL, database connection)
            chunk_size: Size of chunks for reading
            validate: Whether to validate data on load
            **kwargs: Dataset-specific parameters
        """
        self.source = Path(source) if isinstance(source, str) else source
        self.chunk_size = chunk_size
        self.validate = validate
        self.config = self._merge_config(kwargs)
        
        # Dataset info (populated on load)
        self._info: Optional[DatasetInfo] = None
        
    def _merge_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge kwargs with default config."""
        dataset_type = self.__class__.__name__.lower().replace('dataset', '')
        default_config = config.get(f'datasets.{dataset_type}', {})
        return {**default_config, **kwargs}
    
    @abstractmethod
    def load_info(self) -> DatasetInfo:
        """
        Load dataset metadata without reading all data.
        
        Returns:
            DatasetInfo object
        """
        pass
    
    @abstractmethod
    def read_records(self) -> Iterator[Dict[str, Any]]:
        """
        Read records from dataset.
        
        Yields:
            Individual records as dictionaries
        """
        pass
    
    @abstractmethod
    def read_chunks(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Read dataset in chunks.
        
        Yields:
            Chunks of records
        """
        pass
    
    @abstractmethod
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a single record.
        
        Args:
            record: Record to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    def get_info(self) -> DatasetInfo:
        """Get dataset info (load if needed)."""
        if self._info is None:
            self._info = self.load_info()
        return self._info
    
    def filter_records(self, 
                      filter_func: Callable[[Dict[str, Any]], bool],
                      max_records: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Filter records based on a function.
        
        Args:
            filter_func: Function that returns True for records to keep
            max_records: Maximum records to return
            
        Yields:
            Filtered records
        """
        count = 0
        
        for record in self.read_records():
            if filter_func(record):
                yield record
                count += 1
                
                if max_records and count >= max_records:
                    break
    
    def sample_records(self, n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Sample n records from dataset.
        
        Args:
            n: Number of records to sample
            seed: Random seed
            
        Returns:
            List of sampled records
        """
        import random
        
        if seed:
            random.seed(seed)
            
        # For large datasets, use reservoir sampling
        reservoir = []
        
        for i, record in enumerate(self.read_records()):
            if i < n:
                reservoir.append(record)
            else:
                j = random.randint(0, i)
                if j < n:
                    reservoir[j] = record
                    
        return reservoir
    
    def export_subset(self,
                     output_path: Path,
                     filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
                     max_records: Optional[int] = None):
        """
        Export subset of dataset.
        
        Args:
            output_path: Output file path
            filter_func: Optional filter function
            max_records: Maximum records to export
        """
        records = []
        
        if filter_func:
            source = self.filter_records(filter_func, max_records)
        else:
            source = self.read_records()
            
        for i, record in enumerate(source):
            records.append(record)
            
            if max_records and i >= max_records - 1:
                break
                
        # Save based on extension
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {output_path.suffix}")
            
        logger.info(f"Exported {len(records)} records to {output_path}")
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        stats = {
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'field_stats': {}
        }
        
        for record in self.read_records():
            stats['total_records'] += 1
            
            if self.validate:
                is_valid, _ = self.validate_record(record)
                if is_valid:
                    stats['valid_records'] += 1
                else:
                    stats['invalid_records'] += 1
                    
        return stats