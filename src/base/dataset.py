# src/base/dataset.py
"""Base dataset class for data loading and management."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
import json
import numpy as np

from ..core.registry import component_registry
from ..config import config
from .raster_source import RasterTile

logger = logging.getLogger(__name__)

class DataType(Enum):
    """Supported data types for datasets."""
    RASTER = "raster"
    VECTOR = "vector"
    TABULAR = "tabular"
    POINT_CLOUD = "point_cloud"
    TIME_SERIES = "time_series"
    MIXED = "mixed"

@dataclass
class DatasetInfo:
    """Enhanced dataset metadata."""
    name: str
    source: str
    format: str
    size_mb: float
    record_count: int
    bounds: Optional[Tuple[float, float, float, float]]
    crs: Optional[str]
    metadata: Dict[str, Any]
    data_type: DataType
    resolution: Optional[Tuple[float, float]] = None  # For raster data
    tile_count: Optional[int] = None  # For tiled data
    
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
            'metadata': self.metadata,
            'data_type': self.data_type.value,
            'resolution': self.resolution,
            'tile_count': self.tile_count
        }


class BaseDataset(ABC):
    """
    Enhanced base class for dataset loaders.
    
    New Features:
    - data_type property for dataset classification
    - get_resolution() method for spatial datasets
    - tile iterator support for large raster datasets
    - estimate_size() method for memory planning
    - Enhanced metadata with resolution and tile information
    
    Handles:
    - Data loading
    - Format validation
    - Chunked reading for large files
    - Metadata extraction
    - Tile-based access for raster data
    """
    
    def __init__(self,
                 source: Union[str, Path],
                 chunk_size: int = 10000,
                 validate: bool = True,
                 tile_size: Optional[int] = None,
                 lazy_loading: bool = True,
                 **kwargs):
        """
        Initialize enhanced dataset.
        
        Args:
            source: Data source (file path, URL, database connection)
            chunk_size: Size of chunks for reading
            validate: Whether to validate data on load
            tile_size: Size of tiles for raster access (None for auto)
            lazy_loading: Whether to defer loading until access
            **kwargs: Dataset-specific parameters
        """
        self.source = Path(source) if isinstance(source, str) else source
        self.chunk_size = chunk_size
        self.validate = validate
        self.tile_size = tile_size or config.get('datasets.default_tile_size', 512)
        self.lazy_loading = lazy_loading
        self.config = self._merge_config(kwargs)
        
        # Dataset info (populated on load)
        self._info: Optional[DatasetInfo] = None
        
        # Cached resolution for spatial datasets
        self._resolution: Optional[Tuple[float, float]] = None
        
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
        stats: Dict[str, Any] = {
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
    
    # Enhanced methods for base class enhancements
    
    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """
        Get the data type of this dataset.
        
        Returns:
            DataType enum value
        """
        pass
    
    def get_resolution(self) -> Optional[Tuple[float, float]]:
        """
        Get spatial resolution for datasets that support it.
        
        Returns:
            Tuple of (x_resolution, y_resolution) or None for non-spatial data
        """
        if self._resolution is not None:
            return self._resolution
            
        # Try to extract from info
        info = self.get_info()
        if hasattr(info, 'resolution') and info.resolution:
            self._resolution = info.resolution
            return self._resolution
            
        # For raster data, try to calculate from bounds and dimensions
        if self.data_type == DataType.RASTER:
            self._resolution = self._calculate_resolution()
            return self._resolution
            
        return None
        
    def _calculate_resolution(self) -> Optional[Tuple[float, float]]:
        """
        Calculate resolution for raster datasets.
        Override in subclasses with dataset-specific logic.
        """
        return None
    
    def iter_tiles(self, bounds: Optional[Tuple[float, float, float, float]] = None) -> Iterator[RasterTile]:
        """
        Iterate over tiles for raster datasets.
        
        Args:
            bounds: Optional bounding box to limit tiles
            
        Yields:
            RasterTile objects
        """
        if self.data_type != DataType.RASTER:
            raise ValueError(f"Tile iteration not supported for {self.data_type.value} datasets")
            
        # Default implementation - override in raster subclasses
        info = self.get_info()
        if not info.bounds:
            return
            
        minx, miny, maxx, maxy = bounds or info.bounds
        
        # Calculate tile grid
        x_tiles = int((maxx - minx) / self.tile_size) + 1
        y_tiles = int((maxy - miny) / self.tile_size) + 1
        
        for y in range(y_tiles):
            for x in range(x_tiles):
                tile_minx = minx + x * self.tile_size
                tile_maxx = min(tile_minx + self.tile_size, maxx)
                tile_miny = miny + y * self.tile_size
                tile_maxy = min(tile_miny + self.tile_size, maxy)
                
                # Create placeholder tile - subclasses should override this method
                placeholder_data = np.empty((self.tile_size, self.tile_size), dtype=np.float32)
                
                yield RasterTile(
                    data=placeholder_data,
                    bounds=(tile_minx, tile_miny, tile_maxx, tile_maxy),
                    tile_id=f"{x}_{y}",
                    crs=info.crs or "EPSG:4326"
                )
    
    def estimate_size(self, 
                     operation: str = "load",
                     target_dtype: Optional[str] = None) -> Dict[str, float]:
        """
        Estimate memory usage for various operations.
        
        Args:
            operation: Type of operation ('load', 'process', 'tile')
            target_dtype: Target data type for memory calculation
            
        Returns:
            Dictionary with size estimates in MB
        """
        info = self.get_info()
        base_size_mb = info.size_mb
        
        estimates = {
            'source_size_mb': base_size_mb,
            'estimated_memory_mb': base_size_mb
        }
        
        # Operation-specific multipliers
        if operation == "load":
            # Just loading the data
            estimates['estimated_memory_mb'] = base_size_mb * 1.2  # 20% overhead
            
        elif operation == "process":
            # Processing usually requires 2-3x memory
            estimates['estimated_memory_mb'] = base_size_mb * 2.5
            
        elif operation == "tile":
            # Tiled access is more memory efficient
            tile_memory = (self.tile_size * self.tile_size * 8) / (1024 * 1024)  # 8 bytes per pixel
            estimates['estimated_memory_mb'] = tile_memory * 1.5  # Some overhead
            estimates['per_tile_mb'] = tile_memory
            
        # Data type conversion overhead
        if target_dtype:
            if target_dtype in ['float64', 'complex128']:
                estimates['estimated_memory_mb'] *= 2
            elif target_dtype in ['float32', 'int32']:
                estimates['estimated_memory_mb'] *= 1.5
                
        # Add parallelization overhead if configured
        max_workers = config.get('datasets.max_workers', 1)
        if max_workers > 1:
            estimates['estimated_memory_mb'] *= min(max_workers, 4) * 0.3  # Worker overhead
            
        return estimates