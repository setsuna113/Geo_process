from src.abstractions import get_logger
"""Memory optimization utilities for ML pipelines.

Provides memory-efficient data handling, monitoring, and processing utilities.
"""

import numpy as np
import pandas as pd
import logging
import gc
import psutil
import os
from typing import Any, Dict, List, Generator, Optional, Callable
from functools import wraps

logger = get_logger(__name__)


class MemoryMonitor:
    """Monitor and report memory usage during processing."""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        vm = psutil.virtual_memory()
        
        return {
            'rss_gb': mem_info.rss / 1e9,
            'vms_gb': mem_info.vms / 1e9,
            'available_gb': vm.available / 1e9,
            'percent_used': vm.percent,
            'process_percent': process.memory_percent()
        }
    
    @staticmethod
    def log_memory(prefix: str = "", level: int = logging.INFO):
        """Log current memory usage.
        
        Args:
            prefix: Prefix for log message
            level: Logging level
        """
        mem = MemoryMonitor.get_memory_info()
        message = f"{prefix} Memory: RSS={mem['rss_gb']:.2f}GB, " \
                 f"Available={mem['available_gb']:.2f}GB, " \
                 f"Process={mem['process_percent']:.1f}%"
        logger.log(level, message)
    
    @staticmethod
    def memory_tracker(func: Callable) -> Callable:
        """Decorator to track memory usage of a function.
        
        Args:
            func: Function to track
            
        Returns:
            Wrapped function with memory tracking
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Before
            mem_before = MemoryMonitor.get_memory_info()
            logger.debug(f"Before {func.__name__}: RSS={mem_before['rss_gb']:.2f}GB")
            
            # Execute
            result = func(*args, **kwargs)
            
            # After
            mem_after = MemoryMonitor.get_memory_info()
            mem_delta = mem_after['rss_gb'] - mem_before['rss_gb']
            
            logger.debug(f"After {func.__name__}: RSS={mem_after['rss_gb']:.2f}GB "
                        f"(Δ={mem_delta:+.2f}GB)")
            
            return result
        
        return wrapper


class DataChunker:
    """Process large datasets in chunks to minimize memory usage."""
    
    def __init__(self, data: pd.DataFrame, chunk_size: int = 10000):
        """Initialize data chunker.
        
        Args:
            data: DataFrame to chunk
            chunk_size: Size of each chunk
        """
        self.data = data
        self.chunk_size = chunk_size
        self.n_chunks = (len(data) + chunk_size - 1) // chunk_size
    
    def iter_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """Iterate over data chunks lazily.
        
        Yields:
            DataFrame chunks
        """
        for i in range(self.n_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, len(self.data))
            
            logger.debug(f"Yielding chunk {i+1}/{self.n_chunks} (rows {start}:{end})")
            yield self.data.iloc[start:end]
    
    def process_chunks(self, 
                      func: Callable,
                      combine_func: Optional[Callable] = None,
                      **kwargs) -> Any:
        """Process data in chunks and combine results.
        
        Args:
            func: Function to apply to each chunk
            combine_func: Function to combine results (default: list)
            **kwargs: Additional arguments for func
            
        Returns:
            Combined results
        """
        results = []
        
        for i, chunk in enumerate(self.iter_chunks()):
            logger.info(f"Processing chunk {i+1}/{self.n_chunks}")
            
            # Process chunk
            result = func(chunk, **kwargs)
            results.append(result)
            
            # Explicit cleanup
            del chunk
            gc.collect()
        
        # Combine results
        if combine_func:
            return combine_func(results)
        
        return results


class IndexBasedResampler:
    """Memory-efficient resampling using indices instead of data copies."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize resampler.
        
        Args:
            data: DataFrame to resample from
        """
        self.data = data
        self.n = len(data)
        self.indices = np.arange(self.n)
    
    def bootstrap_indices(self, 
                         n_samples: Optional[int] = None,
                         random_state: Optional[int] = None) -> np.ndarray:
        """Generate bootstrap indices.
        
        Args:
            n_samples: Number of samples (default: same as data)
            random_state: Random seed
            
        Returns:
            Array of bootstrap indices
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = n_samples or self.n
        return np.random.choice(self.n, n_samples, replace=True)
    
    def get_bootstrap_view(self, 
                           indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Get bootstrap sample as view (not copy).
        
        Args:
            indices: Bootstrap indices (generates if not provided)
            
        Returns:
            View of data at bootstrap indices
        """
        if indices is None:
            indices = self.bootstrap_indices()
        
        return self.data.iloc[indices]
    
    def bootstrap_generator(self, 
                           n_bootstrap: int = 100,
                           n_samples: Optional[int] = None) -> Generator:
        """Generate bootstrap samples lazily.
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            n_samples: Size of each bootstrap sample
            
        Yields:
            Bootstrap data views
        """
        for i in range(n_bootstrap):
            indices = self.bootstrap_indices(n_samples, random_state=42+i)
            yield self.data.iloc[indices]


class LazyDataLoader:
    """Lazy loading for large datasets."""
    
    def __init__(self, file_path: str, chunk_size: int = 10000):
        """Initialize lazy loader.
        
        Args:
            file_path: Path to data file
            chunk_size: Size of chunks to load
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self._metadata = None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata without loading full data.
        
        Returns:
            Dictionary with metadata
        """
        if self._metadata is None:
            # Detect file type
            if self.file_path.endswith('.parquet'):
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(self.file_path)
                self._metadata = {
                    'num_rows': pf.metadata.num_rows,
                    'num_columns': len(pf.schema),
                    'columns': pf.schema.names,
                    'file_size_mb': os.path.getsize(self.file_path) / 1e6
                }
            elif self.file_path.endswith('.csv'):
                # Read just header
                df_sample = pd.read_csv(self.file_path, nrows=1)
                self._metadata = {
                    'columns': list(df_sample.columns),
                    'file_size_mb': os.path.getsize(self.file_path) / 1e6
                }
        
        return self._metadata
    
    def iter_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """Iterate over file chunks lazily.
        
        Yields:
            DataFrame chunks
        """
        if self.file_path.endswith('.parquet'):
            # Use pandas chunking for parquet
            pf = pd.read_parquet(self.file_path, chunksize=self.chunk_size)
            for chunk in pf:
                yield chunk
        elif self.file_path.endswith('.csv'):
            # Use pandas chunking for CSV
            for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
                yield chunk
        else:
            # Fall back to loading full file
            df = pd.read_pickle(self.file_path) if self.file_path.endswith('.pkl') else pd.read_csv(self.file_path)
            chunker = DataChunker(df, self.chunk_size)
            for chunk in chunker.iter_chunks():
                yield chunk
    
    def process_lazy(self, 
                    func: Callable,
                    combine_func: Optional[Callable] = None,
                    **kwargs) -> Any:
        """Process file in chunks without loading full dataset.
        
        Args:
            func: Function to apply to each chunk
            combine_func: Function to combine results
            **kwargs: Additional arguments for func
            
        Returns:
            Combined results
        """
        results = []
        
        for i, chunk in enumerate(self.iter_chunks()):
            logger.info(f"Processing chunk {i+1} from {self.file_path}")
            result = func(chunk, **kwargs)
            results.append(result)
            
            # Cleanup
            del chunk
            gc.collect()
        
        if combine_func:
            return combine_func(results)
        
        return results


def memory_efficient_concat(dfs: List[pd.DataFrame], 
                           clear_inputs: bool = True) -> pd.DataFrame:
    """Concatenate DataFrames with minimal memory usage.
    
    Args:
        dfs: List of DataFrames to concatenate
        clear_inputs: Whether to clear input DataFrames after concat
        
    Returns:
        Concatenated DataFrame
    """
    # Use concat with copy=False to avoid unnecessary copying
    result = pd.concat(dfs, ignore_index=True, copy=False)
    
    if clear_inputs:
        # Clear original DataFrames to free memory
        for df in dfs:
            df.drop(df.index, inplace=True)
        gc.collect()
    
    return result


def optimize_dtypes(df: pd.DataFrame, 
                   deep: bool = True) -> pd.DataFrame:
    """Optimize DataFrame dtypes to reduce memory usage.
    
    Args:
        df: DataFrame to optimize
        deep: Whether to downcast numeric types
        
    Returns:
        Optimized DataFrame
    """
    # Original memory usage
    mem_before = df.memory_usage(deep=True).sum() / 1e6
    
    # Optimize each column
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            if deep:
                # Downcast numeric types
                if 'int' in str(col_type):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif 'float' in str(col_type):
                    df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            # Convert objects to category if low cardinality
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
    
    # New memory usage
    mem_after = df.memory_usage(deep=True).sum() / 1e6
    reduction_pct = (mem_before - mem_after) / mem_before * 100
    
    logger.info(f"Memory optimization: {mem_before:.1f}MB → {mem_after:.1f}MB "
               f"({reduction_pct:.1f}% reduction)")
    
    return df


# Decorator for automatic memory tracking
def track_memory(func: Callable) -> Callable:
    """Decorator to automatically track memory usage of a function."""
    return MemoryMonitor.memory_tracker(func)


# Context manager for memory monitoring
class MemoryContext:
    """Context manager for monitoring memory usage."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.mem_before = None
    
    def __enter__(self):
        self.mem_before = MemoryMonitor.get_memory_info()
        logger.info(f"{self.name} started - Memory: {self.mem_before['rss_gb']:.2f}GB")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        mem_after = MemoryMonitor.get_memory_info()
        delta = mem_after['rss_gb'] - self.mem_before['rss_gb']
        
        logger.info(f"{self.name} completed - Memory: {mem_after['rss_gb']:.2f}GB "
                   f"(Δ={delta:+.2f}GB)")
        
        # Force garbage collection if memory increased significantly
        if delta > 0.5:  # More than 500MB increase
            logger.info(f"Running garbage collection after {self.name}")
            gc.collect()