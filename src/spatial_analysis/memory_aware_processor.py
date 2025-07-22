"""Memory-aware processing utilities for spatial analysis."""

import numpy as np
import logging
from typing import Dict, Any, Optional, Callable, Generator, Tuple
import psutil
import gc

logger = logging.getLogger(__name__)


class SubsamplingStrategy:
    """Intelligent subsampling for large datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_samples = config.get('max_samples', 500000)
        self.strategy = config.get('strategy', 'stratified')
        self.random_seed = config.get('random_seed', 42)
        self.spatial_block_size = config.get('spatial_block_size', 100)
        
    def subsample_data(self, 
                      data: np.ndarray, 
                      coordinates: Optional[np.ndarray] = None,
                      labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample data intelligently based on strategy.
        
        Args:
            data: Input data array (n_samples, n_features)
            coordinates: Spatial coordinates (n_samples, 2) for spatial sampling
            labels: Class labels for stratified sampling
            
        Returns:
            Tuple of (subsampled_data, sample_indices)
        """
        n_samples = data.shape[0]
        
        # Check if subsampling is needed
        if n_samples <= self.max_samples:
            return data, np.arange(n_samples)
        
        logger.info(f"Subsampling {n_samples:,} samples to {self.max_samples:,}")
        
        if self.strategy == 'random':
            return self._random_subsample(data, n_samples)
        elif self.strategy == 'stratified':
            return self._stratified_subsample(data, coordinates, n_samples)
        elif self.strategy == 'grid':
            return self._grid_subsample(data, coordinates, n_samples)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
    
    def _random_subsample(self, data: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Random subsampling."""
        np.random.seed(self.random_seed)
        indices = np.random.choice(n_samples, self.max_samples, replace=False)
        return data[indices], indices
    
    def _stratified_subsample(self, data: np.ndarray, coordinates: Optional[np.ndarray], 
                             n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Spatially stratified subsampling."""
        if coordinates is None:
            # Fall back to random if no coordinates
            return self._random_subsample(data, n_samples)
        
        # Create spatial grid for stratification
        x_min, y_min = coordinates.min(axis=0)
        x_max, y_max = coordinates.max(axis=0)
        
        n_blocks_x = int((x_max - x_min) / self.spatial_block_size) + 1
        n_blocks_y = int((y_max - y_min) / self.spatial_block_size) + 1
        
        # Check for degenerate cases
        if n_blocks_x * n_blocks_y == 1:
            logger.warning("Only one spatial block created, falling back to random sampling")
            return self._random_subsample(data, n_samples)
        
        # Assign points to blocks
        block_x = ((coordinates[:, 0] - x_min) / self.spatial_block_size).astype(int)
        block_y = ((coordinates[:, 1] - y_min) / self.spatial_block_size).astype(int)
        block_ids = block_y * n_blocks_x + block_x
        
        # Sample from each block proportionally
        unique_blocks, block_counts = np.unique(block_ids, return_counts=True)
        samples_per_block = (block_counts * self.max_samples / n_samples).astype(int)
        samples_per_block[samples_per_block == 0] = 1  # At least 1 sample per block
        
        # Adjust to match target exactly
        while samples_per_block.sum() > self.max_samples:
            largest_block = np.argmax(samples_per_block)
            samples_per_block[largest_block] -= 1
        
        # Collect samples
        selected_indices = []
        np.random.seed(self.random_seed)
        
        for block_id, n_block_samples in zip(unique_blocks, samples_per_block):
            block_mask = block_ids == block_id
            block_indices = np.where(block_mask)[0]
            
            if len(block_indices) > n_block_samples:
                selected = np.random.choice(block_indices, n_block_samples, replace=False)
            else:
                selected = block_indices
                
            selected_indices.extend(selected)
        
        indices = np.array(selected_indices)
        return data[indices], indices
    
    def _grid_subsample(self, data: np.ndarray, coordinates: Optional[np.ndarray], 
                       n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Regular grid subsampling."""
        if coordinates is None:
            return self._random_subsample(data, n_samples)
        
        # Calculate grid spacing
        skip = int(np.sqrt(n_samples / self.max_samples))
        if skip < 1:
            skip = 1
        
        # Create regular grid indices
        indices = []
        for i in range(0, n_samples, skip):
            indices.append(i)
            if len(indices) >= self.max_samples:
                break
        
        indices = np.array(indices)
        return data[indices], indices


class MemoryAwareProcessor:
    """Handle memory-efficient processing of large spatial datasets."""
    
    def __init__(self, memory_limit_gb: Optional[float] = None):
        """
        Initialize processor.
        
        Args:
            memory_limit_gb: Maximum memory to use (defaults to 50% of available)
        """
        if memory_limit_gb is None:
            available_gb = psutil.virtual_memory().available / (1024**3)
            self.memory_limit_gb = available_gb * 0.5
        else:
            self.memory_limit_gb = memory_limit_gb
    
    def process_in_chunks(self, 
                         data_source: Any,
                         processor_func: Callable,
                         chunk_size: Optional[int] = None,
                         overlap: int = 0,
                         progress_callback: Optional[Callable] = None) -> Generator:
        """
        Process data in memory-efficient chunks.
        
        Args:
            data_source: Data source (array, file path, etc.)
            processor_func: Function to process each chunk
            chunk_size: Size of each chunk (auto-calculated if None)
            overlap: Overlap between chunks (for spatial continuity)
            progress_callback: Function to report progress
            
        Yields:
            Processed chunk results
        """
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = self._calculate_optimal_chunk_size(data_source)
        
        logger.info(f"Processing in chunks of {chunk_size:,} with {overlap} overlap")
        
        n_processed = 0
        total_size = self._get_data_size(data_source)
        
        for chunk_data, chunk_info in self._generate_chunks(data_source, chunk_size, overlap):
            # Process chunk
            result = processor_func(chunk_data, **chunk_info)
            
            # Update progress
            n_processed += chunk_info['chunk_size']
            if progress_callback:
                progress = min(100, int(n_processed / total_size * 100))
                progress_callback(progress)
            
            # Force garbage collection
            gc.collect()
            
            yield result
    
    def _calculate_optimal_chunk_size(self, data_source: Any) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Get data dimensions
        if hasattr(data_source, 'shape'):
            n_features = data_source.shape[1] if len(data_source.shape) > 1 else 1
        else:
            n_features = 2  # Default assumption
        
        # Calculate samples that fit in memory limit
        bytes_per_sample = n_features * 8  # Assuming float64
        overhead_factor = 2.0  # Account for processing overhead
        max_samples = int((self.memory_limit_gb * 1024**3) / (bytes_per_sample * overhead_factor))
        
        # Round to reasonable chunk size
        chunk_size = min(max_samples, 100000)  # Cap at 100k for responsiveness
        chunk_size = max(chunk_size, 1000)     # At least 1k samples
        
        return chunk_size
    
    def _get_data_size(self, data_source: Any) -> int:
        """Get total size of data source."""
        if hasattr(data_source, 'shape'):
            return data_source.shape[0]
        elif hasattr(data_source, '__len__'):
            return len(data_source)
        else:
            return -1  # Unknown size
    
    def _generate_chunks(self, data_source: Any, chunk_size: int, 
                        overlap: int) -> Generator:
        """Generate data chunks with optional overlap."""
        if hasattr(data_source, 'shape'):
            # NumPy array or similar
            n_samples = data_source.shape[0]
            
            for start_idx in range(0, n_samples, chunk_size - overlap):
                end_idx = min(start_idx + chunk_size, n_samples)
                
                chunk_info = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'chunk_size': end_idx - start_idx,
                    'is_last': end_idx >= n_samples
                }
                
                yield data_source[start_idx:end_idx], chunk_info
        else:
            # Assume it's already a generator
            for chunk in data_source:
                yield chunk, {'chunk_size': len(chunk)}


def check_memory_usage(data_shape: Tuple[int, ...], dtype: np.dtype = np.dtype(np.float64)) -> Dict[str, Any]:
    """Check if data will fit in memory."""
    # Calculate data size
    element_size = dtype.itemsize
    n_elements = np.prod(data_shape)
    data_size_gb = (n_elements * element_size) / (1024**3)
    
    # Get available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    
    return {
        'data_size_gb': float(data_size_gb),
        'available_gb': float(available_gb),
        'total_gb': float(total_gb),
        'will_fit': bool(data_size_gb < available_gb * 0.5)  # Use only 50% of available
    }


def create_spatial_subsampler(strategy: str = 'stratified') -> Callable:
    """Create a spatial subsampling function."""
    
    def subsample(data: np.ndarray, coordinates: np.ndarray, 
                  target_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample data spatially."""
        n_samples = data.shape[0]
        
        if n_samples <= target_samples:
            return data, np.arange(n_samples)
        
        if strategy == 'stratified':
            # Spatial stratified sampling using KMeans
            try:
                from sklearn.cluster import MiniBatchKMeans
                
                n_clusters = min(100, target_samples // 10)
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(coordinates)
                
                # Sample from each cluster
                samples_per_cluster = target_samples // n_clusters
                indices = []
                
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    if len(cluster_indices) > samples_per_cluster:
                        selected = np.random.choice(cluster_indices, samples_per_cluster, 
                                                  replace=False)
                    else:
                        selected = cluster_indices
                    
                    indices.extend(selected)
                
                indices = np.array(indices[:target_samples])
                return data[indices], indices
                
            except ImportError:
                logger.warning("sklearn not available, falling back to random sampling")
                # Fall through to random sampling
            
        # random sampling (default fallback)
        indices = np.random.choice(n_samples, target_samples, replace=False)
        return data[indices], indices
    
    return subsample
