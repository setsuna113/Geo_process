"""MESF Cache Manager for spatial sampling-aware caching.

This manager ensures:
1. Original data files are never modified
2. MESF is cached per (file, sampling_rate, params) combination
3. Efficient reuse of computed MESF values
"""

import os
import hashlib
import json
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class MESFCacheManager:
    """Manages MESF computation caching with spatial sampling awareness."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory for cache storage (default: /tmp/ml_cache/mesf)
        """
        if cache_dir is None:
            # Use /tmp for better accessibility by H2O Java process
            cache_dir = "/tmp/ml_cache/mesf"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set permissions for H2O access
        os.chmod(self.cache_dir, 0o755)
        
        logger.info(f"MESF cache directory: {self.cache_dir}")
    
    def get_cache_key(self, 
                      file_path: str,
                      spatial_grid_step: int,
                      mesf_params: Dict[str, Any]) -> str:
        """Generate unique cache key based on inputs.
        
        Args:
            file_path: Path to original data file
            spatial_grid_step: Spatial sampling rate (1 = no sampling)
            mesf_params: MESF computation parameters
            
        Returns:
            Unique cache key string
        """
        # Get file stats for cache invalidation
        file_stat = os.stat(file_path)
        
        # Create cache key components
        key_data = {
            'file_path': os.path.abspath(file_path),
            'file_size': file_stat.st_size,
            'file_mtime': file_stat.st_mtime,
            'spatial_grid_step': spatial_grid_step,
            'mesf_params': mesf_params
        }
        
        # Generate hash
        key_str = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        
        return cache_key
    
    def get_cached_path(self, cache_key: str) -> Path:
        """Get the path for cached MESF-augmented data.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Path to cached parquet file
        """
        return self.cache_dir / f"mesf_{cache_key}.parquet"
    
    def get_metadata_path(self, cache_key: str) -> Path:
        """Get the path for cache metadata.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Path to metadata JSON file
        """
        return self.cache_dir / f"mesf_{cache_key}_metadata.json"
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if MESF data is cached.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            True if cached data exists
        """
        cached_path = self.get_cached_path(cache_key)
        metadata_path = self.get_metadata_path(cache_key)
        
        return cached_path.exists() and metadata_path.exists()
    
    def save_to_cache(self,
                      cache_key: str,
                      data_table: pa.Table,
                      metadata: Dict[str, Any]) -> str:
        """Save MESF-augmented data to cache.
        
        Args:
            cache_key: Unique cache key
            data_table: PyArrow table with MESF columns
            metadata: Metadata about the computation
            
        Returns:
            Path to cached file
        """
        cached_path = self.get_cached_path(cache_key)
        metadata_path = self.get_metadata_path(cache_key)
        
        # Save data
        pq.write_table(data_table, cached_path)
        # Set permissions for H2O access
        os.chmod(cached_path, 0o644)
        logger.info(f"Saved MESF data to cache: {cached_path}")
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        os.chmod(metadata_path, 0o644)
        
        return str(cached_path)
    
    def load_from_cache(self, cache_key: str) -> Tuple[str, Dict[str, Any]]:
        """Load MESF-augmented data from cache.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Tuple of (cached_file_path, metadata)
        """
        if not self.is_cached(cache_key):
            raise ValueError(f"No cache found for key: {cache_key}")
        
        cached_path = self.get_cached_path(cache_key)
        metadata_path = self.get_metadata_path(cache_key)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded MESF data from cache: {cached_path}")
        logger.info(f"Cache metadata: {metadata}")
        
        return str(cached_path), metadata
    
    def get_or_compute_mesf(self,
                           file_path: str,
                           spatial_grid_step: int,
                           mesf_params: Dict[str, Any],
                           compute_fn: callable) -> str:
        """Get cached MESF data or compute if not cached.
        
        Args:
            file_path: Path to original data file
            spatial_grid_step: Spatial sampling rate
            mesf_params: MESF computation parameters
            compute_fn: Function to compute MESF if not cached
                       Should return (augmented_table, metadata)
            
        Returns:
            Path to MESF-augmented parquet file
        """
        # Generate cache key
        cache_key = self.get_cache_key(file_path, spatial_grid_step, mesf_params)
        
        # Check cache
        if self.is_cached(cache_key):
            logger.info(f"Using cached MESF data for key: {cache_key}")
            cached_path, metadata = self.load_from_cache(cache_key)
            return cached_path
        
        # Compute MESF
        logger.info(f"Computing MESF (not in cache): key={cache_key}")
        augmented_table, metadata = compute_fn(file_path, spatial_grid_step, mesf_params)
        
        # Save to cache
        cached_path = self.save_to_cache(cache_key, augmented_table, metadata)
        
        return cached_path
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """Clear cache files.
        
        Args:
            older_than_days: Only clear files older than this many days
                           None = clear all
        """
        import time
        
        count = 0
        for cache_file in self.cache_dir.glob("mesf_*.parquet"):
            if older_than_days is not None:
                file_age_days = (time.time() - cache_file.stat().st_mtime) / 86400
                if file_age_days < older_than_days:
                    continue
            
            cache_file.unlink()
            # Also remove metadata
            metadata_file = Path(str(cache_file.with_suffix('')) + '_metadata.json')
            if metadata_file.exists():
                metadata_file.unlink()
            
            count += 1
        
        logger.info(f"Cleared {count} cache files")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache usage.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("mesf_*.parquet"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'num_cached': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_files': [f.name for f in cache_files[:10]]  # First 10
        }


class FungiRichnessCacheManager:
    """Manages fungi_richness computation caching."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory for cache storage
        """
        if cache_dir is None:
            # Use /tmp for better accessibility
            cache_dir = "/tmp/ml_cache/fungi"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set permissions for accessibility
        os.chmod(self.cache_dir, 0o755)
        
        logger.info(f"Fungi richness cache directory: {self.cache_dir}")
    
    def get_cache_key(self, file_path: str) -> str:
        """Generate cache key for fungi richness.
        
        Args:
            file_path: Path to original data file
            
        Returns:
            Unique cache key
        """
        file_stat = os.stat(file_path)
        
        key_data = {
            'file_path': os.path.abspath(file_path),
            'file_size': file_stat.st_size,
            'file_mtime': file_stat.st_mtime
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get_cached_path(self, cache_key: str) -> Path:
        """Get path for cached data with fungi_richness."""
        return self.cache_dir / f"fungi_{cache_key}.parquet"
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if fungi_richness is cached."""
        return self.get_cached_path(cache_key).exists()
    
    def get_or_compute_fungi(self,
                            file_path: str,
                            compute_fn: callable) -> str:
        """Get cached data with fungi_richness or compute if needed.
        
        Args:
            file_path: Path to original data
            compute_fn: Function to compute fungi_richness
                       Should return path to augmented file
            
        Returns:
            Path to file with fungi_richness column
        """
        cache_key = self.get_cache_key(file_path)
        cached_path = self.get_cached_path(cache_key)
        
        if self.is_cached(cache_key):
            logger.info(f"Using cached fungi_richness data: {cached_path}")
            return str(cached_path)
        
        # Compute fungi_richness
        logger.info(f"Computing fungi_richness (not in cache)")
        augmented_path = compute_fn(file_path, str(cached_path))
        
        return augmented_path