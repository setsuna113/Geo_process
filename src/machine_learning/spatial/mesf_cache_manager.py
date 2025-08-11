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
import tempfile
import fcntl
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MESFCacheManager:
    """Manages MESF computation caching with spatial sampling awareness."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory for cache storage (default: system temp dir)
        """
        if cache_dir is None:
            # Use system temp directory with proper user isolation
            base_temp = os.environ.get('ML_CACHE_DIR', tempfile.gettempdir())
            user_id = os.getuid() if hasattr(os, 'getuid') else 'user'
            cache_dir = os.path.join(base_temp, f'ml_cache_{user_id}', 'mesf')
        
        self.cache_dir = Path(cache_dir)
        
        try:
            # Create directory with restricted permissions
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions (owner only)
            os.chmod(self.cache_dir, 0o700)
        except (OSError, PermissionError) as e:
            # Fall back to user's home directory if temp fails
            fallback_dir = Path.home() / '.cache' / 'ml_cache' / 'mesf'
            logger.warning(f"Failed to create cache at {self.cache_dir}: {e}")
            logger.info(f"Using fallback cache directory: {fallback_dir}")
            self.cache_dir = fallback_dir
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.cache_dir, 0o700)
        
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
            
        Raises:
            ValueError: If file_path is invalid or contains path traversal
            FileNotFoundError: If file doesn't exist
        """
        # Validate and sanitize file path
        if not file_path:
            raise ValueError("file_path cannot be empty")
        
        # Resolve to absolute path and check for path traversal
        try:
            abs_path = Path(file_path).resolve()
        except (RuntimeError, OSError) as e:
            raise ValueError(f"Invalid file path: {file_path}") from e
        
        # Check file existence separately to preserve FileNotFoundError
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not abs_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Validate spatial_grid_step
        if not isinstance(spatial_grid_step, int) or spatial_grid_step < 1:
            raise ValueError(f"spatial_grid_step must be positive integer, got {spatial_grid_step}")
        
        # Get file stats for cache invalidation
        try:
            file_stat = abs_path.stat()
        except OSError as e:
            raise OSError(f"Cannot access file: {abs_path}") from e
        
        # Create cache key components using sanitized path
        key_data = {
            'file_path': str(abs_path),
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
    
    @contextmanager
    def _file_lock(self, lock_path: Path, timeout: int = 30):
        """Context manager for file-based locking to prevent race conditions.
        
        Args:
            lock_path: Path to lock file
            timeout: Maximum seconds to wait for lock
            
        Yields:
            Lock file descriptor
        """
        lock_file = None
        try:
            lock_file = open(lock_path, 'w')
            # Try to acquire exclusive lock with timeout
            import time
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Could not acquire lock on {lock_path} after {timeout}s")
                    time.sleep(0.1)
            
            yield lock_file
        finally:
            if lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except:
                    pass
                lock_file.close()
                try:
                    lock_path.unlink()
                except:
                    pass

    def save_to_cache(self,
                      cache_key: str,
                      data_table: pa.Table,
                      metadata: Dict[str, Any]) -> str:
        """Save MESF-augmented data to cache atomically.
        
        Args:
            cache_key: Unique cache key
            data_table: PyArrow table with MESF columns
            metadata: Metadata about the computation
            
        Returns:
            Path to cached file
            
        Raises:
            OSError: If unable to save to cache (disk full, permissions, etc.)
            TimeoutError: If unable to acquire lock
        """
        cached_path = self.get_cached_path(cache_key)
        metadata_path = self.get_metadata_path(cache_key)
        lock_path = self.cache_dir / f".lock_{cache_key}"
        
        # Use file locking to prevent race conditions
        with self._file_lock(lock_path):
            # Check if another process already created the cache
            if cached_path.exists() and metadata_path.exists():
                logger.info(f"Cache already exists (created by another process): {cached_path}")
                return str(cached_path)
            
            # Use atomic write with temporary files
            temp_data_path = cached_path.with_suffix('.tmp')
            temp_meta_path = metadata_path.with_suffix('.tmp')
            
            try:
                # Check available disk space first
                stat = os.statvfs(self.cache_dir)
                free_bytes = stat.f_bavail * stat.f_frsize
                # Estimate size (rough estimate: current table size * 1.2 for safety)
                estimated_size = data_table.nbytes * 1.2 if hasattr(data_table, 'nbytes') else 100 * 1024 * 1024
                
                if free_bytes < estimated_size:
                    raise OSError(f"Insufficient disk space. Need ~{estimated_size/(1024*1024):.1f}MB, "
                                f"have {free_bytes/(1024*1024):.1f}MB")
                
                # Save data to temporary file
                pq.write_table(data_table, temp_data_path)
                # Set restrictive permissions
                os.chmod(temp_data_path, 0o644)
                
                # Save metadata to temporary file
                with open(temp_meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                os.chmod(temp_meta_path, 0o644)
                
                # Atomic rename (on same filesystem)
                temp_data_path.rename(cached_path)
                temp_meta_path.rename(metadata_path)
                
                logger.info(f"Saved MESF data to cache: {cached_path}")
                return str(cached_path)
                
            except Exception as e:
                # Clean up temporary files on error
                for temp_path in [temp_data_path, temp_meta_path]:
                    try:
                        if temp_path.exists():
                            temp_path.unlink()
                    except:
                        pass
                        
                logger.error(f"Failed to save cache: {e}")
                raise OSError(f"Failed to save cache for key {cache_key}") from e
    
    def load_from_cache(self, cache_key: str) -> Tuple[str, Dict[str, Any]]:
        """Load MESF-augmented data from cache with error handling.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Tuple of (cached_file_path, metadata)
            
        Raises:
            ValueError: If cache doesn't exist
            OSError: If unable to read cache files
        """
        if not self.is_cached(cache_key):
            raise ValueError(f"No cache found for key: {cache_key}")
        
        cached_path = self.get_cached_path(cache_key)
        metadata_path = self.get_metadata_path(cache_key)
        lock_path = self.cache_dir / f".lock_{cache_key}"
        
        # Use shared lock for reading (multiple readers OK)
        try:
            # Quick check without lock first
            if not cached_path.exists() or not metadata_path.exists():
                raise ValueError(f"Cache files missing for key: {cache_key}")
            
            # Use file locking to ensure consistent read
            with self._file_lock(lock_path, timeout=10):
                # Verify files still exist (could have been deleted)
                if not cached_path.exists() or not metadata_path.exists():
                    raise ValueError(f"Cache files removed during read for key: {cache_key}")
                
                # Load metadata with error handling
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Failed to read cache metadata: {e}")
                    # Try to clean up corrupted cache
                    self._cleanup_cache(cache_key)
                    raise OSError(f"Corrupted cache metadata for key: {cache_key}") from e
                
                # Verify data file is readable
                try:
                    # Quick check that parquet file is valid
                    pq_file = pq.ParquetFile(cached_path)
                    _ = pq_file.metadata
                except Exception as e:
                    logger.error(f"Failed to verify cache data: {e}")
                    self._cleanup_cache(cache_key)
                    raise OSError(f"Corrupted cache data for key: {cache_key}") from e
                
                logger.info(f"Loaded MESF data from cache: {cached_path}")
                logger.info(f"Cache metadata: {metadata}")
                
                return str(cached_path), metadata
                
        except TimeoutError:
            logger.warning(f"Timeout waiting for cache lock, proceeding without lock")
            # Fall back to reading without lock with validation
            try:
                # Verify files exist first
                if not cached_path.exists() or not metadata_path.exists():
                    raise ValueError(f"Cache files missing for key: {cache_key}")
                
                # Read metadata with error handling
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Quick validation of parquet file
                try:
                    pq_file = pq.ParquetFile(cached_path)
                    _ = pq_file.metadata
                except Exception as e:
                    logger.error(f"Cache data appears corrupted: {e}")
                    self._cleanup_cache(cache_key)
                    raise OSError(f"Corrupted cache data for key: {cache_key}") from e
                
                logger.info(f"Successfully read cache without lock: {cached_path}")
                return str(cached_path), metadata
                
            except (json.JSONDecodeError, IOError, ValueError) as e:
                logger.error(f"Failed to read cache without lock: {e}")
                # Attempt cleanup on error
                self._cleanup_cache(cache_key)
                raise OSError(f"Failed to read cache for key: {cache_key}") from e
    
    def _cleanup_cache(self, cache_key: str):
        """Clean up corrupted or incomplete cache files.
        
        Args:
            cache_key: Cache key to clean up
        """
        cached_path = self.get_cached_path(cache_key)
        metadata_path = self.get_metadata_path(cache_key)
        
        for path in [cached_path, metadata_path]:
            try:
                if path.exists():
                    path.unlink()
                    logger.info(f"Cleaned up cache file: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {path}: {e}")
    
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
            # Use system temp directory with proper user isolation (same as MESF)
            base_temp = os.environ.get('ML_CACHE_DIR', tempfile.gettempdir())
            user_id = os.getuid() if hasattr(os, 'getuid') else 'user'
            cache_dir = os.path.join(base_temp, f'ml_cache_{user_id}', 'fungi')
        
        self.cache_dir = Path(cache_dir)
        
        try:
            # Create directory with secure permissions (same as MESF)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions (owner only)
            os.chmod(self.cache_dir, 0o700)
        except (OSError, PermissionError) as e:
            # Fall back to user's home directory if temp fails
            fallback_dir = Path.home() / '.cache' / 'ml_cache' / 'fungi'
            logger.warning(f"Failed to create cache at {self.cache_dir}: {e}")
            logger.info(f"Using fallback cache directory: {fallback_dir}")
            self.cache_dir = fallback_dir
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.cache_dir, 0o700)
        
        logger.info(f"Fungi richness cache directory: {self.cache_dir}")
    
    def get_cache_key(self, file_path: str) -> str:
        """Generate cache key for fungi richness with validation.
        
        Args:
            file_path: Path to original data file
            
        Returns:
            Unique cache key
            
        Raises:
            ValueError: If file_path is invalid or contains path traversal
            FileNotFoundError: If file doesn't exist
        """
        # Validate and sanitize file path (same as MESF)
        if not file_path:
            raise ValueError("file_path cannot be empty")
        
        # Resolve to absolute path and check for path traversal
        try:
            abs_path = Path(file_path).resolve()
        except (RuntimeError, OSError) as e:
            raise ValueError(f"Invalid file path: {file_path}") from e
        
        # Check file existence separately to preserve FileNotFoundError
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not abs_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Get file stats for cache invalidation
        try:
            file_stat = abs_path.stat()
        except OSError as e:
            raise OSError(f"Cannot access file: {abs_path}") from e
        
        key_data = {
            'file_path': str(abs_path),
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