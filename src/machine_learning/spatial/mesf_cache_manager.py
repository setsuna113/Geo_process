"""MESF Cache Manager for spatial sampling-aware caching.

This manager ensures:
1. Original data files are never modified
2. MESF is cached per (file, sampling_rate, params) combination
3. Efficient reuse of computed MESF values
4. Thread-safe atomic operations
5. Comprehensive error handling
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
import tempfile
import fcntl
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MESFCacheManager:
    """Manages MESF computation caching with spatial sampling awareness."""
    
    # Class-level lock for thread safety
    _lock = threading.Lock()
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size_gb: float = 10.0,
                 max_cache_age_days: int = 30):
        """Initialize the cache manager with configurable and secure cache directory.
        
        Args:
            cache_dir: Directory for cache storage (default from env or system temp dir)
            max_cache_size_gb: Maximum cache size in GB (default 10GB)
            max_cache_age_days: Maximum age of cache entries in days (default 30)
        """
        if cache_dir is None:
            # Allow configuration via environment variable, with user-isolated fallback
            base_temp = os.environ.get('ML_CACHE_DIR', tempfile.gettempdir())
            user_id = os.getuid() if hasattr(os, 'getuid') else 'user'
            cache_dir = os.path.join(base_temp, f'ml_cache_{user_id}', 'mesf')
        
        self.max_cache_size_bytes = max_cache_size_gb * 1024 * 1024 * 1024
        self.max_cache_age_seconds = max_cache_age_days * 24 * 3600
        
        # Validate and sanitize the cache directory path
        try:
            self.cache_dir = Path(cache_dir).resolve()
            
            # Ensure cache directory is not in a dangerous location
            dangerous_paths = ['/', '/etc', '/usr', '/bin', '/sbin', '/boot', '/dev', '/proc', '/sys']
            if str(self.cache_dir) in dangerous_paths or any(str(self.cache_dir).startswith(p + '/') for p in dangerous_paths[:4]):
                raise ValueError(f"Cache directory cannot be in system-critical location: {self.cache_dir}")
            
            # Create directory with proper error handling
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Set restrictive permissions (owner only)
            os.chmod(self.cache_dir, 0o700)
            
            # Verify we can write to the directory
            test_file = self.cache_dir / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
            except (OSError, PermissionError) as e:
                raise PermissionError(f"Cannot write to cache directory {self.cache_dir}: {e}")
                
        except (OSError, PermissionError, ValueError) as e:
            # Fall back to user's home directory if primary fails
            fallback_dir = Path.home() / '.cache' / 'ml_cache' / 'mesf'
            logger.warning(f"Failed to create cache at {cache_dir}: {e}")
            logger.info(f"Using fallback cache directory: {fallback_dir}")
            self.cache_dir = fallback_dir
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.cache_dir, 0o700)
        
        logger.info(f"MESF cache directory initialized: {self.cache_dir}")
    
    def get_cache_key(self, 
                      file_path: str,
                      spatial_grid_step: int,
                      mesf_params: Dict[str, Any]) -> str:
        """Generate unique cache key based on inputs with validation.
        
        Args:
            file_path: Path to original data file
            spatial_grid_step: Spatial sampling rate (1 = no sampling)
            mesf_params: MESF computation parameters
            
        Returns:
            Unique cache key string
            
        Raises:
            ValueError: If inputs are invalid or contain path traversal
            FileNotFoundError: If file doesn't exist
        """
        # Input validation
        if not file_path:
            raise ValueError("file_path cannot be empty")
        
        # Resolve and validate file path to prevent path traversal
        try:
            abs_path = Path(file_path).resolve()
            
            # Check file exists and is a regular file
            if not abs_path.exists():
                raise FileNotFoundError(f"File not found: {abs_path}")
            if not abs_path.is_file():
                raise ValueError(f"Path is not a regular file: {abs_path}")
                
            # Ensure path is not in a dangerous location
            dangerous_prefixes = ['/dev', '/proc', '/sys']
            if any(str(abs_path).startswith(prefix) for prefix in dangerous_prefixes):
                raise ValueError(f"File path in restricted location: {abs_path}")
                
        except (RuntimeError, OSError) as e:
            raise ValueError(f"Invalid file path: {file_path}") from e
        
        # Validate spatial_grid_step
        if not isinstance(spatial_grid_step, int) or spatial_grid_step < 1:
            raise ValueError(f"spatial_grid_step must be positive integer, got {spatial_grid_step}")
        
        # Get file stats for cache invalidation
        try:
            file_stat = abs_path.stat()
        except OSError as e:
            raise FileNotFoundError(f"Cannot access file {abs_path}: {e}") from e
        
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
        
        Uses atomic file operations to prevent race conditions.
        
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
        """Clear cache files safely with proper locking.
        
        Args:
            older_than_days: Only clear files older than this many days
                           None = clear all
        """
        import time
        import shutil
        
        with self._lock:
            count = 0
            errors = 0
            
            # First, check available disk space
            total, used, free = shutil.disk_usage(self.cache_dir)
            logger.info(f"Cache directory disk usage: {used / 1024 / 1024 / 1024:.1f}GB used, "
                       f"{free / 1024 / 1024 / 1024:.1f}GB free")
            
            try:
                for cache_file in self.cache_dir.glob("mesf_*.parquet"):
                    try:
                        if older_than_days is not None:
                            file_age_days = (time.time() - cache_file.stat().st_mtime) / 86400
                            if file_age_days < older_than_days:
                                continue
                        
                        # Remove cache file
                        cache_file.unlink()
                        
                        # Also remove metadata
                        metadata_file = Path(str(cache_file.with_suffix('')) + '_metadata.json')
                        if metadata_file.exists():
                            metadata_file.unlink()
                        
                        count += 1
                        
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Failed to remove cache file {cache_file}: {e}")
                        errors += 1
                        continue
                
                # Also clean up any orphaned temp files
                for temp_file in self.cache_dir.glob("temp_*.parquet"):
                    try:
                        # Only remove if older than 1 hour
                        if (time.time() - temp_file.stat().st_mtime) > 3600:
                            temp_file.unlink()
                            logger.debug(f"Removed orphaned temp file: {temp_file}")
                    except:
                        pass
                
                for temp_file in self.cache_dir.glob("temp_*.json"):
                    try:
                        if (time.time() - temp_file.stat().st_mtime) > 3600:
                            temp_file.unlink()
                            logger.debug(f"Removed orphaned temp file: {temp_file}")
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Error during cache clearing: {e}")
                
            if errors > 0:
                logger.warning(f"Failed to remove {errors} cache files")
            
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
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics including size and age information.
        
        Returns:
            Dictionary with cache statistics
        """
        import time
        
        stats = {
            'cache_dir': str(self.cache_dir),
            'total_files': 0,
            'total_size_mb': 0.0,
            'oldest_file_days': 0.0,
            'newest_file_days': 0.0,
            'avg_file_size_mb': 0.0,
            'disk_free_gb': 0.0,
            'disk_used_gb': 0.0,
            'cache_entries': []
        }
        
        try:
            # Get disk usage
            stat = os.statvfs(self.cache_dir)
            stats['disk_free_gb'] = (stat.f_bavail * stat.f_frsize) / (1024**3)
            stats['disk_used_gb'] = ((stat.f_blocks - stat.f_bavail) * stat.f_frsize) / (1024**3)
            
            cache_files = list(self.cache_dir.glob("mesf_*.parquet"))
            if not cache_files:
                return stats
            
            current_time = time.time()
            oldest_mtime = current_time
            newest_mtime = 0
            total_size = 0
            
            for cache_file in cache_files:
                try:
                    file_stat = cache_file.stat()
                    file_size_mb = file_stat.st_size / (1024**2)
                    file_age_days = (current_time - file_stat.st_mtime) / 86400
                    
                    total_size += file_stat.st_size
                    oldest_mtime = min(oldest_mtime, file_stat.st_mtime)
                    newest_mtime = max(newest_mtime, file_stat.st_mtime)
                    
                    stats['cache_entries'].append({
                        'file': cache_file.name,
                        'size_mb': round(file_size_mb, 2),
                        'age_days': round(file_age_days, 1)
                    })
                except:
                    continue
            
            stats['total_files'] = len(cache_files)
            stats['total_size_mb'] = round(total_size / (1024**2), 2)
            stats['oldest_file_days'] = round((current_time - oldest_mtime) / 86400, 1) if oldest_mtime < current_time else 0
            stats['newest_file_days'] = round((current_time - newest_mtime) / 86400, 1) if newest_mtime > 0 else 0
            stats['avg_file_size_mb'] = round(stats['total_size_mb'] / len(cache_files), 2) if cache_files else 0
            
        except Exception as e:
            logger.warning(f"Failed to get cache statistics: {e}")
        
        return stats
    
    def cleanup_old_entries(self, force: bool = False) -> Dict[str, int]:
        """Clean up old cache entries based on age and size limits.
        
        Args:
            force: Force cleanup even if within limits
            
        Returns:
            Dictionary with cleanup statistics
        """
        import time
        
        cleanup_stats = {
            'removed_by_age': 0,
            'removed_by_size': 0,
            'errors': 0,
            'space_freed_mb': 0
        }
        
        try:
            current_time = time.time()
            cache_files = []
            
            # Collect all cache files with metadata
            for cache_file in self.cache_dir.glob("mesf_*.parquet"):
                try:
                    file_stat = cache_file.stat()
                    cache_files.append({
                        'path': cache_file,
                        'size': file_stat.st_size,
                        'mtime': file_stat.st_mtime,
                        'age_seconds': current_time - file_stat.st_mtime
                    })
                except:
                    continue
            
            # Sort by age (oldest first)
            cache_files.sort(key=lambda x: x['mtime'])
            
            # Remove files older than max age
            for file_info in cache_files[:]:
                if file_info['age_seconds'] > self.max_cache_age_seconds:
                    try:
                        file_info['path'].unlink()
                        # Also remove metadata
                        metadata_path = Path(str(file_info['path'].with_suffix('')) + '_metadata.json')
                        if metadata_path.exists():
                            metadata_path.unlink()
                        
                        cleanup_stats['removed_by_age'] += 1
                        cleanup_stats['space_freed_mb'] += file_info['size'] / (1024**2)
                        cache_files.remove(file_info)
                    except Exception as e:
                        logger.warning(f"Failed to remove old cache file: {e}")
                        cleanup_stats['errors'] += 1
            
            # Check total cache size
            total_size = sum(f['size'] for f in cache_files)
            
            # Remove oldest files if cache exceeds size limit
            while total_size > self.max_cache_size_bytes and cache_files:
                file_info = cache_files.pop(0)
                try:
                    file_info['path'].unlink()
                    # Also remove metadata
                    metadata_path = Path(str(file_info['path'].with_suffix('')) + '_metadata.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    cleanup_stats['removed_by_size'] += 1
                    cleanup_stats['space_freed_mb'] += file_info['size'] / (1024**2)
                    total_size -= file_info['size']
                except Exception as e:
                    logger.warning(f"Failed to remove cache file for size limit: {e}")
                    cleanup_stats['errors'] += 1
            
            if cleanup_stats['removed_by_age'] + cleanup_stats['removed_by_size'] > 0:
                logger.info(f"Cache cleanup: removed {cleanup_stats['removed_by_age']} old files, "
                           f"{cleanup_stats['removed_by_size']} for size limit, "
                           f"freed {cleanup_stats['space_freed_mb']:.1f}MB")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            cleanup_stats['errors'] += 1
        
        return cleanup_stats
    
    def validate_cache_integrity(self, cache_key: str) -> bool:
        """Validate cache entry integrity including schema and data.
        
        Args:
            cache_key: Cache key to validate
            
        Returns:
            True if cache is valid, False otherwise
        """
        cached_path = self.get_cached_path(cache_key)
        metadata_path = self.get_metadata_path(cache_key)
        
        try:
            # Check files exist
            if not cached_path.exists() or not metadata_path.exists():
                return False
            
            # Validate metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required metadata fields
            required_fields = ['n_eigenvectors', 'n_samples']
            if not all(field in metadata for field in required_fields):
                logger.warning(f"Cache metadata missing required fields: {cache_key}")
                return False
            
            # Validate parquet file
            pq_file = pq.ParquetFile(cached_path)
            schema = pq_file.schema_arrow
            
            # Check for MESF columns
            mesf_cols = [name for name in schema.names if name.startswith('ME')]
            if len(mesf_cols) != metadata.get('n_eigenvectors', 0):
                logger.warning(f"Cache eigenvector count mismatch: expected {metadata.get('n_eigenvectors')}, "
                              f"found {len(mesf_cols)}")
                return False
            
            # Quick data integrity check (read first row)
            table = pq_file.read_row_group(0, columns=[mesf_cols[0]] if mesf_cols else None)
            if table.num_rows == 0:
                logger.warning(f"Cache file has no data: {cache_key}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache validation failed for {cache_key}: {e}")
            return False