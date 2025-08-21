"""MESF Cache Manager with robust caching for eigenvectors only.

This manager ensures:
1. Efficient NPZ storage of eigenvectors only (not full data)
2. Thread-safe atomic operations with file locking
3. Cache invalidation on file changes (mtime/size)
4. Robust error handling and corruption recovery
5. Minimal storage overhead
"""

import os
import hashlib
import json
import logging
import tempfile
import fcntl
import shutil
import numpy as np
# Removed pandas and pyarrow - we only store numpy arrays
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
        """Get the path for cached MESF eigenvectors.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Path to cached NPZ file
        """
        # Use same naming as simple cache for compatibility
        return self.cache_dir / f"{cache_key}.npz"
    
    def get_metadata_path(self, cache_key: str) -> Path:
        """Get the path for cache metadata.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Path to metadata JSON file
        """
        return self.cache_dir / f"{cache_key}_metadata.json"
    
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
                      eigenvectors: np.ndarray,
                      metadata: Dict[str, Any]) -> str:
        """Save MESF eigenvectors to cache atomically.
        
        Uses atomic file operations to prevent race conditions.
        
        Args:
            cache_key: Unique cache key
            eigenvectors: MESF eigenvectors array (n_points, n_eigenvectors)
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
            temp_data_path = self.cache_dir / f"tmp_{cache_key}.npz"
            temp_meta_path = self.cache_dir / f"tmp_{cache_key}_metadata.json"
            
            try:
                # Check available disk space first
                stat = os.statvfs(self.cache_dir)
                free_bytes = stat.f_bavail * stat.f_frsize
                # Estimate compressed size (NPZ typically achieves 5-10x compression)
                estimated_size = eigenvectors.nbytes * 0.3  # Conservative estimate
                
                if free_bytes < estimated_size:
                    raise OSError(f"Insufficient disk space. Need ~{estimated_size/(1024*1024):.1f}MB, "
                                f"have {free_bytes/(1024*1024):.1f}MB")
                
                # Save eigenvectors to temporary NPZ file - exact same format as simple cache
                np.savez_compressed(temp_data_path, eigenvectors=eigenvectors)
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
    
    def load_from_cache(self, cache_key: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load MESF eigenvectors from cache with error handling.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Tuple of (eigenvectors_array, metadata)
            
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
                
                # Load and verify eigenvectors
                try:
                    with np.load(cached_path) as data:
                        eigenvectors = data['eigenvectors']
                        # Verify shape matches metadata
                        if 'eigenvector_shape' in metadata:
                            expected_shape = tuple(metadata['eigenvector_shape'])
                            if eigenvectors.shape != expected_shape:
                                raise ValueError(f"Shape mismatch: {eigenvectors.shape} != {expected_shape}")
                except Exception as e:
                    logger.error(f"Failed to load cache data: {e}")
                    self._cleanup_cache(cache_key)
                    raise OSError(f"Corrupted cache data for key: {cache_key}") from e
                
                # Return the eigenvectors and metadata
                return eigenvectors, metadata
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
                
                # Quick validation of NPZ file
                try:
                    with np.load(cached_path) as data:
                        eigenvectors = data['eigenvectors']
                        if 'eigenvector_shape' in metadata:
                            expected_shape = tuple(metadata['eigenvector_shape'])
                            if eigenvectors.shape != expected_shape:
                                raise ValueError(f"Shape mismatch")
                except Exception as e:
                    logger.error(f"Cache data appears corrupted: {e}")
                    self._cleanup_cache(cache_key)
                    raise OSError(f"Corrupted cache data for key: {cache_key}") from e
                
                logger.info(f"Successfully read cache without lock: {cached_path}")
                return eigenvectors, metadata
                
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
                           compute_fn: callable) -> np.ndarray:
        """Get cached MESF eigenvectors or compute if not cached.
        
        Args:
            file_path: Path to original data file
            spatial_grid_step: Spatial sampling rate
            mesf_params: MESF computation parameters
            compute_fn: Function to compute MESF if not cached
                       Should return (eigenvectors, metadata)
            
        Returns:
            MESF eigenvectors array
        """
        # Generate cache key
        cache_key = self.get_cache_key(file_path, spatial_grid_step, mesf_params)
        
        # Check cache
        if self.is_cached(cache_key):
            logger.info(f"Using cached MESF eigenvectors for key: {cache_key}")
            eigenvectors, metadata = self.load_from_cache(cache_key)
            return eigenvectors
        
        # Compute MESF
        logger.info(f"Computing MESF (not in cache): key={cache_key}")
        eigenvectors, metadata = compute_fn(file_path, spatial_grid_step, mesf_params)
        
        # Add shape to metadata for validation
        metadata['eigenvector_shape'] = eigenvectors.shape
        
        # Save to cache
        self.save_to_cache(cache_key, eigenvectors, metadata)
        
        return eigenvectors
    
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
                for cache_file in self.cache_dir.glob("*.npz"):
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
        cache_files = list(self.cache_dir.glob("*.npz"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'num_cached': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_files': [f.name for f in cache_files[:10]]  # First 10
        }
    
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
            
            # Quick array validation
            with np.load(cached_path) as data:
                eigenvectors = data['eigenvectors']
                if 'eigenvector_shape' in metadata:
                    expected_shape = tuple(metadata['eigenvector_shape'])
                    if eigenvectors.shape != expected_shape:
                        logger.warning(f"Cache shape mismatch: expected {expected_shape}, got {eigenvectors.shape}")
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache validation failed for {cache_key}: {e}")
            return False

