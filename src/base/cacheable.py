"""Mixin for caching support with TTL management and memory tracking."""

from abc import ABC
from typing import Any, Optional, Dict, Union, Callable
import time
import threading
import weakref
import hashlib
import pickle
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    value: Any
    timestamp: float
    ttl_seconds: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds <= 0:  # Never expires
            return False
        return (time.time() - self.timestamp) > self.ttl_seconds
        
    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()
        
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.timestamp


class Cacheable(ABC):
    """
    Mixin for caching support.
    
    Provides cache key generation, TTL management, and memory tracking.
    """
    
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.RLock()
        self._cache_config = {
            'default_ttl_days': 7,
            'max_cache_size_mb': 512,
            'cleanup_interval_seconds': 300,  # 5 minutes
            'max_entries': 1000,
            'enable_stats': True
        }
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0,
            'total_size_bytes': 0
        }
        
        # Start background cleanup if enabled
        self._cleanup_thread = None
        self._start_cleanup_thread()
        
        # Register for memory pressure
        self._register_cache_instance()
        
    def configure_cache(self, **config) -> None:
        """Configure cache behavior."""
        self._cache_config.update(config)
        
    def generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Create a deterministic key from arguments
        key_data = {
            'class': self.__class__.__name__,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else None
        }
        
        # Serialize and hash
        try:
            serialized = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
            key_hash = hashlib.md5(serialized).hexdigest()
            return f"{self.__class__.__name__}_{key_hash}"
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            # Fallback to string representation
            return f"{self.__class__.__name__}_{hash(str(key_data))}"
            
    def cache(self, 
              key: str, 
              value: Any, 
              ttl_days: Optional[float] = None,
              force: bool = False) -> None:
        """
        Cache a value with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_days: Time to live in days (None = default)
            force: Force cache even if at capacity
        """
        with self._cache_lock:
            ttl_days = ttl_days or self._cache_config.get('default_ttl_days', 7)
            ttl_seconds = ttl_days * 24 * 3600
            
            # Calculate size estimate
            size_bytes = self._estimate_size(value)
            
            # Check capacity
            if not force and not self._has_capacity(size_bytes):
                self._evict_entries()
                if not self._has_capacity(size_bytes):
                    logger.warning(f"Cannot cache item {key}: insufficient capacity")
                    return
                    
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._cache_stats['total_size_bytes'] -= old_entry.size_bytes
                
            # Create new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._cache_stats['total_size_bytes'] += size_bytes
            
            logger.debug(f"Cached {key} (size: {size_bytes} bytes, TTL: {ttl_days} days)")
            
    def get_cached(self, key: str) -> Optional[Any]:
        """
        Get cached value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._cache_lock:
            if key not in self._cache:
                self._cache_stats['misses'] += 1
                return None
                
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self._cache_stats['misses'] += 1
                return None
                
            # Update access stats
            entry.update_access()
            self._cache_stats['hits'] += 1
            
            return entry.value
            
    def is_cached(self, key: str) -> bool:
        """Check if key is cached and not expired."""
        with self._cache_lock:
            if key not in self._cache:
                return False
            return not self._cache[key].is_expired()
            
    def invalidate(self, key: str) -> bool:
        """
        Remove item from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if item was removed, False if not found
        """
        with self._cache_lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
            
    def clear_cache(self) -> None:
        """Clear all cached items."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_stats['total_size_bytes'] = 0
            logger.info("Cache cleared")
            
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._cache_lock:
            expired_keys = [
                key for key, entry in self._cache.items() 
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                
            if expired_keys:
                self._cache_stats['cleanups'] += 1
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
            return len(expired_keys)
            
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        with self._cache_lock:
            stats: Dict[str, Union[int, float]] = {}
            stats.update(self._cache_stats)
            stats['entries'] = len(self._cache)
            stats['size_mb'] = self._cache_stats['total_size_bytes'] / (1024 * 1024)
            stats['hit_rate'] = (
                self._cache_stats['hits'] / 
                max(1, self._cache_stats['hits'] + self._cache_stats['misses'])
            )
            return stats
            
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self._cache_lock:
            entries_info = []
            for key, entry in self._cache.items():
                entries_info.append({
                    'key': key,
                    'size_bytes': entry.size_bytes,
                    'age_seconds': entry.age_seconds,
                    'access_count': entry.access_count,
                    'expires_in_seconds': (
                        max(0, entry.ttl_seconds - entry.age_seconds) 
                        if entry.ttl_seconds > 0 else -1
                    )
                })
                
            return {
                'stats': self.get_cache_stats(),
                'config': self._cache_config.copy(),
                'entries': entries_info
            }
            
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes."""
        try:
            # Try pickle size as approximation
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimates
            if hasattr(value, '__sizeof__'):
                return value.__sizeof__()
            elif hasattr(value, 'nbytes'):  # numpy arrays
                return value.nbytes
            else:
                # Very rough estimate
                return len(str(value)) * 2
                
    def _has_capacity(self, additional_bytes: int) -> bool:
        """Check if cache has capacity for additional bytes."""
        max_size_bytes = self._cache_config.get('max_cache_size_mb', 512) * 1024 * 1024
        max_entries = self._cache_config.get('max_entries', 1000)
        
        size_ok = (self._cache_stats['total_size_bytes'] + additional_bytes) <= max_size_bytes
        count_ok = len(self._cache) < max_entries
        
        return size_ok and count_ok
        
    def _evict_entries(self) -> None:
        """Evict entries using LRU strategy."""
        with self._cache_lock:
            if not self._cache:
                return
                
            # Sort by last access time (LRU)
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_access
            )
            
            # Remove oldest 25% of entries
            num_to_remove = max(1, len(sorted_items) // 4)
            
            for i in range(num_to_remove):
                key = sorted_items[i][0]
                self._remove_entry(key)
                self._cache_stats['evictions'] += 1
                
            logger.debug(f"Evicted {num_to_remove} cache entries")
            
    def _remove_entry(self, key: str) -> None:
        """Remove a single cache entry."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._cache_stats['total_size_bytes'] -= entry.size_bytes
            
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    interval = self._cache_config.get('cleanup_interval_seconds', 300)
                    time.sleep(interval)
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
    def _register_cache_instance(self) -> None:
        """Register this instance for global cache management."""
        if not hasattr(Cacheable, '_instances'):
            Cacheable._instances = weakref.WeakSet()
        Cacheable._instances.add(self)
        
    @classmethod
    def get_global_cache_stats(cls) -> Dict[str, Any]:
        """Get cache statistics for all instances."""
        if not hasattr(cls, '_instances'):
            return {}
            
        global_stats = {
            'total_instances': 0,
            'total_entries': 0,
            'total_size_mb': 0,
            'total_hits': 0,
            'total_misses': 0
        }
        
        for instance in cls._instances:
            try:
                stats = instance.get_cache_stats()
                global_stats['total_instances'] += 1
                global_stats['total_entries'] += stats.get('entries', 0)
                global_stats['total_size_mb'] += stats.get('size_mb', 0)
                global_stats['total_hits'] += stats.get('hits', 0)
                global_stats['total_misses'] += stats.get('misses', 0)
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
                
        return global_stats
        
    @classmethod
    def clear_all_caches(cls) -> None:
        """Clear all caches across all instances."""
        if not hasattr(cls, '_instances'):
            return
            
        for instance in cls._instances:
            try:
                instance.clear_cache()
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                
    def on_memory_pressure(self, pressure_level: str) -> None:
        """Handle memory pressure by clearing cache."""
        if pressure_level in ['high', 'critical']:
            logger.info(f"Memory pressure ({pressure_level}), clearing cache")
            if pressure_level == 'critical':
                self.clear_cache()
            else:
                # Clear 50% of cache
                with self._cache_lock:
                    items = list(self._cache.items())
                    num_to_remove = len(items) // 2
                    sorted_items = sorted(items, key=lambda x: x[1].last_access)
                    
                    for i in range(num_to_remove):
                        key = sorted_items[i][0]
                        self._remove_entry(key)
