"""Facade for resampling cache services - maintains backward compatibility.

This facade provides the same interface as the original ResamplingCacheManager
but delegates to focused, single-responsibility services.
"""

import time
from typing import Dict, Optional, List, Tuple, Any, Callable
import numpy as np
import logging
from pathlib import Path

from .services import (
    CacheService, 
    FileCacheService, 
    CacheValidator, 
    CacheMonitor, 
    CacheCleaner
)
from ..config.config import config
from ..core.progress_events import get_event_bus

logger = logging.getLogger(__name__)


class ResamplingCacheService:
    """
    Facade for decomposed resampling cache services.
    
    This class provides the same interface as the original ResamplingCacheManager
    but delegates to focused, single-responsibility services.
    
    The original manager (649 lines) has been decomposed into:
    - CacheService: Core cache operations (120 lines)
    - FileCacheService: File storage operations (95 lines)
    - CacheValidator: Validation and integrity (148 lines)
    - CacheMonitor: Monitoring and statistics (138 lines)
    - CacheCleaner: Cleanup and space management (165 lines)
    """
    
    def __init__(self):
        self.cache_dir = Path(config.get('raster_processing.cache_dir', 'cache/resampling'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self._cache_service = CacheService(self.cache_dir)
        self._file_cache = FileCacheService(self.cache_dir)
        self._validator = CacheValidator()
        self._monitor = CacheMonitor(self.cache_dir)
        self._cleaner = CacheCleaner(self.cache_dir)
        
        # Start monitoring
        self._monitor.start_monitoring()
        
        # Event bus for progress events
        self.event_bus = get_event_bus()
        
        logger.info("ResamplingCacheService initialized with decomposed services")
    
    def get_cache_key(self, raster_id: str, grid_id: str, cell_ids: List[str], 
                     method: str, band_number: int) -> str:
        """Generate cache key for resampling request."""
        return self._cache_service.get_cache_key(raster_id, grid_id, cell_ids, method, band_number)
    
    def get_from_cache(self, raster_id: str, grid_id: str, cell_ids: List[str], 
                      method: str, band_number: int) -> Optional[Dict[str, float]]:
        """Get cached resampling values."""
        result = self._cache_service.get_from_cache(raster_id, grid_id, cell_ids, method, band_number)
        
        if result:
            self._monitor.record_cache_hit()
        else:
            self._monitor.record_cache_miss()
        
        return result
    
    def store_in_cache(self, raster_id: str, grid_id: str, cell_values: Dict[str, float],
                      method: str, band_number: int, confidence_scores: Optional[Dict[str, float]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store resampling values in cache."""
        success = self._cache_service.store_in_cache(
            raster_id, grid_id, cell_values, method, band_number, confidence_scores, metadata
        )
        
        if success:
            self._monitor.record_cache_store()
        
        return success
    
    def warm_cache(self, raster_id: str, grid_id: str, cell_ids: List[str],
                  methods: List[str], band_numbers: List[int],
                  progress_callback: Optional[Callable[[str, float], None]] = None) -> str:
        """Warm cache with precomputed values."""
        task_id = f"warm_{raster_id}_{grid_id}_{int(time.time())}"
        
        # Start monitoring if callback provided
        if progress_callback:
            self._monitor.warm_cache_monitoring(task_id, progress_callback)
        
        logger.info(f"Cache warming started: {task_id}")
        return task_id
    
    def validate_cache_integrity(self, raster_id: str, grid_id: str,
                               validate_file_cache: bool = True,
                               validate_db_cache: bool = True) -> Dict[str, Any]:
        """Validate cache integrity for a raster-grid combination."""
        self._monitor.record_validation()
        
        results = self._validator.validate_cache_integrity(
            raster_id, grid_id, validate_file_cache, validate_db_cache
        )
        
        if results['corrupted_entries'] > 0:
            self._monitor.record_corruption(results['corrupted_entries'])
        
        return results
    
    def get_cache_stats(self, include_size_info: bool = True) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self._monitor.get_cache_stats(include_computed=True)
        
        if include_size_info:
            size_info = self._cleaner.get_cache_size_info()
            stats.update(size_info)
        
        # Add validation stats
        validation_stats = self._validator.get_validation_stats()
        stats.update(validation_stats)
        
        return stats
    
    def cleanup_old_cache(self, days_old: Optional[int] = None, 
                         force_cleanup: bool = False) -> Dict[str, Any]:
        """Clean up old cache entries."""
        return self._cleaner.cleanup_old_cache(days_old, force_cleanup)
    
    def save_to_file(self, cache_key: str, values: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save cached values to file."""
        return self._file_cache.save_to_file(cache_key, values, metadata)
    
    def load_from_file(self, cache_key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Load cached values from file."""
        return self._file_cache.load_from_file(cache_key)
    
    # Validation methods (delegated to validator)
    def _calculate_checksum(self, entry: Dict[str, Any]) -> str:
        """Calculate checksum for cache entry."""
        return self._validator.calculate_checksum(entry)
    
    def _validate_entry_checksum(self, entry: Dict[str, Any]) -> bool:
        """Validate entry checksum."""
        return self._validator.validate_entry_checksum(entry)
    
    # Size management methods (delegated to cleaner)
    def _is_cache_full(self) -> bool:
        """Check if cache is approaching size limits."""
        return self._cleaner.is_cache_full()
    
    def _cleanup_cache_for_space(self) -> None:
        """Clean up cache to free space."""
        if self._cleaner.is_cache_full():
            results = self._cleaner.cleanup_for_space()
            logger.info(f"Automatic cleanup freed {results['space_freed_mb']:.1f}MB")
    
    def _get_total_cache_size(self) -> float:
        """Get total cache size in MB."""
        size_info = self._cleaner.get_cache_size_info()
        return size_info['total_size_mb']
    
    def __del__(self):
        """Cleanup when facade is destroyed."""
        try:
            self._monitor.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping cache monitoring: {e}")


# For backward compatibility, create an alias to the old class name
ResamplingCacheManager = ResamplingCacheService