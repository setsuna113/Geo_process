# src/resampling/cache_manager.py
"""Enhanced cache management for resampling with validation and progress support."""

import hashlib
import json
from typing import Dict, Optional, List, Tuple, Any, Callable
import numpy as np
import logging
from pathlib import Path
import time
import threading
from datetime import datetime, timedelta

from ..database.schema import schema
from ..config.config import config
from ..core.progress_events import get_event_bus, create_processing_progress, EventType

logger = logging.getLogger(__name__)


class ResamplingCacheManager:
    """Enhanced cache manager with validation, recovery, and monitoring."""
    
    def __init__(self):
        self.cache_dir = Path(config.get('raster_processing.cache_dir', 'cache/resampling'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_caching = config.get('raster_processing.database_caching', True)
        self.event_bus = get_event_bus()
        
        # Cache configuration
        self.cache_config = {
            'max_cache_size_gb': config.get('raster_processing.max_cache_size_gb', 10),
            'max_cache_age_days': config.get('raster_processing.max_cache_age_days', 30),
            'validate_on_read': config.get('raster_processing.validate_cache_on_read', True),
            'compression': config.get('raster_processing.cache_compression', 'lz4')
        }
        
        # Cache statistics
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'validations': 0,
            'corrupted': 0,
            'total_size_mb': 0
        }
        self._stats_lock = threading.RLock()
        
        # Initialize cache monitoring
        self._start_cache_monitoring()
    
    def get_cache_key(self, 
                      source_id: str,
                      target_resolution: float,
                      method: str,
                      bounds: Optional[Tuple[float, float, float, float]] = None) -> str:
        """Generate cache key for resampling operation."""
        key_parts = [
            source_id,
            f"{target_resolution:.6f}",
            method,
            json.dumps(bounds) if bounds else "full"
        ]
        
        key_string = "_".join(str(p) for p in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_from_cache(self,
                       source_raster_id: str,
                       target_grid_id: str,
                       cell_ids: List[str],
                       method: str,
                       band_number: int = 1,
                       progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, float]:
        """
        Retrieve cached resampling values with validation.
        
        Args:
            source_raster_id: Source raster ID
            target_grid_id: Target grid ID
            cell_ids: List of cell IDs to retrieve
            method: Resampling method
            band_number: Band number
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary of cell_id -> value
        """
        if not self.db_caching:
            return {}
        
        try:
            # Progress reporting
            if progress_callback:
                progress_callback("Checking cache", 10)
            
            # Publish cache event
            if self.event_bus:
                event = create_processing_progress(
                    operation_name="cache_lookup",
                    processed=0,
                    total=len(cell_ids),
                    source=self.__class__.__name__
                )
                self.event_bus.publish(event)
            
            # Retrieve from database
            cached_values = schema.get_cached_resampling_values(
                source_raster_id,
                target_grid_id,
                cell_ids,
                method,
                band_number
            )
            
            # Update statistics
            with self._stats_lock:
                if cached_values:
                    self._cache_stats['hits'] += len(cached_values)
                else:
                    self._cache_stats['misses'] += len(cell_ids)
            
            # Validate if configured
            if self.cache_config['validate_on_read'] and cached_values:
                if progress_callback:
                    progress_callback("Validating cache", 50)
                
                validated_values = self._validate_cached_values(
                    cached_values,
                    source_raster_id,
                    method
                )
                
                # Update statistics
                corrupted = len(cached_values) - len(validated_values)
                if corrupted > 0:
                    with self._stats_lock:
                        self._cache_stats['corrupted'] += corrupted
                    logger.warning(f"Found {corrupted} corrupted cache entries")
                
                if progress_callback:
                    progress_callback("Cache validated", 100)
                
                return validated_values
            
            return cached_values
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            with self._stats_lock:
                self._cache_stats['misses'] += len(cell_ids)
            return {}
    
    def store_in_cache(self,
                       cache_entries: List[Dict[str, Any]],
                       progress_callback: Optional[Callable[[str, float], None]] = None) -> int:
        """
        Store resampling results with progress reporting.
        
        Args:
            cache_entries: List of cache entries to store
            progress_callback: Optional progress callback
            
        Returns:
            Number of entries stored
        """
        if not self.db_caching or not cache_entries:
            return 0
        
        try:
            total_entries = len(cache_entries)
            
            if progress_callback:
                progress_callback("Preparing cache entries", 10)
            
            # Add timestamps and validation checksums
            for entry in cache_entries:
                entry['cached_at'] = datetime.now()
                entry['checksum'] = self._calculate_checksum(entry)
            
            # Check cache size limits
            if self._is_cache_full():
                if progress_callback:
                    progress_callback("Cleaning old cache entries", 20)
                
                self._cleanup_cache_for_space()
            
            # Store in batches for progress reporting
            batch_size = 1000
            stored_count = 0
            
            for i in range(0, total_entries, batch_size):
                batch = cache_entries[i:i + batch_size]
                
                if progress_callback:
                    progress_pct = 20 + (i / total_entries) * 70
                    progress_callback(f"Storing batch {i//batch_size + 1}", progress_pct)
                
                stored = schema.store_resampling_cache_batch(batch)
                stored_count += stored
                
                # Publish progress event
                if self.event_bus:
                    event = create_processing_progress(
                        operation_name="cache_store",
                        processed=i + len(batch),
                        total=total_entries,
                        source=self.__class__.__name__
                    )
                    self.event_bus.publish(event)
            
            # Update statistics
            with self._stats_lock:
                self._cache_stats['total_size_mb'] += sum(
                    self._estimate_entry_size(e) for e in cache_entries
                ) / (1024 * 1024)
            
            if progress_callback:
                progress_callback("Cache storage complete", 100)
            
            logger.info(f"Stored {stored_count} entries in cache")
            return stored_count
            
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
            return 0
    
    def warm_cache(self,
                   source_raster_id: str,
                   target_grid_id: str,
                   priority_bounds: Optional[Tuple[float, float, float, float]] = None,
                   method: str = 'bilinear',
                   progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Pre-populate cache with progress tracking.
        
        Args:
            source_raster_id: Source raster ID
            target_grid_id: Target grid ID
            priority_bounds: Priority region bounds
            method: Resampling method
            progress_callback: Progress callback
            
        Returns:
            Task information
        """
        logger.info(f"Warming cache for raster {source_raster_id} -> grid {target_grid_id}")
        
        if progress_callback:
            progress_callback("Initializing cache warming task", 10)
        
        # Add to processing queue
        task_id = schema.add_processing_task(
            queue_type='resampling_cache_warm',
            raster_id=source_raster_id,
            grid_id=target_grid_id,
            parameters={
                'method': method,
                'bounds': priority_bounds,
                'priority_region': True
            },
            priority=10
        )
        
        # Monitor task progress in background
        if progress_callback:
            self._monitor_warm_task(task_id, progress_callback)
        
        return {
            'task_id': task_id,
            'status': 'queued',
            'priority': 10,
            'estimated_entries': self._estimate_cache_entries(
                source_raster_id, target_grid_id, priority_bounds
            )
        }
    
    def validate_cache_integrity(self,
                               source_raster_id: Optional[str] = None,
                               progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        Validate cache integrity with progress reporting.
        
        Args:
            source_raster_id: Optional source raster to validate
            progress_callback: Progress callback
            
        Returns:
            Validation report
        """
        logger.info("Starting cache integrity validation")
        
        validation_report = {
            'total_entries': 0,
            'valid_entries': 0,
            'corrupted_entries': 0,
            'removed_entries': 0,
            'validation_time_seconds': 0
        }
        
        start_time = time.time()
        
        try:
            # Get all cache entries
            if progress_callback:
                progress_callback("Loading cache entries", 10)
            
            entries = schema.get_all_cache_entries(source_raster_id)
            validation_report['total_entries'] = len(entries)
            
            # Validate in batches
            batch_size = 1000
            corrupted_ids = []
            
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                
                if progress_callback:
                    progress_pct = 10 + (i / len(entries)) * 80
                    progress_callback(f"Validating batch {i//batch_size + 1}", progress_pct)
                
                # Validate checksums
                for entry in batch:
                    if not self._validate_entry_checksum(entry):
                        corrupted_ids.append(entry['id'])
                
                # Update statistics
                with self._stats_lock:
                    self._cache_stats['validations'] += len(batch)
            
            validation_report['corrupted_entries'] = len(corrupted_ids)
            validation_report['valid_entries'] = len(entries) - len(corrupted_ids)
            
            # Remove corrupted entries
            if corrupted_ids:
                if progress_callback:
                    progress_callback("Removing corrupted entries", 90)
                
                removed = schema.remove_cache_entries(corrupted_ids)
                validation_report['removed_entries'] = removed
                
                logger.warning(f"Removed {removed} corrupted cache entries")
            
            validation_report['validation_time_seconds'] = time.time() - start_time
            
            if progress_callback:
                progress_callback("Validation complete", 100)
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")
            validation_report['error'] = str(e)
            return validation_report
    
    def get_cache_stats(self,
                        source_raster_id: Optional[str] = None,
                        target_grid_id: Optional[str] = None) -> Dict[str, Any]:
        """Get enhanced cache statistics."""
        # Get database statistics
        db_stats = schema.get_cache_efficiency_summary(source_raster_id, target_grid_id)
        
        # Combine with local statistics
        with self._stats_lock:
            combined_stats = {
                'database_stats': db_stats,
                'session_stats': self._cache_stats.copy(),
                'cache_config': self.cache_config.copy(),
                'cache_directory': str(self.cache_dir),
                'file_cache_size_mb': self._get_file_cache_size()
            }
        
        # Calculate hit rate
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        if total_requests > 0:
            combined_stats['session_stats']['hit_rate'] = (
                self._cache_stats['hits'] / total_requests * 100
            )
        
        return combined_stats
    
    def cleanup_old_cache(self, 
                         days_old: int = 30, 
                         min_access_count: int = 1,
                         progress_callback: Optional[Callable[[str, float], None]] = None) -> int:
        """
        Enhanced cache cleanup with progress reporting.
        
        Args:
            days_old: Age threshold in days
            min_access_count: Minimum access count to keep
            progress_callback: Progress callback
            
        Returns:
            Number of entries cleaned
        """
        logger.info(f"Cleaning cache entries older than {days_old} days")
        
        if progress_callback:
            progress_callback("Analyzing cache entries", 10)
        
        # Clean database cache
        db_cleaned = schema.cleanup_old_cache(days_old, min_access_count)
        
        if progress_callback:
            progress_callback("Cleaning file cache", 50)
        
        # Clean file cache
        file_cleaned = self._cleanup_file_cache(days_old)
        
        total_cleaned = db_cleaned + file_cleaned
        
        # Update statistics
        with self._stats_lock:
            self._cache_stats['total_size_mb'] = self._get_total_cache_size()
        
        if progress_callback:
            progress_callback("Cleanup complete", 100)
        
        logger.info(f"Cleaned {total_cleaned} cache entries")
        return total_cleaned
    
    def save_to_file(self, 
                     data: np.ndarray,
                     cache_key: str,
                     metadata: Dict[str, Any]) -> Path:
        """Enhanced file cache saving with compression."""
        cache_path = self.cache_dir / f"{cache_key}.npz"
        
        # Add timestamp and checksum to metadata
        metadata['saved_at'] = datetime.now().isoformat()
        metadata['checksum'] = hashlib.md5(data.tobytes()).hexdigest()
        
        # Save with compression based on configuration
        if self.cache_config['compression'] == 'lz4':
            try:
                import lz4.frame
                compressed = lz4.frame.compress(data.tobytes())
                cache_path = cache_path.with_suffix('.lz4')
                
                with open(cache_path, 'wb') as f:
                    f.write(compressed)
                
                # Save metadata separately
                meta_path = cache_path.with_suffix('.meta.json')
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                    
            except ImportError:
                # Fallback to numpy compression
                np.savez_compressed(
                    cache_path,
                    data=data,
                    metadata=json.dumps(metadata)
                )
        else:
            np.savez_compressed(
                cache_path,
                data=data,
                metadata=json.dumps(metadata)
            )
        
        logger.debug(f"Saved to file cache: {cache_path}")
        return cache_path
    
    def load_from_file(self, cache_key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Enhanced file cache loading with validation."""
        # Try different file extensions
        for ext in ['.lz4', '.npz']:
            cache_path = self.cache_dir / f"{cache_key}{ext}"
            
            if not cache_path.exists():
                continue
            
            try:
                if ext == '.lz4':
                    import lz4.frame
                    
                    # Load compressed data
                    with open(cache_path, 'rb') as f:
                        compressed = f.read()
                    
                    # Load metadata
                    meta_path = cache_path.with_suffix('.meta.json')
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Decompress data
                    decompressed = lz4.frame.decompress(compressed)
                    
                    # Reconstruct array from metadata
                    shape = tuple(metadata['shape'])
                    dtype = np.dtype(metadata['dtype'])
                    data = np.frombuffer(decompressed, dtype=dtype).reshape(shape)
                    
                else:
                    # Load numpy compressed
                    with np.load(cache_path) as npz:
                        data = npz['data']
                        metadata = json.loads(npz['metadata'].item())
                
                # Validate if configured
                if self.cache_config['validate_on_read']:
                    stored_checksum = metadata.get('checksum')
                    if stored_checksum:
                        actual_checksum = hashlib.md5(data.tobytes()).hexdigest()
                        if stored_checksum != actual_checksum:
                            logger.warning(f"Cache file corrupted: {cache_path}")
                            cache_path.unlink()  # Remove corrupted file
                            return None
                
                logger.debug(f"Loaded from file cache: {cache_path}")
                return data, metadata
                
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_path}: {e}")
                continue
        
        return None
    
    def _validate_cached_values(self,
                              cached_values: Dict[str, float],
                              source_raster_id: str,
                              method: str) -> Dict[str, float]:
        """Validate cached values for consistency."""
        validated = {}
        
        for cell_id, value in cached_values.items():
            # Basic validation checks
            if np.isfinite(value):  # Check for NaN or Inf
                validated[cell_id] = value
            else:
                logger.warning(f"Invalid cached value for cell {cell_id}: {value}")
        
        return validated
    
    def _calculate_checksum(self, entry: Dict[str, Any]) -> str:
        """Calculate checksum for cache entry."""
        # Create deterministic string from entry
        checksum_data = f"{entry.get('source_raster_id', '')}_{entry.get('cell_id', '')}_{entry.get('value', '')}"
        return hashlib.md5(checksum_data.encode()).hexdigest()
    
    def _validate_entry_checksum(self, entry: Dict[str, Any]) -> bool:
        """Validate entry checksum."""
        stored_checksum = entry.get('checksum')
        if not stored_checksum:
            return True  # No checksum to validate
        
        actual_checksum = self._calculate_checksum(entry)
        return stored_checksum == actual_checksum
    
    def _estimate_entry_size(self, entry: Dict[str, Any]) -> int:
        """Estimate size of cache entry in bytes."""
        # Rough estimate based on data
        return len(json.dumps(entry))
    
    def _estimate_cache_entries(self,
                              source_raster_id: str,
                              target_grid_id: str,
                              bounds: Optional[Tuple[float, float, float, float]]) -> int:
        """Estimate number of cache entries for operation."""
        # This would query the grid system for cell count
        # Simplified implementation
        return 10000  # Placeholder
    
    def _is_cache_full(self) -> bool:
        """Check if cache is full."""
        total_size_gb = self._get_total_cache_size() / 1024
        return total_size_gb >= self.cache_config['max_cache_size_gb']
    
    def _cleanup_cache_for_space(self) -> None:
        """Clean up cache to make space."""
        # Remove oldest entries first
        removed = schema.cleanup_old_cache(
            days_old=1,  # Start with 1 day old
            min_access_count=1
        )
        logger.info(f"Removed {removed} entries to make cache space")
    
    def _get_file_cache_size(self) -> float:
        """Get total size of file cache in MB."""
        total_size = 0
        
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                total_size += cache_file.stat().st_size
        
        return total_size / (1024 * 1024)
    
    def _get_total_cache_size(self) -> float:
        """Get total cache size in MB."""
        return self._get_file_cache_size()  # Simplified - would also query DB
    
    def _cleanup_file_cache(self, days_old: int) -> int:
        """Clean up old file cache entries."""
        cleaned = 0
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file() and cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                cleaned += 1
        
        return cleaned
    
    def _monitor_warm_task(self, task_id: str, progress_callback: Callable[[str, float], None]) -> None:
        """Monitor cache warming task progress."""
        def monitor():
            while True:
                task_status = schema.get_task_status(task_id)
                if not task_status:
                    break
                
                if task_status['status'] == 'completed':
                    progress_callback("Cache warming complete", 100)
                    break
                elif task_status['status'] == 'failed':
                    progress_callback("Cache warming failed", 100)
                    break
                else:
                    progress = task_status.get('progress', 0)
                    progress_callback(f"Cache warming: {progress}%", 10 + progress * 0.9)
                
                time.sleep(2)
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _start_cache_monitoring(self) -> None:
        """Start background cache monitoring."""
        def monitor():
            while True:
                try:
                    # Periodic cache validation
                    if self.cache_config.get('auto_validate', False):
                        self.validate_cache_integrity()
                    
                    # Check cache size
                    if self._is_cache_full():
                        self._cleanup_cache_for_space()
                    
                    # Sleep for monitoring interval
                    time.sleep(3600)  # Check every hour
                    
                except Exception as e:
                    logger.error(f"Cache monitoring error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()