"""Cache monitoring and statistics service."""

import threading
import time
from typing import Dict, Any, Callable, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheMonitor:
    """Cache monitoring and statistics service."""
    
    def __init__(self, cache_dir: Path, monitoring_interval: int = 300):
        self.cache_dir = cache_dir
        self.monitoring_interval = monitoring_interval
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Cache statistics
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'validations': 0,
            'corrupted': 0,
            'total_size_mb': 0.0,
            'file_count': 0
        }
        self._stats_lock = threading.RLock()
    
    def start_monitoring(self):
        """Start background cache monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Cache monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_cache,
            name="CacheMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Cache monitoring started")
    
    def stop_monitoring(self):
        """Stop background cache monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
            logger.info("Cache monitoring stopped")
    
    def record_cache_hit(self):
        """Record cache hit."""
        with self._stats_lock:
            self._cache_stats['hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        with self._stats_lock:
            self._cache_stats['misses'] += 1
    
    def record_cache_store(self):
        """Record cache store operation."""
        with self._stats_lock:
            self._cache_stats['stores'] += 1
    
    def record_validation(self):
        """Record cache validation."""
        with self._stats_lock:
            self._cache_stats['validations'] += 1
    
    def record_corruption(self, count: int = 1):
        """Record cache corruption."""
        with self._stats_lock:
            self._cache_stats['corrupted'] += count
    
    def get_cache_stats(self, include_computed: bool = True) -> Dict[str, Any]:
        """Get current cache statistics."""
        with self._stats_lock:
            stats = self._cache_stats.copy()
        
        if include_computed:
            total_requests = stats['hits'] + stats['misses']
            stats['hit_rate'] = (stats['hits'] / total_requests) if total_requests > 0 else 0.0
            stats['miss_rate'] = (stats['misses'] / total_requests) if total_requests > 0 else 0.0
            
            # Update size and count from filesystem
            stats.update(self._get_filesystem_stats())
        
        return stats
    
    def _monitor_cache(self):
        """Background cache monitoring loop."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                # Update filesystem stats
                fs_stats = self._get_filesystem_stats()
                with self._stats_lock:
                    self._cache_stats.update(fs_stats)
                
                # Log periodic stats
                stats = self.get_cache_stats(include_computed=True)
                logger.info(
                    f"Cache stats: {stats['hit_rate']:.1%} hit rate, "
                    f"{stats['file_count']} files, {stats['total_size_mb']:.1f}MB"
                )
                
                # Check for issues
                if stats['corrupted'] > 0:
                    logger.warning(f"Cache corruption detected: {stats['corrupted']} entries")
                
            except Exception as e:
                logger.error(f"Cache monitoring error: {e}")
    
    def _get_filesystem_stats(self) -> Dict[str, Any]:
        """Get filesystem-based cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            file_count = len(cache_files)
            
            total_size_bytes = sum(f.stat().st_size for f in cache_files)
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            return {
                'file_count': file_count,
                'total_size_mb': total_size_mb
            }
            
        except Exception as e:
            logger.error(f"Failed to get filesystem stats: {e}")
            return {
                'file_count': 0,
                'total_size_mb': 0.0
            }
    
    def warm_cache_monitoring(self, task_id: str, progress_callback: Optional[Callable[[str, float], None]] = None):
        """Monitor cache warming task progress."""
        if not progress_callback:
            return
        
        def monitor():
            try:
                # This would monitor the actual warming task
                # For now, simulate monitoring
                start_time = time.time()
                max_duration = 300  # 5 minutes max
                
                while time.time() - start_time < max_duration:
                    elapsed = time.time() - start_time
                    progress = min(elapsed / max_duration, 1.0)
                    
                    progress_callback(task_id, progress * 100)
                    
                    if progress >= 1.0:
                        break
                    
                    time.sleep(5)
                
                logger.info(f"Cache warming monitoring complete for task {task_id}")
                
            except Exception as e:
                logger.error(f"Cache warming monitoring failed for {task_id}: {e}")
        
        monitor_thread = threading.Thread(
            target=monitor,
            name=f"CacheWarmMonitor-{task_id}",
            daemon=True
        )
        monitor_thread.start()
        
        return monitor_thread