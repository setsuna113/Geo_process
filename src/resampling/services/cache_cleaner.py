"""Cache cleanup service for resampling operations."""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from pathlib import Path

from ...config.config import config

logger = logging.getLogger(__name__)


class CacheCleaner:
    """Cache cleanup and space management service."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        
        # Cleanup configuration
        self.cleanup_config = {
            'max_cache_size_gb': config.get('raster_processing.max_cache_size_gb', 10),
            'max_cache_age_days': config.get('raster_processing.max_cache_age_days', 30),
            'cleanup_threshold': 0.8,  # Start cleanup at 80% of max size
            'target_usage': 0.6        # Clean down to 60% of max size
        }
    
    def cleanup_old_cache(self, days_old: Optional[int] = None, 
                         force_cleanup: bool = False) -> Dict[str, Any]:
        """Clean up old cache entries."""
        days_old = days_old or self.cleanup_config['max_cache_age_days']
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        results = {
            'files_removed': 0,
            'space_freed_mb': 0.0,
            'files_kept': 0,
            'errors': []
        }
        
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            
            for cache_file in cache_files:
                try:
                    file_mtime = cache_file.stat().st_mtime
                    file_size = cache_file.stat().st_size
                    
                    if force_cleanup or file_mtime < cutoff_time:
                        cache_file.unlink()
                        results['files_removed'] += 1
                        results['space_freed_mb'] += file_size / (1024 * 1024)
                        logger.debug(f"Removed old cache file: {cache_file.name}")
                    else:
                        results['files_kept'] += 1
                        
                except Exception as e:
                    error_msg = f"Failed to process {cache_file.name}: {e}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(
                f"Cache cleanup complete: removed {results['files_removed']} files, "
                f"freed {results['space_freed_mb']:.1f}MB"
            )
            
        except Exception as e:
            error_msg = f"Cache cleanup failed: {e}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def cleanup_for_space(self) -> Dict[str, Any]:
        """Clean up cache to free space based on size limits."""
        max_size_bytes = self.cleanup_config['max_cache_size_gb'] * 1024 * 1024 * 1024
        threshold_bytes = max_size_bytes * self.cleanup_config['cleanup_threshold']
        target_bytes = max_size_bytes * self.cleanup_config['target_usage']
        
        results = {
            'files_removed': 0,
            'space_freed_mb': 0.0,
            'current_size_mb': 0.0,
            'target_achieved': False
        }
        
        try:
            # Get all cache files with their stats
            cache_files = []
            total_size = 0
            
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    stat = cache_file.stat()
                    cache_files.append({
                        'path': cache_file,
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'atime': stat.st_atime
                    })
                    total_size += stat.st_size
                except Exception as e:
                    logger.warning(f"Failed to stat {cache_file}: {e}")
            
            results['current_size_mb'] = total_size / (1024 * 1024)
            
            # Check if cleanup is needed
            if total_size < threshold_bytes:
                logger.debug(f"Cache size {results['current_size_mb']:.1f}MB below threshold")
                results['target_achieved'] = True
                return results
            
            # Sort by access time (least recently accessed first)
            cache_files.sort(key=lambda x: x['atime'])
            
            # Remove files until we reach target size
            for file_info in cache_files:
                if total_size <= target_bytes:
                    results['target_achieved'] = True
                    break
                
                try:
                    file_info['path'].unlink()
                    results['files_removed'] += 1
                    results['space_freed_mb'] += file_info['size'] / (1024 * 1024)
                    total_size -= file_info['size']
                    
                    logger.debug(f"Removed cache file for space: {file_info['path'].name}")
                    
                except Exception as e:
                    logger.error(f"Failed to remove {file_info['path']}: {e}")
            
            final_size_mb = total_size / (1024 * 1024)
            logger.info(
                f"Space cleanup complete: removed {results['files_removed']} files, "
                f"freed {results['space_freed_mb']:.1f}MB, "
                f"current size: {final_size_mb:.1f}MB"
            )
            
        except Exception as e:
            logger.error(f"Space cleanup failed: {e}")
        
        return results
    
    def is_cache_full(self) -> bool:
        """Check if cache is approaching size limits."""
        try:
            max_size_bytes = self.cleanup_config['max_cache_size_gb'] * 1024 * 1024 * 1024
            threshold_bytes = max_size_bytes * self.cleanup_config['cleanup_threshold']
            
            total_size = sum(
                f.stat().st_size 
                for f in self.cache_dir.glob("*.cache")
            )
            
            return total_size >= threshold_bytes
            
        except Exception as e:
            logger.error(f"Failed to check cache size: {e}")
            return False
    
    def get_cache_size_info(self) -> Dict[str, Any]:
        """Get detailed cache size information."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            max_size_bytes = self.cleanup_config['max_cache_size_gb'] * 1024 * 1024 * 1024
            
            return {
                'file_count': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'total_size_gb': total_size / (1024 * 1024 * 1024),
                'max_size_gb': self.cleanup_config['max_cache_size_gb'],
                'usage_percent': (total_size / max_size_bytes * 100) if max_size_bytes > 0 else 0,
                'cleanup_needed': total_size >= (max_size_bytes * self.cleanup_config['cleanup_threshold'])
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache size info: {e}")
            return {
                'file_count': 0,
                'total_size_mb': 0.0,
                'total_size_gb': 0.0,
                'max_size_gb': self.cleanup_config['max_cache_size_gb'],
                'usage_percent': 0.0,
                'cleanup_needed': False
            }