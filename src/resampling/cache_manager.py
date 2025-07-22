# src/resampling/cache_manager.py
"""Cache management for resampling results."""

import hashlib
import json
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import logging
from pathlib import Path

from ..database.schema import schema
from ..config.config import config

logger = logging.getLogger(__name__)


class ResamplingCacheManager:
    """Manage caching of resampling results."""
    
    def __init__(self):
        self.cache_dir = Path(config.get('raster_processing.cache_dir', 'cache/resampling'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_caching = config.get('raster_processing.database_caching', True)
    
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
                       band_number: int = 1) -> Dict[str, float]:
        """Retrieve cached resampling values from database."""
        if not self.db_caching:
            return {}
        
        try:
            return schema.get_cached_resampling_values(
                source_raster_id,
                target_grid_id,
                cell_ids,
                method,
                band_number
            )
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return {}
    
    def store_in_cache(self,
                       cache_entries: List[Dict[str, Any]]) -> int:
        """Store resampling results in cache."""
        if not self.db_caching:
            return 0
        
        try:
            return schema.store_resampling_cache_batch(cache_entries)
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
            return 0
    
    def warm_cache(self,
                   source_raster_id: str,
                   target_grid_id: str,
                   priority_bounds: Optional[Tuple[float, float, float, float]] = None,
                   method: str = 'bilinear') -> Dict[str, Any]:
        """Pre-populate cache for frequently accessed regions."""
        logger.info(f"Warming cache for raster {source_raster_id} -> grid {target_grid_id}")
        
        # Add to processing queue with high priority
        task_id = schema.add_processing_task(
            queue_type='resampling_cache_warm',
            raster_id=source_raster_id,
            grid_id=target_grid_id,
            parameters={
                'method': method,
                'bounds': priority_bounds,
                'priority_region': True
            },
            priority=10  # High priority
        )
        
        return {
            'task_id': task_id,
            'status': 'queued',
            'priority': 10
        }
    
    def get_cache_stats(self,
                        source_raster_id: Optional[str] = None,
                        target_grid_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cache efficiency statistics."""
        return schema.get_cache_efficiency_summary(source_raster_id, target_grid_id)
    
    def cleanup_old_cache(self, days_old: int = 30, min_access_count: int = 1) -> int:
        """Clean up old cache entries."""
        logger.info(f"Cleaning cache entries older than {days_old} days with < {min_access_count} accesses")
        return schema.cleanup_old_cache(days_old, min_access_count)
    
    def save_to_file(self, 
                     data: np.ndarray,
                     cache_key: str,
                     metadata: Dict[str, Any]) -> Path:
        """Save resampled data to file cache."""
        cache_path = self.cache_dir / f"{cache_key}.npz"
        
        np.savez_compressed(
            cache_path,
            data=data,
            metadata=json.dumps(metadata)
        )
        
        logger.debug(f"Saved to file cache: {cache_path}")
        return cache_path
    
    def load_from_file(self, cache_key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Load resampled data from file cache."""
        cache_path = self.cache_dir / f"{cache_key}.npz"
        
        if not cache_path.exists():
            return None
        
        try:
            with np.load(cache_path) as npz:
                data = npz['data']
                metadata = json.loads(npz['metadata'].item())
                
            logger.debug(f"Loaded from file cache: {cache_path}")
            return data, metadata
            
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_path}: {e}")
            return None