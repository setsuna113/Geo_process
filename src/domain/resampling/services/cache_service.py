"""Core cache service for resampling operations."""

import hashlib
import json
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import logging
from pathlib import Path

from src.database.schema import schema
from src.config import config

logger = logging.getLogger(__name__)


class CacheService:
    """Core cache management service for resampling operations."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(config.get('raster_processing.cache_dir', 'cache/resampling'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_caching = config.get('raster_processing.database_caching', True)
        
        # Cache configuration
        self.cache_config = {
            'validate_on_read': config.get('raster_processing.validate_cache_on_read', True),
            'compression': config.get('raster_processing.cache_compression', 'lz4')
        }
    
    def get_cache_key(self, raster_id: str, grid_id: str, cell_ids: List[str], 
                     method: str, band_number: int) -> str:
        """Generate cache key for resampling request."""
        key_data = {
            'raster_id': raster_id,
            'grid_id': grid_id,
            'cell_ids': sorted(cell_ids),
            'method': method,
            'band_number': band_number
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get_from_cache(self, raster_id: str, grid_id: str, cell_ids: List[str], 
                      method: str, band_number: int) -> Optional[Dict[str, float]]:
        """Get cached resampling values."""
        cache_key = self.get_cache_key(raster_id, grid_id, cell_ids, method, band_number)
        
        # Try database cache first
        if self.db_caching:
            try:
                cached_values = schema.get_cached_resampling_values(
                    raster_id, grid_id, cell_ids, method, band_number
                )
                if cached_values:
                    logger.debug(f"✅ Database cache hit for {cache_key}")
                    return cached_values
            except Exception as e:
                logger.warning(f"Database cache read failed: {e}")
        
        # Try file cache
        try:
            from .file_cache_service import FileCacheService
            file_cache = FileCacheService(self.cache_dir)
            cached_data = file_cache.load_from_file(cache_key)
            if cached_data:
                values, metadata = cached_data
                logger.debug(f"✅ File cache hit for {cache_key}")
                
                # Convert numpy array back to cell_id -> value mapping
                result = {}
                for i, cell_id in enumerate(cell_ids):
                    if i < len(values):
                        result[cell_id] = float(values[i])
                
                return result
        except Exception as e:
            logger.warning(f"File cache read failed: {e}")
        
        logger.debug(f"❌ Cache miss for {cache_key}")
        return None
    
    def store_in_cache(self, raster_id: str, grid_id: str, cell_values: Dict[str, float],
                      method: str, band_number: int, confidence_scores: Optional[Dict[str, float]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store resampling values in cache."""
        cell_ids = list(cell_values.keys())
        cache_key = self.get_cache_key(raster_id, grid_id, cell_ids, method, band_number)
        
        success = False
        
        # Store in database cache
        if self.db_caching:
            try:
                cache_data = []
                for cell_id, value in cell_values.items():
                    cache_data.append({
                        'source_raster_id': raster_id,
                        'target_grid_id': grid_id,
                        'cell_id': cell_id,
                        'method': method,
                        'band_number': band_number,
                        'value': value,
                        'confidence_score': confidence_scores.get(cell_id, 1.0) if confidence_scores else 1.0,
                        'computation_metadata': metadata or {}
                    })
                
                schema.store_resampling_cache_batch(cache_data)
                logger.debug(f"✅ Stored {len(cache_data)} values in database cache")
                success = True
                
            except Exception as e:
                logger.error(f"Failed to store in database cache: {e}")
        
        # Store in file cache
        try:
            from .file_cache_service import FileCacheService
            file_cache = FileCacheService(self.cache_dir)
            
            # Convert to numpy array for efficient storage
            values_array = np.array([cell_values[cell_id] for cell_id in cell_ids])
            file_metadata = {
                'cell_ids': cell_ids,
                'raster_id': raster_id,
                'grid_id': grid_id,
                'method': method,
                'band_number': band_number,
                'timestamp': int(time.time()),
                'confidence_scores': confidence_scores,
                'metadata': metadata
            }
            
            file_cache.save_to_file(cache_key, values_array, file_metadata)
            logger.debug(f"✅ Stored values in file cache: {cache_key}")
            success = True
            
        except Exception as e:
            logger.error(f"Failed to store in file cache: {e}")
        
        return success