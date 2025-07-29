"""Cache validation service for resampling operations."""

import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
import threading

logger = logging.getLogger(__name__)


class CacheValidator:
    """Cache validation and integrity checking service."""
    
    def __init__(self):
        self._validation_stats = {
            'validations': 0,
            'corrupted': 0,
            'repaired': 0
        }
        self._stats_lock = threading.RLock()
    
    def validate_cache_integrity(self, raster_id: str, grid_id: str, 
                               validate_file_cache: bool = True,
                               validate_db_cache: bool = True) -> Dict[str, Any]:
        """Validate cache integrity for a raster-grid combination."""
        with self._stats_lock:
            self._validation_stats['validations'] += 1
        
        results = {
            'total_entries': 0,
            'valid_entries': 0,
            'corrupted_entries': 0,
            'repaired_entries': 0,
            'errors': []
        }
        
        # Validate database cache
        if validate_db_cache:
            try:
                db_results = self._validate_database_cache(raster_id, grid_id)
                results['db_cache'] = db_results
                results['total_entries'] += db_results['total_entries']
                results['valid_entries'] += db_results['valid_entries']
                results['corrupted_entries'] += db_results['corrupted_entries']
            except Exception as e:
                error_msg = f"Database cache validation failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Validate file cache
        if validate_file_cache:
            try:
                file_results = self._validate_file_cache(raster_id, grid_id)
                results['file_cache'] = file_results
                results['total_entries'] += file_results['total_entries']
                results['valid_entries'] += file_results['valid_entries']
                results['corrupted_entries'] += file_results['corrupted_entries']
            except Exception as e:
                error_msg = f"File cache validation failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Update corruption stats
        if results['corrupted_entries'] > 0:
            with self._stats_lock:
                self._validation_stats['corrupted'] += results['corrupted_entries']
        
        logger.info(f"Cache validation complete: {results['valid_entries']}/{results['total_entries']} valid")
        return results
    
    def _validate_database_cache(self, raster_id: str, grid_id: str) -> Dict[str, Any]:
        """Validate database cache entries."""
        from src.database.schema import schema
        
        results = {
            'total_entries': 0,
            'valid_entries': 0,
            'corrupted_entries': 0,
            'validation_errors': []
        }
        
        # This would need to be implemented based on database schema
        # For now, assume database entries are valid
        logger.debug(f"Database cache validation skipped (not implemented)")
        
        return results
    
    def _validate_file_cache(self, raster_id: str, grid_id: str) -> Dict[str, Any]:
        """Validate file cache entries."""
        from .file_cache_service import FileCacheService
        from pathlib import Path
        from src.config import config
        
        results = {
            'total_entries': 0,
            'valid_entries': 0,
            'corrupted_entries': 0,
            'validation_errors': []
        }
        
        try:
            cache_dir = Path(config.get('raster_processing.cache_dir', 'cache/resampling'))
            file_cache = FileCacheService(cache_dir)
            
            cache_files = file_cache.list_cache_files()
            results['total_entries'] = len(cache_files)
            
            for cache_key in cache_files:
                try:
                    # Try to load the cache file
                    cached_data = file_cache.load_from_file(cache_key)
                    if cached_data:
                        values, metadata = cached_data
                        
                        # Validate metadata structure
                        if self._validate_cache_metadata(metadata):
                            results['valid_entries'] += 1
                        else:
                            results['corrupted_entries'] += 1
                            results['validation_errors'].append(f"Invalid metadata: {cache_key}")
                    else:
                        results['corrupted_entries'] += 1
                        results['validation_errors'].append(f"Failed to load: {cache_key}")
                        
                except Exception as e:
                    results['corrupted_entries'] += 1
                    results['validation_errors'].append(f"Cache file error {cache_key}: {e}")
            
        except Exception as e:
            results['validation_errors'].append(f"File cache validation error: {e}")
        
        return results
    
    def _validate_cache_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate cache metadata structure."""
        required_keys = ['cell_ids', 'raster_id', 'grid_id', 'method', 'band_number']
        
        for key in required_keys:
            if key not in metadata:
                return False
        
        # Additional validation checks
        if not isinstance(metadata.get('cell_ids'), list):
            return False
        
        if not isinstance(metadata.get('band_number'), int):
            return False
        
        return True
    
    def calculate_checksum(self, entry: Dict[str, Any]) -> str:
        """Calculate checksum for cache entry."""
        # Create a normalized representation for checksumming
        checksum_data = {
            'raster_id': entry.get('raster_id'),
            'grid_id': entry.get('grid_id'),
            'cell_ids': sorted(entry.get('cell_ids', [])),
            'method': entry.get('method'),
            'band_number': entry.get('band_number'),
            'values': entry.get('values')
        }
        
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        return hashlib.sha256(checksum_str.encode()).hexdigest()
    
    def validate_entry_checksum(self, entry: Dict[str, Any]) -> bool:
        """Validate entry checksum."""
        if 'checksum' not in entry:
            return True  # No checksum to validate
        
        expected_checksum = entry.pop('checksum')
        calculated_checksum = self.calculate_checksum(entry)
        entry['checksum'] = expected_checksum  # Restore checksum
        
        return expected_checksum == calculated_checksum
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        with self._stats_lock:
            return self._validation_stats.copy()