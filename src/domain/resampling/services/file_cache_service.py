"""File-based cache service for resampling operations."""

import json
import pickle
import lz4.frame
from typing import Dict, Optional, Tuple, Any
import numpy as np
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class FileCacheService:
    """File-based cache storage service."""
    
    def __init__(self, cache_dir: Path, compression: str = 'lz4'):
        self.cache_dir = cache_dir
        self.compression = compression
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_to_file(self, cache_key: str, values: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save cached values to file."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.cache"
            
            # Prepare data for serialization
            cache_data = {
                'values': values,
                'metadata': metadata,
                'created_at': time.time(),
                'version': '1.0'
            }
            
            # Serialize and compress
            serialized = pickle.dumps(cache_data)
            
            if self.compression == 'lz4':
                compressed = lz4.frame.compress(serialized)
            else:
                compressed = serialized
            
            # Write to file atomically
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                f.write(compressed)
            
            temp_file.rename(cache_file)
            
            logger.debug(f"✅ Saved cache file: {cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save cache file {cache_key}: {e}")
            return False
    
    def load_from_file(self, cache_key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Load cached values from file."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.cache"
            
            if not cache_file.exists():
                return None
            
            # Read and decompress
            with open(cache_file, 'rb') as f:
                compressed = f.read()
            
            if self.compression == 'lz4':
                try:
                    serialized = lz4.frame.decompress(compressed)
                except lz4.frame.LZ4FrameError:
                    # Fallback for uncompressed files
                    serialized = compressed
            else:
                serialized = compressed
            
            # Deserialize
            cache_data = pickle.loads(serialized)
            
            # Validate cache format
            if not isinstance(cache_data, dict) or 'values' not in cache_data:
                logger.warning(f"Invalid cache format: {cache_key}")
                return None
            
            values = cache_data['values']
            metadata = cache_data.get('metadata', {})
            
            logger.debug(f"✅ Loaded cache file: {cache_file}")
            return values, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load cache file {cache_key}: {e}")
            return None
    
    def delete_file(self, cache_key: str) -> bool:
        """Delete cache file."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.cache"
            if cache_file.exists():
                cache_file.unlink()
                logger.debug(f"✅ Deleted cache file: {cache_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to delete cache file {cache_key}: {e}")
            return False
    
    def get_file_size(self, cache_key: str) -> int:
        """Get cache file size in bytes."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.cache"
            return cache_file.stat().st_size if cache_file.exists() else 0
        except Exception as e:
            logger.error(f"❌ Failed to get file size {cache_key}: {e}")
            return 0
    
    def list_cache_files(self) -> List[str]:
        """List all cache files."""
        try:
            return [f.stem for f in self.cache_dir.glob("*.cache")]
        except Exception as e:
            logger.error(f"❌ Failed to list cache files: {e}")
            return []