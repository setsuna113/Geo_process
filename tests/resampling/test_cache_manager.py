# tests/test_resampling/test_cache_manager.py
"""Unit tests for resampling cache manager."""

import pytest
import numpy as np
from unittest.mock import patch
import tempfile
from pathlib import Path
import json

from src.resampling.cache_manager import ResamplingCacheManager


class TestResamplingCacheManager:
    """Test cache manager functionality."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.resampling.cache_manager.config') as mock_config:
                mock_config.get.return_value = tmpdir
                manager = ResamplingCacheManager()
                manager.cache_dir = Path(tmpdir)
                yield manager
    
    def test_get_cache_key(self, cache_manager):
        """Test cache key generation."""
        key1 = cache_manager.get_cache_key(
            'raster_1', 0.1, 'bilinear', (0, 0, 10, 10)
        )
        
        key2 = cache_manager.get_cache_key(
            'raster_1', 0.1, 'bilinear', (0, 0, 10, 10)
        )
        
        key3 = cache_manager.get_cache_key(
            'raster_2', 0.1, 'bilinear', (0, 0, 10, 10)
        )
        
        assert key1 == key2  # Same parameters
        assert key1 != key3  # Different raster
    
    @patch('src.resampling.cache_manager.schema')
    def test_get_from_cache(self, mock_schema, cache_manager):
        """Test cache retrieval from database."""
        mock_schema.get_cached_resampling_values.return_value = {
            'cell_1': 42.0,
            'cell_2': 43.0
        }
        
        result = cache_manager.get_from_cache(
            'raster_1', 'grid_1', ['cell_1', 'cell_2'], 
            'bilinear', 1
        )
        
        assert result['cell_1'] == 42.0
        assert result['cell_2'] == 43.0
        
        mock_schema.get_cached_resampling_values.assert_called_once()
    
    @patch('src.resampling.cache_manager.schema')
    def test_store_in_cache(self, mock_schema, cache_manager):
        """Test cache storage."""
        mock_schema.store_resampling_cache_batch.return_value = 2
        
        cache_entries = [
            {'source_raster_id': 'r1', 'value': 1.0},
            {'source_raster_id': 'r1', 'value': 2.0}
        ]
        
        count = cache_manager.store_in_cache(cache_entries)
        
        assert count == 2
        mock_schema.store_resampling_cache_batch.assert_called_once_with(cache_entries)
    
    def test_save_to_file(self, cache_manager):
        """Test file cache saving."""
        data = np.random.rand(10, 10)
        metadata = {'test': 'value'}
        
        path = cache_manager.save_to_file(data, 'test_key', metadata)
        
        assert path.exists()
        assert path.suffix == '.npz'
        
        # Verify contents
        with np.load(path) as npz:
            assert np.array_equal(npz['data'], data)
            assert json.loads(npz['metadata'].item()) == metadata
    
    def test_load_from_file(self, cache_manager):
        """Test file cache loading."""
        # Save first
        data = np.random.rand(5, 5)
        metadata = {'method': 'test'}
        cache_manager.save_to_file(data, 'load_test', metadata)
        
        # Load back
        loaded_data, loaded_meta = cache_manager.load_from_file('load_test')
        
        assert np.array_equal(loaded_data, data)
        assert loaded_meta == metadata
    
    def test_load_from_file_missing(self, cache_manager):
        """Test loading non-existent cache file."""
        result = cache_manager.load_from_file('nonexistent')
        assert result is None
    
    @patch('src.resampling.cache_manager.schema')
    def test_warm_cache(self, mock_schema, cache_manager):
        """Test cache warming."""
        mock_schema.add_processing_task.return_value = 'task_123'
        
        result = cache_manager.warm_cache(
            'raster_1', 'grid_1', 
            priority_bounds=(0, 0, 10, 10),
            method='bilinear'
        )
        
        assert result['task_id'] == 'task_123'
        assert result['status'] == 'queued'
        assert result['priority'] == 10
        
        mock_schema.add_processing_task.assert_called_once()
    
    @patch('src.resampling.cache_manager.schema')
    def test_cleanup_old_cache(self, mock_schema, cache_manager):
        """Test cache cleanup."""
        mock_schema.cleanup_old_cache.return_value = 150
        
        count = cache_manager.cleanup_old_cache(days_old=7, min_access_count=2)
        
        assert count == 150
        mock_schema.cleanup_old_cache.assert_called_once_with(7, 2)