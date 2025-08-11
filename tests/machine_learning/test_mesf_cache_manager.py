"""Tests for MESF Cache Manager."""

import os
import json
import tempfile
import pytest
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.machine_learning.spatial.mesf_cache_manager import (
    MESFCacheManager, FungiRichnessCacheManager
)


class TestMESFCacheManager:
    """Test suite for MESF Cache Manager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_parquet_file(self, tmp_path):
        """Create a sample parquet file for testing."""
        df = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'value': np.random.randn(100)
        })
        file_path = tmp_path / "test_data.parquet"
        df.to_parquet(file_path)
        return str(file_path)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create cache manager with temp directory."""
        return MESFCacheManager(cache_dir=temp_cache_dir)
    
    def test_init_creates_directory(self, temp_cache_dir):
        """Test that initialization creates cache directory."""
        cache_dir = Path(temp_cache_dir) / "test_cache"
        manager = MESFCacheManager(cache_dir=str(cache_dir))
        assert cache_dir.exists()
        assert cache_dir.is_dir()
    
    def test_init_sets_secure_permissions(self, temp_cache_dir):
        """Test that cache directory has secure permissions."""
        cache_dir = Path(temp_cache_dir) / "test_cache"
        manager = MESFCacheManager(cache_dir=str(cache_dir))
        # Check permissions (owner only)
        stat_info = cache_dir.stat()
        mode = stat_info.st_mode & 0o777
        assert mode == 0o700
    
    def test_init_fallback_to_home_on_permission_error(self):
        """Test fallback to home directory on permission error."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            # First call raises PermissionError, second succeeds
            mock_mkdir.side_effect = [PermissionError("No permission"), None, None]
            with patch('os.chmod'):  # Mock chmod as well
                manager = MESFCacheManager()
                assert '.cache' in str(manager.cache_dir)
    
    def test_get_cache_key_validates_empty_path(self, cache_manager):
        """Test that empty file path raises ValueError."""
        with pytest.raises(ValueError, match="file_path cannot be empty"):
            cache_manager.get_cache_key("", 1, {})
    
    def test_get_cache_key_validates_nonexistent_file(self, cache_manager):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            cache_manager.get_cache_key("/nonexistent/file.parquet", 1, {})
    
    def test_get_cache_key_validates_directory_path(self, cache_manager, temp_cache_dir):
        """Test that directory path raises ValueError."""
        with pytest.raises(ValueError, match="Path is not a file"):
            cache_manager.get_cache_key(temp_cache_dir, 1, {})
    
    def test_get_cache_key_validates_spatial_grid_step(self, cache_manager, sample_parquet_file):
        """Test that invalid spatial_grid_step raises ValueError."""
        with pytest.raises(ValueError, match="spatial_grid_step must be positive"):
            cache_manager.get_cache_key(sample_parquet_file, 0, {})
        
        with pytest.raises(ValueError, match="spatial_grid_step must be positive"):
            cache_manager.get_cache_key(sample_parquet_file, -1, {})
    
    def test_get_cache_key_generates_unique_keys(self, cache_manager, sample_parquet_file):
        """Test that different parameters generate different keys."""
        key1 = cache_manager.get_cache_key(sample_parquet_file, 1, {'param': 1})
        key2 = cache_manager.get_cache_key(sample_parquet_file, 2, {'param': 1})
        key3 = cache_manager.get_cache_key(sample_parquet_file, 1, {'param': 2})
        
        assert key1 != key2  # Different spatial_grid_step
        assert key1 != key3  # Different params
        assert key2 != key3
    
    @pytest.mark.skip(reason="PyArrow table.nbytes attribute handling varies by version")
    def test_save_to_cache_checks_disk_space(self, cache_manager, sample_parquet_file):
        """Test that save_to_cache checks available disk space."""
        # Create a small test table
        data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        table = pa.Table.from_pydict(data)
        
        # The implementation uses 100MB as default when nbytes not available
        # So we need very small space to trigger the error
        with patch('os.statvfs') as mock_statvfs:
            # Simulate insufficient disk space (less than 100MB default)
            mock_stat = MagicMock()
            mock_stat.f_bavail = 10  # Very small available blocks (10 * 1024 = 10KB)
            mock_stat.f_frsize = 1024  # 1KB blocks
            mock_statvfs.return_value = mock_stat
            
            with pytest.raises(OSError, match="Insufficient disk space"):
                cache_manager.save_to_cache("test_key", table, {})
    
    def test_save_to_cache_atomic_write(self, cache_manager, sample_parquet_file):
        """Test that save_to_cache uses atomic write with temp files."""
        data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        table = pa.Table.from_pydict(data)
        metadata = {'test': 'metadata'}
        
        # Save to cache
        cache_key = "test_atomic"
        cached_path = cache_manager.save_to_cache(cache_key, table, metadata)
        
        # Verify files exist
        assert Path(cached_path).exists()
        metadata_path = cache_manager.get_metadata_path(cache_key)
        assert metadata_path.exists()
        
        # Verify metadata
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata == metadata
    
    def test_load_from_cache_with_corrupted_data(self, cache_manager):
        """Test that corrupted cache is handled properly."""
        cache_key = "corrupted_test"
        cached_path = cache_manager.get_cached_path(cache_key)
        metadata_path = cache_manager.get_metadata_path(cache_key)
        
        # Create corrupted files
        cached_path.write_text("corrupted data")
        metadata_path.write_text('{"test": "metadata"}')
        
        # Try to load - should raise and cleanup
        with pytest.raises(OSError, match="Corrupted cache"):
            cache_manager.load_from_cache(cache_key)
        
        # Files should be cleaned up
        assert not cached_path.exists()
        assert not metadata_path.exists()
    
    def test_load_from_cache_timeout_fallback(self, cache_manager, monkeypatch):
        """Test timeout fallback in load_from_cache."""
        cache_key = "timeout_test"
        cached_path = cache_manager.get_cached_path(cache_key)
        metadata_path = cache_manager.get_metadata_path(cache_key)
        
        # Create valid cache files
        data = {'col1': [1, 2, 3]}
        table = pa.Table.from_pydict(data)
        pq.write_table(table, cached_path)
        metadata_path.write_text('{"test": "metadata"}')
        
        # Mock file lock to simulate timeout
        with patch.object(cache_manager, '_file_lock') as mock_lock:
            mock_lock.side_effect = TimeoutError("Lock timeout")
            
            # Should fall back to reading without lock
            path, metadata = cache_manager.load_from_cache(cache_key)
            assert path == str(cached_path)
            assert metadata == {"test": "metadata"}


class TestFungiRichnessCacheManager:
    """Test suite for Fungi Richness Cache Manager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample file for testing."""
        file_path = tmp_path / "test_data.txt"
        file_path.write_text("test data")
        return str(file_path)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create cache manager with temp directory."""
        return FungiRichnessCacheManager(cache_dir=temp_cache_dir)
    
    def test_init_creates_secure_directory(self, temp_cache_dir):
        """Test that initialization creates directory with secure permissions."""
        cache_dir = Path(temp_cache_dir) / "fungi_cache"
        manager = FungiRichnessCacheManager(cache_dir=str(cache_dir))
        assert cache_dir.exists()
        # Check secure permissions
        mode = cache_dir.stat().st_mode & 0o777
        assert mode == 0o700
    
    def test_get_cache_key_validates_empty_path(self, cache_manager):
        """Test that empty file path raises ValueError."""
        with pytest.raises(ValueError, match="file_path cannot be empty"):
            cache_manager.get_cache_key("")
    
    def test_get_cache_key_validates_nonexistent_file(self, cache_manager):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            cache_manager.get_cache_key("/nonexistent/file.txt")
    
    def test_get_cache_key_validates_directory(self, cache_manager, temp_cache_dir):
        """Test that directory path raises ValueError."""
        with pytest.raises(ValueError, match="Path is not a file"):
            cache_manager.get_cache_key(temp_cache_dir)
    
    def test_get_cache_key_handles_path_traversal(self, cache_manager, sample_file):
        """Test that path traversal is handled safely."""
        # Path with .. should be resolved
        dir_path = Path(sample_file).parent
        traversal_path = str(dir_path / ".." / dir_path.name / Path(sample_file).name)
        
        # Should resolve to same key as direct path
        key1 = cache_manager.get_cache_key(sample_file)
        key2 = cache_manager.get_cache_key(traversal_path)
        assert key1 == key2
    
    def test_cache_operations(self, cache_manager, sample_file):
        """Test basic cache operations."""
        cache_key = cache_manager.get_cache_key(sample_file)
        
        # Initially not cached
        assert not cache_manager.is_cached(cache_key)
        
        # Get cached path
        cached_path = cache_manager.get_cached_path(cache_key)
        assert "fungi_" in cached_path.name
        assert cached_path.suffix == ".parquet"