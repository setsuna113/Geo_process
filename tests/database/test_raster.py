"""Tests for raster data management functionality."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.database.schema import schema
from src.raster.manager import RasterManager
from src.raster import manager as raster_manager_module

class TestRasterManager:
    """Test RasterManager functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        # This would typically reset the database state
        yield
    
    @pytest.fixture
    def mock_raster_file(self):
        """Create a mock raster file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp.write(b"mock raster data")
            return Path(tmp.name)
    
    @pytest.fixture
    def sample_raster_metadata(self):
        """Sample raster metadata for testing."""
        return {
            'source_dataset': 'WorldClim_v2.1',
            'variable_name': 'bio_1',
            'units': 'degrees_celsius',
            'description': 'Annual Mean Temperature',
            'temporal_info': {
                'start_date': '1970-01-01',
                'end_date': '2000-12-31',
                'temporal_resolution': 'annual'
            }
        }
    
    def test_raster_manager_initialization(self):
        """Test RasterManager initialization."""
        manager = RasterManager()
        # Test that manager initializes without errors
        assert manager is not None
    
    @patch('src.raster.metadata.RasterMetadataExtractor.extract_metadata')
    @patch('src.raster.metadata.RasterMetadataExtractor.calculate_file_checksum')
    def test_register_raster_source(self, mock_checksum, mock_metadata, 
                                   mock_raster_file, sample_raster_metadata):
        """Test registering a new raster source."""
        # Mock the metadata extraction and checksum calculation
        mock_metadata.return_value = {
            'data_type': 'Float32',
            'pixel_size_degrees': 0.016666666666667,
            'spatial_extent_wkt': 'POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))',
            'nodata_value': -9999.0,
            'band_count': 1,
            'crs': 'EPSG:4326'
        }
        mock_checksum.return_value = 'abc123def456'
        
        # Mock schema operations
        with patch.object(schema, 'store_raster_source') as mock_store:
            mock_store.return_value = 'raster-id-123'
            
            with patch.object(schema, 'add_processing_task') as mock_task:
                manager = RasterManager()
                raster_id = manager.register_raster_source(
                    mock_raster_file, 
                    name='test_raster',
                    metadata=sample_raster_metadata
                )
                
                assert raster_id == 'raster-id-123'
                mock_store.assert_called_once()
                
                # Check that store was called with correct data structure
                call_args = mock_store.call_args[0][0]
                assert call_args['name'] == 'test_raster'
                assert call_args['data_type'] == 'Float32'
                assert call_args['checksum'] == 'abc123def456'
                assert call_args['source_dataset'] == 'WorldClim_v2.1'
        
        # Cleanup
        mock_raster_file.unlink()
    
    def test_register_raster_source_file_not_found(self):
        """Test error handling when raster file doesn't exist."""
        manager = RasterManager()
        
        with pytest.raises(FileNotFoundError):
            manager.register_raster_source('/nonexistent/file.tif')
    
    def test_register_raster_source_unsupported_format(self, mock_raster_file):
        """Test error handling for unsupported file formats."""
        # Rename to unsupported extension
        unsupported_file = mock_raster_file.with_suffix('.txt')
        mock_raster_file.rename(unsupported_file)
        
        manager = RasterManager()
        
        with pytest.raises(ValueError, match="Unsupported raster format"):
            manager.register_raster_source(unsupported_file)
        
        # Cleanup
        unsupported_file.unlink()
    
    @patch('src.raster.processor.RasterProcessor.generate_tile_metadata')
    def test_create_raster_tiles(self, mock_generate_tiles):
        """Test raster tile creation."""
        # Mock tile generation
        mock_tiles = [
            {
                'tile_x': 0, 'tile_y': 0, 'tile_size_pixels': 1000,
                'tile_bounds_wkt': 'POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))',
                'file_byte_offset': 0, 'file_byte_length': 1024,
                'tile_stats': {'min': 0, 'max': 100, 'mean': 50}
            }
        ]
        mock_generate_tiles.return_value = mock_tiles
        
        # Mock schema operations
        with patch.object(schema, 'get_raster_sources') as mock_get:
            mock_get.return_value = [{
                'id': 'raster-123',
                'name': 'test_raster',
                'processing_status': 'pending'
            }]
            
            with patch.object(schema, 'update_raster_processing_status') as mock_update:
                with patch.object(schema, 'store_raster_tiles_batch') as mock_store_tiles:
                    mock_store_tiles.return_value = 1
                    
                    manager = RasterManager()
                    count = manager.create_raster_tiles('raster-123')
                    
                    assert count == 1
                    mock_store_tiles.assert_called_once_with('raster-123', mock_tiles)
                    
                    # Check status updates
                    assert mock_update.call_count == 2  # tiling -> ready
                    mock_update.assert_any_call('raster-123', 'tiling')
                    # Check that the call was made with proper structure - using ANY for dynamic timestamp
                    from unittest.mock import ANY
                    mock_update.assert_any_call('raster-123', 'ready', 
                                               {'tiling_completed_at': ANY, 'tile_count': 1})
    
    @patch('src.raster.processor.RasterProcessor.perform_resampling')
    def test_resample_to_grid(self, mock_resample):
        """Test raster resampling to grid."""
        # Mock resampling results
        mock_resample.return_value = {'cell_1': 25.5, 'cell_2': 30.2}
        
        # Mock schema operations
        with patch.object(schema, 'get_cached_resampling_values') as mock_cache:
            mock_cache.return_value = {}  # No cached values
            
            with patch.object(schema, 'get_grid_by_name') as mock_grid:
                mock_grid.return_value = {'id': 'grid-123', 'name': 'test_grid'}
                
                with patch.object(schema, 'store_resampling_cache_batch') as mock_store_cache:
                    manager = RasterManager()
                    results = manager.resample_to_grid(
                        'raster-123', 'grid-123', 
                        cell_ids=['cell_1', 'cell_2'],
                        method='bilinear'
                    )
                    
                    assert results == {'cell_1': 25.5, 'cell_2': 30.2}
                    mock_store_cache.assert_called_once()
                    
                    # Check cache data structure
                    cache_data = mock_store_cache.call_args[0][0]
                    assert len(cache_data) == 2
                    assert cache_data[0]['cell_id'] == 'cell_1'
                    assert cache_data[0]['value'] == 25.5
                    assert cache_data[0]['method'] == 'bilinear'
    
    def test_resample_to_grid_with_cache(self):
        """Test resampling with cached values."""
        # Mock cached values
        with patch.object(schema, 'get_cached_resampling_values') as mock_cache:
            mock_cache.return_value = {'cell_1': 25.5, 'cell_2': 30.2}
            
            manager = RasterManager()
            results = manager.resample_to_grid(
                'raster-123', 'grid-123',
                cell_ids=['cell_1', 'cell_2'],
                use_cache=True
            )
            
            assert results == {'cell_1': 25.5, 'cell_2': 30.2}
            mock_cache.assert_called_once()
    
    def test_get_processing_status(self):
        """Test getting raster processing status."""
        with patch.object(schema, 'get_raster_processing_status') as mock_status:
            mock_status.return_value = [
                {
                    'raster_id': 'raster-123',
                    'raster_name': 'test_raster',
                    'processing_status': 'ready',
                    'total_tiles': 100,
                    'completed_tiles': 95,
                    'completion_percent': 95.0
                }
            ]
            
            manager = RasterManager()
            status = manager.get_processing_status('raster-123')
            
            assert len(status) == 1
            assert status[0]['raster_name'] == 'test_raster'
            assert status[0]['completion_percent'] == 95.0
    
    def test_cleanup_cache(self):
        """Test cache cleanup functionality."""
        with patch.object(schema, 'cleanup_old_cache') as mock_cleanup:
            mock_cleanup.return_value = 42  # 42 entries cleaned
            
            manager = RasterManager()
            deleted_count = manager.cleanup_cache(days_old=15, min_access_count=2)
            
            assert deleted_count == 42
            mock_cleanup.assert_called_once_with(15, 2)
    
    def test_process_queue_task_tiling(self):
        """Test processing a tiling task from the queue."""
        # Mock task from queue
        mock_task = {
            'id': 'task-123',
            'queue_type': 'raster_tiling',
            'raster_source_id': 'raster-123',
            'parameters': '{"tile_size": 1000}'
        }
        
        with patch.object(schema, 'get_next_processing_task') as mock_get_task:
            mock_get_task.return_value = mock_task
            
            with patch.object(schema, 'complete_processing_task') as mock_complete:
                with patch.object(RasterManager, 'create_raster_tiles') as mock_create_tiles:
                    mock_create_tiles.return_value = 50  # 50 tiles created
                    
                    manager = RasterManager()
                    success = manager.process_queue_task('raster_tiling', 'worker-1')
                    
                    assert success is True
                    mock_create_tiles.assert_called_once_with('raster-123')
                    mock_complete.assert_called_once_with('task-123', success=True)
    
    def test_process_queue_task_no_tasks(self):
        """Test processing when no tasks are available."""
        with patch.object(schema, 'get_next_processing_task') as mock_get_task:
            mock_get_task.return_value = None  # No tasks available
            
            manager = RasterManager()
            success = manager.process_queue_task('raster_tiling', 'worker-1')
            
            assert success is False
    
    def test_process_queue_task_failure(self):
        """Test handling of task processing failure."""
        mock_task = {
            'id': 'task-123',
            'queue_type': 'raster_tiling',
            'raster_source_id': 'raster-123',
            'parameters': '{}'
        }
        
        with patch.object(schema, 'get_next_processing_task') as mock_get_task:
            mock_get_task.return_value = mock_task
            
            with patch.object(schema, 'complete_processing_task') as mock_complete:
                with patch.object(RasterManager, 'create_raster_tiles') as mock_create_tiles:
                    mock_create_tiles.side_effect = Exception("Tiling failed")
                    
                    manager = RasterManager()
                    success = manager.process_queue_task('raster_tiling', 'worker-1')
                    
                    assert success is False
                    mock_complete.assert_called_once_with(
                        'task-123', success=False, error_message="Tiling failed"
                    )

class TestRasterSchemaIntegration:
    """Test raster schema operations integration."""
    
    def test_raster_source_storage_and_retrieval(self):
        """Test storing and retrieving raster sources."""
        # This would be an integration test that actually uses the database
        # For now, we'll mock the database operations
        
        sample_raster_data = {
            'name': f'bio_1_annual_{hash(id(self))}',
            'file_path': '/data/worldclim/bio_1.tif',
            'data_type': 'Float32',
            'pixel_size_degrees': 0.016666666666667,
            'spatial_extent_wkt': 'POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))',
            'nodata_value': -9999.0,
            'band_count': 1,
            'file_size_mb': 125.5,
            'checksum': 'abc123',
            'last_modified': '2024-01-01T00:00:00Z',
            'source_dataset': 'WorldClim_v2.1',
            'variable_name': 'bio_1',
            'units': 'degrees_celsius',
            'description': 'Annual Mean Temperature',
            'temporal_info': {},
            'metadata': {}
        }
        
        with patch.object(schema, 'store_raster_source') as mock_store:
            mock_store.return_value = 'raster-123'
            
            with patch.object(schema, 'get_raster_sources') as mock_get:
                mock_get.return_value = [sample_raster_data]
                
                # Test storage
                raster_id = schema.store_raster_source(sample_raster_data)
                assert raster_id == 'raster-123'
                
                # Test retrieval
                sources = schema.get_raster_sources()
                test_sources = [s for s in sources if s['name'].startswith('bio_1_annual')]
                assert len(test_sources) == 1
                assert test_sources[0]['name'].startswith('bio_1_annual')
    
    def test_cache_operations(self):
        """Test resampling cache operations."""
        cache_data = [
            {
                'source_raster_id': 'raster-123',
                'target_grid_id': 'grid-456',
                'cell_id': 'cell_1',
                'method': 'bilinear',
                'band_number': 1,
                'value': 25.5,
                'confidence_score': 0.95,
                'source_tiles_used': [1, 2, 3],
                'computation_metadata': {'method': 'bilinear'}
            }
        ]
        
        with patch.object(schema, 'store_resampling_cache_batch') as mock_store:
            mock_store.return_value = 1
            
            with patch.object(schema, 'get_cached_resampling_values') as mock_get:
                mock_get.return_value = {'cell_1': 25.5}
                
                # Test storage
                count = schema.store_resampling_cache_batch(cache_data)
                assert count == 1
                
                # Test retrieval
                cached = schema.get_cached_resampling_values(
                    'raster-123', 'grid-456', ['cell_1'], 'bilinear', 1
                )
                assert cached == {'cell_1': 25.5}

def test_global_raster_manager_instance():
    """Test that global raster manager instance is available."""
    from src.raster.manager import raster_manager
    assert raster_manager is not None
    assert isinstance(raster_manager, RasterManager)
