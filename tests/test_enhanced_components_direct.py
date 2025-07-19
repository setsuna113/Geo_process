"""Direct test of enhanced base classes without full module imports."""

import pytest
import numpy as np
import time
import threading
from pathlib import Path
import sys
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the specific modules we need without triggering the full src import
from src.base.raster_source import RasterTile
from src.base.tile_processor import ProcessingStatus, TileProgress  
from src.base.memory_tracker import get_memory_tracker


class TestRasterTileEnhanced:
    """Test enhanced RasterTile functionality."""
    
    def test_raster_tile_creation_basic(self):
        """Test basic raster tile creation."""
        data = np.ones((50, 50), dtype=np.float32)
        tile = RasterTile(
            data=data,
            bounds=(0, 0, 50, 50), 
            tile_id="test_basic",
            crs="EPSG:4326"
        )
        
        assert tile.tile_id == "test_basic"
        assert tile.bounds == (0, 0, 50, 50)
        assert tile.crs == "EPSG:4326"
        assert tile.shape == (50, 50)
        assert tile.dtype == np.float32
        
    def test_memory_size_calculation(self):
        """Test memory size calculation accuracy."""
        # Test with different sizes and data types
        test_cases = [
            ((10, 10), np.uint8),
            ((100, 100), np.float32),
            ((50, 50), np.float64),
        ]
        
        for shape, dtype in test_cases:
            data = np.ones(shape, dtype=dtype)
            tile = RasterTile(data, (0, 0, 1, 1), f"test_{dtype}")
            
            expected_bytes = data.nbytes
            expected_mb = expected_bytes / (1024 * 1024)
            
            assert abs(tile.memory_size_mb - expected_mb) < 0.001
            
    def test_nodata_handling(self):
        """Test nodata value handling."""
        # Test with nodata values
        data = np.full((20, 20), -9999, dtype=np.float32)
        tile = RasterTile(data, (0, 0, 1, 1), "nodata_test", nodata=-9999)
        
        assert tile.is_empty()
        
        # Add some valid data
        data[10, 10] = 100.0
        tile_with_data = RasterTile(data, (0, 0, 1, 1), "mixed_test", nodata=-9999)
        
        assert not tile_with_data.is_empty()
        
    def test_tile_properties(self):
        """Test tile property access."""
        data = np.random.rand(64, 64).astype(np.float32)
        tile = RasterTile(
            data=data,
            bounds=(100, 200, 164, 264),
            tile_id="props_test",
            crs="EPSG:3857",
            nodata=-999
        )
        
        assert tile.data is data
        assert tile.bounds == (100, 200, 164, 264)
        assert tile.tile_id == "props_test"
        assert tile.crs == "EPSG:3857"
        assert tile.nodata == -999
        assert tile.shape == (64, 64)
        assert tile.dtype == np.float32


class TestTileProgressEnhanced:
    """Test enhanced TileProgress functionality."""
    
    def test_progress_creation_and_properties(self):
        """Test creating and accessing tile progress properties."""
        progress = TileProgress(
            tile_id="progress_test",
            status=ProcessingStatus.PROCESSING,
            progress_percentage=75.5,
            memory_usage_mb=12.3
        )
        
        assert progress.tile_id == "progress_test"
        assert progress.status == ProcessingStatus.PROCESSING
        assert progress.progress_percentage == 75.5
        assert progress.memory_usage_mb == 12.3
        
    def test_processing_time_tracking(self):
        """Test processing time calculation."""
        progress = TileProgress("timing_test")
        
        # Initially no timing info
        assert progress.processing_time_seconds is None
        assert progress.start_time is None
        assert progress.end_time is None
        
        # Start processing
        start_time = time.time()
        progress.start_time = start_time
        progress.status = ProcessingStatus.PROCESSING
        
        # Should calculate current processing time
        time.sleep(0.05)  # 50ms
        current_time = progress.processing_time_seconds
        assert current_time is not None
        assert current_time >= 0.05
        
        # End processing
        end_time = time.time()
        progress.end_time = end_time
        progress.status = ProcessingStatus.COMPLETED
        
        # Should calculate total processing time
        total_time = progress.processing_time_seconds
        assert total_time is not None
        assert total_time >= 0.05
        assert abs(total_time - (end_time - start_time)) < 0.001
        
    def test_completion_status_tracking(self):
        """Test completion status logic."""
        progress = TileProgress("completion_test")
        
        # Test all non-complete states
        for status in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]:
            progress.status = status
            assert not progress.is_complete
            
        # Test all complete states
        for status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.SKIPPED]:
            progress.status = status
            assert progress.is_complete
            
    def test_progress_workflow_simulation(self):
        """Test a complete progress workflow."""
        progress = TileProgress("workflow_test")
        
        # Step 1: Initialize
        assert progress.status == ProcessingStatus.PENDING
        assert progress.progress_percentage == 0.0
        assert not progress.is_complete
        
        # Step 2: Start processing
        progress.status = ProcessingStatus.PROCESSING
        progress.start_time = time.time()
        progress.progress_percentage = 10.0
        
        assert not progress.is_complete
        assert progress.processing_time_seconds is not None
        
        # Step 3: Update progress
        progress.progress_percentage = 50.0
        progress.memory_usage_mb = 25.0
        
        # Step 4: Nearly complete
        progress.progress_percentage = 90.0
        
        # Step 5: Complete
        progress.status = ProcessingStatus.COMPLETED
        progress.end_time = time.time()
        progress.progress_percentage = 100.0
        
        assert progress.is_complete
        assert progress.progress_percentage == 100.0
        assert progress.processing_time_seconds > 0


class TestMemoryTrackerEnhanced:
    """Test enhanced memory tracker functionality."""
    
    def test_singleton_pattern(self):
        """Test memory tracker singleton pattern."""
        tracker1 = get_memory_tracker()
        tracker2 = get_memory_tracker()
        
        assert tracker1 is tracker2
        assert id(tracker1) == id(tracker2)
        
    def test_memory_snapshot_creation(self):
        """Test memory snapshot creation and properties."""
        tracker = get_memory_tracker()
        snapshot = tracker.get_current_snapshot()
        
        # Basic snapshot properties
        assert snapshot is not None
        assert hasattr(snapshot, 'timestamp')
        assert snapshot.timestamp > 0
        
        # Memory properties should exist and be reasonable
        assert hasattr(snapshot, 'heap_memory_mb')
        assert snapshot.heap_memory_mb >= 0
        
        assert hasattr(snapshot, 'system_available_mb')
        assert snapshot.system_available_mb > 0
        
        # Should have mapped files tracking
        assert hasattr(snapshot, 'mapped_files_mb')
        assert snapshot.mapped_files_mb >= 0
        
    def test_memory_prediction_basic(self):
        """Test basic memory prediction functionality."""
        tracker = get_memory_tracker()
        
        prediction = tracker.predict_memory_usage(
            data_size_mb=50.0,
            operation_type="test_operation"
        )
        
        # Prediction should have required fields
        assert hasattr(prediction, 'predicted_peak_mb')
        assert prediction.predicted_peak_mb > 0
        
        assert hasattr(prediction, 'predicted_duration_seconds')
        assert prediction.predicted_duration_seconds > 0
        
        assert hasattr(prediction, 'confidence')
        assert 0.0 <= prediction.confidence <= 1.0
        
    def test_mapped_file_tracking_basic(self):
        """Test basic mapped file tracking."""
        tracker = get_memory_tracker()
        
        # Get initial state
        initial_snapshot = tracker.get_current_snapshot()
        initial_mapped = initial_snapshot.mapped_files_mb
        
        # Track a file
        test_file = "/test/mapped_file.dat"
        file_size_mb = 25.0
        tracker.track_mapped_file(test_file, file_size_mb)
        
        # Check it's tracked
        after_track_snapshot = tracker.get_current_snapshot()
        assert after_track_snapshot.mapped_files_mb >= initial_mapped + file_size_mb
        
        # Untrack the file
        tracker.untrack_mapped_file(test_file)
        
        # Check it's removed
        after_untrack_snapshot = tracker.get_current_snapshot()
        assert test_file not in tracker._mapped_files
        
    def test_memory_summary(self):
        """Test memory summary functionality."""
        tracker = get_memory_tracker()
        
        summary = tracker.get_memory_summary()
        
        # Summary should contain key information
        assert isinstance(summary, dict)
        assert 'current' in summary
        assert 'system' in summary


class TestProcessingStatusEnum:
    """Test ProcessingStatus enum functionality."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
        assert ProcessingStatus.SKIPPED.value == "skipped"
        
    def test_enum_comparison(self):
        """Test enum comparison functionality."""
        # Test equality
        assert ProcessingStatus.PENDING == ProcessingStatus.PENDING
        assert ProcessingStatus.COMPLETED != ProcessingStatus.FAILED
        
        # Test enum instances
        status1 = ProcessingStatus.PROCESSING
        status2 = ProcessingStatus.PROCESSING
        assert status1 == status2
        assert status1 is status2


class TestIntegrationScenarios:
    """Test integration scenarios with enhanced base classes."""
    
    def test_tile_processing_workflow(self):
        """Test a complete tile processing workflow."""
        # Create tiles
        tiles = []
        progress_trackers = []
        
        for i in range(3):
            # Create tile
            size = 32 + i * 16
            data = np.random.rand(size, size).astype(np.float32)
            tile = RasterTile(
                data=data,
                bounds=(i*100, 0, (i+1)*100, 100),
                tile_id=f"workflow_tile_{i}",
                crs="EPSG:4326"
            )
            tiles.append(tile)
            
            # Create progress tracker
            progress = TileProgress(f"workflow_tile_{i}")
            progress.status = ProcessingStatus.PENDING
            progress.memory_usage_mb = tile.memory_size_mb
            progress_trackers.append(progress)
            
        # Process tiles
        for i, (tile, progress) in enumerate(zip(tiles, progress_trackers)):
            # Start processing
            progress.status = ProcessingStatus.PROCESSING
            progress.start_time = time.time()
            progress.progress_percentage = 0.0
            
            # Simulate processing steps
            for step_progress in [25.0, 50.0, 75.0, 100.0]:
                time.sleep(0.01)  # Simulate work
                progress.progress_percentage = step_progress
                
            # Complete processing
            progress.status = ProcessingStatus.COMPLETED
            progress.end_time = time.time()
            
        # Verify all completed
        assert len(tiles) == 3
        assert len(progress_trackers) == 3
        assert all(p.is_complete for p in progress_trackers)
        assert all(p.progress_percentage == 100.0 for p in progress_trackers)
        
    def test_memory_constrained_processing(self):
        """Test processing under memory constraints."""
        tracker = get_memory_tracker()
        
        # Get initial memory state
        initial_snapshot = tracker.get_current_snapshot()
        
        # Create memory-intensive tiles
        tiles = []
        total_memory = 0.0
        
        for i in range(5):
            size = 64 + i * 32
            data = np.random.rand(size, size).astype(np.float64)  # Use float64 for more memory
            tile = RasterTile(data, (i*100, 0, (i+1)*100, 100), f"memory_tile_{i}")
            
            tiles.append(tile)
            total_memory += tile.memory_size_mb
            
        # Track memory usage
        current_snapshot = tracker.get_current_snapshot()
        
        # Should have created tiles successfully
        assert len(tiles) == 5
        assert total_memory > 0
        
        # Memory tracking should work
        assert current_snapshot.heap_memory_mb >= initial_snapshot.heap_memory_mb
        
        # Clean up
        del tiles
        
    def test_concurrent_access_safety(self):
        """Test concurrent access to enhanced components."""
        import concurrent.futures
        
        tracker = get_memory_tracker()
        results = []
        
        def concurrent_task(task_id):
            """Task that uses enhanced components concurrently."""
            try:
                # Create tile
                data = np.random.rand(32, 32).astype(np.float32)
                tile = RasterTile(data, (task_id, 0, task_id+1, 1), f"concurrent_{task_id}")
                
                # Track progress
                progress = TileProgress(f"concurrent_{task_id}")
                progress.status = ProcessingStatus.PROCESSING
                progress.start_time = time.time()
                progress.memory_usage_mb = tile.memory_size_mb
                
                # Track memory file
                file_path = f"/test/concurrent_{task_id}.dat"
                tracker.track_mapped_file(file_path, tile.memory_size_mb)
                
                # Simulate some work
                time.sleep(0.01)
                
                # Complete
                progress.status = ProcessingStatus.COMPLETED
                progress.end_time = time.time()
                
                # Clean up
                tracker.untrack_mapped_file(file_path)
                
                return {
                    'task_id': task_id,
                    'tile_memory': tile.memory_size_mb,
                    'processing_time': progress.processing_time_seconds,
                    'success': True
                }
                
            except Exception as e:
                return {
                    'task_id': task_id,
                    'error': str(e),
                    'success': False
                }
                
        # Run concurrent tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(concurrent_task, i) for i in range(8)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
        # All tasks should succeed
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        assert len(successful) == 8
        assert len(failed) == 0
        
        # All should have reasonable values
        for result in successful:
            assert result['tile_memory'] > 0
            assert result['processing_time'] > 0
            
    def test_error_handling_robustness(self):
        """Test error handling in enhanced components."""
        # Test with edge case data
        edge_cases = [
            # Very small tile
            (np.array([[1]], dtype=np.float32), "single_pixel"),
            # Large tile (but reasonable for testing)
            (np.random.rand(256, 256).astype(np.float32), "large_tile"),
            # All zeros
            (np.zeros((50, 50), dtype=np.float32), "all_zeros"),
        ]
        
        for data, test_name in edge_cases:
            try:
                tile = RasterTile(data, (0, 0, 1, 1), test_name)
                
                # Should be able to access basic properties
                assert tile.tile_id == test_name
                assert tile.memory_size_mb >= 0
                assert tile.shape == data.shape
                
                # Progress tracking should work
                progress = TileProgress(f"progress_{test_name}")
                progress.memory_usage_mb = tile.memory_size_mb
                progress.status = ProcessingStatus.COMPLETED
                
                assert progress.is_complete
                
            except Exception as e:
                # If there are errors, they should be reasonable
                assert isinstance(e, (ValueError, TypeError, IndexError))
                
        # Test progress with error states
        error_progress = TileProgress("error_test")
        error_progress.status = ProcessingStatus.FAILED
        error_progress.error_message = "Test error condition"
        
        assert error_progress.is_complete
        assert error_progress.error_message == "Test error condition"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
