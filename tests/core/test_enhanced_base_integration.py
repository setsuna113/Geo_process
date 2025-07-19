"""Test suite for enhanced base class functionality - focused testing."""

import pytest
import numpy as np
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Direct imports to avoid dependency issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.base.raster_source import RasterTile
from src.base.tile_processor import ProcessingStatus, TileProgress
from src.base.memory_tracker import get_memory_tracker


class TestRasterTileCore:
    """Test core RasterTile functionality."""
    
    def test_raster_tile_creation(self):
        """Test creating a raster tile."""
        data = np.random.rand(100, 100).astype(np.float32)
        bounds = (0.0, 0.0, 1.0, 1.0)
        
        tile = RasterTile(
            data=data,
            bounds=bounds,
            tile_id="test_tile",
            crs="EPSG:4326",
            nodata=-9999
        )
        
        assert tile.tile_id == "test_tile"
        assert tile.bounds == bounds
        assert tile.crs == "EPSG:4326"
        assert tile.nodata == -9999
        assert tile.shape == (100, 100)
        assert tile.dtype == np.float32
        
    def test_memory_size_calculation(self):
        """Test memory size calculation."""
        data = np.ones((100, 100), dtype=np.float32)
        tile = RasterTile(data, (0, 0, 1, 1), "test")
        
        expected_mb = (100 * 100 * 4) / (1024 * 1024)  # 4 bytes per float32
        assert abs(tile.memory_size_mb - expected_mb) < 0.01
        
    def test_is_empty_detection(self):
        """Test empty tile detection."""
        # Tile with all nodata values
        data = np.full((10, 10), -9999, dtype=np.float32)
        tile = RasterTile(data, (0, 0, 1, 1), "empty", nodata=-9999)
        assert tile.is_empty()
        
        # Tile with some valid data
        data[5, 5] = 100
        tile = RasterTile(data, (0, 0, 1, 1), "not_empty", nodata=-9999)
        assert not tile.is_empty()
        
    def test_tile_with_different_dtypes(self):
        """Test tiles with different data types."""
        dtypes = [np.uint8, np.int16, np.uint16, np.int32, np.float32, np.float64]
        
        for dtype in dtypes:
            data = np.ones((10, 10), dtype=dtype)
            tile = RasterTile(data, (0, 0, 1, 1), f"test_{dtype}")
            
            assert tile.dtype == dtype
            assert tile.data.dtype == dtype
            
    def test_memory_calculation_accuracy(self):
        """Test memory calculation accuracy for different data types."""
        data_uint8 = np.ones((100, 100), dtype=np.uint8)
        data_float64 = np.ones((100, 100), dtype=np.float64)
        
        tile_uint8 = RasterTile(data_uint8, (0, 0, 1, 1), "uint8")
        tile_float64 = RasterTile(data_float64, (0, 0, 1, 1), "float64")
        
        # float64 should use 8x more memory than uint8
        ratio = tile_float64.memory_size_mb / tile_uint8.memory_size_mb
        assert abs(ratio - 8.0) < 0.1


class TestTileProgressCore:
    """Test TileProgress functionality."""
    
    def test_tile_progress_creation(self):
        """Test creating tile progress."""
        progress = TileProgress(
            tile_id="tile_001",
            status=ProcessingStatus.PROCESSING,
            progress_percentage=50.0,
            memory_usage_mb=25.5
        )
        
        assert progress.tile_id == "tile_001"
        assert progress.status == ProcessingStatus.PROCESSING
        assert progress.progress_percentage == 50.0
        assert progress.memory_usage_mb == 25.5
        
    def test_processing_time_calculation(self):
        """Test processing time calculation."""
        progress = TileProgress("test")
        
        # No start time
        assert progress.processing_time_seconds is None
        
        # With start time
        progress.start_time = time.time()
        time.sleep(0.1)
        progress.end_time = time.time()
        
        processing_time = progress.processing_time_seconds
        assert processing_time is not None
        assert processing_time >= 0.1
        
    def test_completion_status(self):
        """Test completion status checking."""
        progress = TileProgress("test")
        
        # Initially not complete
        assert not progress.is_complete
        
        # Completed
        progress.status = ProcessingStatus.COMPLETED
        assert progress.is_complete
        
        # Failed
        progress.status = ProcessingStatus.FAILED
        assert progress.is_complete
        
        # Skipped
        progress.status = ProcessingStatus.SKIPPED
        assert progress.is_complete
        
    def test_progress_lifecycle(self):
        """Test complete progress lifecycle."""
        progress = TileProgress("lifecycle_test")
        
        # Initial state
        assert progress.status == ProcessingStatus.PENDING
        assert progress.start_time is None
        assert progress.end_time is None
        assert not progress.is_complete
        
        # Start processing
        progress.status = ProcessingStatus.PROCESSING
        progress.start_time = time.time()
        progress.progress_percentage = 25.0
        
        assert not progress.is_complete
        assert progress.processing_time_seconds is not None
        
        # Update progress
        progress.progress_percentage = 75.0
        progress.memory_usage_mb = 15.5
        
        # Complete processing
        progress.status = ProcessingStatus.COMPLETED
        progress.end_time = time.time()
        progress.progress_percentage = 100.0
        
        assert progress.is_complete
        assert progress.processing_time_seconds > 0


class TestProcessingStatusEnum:
    """Test ProcessingStatus enum."""
    
    def test_status_values(self):
        """Test processing status enum values."""
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
        assert ProcessingStatus.SKIPPED.value == "skipped"


class TestMemoryTrackerCore:
    """Test enhanced memory tracker."""
    
    def test_memory_tracker_singleton(self):
        """Test memory tracker singleton pattern."""
        tracker1 = get_memory_tracker()
        tracker2 = get_memory_tracker()
        
        assert tracker1 is tracker2
        
    def test_memory_snapshot(self):
        """Test memory snapshot creation."""
        tracker = get_memory_tracker()
        snapshot = tracker.get_current_snapshot()
        
        assert snapshot is not None
        assert snapshot.timestamp is not None
        assert snapshot.heap_memory_mb >= 0
        assert snapshot.system_available_mb > 0
        
    def test_memory_prediction(self):
        """Test memory usage prediction."""
        tracker = get_memory_tracker()
        
        prediction = tracker.predict_memory_usage(
            data_size_mb=100.0,
            operation_type="test_operation"
        )
        
        assert prediction.predicted_peak_mb > 0
        assert prediction.predicted_duration_seconds > 0
        assert 0.0 <= prediction.confidence <= 1.0
        
    def test_mapped_file_tracking(self):
        """Test memory-mapped file tracking."""
        tracker = get_memory_tracker()
        
        # Track a file
        tracker.track_mapped_file("/test/file.dat", 50.0)
        
        snapshot = tracker.get_current_snapshot()
        assert snapshot.mapped_files_mb >= 50.0
        
        # Untrack the file
        tracker.untrack_mapped_file("/test/file.dat")
        
        snapshot = tracker.get_current_snapshot()
        assert "/test/file.dat" not in tracker._mapped_files


class TestThreadSafetyCore:
    """Test thread safety of core components."""
    
    def test_memory_tracker_thread_safety(self):
        """Test memory tracker thread safety."""
        tracker = get_memory_tracker()
        results = []
        errors = []
        
        def worker():
            try:
                # Each thread tracks some files
                thread_id = threading.current_thread().ident
                for i in range(5):
                    tracker.track_mapped_file(f"/test/file_{thread_id}_{i}.dat", 1.0)
                    
                snapshot = tracker.get_current_snapshot()
                results.append(snapshot.mapped_files_mb)
                
                # Clean up
                for i in range(5):
                    tracker.untrack_mapped_file(f"/test/file_{thread_id}_{i}.dat")
                    
            except Exception as e:
                errors.append(e)
                
        # Run multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should complete without errors
        assert len(errors) == 0
        assert len(results) > 0


class TestMemoryManagementIntegration:
    """Test memory management integration scenarios."""
    
    def test_memory_tracking_integration(self):
        """Test memory tracking with tile processing."""
        tracker = get_memory_tracker()
        
        # Get initial memory state
        initial_snapshot = tracker.get_current_snapshot()
        
        # Create tiles and track their memory usage
        tiles = []
        for i in range(5):
            # Create progressively larger tiles
            size = 50 + i * 25
            data = np.random.rand(size, size).astype(np.float32)
            tile = RasterTile(data, (i, 0, i+1, 1), f"memory_test_{i}")
            tiles.append(tile)
            
        # Calculate total memory usage
        total_memory = sum(tile.memory_size_mb for tile in tiles)
        
        # Get memory state after allocation
        after_snapshot = tracker.get_current_snapshot()
        
        # Memory usage should be tracked
        assert total_memory > 0
        assert len(tiles) == 5
        
        # Clean up
        del tiles
        
    def test_tile_progress_tracking_integration(self):
        """Test tile progress tracking integration."""
        progress_items = []
        
        # Create multiple tile progress trackers
        for i in range(10):
            progress = TileProgress(f"tile_{i}")
            progress.status = ProcessingStatus.PROCESSING
            progress.start_time = time.time()
            progress.progress_percentage = i * 10.0
            progress.memory_usage_mb = i * 2.5
            
            progress_items.append(progress)
            
        # Verify all are tracked correctly
        assert len(progress_items) == 10
        
        for i, progress in enumerate(progress_items):
            assert progress.tile_id == f"tile_{i}"
            assert progress.progress_percentage == i * 10.0
            assert progress.memory_usage_mb == i * 2.5
            
        # Complete some tiles
        for i in [2, 5, 8]:
            progress_items[i].status = ProcessingStatus.COMPLETED
            progress_items[i].end_time = time.time()
            progress_items[i].progress_percentage = 100.0
            
        completed_count = sum(1 for p in progress_items if p.is_complete)
        assert completed_count == 3
        
    def test_error_handling_scenarios(self):
        """Test error handling in enhanced components."""
        # Test with invalid tile data scenarios
        try:
            # Create tile with minimal data
            data = np.array([[1]], dtype=np.float32)
            tile = RasterTile(data, (0, 0, 1, 1), "minimal")
            
            # Should still work
            assert tile.tile_id == "minimal"
            assert tile.memory_size_mb > 0
            
        except Exception as e:
            # If it fails, should be a reasonable error
            assert isinstance(e, (ValueError, IndexError))
            
        # Test progress with error states
        progress = TileProgress("error_test")
        progress.status = ProcessingStatus.FAILED
        progress.error_message = "Simulated processing error"
        
        assert progress.is_complete
        assert progress.error_message is not None
        
    def test_concurrent_processing_simulation(self):
        """Test concurrent processing scenarios."""
        import concurrent.futures
        
        tracker = get_memory_tracker()
        results = []
        
        def worker_task(worker_id: int):
            """Simulate a processing task."""
            try:
                # Create some data
                data = np.random.rand(50, 50).astype(np.float32)
                tile = RasterTile(data, (worker_id, 0, worker_id+1, 1), f"worker_{worker_id}")
                
                # Track progress
                progress = TileProgress(f"worker_{worker_id}")
                progress.status = ProcessingStatus.PROCESSING
                progress.start_time = time.time()
                progress.memory_usage_mb = tile.memory_size_mb
                
                # Simulate processing time
                time.sleep(0.01)
                
                # Complete
                progress.status = ProcessingStatus.COMPLETED
                progress.end_time = time.time()
                progress.progress_percentage = 100.0
                
                return {
                    'worker_id': worker_id,
                    'tile_memory': tile.memory_size_mb,
                    'processing_time': progress.processing_time_seconds,
                    'success': True
                }
                
            except Exception as e:
                return {'worker_id': worker_id, 'error': str(e), 'success': False}
                
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_task, i) for i in range(10)]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
        # All workers should complete successfully
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        assert len(successful) == 10
        assert len(failed) == 0
        
        # All should have reasonable processing times and memory usage
        for result in successful:
            assert result['tile_memory'] > 0
            assert result['processing_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
