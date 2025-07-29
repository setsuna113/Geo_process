"""Tests for unified monitoring system."""
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.monitoring.unified_monitor import UnifiedMonitor
from src.infrastructure.monitoring.database_progress_backend import DatabaseProgressBackend


class TestUnifiedMonitor:
    """Test unified monitoring functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            'monitoring.metrics_interval': 1.0,
            'monitoring.progress_update_interval': 0.5,
            'monitoring.enable_metrics': True
        }.get(key, default)
        return config
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        mock_db = Mock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        mock_db.get_cursor.return_value = mock_cursor
        return mock_db, mock_cursor
    
    @pytest.fixture
    def monitor(self, mock_config, mock_db_manager):
        """Create monitor instance."""
        mock_db, _ = mock_db_manager
        monitor = UnifiedMonitor(mock_config, mock_db)
        yield monitor
        # Cleanup
        monitor.stop()
    
    def test_monitor_initialization(self, monitor):
        """Test monitor is properly initialized."""
        assert monitor.metrics_interval == 1.0
        assert monitor.progress_update_interval == 0.5
        assert monitor.enable_metrics is True
        assert monitor._monitoring_thread is None
        assert not monitor._stop_monitoring.is_set()
    
    def test_start_monitoring(self, monitor):
        """Test starting monitoring."""
        monitor.start("test-exp-123", "job-456")
        
        assert monitor.experiment_id == "test-exp-123"
        assert monitor.job_id == "job-456"
        assert monitor._monitoring_thread is not None
        assert monitor._monitoring_thread.is_alive()
        
        # Stop monitoring
        monitor.stop()
        assert monitor._stop_monitoring.is_set()
        assert not monitor._monitoring_thread.is_alive()
    
    def test_track_stage_context(self, mock_db_manager, monitor):
        """Test stage tracking context manager."""
        mock_db, mock_cursor = mock_db_manager
        
        monitor.start("test-exp", "test-job")
        
        # Track a stage
        with monitor.track_stage("data_loading") as progress:
            # Should create progress node
            assert mock_cursor.execute.called
            
            # Update progress
            progress.update(50, 100, "Processing...")
            
            # Should update in database
            execute_calls = mock_cursor.execute.call_args_list
            assert any("UPDATE pipeline_progress" in str(call) for call in execute_calls)
        
        # Stage should be marked complete
        assert any("status = 'completed'" in str(call) for call in mock_cursor.execute.call_args_list)
        
        monitor.stop()
    
    def test_record_metrics(self, mock_db_manager, monitor):
        """Test metric recording."""
        mock_db, mock_cursor = mock_db_manager
        
        monitor.start("test-exp", "test-job")
        
        # Record metrics
        monitor.record_metrics(
            memory_mb=1024.5,
            cpu_percent=75.3,
            throughput_per_sec=100.0,
            custom_metric=42
        )
        
        # Should insert into database
        assert mock_cursor.execute.called
        sql = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO pipeline_metrics" in sql
        
        # Check values
        values = mock_cursor.execute.call_args[0][1]
        assert values['memory_mb'] == 1024.5
        assert values['cpu_percent'] == 75.3
        assert values['throughput_per_sec'] == 100.0
        
        monitor.stop()
    
    def test_background_monitoring(self, mock_db_manager, monitor):
        """Test background resource monitoring."""
        mock_db, mock_cursor = mock_db_manager
        
        # Mock psutil
        with patch('psutil.Process') as mock_process_class:
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(rss=1024*1024*512)  # 512MB
            mock_process.cpu_percent.return_value = 25.5
            mock_process_class.return_value = mock_process
            
            # Start monitoring with short interval
            monitor.metrics_interval = 0.1
            monitor.start("test-exp", "test-job")
            
            # Wait for at least one monitoring cycle
            time.sleep(0.3)
            
            # Should have recorded system metrics
            execute_calls = mock_cursor.execute.call_args_list
            metrics_calls = [call for call in execute_calls 
                           if "INSERT INTO pipeline_metrics" in str(call)]
            
            assert len(metrics_calls) > 0
            
            # Check that system metrics were recorded
            for call in metrics_calls:
                values = call[0][1]
                if 'memory_mb' in values:
                    assert values['memory_mb'] == 512.0
                if 'cpu_percent' in values:
                    assert values['cpu_percent'] == 25.5
        
        monitor.stop()
    
    def test_monitoring_client_interface(self, mock_db_manager):
        """Test monitoring client query interface."""
        from src.infrastructure.monitoring.monitoring_client import MonitoringClient
        
        mock_db, mock_cursor = mock_db_manager
        client = MonitoringClient(mock_db)
        
        # Test get_experiment_status
        mock_cursor.fetchone.return_value = {
            'id': 'exp-123',
            'name': 'test_exp',
            'status': 'running',
            'started_at': datetime.now(),
            'completed_at': None
        }
        mock_cursor.fetchall.return_value = []
        
        status = client.get_experiment_status('test_exp')
        assert status['name'] == 'test_exp'
        assert status['status'] == 'running'
        
        # Test query_logs
        mock_cursor.fetchall.return_value = [
            {
                'id': 'log-1',
                'timestamp': datetime.now(),
                'level': 'ERROR',
                'message': 'Test error',
                'stage': 'processing',
                'traceback': 'Traceback...'
            }
        ]
        
        logs = client.query_logs(experiment_id='exp-123', level='ERROR')
        assert len(logs) == 1
        assert logs[0]['level'] == 'ERROR'
        assert logs[0]['message'] == 'Test error'
    
    def test_progress_aggregation(self, mock_db_manager):
        """Test progress aggregation for hierarchical nodes."""
        mock_db, mock_cursor = mock_db_manager
        backend = DatabaseProgressBackend(mock_db, "test-exp")
        
        # Create hierarchical structure
        backend.create_node("pipeline", None, "pipeline", "Main Pipeline", 100)
        backend.create_node("pipeline/stage1", "pipeline", "stage", "Stage 1", 50)
        backend.create_node("pipeline/stage2", "pipeline", "stage", "Stage 2", 50)
        
        # Update children
        backend.update_progress("pipeline/stage1", 50, "completed")
        backend.update_progress("pipeline/stage2", 25, "running")
        
        # Check SQL calls for aggregation
        execute_calls = mock_cursor.execute.call_args_list
        
        # Should have CREATE and UPDATE calls
        create_calls = [call for call in execute_calls if "INSERT INTO pipeline_progress" in str(call)]
        update_calls = [call for call in execute_calls if "UPDATE pipeline_progress" in str(call)]
        
        assert len(create_calls) == 3  # 3 nodes created
        assert len(update_calls) == 2  # 2 progress updates
    
    def test_error_handling(self, mock_db_manager, monitor):
        """Test error handling in monitoring."""
        mock_db, mock_cursor = mock_db_manager
        
        # Make database operations fail
        mock_cursor.execute.side_effect = Exception("Database error")
        
        # Should not crash when starting
        monitor.start("test-exp", "test-job")
        
        # Try to record metrics - should handle error gracefully
        monitor.record_metrics(memory_mb=100)
        
        # Stop should work even with errors
        monitor.stop()
        assert monitor._stop_monitoring.is_set()
    
    def test_concurrent_metric_recording(self, mock_db_manager, monitor):
        """Test thread-safe metric recording."""
        mock_db, mock_cursor = mock_db_manager
        
        monitor.start("test-exp", "test-job")
        
        # Record metrics from multiple threads
        def record_worker(worker_id):
            for i in range(5):
                monitor.record_metrics(
                    worker_id=worker_id,
                    iteration=i,
                    value=worker_id * 10 + i
                )
                time.sleep(0.01)
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=record_worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have recorded all metrics without errors
        insert_calls = [call for call in mock_cursor.execute.call_args_list
                       if "INSERT INTO pipeline_metrics" in str(call)]
        
        # 3 workers * 5 iterations = 15 metrics
        assert len(insert_calls) >= 15
        
        monitor.stop()