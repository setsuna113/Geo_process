"""Tests for database log handler."""
import pytest
import json
import logging
import time
import queue
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.logging.handlers.database_handler import DatabaseLogHandler


class TestDatabaseLogHandler:
    """Test database handler for structured logging."""
    
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
    def handler(self, mock_db_manager):
        """Create handler with mock database."""
        mock_db, _ = mock_db_manager
        handler = DatabaseLogHandler(
            db_manager=mock_db,
            batch_size=5,
            flush_interval=0.1
        )
        yield handler
        # Cleanup
        handler.close()
    
    def test_handler_initialization(self, handler):
        """Test handler is properly initialized."""
        assert handler.batch_size == 5
        assert handler.flush_interval == 0.1
        assert handler.queue.maxsize == 10000
        assert handler._worker_thread.is_alive()
    
    def test_emit_queues_record(self, handler):
        """Test that emit queues log records."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add context to record
        record.context = {'experiment_id': 'test-123'}
        record.traceback = None
        record.performance = {}
        
        # Emit record
        handler.emit(record)
        
        # Should be in queue
        assert not handler.queue.empty()
        queued_record = handler.queue.get_nowait()
        assert queued_record.msg == "Test message"
    
    def test_batch_flushing(self, mock_db_manager, handler):
        """Test that batches are flushed to database."""
        mock_db, mock_cursor = mock_db_manager
        
        # Create test records
        records = []
        for i in range(3):
            record = logging.LogRecord(
                name=f"test.logger.{i}",
                level=logging.INFO,
                pathname="test.py",
                lineno=i,
                msg=f"Test message {i}",
                args=(),
                exc_info=None
            )
            record.context = {'experiment_id': f'test-{i}'}
            record.traceback = None
            record.performance = {}
            records.append(record)
            handler.emit(record)
        
        # Wait for batch to be processed
        time.sleep(0.3)  # Longer than flush_interval
        
        # Should have executed batch insert
        assert mock_cursor.executemany.called
        call_args = mock_cursor.executemany.call_args
        
        # Check SQL
        sql = call_args[0][0]
        assert "INSERT INTO pipeline_logs" in sql
        
        # Check values
        values = call_args[0][1]
        assert len(values) == 3
        
        for i, value in enumerate(values):
            assert value['message'] == f"Test message {i}"
            assert json.loads(value['context'])['experiment_id'] == f"test-{i}"
    
    def test_queue_full_handling(self, handler):
        """Test behavior when queue is full."""
        # Fill the queue
        handler.queue.maxsize = 5  # Make it small for testing
        
        for i in range(10):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=f"Message {i}",
                args=(),
                exc_info=None
            )
            handler.emit(record)
        
        # Queue should be at max size
        assert handler.queue.qsize() <= handler.queue.maxsize
    
    def test_error_handling_in_flush(self, mock_db_manager, handler):
        """Test error handling during database flush."""
        mock_db, mock_cursor = mock_db_manager
        
        # Make executemany raise an error
        mock_cursor.executemany.side_effect = Exception("Database error")
        
        # Add records
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg=f"Error {i}",
                args=(),
                exc_info=None
            )
            record.context = {}
            record.traceback = None
            record.performance = {}
            handler.emit(record)
        
        # Wait for flush attempt
        time.sleep(0.3)
        
        # Should have attempted to flush
        assert mock_cursor.executemany.called
        
        # Batch should be cleared despite error
        assert len(handler.batch) == 0
    
    def test_traceback_handling(self, mock_db_manager, handler):
        """Test that tracebacks are properly stored."""
        mock_db, mock_cursor = mock_db_manager
        
        # Create record with traceback
        record = logging.LogRecord(
            name="test.error",
            level=logging.ERROR,
            pathname="test.py",
            lineno=50,
            msg="Error with traceback",
            args=(),
            exc_info=None
        )
        
        # Add traceback
        record.context = {'stage': 'processing'}
        record.traceback = "Traceback (most recent call last):\n  File...\nValueError: Test"
        record.performance = {}
        
        handler.emit(record)
        
        # Wait for flush
        time.sleep(0.3)
        
        # Check that traceback was included
        values = mock_cursor.executemany.call_args[0][1]
        assert len(values) == 1
        assert values[0]['traceback'] == record.traceback
    
    def test_performance_data_handling(self, mock_db_manager, handler):
        """Test that performance data is properly stored."""
        mock_db, mock_cursor = mock_db_manager
        
        # Create record with performance data
        record = logging.LogRecord(
            name="test.perf",
            level=logging.INFO,
            pathname="test.py",
            lineno=100,
            msg="Performance: operation",
            args=(),
            exc_info=None
        )
        
        # Add performance data
        record.context = {}
        record.traceback = None
        record.performance = {
            'operation': 'data_load',
            'duration_seconds': 1.23,
            'items_processed': 1000
        }
        
        handler.emit(record)
        
        # Wait for flush
        time.sleep(0.3)
        
        # Check that performance data was included
        values = mock_cursor.executemany.call_args[0][1]
        assert len(values) == 1
        assert values[0]['performance'] is not None
        
        perf_data = json.loads(values[0]['performance'])
        assert perf_data['operation'] == 'data_load'
        assert perf_data['duration_seconds'] == 1.23
    
    def test_handler_close(self, mock_db_manager):
        """Test handler cleanup on close."""
        mock_db, mock_cursor = mock_db_manager
        
        # Create handler
        handler = DatabaseLogHandler(mock_db, batch_size=10, flush_interval=1.0)
        
        # Add some records
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=f"Message {i}",
                args=(),
                exc_info=None
            )
            record.context = {}
            record.traceback = None
            record.performance = {}
            handler.emit(record)
        
        # Close handler
        handler.close()
        
        # Worker thread should stop
        assert handler._stop_event.is_set()
        assert not handler._worker_thread.is_alive()
        
        # Final flush should have occurred
        assert mock_cursor.executemany.called
    
    def test_timestamp_handling(self, mock_db_manager, handler):
        """Test that timestamps are properly converted."""
        mock_db, mock_cursor = mock_db_manager
        
        # Create record
        record = logging.LogRecord(
            name="test.time",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test timestamp",
            args=(),
            exc_info=None
        )
        record.context = {}
        record.traceback = None
        record.performance = {}
        
        handler.emit(record)
        
        # Wait for flush
        time.sleep(0.3)
        
        # Check timestamp conversion
        values = mock_cursor.executemany.call_args[0][1]
        assert len(values) == 1
        
        timestamp = values[0]['timestamp']
        assert isinstance(timestamp, datetime)
        assert timestamp.timestamp() == pytest.approx(record.created, rel=1e-3)