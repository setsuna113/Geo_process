"""Tests for structured logging functionality."""
import pytest
import json
import logging
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.logging.structured_logger import (
    StructuredLogger, get_logger, experiment_context, 
    node_context, stage_context
)


class TestStructuredLogger:
    """Test structured logging functionality."""
    
    def test_logger_creation(self):
        """Test that get_logger creates StructuredLogger instances."""
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "test.module"
    
    def test_context_variables(self):
        """Test context variable setting and retrieval."""
        # Test experiment context
        assert experiment_context.get() is None
        experiment_context.set("test-exp-123")
        assert experiment_context.get() == "test-exp-123"
        
        # Test node context
        assert node_context.get() is None
        node_context.set("pipeline/stage1")
        assert node_context.get() == "pipeline/stage1"
        
        # Test stage context
        assert stage_context.get() is None
        stage_context.set("data_loading")
        assert stage_context.get() == "data_loading"
        
        # Reset contexts
        experiment_context.set(None)
        node_context.set(None)
        stage_context.set(None)
    
    def test_structured_log_output(self):
        """Test that logs include structured context."""
        logger = get_logger("test.structured")
        
        # Set contexts
        experiment_context.set("exp-456")
        node_context.set("pipeline/test")
        stage_context.set("testing")
        
        # Mock the parent _log method
        with patch.object(logging.Logger, '_log') as mock_log:
            logger.info("Test message")
            
            # Verify call was made
            assert mock_log.called
            call_args = mock_log.call_args
            
            # Check structured extra data
            extra = call_args.kwargs.get('extra', {})
            assert 'context' in extra
            context = extra['context']
            
            assert context['experiment_id'] == "exp-456"
            assert context['node_id'] == "pipeline/test"
            assert context['stage'] == "testing"
            assert context['logger_name'] == "test.structured"
            assert 'timestamp' in context
        
        # Cleanup
        experiment_context.set(None)
        node_context.set(None)
        stage_context.set(None)
    
    def test_traceback_capture(self):
        """Test error traceback capture."""
        logger = get_logger("test.errors")
        
        try:
            # Simulate an error
            raise ValueError("Test error for traceback")
        except ValueError:
            with patch.object(logging.Logger, '_log') as mock_log:
                logger.error("Caught error", exc_info=True)
                
                # Check traceback was captured
                extra = mock_log.call_args.kwargs.get('extra', {})
                assert 'traceback' in extra
                assert extra['traceback'] is not None
                assert "ValueError: Test error for traceback" in extra['traceback']
                assert "test_traceback_capture" in extra['traceback']
    
    def test_add_context(self):
        """Test adding persistent context fields."""
        logger = get_logger("test.context")
        
        # Add context fields
        logger.add_context(user_id="user123", session_id="sess456")
        
        with patch.object(logging.Logger, '_log') as mock_log:
            logger.info("Test with context")
            
            extra = mock_log.call_args.kwargs.get('extra', {})
            context = extra['context']
            
            assert context['user_id'] == "user123"
            assert context['session_id'] == "sess456"
    
    def test_log_performance(self):
        """Test performance logging method."""
        logger = get_logger("test.performance")
        
        with patch.object(logging.Logger, '_log') as mock_log:
            logger.log_performance(
                "data_processing",
                duration=1.234,
                items_processed=1000,
                throughput=810.37
            )
            
            # Verify performance data
            assert mock_log.called
            msg = mock_log.call_args.args[1]
            assert "Performance: data_processing" in msg
            
            extra = mock_log.call_args.kwargs.get('extra', {})
            assert 'performance' in extra
            perf = extra['performance']
            
            assert perf['operation'] == "data_processing"
            assert perf['duration_seconds'] == 1.234
            assert perf['items_processed'] == 1000
            assert perf['throughput'] == 810.37
    
    def test_extra_context_merge(self):
        """Test that extra context merges with persistent context."""
        logger = get_logger("test.merge")
        
        # Set persistent context
        logger.add_context(persistent_field="persistent_value")
        experiment_context.set("exp-789")
        
        with patch.object(logging.Logger, '_log') as mock_log:
            # Log with additional extra context
            logger.info("Test merge", extra={
                'request_id': 'req-123',
                'custom_field': 'custom_value'
            })
            
            extra = mock_log.call_args.kwargs.get('extra', {})
            context = extra['context']
            
            # Should have all contexts merged
            assert context['persistent_field'] == "persistent_value"
            assert context['experiment_id'] == "exp-789"
            assert context['request_id'] == "req-123"
            assert context['custom_field'] == "custom_value"
        
        experiment_context.set(None)
    
    def test_log_levels(self):
        """Test different log levels."""
        logger = get_logger("test.levels")
        
        test_cases = [
            (logger.debug, "Debug message", logging.DEBUG),
            (logger.info, "Info message", logging.INFO),
            (logger.warning, "Warning message", logging.WARNING),
            (logger.error, "Error message", logging.ERROR),
            (logger.critical, "Critical message", logging.CRITICAL)
        ]
        
        for log_method, message, expected_level in test_cases:
            with patch.object(logging.Logger, '_log') as mock_log:
                log_method(message)
                
                assert mock_log.called
                actual_level = mock_log.call_args.args[0]
                assert actual_level == expected_level
                
                actual_msg = mock_log.call_args.args[1]
                assert actual_msg == message


class TestLoggingIntegration:
    """Test integration with other components."""
    
    def test_context_isolation(self):
        """Test that contexts are isolated between operations."""
        logger = get_logger("test.isolation")
        
        # Set context for operation 1
        experiment_context.set("exp-1")
        node_context.set("node-1")
        
        with patch.object(logging.Logger, '_log') as mock_log:
            logger.info("Operation 1")
            context1 = mock_log.call_args.kwargs['extra']['context']
            
            # Change context for operation 2
            experiment_context.set("exp-2")
            node_context.set("node-2")
            
            logger.info("Operation 2")
            context2 = mock_log.call_args.kwargs['extra']['context']
            
            # Contexts should be different
            assert context1['experiment_id'] == "exp-1"
            assert context1['node_id'] == "node-1"
            assert context2['experiment_id'] == "exp-2"
            assert context2['node_id'] == "node-2"
        
        # Cleanup
        experiment_context.set(None)
        node_context.set(None)
    
    def test_thread_safety(self):
        """Test that context variables are thread-safe."""
        import threading
        
        logger = get_logger("test.threads")
        results = {}
        
        def worker(worker_id, exp_id):
            """Worker that sets and logs with its own context."""
            experiment_context.set(exp_id)
            time.sleep(0.01)  # Small delay to encourage race conditions
            
            with patch.object(logging.Logger, '_log') as mock_log:
                logger.info(f"Worker {worker_id}")
                context = mock_log.call_args.kwargs['extra']['context']
                results[worker_id] = context['experiment_id']
        
        # Create multiple threads with different contexts
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=worker, 
                args=(f"worker-{i}", f"exp-thread-{i}")
            )
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Each worker should have logged with its own context
        for i in range(5):
            assert results[f"worker-{i}"] == f"exp-thread-{i}"
        
        # Cleanup
        experiment_context.set(None)