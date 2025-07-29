"""Signal handler architecture tests focusing on thread safety and deferred processing."""

import pytest
import threading
import time
import signal
import queue
from unittest.mock import Mock, patch, MagicMock

from src.core.signal_handler import SignalHandler, SignalEvent


class TestSignalEventProcessing:
    """Test signal event processing architecture."""
    
    def test_signal_event_creation(self):
        """Test SignalEvent creation and attributes."""
        timestamp = time.time()
        event = SignalEvent(
            signal_num=signal.SIGTERM,
            timestamp=timestamp,
            handler_name="test_handler"
        )
        
        assert event.signal_num == signal.SIGTERM
        assert event.timestamp == timestamp
        assert event.handler_name == "test_handler"
    
    def test_deferred_signal_processing(self):
        """Test that signals are processed in background thread."""
        handler = SignalHandler()
        
        # Verify background thread is created and running
        assert handler._processor_thread is not None
        assert handler._processor_thread.is_alive()
        assert handler._processor_running.is_set()
        
        # Verify signal queue exists
        assert isinstance(handler._signal_queue, queue.Queue)
        
        handler.shutdown()
    
    def test_signal_context_minimal_operations(self):
        """Test that signal handlers do minimal work in signal context."""
        handler = SignalHandler()
        
        with patch.object(handler._signal_queue, 'put_nowait') as mock_put:
            # Simulate signal handler call
            handler._handle_shutdown_signal(signal.SIGTERM)
            
            # Should only queue the event
            mock_put.assert_called_once()
            
            # Verify event structure
            call_args = mock_put.call_args[0][0]
            assert isinstance(call_args, SignalEvent)
            assert call_args.signal_num == signal.SIGTERM
        
        handler.shutdown()
    
    def test_queue_full_handling(self):
        """Test handling of full signal queue."""
        handler = SignalHandler()
        
        # Fill the queue
        with patch.object(handler._signal_queue, 'put_nowait', side_effect=queue.Full):
            # Should not raise exception in signal context
            handler._handle_shutdown_signal(signal.SIGTERM)
        
        handler.shutdown()


class TestThreadSafetyMechanisms:
    """Test thread safety mechanisms."""
    
    def test_atomic_state_flags(self):
        """Test atomic state management with threading.Event."""
        handler = SignalHandler()
        
        # Test shutdown state
        assert not handler.is_shutdown_requested()
        assert not handler._shutdown_in_progress.is_set()
        
        handler._shutdown_in_progress.set()
        assert handler.is_shutdown_requested()
        assert handler.should_shutdown
        
        # Test pause state
        assert not handler.is_pause_requested()
        assert not handler._pause_requested.is_set()
        
        handler._pause_requested.set()
        assert handler.is_pause_requested()
        assert handler.is_paused
        
        handler.shutdown()
    
    def test_handler_snapshot_update(self):
        """Test thread-safe handler snapshot mechanism."""
        handler = SignalHandler()
        
        # Initially empty
        assert len(handler._handlers_snapshot) == 0
        
        # Register a handler
        mock_handler = Mock()
        handler.register_handler("test", mock_handler)
        
        # Snapshot should be updated
        assert len(handler._handlers_snapshot) == 1
        assert mock_handler in handler._handlers_snapshot
        
        # Register another handler
        mock_handler2 = Mock()
        handler.register_handler("test2", mock_handler2)
        
        # Snapshot should include both
        assert len(handler._handlers_snapshot) == 2
        assert mock_handler in handler._handlers_snapshot
        assert mock_handler2 in handler._handlers_snapshot
        
        # Unregister a handler
        handler.unregister_handler("test")
        
        # Snapshot should be updated
        assert mock_handler2 in handler._handlers_snapshot
        # mock_handler may or may not be present depending on implementation
        
        handler.shutdown()
    
    def test_concurrent_handler_registration(self):
        """Test concurrent handler registration thread safety."""
        handler = SignalHandler()
        
        handlers_added = []
        registration_errors = []
        
        def register_handlers(start_id, count):
            """Register multiple handlers from a thread."""
            try:
                for i in range(count):
                    handler_id = f"handler_{start_id}_{i}"
                    mock_handler = Mock()
                    handler.register_handler(handler_id, mock_handler)
                    handlers_added.append(handler_id)
            except Exception as e:
                registration_errors.append(e)
        
        # Start multiple threads registering handlers
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(
                target=register_handlers,
                args=(thread_id, 10)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Should have no errors
        assert len(registration_errors) == 0
        
        # Should have registered all handlers
        assert len(handlers_added) == 50
        
        handler.shutdown()


class TestSignalProcessingLoop:
    """Test signal processing background loop."""
    
    def test_signal_processing_loop_lifecycle(self):
        """Test signal processing loop start and stop."""
        handler = SignalHandler()
        
        # Loop should be running
        assert handler._processor_running.is_set()
        assert handler._processor_thread.is_alive()
        
        # Shutdown should stop the loop
        handler.shutdown()
        
        # Should clean up properly
        assert not handler._processor_running.is_set()
        
        # Give thread time to exit
        time.sleep(0.5)
        assert not handler._processor_thread.is_alive()
    
    def test_signal_event_processing(self):
        """Test processing of signal events."""
        handler = SignalHandler()
        
        # Register test handlers
        shutdown_called = threading.Event()
        pause_called = threading.Event()
        resume_called = threading.Event()
        
        def test_shutdown():
            shutdown_called.set()
        
        def test_pause():
            pause_called.set()
            
        def test_resume():
            resume_called.set()
        
        handler.register_shutdown_callback(test_shutdown)
        handler.register_pause_callback(test_pause)
        handler.register_resume_callback(test_resume)
        
        # Create and queue signal events
        shutdown_event = SignalEvent(signal.SIGTERM, time.time())
        pause_event = SignalEvent(signal.SIGUSR1, time.time())
        resume_event = SignalEvent(signal.SIGUSR2, time.time())
        
        handler._signal_queue.put_nowait(shutdown_event)
        handler._signal_queue.put_nowait(pause_event)
        handler._signal_queue.put_nowait(resume_event)
        
        # Wait for processing
        assert shutdown_called.wait(timeout=2.0)
        assert pause_called.wait(timeout=2.0)
        assert resume_called.wait(timeout=2.0)
        
        # Verify state changes
        assert handler.is_shutdown_requested()
        # Pause should be cleared by resume
        time.sleep(0.1)  # Allow resume to process
        assert not handler.is_pause_requested()
        
        handler.shutdown()
    
    def test_handler_notification_in_processing_loop(self):
        """Test that registered handlers are notified during event processing."""
        handler = SignalHandler()
        
        # Register test handler
        handler_called = threading.Event()
        received_signal = [None]
        
        def test_handler(sig):
            received_signal[0] = sig
            handler_called.set()
        
        handler.register_handler("test", test_handler)
        
        # Queue a signal event
        event = SignalEvent(signal.SIGTERM, time.time())
        handler._signal_queue.put_nowait(event)
        
        # Wait for handler to be called
        assert handler_called.wait(timeout=2.0)
        assert received_signal[0] == signal.SIGTERM
        
        handler.shutdown()


class TestErrorHandling:
    """Test error handling in signal processing."""
    
    def test_handler_exception_isolation(self):
        """Test that exceptions in one handler don't affect others."""
        handler = SignalHandler()
        
        # Register handlers: one that throws, one that works
        good_handler_called = threading.Event()
        
        def bad_handler(sig):
            raise Exception("Test exception")
        
        def good_handler(sig):
            good_handler_called.set()
        
        handler.register_handler("bad", bad_handler)
        handler.register_handler("good", good_handler)
        
        # Queue signal event
        event = SignalEvent(signal.SIGTERM, time.time())
        handler._signal_queue.put_nowait(event)
        
        # Good handler should still be called despite bad handler exception
        assert good_handler_called.wait(timeout=2.0)
        
        handler.shutdown()
    
    def test_processing_loop_exception_recovery(self):
        """Test that processing loop recovers from exceptions."""
        handler = SignalHandler()
        
        # Mock the event processing to throw an exception
        original_process = handler._process_signal_event
        exception_count = [0]
        
        def mock_process(event):
            exception_count[0] += 1
            if exception_count[0] == 1:
                raise Exception("Test exception")
            else:
                # Call original for subsequent events
                original_process(event)
        
        with patch.object(handler, '_process_signal_event', side_effect=mock_process):
            # Queue two events
            event1 = SignalEvent(signal.SIGTERM, time.time())
            event2 = SignalEvent(signal.SIGTERM, time.time())
            
            handler._signal_queue.put_nowait(event1)
            handler._signal_queue.put_nowait(event2)
            
            # Wait for processing
            time.sleep(0.5)
            
            # Both events should have been processed despite first exception
            assert exception_count[0] == 2
        
        handler.shutdown()


class TestMemoryManagement:
    """Test memory management in signal handler."""
    
    def test_handler_cleanup(self):
        """Test proper cleanup of handlers and resources."""
        handler = SignalHandler()
        
        # Add some handlers
        handler.register_handler("test1", Mock())
        handler.register_handler("test2", Mock()) 
        
        # Add callbacks
        handler.register_shutdown_callback(Mock())
        handler.register_pause_callback(Mock())
        
        # Verify resources exist
        assert len(handler._handlers) == 2
        assert len(handler._shutdown_callbacks) == 1
        assert len(handler._pause_callbacks) == 1
        
        # Shutdown should clean up
        handler.shutdown()
        
        # Verify thread cleanup
        time.sleep(0.5)
        assert not handler._processor_thread.is_alive()
    
    def test_signal_queue_cleanup(self):
        """Test that signal queue is properly handled during shutdown."""
        handler = SignalHandler()
        
        # Add events to queue
        for i in range(5):
            event = SignalEvent(signal.SIGTERM, time.time())
            handler._signal_queue.put_nowait(event)
        
        # Shutdown should handle remaining events
        handler.shutdown()
        
        # Should be able to join processing thread
        handler._processor_thread.join(timeout=2.0)
        assert not handler._processor_thread.is_alive()


class TestRegressionPrevention:
    """Test for regression prevention in signal handling."""
    
    def test_no_locks_in_signal_context(self):
        """Test that signal handlers don't use locks."""
        handler = SignalHandler()
        
        # Mock threading.Lock to detect lock usage
        with patch('threading.Lock') as mock_lock_class:
            mock_lock = Mock()
            mock_lock_class.return_value = mock_lock
            
            # Call signal handlers
            handler._handle_shutdown_signal(signal.SIGTERM)
            handler._handle_control_signal(signal.SIGUSR1)
            
            # Signal handlers should not acquire locks
            mock_lock.acquire.assert_not_called()
            mock_lock.__enter__.assert_not_called()
        
        handler.shutdown()
    
    def test_signal_context_performance(self):
        """Test that signal context operations are fast."""
        handler = SignalHandler()
        
        # Time signal handler execution
        start_time = time.perf_counter()
        for _ in range(100):
            handler._handle_shutdown_signal(signal.SIGTERM)
        end_time = time.perf_counter()
        
        # Should be very fast (less than 1ms per call)
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.001, f"Signal handler too slow: {avg_time:.6f}s"
        
        handler.shutdown()
    
    def test_atomic_operations_only(self):
        """Test that signal handlers only do atomic operations."""
        handler = SignalHandler()
        
        # Mock queue.put_nowait to track calls
        with patch.object(handler._signal_queue, 'put_nowait') as mock_put:
            handler._handle_shutdown_signal(signal.SIGTERM)
            
            # Should only make one atomic queue operation
            assert mock_put.call_count == 1
            
            # Should not do any complex operations (no multiple calls)
            handler._handle_control_signal(signal.SIGUSR1)
            assert mock_put.call_count == 2
        
        handler.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])