# src/core/signal_handler.py
"""Process signal management for the biodiversity pipeline."""

import signal
import logging
import threading
import queue
import time
from typing import Dict, Callable, Optional, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SignalEvent:
    """Signal event for deferred processing outside signal context."""
    signal_num: signal.Signals
    timestamp: float
    handler_name: Optional[str] = None


class SignalHandler:
    """
    Centralized signal handling for graceful shutdown and process control.
    
    Features:
    - Graceful shutdown on SIGTERM/SIGINT
    - Pause/resume on SIGUSR1/SIGUSR2
    - Progress report trigger
    - Integration with checkpoint system
    
    Note: Singleton pattern removed to support dependency injection.
    Use dependency injection container or pass instances explicitly.
    """
    
    def __init__(self):
        """Initialize signal handler."""
        # Signal handlers registry
        self._handlers: Dict[str, List[Callable]] = {}
        self._handler_lock = threading.RLock()
        
        # Thread-safe snapshot for signal context (avoid locks in signal handlers)
        self._handlers_snapshot: List[Callable] = []
        
        # Original signal handlers
        self._original_handlers: Dict[signal.Signals, Any] = {}
        
        # Atomic state flags (safe for signal context)
        self._signals_registered = False
        self._shutdown_in_progress = threading.Event()
        self._pause_requested = threading.Event()
        
        # Deferred signal processing queue (thread-safe)
        self._signal_queue: queue.Queue[SignalEvent] = queue.Queue()
        self._processor_thread: Optional[threading.Thread] = None
        self._processor_running = threading.Event()
        
        # Callbacks (processed outside signal context)
        self._shutdown_callbacks: List[Callable] = []
        self._pause_callbacks: List[Callable] = []
        self._resume_callbacks: List[Callable] = []
        
        # Start deferred signal processor
        self._start_signal_processor()
        
        logger.info("Signal handler initialized")
    
    def _start_signal_processor(self) -> None:
        """Start the deferred signal processing thread."""
        if self._processor_thread is None or not self._processor_thread.is_alive():
            self._processor_running.set()
            self._processor_thread = threading.Thread(
                target=self._process_signals_loop,
                name="SignalProcessor",
                daemon=True
            )
            self._processor_thread.start()
    
    def _process_signals_loop(self) -> None:
        """Main loop for processing signals outside signal context."""
        logger.debug("Signal processor thread started")
        
        while self._processor_running.is_set():
            try:
                # Wait for signal events with timeout
                event = self._signal_queue.get(timeout=1.0)
                self._process_signal_event(event)
                self._signal_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing signal event: {e}")
        
        logger.debug("Signal processor thread stopped")
    
    def _process_signal_event(self, event: SignalEvent) -> None:
        """Process a signal event outside signal context (safe for complex operations)."""
        try:
            # Process callbacks based on signal type
            if event.signal_num in [signal.SIGTERM, signal.SIGINT]:
                self._handle_shutdown_event(event)
            elif event.signal_num == signal.SIGUSR1:
                self._handle_pause_event(event)
            elif event.signal_num == signal.SIGUSR2:
                self._handle_resume_event(event)
            
            # Notify registered handlers using thread-safe snapshot
            for handler in self._handlers_snapshot:
                try:
                    handler(event.signal_num)
                except Exception as e:
                    logger.error(f"Error in signal handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing signal event: {e}")
    
    def _handle_shutdown_event(self, event: SignalEvent) -> None:
        """Handle shutdown signal event (outside signal context)."""
        if self._shutdown_in_progress.is_set():
            logger.warning("Shutdown already in progress")
            return
        
        self._shutdown_in_progress.set()
        logger.info(f"Processing shutdown signal: {event.signal_num.name if hasattr(event.signal_num, 'name') else event.signal_num}")
        
        # Call shutdown callbacks safely
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")
    
    def _handle_pause_event(self, event: SignalEvent) -> None:
        """Handle pause signal event (outside signal context)."""
        if self._pause_requested.is_set():
            logger.warning("Already paused")
            return
        
        self._pause_requested.set()
        logger.info("Processing pause signal")
        
        for callback in self._pause_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in pause callback: {e}")
    
    def _handle_resume_event(self, event: SignalEvent) -> None:
        """Handle resume signal event (outside signal context)."""
        if not self._pause_requested.is_set():
            logger.warning("Not currently paused")
            return
        
        self._pause_requested.clear()
        logger.info("Processing resume signal")
        
        for callback in self._resume_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in resume callback: {e}")
    
    def shutdown(self) -> None:
        """Clean shutdown of signal handler."""
        logger.info("Shutting down signal handler")
        
        # Stop processor thread
        self._processor_running.clear()
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=2.0)
        
        # Restore original signal handlers
        for sig, original_handler in self._original_handlers.items():
            try:
                signal.signal(sig, original_handler)
            except Exception as e:
                logger.warning(f"Failed to restore signal handler for {sig}: {e}")
    
    def register_handler(self, name: str, handler: Callable[[signal.Signals], None]) -> None:
        """
        Register a signal handler.
        
        Args:
            name: Handler name
            handler: Handler function(signal)
        """
        with self._handler_lock:
            if name not in self._handlers:
                self._handlers[name] = []
            self._handlers[name].append(handler)
            
            # Update thread-safe snapshot
            self._update_handlers_snapshot()
            
            # Register system signal handlers if not done
            if not self._signals_registered:
                self._register_system_handlers()
    
    def unregister_handler(self, name: str) -> None:
        """Unregister a signal handler."""
        with self._handler_lock:
            self._handlers.pop(name, None)
            # Update thread-safe snapshot
            self._update_handlers_snapshot()
    
    def _update_handlers_snapshot(self) -> None:
        """Update the handlers snapshot for thread-safe signal context access."""
        # Must be called with _handler_lock held
        all_handlers = []
        for handler_list in self._handlers.values():
            all_handlers.extend(handler_list)
        self._handlers_snapshot = all_handlers
    
    def register_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for shutdown."""
        self._shutdown_callbacks.append(callback)
    
    def register_pause_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for pause."""
        self._pause_callbacks.append(callback)
    
    def register_resume_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for resume."""
        self._resume_callbacks.append(callback)
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_in_progress.is_set()
    
    @property
    def is_paused(self) -> bool:
        """Check if pause has been requested."""
        return self._pause_requested.is_set()
    
    @property
    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_in_progress.is_set()
    
    def is_pause_requested(self) -> bool:
        """Check if pause has been requested."""
        return self._pause_requested.is_set()
    
    def trigger_shutdown(self) -> None:
        """Manually trigger shutdown."""
        logger.info("Manual shutdown triggered")
        self._handle_shutdown_signal(signal.SIGTERM)
    
    def trigger_pause(self) -> None:
        """Manually trigger pause."""
        logger.info("Manual pause triggered")
        self._handle_pause_signal()
    
    def trigger_resume(self) -> None:
        """Manually trigger resume."""
        logger.info("Manual resume triggered")
        self._handle_resume_signal()
    
    def _register_system_handlers(self) -> None:
        """Register system signal handlers."""
        try:
            # Shutdown signals
            for sig in [signal.SIGTERM, signal.SIGINT]:
                self._original_handlers[sig] = signal.signal(sig, self._handle_shutdown_signal)
            
            # Control signals (Unix only)
            if hasattr(signal, 'SIGUSR1'):
                self._original_handlers[signal.SIGUSR1] = signal.signal(
                    signal.SIGUSR1, 
                    lambda sig, frame: self._handle_control_signal(sig)
                )
            
            if hasattr(signal, 'SIGUSR2'):
                self._original_handlers[signal.SIGUSR2] = signal.signal(
                    signal.SIGUSR2,
                    lambda sig, frame: self._handle_control_signal(sig)
                )
            
            self._signals_registered = True
            logger.info("System signal handlers registered")
            
        except Exception as e:
            logger.error(f"Failed to register signal handlers: {e}")
    
    def _handle_shutdown_signal(self, sig, frame=None) -> None:
        """Handle shutdown signals - minimal processing in signal context."""
        # Defer complex processing to background thread
        try:
            event = SignalEvent(signal_num=sig, timestamp=time.time())
            self._signal_queue.put_nowait(event)
        except queue.Full:
            # Queue full - this is a critical situation but we can't do much in signal context
            print(f"Signal queue full, dropping signal {sig}")
        except Exception:
            # Avoid any exceptions in signal context
            pass
    
    def _handle_control_signal(self, sig) -> None:
        """Handle control signals - minimal processing in signal context."""
        # Defer complex processing to background thread
        try:
            event = SignalEvent(signal_num=sig, timestamp=time.time())
            self._signal_queue.put_nowait(event)
        except queue.Full:
            print(f"Signal queue full, dropping signal {sig}")
        except Exception:
            # Avoid any exceptions in signal context
            pass
    
    
    def cleanup(self) -> None:
        """Clean up and restore original signal handlers."""
        logger.info("Cleaning up signal handlers")
        
        # Restore original handlers
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception as e:
                logger.error(f"Failed to restore signal handler for {sig}: {e}")
        
        self._original_handlers.clear()
        self._signals_registered = False


# Factory function for creating signal handler instances
# Note: Global state removed - use dependency injection or create instances explicitly
def create_signal_handler() -> SignalHandler:
    """
    Create a new signal handler instance.
    
    Note: This replaces the global singleton pattern. Applications should:
    1. Create instances explicitly when needed
    2. Use dependency injection containers
    3. Pass instances through constructors
    
    Returns:
        New SignalHandler instance
    """
    return SignalHandler()