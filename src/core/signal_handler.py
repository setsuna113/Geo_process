# src/core/signal_handler.py
"""Process signal management for the biodiversity pipeline."""

import signal
import logging
import threading
from typing import Dict, Callable, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class SignalHandler:
    """
    Centralized signal handling for graceful shutdown and process control.
    
    Features:
    - Graceful shutdown on SIGTERM/SIGINT
    - Pause/resume on SIGUSR1/SIGUSR2
    - Progress report trigger
    - Integration with checkpoint system
    """
    
    _instance: Optional['SignalHandler'] = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        """Initialize signal handler."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            
            # Signal handlers registry
            self._handlers: Dict[str, List[Callable]] = {}
            self._handler_lock = threading.RLock()
            
            # Original signal handlers
            self._original_handlers: Dict[signal.Signals, Any] = {}
            
            # State
            self._signals_registered = False
            self._shutdown_in_progress = False
            self._pause_requested = False
            
            # Callbacks
            self._shutdown_callbacks: List[Callable] = []
            self._pause_callbacks: List[Callable] = []
            self._resume_callbacks: List[Callable] = []
            
            logger.info("Signal handler initialized")
    
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
            
            # Register system signal handlers if not done
            if not self._signals_registered:
                self._register_system_handlers()
    
    def unregister_handler(self, name: str) -> None:
        """Unregister a signal handler."""
        with self._handler_lock:
            self._handlers.pop(name, None)
    
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
        return self._shutdown_in_progress
    
    @property
    def is_paused(self) -> bool:
        """Check if pause has been requested."""
        return self._pause_requested
    
    @property
    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_in_progress
    
    def is_pause_requested(self) -> bool:
        """Check if pause has been requested."""
        return self._pause_requested
    
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
        """Handle shutdown signals."""
        if self._shutdown_in_progress:
            logger.warning("Shutdown already in progress")
            return
        
        self._shutdown_in_progress = True
        logger.info(f"Received shutdown signal: {sig.name if hasattr(sig, 'name') else sig}")
        
        # Notify all registered handlers
        with self._handler_lock:
            for handler_list in self._handlers.values():
                for handler in handler_list:
                    try:
                        handler(sig)
                    except Exception as e:
                        logger.error(f"Error in signal handler: {e}")
        
        # Call shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in shutdown callback: {e}")
        
        # Restore original handler and re-raise if needed
        if sig in self._original_handlers:
            signal.signal(sig, self._original_handlers[sig])
            if frame is not None:
                self._original_handlers[sig](sig, frame)
    
    def _handle_control_signal(self, sig) -> None:
        """Handle control signals (SIGUSR1/SIGUSR2)."""
        if sig == signal.SIGUSR1:
            self._handle_pause_signal()
        elif sig == signal.SIGUSR2:
            self._handle_resume_signal()
        
        # Also notify registered handlers
        with self._handler_lock:
            for handler_list in self._handlers.values():
                for handler in handler_list:
                    try:
                        handler(sig)
                    except Exception as e:
                        logger.error(f"Error in signal handler: {e}")
    
    def _handle_pause_signal(self) -> None:
        """Handle pause signal."""
        if self._pause_requested:
            logger.info("Already paused")
            return
        
        self._pause_requested = True
        logger.info("Processing paused")
        
        # Call pause callbacks
        for callback in self._pause_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in pause callback: {e}")
    
    def _handle_resume_signal(self) -> None:
        """Handle resume signal."""
        if not self._pause_requested:
            logger.info("Not paused")
            return
        
        self._pause_requested = False
        logger.info("Processing resumed")
        
        # Call resume callbacks
        for callback in self._resume_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in resume callback: {e}")
    
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


# Global signal handler instance
_signal_handler: Optional[SignalHandler] = None


def get_signal_handler() -> SignalHandler:
    """Get the global signal handler instance."""
    global _signal_handler
    if _signal_handler is None:
        _signal_handler = SignalHandler()
    return _signal_handler