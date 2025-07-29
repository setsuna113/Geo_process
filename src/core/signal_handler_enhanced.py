"""Enhanced signal handler with structured logging and proper error capture."""

import signal
import sys
import threading
import traceback
from typing import Callable, Optional, Dict, Any
from datetime import datetime

from .signal_handler import SignalHandler
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class EnhancedSignalHandler(SignalHandler):
    """Enhanced signal handler that captures full context and tracebacks.
    
    Features:
    - Structured logging of all signals
    - Full traceback capture on unexpected exits
    - Context preservation for debugging
    - Graceful shutdown with cleanup logging
    """
    
    def __init__(self, cleanup_callback: Optional[Callable] = None):
        """Initialize enhanced signal handler.
        
        Args:
            cleanup_callback: Optional cleanup function to call on shutdown
        """
        super().__init__(cleanup_callback)
        
        # Additional state for enhanced handling
        self._shutdown_context: Dict[str, Any] = {}
        self._unexpected_exit_handler = None
        
        # Register enhanced handlers
        self._register_enhanced_handlers()
        
        logger.info("Enhanced signal handler initialized")
    
    def _register_enhanced_handlers(self):
        """Register additional handlers for better error capture."""
        # Override parent handlers with enhanced versions
        for sig in self._handled_signals:
            signal.signal(sig, self._enhanced_signal_handler)
        
        # Register unexpected exit handler
        sys.excepthook = self._handle_uncaught_exception
        
        # Register thread exception handler (Python 3.8+)
        if hasattr(threading, 'excepthook'):
            threading.excepthook = self._handle_thread_exception
    
    def _enhanced_signal_handler(self, signum: int, frame):
        """Enhanced signal handler with full context logging."""
        signal_name = signal.Signals(signum).name
        
        # Log signal with context
        logger.warning(
            f"Signal {signal_name} received",
            extra={
                'context': {
                    'signal': signal_name,
                    'signal_number': signum,
                    'thread': threading.current_thread().name,
                    'shutdown_requested': self._shutdown_requested,
                    'frame_info': {
                        'filename': frame.f_code.co_filename,
                        'function': frame.f_code.co_name,
                        'line': frame.f_lineno
                    } if frame else None
                }
            }
        )
        
        # Special handling for different signals
        if signum == signal.SIGTERM:
            self._handle_termination()
        elif signum == signal.SIGINT:
            self._handle_interruption()
        elif signum in (signal.SIGUSR1, signal.SIGUSR2):
            self._handle_user_signal(signum)
        
        # Call parent handler
        super()._signal_handler(signum, frame)
    
    def _handle_termination(self):
        """Handle SIGTERM with proper logging."""
        logger.info(
            "Received termination signal, initiating graceful shutdown",
            extra={
                'context': {
                    'signal': 'SIGTERM',
                    'action': 'graceful_shutdown'
                }
            }
        )
        
        # Set context for cleanup
        self._shutdown_context['reason'] = 'SIGTERM'
        self._shutdown_context['timestamp'] = datetime.utcnow().isoformat()
    
    def _handle_interruption(self):
        """Handle SIGINT (Ctrl+C) with proper logging."""
        if self._shutdown_requested:
            logger.warning(
                "Received second interrupt, forcing immediate shutdown",
                extra={
                    'context': {
                        'signal': 'SIGINT',
                        'action': 'force_shutdown'
                    }
                }
            )
            # Force exit
            sys.exit(130)  # 128 + SIGINT
        else:
            logger.info(
                "Received interrupt signal, initiating graceful shutdown",
                extra={
                    'context': {
                        'signal': 'SIGINT',
                        'action': 'graceful_shutdown'
                    }
                }
            )
            self._shutdown_context['reason'] = 'SIGINT'
    
    def _handle_user_signal(self, signum: int):
        """Handle SIGUSR1/SIGUSR2 for custom actions."""
        signal_name = signal.Signals(signum).name
        
        if signum == signal.SIGUSR1:
            # Could trigger memory dump, status report, etc.
            logger.info(
                f"Received {signal_name} - triggering status report",
                extra={'context': {'signal': signal_name}}
            )
            self._log_process_status()
        else:
            # SIGUSR2 - could trigger different action
            logger.info(
                f"Received {signal_name}",
                extra={'context': {'signal': signal_name}}
            )
    
    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions with full logging."""
        # Don't handle KeyboardInterrupt (let it propagate)
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the full exception with traceback
        logger.critical(
            f"Uncaught exception: {exc_type.__name__}: {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback),
            extra={
                'context': {
                    'exception_type': exc_type.__name__,
                    'exception_module': exc_type.__module__,
                    'thread': threading.current_thread().name,
                    'is_daemon_thread': threading.current_thread().daemon
                }
            }
        )
        
        # Call cleanup if not already shutting down
        if not self._shutdown_requested:
            self._shutdown_requested = True
            if self._cleanup_callback:
                try:
                    logger.info("Executing emergency cleanup after uncaught exception")
                    self._cleanup_callback()
                except Exception as e:
                    logger.error(f"Cleanup failed during exception handling: {e}")
        
        # Call default handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    def _handle_thread_exception(self, args):
        """Handle exceptions in threads (Python 3.8+)."""
        logger.error(
            f"Exception in thread '{args.thread.name}'",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            extra={
                'context': {
                    'thread_name': args.thread.name,
                    'thread_id': args.thread.ident,
                    'is_daemon': args.thread.daemon,
                    'exception_type': args.exc_type.__name__
                }
            }
        )
    
    def _log_process_status(self):
        """Log current process status for debugging."""
        import psutil
        
        try:
            process = psutil.Process()
            
            # Gather process info
            status_info = {
                'pid': process.pid,
                'name': process.name(),
                'status': process.status(),
                'cpu_percent': process.cpu_percent(interval=0.1),
                'memory_info': {
                    'rss_mb': process.memory_info().rss / 1024 / 1024,
                    'vms_mb': process.memory_info().vms / 1024 / 1024,
                    'percent': process.memory_percent()
                },
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None,
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
            }
            
            # Log thread info
            thread_info = []
            for thread in threading.enumerate():
                thread_info.append({
                    'name': thread.name,
                    'alive': thread.is_alive(),
                    'daemon': thread.daemon,
                    'ident': thread.ident
                })
            
            status_info['threads'] = thread_info
            
            logger.info(
                "Process status report",
                extra={'context': status_info}
            )
            
        except Exception as e:
            logger.error(f"Failed to gather process status: {e}")
    
    def cleanup(self):
        """Perform cleanup with enhanced logging."""
        if self._cleanup_callback:
            logger.info(
                "Executing cleanup callback",
                extra={'context': self._shutdown_context}
            )
            
            try:
                self._cleanup_callback()
                logger.info("Cleanup completed successfully")
            except Exception as e:
                logger.error(
                    "Cleanup callback failed",
                    exc_info=True,
                    extra={
                        'context': {
                            'error_type': type(e).__name__,
                            'shutdown_context': self._shutdown_context
                        }
                    }
                )
                raise
    
    def set_context(self, **kwargs):
        """Set additional context for shutdown logging.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self._shutdown_context.update(kwargs)


def create_enhanced_signal_handler(cleanup_callback: Optional[Callable] = None) -> EnhancedSignalHandler:
    """Create and return an enhanced signal handler instance.
    
    Args:
        cleanup_callback: Optional cleanup function
        
    Returns:
        EnhancedSignalHandler instance
    """
    return EnhancedSignalHandler(cleanup_callback)