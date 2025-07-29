"""Progress event broadcasting service."""

import threading
import logging
from typing import Dict, Any, List, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProgressEventService:
    """Service for broadcasting progress events to registered callbacks."""
    
    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._global_callbacks: List[Callable] = []
        self._callback_lock = threading.RLock()
    
    def register_callback(self, callback: Callable[[str, Dict[str, Any]], None],
                         node_pattern: str = "*") -> None:
        """Register a progress callback."""
        with self._callback_lock:
            if node_pattern == "*":
                self._global_callbacks.append(callback)
                logger.debug("Registered global progress callback")
            else:
                self._callbacks[node_pattern].append(callback)
                logger.debug(f"Registered progress callback for pattern: {node_pattern}")
    
    def unregister_callback(self, callback: Callable[[str, Dict[str, Any]], None],
                          node_pattern: str = "*") -> bool:
        """Unregister a progress callback."""
        with self._callback_lock:
            try:
                if node_pattern == "*":
                    self._global_callbacks.remove(callback)
                    logger.debug("Unregistered global progress callback")
                else:
                    self._callbacks[node_pattern].remove(callback)
                    logger.debug(f"Unregistered progress callback for pattern: {node_pattern}")
                return True
            except ValueError:
                logger.warning(f"Callback not found for pattern: {node_pattern}")
                return False
    
    def emit_event(self, event_type: str, node_id: str, progress_data: Dict[str, Any]) -> None:
        """Emit a progress event to registered callbacks."""
        with self._callback_lock:
            # Call global callbacks
            for callback in self._global_callbacks:
                try:
                    callback(node_id, progress_data)
                except Exception as e:
                    logger.error(f"Global callback error: {e}")
            
            # Call pattern-specific callbacks
            for pattern, callbacks in self._callbacks.items():
                if self._matches_pattern(node_id, pattern):
                    for callback in callbacks:
                        try:
                            callback(node_id, progress_data)
                        except Exception as e:
                            logger.error(f"Pattern callback error for {pattern}: {e}")
        
        logger.debug(f"Emitted {event_type} event for {node_id}")
    
    def _matches_pattern(self, node_id: str, pattern: str) -> bool:
        """Check if node_id matches the pattern."""
        if pattern == "*":
            return True
        
        # Simple pattern matching (can be enhanced)
        if pattern.endswith("*"):
            return node_id.startswith(pattern[:-1])
        elif pattern.startswith("*"):
            return node_id.endswith(pattern[1:])
        else:
            return node_id == pattern
    
    def clear_callbacks(self) -> None:
        """Clear all registered callbacks."""
        with self._callback_lock:
            self._callbacks.clear()
            self._global_callbacks.clear()
            logger.info("Cleared all progress callbacks")
    
    def get_callback_count(self) -> Dict[str, int]:
        """Get count of registered callbacks."""
        with self._callback_lock:
            return {
                'global': len(self._global_callbacks),
                'pattern_specific': sum(len(callbacks) for callbacks in self._callbacks.values()),
                'total_patterns': len(self._callbacks)
            }