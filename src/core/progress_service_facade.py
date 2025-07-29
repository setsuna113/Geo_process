"""Facade for progress services - maintains backward compatibility.

This facade provides the same interface as the original ProgressManager
but delegates to focused, single-responsibility services.
"""

import threading  
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

from .services import ProgressService, ProgressEventService, ProgressHistoryService, ProgressNode

logger = logging.getLogger(__name__)


class ProgressManager:
    """
    Facade for decomposed progress services.
    
    This class provides the same interface as the original ProgressManager
    but delegates to focused, single-responsibility services.
    
    The original manager (573 lines) has been decomposed into:
    - ProgressService: Core progress tracking (194 lines)
    - ProgressEventService: Event broadcasting (82 lines)
    - ProgressHistoryService: History recording (130 lines)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        # Initialize services
        self._progress_service = ProgressService()
        self._event_service = ProgressEventService()
        self._history_service = ProgressHistoryService()
        
        # Register history recording with event service
        self._event_service.register_callback(
            lambda node_id, data: self._history_service.record_event("progress_update", node_id, data)
        )
        
        self._initialized = True
        logger.info("ProgressManager initialized with decomposed services")
    
    def create_pipeline(self, pipeline_id: str, name: str, 
                       total_phases: int = 1, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a pipeline progress node."""
        return self._progress_service.create_node(
            pipeline_id, name, "pipeline", 
            total_units=total_phases, metadata=metadata
        )
    
    def create_phase(self, phase_id: str, name: str, pipeline_id: str,
                    total_steps: int = 1, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a phase progress node."""
        return self._progress_service.create_node(
            phase_id, name, "phase", parent=pipeline_id,
            total_units=total_steps, metadata=metadata
        )
    
    def create_step(self, step_id: str, name: str, phase_id: str,
                   total_substeps: int = 100, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a step progress node."""
        return self._progress_service.create_node(
            step_id, name, "step", parent=phase_id,
            total_units=total_substeps, metadata=metadata
        )
    
    def create_substep(self, substep_id: str, name: str, step_id: str,
                      total_units: int = 100, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a substep progress node."""
        return self._progress_service.create_node(
            substep_id, name, "substep", parent=step_id,
            total_units=total_units, metadata=metadata
        )
    
    def start(self, node_id: str) -> None:
        """Start a progress node."""
        self._progress_service.start_node(node_id)
        
        # Emit start event
        progress_data = self._progress_service.get_node_progress(node_id)
        self._event_service.emit_event("start", node_id, progress_data)
    
    def update(self, node_id: str, completed_units: Optional[int] = None,
              delta_units: Optional[int] = None, status: Optional[str] = None,
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update a progress node."""
        self._progress_service.update_node(node_id, completed_units, delta_units, status, metadata)
        
        # Emit update event
        progress_data = self._progress_service.get_node_progress(node_id)
        self._event_service.emit_event("update", node_id, progress_data)
    
    def complete(self, node_id: str, status: str = "completed",
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """Complete a progress node."""
        self._progress_service.complete_node(node_id, status, metadata)
        
        # Emit complete event
        progress_data = self._progress_service.get_node_progress(node_id)
        self._event_service.emit_event("complete", node_id, progress_data)
    
    def get_progress(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """Get progress information for a node or all nodes."""
        if node_id:
            return self._progress_service.get_node_progress(node_id)
        else:
            # Return all nodes
            all_progress = {}
            for node_id in self._progress_service._nodes:
                all_progress[node_id] = self._progress_service.get_node_progress(node_id)
            return all_progress
    
    def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get all active progress nodes."""
        return self._progress_service.get_active_nodes()
    
    def register_callback(self, callback: Callable[[str, Dict[str, Any]], None],
                         node_pattern: str = "*") -> None:
        """Register a progress callback."""
        self._event_service.register_callback(callback, node_pattern)
    
    def set_history_file(self, history_file: Path) -> None:
        """Set the progress history file."""
        self._history_service.set_history_file(history_file)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        progress_summary = self._progress_service.get_summary()
        
        # Add event and history info
        callback_info = self._event_service.get_callback_count()
        history_info = self._history_service.get_history_stats()
        
        return {
            **progress_summary,
            'callbacks': callback_info,
            'history': history_info
        }
    
    def clear(self, pipeline_id: Optional[str] = None) -> None:
        """Clear progress nodes."""
        self._progress_service.clear_nodes(pipeline_id)
        
        # Clear callbacks and history if clearing all
        if pipeline_id is None:
            self._event_service.clear_callbacks()
            self._history_service.clear_history()
    
    # Backward compatibility methods
    def _create_node(self, node_id: str, name: str, level: str, 
                    parent: Optional[str] = None, total_units: int = 100,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a node (backward compatibility)."""
        return self._progress_service.create_node(node_id, name, level, parent, total_units, metadata)
    
    def _get_node_progress(self, node_id: str) -> Dict[str, Any]:
        """Get node progress (backward compatibility)."""
        return self._progress_service.get_node_progress(node_id)
    
    def _propagate_progress(self, node_id: str) -> None:
        """Propagate progress (backward compatibility - now handled automatically)."""
        # Progress propagation is now handled automatically in get_node_progress
        pass
    
    def _should_update(self, node_id: str) -> bool:
        """Check if should update (backward compatibility)."""
        return self._progress_service._should_update(node_id)
    
    def _emit_event(self, event_type: str, node_id: str) -> None:
        """Emit event (backward compatibility)."""
        progress_data = self._progress_service.get_node_progress(node_id)
        self._event_service.emit_event(event_type, node_id, progress_data)
    
    def _record_history(self, event_type: str, node_id: str, 
                       progress_data: Dict[str, Any]) -> None:
        """Record history (backward compatibility)."""
        self._history_service.record_event(event_type, node_id, progress_data)
    
    def _get_descendants(self, node_id: str) -> set:
        """Get descendants (backward compatibility)."""
        return self._progress_service.get_descendants(node_id)


# Backward compatibility callbacks
def console_progress_callback(node_id: str, progress_data: Dict[str, Any]) -> None:
    """Console progress callback."""
    progress = progress_data.get('progress_percent', 0)
    name = progress_data.get('name', 'Unknown')
    status = progress_data.get('status', 'unknown')
    
    print(f"[{status.upper()}] {name} ({node_id}): {progress:.1f}%")


def file_progress_callback(log_file: Path) -> Callable:
    """File progress callback factory."""
    def callback(node_id: str, progress_data: Dict[str, Any]) -> None:
        try:
            with open(log_file, 'a') as f:
                import json
                from datetime import datetime
                
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'node_id': node_id,
                    'progress_data': progress_data
                }
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"File callback error: {e}")
    
    return callback


def database_progress_callback(db_connection) -> Callable:
    """Database progress callback factory."""
    def callback(node_id: str, progress_data: Dict[str, Any]) -> None:
        try:
            # This would store progress in database
            # Implementation depends on database schema
            logger.debug(f"Database callback for {node_id}: {progress_data}")
        except Exception as e:
            logger.error(f"Database callback error: {e}")
    
    return callback


# Singleton instance getter
def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance."""
    return ProgressManager()