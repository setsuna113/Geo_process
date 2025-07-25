# src/core/progress_manager.py
"""Hierarchical progress tracking system for the biodiversity pipeline."""

import threading
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProgressNode:
    """Node in the progress hierarchy."""
    name: str
    level: str  # 'pipeline', 'phase', 'step', 'substep'
    parent: Optional[str] = None
    total_units: int = 100
    completed_units: int = 0
    status: str = "pending"  # pending, running, completed, failed, cancelled
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_units == 0:
            return 100.0 if self.status == "completed" else 0.0
        return (self.completed_units / self.total_units) * 100.0
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def is_active(self) -> bool:
        """Check if node is currently active."""
        return self.status == "running"
    
    @property
    def is_complete(self) -> bool:
        """Check if node is complete."""
        return self.status in ["completed", "failed", "cancelled"]


class ProgressManager:
    """
    Hierarchical progress tracking with aggregation and reporting.
    
    Supports pipeline → phase → step → substep hierarchy.
    """
    
    _instance: Optional['ProgressManager'] = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        """Initialize progress manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            
            # Progress hierarchy
            self._nodes: Dict[str, ProgressNode] = {}
            self._node_lock = threading.RLock()
            
            # Callbacks
            self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
            
            # Progress history
            self._history: List[Dict[str, Any]] = []
            self._history_file: Optional[Path] = None
            
            # Update tracking
            self._last_update_time: Dict[str, float] = {}
            self._update_interval = 1.0  # Minimum seconds between updates
            
            logger.info("Progress manager initialized")
    
    def create_pipeline(self, 
                       pipeline_id: str,
                       total_phases: int,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a pipeline-level progress node.
        
        Args:
            pipeline_id: Unique pipeline identifier
            total_phases: Total number of phases
            metadata: Optional metadata
            
        Returns:
            Pipeline node ID
        """
        return self._create_node(
            node_id=pipeline_id,
            name=pipeline_id,
            level="pipeline",
            total_units=total_phases,
            metadata=metadata
        )
    
    def create_phase(self,
                    phase_id: str,
                    pipeline_id: str,
                    total_steps: int,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a phase-level progress node."""
        return self._create_node(
            node_id=phase_id,
            name=phase_id,
            level="phase",
            parent=pipeline_id,
            total_units=total_steps,
            metadata=metadata
        )
    
    def create_step(self,
                   step_id: str,
                   phase_id: str,
                   total_substeps: int = 100,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a step-level progress node."""
        return self._create_node(
            node_id=step_id,
            name=step_id,
            level="step",
            parent=phase_id,
            total_units=total_substeps,
            metadata=metadata
        )
    
    def create_substep(self,
                      substep_id: str,
                      step_id: str,
                      total_units: int = 100,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a substep-level progress node."""
        return self._create_node(
            node_id=substep_id,
            name=substep_id,
            level="substep",
            parent=step_id,
            total_units=total_units,
            metadata=metadata
        )
    
    def start(self, node_id: str) -> None:
        """Mark a node as started."""
        with self._node_lock:
            if node_id not in self._nodes:
                raise ValueError(f"Unknown node: {node_id}")
            
            node = self._nodes[node_id]
            node.status = "running"
            node.start_time = time.time()
            
            self._emit_event("started", node_id)
            self._record_history("start", node_id)
    
    def update(self, 
              node_id: str,
              completed_units: Optional[int] = None,
              increment: Optional[int] = None,
              metadata: Optional[Dict[str, Any]] = None,
              force: bool = False) -> None:
        """
        Update progress for a node.
        
        Args:
            node_id: Node to update
            completed_units: Absolute completed units
            increment: Increment completed units by this amount
            metadata: Additional metadata to merge
            force: Force update even if within rate limit
        """
        with self._node_lock:
            if node_id not in self._nodes:
                raise ValueError(f"Unknown node: {node_id}")
            
            # Rate limiting
            if not force and not self._should_update(node_id):
                return
            
            node = self._nodes[node_id]
            
            # Update progress
            if completed_units is not None:
                node.completed_units = min(completed_units, node.total_units)
            elif increment is not None:
                node.completed_units = min(node.completed_units + increment, node.total_units)
            
            # Update metadata
            if metadata:
                node.metadata.update(metadata)
            
            # Propagate to parents
            self._propagate_progress(node_id)
            
            self._emit_event("progress", node_id)
            self._last_update_time[node_id] = time.time()
    
    def complete(self, 
                node_id: str,
                status: str = "completed",
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark a node as complete.
        
        Args:
            node_id: Node to complete
            status: Final status (completed, failed, cancelled)
            metadata: Final metadata
        """
        with self._node_lock:
            if node_id not in self._nodes:
                raise ValueError(f"Unknown node: {node_id}")
            
            node = self._nodes[node_id]
            node.status = status
            node.end_time = time.time()
            
            if status == "completed":
                node.completed_units = node.total_units
            
            if metadata:
                node.metadata.update(metadata)
            
            # Complete all children if successful
            if status == "completed":
                for child_id in node.children:
                    if child_id in self._nodes and not self._nodes[child_id].is_complete:
                        self.complete(child_id, "completed")
            
            self._propagate_progress(node_id)
            self._emit_event("completed", node_id)
            self._record_history("complete", node_id, {"status": status})
    
    def get_progress(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get progress information.
        
        Args:
            node_id: Specific node or None for all
            
        Returns:
            Progress information
        """
        with self._node_lock:
            if node_id:
                return self._get_node_progress(node_id)
            else:
                # Return all root nodes (pipelines)
                pipelines = [
                    self._get_node_progress(nid)
                    for nid, node in self._nodes.items()
                    if node.parent is None
                ]
                return {"pipelines": pipelines}
    
    def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get all currently active nodes."""
        with self._node_lock:
            active = []
            for node_id, node in self._nodes.items():
                if node.is_active:
                    active.append(self._get_node_progress(node_id))
            return active
    
    def register_callback(self, 
                         event_type: str,
                         callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Register a progress callback.
        
        Args:
            event_type: Event type (started, progress, completed, any)
            callback: Callback function(node_id, progress_data)
        """
        self._callbacks[event_type].append(callback)
    
    def set_history_file(self, history_file: Path) -> None:
        """Set file for saving progress history."""
        self._history_file = history_file
        
        # Load existing history
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall progress summary."""
        with self._node_lock:
            total_nodes = len(self._nodes)
            completed_nodes = sum(1 for n in self._nodes.values() if n.is_complete)
            active_nodes = sum(1 for n in self._nodes.values() if n.is_active)
            
            # Calculate weighted progress
            total_progress = 0.0
            pipeline_count = 0
            
            for node in self._nodes.values():
                if node.level == "pipeline":
                    total_progress += node.progress_percent
                    pipeline_count += 1
            
            avg_progress = total_progress / pipeline_count if pipeline_count > 0 else 0.0
            
            return {
                'total_nodes': total_nodes,
                'completed_nodes': completed_nodes,
                'active_nodes': active_nodes,
                'average_progress': avg_progress,
                'pipelines': pipeline_count
            }
    
    def clear(self, pipeline_id: Optional[str] = None) -> None:
        """
        Clear progress data.
        
        Args:
            pipeline_id: Clear specific pipeline or all if None
        """
        with self._node_lock:
            if pipeline_id:
                # Clear specific pipeline and its children
                nodes_to_remove = self._get_descendants(pipeline_id)
                nodes_to_remove.add(pipeline_id)
                
                for node_id in nodes_to_remove:
                    self._nodes.pop(node_id, None)
                    self._last_update_time.pop(node_id, None)
            else:
                # Clear all
                self._nodes.clear()
                self._last_update_time.clear()
                self._history.clear()
    
    def _create_node(self,
                    node_id: str,
                    name: str,
                    level: str,
                    parent: Optional[str] = None,
                    total_units: int = 100,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a progress node."""
        with self._node_lock:
            if node_id in self._nodes:
                raise ValueError(f"Node already exists: {node_id}")
            
            node = ProgressNode(
                name=name,
                level=level,
                parent=parent,
                total_units=total_units,
                metadata=metadata or {}
            )
            
            self._nodes[node_id] = node
            
            # Update parent's children
            if parent and parent in self._nodes:
                self._nodes[parent].children.append(node_id)
            
            self._record_history("create", node_id, {"level": level, "parent": parent})
            
            # Emit creation event both for specific level and generic created
            self._emit_event(level, node_id)  # 'pipeline', 'phase', 'step'
            self._emit_event("created", node_id)  # Generic creation event
            
            return node_id
    
    def _get_node_progress(self, node_id: str) -> Dict[str, Any]:
        """Get progress data for a node."""
        if node_id not in self._nodes:
            raise ValueError(f"Unknown node: {node_id}")
        
        node = self._nodes[node_id]
        
        # Basic progress data
        progress_data = {
            'id': node_id,
            'name': node.name,
            'level': node.level,
            'status': node.status,
            'progress_percent': node.progress_percent,
            'completed_units': node.completed_units,
            'total_units': node.total_units,
            'elapsed_time': node.elapsed_time,
            'metadata': node.metadata
        }
        
        # Add children progress
        if node.children:
            children_progress = []
            for child_id in node.children:
                if child_id in self._nodes:
                    children_progress.append(self._get_node_progress(child_id))
            progress_data['children'] = children_progress
        
        # Add estimated remaining time
        if node.is_active and node.completed_units > 0:
            rate = node.completed_units / node.elapsed_time
            remaining = node.total_units - node.completed_units
            progress_data['estimated_remaining_seconds'] = remaining / rate if rate > 0 else None
        
        return progress_data
    
    def _propagate_progress(self, node_id: str) -> None:
        """Propagate progress updates to parent nodes."""
        node = self._nodes.get(node_id)
        if not node or not node.parent:
            return
        
        parent = self._nodes.get(node.parent)
        if not parent:
            return
        
        # Calculate parent progress based on children
        if parent.children:
            total_progress = 0
            completed_children = 0
            
            for child_id in parent.children:
                if child_id in self._nodes:
                    child = self._nodes[child_id]
                    if child.is_complete:
                        completed_children += 1
                    total_progress += child.progress_percent
            
            # Update parent based on children average
            if parent.level in ["pipeline", "phase"]:
                parent.completed_units = completed_children
            else:
                avg_progress = total_progress / len(parent.children)
                parent.completed_units = int(parent.total_units * avg_progress / 100)
        
        # Recursively propagate
        if parent.parent:
            self._propagate_progress(parent.parent)
    
    def _should_update(self, node_id: str) -> bool:
        """Check if update should be emitted based on rate limiting."""
        last_update = self._last_update_time.get(node_id, 0)
        return (time.time() - last_update) >= self._update_interval
    
    def _emit_event(self, event_type: str, node_id: str) -> None:
        """Emit progress event to callbacks."""
        try:
            progress_data = self._get_node_progress(node_id)
            
            # Call specific event callbacks
            for callback in self._callbacks.get(event_type, []):
                try:
                    callback(node_id, progress_data)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
            
            # Call 'any' callbacks
            for callback in self._callbacks.get("any", []):
                try:
                    callback(node_id, progress_data)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")
    
    def _record_history(self, 
                       action: str,
                       node_id: str,
                       data: Optional[Dict[str, Any]] = None) -> None:
        """Record action in history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'node_id': node_id,
            'data': data or {}
        }
        
        self._history.append(entry)
        
        # Save to file if configured
        if self._history_file:
            try:
                with open(self._history_file, 'w') as f:
                    json.dump(self._history, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save history: {e}")
    
    def _get_descendants(self, node_id: str) -> set:
        """Get all descendant node IDs."""
        descendants = set()
        
        node = self._nodes.get(node_id)
        if not node:
            return descendants
        
        for child_id in node.children:
            descendants.add(child_id)
            descendants.update(self._get_descendants(child_id))
        
        return descendants


# Convenience functions for common progress patterns
def console_progress_callback(node_id: str, progress_data: Dict[str, Any]) -> None:
    """Console progress callback."""
    level = progress_data['level']
    name = progress_data['name']
    percent = progress_data['progress_percent']
    status = progress_data['status']
    
    if level in ['pipeline', 'phase']:
        print(f"\r{name}: {percent:.1f}% [{status}]", end='', flush=True)
        if status in ['completed', 'failed', 'cancelled']:
            print()  # New line when complete


def file_progress_callback(log_file: Path) -> Callable:
    """Create a file progress callback."""
    def callback(node_id: str, progress_data: Dict[str, Any]) -> None:
        with open(log_file, 'a') as f:
            entry = {
                'timestamp': datetime.now().isoformat(),
                'node_id': node_id,
                'progress': progress_data
            }
            f.write(json.dumps(entry) + '\n')
    return callback


def database_progress_callback(db_connection) -> Callable:
    """Create a database progress callback."""
    def callback(node_id: str, progress_data: Dict[str, Any]) -> None:
        # Implementation depends on database schema
        # This is a placeholder
        try:
            # Store progress in database
            pass
        except Exception as e:
            logger.error(f"Failed to store progress in database: {e}")
    return callback


# Global progress manager instance
_progress_manager: Optional[ProgressManager] = None


def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance."""
    global _progress_manager
    if _progress_manager is None:
        _progress_manager = ProgressManager()
    return _progress_manager