"""Core progress tracking service."""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

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


class ProgressService:
    """Core progress tracking service."""
    
    def __init__(self):
        self._nodes: Dict[str, ProgressNode] = {}
        self._last_update_times: Dict[str, float] = {}
        self._update_threshold = 0.1  # Minimum seconds between updates
    
    def create_node(self, node_id: str, name: str, level: str, 
                   parent: Optional[str] = None, total_units: int = 100,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new progress node."""
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already exists")
        
        node = ProgressNode(
            name=name,
            level=level,
            parent=parent,
            total_units=total_units,
            metadata=metadata or {}
        )
        
        self._nodes[node_id] = node
        
        # Add to parent's children
        if parent and parent in self._nodes:
            if node_id not in self._nodes[parent].children:
                self._nodes[parent].children.append(node_id)
        
        logger.debug(f"Created {level} node: {node_id} ({name})")
        return node_id
    
    def start_node(self, node_id: str) -> None:
        """Start a progress node."""
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self._nodes[node_id]
        node.status = "running"
        node.start_time = time.time()
        
        logger.debug(f"Started node: {node_id}")
    
    def update_node(self, node_id: str, completed_units: Optional[int] = None,
                   delta_units: Optional[int] = None, status: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update a progress node."""
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not found")
        
        # Check update throttling
        if not self._should_update(node_id):
            return
        
        node = self._nodes[node_id]
        
        # Update completed units
        if completed_units is not None:
            node.completed_units = min(completed_units, node.total_units)
        elif delta_units is not None:
            node.completed_units = min(node.completed_units + delta_units, node.total_units)
        
        # Update status
        if status:
            node.status = status
        
        # Update metadata
        if metadata:
            node.metadata.update(metadata)
        
        # Auto-complete if fully processed
        if node.completed_units >= node.total_units and node.status == "running":
            node.status = "completed"
            node.end_time = time.time()
        
        self._last_update_times[node_id] = time.time()
        logger.debug(f"Updated node {node_id}: {node.progress_percent:.1f}%")
    
    def complete_node(self, node_id: str, status: str = "completed",
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Complete a progress node."""
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self._nodes[node_id]
        node.status = status
        node.end_time = time.time()
        
        if status == "completed":
            node.completed_units = node.total_units
        
        if metadata:
            node.metadata.update(metadata)
        
        logger.debug(f"Completed node {node_id} with status: {status}")
    
    def get_node(self, node_id: str) -> Optional[ProgressNode]:
        """Get a progress node."""
        return self._nodes.get(node_id)
    
    def get_node_progress(self, node_id: str) -> Dict[str, Any]:
        """Get detailed progress information for a node."""
        if node_id not in self._nodes:
            return {}
        
        node = self._nodes[node_id]
        
        # Calculate child progress if has children
        child_progress = 0.0
        if node.children:
            child_progresses = []
            for child_id in node.children:
                if child_id in self._nodes:
                    child_node = self._nodes[child_id]
                    child_progresses.append(child_node.progress_percent)
            
            if child_progresses:
                child_progress = sum(child_progresses) / len(child_progresses)
        
        # Use child progress if available and higher than direct progress
        effective_progress = max(node.progress_percent, child_progress)
        
        return {
            'node_id': node_id,
            'name': node.name,
            'level': node.level,
            'parent': node.parent,
            'status': node.status,
            'progress_percent': effective_progress,
            'completed_units': node.completed_units,
            'total_units': node.total_units,
            'elapsed_time': node.elapsed_time,
            'start_time': node.start_time,
            'end_time': node.end_time,
            'metadata': node.metadata,
            'children': node.children,
            'child_progress': child_progress,
            'is_active': node.is_active,
            'is_complete': node.is_complete
        }
    
    def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get all active progress nodes."""
        active_nodes = []
        for node_id, node in self._nodes.items():
            if node.is_active:
                active_nodes.append(self.get_node_progress(node_id))
        return active_nodes
    
    def get_descendants(self, node_id: str) -> set:
        """Get all descendant node IDs."""
        descendants = set()
        
        if node_id not in self._nodes:
            return descendants
        
        node = self._nodes[node_id]
        for child_id in node.children:
            descendants.add(child_id)
            descendants.update(self.get_descendants(child_id))
        
        return descendants
    
    def clear_nodes(self, pipeline_id: Optional[str] = None) -> None:
        """Clear progress nodes."""
        if pipeline_id:
            # Clear specific pipeline and descendants
            to_remove = {pipeline_id}
            to_remove.update(self.get_descendants(pipeline_id))
            
            for node_id in to_remove:
                self._nodes.pop(node_id, None)
                self._last_update_times.pop(node_id, None)
        else:
            # Clear all nodes
            self._nodes.clear()
            self._last_update_times.clear()
        
        logger.info(f"Cleared progress nodes: {pipeline_id or 'all'}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        total_nodes = len(self._nodes)
        active_nodes = len([n for n in self._nodes.values() if n.is_active])
        completed_nodes = len([n for n in self._nodes.values() if n.is_complete])
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_nodes,
            'completed_nodes': completed_nodes,
            'pending_nodes': total_nodes - active_nodes - completed_nodes
        }
    
    def _should_update(self, node_id: str) -> bool:
        """Check if node should be updated based on throttling."""
        last_update = self._last_update_times.get(node_id, 0)
        return (time.time() - last_update) >= self._update_threshold