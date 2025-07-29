"""In-memory backend for progress tracking (backward compatibility)."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import time
from collections import defaultdict

from src.base.monitoring.progress_backend import ProgressBackend


class MemoryProgressBackend(ProgressBackend):
    """In-memory progress backend for compatibility with existing ProgressManager."""
    
    def __init__(self):
        """Initialize memory backend."""
        # Store nodes by experiment_id -> node_id -> node_data
        self._nodes: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        # Track parent-child relationships
        self._children: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    def create_node(self, 
                   experiment_id: str,
                   node_id: str, 
                   parent_id: Optional[str],
                   level: str, 
                   name: str, 
                   total_units: int,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a progress node in memory."""
        node = {
            'node_id': node_id,
            'parent_id': parent_id,
            'level': level,
            'name': name,
            'status': 'pending',
            'progress_percent': 0.0,
            'completed_units': 0,
            'total_units': total_units,
            'start_time': None,
            'end_time': None,
            'metadata': metadata or {},
            'experiment_id': experiment_id
        }
        
        self._nodes[experiment_id][node_id] = node
        
        # Track parent-child relationship
        if parent_id:
            self._children[experiment_id][parent_id].append(node_id)
    
    def update_progress(self, 
                       experiment_id: str,
                       node_id: str, 
                       completed_units: int,
                       status: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update node progress in memory."""
        if experiment_id not in self._nodes or node_id not in self._nodes[experiment_id]:
            return
        
        node = self._nodes[experiment_id][node_id]
        node['completed_units'] = completed_units
        node['progress_percent'] = self.calculate_progress_percent(
            completed_units, node['total_units']
        )
        node['status'] = status
        
        if metadata:
            node['metadata'].update(metadata)
        
        # Update timestamps
        if status == 'running' and node['start_time'] is None:
            node['start_time'] = datetime.utcnow()
        elif status in ['completed', 'failed', 'cancelled']:
            node['end_time'] = datetime.utcnow()
    
    def get_node(self, experiment_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node information from memory."""
        if experiment_id in self._nodes and node_id in self._nodes[experiment_id]:
            node = self._nodes[experiment_id][node_id].copy()
            # Calculate elapsed time
            if node['start_time']:
                end = node['end_time'] or datetime.utcnow()
                node['elapsed_seconds'] = (end - node['start_time']).total_seconds()
            else:
                node['elapsed_seconds'] = 0
            return node
        return None
    
    def get_children(self, experiment_id: str, parent_id: str) -> List[Dict[str, Any]]:
        """Get child nodes from memory."""
        children = []
        if experiment_id in self._children and parent_id in self._children[experiment_id]:
            for child_id in self._children[experiment_id][parent_id]:
                child = self.get_node(experiment_id, child_id)
                if child:
                    children.append(child)
        return children
    
    def get_experiment_nodes(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all nodes for an experiment."""
        nodes = []
        if experiment_id in self._nodes:
            for node_id in self._nodes[experiment_id]:
                node = self.get_node(experiment_id, node_id)
                if node:
                    nodes.append(node)
        # Sort by level and node_id
        level_order = {'pipeline': 0, 'phase': 1, 'step': 2, 'substep': 3}
        nodes.sort(key=lambda n: (level_order.get(n['level'], 4), n['node_id']))
        return nodes
    
    def update_node_status(self, 
                          experiment_id: str,
                          node_id: str,
                          status: str,
                          end_time: Optional[datetime] = None) -> None:
        """Update node status and optionally end time."""
        if experiment_id in self._nodes and node_id in self._nodes[experiment_id]:
            node = self._nodes[experiment_id][node_id]
            node['status'] = status
            if end_time:
                node['end_time'] = end_time
            elif status in ['completed', 'failed', 'cancelled']:
                node['end_time'] = datetime.utcnow()
    
    def get_active_nodes(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently active (running) nodes."""
        active = []
        
        if experiment_id:
            # Single experiment
            if experiment_id in self._nodes:
                for node in self._nodes[experiment_id].values():
                    if node['status'] == 'running':
                        active.append(self.get_node(experiment_id, node['node_id']))
        else:
            # All experiments
            for exp_id, nodes in self._nodes.items():
                for node in nodes.values():
                    if node['status'] == 'running':
                        active.append(self.get_node(exp_id, node['node_id']))
        
        # Sort by start time
        active.sort(key=lambda n: n.get('start_time') or datetime.min, reverse=True)
        return active