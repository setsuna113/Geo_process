"""Abstract base class for progress tracking backends."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime


class ProgressBackend(ABC):
    """Abstract backend for progress storage.
    
    Defines the interface that all progress tracking backends must implement.
    This allows switching between in-memory, database, or other storage backends.
    """
    
    @abstractmethod
    def create_node(self, 
                   experiment_id: str,
                   node_id: str, 
                   parent_id: Optional[str],
                   level: str, 
                   name: str, 
                   total_units: int,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a progress node.
        
        Args:
            experiment_id: Experiment UUID
            node_id: Unique node identifier
            parent_id: Parent node ID (None for root)
            level: Node level (pipeline, phase, step, substep)
            name: Human-readable node name
            total_units: Total units for progress calculation
            metadata: Optional metadata dict
        """
        pass
    
    @abstractmethod
    def update_progress(self, 
                       experiment_id: str,
                       node_id: str, 
                       completed_units: int,
                       status: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update node progress.
        
        Args:
            experiment_id: Experiment UUID
            node_id: Node identifier
            completed_units: Number of completed units
            status: Status (pending, running, completed, failed, cancelled)
            metadata: Optional metadata to merge
        """
        pass
    
    @abstractmethod
    def get_node(self, experiment_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node information.
        
        Args:
            experiment_id: Experiment UUID
            node_id: Node identifier
            
        Returns:
            Node data dict or None if not found
        """
        pass
    
    @abstractmethod
    def get_children(self, experiment_id: str, parent_id: str) -> List[Dict[str, Any]]:
        """Get child nodes.
        
        Args:
            experiment_id: Experiment UUID
            parent_id: Parent node ID
            
        Returns:
            List of child node data
        """
        pass
    
    @abstractmethod
    def get_experiment_nodes(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all nodes for an experiment.
        
        Args:
            experiment_id: Experiment UUID
            
        Returns:
            List of all nodes
        """
        pass
    
    @abstractmethod
    def update_node_status(self, 
                          experiment_id: str,
                          node_id: str,
                          status: str,
                          end_time: Optional[datetime] = None) -> None:
        """Update node status and optionally end time.
        
        Args:
            experiment_id: Experiment UUID
            node_id: Node identifier
            status: New status
            end_time: Optional end timestamp
        """
        pass
    
    @abstractmethod
    def get_active_nodes(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently active (running) nodes.
        
        Args:
            experiment_id: Optional experiment filter
            
        Returns:
            List of active nodes
        """
        pass
    
    def calculate_progress_percent(self, completed_units: int, total_units: int) -> float:
        """Calculate progress percentage.
        
        Args:
            completed_units: Number of completed units
            total_units: Total number of units
            
        Returns:
            Progress percentage (0-100)
        """
        if total_units <= 0:
            return 0.0
        return min(100.0, (completed_units / total_units) * 100.0)