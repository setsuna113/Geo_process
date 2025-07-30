"""Enhanced progress manager with database backend support."""

import threading
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from src.base.monitoring import ProgressBackend
from src.infrastructure.monitoring import DatabaseProgressBackend, MemoryProgressBackend
from src.infrastructure.logging import get_logger
from .progress_manager import ProgressManager, ProgressNode

logger = get_logger(__name__)


class EnhancedProgressManager(ProgressManager):
    """Enhanced progress manager that supports database persistence.
    
    Extends the existing ProgressManager to optionally use database backend
    while maintaining backward compatibility.
    """
    
    def __new__(cls, *args, **kwargs):
        """Override parent's singleton pattern to allow instance creation."""
        return object.__new__(cls)
    
    def __init__(self, 
                 experiment_id: Optional[str] = None,
                 db_manager = None,
                 use_database: bool = True):
        """Initialize enhanced progress manager.
        
        Args:
            experiment_id: Current experiment ID
            db_manager: Optional DatabaseManager for persistence
            use_database: Whether to use database backend if available
        """
        # Initialize parent
        super().__init__()
        
        self.experiment_id = experiment_id
        
        # Setup backend
        if db_manager and use_database and experiment_id:
            self.backend: ProgressBackend = DatabaseProgressBackend(db_manager)
            self.use_backend = True
            logger.info(f"Using database backend for progress tracking (experiment: {experiment_id})")
        else:
            # Fallback to memory backend for compatibility
            self.backend = MemoryProgressBackend()
            self.use_backend = False
            logger.info("Using memory-only progress tracking (no persistence)")
    
    def set_experiment(self, experiment_id: str):
        """Set or update the experiment ID.
        
        Args:
            experiment_id: Experiment UUID
        """
        self.experiment_id = experiment_id
        logger.debug(f"Set experiment ID: {experiment_id}")
    
    def _create_node(self,
                    node_id: str,
                    name: str,
                    level: str,
                    parent: Optional[str] = None,
                    total_units: int = 100,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a progress node (override parent method).
        
        Adds database persistence to the existing node creation.
        """
        # Call parent implementation
        node_id = super()._create_node(
            node_id=node_id,
            name=name,
            level=level,
            parent=parent,
            total_units=total_units,
            metadata=metadata
        )
        
        # Also persist to backend if available
        if self.use_backend and self.experiment_id:
            try:
                self.backend.create_node(
                    experiment_id=self.experiment_id,
                    node_id=node_id,
                    parent_id=parent,
                    level=level,
                    name=name,
                    total_units=total_units,
                    metadata=metadata
                )
            except Exception as e:
                logger.error(f"Failed to persist node to backend: {e}")
        
        return node_id
    
    def update_progress(self,
                       node_id: str,
                       completed_units: Optional[int] = None,
                       increment: Optional[int] = None,
                       status: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update progress for a node (override parent method).
        
        Adds database persistence to progress updates.
        """
        # Call parent implementation
        super().update_progress(
            node_id=node_id,
            completed_units=completed_units,
            increment=increment,
            status=status,
            metadata=metadata
        )
        
        # Get updated node info
        node = self._nodes.get(node_id)
        if not node:
            return
        
        # Persist to backend if available
        if self.use_backend and self.experiment_id:
            try:
                self.backend.update_progress(
                    experiment_id=self.experiment_id,
                    node_id=node_id,
                    completed_units=node.completed_units,
                    status=node.status,
                    metadata=metadata
                )
            except Exception as e:
                logger.error(f"Failed to persist progress to backend: {e}")
    
    def get_experiment_progress(self) -> List[Dict[str, Any]]:
        """Get all progress nodes for current experiment.
        
        Returns:
            List of progress nodes from backend
        """
        if self.use_backend and self.experiment_id:
            try:
                return self.backend.get_experiment_nodes(self.experiment_id)
            except Exception as e:
                logger.error(f"Failed to get progress from backend: {e}")
        
        # Fallback to in-memory nodes
        return self.get_all_nodes()
    
    def sync_to_backend(self):
        """Sync all in-memory nodes to backend.
        
        Useful for migrating existing progress to database.
        """
        if not self.use_backend or not self.experiment_id:
            return
        
        with self._node_lock:
            for node_id, node in self._nodes.items():
                try:
                    # Create node in backend
                    self.backend.create_node(
                        experiment_id=self.experiment_id,
                        node_id=node_id,
                        parent_id=node.parent,
                        level=node.level,
                        name=node.name,
                        total_units=node.total_units,
                        metadata=node.metadata
                    )
                    
                    # Update progress
                    self.backend.update_progress(
                        experiment_id=self.experiment_id,
                        node_id=node_id,
                        completed_units=node.completed_units,
                        status=node.status
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to sync node {node_id} to backend: {e}")
        
        logger.info(f"Synced {len(self._nodes)} nodes to backend")


# Global enhanced instance
_enhanced_instance: Optional[EnhancedProgressManager] = None
_enhanced_lock = threading.Lock()


def get_enhanced_progress_manager(experiment_id: Optional[str] = None,
                                 db_manager = None) -> EnhancedProgressManager:
    """Get or create enhanced progress manager singleton.
    
    Args:
        experiment_id: Current experiment ID
        db_manager: Optional DatabaseManager
        
    Returns:
        EnhancedProgressManager instance
    """
    global _enhanced_instance
    
    with _enhanced_lock:
        if _enhanced_instance is None:
            _enhanced_instance = EnhancedProgressManager(
                experiment_id=experiment_id,
                db_manager=db_manager
            )
        elif experiment_id and _enhanced_instance.experiment_id != experiment_id:
            # Update experiment ID if changed
            _enhanced_instance.set_experiment(experiment_id)
            
    return _enhanced_instance