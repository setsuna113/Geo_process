"""
Simple progress manager to replace EnhancedProgressManager
"""

import time
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SimpleProgressManager:
    """Simplified progress tracking without database dependencies."""
    
    def __init__(self, experiment_id: Optional[str] = None, 
                 db_manager: Optional[Any] = None,
                 use_database: bool = False):
        """Initialize simple progress manager.
        
        Args:
            experiment_id: Experiment identifier (optional)
            db_manager: Database manager (ignored)
            use_database: Whether to use database (ignored, always False)
        """
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.stages = {}
        self.current_stage = None
        self.start_time = time.time()
        
    def start_stage(self, stage_name: str, total_items: int = 100):
        """Start tracking a new stage."""
        self.current_stage = stage_name
        self.stages[stage_name] = {
            'total': total_items,
            'completed': 0,
            'start_time': time.time(),
            'status': 'running'
        }
        logger.info(f"Started stage: {stage_name} ({total_items} items)")
        
    def update_progress(self, stage_name: Optional[str] = None, 
                       completed: Optional[int] = None,
                       increment: int = 1):
        """Update progress for a stage."""
        stage = stage_name or self.current_stage
        if stage and stage in self.stages:
            if completed is not None:
                self.stages[stage]['completed'] = completed
            else:
                self.stages[stage]['completed'] += increment
                
            progress = self.stages[stage]['completed'] / self.stages[stage]['total'] * 100
            logger.info(f"{stage}: {progress:.1f}% ({self.stages[stage]['completed']}/{self.stages[stage]['total']})")
    
    def complete_stage(self, stage_name: Optional[str] = None):
        """Mark a stage as complete."""
        stage = stage_name or self.current_stage
        if stage and stage in self.stages:
            self.stages[stage]['status'] = 'completed'
            elapsed = time.time() - self.stages[stage]['start_time']
            logger.info(f"Completed stage: {stage} in {elapsed:.1f}s")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        total_elapsed = time.time() - self.start_time
        return {
            'experiment_id': self.experiment_id,
            'total_time': total_elapsed,
            'stages': self.stages
        }
        
    def cleanup(self):
        """Cleanup resources (no-op for simple version)."""
        pass


# Alias for compatibility
EnhancedProgressManager = SimpleProgressManager