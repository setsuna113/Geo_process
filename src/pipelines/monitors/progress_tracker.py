# src/pipelines/monitors/progress_tracker.py
"""Progress tracking for pipeline execution."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StageProgress:
    """Progress information for a single stage."""
    name: str
    status: str = "pending"
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    operations_completed: int = 0
    total_operations: int = 100
    error: Optional[str] = None


class ProgressTracker:
    """Track progress of pipeline execution."""
    
    def __init__(self):
        self.stages: Dict[str, StageProgress] = {}
        self.total_stages = 0
        self.completed_stages = 0
        self.start_time: Optional[datetime] = None
        self.callbacks = []
    
    def initialize(self, total_stages: int, total_operations: Optional[int] = None):
        """Initialize progress tracking."""
        self.total_stages = total_stages
        self.start_time = datetime.now()
        self.completed_stages = 0
        
        logger.info(f"Progress tracking initialized: {total_stages} stages")
    
    def start_stage(self, stage_name: str, total_operations: int = 100):
        """Mark stage as started."""
        self.stages[stage_name] = StageProgress(
            name=stage_name,
            status="running",
            start_time=datetime.now(),
            total_operations=total_operations
        )
        
        self._notify_callbacks()
    
    def update_stage(self, stage_name: str, operations_completed: Optional[int] = None, 
                    progress_percent: Optional[float] = None, message: Optional[str] = None):
        """Update stage progress."""
        if stage_name not in self.stages:
            return
        
        stage = self.stages[stage_name]
        
        if operations_completed is not None:
            stage.operations_completed = operations_completed
            stage.progress = (operations_completed / stage.total_operations) * 100
        elif progress_percent is not None:
            stage.progress = progress_percent
            stage.operations_completed = int((progress_percent / 100) * stage.total_operations)
        
        # Estimate completion time
        if stage.progress > 0 and stage.start_time:
            elapsed = datetime.now() - stage.start_time
            total_estimated = elapsed * (100 / stage.progress)
            stage.estimated_completion = stage.start_time + total_estimated
        
        self._notify_callbacks()
    
    def complete_stage(self, stage_name: str):
        """Mark stage as completed."""
        if stage_name not in self.stages:
            return
        
        stage = self.stages[stage_name]
        stage.status = "completed"
        stage.progress = 100.0
        stage.end_time = datetime.now()
        stage.operations_completed = stage.total_operations
        
        self.completed_stages += 1
        self._notify_callbacks()
        
        if stage.start_time:
            duration = stage.end_time - stage.start_time
            logger.info(f"Stage '{stage_name}' completed in {duration}")
    
    def fail_stage(self, stage_name: str, error: str):
        """Mark stage as failed."""
        if stage_name not in self.stages:
            return
        
        stage = self.stages[stage_name]
        stage.status = "failed"
        stage.error = error
        stage.end_time = datetime.now()
        
        self._notify_callbacks()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        overall_progress = (self.completed_stages / self.total_stages * 100) if self.total_stages > 0 else 0
        
        # Calculate ETA
        eta = None
        if self.start_time and overall_progress > 0:
            elapsed = datetime.now() - self.start_time
            total_estimated = elapsed * (100 / overall_progress)
            eta = self.start_time + total_estimated
        
        return {
            'overall_progress': overall_progress,
            'completed_stages': self.completed_stages,
            'total_stages': self.total_stages,
            'current_stage': self._get_current_stage(),
            'eta': eta,
            'elapsed_time': datetime.now() - self.start_time if self.start_time else None,
            'stages': {
                name: {
                    'status': stage.status,
                    'progress': stage.progress,
                    'elapsed': (datetime.now() - stage.start_time).total_seconds() if stage.start_time and stage.status == 'running' else None
                }
                for name, stage in self.stages.items()
            }
        }
    
    def _get_current_stage(self) -> Optional[str]:
        """Get currently running stage."""
        for name, stage in self.stages.items():
            if stage.status == "running":
                return name
        return None
    
    def register_callback(self, callback):
        """Register progress callback."""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks."""
        progress = self.get_progress()
        for callback in self.callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def get_summary(self) -> str:
        """Get progress summary as string."""
        progress = self.get_progress()
        
        lines = [
            f"Pipeline Progress: {progress['overall_progress']:.1f}%",
            f"Stages: {progress['completed_stages']}/{progress['total_stages']}",
        ]
        
        if progress['current_stage']:
            current = self.stages[progress['current_stage']]
            lines.append(f"Current: {current.name} ({current.progress:.1f}%)")
        
        if progress['eta']:
            lines.append(f"ETA: {progress['eta'].strftime('%H:%M:%S')}")
        
        return " | ".join(lines)
    
    def start(self):
        """Start progress tracking."""
        pass  # Placeholder for consistency
    
    def stop(self):
        """Stop progress tracking."""
        pass  # Placeholder for consistency
    
    def update(self):
        """Update progress (called periodically)."""
        # Could implement auto-progress estimation here
        pass
    
    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Restore progress from checkpoint."""
        if 'stages' in checkpoint_data:
            for stage_name, stage_data in checkpoint_data['stages'].items():
                self.stages[stage_name] = StageProgress(
                    name=stage_name,
                    status=stage_data.get('status', 'pending'),
                    progress=stage_data.get('progress', 0.0)
                )
        
        self.completed_stages = checkpoint_data.get('completed_stages', 0)