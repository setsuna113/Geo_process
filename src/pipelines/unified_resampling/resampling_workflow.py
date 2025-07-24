from typing import Dict, Any, Optional, Callable
# src/pipelines/unified_resampling/resampling_workflow.py
"""Workflow management for resampling operations."""

import logging
from datetime import datetime

from src.config.config import Config

logger = logging.getLogger(__name__)


class ResamplingWorkflow:
    """Manages workflow and progress tracking for resampling operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.workflow_state = {}
        self.progress_callbacks = []
        self.start_time = None
        self.steps_completed = 0
        self.total_steps = 0
        
    def initialize_workflow(self, datasets_count: int, include_som: bool = True):
        """Initialize workflow with total step count."""
        self.total_steps = datasets_count  # One step per dataset
        if include_som:
            self.total_steps += 2  # Merging + SOM analysis
        
        self.steps_completed = 0
        self.start_time = datetime.now()
        
        self.workflow_state = {
            'status': 'initialized',
            'total_datasets': datasets_count,
            'completed_datasets': 0,
            'current_step': 'initialization',
            'start_time': self.start_time,
            'estimated_completion': None,
            'errors': []
        }
        
        logger.info(f"Workflow initialized: {self.total_steps} total steps")
    
    def register_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback function for progress updates."""
        self.progress_callbacks.append(callback)
    
    def update_progress(self, step_name: str, step_progress: float = 100.0, 
                       status_message: Optional[str] = None):
        """Update workflow progress."""
        if step_progress >= 100.0:
            self.steps_completed += 1
        
        overall_progress = (self.steps_completed / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        # Update workflow state
        self.workflow_state.update({
            'current_step': step_name,
            'step_progress': step_progress,
            'overall_progress': overall_progress,
            'status_message': status_message,
            'last_update': datetime.now()
        })
        
        # Estimate completion time
        if self.start_time and overall_progress > 0:
            elapsed = datetime.now() - self.start_time
            estimated_total = elapsed * (100 / overall_progress)
            self.workflow_state['estimated_completion'] = self.start_time + estimated_total
        
        # Call progress callbacks
        for callback in self.progress_callbacks:
            try:
                callback(self.workflow_state.copy())
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Log progress
        if step_progress >= 100.0:
            logger.info(f"✅ Completed: {step_name} (Overall: {overall_progress:.1f}%)")
        elif status_message:
            logger.info(f"🔄 {step_name}: {status_message} ({step_progress:.1f}%)")
    
    def log_error(self, step_name: str, error_message: str, exception: Optional[Exception] = None):
        """Log an error in the workflow."""
        error_info = {
            'step': step_name,
            'message': error_message,
            'timestamp': datetime.now(),
            'exception_type': type(exception).__name__ if exception else None
        }
        
        self.workflow_state['errors'].append(error_info)
        logger.error(f"❌ Error in {step_name}: {error_message}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return self.workflow_state.copy()
    
    def create_progress_report(self) -> str:
        """Create a human-readable progress report."""
        state = self.workflow_state
        
        report_lines = [
            "Resampling Workflow Progress Report",
            "=" * 40,
            f"Status: {state.get('status', 'unknown')}",
            f"Overall Progress: {state.get('overall_progress', 0):.1f}%",
            f"Current Step: {state.get('current_step', 'unknown')}",
            f"Completed Steps: {self.steps_completed}/{self.total_steps}",
            ""
        ]
        
        # Time information
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            report_lines.extend([
                f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Elapsed Time: {self._format_duration(elapsed)}",
            ])
            
            if state.get('estimated_completion'):
                eta = state['estimated_completion']
                remaining = eta - datetime.now()
                report_lines.extend([
                    f"Estimated Completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Estimated Remaining: {self._format_duration(remaining)}"
                ])
        
        # Dataset progress
        if state.get('total_datasets'):
            report_lines.extend([
                "",
                "Dataset Progress:",
                f"  Completed: {state.get('completed_datasets', 0)}/{state['total_datasets']}"
            ])
        
        # Current status
        if state.get('status_message'):
            report_lines.extend([
                "",
                f"Current Status: {state['status_message']}"
            ])
        
        # Errors
        if state.get('errors'):
            report_lines.extend([
                "",
                "Errors Encountered:",
                "-" * 20
            ])
            for error in state['errors'][-5:]:  # Show last 5 errors
                report_lines.append(f"  {error['step']}: {error['message']}")
        
        return "\n".join(report_lines)
    
    def _format_duration(self, duration) -> str:
        """Format duration as human-readable string."""
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def create_dataset_progress_tracker(self, dataset_name: str, total_operations: int = 100):
        """Create a progress tracker for a specific dataset."""
        return DatasetProgressTracker(self, dataset_name, total_operations)
    
    def finalize_workflow(self, success: bool = True, final_message: Optional[str] = None):
        """Finalize the workflow."""
        end_time = datetime.now()
        total_duration = end_time - self.start_time if self.start_time else None
        
        self.workflow_state.update({
            'status': 'completed' if success else 'failed',
            'end_time': end_time,
            'total_duration': total_duration,
            'final_message': final_message,
            'overall_progress': 100.0 if success else self.workflow_state.get('overall_progress', 0)
        })
        
        # Final progress update
        for callback in self.progress_callbacks:
            try:
                callback(self.workflow_state.copy())
            except Exception as e:
                logger.warning(f"Final progress callback failed: {e}")
        
        duration_str = self._format_duration(total_duration) if total_duration else "unknown"
        
        if success:
            logger.info(f"✅ Workflow completed successfully in {duration_str}")
        else:
            logger.error(f"❌ Workflow failed after {duration_str}")


class DatasetProgressTracker:
    """Progress tracker for individual dataset processing."""
    
    def __init__(self, workflow: ResamplingWorkflow, dataset_name: str, total_operations: int):
        self.workflow = workflow
        self.dataset_name = dataset_name
        self.total_operations = total_operations
        self.completed_operations = 0
    
    def update(self, operations_completed: Optional[int] = None, message: Optional[str] = None):
        """Update progress for this dataset."""
        if operations_completed is not None:
            self.completed_operations = operations_completed
        else:
            self.completed_operations += 1
        
        progress = (self.completed_operations / self.total_operations) * 100
        
        status_message = message or f"Processing {self.dataset_name}"
        
        self.workflow.update_progress(
            step_name=f"Dataset: {self.dataset_name}",
            step_progress=progress,
            status_message=status_message
        )
    
    def complete(self, message: Optional[str] = None):
        """Mark dataset processing as complete."""
        final_message = message or f"Completed {self.dataset_name}"
        self.workflow.update_progress(
            step_name=f"Dataset: {self.dataset_name}",
            step_progress=100.0,
            status_message=final_message
        )
        
        # Update workflow dataset counter
        self.workflow.workflow_state['completed_datasets'] = \
            self.workflow.workflow_state.get('completed_datasets', 0) + 1
    
    def error(self, error_message: str, exception: Optional[Exception] = None):
        """Log an error for this dataset."""
        self.workflow.log_error(
            step_name=f"Dataset: {self.dataset_name}",
            error_message=error_message,
            exception=exception
        )