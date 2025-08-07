"""GEE Task Monitor with Local Integration

Monitors GEE tasks and integrates with local progress tracking.
"""

import ee
import time
from typing import Dict, Optional, List
from datetime import datetime

from src.infrastructure.logging import get_logger
from src.core.progress_events import get_event_bus, create_file_progress

logger = get_logger(__name__)


class GEETaskMonitor:
    """Monitor GEE tasks with local process integration"""
    
    def __init__(self):
        """Initialize task monitor"""
        self.logger = logger
        self.event_bus = get_event_bus()
        self.active_monitors = {}
    
    def create_task_monitor(self, task_id: str, description: str,
                          total_steps: int = 100) -> Dict:
        """
        Create a progress monitor for a GEE task.
        
        Args:
            task_id: GEE task ID
            description: Task description
            total_steps: Total progress steps
            
        Returns:
            Monitor info dict
        """
        # Create progress tracker - create_file_progress requires file_path, processed, total
        # For now, use a dummy file path for the task
        progress = create_file_progress(
            file_path=f"gee_task_{task_id}",
            processed=0,
            total=total_steps,
            operation="export",
            source=f"GEE: {description}"
        )
        
        monitor = {
            'task_id': task_id,
            'description': description,
            'progress': progress,
            'start_time': time.time(),
            'last_update': time.time(),
            'last_progress': 0
        }
        
        self.active_monitors[task_id] = monitor
        return monitor
    
    def update_task_progress(self, task_id: str, task_status: Dict) -> bool:
        """
        Update progress based on GEE task status.
        
        Args:
            task_id: GEE task ID
            task_status: Status dict from GEE
            
        Returns:
            bool: True if task is still running
        """
        if task_id not in self.active_monitors:
            return False
        
        monitor = self.active_monitors[task_id]
        state = task_status.get('state', 'UNKNOWN')
        
        # Update progress
        if state == 'RUNNING':
            # GEE provides progress as 0-1 float
            gee_progress = task_status.get('progress', 0)
            progress_pct = int(gee_progress * 100)
            
            # Only update if progress increased
            if progress_pct > monitor['last_progress']:
                monitor['progress'].update(progress_pct - monitor['last_progress'])
                monitor['last_progress'] = progress_pct
                monitor['last_update'] = time.time()
            
            # Estimate time remaining
            if progress_pct > 0:
                elapsed = time.time() - monitor['start_time']
                estimated_total = elapsed / (progress_pct / 100)
                remaining = estimated_total - elapsed
                
                if remaining > 0:
                    monitor['progress'].set_postfix({
                        'eta': f"{int(remaining/60)}m {int(remaining%60)}s"
                    })
            
            return True
            
        elif state == 'COMPLETED':
            # Complete the progress
            monitor['progress'].update(100 - monitor['last_progress'])
            monitor['progress'].close()
            
            elapsed = time.time() - monitor['start_time']
            self.logger.info(f"✅ GEE task completed: {monitor['description']} ({elapsed/60:.1f}m)")
            
            del self.active_monitors[task_id]
            return False
            
        elif state in ['FAILED', 'CANCELLED']:
            # Mark as failed
            monitor['progress'].close()
            
            error_msg = task_status.get('error_message', 'Unknown error')
            if state == 'FAILED':
                self.logger.error(f"❌ GEE task failed: {monitor['description']}")
                self.logger.error(f"   Error: {error_msg}")
            else:
                self.logger.warning(f"⚠️ GEE task cancelled: {monitor['description']}")
            
            del self.active_monitors[task_id]
            return False
        
        return True
    
    def monitor_with_callback(self, task: ee.batch.Task, 
                            description: str,
                            callback: Optional[callable] = None,
                            check_interval: int = 30) -> Dict:
        """
        Monitor a task with optional callback.
        
        Args:
            task: Earth Engine task
            description: Task description
            callback: Optional callback function(status_dict)
            check_interval: Check interval in seconds
            
        Returns:
            Final task status
        """
        task_id = task.id
        self.create_task_monitor(task_id, description)
        
        while True:
            # Get task status
            status = task.status()
            
            # Update progress
            still_running = self.update_task_progress(task_id, status)
            
            # Call callback if provided
            if callback:
                callback(status)
            
            if not still_running:
                return status
            
            time.sleep(check_interval)
    
    def batch_monitor(self, tasks: Dict[str, ee.batch.Task],
                     check_interval: int = 30) -> Dict[str, Dict]:
        """
        Monitor multiple tasks simultaneously.
        
        Args:
            tasks: Dict mapping description to task
            check_interval: Check interval in seconds
            
        Returns:
            Dict mapping description to final status
        """
        # Create monitors for all tasks
        for desc, task in tasks.items():
            self.create_task_monitor(task.id, desc)
        
        task_statuses = {}
        active_tasks = dict(tasks)
        
        while active_tasks:
            completed_tasks = []
            
            for desc, task in active_tasks.items():
                # Get status
                status = task.status()
                
                # Update progress
                still_running = self.update_task_progress(task.id, status)
                
                if not still_running:
                    task_statuses[desc] = status
                    completed_tasks.append(desc)
            
            # Remove completed tasks
            for desc in completed_tasks:
                del active_tasks[desc]
            
            if active_tasks:
                time.sleep(check_interval)
        
        return task_statuses
    
    def get_active_tasks(self) -> List[Dict]:
        """Get info about all active monitors"""
        active = []
        
        for task_id, monitor in self.active_monitors.items():
            active.append({
                'task_id': task_id,
                'description': monitor['description'],
                'progress': monitor['last_progress'],
                'duration': time.time() - monitor['start_time'],
                'last_update': time.time() - monitor['last_update']
            })
        
        return active