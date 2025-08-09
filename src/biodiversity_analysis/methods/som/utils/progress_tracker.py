"""Progress tracking for SOM training with file-based updates."""

import os
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import threading


class SOMProgressTracker:
    """Track SOM training progress to a file for external monitoring."""
    
    def __init__(self, output_dir: str, experiment_name: str):
        """Initialize progress tracker.
        
        Args:
            output_dir: Directory for progress files
            experiment_name: Name of the experiment
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.progress_file = os.path.join(output_dir, f"som_progress_{experiment_name}.json")
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize progress data
        self.progress_data = {
            'experiment': experiment_name,
            'start_time': datetime.now().isoformat(),
            'status': 'initializing',
            'current_phase': 'setup',
            'cv_fold': None,
            'epoch': 0,
            'max_epochs': 0,
            'quantization_error': None,
            'learning_rate': None,
            'elapsed_seconds': 0,
            'estimated_remaining': None,
            'last_update': datetime.now().isoformat()
        }
        
        self.start_time = time.time()
        self._write_progress()
        
    def update_phase(self, phase: str, cv_fold: Optional[int] = None):
        """Update current training phase."""
        self.progress_data['current_phase'] = phase
        self.progress_data['cv_fold'] = cv_fold
        self.progress_data['status'] = 'running'
        self._write_progress()
        
    def update_epoch(self, epoch: int, max_epochs: int, qe: float, 
                    lr: float, radius: float):
        """Update epoch-level progress."""
        self.progress_data.update({
            'epoch': epoch,
            'max_epochs': max_epochs,
            'quantization_error': float(qe),
            'learning_rate': float(lr),
            'radius': float(radius),
            'progress_percent': (epoch / max_epochs * 100) if max_epochs > 0 else 0
        })
        
        # Estimate remaining time
        elapsed = time.time() - self.start_time
        if epoch > 0:
            time_per_epoch = elapsed / epoch
            remaining_epochs = max_epochs - epoch
            self.progress_data['estimated_remaining'] = time_per_epoch * remaining_epochs
        
        self._write_progress()
        
    def mark_complete(self, success: bool = True):
        """Mark training as complete."""
        self.progress_data['status'] = 'completed' if success else 'failed'
        self.progress_data['end_time'] = datetime.now().isoformat()
        self.progress_data['total_time'] = time.time() - self.start_time
        self._write_progress()
        
    def _write_progress(self):
        """Write progress to file atomically."""
        self.progress_data['elapsed_seconds'] = time.time() - self.start_time
        self.progress_data['last_update'] = datetime.now().isoformat()
        
        # Write to temp file first
        temp_file = self.progress_file + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2)
        
        # Atomic rename
        os.replace(temp_file, self.progress_file)


def create_progress_callback(tracker: SOMProgressTracker, phase: str = 'training'):
    """Create a progress callback function for SOM training.
    
    Args:
        tracker: Progress tracker instance
        phase: Current phase name
        
    Returns:
        Callback function compatible with GeoSOM
    """
    def callback(progress_info):
        if isinstance(progress_info, dict):
            # Detailed progress info
            tracker.update_epoch(
                epoch=progress_info['epoch'],
                max_epochs=progress_info['max_epochs'],
                qe=progress_info['qe'],
                lr=progress_info['learning_rate'],
                radius=progress_info['radius']
            )
        else:
            # Simple progress (float)
            # Just update the phase progress
            tracker.progress_data['progress_percent'] = progress_info * 100
            tracker._write_progress()
    
    return callback