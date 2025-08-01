"""Monitoring and progress tracking for rasterio resampler."""

import json
import os
import time
import psutil
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ResamplingMonitor:
    """Monitor memory, CPU, and progress for resampling operations."""
    
    def __init__(self, config: 'ResamplingConfig', total_windows: int):
        """Initialize monitor.
        
        Args:
            config: Resampling configuration
            total_windows: Total number of windows to process
        """
        self.config = config
        self.total_windows = total_windows
        self.processed_windows = 0
        self.start_time = time.time()
        self.last_checkpoint = 0
        
        # Progress tracking
        self.progress_file = config.progress_file
        self.checkpoint_interval = config.checkpoint_interval
        
        # Resource monitoring
        self.process = psutil.Process()
        self.peak_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        
        # Error tracking
        self.errors = []
        self.warnings = []
        
        # Load existing progress if resuming
        self.completed_windows = set()
        self._load_progress()
    
    def start_monitoring(self):
        """Start background resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_resources(self):
        """Background thread for resource monitoring."""
        while self.monitoring:
            try:
                # Memory usage
                memory_info = self.process.memory_info()
                current_memory_mb = memory_info.rss / (1024**2)
                self.peak_memory = max(self.peak_memory, current_memory_mb)
                
                # Check memory pressure
                memory_percent = self.process.memory_percent()
                if memory_percent > 80:
                    logger.warning(f"High memory usage: {memory_percent:.1f}%")
                
                # CPU usage
                cpu_percent = self.process.cpu_percent(interval=1)
                
                # Log periodically in debug mode
                if self.config.debug and int(time.time()) % 30 == 0:
                    logger.debug(f"Resources - Memory: {current_memory_mb:.1f} MB "
                               f"({memory_percent:.1f}%), CPU: {cpu_percent:.1f}%")
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                
            time.sleep(5)  # Check every 5 seconds
    
    def _json_encoder(self, obj):
        """Custom JSON encoder for numpy types."""
        import numpy as np
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    def update_progress(self, window_id: str, window_info: Dict[str, Any]):
        """Update progress for a completed window.
        
        Args:
            window_id: Unique identifier for the window
            window_info: Information about the processed window
        """
        self.processed_windows += 1
        self.completed_windows.add(window_id)
        
        # Calculate metrics
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.processed_windows / self.total_windows) * 100
        windows_per_second = self.processed_windows / elapsed_time if elapsed_time > 0 else 0
        eta_seconds = (self.total_windows - self.processed_windows) / windows_per_second if windows_per_second > 0 else 0
        
        # Log progress
        logger.info(f"Progress: {self.processed_windows}/{self.total_windows} "
                   f"({progress_percent:.1f}%) - "
                   f"{windows_per_second:.2f} windows/sec - "
                   f"ETA: {self._format_time(eta_seconds)}")
        
        # Save checkpoint if needed
        if self.processed_windows - self.last_checkpoint >= self.checkpoint_interval:
            self._save_checkpoint(window_info)
            self.last_checkpoint = self.processed_windows
    
    def is_window_completed(self, window_id: str) -> bool:
        """Check if a window has already been processed."""
        return window_id in self.completed_windows
    
    def add_error(self, error: str, window_id: Optional[str] = None):
        """Record an error."""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'window_id': window_id,
            'error': error
        })
        logger.error(f"Error processing window {window_id}: {error}")
    
    def add_warning(self, warning: str):
        """Record a warning."""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'warning': warning
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        elapsed_time = time.time() - self.start_time
        
        return {
            'total_windows': self.total_windows,
            'processed_windows': self.processed_windows,
            'progress_percent': (self.processed_windows / self.total_windows) * 100 if self.total_windows > 0 else 0,
            'elapsed_time': self._format_time(elapsed_time),
            'elapsed_seconds': elapsed_time,
            'windows_per_second': self.processed_windows / elapsed_time if elapsed_time > 0 else 0,
            'peak_memory_mb': self.peak_memory,
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'completed': self.processed_windows == self.total_windows
        }
    
    def _save_checkpoint(self, window_info: Dict[str, Any]):
        """Save progress checkpoint."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'processed_windows': self.processed_windows,
            'total_windows': self.total_windows,
            'completed_windows': list(self.completed_windows),
            'last_window_info': window_info,
            'errors': self.errors[-10:],  # Keep last 10 errors
            'warnings': self.warnings[-10:],  # Keep last 10 warnings
            'peak_memory_mb': self.peak_memory
        }
        
        # Atomic write with custom JSON encoder for numpy types
        temp_file = f"{self.progress_file}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=self._json_encoder)
        os.replace(temp_file, self.progress_file)
        
        logger.info(f"Checkpoint saved: {self.processed_windows}/{self.total_windows} windows")
    
    def _load_progress(self):
        """Load existing progress if available."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    checkpoint = json.load(f)
                
                self.completed_windows = set(checkpoint.get('completed_windows', []))
                self.processed_windows = len(self.completed_windows)
                self.errors = checkpoint.get('errors', [])
                self.warnings = checkpoint.get('warnings', [])
                
                logger.info(f"Resumed from checkpoint: {self.processed_windows}/{self.total_windows} windows")
                
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def cleanup(self):
        """Clean up resources and save final state."""
        self.stop_monitoring()
        
        # Save final checkpoint
        self._save_checkpoint({
            'status': 'completed' if self.processed_windows == self.total_windows else 'interrupted',
            'summary': self.get_summary()
        })
        
        # Remove progress file if completed successfully
        if self.processed_windows == self.total_windows and not self.errors:
            try:
                os.remove(self.progress_file)
                logger.info("Progress file removed after successful completion")
            except:
                pass
        
        # Clean up old progress files
        self._cleanup_old_progress_files()
    
    def _cleanup_old_progress_files(self, max_age_days: int = 7):
        """Remove progress files older than specified days."""
        try:
            progress_dir = os.path.dirname(self.progress_file)
            if not os.path.exists(progress_dir):
                return
            
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            
            for filename in os.listdir(progress_dir):
                if filename.endswith('.progress.json'):
                    filepath = os.path.join(progress_dir, filename)
                    try:
                        # Check file age
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > max_age_seconds:
                            # Read file to check if it's an incomplete run
                            with open(filepath, 'r') as f:
                                data = json.load(f)
                            
                            # Only remove if it's completed or very old (30+ days)
                            if data.get('status') == 'completed' or file_age > (30 * 24 * 60 * 60):
                                os.remove(filepath)
                                logger.debug(f"Removed old progress file: {filename}")
                    except Exception as e:
                        logger.debug(f"Could not clean up {filename}: {e}")
        except Exception as e:
            logger.debug(f"Error during progress file cleanup: {e}")