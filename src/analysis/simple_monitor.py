"""
Simple monitoring to replace UnifiedMonitor
"""

import psutil
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleMonitor:
    """Simplified system monitoring without database dependencies."""
    
    def __init__(self, config: Optional[Dict] = None, db_manager: Optional[Any] = None):
        """Initialize simple monitor.
        
        Args:
            config: Configuration dict (optional)
            db_manager: Database manager (ignored)
        """
        self.config = config or {}
        self.start_time = time.time()
        self.checkpoints = []
        
    def log_memory(self, stage: str = ""):
        """Log current memory usage."""
        memory = psutil.virtual_memory()
        logger.info(f"Memory [{stage}]: {memory.percent:.1f}% used ({memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB)")
        
    def log_cpu(self, stage: str = ""):
        """Log current CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"CPU [{stage}]: {cpu_percent:.1f}% used")
        
    def log_disk(self, path: str = "/"):
        """Log disk usage."""
        disk = psutil.disk_usage(path)
        logger.info(f"Disk [{path}]: {disk.percent:.1f}% used ({disk.used / 1e9:.1f}GB / {disk.total / 1e9:.1f}GB)")
        
    def checkpoint(self, name: str, metrics: Optional[Dict] = None):
        """Create a monitoring checkpoint."""
        checkpoint = {
            'name': name,
            'timestamp': time.time(),
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(interval=0),
            'metrics': metrics or {}
        }
        self.checkpoints.append(checkpoint)
        logger.info(f"Checkpoint '{name}': Memory={checkpoint['memory_percent']:.1f}%, CPU={checkpoint['cpu_percent']:.1f}%")
        
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        elapsed = time.time() - self.start_time
        return {
            'elapsed_time': elapsed,
            'checkpoints': len(self.checkpoints),
            'current_memory': psutil.virtual_memory().percent,
            'current_cpu': psutil.cpu_percent(interval=0)
        }
        
    def cleanup(self):
        """Cleanup resources (no-op for simple version)."""
        pass
        
    def start_monitoring(self):
        """Start monitoring (no-op for simple version)."""
        logger.info("Monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring (no-op for simple version)."""
        logger.info("Monitoring stopped")


# Alias for compatibility
UnifiedMonitor = SimpleMonitor