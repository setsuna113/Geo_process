# tests/integration/monitors.py
import psutil
import time
import threading
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from datetime import datetime

class SystemMonitor:
    """Monitor system resources during tests."""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.data = {
            'timestamps': [],
            'memory': [],
            'cpu': [],
            'disk_io': [],
            'db_connections': []
        }
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            timestamp = time.time()
            
            # Collect metrics
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Disk I/O (if available)
            try:
                io_counters = process.io_counters()
                disk_io_mb = (io_counters.read_bytes + io_counters.write_bytes) / 1024 / 1024
            except:
                disk_io_mb = 0
            
            # Store data
            self.data['timestamps'].append(timestamp)
            self.data['memory'].append(memory_mb)
            self.data['cpu'].append(cpu_percent)
            self.data['disk_io'].append(disk_io_mb)
            
            time.sleep(self.sample_interval)
    
    def generate_report(self, output_dir: Path) -> Dict[str, Any]:
        """Generate monitoring report."""
        if not self.data['timestamps']:
            return {'error': 'No monitoring data collected'}
        
        # Calculate statistics
        stats = {
            'duration': self.data['timestamps'][-1] - self.data['timestamps'][0],
            'memory': {
                'min': min(self.data['memory']),
                'max': max(self.data['memory']),
                'mean': sum(self.data['memory']) / len(self.data['memory'])
            },
            'cpu': {
                'min': min(self.data['cpu']),
                'max': max(self.data['cpu']),
                'mean': sum(self.data['cpu']) / len(self.data['cpu'])
            }
        }
        
        # Generate plots
        self._generate_plots(output_dir)
        
        return stats
    
    def _generate_plots(self, output_dir: Path):
        """Generate resource usage plots."""
        output_dir.mkdir(exist_ok=True)
        
        # Convert timestamps to relative seconds
        start_time = self.data['timestamps'][0]
        relative_times = [(t - start_time) for t in self.data['timestamps']]
        
        # Memory plot
        plt.figure(figsize=(10, 6))
        plt.plot(relative_times, self.data['memory'], 'b-', label='Memory (MB)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage During Test')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'memory_usage.png')
        plt.close()
        
        # CPU plot
        plt.figure(figsize=(10, 6))
        plt.plot(relative_times, self.data['cpu'], 'r-', label='CPU (%)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('CPU Usage (%)')
        plt.title('CPU Usage During Test')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'cpu_usage.png')
        plt.close()