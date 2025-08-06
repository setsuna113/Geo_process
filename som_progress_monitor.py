#!/usr/bin/env python3
"""
Essential SOM Progress Monitor
Monitors SOM analysis progress with live updates
"""

import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import re
import psutil

class SOMProgressMonitor:
    def __init__(self):
        self.log_dir = Path("logs")
        self.checkpoint_dir = Path("checkpoints/som")
        self.output_dir = Path("outputs/analysis_results/som")
        self.last_update = {}
        self.start_time = None
        
    def find_latest_log(self):
        """Find the most recent SOM log file."""
        patterns = ["analysis_standalone_*.log", "som_*.log", "analysis_som_*.log"]
        all_logs = []
        
        for pattern in patterns:
            all_logs.extend(self.log_dir.glob(pattern))
        
        if not all_logs:
            return None
            
        # Sort by modification time
        return max(all_logs, key=lambda x: x.stat().st_mtime)
    
    def parse_log_line(self, line):
        """Extract key information from log line."""
        info = {}
        
        # Epoch information
        if "Starting epoch" in line:
            match = re.search(r"Starting epoch (\d+)/(\d+)", line)
            if match:
                info['current_epoch'] = int(match.group(1))
                info['total_epochs'] = int(match.group(2))
                
        # CV fold
        elif "Processing CV fold" in line:
            match = re.search(r"Processing CV fold (\d+)/(\d+)", line)
            if match:
                info['current_fold'] = int(match.group(1))
                info['total_folds'] = int(match.group(2))
        
        # Quantization error
        elif "QE:" in line or "QE=" in line:
            match = re.search(r"QE[=:]\s*([\d.]+)", line)
            if match:
                info['qe'] = float(match.group(1))
        
        # Worker/chunk info
        elif "Processing" in line and "chunks" in line:
            match = re.search(r"Processing (\d+) chunks.*with (\d+) workers", line)
            if match:
                info['chunks'] = int(match.group(1))
                info['workers'] = int(match.group(2))
                
        # Memory/CPU info
        elif "Memory:" in line and "CPU:" in line:
            mem_match = re.search(r"Memory:\s*([\d.]+)%", line)
            cpu_match = re.search(r"CPU:\s*([\d.]+)%", line)
            if mem_match:
                info['memory_percent'] = float(mem_match.group(1))
            if cpu_match:
                info['cpu_percent'] = float(cpu_match.group(1))
                
        return info
    
    def get_system_stats(self):
        """Get current system statistics."""
        stats = {}
        
        # CPU load
        try:
            load_avg = os.getloadavg()
            stats['load_1m'] = load_avg[0]
            stats['load_5m'] = load_avg[1]
        except:
            pass
        
        # Python processes
        try:
            result = subprocess.run(
                ['ps', 'aux'], 
                capture_output=True, 
                text=True
            )
            python_procs = len([l for l in result.stdout.split('\n') 
                              if 'python' in l and 'som' in l.lower()])
            stats['python_processes'] = python_procs
        except:
            pass
            
        return stats
    
    def display_status(self, status):
        """Display formatted status update."""
        os.system('clear')
        print("╔══════════════════════════════════════════════════════════╗")
        print("║               SOM Analysis Progress Monitor              ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("─" * 60)
        
        # Analysis progress
        if 'current_fold' in status:
            print(f"📊 CV Fold: {status['current_fold']}/{status['total_folds']}")
            
        if 'current_epoch' in status:
            epoch_pct = (status['current_epoch'] / status['total_epochs']) * 100
            print(f"🔄 Epoch: {status['current_epoch']}/{status['total_epochs']} ({epoch_pct:.1f}%)")
            
        if 'qe' in status:
            print(f"📈 QE: {status['qe']:.6f}")
            
        # Processing info
        print("\n💻 Processing:")
        if 'chunks' in status:
            print(f"   Chunks: {status['chunks']}")
        if 'workers' in status:
            print(f"   Workers: {status['workers']}")
            
        # System stats
        sys_stats = self.get_system_stats()
        print("\n🖥️  System:")
        if 'load_1m' in sys_stats:
            print(f"   Load: {sys_stats['load_1m']:.2f} (1m), {sys_stats['load_5m']:.2f} (5m)")
        if 'python_processes' in sys_stats:
            print(f"   Python processes: {sys_stats['python_processes']}")
            
        if 'memory_percent' in status:
            print(f"   Memory: {status['memory_percent']:.1f}%")
        if 'cpu_percent' in status:
            print(f"   CPU: {status['cpu_percent']:.1f}%")
            
        print("\n" + "─" * 60)
        print("Press Ctrl+C to exit")
    
    def monitor(self):
        """Main monitoring loop."""
        print("Starting SOM progress monitor...")
        
        # Find log file
        log_file = self.find_latest_log()
        if not log_file:
            print("Waiting for log file...")
            while not log_file:
                time.sleep(1)
                log_file = self.find_latest_log()
                
        print(f"Monitoring: {log_file}")
        
        # Open log file and seek to end
        with open(log_file, 'r') as f:
            # Go to end of file
            f.seek(0, 2)
            
            try:
                while True:
                    line = f.readline()
                    if line:
                        info = self.parse_log_line(line.strip())
                        if info:
                            self.last_update.update(info)
                            self.display_status(self.last_update)
                    else:
                        # Update system stats even if no new log lines
                        self.display_status(self.last_update)
                        time.sleep(2)
                        
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    monitor = SOMProgressMonitor()
    monitor.monitor()