#!/usr/bin/env python3
"""Monitor the fixed SOM run progress."""

import json
import time
import psutil
import os
from datetime import datetime

def check_som_progress():
    # Find the process
    som_process = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'run_analysis.py' in ' '.join(proc.info['cmdline'] or []):
                if '--method som' in ' '.join(proc.info['cmdline']):
                    som_process = proc
                    break
        except:
            pass
    
    if not som_process:
        print("SOM process not found")
        return
    
    # Get process info
    pid = som_process.pid
    cpu_percent = som_process.cpu_percent(interval=1)
    memory_info = som_process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    memory_gb = memory_mb / 1024
    
    # Calculate runtime
    create_time = datetime.fromtimestamp(som_process.create_time())
    runtime = datetime.now() - create_time
    
    print(f"\n=== SOM Process Status ===")
    print(f"PID: {pid}")
    print(f"CPU Usage: {cpu_percent:.1f}%")
    print(f"Memory Usage: {memory_gb:.2f} GB ({memory_mb:.0f} MB)")
    print(f"Runtime: {runtime}")
    
    # Check latest progress file
    progress_files = []
    output_dir = "outputs/analysis_results/som/production_run_20250801"
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith("som_progress_") and f.endswith(".json"):
                progress_files.append(os.path.join(output_dir, f))
    
    if progress_files:
        latest_file = max(progress_files, key=os.path.getmtime)
        print(f"\nLatest progress file: {os.path.basename(latest_file)}")
        
        try:
            with open(latest_file, 'r') as f:
                progress = json.load(f)
            
            print(f"Status: {progress.get('status', 'unknown')}")
            print(f"Phase: {progress.get('current_phase', 'unknown')}")
            print(f"CV Fold: {progress.get('cv_fold', '?')}/3")
            print(f"Epoch: {progress.get('epoch', '?')}/{progress.get('max_epochs', '?')}")
            
            if progress.get('quantization_error'):
                print(f"QE: {progress['quantization_error']:.6f}")
            if progress.get('learning_rate'):
                print(f"LR: {progress['learning_rate']:.4f}")
            
            # Estimate time remaining
            if progress.get('epoch') and progress.get('max_epochs') and progress.get('elapsed_seconds'):
                epoch_progress = progress['epoch'] / progress['max_epochs']
                if epoch_progress > 0:
                    total_time_est = progress['elapsed_seconds'] / epoch_progress
                    remaining = total_time_est - progress['elapsed_seconds']
                    hours = int(remaining // 3600)
                    minutes = int((remaining % 3600) // 60)
                    print(f"Estimated time remaining: {hours}h {minutes}m")
        except Exception as e:
            print(f"Error reading progress: {e}")
    
    # Check log file
    log_file = "som_fixed_run.log"
    if os.path.exists(log_file):
        # Get last few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-10:]
        
        print(f"\n=== Recent Log Entries ===")
        for line in recent_lines:
            if any(keyword in line for keyword in ['Epoch', 'QE=', 'progress', 'batch update', 'chunk']):
                print(line.strip())
    
    # Memory analysis
    print(f"\n=== Memory Analysis ===")
    print(f"Working set: {memory_gb:.2f} GB")
    print(f"Chunks: 768 (7.67M samples / 10k per chunk)")
    print(f"Processing with geographic distances")
    
    # Performance estimate
    if cpu_percent > 100:
        print(f"\nPerformance: Using multiple cores ({cpu_percent:.0f}% CPU)")
    
    elapsed_minutes = runtime.total_seconds() / 60
    print(f"\nProcessing rate: ~{7671105 / elapsed_minutes:.0f} samples/minute")

if __name__ == "__main__":
    check_som_progress()