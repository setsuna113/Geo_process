#!/usr/bin/env python3
"""Live monitoring of SOM training progress."""

import json
import time
import os
from datetime import datetime

def monitor_progress(progress_file, interval=30):
    """Monitor progress file and display updates."""
    print(f"Monitoring SOM progress: {progress_file}")
    print("Press Ctrl+C to stop\n")
    
    last_update = None
    
    try:
        while True:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                
                current_update = data.get('last_update')
                
                # Print header every 10 updates
                if last_update is None:
                    print(f"{'Time':<8} {'Phase':<12} {'Epoch':<6} {'QE':<10} {'LR':<8} {'Elapsed':<10}")
                    print("-" * 60)
                
                # Only print if there's a change
                if current_update != last_update:
                    now = datetime.now().strftime("%H:%M:%S")
                    phase = data.get('current_phase', 'unknown')
                    epoch = data.get('epoch', 0)
                    max_epochs = data.get('max_epochs', 0)
                    qe = data.get('quantization_error', 0)
                    lr = data.get('learning_rate', 0)
                    elapsed = data.get('elapsed_seconds', 0)
                    
                    if isinstance(qe, (int, float)) and qe > 0:
                        qe_str = f"{qe:.6f}"
                    else:
                        qe_str = "N/A"
                    
                    if isinstance(lr, (int, float)) and lr > 0:
                        lr_str = f"{lr:.4f}"
                    else:
                        lr_str = "N/A"
                    
                    print(f"{now:<8} {phase:<12} {epoch:>3}/{max_epochs:<3} {qe_str:<10} {lr_str:<8} {elapsed/60:>6.1f}min")
                    
                    last_update = current_update
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    progress_file = "/home/yl998/dev/geo/outputs/analysis_results/som/som_progress_som_20250801_101125.json"
    monitor_progress(progress_file, interval=10)