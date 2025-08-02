#!/usr/bin/env python3
from pathlib import Path
import json
import time
import os

progress_file = "Path(__file__).parent.parent.parent / outputs/analysis_results/som/production_run_20250801/som_progress_som_20250801_112047.json"

print("Checking SOM progress...")
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        data = json.load(f)
    
    elapsed = time.time() - 1738330847.696116  # start_time from file
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Current phase: {data['current_phase']}")
    print(f"Epoch: {data['epoch']}/{data['max_epochs']}")
    print(f"Last update: {data['last_update']}")
    
    # Check process
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if '3832723' in line and 'grep' not in line:
            print(f"\nProcess status: {line}")
else:
    print("Progress file not found")