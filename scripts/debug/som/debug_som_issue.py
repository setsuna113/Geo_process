#!/usr/bin/env python3
"""Debug why SOM is stuck at epoch 0."""

import sys
import psutil
import subprocess

# Check if the process is actually doing something
pid = 3817140
try:
    p = psutil.Process(pid)
    print(f"Process {pid} status:")
    print(f"  Status: {p.status()}")
    print(f"  CPU %: {p.cpu_percent(interval=1.0)}")
    print(f"  Memory: {p.memory_info().rss / 1024**3:.2f} GB")
    print(f"  Threads: {p.num_threads()}")
    
    # Check what the process is doing
    print("\nProcess stack trace sample:")
    try:
        # Use py-spy to sample what Python is doing
        result = subprocess.run(['py-spy', 'dump', '--pid', str(pid)], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Show first 50 lines of stack trace
            lines = result.stdout.split('\n')[:50]
            print('\n'.join(lines))
        else:
            print("py-spy not available or failed")
    except:
        print("Could not get stack trace")
        
except psutil.NoSuchProcess:
    print(f"Process {pid} not found")
except Exception as e:
    print(f"Error: {e}")

# Check if there's an infinite loop issue
print("\nChecking for potential infinite loop...")
print("If max_epochs is 0, the for loop would exit immediately")
print("If it's stuck at epoch 0, it might be stuck in _batch_update or calculate_quantization_error")