#!/usr/bin/env python3
"""Debug script to understand why progress isn't updating."""

import numpy as np
import time

# Simulate the computational complexity
n_samples = 9_000_000  # ~80% of 11.3M for training
n_neurons = 225  # 15x15 grid
n_features = 4
missing_rate = 0.948

print(f"Estimating computation time for partial Bray-Curtis...")
print(f"Samples: {n_samples:,}")
print(f"Neurons: {n_neurons}")
print(f"Features: {n_features}")
print(f"Missing rate: {missing_rate:.1%}")

# Estimate operations per epoch
operations_per_epoch = n_samples * n_neurons  # For finding BMUs
print(f"\nOperations per epoch: {operations_per_epoch:,}")

# Estimate based on ~100 microseconds per distance calculation with missing data
time_per_operation = 100e-6  # 100 microseconds
estimated_epoch_time = operations_per_epoch * time_per_operation

print(f"Estimated time per epoch: {estimated_epoch_time/60:.1f} minutes")
print(f"For 1000 epochs: {estimated_epoch_time*1000/3600:.1f} hours")

# Check what's in the output directory
import os
import json

output_dir = "/home/yl998/dev/geo/outputs/analysis_results/som"
print(f"\nFiles in {output_dir}:")
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.json'):
        path = os.path.join(output_dir, f)
        mtime = os.path.getmtime(path)
        print(f"  {f} - modified {time.time() - mtime:.1f} seconds ago")
        
# Check if we can read the progress file
progress_file = "/home/yl998/dev/geo/outputs/analysis_results/som/som_progress_som_20250801_101125.json"
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        data = json.load(f)
    print(f"\nProgress file content:")
    for k, v in data.items():
        print(f"  {k}: {v}")