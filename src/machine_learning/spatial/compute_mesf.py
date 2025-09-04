#!/usr/bin/env python3
"""
Compute MESF from Arrow files.

Usage: python compute_mesf.py [input_file] [spatial_step]

Defaults to /scratch/yl998/*.arrow if no file specified.
"""

import sys

# CRITICAL: Configure threads BEFORE importing anything that uses Numba/OpenMP
from worker_control import WorkerController
controller = WorkerController(64)
controller.set_all_thread_limits()  # Sets environment variables

# NOW we can safely import everything else
import numpy as np
import pyarrow.feather as pf
from glob import glob

# Get input file - default to first arrow in scratch
input_file = sys.argv[1] if len(sys.argv) > 1 else (glob('/scratch/yl998/*.arrow') + [None])[0]
if not input_file:
    print("No arrow file found")
    sys.exit(1)

# Spatial sampling step
step = int(sys.argv[2]) if len(sys.argv) > 2 else 1

# Read with sampling - Arrow reader handles the step efficiently
print(f"Reading {input_file} (step={step})")
table = pf.read_table(input_file)

# Extract coordinates - look for any coord-like columns
coord_pairs = [('x', 'y'), ('longitude', 'latitude')]
for c1, c2 in coord_pairs:
    if c1 in table.column_names:
        # Simple extraction with sampling - ensure fp32
        x = table[c1].to_numpy()[::step].astype(np.float32)
        y = table[c2].to_numpy()[::step].astype(np.float32)
        coords = np.column_stack([x, y]).astype(np.float32)
        break
else:
    print(f"No coordinates in {table.column_names}")
    sys.exit(1)

print(f"Computing MESF for {len(coords):,} points...")

# Compute MESF
from mesf import compute_mesf_with_cache
result = compute_mesf_with_cache(coords)

print(f"Done: {result.shape}")
print(f"Cache: /scratch/yl998/.cache/mesf/")
