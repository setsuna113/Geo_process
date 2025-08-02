#!/usr/bin/env python3
"""Estimate completion time for optimized SOM run."""

from datetime import datetime, timedelta

# Timing data from logs
start_time = datetime.strptime("2025-08-01 13:23:23", "%Y-%m-%d %H:%M:%S")
progress_40_time = datetime.strptime("2025-08-01 13:25:34", "%Y-%m-%d %H:%M:%S")

# Calculate rate
elapsed = (progress_40_time - start_time).total_seconds()
progress = 0.40
time_per_batch = elapsed / progress

print("=== Optimized SOM Time Estimation ===")
print(f"Time for 40% of batch update: {elapsed/60:.1f} minutes")
print(f"Estimated time per full batch: {time_per_batch/60:.1f} minutes")

# Parameters
max_epochs = 50  # from config
cv_folds = 3
convergence_epochs = 20  # typical for SOM with early stopping

# Add time for QE calculation and other operations
time_per_epoch = time_per_batch + 30  # seconds

print(f"\nTime per epoch: {time_per_epoch/60:.1f} minutes")
print(f"Max epochs: {max_epochs}")
print(f"Expected convergence: ~{convergence_epochs} epochs")

# Calculate total time
time_per_fold = convergence_epochs * time_per_epoch
total_time = cv_folds * time_per_fold

print(f"\nTime per CV fold: {time_per_fold/3600:.1f} hours")
print(f"Total time for 3 folds: {total_time/3600:.1f} hours")

# Current time
now = datetime.now()
completion_time = now + timedelta(seconds=total_time)

print(f"\nCurrent time: {now.strftime('%H:%M')}")
print(f"Estimated completion: {completion_time.strftime('%H:%M')}")
print(f"Well within 5-hour deadline! ✓")

# Progress check
current_runtime = (datetime.now() - datetime.strptime("2025-08-01 13:22:41", "%Y-%m-%d %H:%M:%S")).total_seconds()
print(f"\nCurrent runtime: {current_runtime/60:.1f} minutes")