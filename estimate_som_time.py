#!/usr/bin/env python3
"""Estimate SOM completion time."""

# Parameters
n_samples_cv = 7_671_105  # samples per CV fold
n_cv_folds = 3
n_epochs = 1000  # max epochs
n_neurons = 225
chunk_size = 10_000
n_chunks = (n_samples_cv + chunk_size - 1) // chunk_size

# From our test: 0.0392s for 1000 samples with combined distances
# Scale up: 7.67M samples
time_per_chunk = 0.0392  # seconds
time_per_batch = n_chunks * time_per_chunk
time_per_epoch = time_per_batch + 10  # add time for QE calculation etc

print("=== SOM Time Estimation ===")
print(f"Samples per CV fold: {n_samples_cv:,}")
print(f"CV folds: {n_cv_folds}")
print(f"Max epochs per fold: {n_epochs}")
print(f"Neurons: {n_neurons}")
print(f"Chunk size: {chunk_size:,}")
print(f"Chunks per epoch: {n_chunks}")

print(f"\nTime per chunk: {time_per_chunk:.3f}s")
print(f"Time per batch update: {time_per_batch/60:.1f} minutes")
print(f"Time per epoch: {time_per_epoch/60:.1f} minutes")

# Assume convergence at 10% of max epochs (typical for SOM)
convergence_epochs = int(n_epochs * 0.1)
time_per_fold = convergence_epochs * time_per_epoch
total_time = n_cv_folds * time_per_fold

print(f"\nAssuming convergence at {convergence_epochs} epochs:")
print(f"Time per CV fold: {time_per_fold/3600:.1f} hours")
print(f"Total time for 3 folds: {total_time/3600:.1f} hours")

# Current progress
runtime_minutes = 8.5  # from monitoring
epochs_completed = runtime_minutes / (time_per_epoch/60)
print(f"\nCurrent progress estimate:")
print(f"Runtime: {runtime_minutes:.1f} minutes")
print(f"Estimated epochs completed: {epochs_completed:.1f}")

# But the chunk processing is taking longer than expected
# Let's recalculate based on actual runtime
if epochs_completed < 1:
    # Still in first epoch
    actual_time_per_epoch = runtime_minutes * 60  # seconds
    print(f"\nActual time for first epoch: {actual_time_per_epoch/60:.1f} minutes")
    
    # Recalculate total time
    time_per_fold_actual = convergence_epochs * actual_time_per_epoch
    total_time_actual = n_cv_folds * time_per_fold_actual
    
    print(f"\nRevised estimate based on actual performance:")
    print(f"Time per CV fold: {time_per_fold_actual/3600:.1f} hours")
    print(f"Total time for 3 folds: {total_time_actual/3600:.1f} hours")
    
    if total_time_actual/3600 > 5:
        print(f"\n⚠️  WARNING: Estimated time ({total_time_actual/3600:.1f}h) exceeds 5-hour deadline!")