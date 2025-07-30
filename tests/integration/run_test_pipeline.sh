#!/bin/bash
# Simple test pipeline run

echo "Running test pipeline..."
export EXPERIMENT_NAME="integration_test_$(date +%s)"

# Use the standard run script with small test
timeout 120 python scripts/run_pipeline.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --stages data_load resample \
    --no-resume \
    2>&1 | head -200

echo "Test complete!"