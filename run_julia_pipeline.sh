#!/bin/bash
# Script to run the Julia pipeline with all three methods sequentially

set -e  # Exit on error

echo "=============================================="
echo "Starting Julia Pipeline for Methods 1, 2, 3"
echo "Date: $(date)"
echo "=============================================="

# Set paths
DATA_PATH="/scratch/yl998/biodiversity_climate_production_1x_corrected.arrow"
JULIA_BIN="$HOME/julia/bin/julia"
PIPELINE_SCRIPT="/home/yl998/Geo_process/src/machine_learning/pipelines/julia_hybrid_pipeline.jl"

# Method 1
echo ""
echo "=============================================="
echo "RUNNING METHOD 1 (H0-H1 Testing)"
echo "Time: $(date)"
echo "=============================================="
$JULIA_BIN $PIPELINE_SCRIPT method1 $DATA_PATH

if [ $? -eq 0 ]; then
    echo "✓ Method 1 completed successfully at $(date)"
else
    echo "✗ Method 1 failed at $(date)"
    exit 1
fi

# Method 2
echo ""
echo "=============================================="
echo "RUNNING METHOD 2 (Interactions)"
echo "Time: $(date)"
echo "=============================================="
$JULIA_BIN $PIPELINE_SCRIPT method2 $DATA_PATH

if [ $? -eq 0 ]; then
    echo "✓ Method 2 completed successfully at $(date)"
else
    echo "✗ Method 2 failed at $(date)"
    exit 1
fi

# Method 3 (DML)
echo ""
echo "=============================================="
echo "RUNNING METHOD 3 (Double Machine Learning)"
echo "Time: $(date)"
echo "=============================================="
$JULIA_BIN $PIPELINE_SCRIPT method3 $DATA_PATH

if [ $? -eq 0 ]; then
    echo "✓ Method 3 completed successfully at $(date)"
else
    echo "✗ Method 3 failed at $(date)"
    exit 1
fi

echo ""
echo "=============================================="
echo "ALL METHODS COMPLETED SUCCESSFULLY"
echo "End time: $(date)"
echo "=============================================="
echo ""
echo "Results saved to: results/julia_models/"
ls -lh results/julia_models/