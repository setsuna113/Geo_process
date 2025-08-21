#!/bin/bash
# Integrated Julia-Python Pipeline Runner
# Uses Julia for all processing, Python only for hypothesis testing and reporting

# Configuration
DATA_FILE="/scratch/yl998/biodiversity_climate_production_1x_corrected.arrow"
OUTPUT_DIR="results/integrated_pipeline"
LOG_DIR="logs/integrated"

# Spatial sampling options
# 1 = all rows (51M), 100 = 519k rows, 1000 = 51k rows, 10000 = 5k rows, 100000 = 519 rows
SPATIAL_STEP="${1:-10000}"  # Default to 10000 (approx 5k rows)

# MESF options
COMPUTE_MESF="${2:-true}"   # true or false
N_MESF="${3:-10}"           # Number of eigenvectors

echo "========================================================================"
echo "INTEGRATED JULIA-PYTHON ML PIPELINE"
echo "========================================================================"
echo "Data: $DATA_FILE"
echo "Spatial sampling: 1/$SPATIAL_STEP (approx $(echo "51901336 / $SPATIAL_STEP" | bc) rows)"
echo "MESF computation: $COMPUTE_MESF ($N_MESF eigenvectors)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Set timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

echo "Running pipeline (log: $LOG_FILE)..."
echo ""

# Option 1: Run Julia directly (faster, no Python overhead)
echo "Option 1: Direct Julia execution"
echo "--------------------------------"
~/julia/bin/julia julia_complete_pipeline.jl run_pipeline \
    "$DATA_FILE" \
    "$SPATIAL_STEP" \
    "$COMPUTE_MESF" \
    "$N_MESF" \
    2>&1 | tee "$LOG_FILE" | grep -E "^(✓|ERROR:|METHOD|Pipeline|AIC:|p-value:|effect:)"

echo ""
echo "Option 2: Python-integrated execution (with hypothesis testing and reports)"
echo "--------------------------------------------------------------------------"
~/anaconda3/envs/geo/bin/python test_integrated_pipeline.py 2>&1 | tee -a "$LOG_FILE" | grep -E "(TEST SUCCESSFUL|Failed|Hypothesis|consensus)"

echo ""
echo "========================================================================"
echo "Pipeline completed. Full log saved to: $LOG_FILE"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================================================"

# Display summary of results if successful
if grep -q "PIPELINE COMPLETED SUCCESSFULLY" "$LOG_FILE"; then
    echo ""
    echo "SUMMARY OF RESULTS"
    echo "------------------"
    grep -A 2 "Method 1 models fitted" "$LOG_FILE" | tail -2
    grep -A 2 "Method 2 models fitted" "$LOG_FILE" | tail -2
    grep -A 2 "Method 3 DML fitted" "$LOG_FILE" | tail -2
fi