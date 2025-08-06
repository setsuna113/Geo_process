#!/bin/bash

# Run GEE Production Workflow
# This script executes the complete GEE pipeline to process biodiversity and climate data

set -e

echo "🌍 Starting GEE Production Workflow"
echo "===================================="
echo ""

# Configuration
CONFIG_FILE="config/gee_config.yml"
EXPERIMENT_NAME="production_$(date +%Y%m%d_%H%M%S)"

echo "📋 Configuration:"
echo "  - Config: $CONFIG_FILE"
echo "  - Experiment: $EXPERIMENT_NAME"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "🐍 Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate geo
    echo "  - Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
fi

echo ""
echo "🚀 Running GEE workflow..."
echo "--------------------------"

# Run the workflow with resume capability
python scripts/run_gee_workflow.py \
    --config "$CONFIG_FILE" \
    --experiment-name "$EXPERIMENT_NAME" \
    --resume

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GEE workflow completed successfully!"
    echo ""
    echo "📦 Output files:"
    echo "  - Fine resolution (1km): data/merged/biodiversity_climate_fine_*.parquet"
    echo "  - Coarse resolution (10km): data/merged/biodiversity_climate_coarse_*.parquet"
    echo ""
    echo "To transfer to Cambridge server:"
    echo "  rsync -avP data/merged/*.parquet yl998@sherwood.cl.cam.ac.uk:/sparse/yl998/data/"
else
    echo ""
    echo "❌ GEE workflow failed. Check logs for details."
    exit 1
fi