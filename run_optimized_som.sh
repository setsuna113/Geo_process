#!/bin/bash
# Run SOM with optimized parameters for sparse biodiversity data

echo "🚀 Starting Optimized SOM Analysis"
echo "=================================="
echo "Key optimizations:"
echo "  - 50x50 grid (2,500 neurons) for detailed patterns"
echo "  - Spatial weight 0.4 (60% biodiversity, 40% spatial)" 
echo "  - Equal feature weights [0.25 x4] for correlation discovery"
echo "  - Log transformation preserved for richness distributions"
echo "  - Exponential radius decay for sparse data"
echo "  - 200 epoch patience with 1000 max epochs"
echo "  - 5-fold CV with 500km spatial blocks"
echo ""

# Set experiment name with timestamp
EXPERIMENT_NAME="som_optimized_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/som_optimized.log"

# Clean any previous runs
echo "Cleaning previous tmux sessions..."
tmux kill-session -t som_optimized 2>/dev/null || true

# Create new tmux session
echo "Creating tmux session 'som_optimized'..."
tmux new-session -d -s som_optimized -n "Optimized SOM"

# Set environment and run
tmux send-keys -t som_optimized "cd /home/yl998/dev/geo" Enter
tmux send-keys -t som_optimized "export PYTHONUNBUFFERED=1" Enter
tmux send-keys -t som_optimized "export SOM_CONFIG_PATH=/home/yl998/dev/geo/src/config/som/som_config_optimized.yml" Enter

# Run with optimized config
tmux send-keys -t som_optimized "python scripts/run_analysis.py --experiment test_som_optimized --config-override som_config_path=/home/yl998/dev/geo/src/config/som/som_config_optimized.yml 2>&1 | tee $LOG_FILE" Enter

echo ""
echo "✅ Optimized SOM started in tmux session 'som_optimized'"
echo "📊 Log file: $LOG_FILE"
echo ""
echo "Monitor with:"
echo "  tmux attach -t som_optimized"
echo "  tail -f $LOG_FILE"
echo ""
echo "Expected improvements:"
echo "  - Lower QE (<0.05) through 50x50 grid"
echo "  - Better Moran's I through balanced spatial weight"
echo "  - Discovery of richness intercorrelations"
echo "  - Equal importance for all 4 richness measures"
echo "  - More stable convergence with patience=200"