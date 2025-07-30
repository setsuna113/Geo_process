#!/bin/bash
# Test pipeline runner with custom config

echo "🧪 Running Test Pipeline with Test Rasters"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set test config
export GEO_CONFIG_PATH="$SCRIPT_DIR/config_test.yml"
echo "📋 Using config: $GEO_CONFIG_PATH"

# Create test data if not exists
if [ ! -d "test_data/rasters" ]; then
    echo "📊 Creating test rasters..."
    python scripts/create_test_rasters.py
fi

# Clean up any previous test runs
echo "🧹 Cleaning up previous test runs..."
rm -rf outputs/test_integration_* 2>/dev/null || true
rm -rf checkpoint_outputs/test_integration_* 2>/dev/null || true

# Run pipeline with test experiment name
echo ""
echo "🚀 Starting pipeline..."
echo ""

# Use the monitored pipeline script
./scripts/production/run_monitored_pipeline.sh \
    --experiment-name "test_integration_$(date +%Y%m%d_%H%M%S)" \
    --analysis-method "som" \
    "$@"