#!/bin/bash
# Main Pipeline Launcher - Geo Biodiversity Analysis System
# Redirects to the organized production script

echo "🌍 Geo Biodiversity Analysis Pipeline"
echo "====================================="
echo "   Redirecting to production script..."
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Execute the monitored production script from the project root
cd "$SCRIPT_DIR"
exec ./scripts/production/run_monitored_pipeline.sh "$@"