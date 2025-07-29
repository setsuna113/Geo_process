#!/bin/bash
# Quick launcher for monitored pipeline execution

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Execute the monitored pipeline script
exec "$SCRIPT_DIR/scripts/production/run_monitored_pipeline.sh" "$@"