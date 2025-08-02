#!/bin/bash
# Simple SOM progress monitor

LOG_FILE="/home/yl998/dev/geo/logs/som_parallel_production.log"

while true; do
    clear
    echo "=== SOM Training Progress ==="
    echo "Time: $(date)"
    echo ""
    
    # Get current epoch
    CURRENT_EPOCH=$(grep "Epoch [0-9]\+/1000:" "$LOG_FILE" | tail -1)
    if [ ! -z "$CURRENT_EPOCH" ]; then
        echo "Current: $CURRENT_EPOCH"
    fi
    
    # Get CV fold
    CV_FOLD=$(grep "Processing CV fold" "$LOG_FILE" | tail -1)
    echo "$CV_FOLD"
    
    # Check for convergence
    CONVERGED=$(grep "Converged at epoch" "$LOG_FILE" | tail -1)
    if [ ! -z "$CONVERGED" ]; then
        echo "Status: $CONVERGED"
    fi
    
    # Show latest chunk progress
    echo ""
    echo "Latest activity:"
    grep -E "Done.*tasks|Parallel processing complete|Starting epoch" "$LOG_FILE" | tail -5
    
    sleep 5
done