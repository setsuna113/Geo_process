#!/bin/bash
# Quick SOM progress summary

LOG_FILE=$(ls -t logs/analysis_standalone_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No active SOM log found"
    exit 1
fi

echo "=== SOM TRAINING PROGRESS SUMMARY ==="
echo "Log: $LOG_FILE"
echo "Time: $(date)"
echo ""

# Get current epoch
CURRENT_EPOCH=$(grep -E "Epoch [0-9]+/[0-9]+" "$LOG_FILE" | tail -1 | grep -oE "Epoch [0-9]+/[0-9]+")
echo "Progress: $CURRENT_EPOCH"

# Get latest QE
LATEST_QE=$(grep -E "QE=[0-9.]+" "$LOG_FILE" | tail -1 | grep -oE "QE=[0-9.]+" | cut -d'=' -f2)
echo "Current QE: $LATEST_QE"

# Get convergence status
if grep -q "converged" "$LOG_FILE"; then
    echo "Status: CONVERGED ✅"
else
    echo "Status: Training..."
fi

# Get timing
echo ""
echo "=== PERFORMANCE ==="
# Count epochs and calculate average time
EPOCH_COUNT=$(grep -c "Completed epoch" "$LOG_FILE")
if [ $EPOCH_COUNT -gt 1 ]; then
    FIRST_TIME=$(grep "Starting epoch 1/1000" "$LOG_FILE" | head -1 | cut -d' ' -f2)
    LAST_TIME=$(grep "Completed epoch" "$LOG_FILE" | tail -1 | cut -d' ' -f2)
    
    # Convert to seconds
    FIRST_SEC=$(date -d "$FIRST_TIME" +%s 2>/dev/null || echo "0")
    LAST_SEC=$(date -d "$LAST_TIME" +%s 2>/dev/null || echo "0")
    
    if [ $FIRST_SEC -ne 0 ] && [ $LAST_SEC -ne 0 ]; then
        TOTAL_SEC=$((LAST_SEC - FIRST_SEC))
        AVG_SEC=$((TOTAL_SEC / EPOCH_COUNT))
        echo "Epochs completed: $EPOCH_COUNT"
        echo "Average time per epoch: ${AVG_SEC}s"
        
        # Estimate remaining time
        MAX_EPOCHS=$(echo "$CURRENT_EPOCH" | cut -d'/' -f2)
        REMAINING=$((MAX_EPOCHS - EPOCH_COUNT))
        EST_REMAINING=$((REMAINING * AVG_SEC))
        EST_HOURS=$((EST_REMAINING / 3600))
        EST_MINS=$(((EST_REMAINING % 3600) / 60))
        echo "Estimated time remaining: ${EST_HOURS}h ${EST_MINS}m"
    fi
fi

# Check resource usage
echo ""
echo "=== RESOURCES ==="
WORKERS=$(grep -E "using [0-9]+ workers" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+ workers" | cut -d' ' -f1)
echo "Parallel workers: $WORKERS"

# Latest memory/CPU
LATEST_RESOURCE=$(grep -E "Memory: [0-9.]+%, CPU: [0-9.]+%" "$LOG_FILE" | tail -1)
if [ ! -z "$LATEST_RESOURCE" ]; then
    echo "Latest: $LATEST_RESOURCE"
fi

# Check for current CV fold
echo ""
echo "=== CROSS-VALIDATION ==="
CV_STATUS=$(grep -E "Processing CV fold [0-9]+/[0-9]+" "$LOG_FILE" | tail -1)
if [ ! -z "$CV_STATUS" ]; then
    echo "$CV_STATUS"
fi