#!/bin/bash
# Monitor SOM progress in a loop

echo "=== Monitoring SOM Progress ==="
echo "Started at: $(date '+%H:%M:%S')"
echo "Press Ctrl+C to stop"

while true; do
    # Get latest progress
    LATEST=$(tail -100 som_optimized_run.log | grep -E "(Epoch|QE=|progress|fold)" | tail -5)
    
    # Clear screen and show update
    clear
    echo "=== SOM Progress Monitor ==="
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    echo "Latest updates:"
    echo "$LATEST"
    
    # Check if completed
    if grep -q "Analysis complete" som_optimized_run.log; then
        echo ""
        echo "✓ Analysis complete!"
        break
    fi
    
    # Wait 30 seconds
    sleep 30
done