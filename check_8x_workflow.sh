#!/bin/bash
# Monitor script for 8x-only GEE workflow

echo "======================================"
echo "8x-Only GEE Workflow Status"
echo "======================================"
echo ""

# Check if tmux session is running
if tmux has-session -t gee_8x_workflow 2>/dev/null; then
    echo "✅ Workflow session is RUNNING"
    echo ""
    
    # Show last few lines from tmux
    echo "Latest output from tmux session:"
    echo "--------------------------------"
    tmux capture-pane -t gee_8x_workflow -p | tail -10
    echo ""
else
    echo "❌ Workflow session is NOT running"
    echo ""
fi

# Check log file
if [ -f workflow_8x_debug.log ]; then
    echo "Log file status:"
    echo "----------------"
    echo "Size: $(ls -lh workflow_8x_debug.log | awk '{print $5}')"
    echo "Last modified: $(date -r workflow_8x_debug.log '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "Last 20 lines of log:"
    echo "---------------------"
    tail -20 workflow_8x_debug.log
else
    echo "Log file not found yet..."
fi

echo ""
echo "======================================"
echo "To interact with the workflow:"
echo "  tmux attach -t gee_8x_workflow    # Attach to session"
echo "  tail -f workflow_8x_debug.log     # Follow log"
echo "  ./check_8x_workflow.sh            # Run this status check"
echo "======================================" 