#!/bin/bash
# Script to start Claude monitoring in tmux

echo "Starting Claude pipeline monitor in tmux..."

# Create the monitoring command
tmux new-session -d -s monitor_pipeline -n claude_monitor \
"claude -p \"\$(cat /home/yl998/dev/geo/monitor_pipeline_prompt.txt)\" --dangerously-allow-filesystem-access"

echo "Monitor started!"
echo ""
echo "Commands:"
echo "  View monitor:     tmux attach -t monitor_pipeline"
echo "  Check log:        tail -f ~/pipeline_monitor.log"
echo "  Stop monitor:     tmux kill-session -t monitor_pipeline"
echo ""
echo "The monitor will:"
echo "  - Check every 15 minutes"
echo "  - Log to ~/pipeline_monitor.log"
echo "  - Detect completion or failure"
echo "  - Provide intelligent debugging"