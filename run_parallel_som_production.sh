#!/bin/bash
# Run SOM analysis with parallel processing on real biodiversity data

echo "🚀 Starting Parallel SOM Production Run"
echo "========================================"

# Project directory
PROJECT_DIR="/home/yl998/dev/geo"
cd "$PROJECT_DIR"

# Create logs directory
mkdir -p logs

# Set experiment name with timestamp
EXPERIMENT_NAME="som_parallel_production_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/som_parallel_production.log"

echo "Experiment: $EXPERIMENT_NAME"
echo "Log file: $LOG_FILE"
echo ""

# Create tmux session for monitoring
SESSION_NAME="som_parallel_test"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create new session
tmux new-session -d -s "$SESSION_NAME" -n "SOM Analysis"

# Main pane: Run the analysis
tmux send-keys -t "$SESSION_NAME:0.0" "cd $PROJECT_DIR" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "./run_analysis.sh --experiment test_som 2>&1 | tee $LOG_FILE" C-m

# Split horizontally for monitoring
tmux split-window -h -t "$SESSION_NAME:0"
tmux send-keys -t "$SESSION_NAME:0.1" "cd $PROJECT_DIR && python monitor_parallel_som.py" C-m

# Split vertically for htop
tmux split-window -v -t "$SESSION_NAME:0.0"
tmux send-keys -t "$SESSION_NAME:0.2" "htop -d 2" C-m

# Split for log tail
tmux split-window -v -t "$SESSION_NAME:0.1"
tmux send-keys -t "$SESSION_NAME:0.3" "tail -f $LOG_FILE | grep -E '(Epoch|chunk|parallel|QE=|workers|Processing)'" C-m

# Set pane titles
tmux select-pane -t "$SESSION_NAME:0.0" -T "SOM Analysis"
tmux select-pane -t "$SESSION_NAME:0.1" -T "Monitor"
tmux select-pane -t "$SESSION_NAME:0.2" -T "System Resources"
tmux select-pane -t "$SESSION_NAME:0.3" -T "Progress Log"

# Attach to session
echo "📺 Attaching to tmux session '$SESSION_NAME'..."
echo "Tips:"
echo "  - Ctrl+B then arrow keys to navigate panes"
echo "  - Ctrl+B then D to detach"
echo "  - Run 'tmux attach -t $SESSION_NAME' to reattach"

tmux attach-session -t "$SESSION_NAME"