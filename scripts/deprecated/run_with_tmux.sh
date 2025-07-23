#!/bin/bash
# Enhanced tmux runner for richness analysis with monitoring

# Don't exit on error - we want to handle errors gracefully
set +e

# Auto-detect Python environment
detect_python_env() {
    # Try common conda environment paths
    local possible_paths=(
        "$HOME/anaconda3/envs/geo_py311/bin/python"
        "$HOME/miniconda3/envs/geo_py311/bin/python" 
        "$HOME/conda/envs/geo_py311/bin/python"
        "/opt/anaconda3/envs/geo_py311/bin/python"
        "/opt/miniconda3/envs/geo_py311/bin/python"
    )
    
    # Try current conda environment if active
    if [ ! -z "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
        echo "$CONDA_PREFIX/bin/python"
        return 0
    fi
    
    # Try common paths
    for path in "${possible_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    
    # Fall back to system python if it has required packages
    if command -v python3 &> /dev/null; then
        if python3 -c "import sys; sys.path.insert(0, 'src'); from src.database.connection import DatabaseManager" 2>/dev/null; then
            echo "python3"
            return 0
        fi
    fi
    
    return 1
}

SCRIPT_NAME="richness_analysis"
PYTHON_ENV=$(detect_python_env)
if [ $? -ne 0 ] || [ -z "$PYTHON_ENV" ]; then
    echo "❌ Error: Python environment with geo dependencies not found"
    echo "   Cannot run analysis without proper Python environment"
    exit 1
else
    echo "✅ Python environment found: $PYTHON_ENV"
fi

LOG_DIR="logs"

echo "🚀 Starting Richness Analysis with tmux monitoring"

# Create logs directory
mkdir -p "$LOG_DIR"

# Install tqdm if not available (for progress bars)
echo "📦 Checking dependencies..."
if ! $PYTHON_ENV -c "import tqdm" 2>/dev/null; then
    echo "📦 Installing tqdm for progress bars..."
    if ! $PYTHON_ENV -m pip install tqdm; then
        echo "⚠️ Failed to install tqdm - progress bars will be basic"
    fi
else
    echo "✅ tqdm already installed"
fi

# Kill existing tmux session if it exists
if tmux has-session -t "$SCRIPT_NAME" 2>/dev/null; then
    echo "🛑 Killing existing tmux session: $SCRIPT_NAME"
    tmux kill-session -t "$SCRIPT_NAME"
fi

# Start new tmux session with multiple panes for monitoring
echo "🎬 Creating tmux session: $SCRIPT_NAME"

# Create tmux session with the main process
tmux new-session -d -s "$SCRIPT_NAME" -c "$(pwd)" \
    "$PYTHON_ENV run_richness_analysis.py --memory-limit 8.0 --max-samples 500000 --sampling-strategy stratified"

# Split window to create monitoring panes
tmux split-window -h -t "$SCRIPT_NAME" -c "$(pwd)"
tmux split-window -v -t "$SCRIPT_NAME:0.1" -c "$(pwd)"

# Setup pane 1 (right-top): System monitoring
tmux send-keys -t "$SCRIPT_NAME:0.1" "
echo '📊 SYSTEM MONITORING'
echo '=================='
while true; do
    clear
    echo '📊 SYSTEM MONITORING - $(date)'
    echo '=================='
    echo
    echo '🖥️  CPU and Memory:'
    top -bn1 | head -5
    echo
    echo '💾 Memory Usage:'
    free -h
    echo
    echo '💿 Disk Usage:'
    df -h /
    echo
    echo '🐍 Python Processes:'
    ps aux | grep -E '(run_richness|python)' | grep -v grep || echo 'No Python processes'
    echo
    echo '⏱️  Last Updated: $(date)'
    sleep 10
done
" Enter

# Setup pane 2 (right-bottom): Log monitoring  
tmux send-keys -t "$SCRIPT_NAME:0.2" "
echo '📋 LOG MONITORING'
echo '================='
# Wait a bit for log file to be created
sleep 2
LOG_FILE=\$(ls -t logs/richness_analysis_*.log 2>/dev/null | head -1)
if [ -n \"\$LOG_FILE\" ]; then
    echo \"📄 Monitoring: \$LOG_FILE\"
    tail -f \"\$LOG_FILE\"
else
    echo '⏳ Waiting for log file to be created...'
    sleep 5
    # Try again
    LOG_FILE=\$(ls -t logs/richness_analysis_*.log 2>/dev/null | head -1)
    if [ -n \"\$LOG_FILE\" ]; then
        tail -f \"\$LOG_FILE\"
    else
        echo '❌ No log file found'
    fi
fi
" Enter

# Setup pane 0 (left): Main process is already running

echo
echo "✅ tmux session '$SCRIPT_NAME' started successfully!"
echo
echo "🎛️  CONTROL COMMANDS:"
echo "   tmux attach -t $SCRIPT_NAME      # Attach to session"
echo "   tmux detach                      # Detach from session" 
echo "   tmux kill-session -t $SCRIPT_NAME # Kill session"
echo
echo "📱 MONITORING:"
echo "   Pane 0 (left): Main processing script"
echo "   Pane 1 (top-right): System monitoring (CPU, Memory, Disk)"
echo "   Pane 2 (bottom-right): Real-time log monitoring"
echo
echo "🎮 TMUX NAVIGATION:"
echo "   Ctrl+B then arrow keys: Switch between panes"
echo "   Ctrl+B then d: Detach from session (keeps running)"
echo "   Ctrl+B then &: Kill current session"
echo "   Ctrl+B then ?: Help"
echo
echo "📁 OUTPUT LOCATIONS:"
echo "   Logs: ./logs/richness_analysis_*.log"
echo "   Results: ./outputs/spatial_analysis/"
echo "   Checkpoints: ./checkpoint_*.json"
echo
echo "🔄 RESUMING AFTER INTERRUPTION:"
echo "   Just run this script again - it will resume from checkpoints!"
echo
echo "🚀 Attaching to tmux session now..."
echo "   (Press Ctrl+B then d to detach and let it run in background)"
sleep 2

# Attach to the session
exec tmux attach -t "$SCRIPT_NAME"