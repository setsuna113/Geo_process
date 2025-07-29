#!/bin/bash
# scripts/production/run_monitored_pipeline.sh
# Enhanced pipeline launcher with integrated monitoring in tmux

echo "🚀 Monitored Pipeline Launcher with tmux"
echo "======================================"

# Get script directory and find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
echo "📍 Project root: $PROJECT_ROOT"

# Default values
EXPERIMENT_NAME=""
ANALYSIS_METHOD="som"
RESUME_MODE=false
SESSION_NAME=""
DAEMON_MODE=false
PROCESS_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment-name|-e)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --analysis-method|-a)
            ANALYSIS_METHOD="$2"
            shift 2
            ;;
        --resume|-r)
            RESUME_MODE=true
            shift
            ;;
        --session-name|-s)
            SESSION_NAME="$2"
            shift 2
            ;;
        --daemon|-d)
            DAEMON_MODE=true
            shift
            ;;
        --process-name|-p)
            PROCESS_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --experiment-name NAME   Experiment name (required)"
            echo "  -a, --analysis-method METHOD Analysis method (default: som)"
            echo "  -r, --resume                 Resume from checkpoint"
            echo "  -s, --session-name NAME      tmux session name"
            echo "  -d, --daemon                 Run in daemon mode"
            echo "  -p, --process-name NAME      Process name for daemon"
            echo "  -h, --help                   Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "❌ Error: Experiment name is required"
    echo "Usage: $0 --experiment-name <name>"
    exit 1
fi

# Auto-generate session/process name if not provided
if [ -z "$SESSION_NAME" ]; then
    SESSION_NAME="geo_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

if [ -z "$PROCESS_NAME" ]; then
    PROCESS_NAME="$SESSION_NAME"
fi

# Auto-detect Python environment
detect_python_env() {
    local possible_paths=(
        "$HOME/anaconda3/envs/geo/bin/python"
        "$HOME/miniconda3/envs/geo/bin/python"
        "$HOME/conda/envs/geo/bin/python"
        "/opt/anaconda3/envs/geo/bin/python"
        "/opt/miniconda3/envs/geo/bin/python"
        "$HOME/anaconda3/envs/geo_py311/bin/python"
        "$HOME/miniconda3/envs/geo_py311/bin/python"
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
    
    # Fall back to system python
    if command -v python3 &> /dev/null; then
        echo "python3"
        return 0
    fi
    
    return 1
}

PYTHON_ENV=$(detect_python_env)
if [ $? -ne 0 ] || [ -z "$PYTHON_ENV" ]; then
    echo "❌ Error: Python environment not found"
    exit 1
fi

echo "✅ Python environment: $PYTHON_ENV"
echo "📝 Experiment: $EXPERIMENT_NAME"
echo "🔧 Analysis method: $ANALYSIS_METHOD"
echo "🖥️  Session name: $SESSION_NAME"

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "❌ Error: tmux is not installed"
    echo "Install with: sudo apt-get install tmux (Ubuntu/Debian) or equivalent"
    exit 1
fi

# Function to setup tmux session with monitoring
setup_tmux_session() {
    # Kill existing session if it exists
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    
    # Create new session with main window
    tmux new-session -d -s "$SESSION_NAME" -n "pipeline"
    
    # Window 1: Main pipeline execution
    tmux send-keys -t "$SESSION_NAME:pipeline" "cd $PROJECT_ROOT" C-m
    
    # Build pipeline command
    PIPELINE_CMD="$PYTHON_ENV scripts/process_manager.py start"
    PIPELINE_CMD="$PIPELINE_CMD --name $PROCESS_NAME"
    PIPELINE_CMD="$PIPELINE_CMD --experiment-name $EXPERIMENT_NAME"
    PIPELINE_CMD="$PIPELINE_CMD --analysis-method $ANALYSIS_METHOD"
    
    if [ "$RESUME_MODE" = true ]; then
        PIPELINE_CMD="$PIPELINE_CMD --resume"
    fi
    
    if [ "$DAEMON_MODE" = true ]; then
        PIPELINE_CMD="$PIPELINE_CMD --daemon"
    fi
    
    # Start pipeline
    tmux send-keys -t "$SESSION_NAME:pipeline" "$PIPELINE_CMD" C-m
    
    # Give pipeline time to start
    sleep 2
    
    # Window 2: Monitoring
    tmux new-window -t "$SESSION_NAME:1" -n "monitoring"
    
    # Split monitoring window into panes
    # Pane 0: Live monitoring
    tmux send-keys -t "$SESSION_NAME:monitoring.0" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$SESSION_NAME:monitoring.0" "$PYTHON_ENV scripts/monitor.py watch $EXPERIMENT_NAME" C-m
    
    # Pane 1: Resource metrics
    tmux split-window -t "$SESSION_NAME:monitoring" -h
    tmux send-keys -t "$SESSION_NAME:monitoring.1" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$SESSION_NAME:monitoring.1" "watch -n 5 '$PYTHON_ENV scripts/monitor.py metrics $EXPERIMENT_NAME --type memory'" C-m
    
    # Pane 2: Error monitoring
    tmux split-window -t "$SESSION_NAME:monitoring.0" -v
    tmux send-keys -t "$SESSION_NAME:monitoring.2" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$SESSION_NAME:monitoring.2" "watch -n 10 '$PYTHON_ENV scripts/monitor.py errors $EXPERIMENT_NAME'" C-m
    
    # Window 3: Logs
    tmux new-window -t "$SESSION_NAME:2" -n "logs"
    
    # Split logs window
    # Pane 0: All logs
    tmux send-keys -t "$SESSION_NAME:logs.0" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$SESSION_NAME:logs.0" "$PYTHON_ENV scripts/monitor.py logs $EXPERIMENT_NAME --limit 50" C-m
    
    # Pane 1: Error logs only
    tmux split-window -t "$SESSION_NAME:logs" -h
    tmux send-keys -t "$SESSION_NAME:logs.1" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$SESSION_NAME:logs.1" "$PYTHON_ENV scripts/monitor.py logs $EXPERIMENT_NAME --level ERROR --traceback" C-m
    
    # Window 4: System resources
    tmux new-window -t "$SESSION_NAME:3" -n "system"
    
    # System monitoring
    tmux send-keys -t "$SESSION_NAME:system" "htop || top" C-m
    
    # Split for disk usage
    tmux split-window -t "$SESSION_NAME:system" -v
    tmux send-keys -t "$SESSION_NAME:system.1" "cd $PROJECT_ROOT && watch -n 30 'df -h . && echo && du -sh checkpoint_outputs/ output/ logs/ 2>/dev/null | sort -h'" C-m
    
    # Window 5: Control panel
    tmux new-window -t "$SESSION_NAME:4" -n "control"
    tmux send-keys -t "$SESSION_NAME:control" "cd $PROJECT_ROOT" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '🎮 Control Panel'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '==============='" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo ''" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo 'Pipeline Control Commands:'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Pause:   $PYTHON_ENV scripts/process_manager.py pause $PROCESS_NAME'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Resume:  $PYTHON_ENV scripts/process_manager.py resume $PROCESS_NAME'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Stop:    $PYTHON_ENV scripts/process_manager.py stop $PROCESS_NAME'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Status:  $PYTHON_ENV scripts/process_manager.py status $PROCESS_NAME'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo ''" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo 'Monitoring Commands:'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Status:  $PYTHON_ENV scripts/monitor.py status $EXPERIMENT_NAME'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Logs:    $PYTHON_ENV scripts/monitor.py logs $EXPERIMENT_NAME'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Metrics: $PYTHON_ENV scripts/monitor.py metrics $EXPERIMENT_NAME'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo ''" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo 'Session Management:'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Attach:  tmux attach -t $SESSION_NAME'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Detach:  Ctrl+b d'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo '  Switch:  Ctrl+b [0-4]'" C-m
    tmux send-keys -t "$SESSION_NAME:control" "echo ''" C-m
    
    # Select first window
    tmux select-window -t "$SESSION_NAME:0"
}

# Setup tmux session
echo ""
echo "🔧 Setting up tmux session..."
setup_tmux_session

# Show status
echo ""
echo "✅ tmux session '$SESSION_NAME' created with integrated monitoring"
echo ""
echo "📺 Windows:"
echo "  0: pipeline   - Main pipeline execution"
echo "  1: monitoring - Live progress and metrics"
echo "  2: logs       - Log viewing"
echo "  3: system     - System resource monitoring"
echo "  4: control    - Control panel with commands"
echo ""

if [ "$DAEMON_MODE" = true ]; then
    echo "🌙 Pipeline started in daemon mode"
    echo ""
    echo "📋 Useful commands:"
    echo "  Attach to session:  tmux attach -t $SESSION_NAME"
    echo "  Check status:       $PYTHON_ENV scripts/monitor.py status $EXPERIMENT_NAME"
    echo "  View logs:          $PYTHON_ENV scripts/monitor.py logs $EXPERIMENT_NAME -f"
    echo "  Stop pipeline:      $PYTHON_ENV scripts/process_manager.py stop $PROCESS_NAME"
else
    echo "🎯 Attaching to tmux session..."
    echo "   (Use Ctrl+b d to detach, Ctrl+b [0-4] to switch windows)"
    echo ""
    sleep 1
    tmux attach -t "$SESSION_NAME"
fi