#!/bin/bash
# ML Pipeline Launcher - Standalone Machine Learning Pipeline for Biodiversity Analysis
# Supports both simple execution and advanced monitoring with tmux

echo "🤖 Machine Learning Pipeline for Biodiversity Analysis"
echo "====================================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"
echo "📍 Project root: $PROJECT_ROOT"

# Default values
EXPERIMENT_NAME=""
INPUT_PARQUET=""
MONITOR_MODE=false
DAEMON_MODE=false
RESUME_MODE=false
PROCESS_NAME=""
CONFIG_OVERRIDE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment|-e)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --input|-i)
            INPUT_PARQUET="$2"
            shift 2
            ;;
        --monitor|-m)
            MONITOR_MODE=true
            shift
            ;;
        --daemon|-d)
            DAEMON_MODE=true
            shift
            ;;
        --resume|-r)
            RESUME_MODE=true
            shift
            ;;
        --process-name|-p)
            PROCESS_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --experiment NAME    ML experiment name from config.yml"
            echo "  -i, --input PATH         Input parquet file (overrides experiment config)"
            echo "  -m, --monitor            Run with tmux monitoring"
            echo "  -d, --daemon             Run as daemon process"
            echo "  -r, --resume             Resume from checkpoint"
            echo "  -p, --process-name NAME  Process name for daemon mode"
            echo "  --config KEY=VALUE       Override config values"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  # Run named experiment from config.yml"
            echo "  $0 --experiment production_lgb"
            echo ""
            echo "  # Run with specific parquet file"
            echo "  $0 --input outputs/biodiversity_20240730.parquet --experiment test_linear"
            echo ""
            echo "  # Run with monitoring in tmux"
            echo "  $0 --experiment production_lgb --monitor"
            echo ""
            echo "  # Run as daemon"
            echo "  $0 --experiment production_lgb --daemon --process-name ml_prod_run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ -z "$EXPERIMENT_NAME" ] && [ -z "$INPUT_PARQUET" ]; then
    echo "❌ Error: Either --experiment or --input must be specified"
    echo "Use --help for usage information"
    exit 1
fi

# Auto-generate process name if not provided
if [ -z "$PROCESS_NAME" ]; then
    if [ -n "$EXPERIMENT_NAME" ]; then
        PROCESS_NAME="ml_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
    else
        PROCESS_NAME="ml_custom_$(date +%Y%m%d_%H%M%S)"
    fi
fi

# Auto-detect Python environment
detect_python_env() {
    local possible_paths=(
        "$HOME/anaconda3/envs/geo/bin/python"
        "$HOME/miniconda3/envs/geo/bin/python"
        "$HOME/conda/envs/geo/bin/python"
        "/opt/anaconda3/envs/geo/bin/python"
        "/opt/miniconda3/envs/geo/bin/python"
        "$CONDA_PREFIX/bin/python"
    )
    
    for python_path in "${possible_paths[@]}"; do
        if [ -f "$python_path" ]; then
            echo "$python_path"
            return 0
        fi
    done
    
    # Fallback to system python
    echo "python"
}

PYTHON_CMD=$(detect_python_env)
echo "🐍 Using Python: $PYTHON_CMD"

# Build ML pipeline command
ML_CMD="$PYTHON_CMD $PROJECT_ROOT/scripts/run_ml.py"

if [ -n "$EXPERIMENT_NAME" ]; then
    ML_CMD="$ML_CMD --experiment $EXPERIMENT_NAME"
fi

if [ -n "$INPUT_PARQUET" ]; then
    ML_CMD="$ML_CMD --input $INPUT_PARQUET"
fi

if [ "$RESUME_MODE" = true ]; then
    ML_CMD="$ML_CMD --resume"
fi

if [ -n "$CONFIG_OVERRIDE" ]; then
    ML_CMD="$ML_CMD --config-override \"$CONFIG_OVERRIDE\""
fi

# Handle different execution modes
if [ "$DAEMON_MODE" = true ]; then
    echo "🌙 Starting ML pipeline in daemon mode..."
    echo "Process name: $PROCESS_NAME"
    
    # Run as background process with nohup
    nohup $ML_CMD > "$PROJECT_ROOT/logs/${PROCESS_NAME}.log" 2>&1 &
    PID=$!
    echo "🌙 ML pipeline started as daemon (PID: $PID)"
    echo "📝 Logs: $PROJECT_ROOT/logs/${PROCESS_NAME}.log"
    echo "To stop: kill $PID"
    
elif [ "$MONITOR_MODE" = true ]; then
    echo "📊 Starting ML pipeline with tmux monitoring..."
    
    # Create tmux session for monitoring
    SESSION_NAME="ml_${PROCESS_NAME}"
    
    # Check if session already exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️  tmux session '$SESSION_NAME' already exists"
        echo "Attaching to existing session..."
        tmux attach-session -t "$SESSION_NAME"
        exit 0
    fi
    
    # Create new tmux session with monitoring layout
    tmux new-session -d -s "$SESSION_NAME" -n "ML Pipeline"
    
    # Main pane: ML pipeline
    tmux send-keys -t "$SESSION_NAME:0.0" "$ML_CMD" C-m
    
    # Split horizontally for logs
    tmux split-window -h -t "$SESSION_NAME:0"
    tmux send-keys -t "$SESSION_NAME:0.1" "tail -f $PROJECT_ROOT/logs/ml_pipeline.log 2>/dev/null || echo 'Waiting for logs...'" C-m
    
    # Split vertically for monitoring
    tmux split-window -v -t "$SESSION_NAME:0.0"
    tmux send-keys -t "$SESSION_NAME:0.2" "watch -n 2 'echo \"=== ML Pipeline Status ===\"; $PYTHON_CMD $PROJECT_ROOT/scripts/process_manager.py status $PROCESS_NAME 2>/dev/null || echo \"Pipeline starting...\"'" C-m
    
    # Split for resource monitoring
    tmux split-window -v -t "$SESSION_NAME:0.1"
    tmux send-keys -t "$SESSION_NAME:0.3" "htop || top" C-m
    
    # Set pane titles
    tmux select-pane -t "$SESSION_NAME:0.0" -T "ML Pipeline"
    tmux select-pane -t "$SESSION_NAME:0.1" -T "Logs"
    tmux select-pane -t "$SESSION_NAME:0.2" -T "Status"
    tmux select-pane -t "$SESSION_NAME:0.3" -T "Resources"
    
    # Attach to session
    echo "📺 Attaching to tmux session '$SESSION_NAME'..."
    echo "💡 Tips:"
    echo "   - Use Ctrl+B then arrow keys to navigate panes"
    echo "   - Use Ctrl+B then D to detach"
    echo "   - Run 'tmux attach -t $SESSION_NAME' to reattach"
    
    tmux attach-session -t "$SESSION_NAME"
    
else
    echo "🚀 Starting ML pipeline..."
    echo "Command: $ML_CMD"
    echo ""
    
    # Run directly
    exec $ML_CMD
fi