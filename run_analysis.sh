#!/bin/bash
# Analysis Pipeline Launcher - Standalone Spatial Analysis Pipeline for Biodiversity Data
# Supports SOM, GWPCA, and MaxP regionalization with monitoring and daemon capabilities

echo "🧠 Spatial Analysis Pipeline for Biodiversity Data"
echo "================================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"
echo "📍 Project root: $PROJECT_ROOT"

# Default values
EXPERIMENT_NAME=""
INPUT_PARQUET=""
ANALYSIS_METHOD=""
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
        --method|-m)
            ANALYSIS_METHOD="$2"
            shift 2
            ;;
        --monitor)
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
            echo "  -e, --experiment NAME    Analysis experiment name from config.yml"
            echo "  -i, --input PATH         Input parquet file (overrides experiment config)"
            echo "  -m, --method METHOD      Analysis method: som|gwpca|maxp_regions"
            echo "  --monitor                Run with tmux monitoring"
            echo "  -d, --daemon             Run as daemon process"
            echo "  -r, --resume             Resume from checkpoint"
            echo "  -p, --process-name NAME  Process name for daemon mode"
            echo "  --config KEY=VALUE       Override config values"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  # Run named experiment from config.yml"
            echo "  $0 --experiment test_som"
            echo ""
            echo "  # Run with specific parquet file and method"
            echo "  $0 --input outputs/biodiversity_20240730.parquet --method som --experiment test_som"
            echo ""
            echo "  # Run production GWPCA with monitoring"
            echo "  $0 --experiment production_gwpca --monitor"
            echo ""
            echo "  # Run as daemon"
            echo "  $0 --experiment production_som --daemon --process-name som_prod_run"
            echo ""
            echo "Available methods:"
            echo "  som          - Self-Organizing Maps for clustering biodiversity patterns"
            echo "  gwpca        - Geographically Weighted Principal Component Analysis"
            echo "  maxp_regions - MaxP regionalization for spatial clustering"
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
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "❌ Error: --experiment must be specified"
    echo "Use --help for usage information"
    exit 1
fi

# Auto-generate process name if not provided
if [ -z "$PROCESS_NAME" ]; then
    if [ -n "$ANALYSIS_METHOD" ]; then
        PROCESS_NAME="analysis_${ANALYSIS_METHOD}_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
    else
        PROCESS_NAME="analysis_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
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

# Build analysis pipeline command
ANALYSIS_CMD="$PYTHON_CMD $PROJECT_ROOT/scripts/run_analysis.py"

# Add required experiment parameter
ANALYSIS_CMD="$ANALYSIS_CMD --experiment $EXPERIMENT_NAME"

# Add optional parameters
if [ -n "$INPUT_PARQUET" ]; then
    ANALYSIS_CMD="$ANALYSIS_CMD --input $INPUT_PARQUET"
fi

if [ -n "$ANALYSIS_METHOD" ]; then
    ANALYSIS_CMD="$ANALYSIS_CMD --method $ANALYSIS_METHOD"
fi

if [ "$RESUME_MODE" = true ]; then
    ANALYSIS_CMD="$ANALYSIS_CMD --resume"
fi

if [ -n "$CONFIG_OVERRIDE" ]; then
    ANALYSIS_CMD="$ANALYSIS_CMD --config-override \"$CONFIG_OVERRIDE\""
fi

# Handle different execution modes
if [ "$DAEMON_MODE" = true ]; then
    echo "🌙 Starting analysis pipeline in daemon mode..."
    echo "Process name: $PROCESS_NAME"
    echo "Experiment: $EXPERIMENT_NAME"
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Run as background process with nohup
    nohup $ANALYSIS_CMD > "$PROJECT_ROOT/logs/${PROCESS_NAME}.log" 2>&1 &
    PID=$!
    echo "🌙 Analysis pipeline started as daemon (PID: $PID)"
    echo "📝 Logs: $PROJECT_ROOT/logs/${PROCESS_NAME}.log"
    echo "To monitor: tail -f $PROJECT_ROOT/logs/${PROCESS_NAME}.log"
    echo "To stop: kill $PID"
    
elif [ "$MONITOR_MODE" = true ]; then
    echo "📊 Starting analysis pipeline with tmux monitoring..."
    
    # Create tmux session for monitoring
    SESSION_NAME="analysis_${PROCESS_NAME}"
    
    # Check if session already exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️  tmux session '$SESSION_NAME' already exists"
        echo "Attaching to existing session..."
        tmux attach-session -t "$SESSION_NAME"
        exit 0
    fi
    
    # Create new tmux session with monitoring layout
    tmux new-session -d -s "$SESSION_NAME" -n "Analysis Pipeline"
    
    # Main pane: Analysis pipeline
    tmux send-keys -t "$SESSION_NAME:0.0" "$ANALYSIS_CMD" C-m
    
    # Split horizontally for logs
    tmux split-window -h -t "$SESSION_NAME:0"
    tmux send-keys -t "$SESSION_NAME:0.1" "tail -f $PROJECT_ROOT/logs/analysis_*.log 2>/dev/null || echo 'Waiting for logs...'" C-m
    
    # Split vertically for status monitoring
    tmux split-window -v -t "$SESSION_NAME:0.0"
    tmux send-keys -t "$SESSION_NAME:0.2" "watch -n 2 'echo \"=== Analysis Pipeline Status ===\"; $PYTHON_CMD $PROJECT_ROOT/scripts/process_manager.py status $PROCESS_NAME 2>/dev/null || echo \"Pipeline starting...\"'" C-m
    
    # Split for resource monitoring
    tmux split-window -v -t "$SESSION_NAME:0.1"
    tmux send-keys -t "$SESSION_NAME:0.3" "htop || top" C-m
    
    # Set pane titles and adjust layout
    tmux select-pane -t "$SESSION_NAME:0.0" -T "Analysis Pipeline"
    tmux select-pane -t "$SESSION_NAME:0.1" -T "Logs"
    tmux select-pane -t "$SESSION_NAME:0.2" -T "Status"
    tmux select-pane -t "$SESSION_NAME:0.3" -T "Resources"
    
    # Focus on main pane
    tmux select-pane -t "$SESSION_NAME:0.0"
    
    # Attach to session
    echo "📺 Attaching to tmux session '$SESSION_NAME'..."
    echo "💡 Tips:"
    echo "   - Use Ctrl+B then arrow keys to navigate panes"
    echo "   - Use Ctrl+B then D to detach"
    echo "   - Run 'tmux attach -t $SESSION_NAME' to reattach"
    echo "   - Use Ctrl+C in main pane to stop analysis"
    
    tmux attach-session -t "$SESSION_NAME"
    
else
    echo "🚀 Starting analysis pipeline..."
    echo "Experiment: $EXPERIMENT_NAME"
    if [ -n "$ANALYSIS_METHOD" ]; then
        echo "Method: $ANALYSIS_METHOD"
    fi
    if [ -n "$INPUT_PARQUET" ]; then
        echo "Input: $INPUT_PARQUET"
    fi
    echo "Command: $ANALYSIS_CMD"
    echo ""
    
    # Run directly
    exec $ANALYSIS_CMD
fi