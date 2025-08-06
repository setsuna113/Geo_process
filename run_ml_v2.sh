#!/bin/bash
# ML Pipeline Launcher V2 - Supports both traditional ML and regression analysis
# Based on regression_analysis_decisions.md

echo "🤖 Machine Learning Pipeline for Biodiversity Analysis V2"
echo "========================================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"
echo "📍 Project root: $PROJECT_ROOT"

# Default values
ANALYSIS_TYPE="ml"  # ml or regression
EXPERIMENT_NAME=""
INPUT_DATA=""
MONITOR_MODE=false
DAEMON_MODE=false
RESUME_MODE=false
PROCESS_NAME=""
CONFIG_PATH=""
OUTPUT_DIR=""
METHOD="all"
DIAGNOSTICS_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --analysis|-a)
            ANALYSIS_TYPE="$2"
            shift 2
            ;;
        --experiment|-e)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --input|-i)
            INPUT_DATA="$2"
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
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --diagnostics-only)
            DIAGNOSTICS_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -a, --analysis TYPE      Analysis type: 'ml' or 'regression' (default: ml)"
            echo "  -e, --experiment NAME    Experiment name from config"
            echo "  -i, --input PATH         Input data file (parquet/csv)"
            echo "  -m, --monitor            Run with tmux monitoring"
            echo "  -d, --daemon             Run as daemon process"
            echo "  -r, --resume             Resume from checkpoint"
            echo "  -p, --process-name NAME  Process name for daemon mode"
            echo "  --config PATH            Configuration file path"
            echo "  --output PATH            Output directory"
            echo "  --method NAME            For regression: all|glm-continuous|glm-categorical|dml"
            echo "  --diagnostics-only       Run only diagnostic checks"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  # Run traditional ML experiment"
            echo "  $0 --analysis ml --experiment production_lgb"
            echo ""
            echo "  # Run regression analysis for hypothesis testing"
            echo "  $0 --analysis regression --input outputs/biodiversity_data.parquet"
            echo ""
            echo "  # Run specific regression method"
            echo "  $0 --analysis regression --input data.parquet --method dml"
            echo ""
            echo "  # Run with monitoring in tmux"
            echo "  $0 --analysis regression --input data.parquet --monitor"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

# Build command based on analysis type
if [ "$ANALYSIS_TYPE" = "regression" ]; then
    echo "📊 Running regression analysis for hypothesis testing"
    
    # Validate inputs for regression
    if [ -z "$INPUT_DATA" ]; then
        echo "❌ Error: --input is required for regression analysis"
        exit 1
    fi
    
    # Build regression command
    ML_CMD="$PYTHON_CMD $PROJECT_ROOT/run_ml_regression.py $INPUT_DATA"
    
    if [ -n "$CONFIG_PATH" ]; then
        ML_CMD="$ML_CMD --config $CONFIG_PATH"
    else
        ML_CMD="$ML_CMD --config config/ml/ml_config.yml"
    fi
    
    if [ -n "$OUTPUT_DIR" ]; then
        ML_CMD="$ML_CMD --output-dir $OUTPUT_DIR"
    else
        ML_CMD="$ML_CMD --output-dir outputs/regression_results"
    fi
    
    if [ -n "$METHOD" ]; then
        ML_CMD="$ML_CMD --method $METHOD"
    fi
    
    if [ "$DIAGNOSTICS_ONLY" = true ]; then
        ML_CMD="$ML_CMD --diagnostics-only"
    fi
    
    # Auto-generate process name
    if [ -z "$PROCESS_NAME" ]; then
        PROCESS_NAME="regression_$(date +%Y%m%d_%H%M%S)"
    fi
    
else
    echo "🚀 Running traditional ML pipeline"
    
    # Validate inputs for ML
    if [ -z "$EXPERIMENT_NAME" ] && [ -z "$INPUT_DATA" ]; then
        echo "❌ Error: Either --experiment or --input must be specified"
        exit 1
    fi
    
    # Build ML command
    ML_CMD="$PYTHON_CMD $PROJECT_ROOT/scripts/run_ml.py"
    
    if [ -n "$EXPERIMENT_NAME" ]; then
        ML_CMD="$ML_CMD --experiment $EXPERIMENT_NAME"
    fi
    
    if [ -n "$INPUT_DATA" ]; then
        ML_CMD="$ML_CMD --input $INPUT_DATA"
    fi
    
    if [ "$RESUME_MODE" = true ]; then
        ML_CMD="$ML_CMD --resume"
    fi
    
    # Auto-generate process name
    if [ -z "$PROCESS_NAME" ]; then
        if [ -n "$EXPERIMENT_NAME" ]; then
            PROCESS_NAME="ml_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
        else
            PROCESS_NAME="ml_custom_$(date +%Y%m%d_%H%M%S)"
        fi
    fi
fi

# Handle different execution modes
if [ "$DAEMON_MODE" = true ]; then
    echo "🌙 Starting in daemon mode..."
    echo "Process name: $PROCESS_NAME"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Run as background process with nohup
    nohup $ML_CMD > "$PROJECT_ROOT/logs/${PROCESS_NAME}.log" 2>&1 &
    PID=$!
    echo "🌙 Pipeline started as daemon (PID: $PID)"
    echo "📝 Logs: $PROJECT_ROOT/logs/${PROCESS_NAME}.log"
    echo "To stop: kill $PID"
    
elif [ "$MONITOR_MODE" = true ]; then
    echo "📊 Starting with tmux monitoring..."
    
    # Create tmux session for monitoring
    SESSION_NAME="${ANALYSIS_TYPE}_${PROCESS_NAME}"
    
    # Check if session already exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️  tmux session '$SESSION_NAME' already exists"
        echo "Attaching to existing session..."
        tmux attach-session -t "$SESSION_NAME"
        exit 0
    fi
    
    # Create new tmux session with monitoring layout
    tmux new-session -d -s "$SESSION_NAME" -n "${ANALYSIS_TYPE^^} Pipeline"
    
    # Main pane: Pipeline execution
    tmux send-keys -t "$SESSION_NAME:0.0" "$ML_CMD" C-m
    
    # Split horizontally for logs
    tmux split-window -h -t "$SESSION_NAME:0"
    LOG_FILE="$PROJECT_ROOT/logs/${PROCESS_NAME}.log"
    if [ "$ANALYSIS_TYPE" = "regression" ]; then
        LOG_FILE="$PROJECT_ROOT/regression_analysis.log"
    fi
    tmux send-keys -t "$SESSION_NAME:0.1" "tail -f $LOG_FILE 2>/dev/null || echo 'Waiting for logs...'" C-m
    
    # Split vertically for monitoring
    tmux split-window -v -t "$SESSION_NAME:0.0"
    if [ "$ANALYSIS_TYPE" = "regression" ]; then
        # For regression, show progress differently
        tmux send-keys -t "$SESSION_NAME:0.2" "watch -n 2 'echo \"=== Regression Analysis Status ===\"; tail -20 $LOG_FILE | grep -E \"(Method|H[012]|COMPLETE)\" || echo \"Analysis in progress...\"'" C-m
    else
        tmux send-keys -t "$SESSION_NAME:0.2" "watch -n 2 'echo \"=== ML Pipeline Status ===\"; $PYTHON_CMD $PROJECT_ROOT/scripts/process_manager.py status $PROCESS_NAME 2>/dev/null || echo \"Pipeline starting...\"'" C-m
    fi
    
    # Split for resource monitoring
    tmux split-window -v -t "$SESSION_NAME:0.1"
    tmux send-keys -t "$SESSION_NAME:0.3" "htop || top" C-m
    
    # Set pane titles
    tmux select-pane -t "$SESSION_NAME:0.0" -T "${ANALYSIS_TYPE^^} Pipeline"
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
    echo "🚀 Starting pipeline..."
    echo "Command: $ML_CMD"
    echo ""
    
    # Run directly
    exec $ML_CMD
fi