#!/bin/bash
# Machine Learning Pipeline Launcher with Method Selection and Monitoring
# Supports all three hypothesis testing methods with tmux monitoring

echo "🤖 Machine Learning Regression Pipeline (H2O-Optimized)"
echo "=================================================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"
echo "📍 Project root: $PROJECT_ROOT"

# Default values
DATA_PATH="/home/jason/geo/biodiversity_climate_production_1x_corrected.parquet"
METHOD="all"
SPATIAL_GRID_STEP=""
OUTPUT_DIR=""
CONFIG_PATH=""
LOG_LEVEL="INFO"
MONITOR_MODE=false
DAEMON_MODE=false
PROCESS_NAME=""
SKIP_H2O_CHECK=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data|-d)
            DATA_PATH="$2"
            shift 2
            ;;
        --method|-m)
            METHOD="$2"
            shift 2
            ;;
        --spatial-grid-step|-s)
            SPATIAL_GRID_STEP="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --log-level|-l)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --monitor)
            MONITOR_MODE=true
            shift
            ;;
        --daemon)
            DAEMON_MODE=true
            shift
            ;;
        --process-name|-p)
            PROCESS_NAME="$2"
            shift 2
            ;;
        --skip-h2o-check)
            SKIP_H2O_CHECK=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --data DATA_PATH [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  -d, --data PATH          Input parquet/csv file with biodiversity data"
            echo ""
            echo "Options:"
            echo "  -m, --method METHOD      Method to run (default: all)"
            echo "                           Options: all, method1, method2, method3,"
            echo "                                   glm-continuous, glm-categorical, dml"
            echo "  -s, --spatial-grid-step N  Spatial sampling step (e.g., 10 = every 10th point)"
            echo "  -o, --output DIR         Output directory for results"
            echo "  -c, --config PATH        Path to configuration file"
            echo "  -l, --log-level LEVEL    Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
            echo "  --monitor                Run with tmux monitoring (4-pane layout)"
            echo "  --daemon                 Run as background daemon process"
            echo "  -p, --process-name NAME  Process name for monitoring/daemon mode"
            echo "  --skip-h2o-check         Skip H2O availability check"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Methods:"
            echo "  all             - Run all three methods (complete analysis)"
            echo "  method1         - H0-H1 Testing: Continuous GLM (biodiversity begets biodiversity)"
            echo "  method2         - H2 Testing: Categorical GLM (temperate mismatch hypothesis)"
            echo "  method3         - H0-H1 Causal: Double Machine Learning (causal inference)"
            echo "  glm-continuous  - Alias for method1"
            echo "  glm-categorical - Alias for method2"
            echo "  dml             - Alias for method3"
            echo ""
            echo "Examples:"
            echo "  # Run complete analysis"
            echo "  $0 --data biodiversity.parquet"
            echo ""
            echo "  # Run specific method with output"
            echo "  $0 --data biodiversity.parquet --method method2 --output results/"
            echo ""
            echo "  # Run with spatial sampling and monitoring"
            echo "  $0 --data biodiversity.parquet --spatial-grid-step 10 --monitor"
            echo ""
            echo "  # Run as daemon for long-running analysis"
            echo "  $0 --data biodiversity.parquet --method all --daemon --process-name ml_full_analysis"
            echo ""
            echo "Monitor mode (--monitor):"
            echo "  Creates a 4-pane tmux layout:"
            echo "  - Top-left:     Main ML pipeline output"
            echo "  - Top-right:    Live logs (tail -f)"
            echo "  - Bottom-left:  H2O cluster status"
            echo "  - Bottom-right: Method progress & results"
            echo ""
            echo "  Navigation: Ctrl+B then arrow keys to switch panes"
            echo "  Detach:     Ctrl+B then D"
            echo "  Reattach:   tmux attach -t ml_<process_name>"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required inputs
if [ -z "$DATA_PATH" ]; then
    echo "❌ Error: --data must be specified"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Auto-generate process name if not provided
if [ -z "$PROCESS_NAME" ]; then
    PROCESS_NAME="ml_${METHOD}_$(date +%Y%m%d_%H%M%S)"
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

# Check H2O availability
if [ "$SKIP_H2O_CHECK" = false ]; then
    echo ""
    echo "🔍 Checking H2O availability..."
    
    H2O_CHECK=$($PYTHON_CMD -c "
import sys
try:
    import h2o
    print('✅ H2O is available')
    sys.exit(0)
except ImportError:
    print('❌ H2O is not installed')
    print('Install with: pip install h2o')
    sys.exit(1)
" 2>&1)
    
    H2O_CHECK_EXIT=$?
    echo "$H2O_CHECK"
    
    if [ $H2O_CHECK_EXIT -ne 0 ]; then
        echo ""
        echo "⚠️  H2O is required for ML pipeline"
        echo "Please install H2O or use --skip-h2o-check to proceed anyway"
        exit 1
    fi
    echo ""
fi

# Build ML pipeline command
ML_CMD="$PYTHON_CMD $PROJECT_ROOT/run_ml_pipeline.py"

# Add required data parameter
ML_CMD="$ML_CMD $DATA_PATH"

# Add optional parameters
ML_CMD="$ML_CMD --method $METHOD"

if [ -n "$SPATIAL_GRID_STEP" ]; then
    ML_CMD="$ML_CMD --spatial-grid-step $SPATIAL_GRID_STEP"
fi

if [ -n "$OUTPUT_DIR" ]; then
    ML_CMD="$ML_CMD --output-dir $OUTPUT_DIR"
fi

if [ -n "$CONFIG_PATH" ]; then
    ML_CMD="$ML_CMD --config $CONFIG_PATH"
fi

ML_CMD="$ML_CMD --log-level $LOG_LEVEL"

# Handle different execution modes
if [ "$DAEMON_MODE" = true ]; then
    echo "🌙 Starting ML pipeline in daemon mode..."
    echo "Process name: $PROCESS_NAME"
    echo "Method: $METHOD"
    echo "Data: $DATA_PATH"
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # Validate process name for security
    if [[ ! "$PROCESS_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "❌ Error: Invalid process name contains unsafe characters"
        exit 1
    fi
    
    # Run as background process with nohup
    LOG_FILE="$PROJECT_ROOT/logs/${PROCESS_NAME}.log"
    nohup $ML_CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    
    echo "🌙 ML pipeline started as daemon (PID: $PID)"
    echo "📝 Logs: $LOG_FILE"
    echo "To monitor: tail -f $LOG_FILE"
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
    LOG_FILE="$PROJECT_ROOT/logs/ml_pipeline_*.log"
    tmux send-keys -t "$SESSION_NAME:0.1" "tail -f $LOG_FILE 2>/dev/null || echo 'Waiting for logs...'; while true; do sleep 1; if ls $LOG_FILE 2>/dev/null; then tail -f $LOG_FILE; break; fi; done" C-m
    
    # Split vertically for H2O status
    tmux split-window -v -t "$SESSION_NAME:0.0"
    tmux send-keys -t "$SESSION_NAME:0.2" "watch -n 5 '$PYTHON_CMD -c \"
import h2o
try:
    h2o.init()
    cluster_info = h2o.cluster()
    print(\\\"=== H2O Cluster Status ===\")
    print(f\\\"Status: {cluster_info.cloud_healthy}\\\")
    print(f\\\"Nodes: {len(cluster_info.nodes)}\\\")
    print(f\\\"Memory: {cluster_info.free_mem // (1024**3)}GB free\\\")
    print(f\\\"CPUs: {cluster_info.cores} cores\\\")
except:
    print(\\\"H2O cluster not running\\\")
\"'" C-m
    
    # Split for method progress monitoring
    tmux split-window -v -t "$SESSION_NAME:0.1"
    
    # Method-specific monitoring
    if [ "$METHOD" = "method1" ] || [ "$METHOD" = "glm-continuous" ]; then
        MONITOR_MSG="Method 1: H0-H1 Testing (Continuous GLM)"
    elif [ "$METHOD" = "method2" ] || [ "$METHOD" = "glm-categorical" ]; then
        MONITOR_MSG="Method 2: H2 Testing (Categorical GLM)"
    elif [ "$METHOD" = "method3" ] || [ "$METHOD" = "dml" ]; then
        MONITOR_MSG="Method 3: H0-H1 Causal (DML)"
    else
        MONITOR_MSG="All Methods Running"
    fi
    
    tmux send-keys -t "$SESSION_NAME:0.3" "watch -n 2 'echo \"=== $MONITOR_MSG ===\"; echo; if [ -d \"$OUTPUT_DIR\" ]; then echo \"Output files:\"; ls -lht \"$OUTPUT_DIR\" 2>/dev/null | head -10; fi'" C-m
    
    # Set pane titles and adjust layout
    tmux select-pane -t "$SESSION_NAME:0.0" -T "ML Pipeline"
    tmux select-pane -t "$SESSION_NAME:0.1" -T "Logs"
    tmux select-pane -t "$SESSION_NAME:0.2" -T "H2O Status"
    tmux select-pane -t "$SESSION_NAME:0.3" -T "Progress"
    
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
    echo "🚀 Starting ML pipeline..."
    echo "Method: $METHOD"
    echo "Data: $DATA_PATH"
    if [ -n "$OUTPUT_DIR" ]; then
        echo "Output: $OUTPUT_DIR"
    fi
    echo "Command: $ML_CMD"
    echo ""
    
    # Run directly
    exec $ML_CMD
fi