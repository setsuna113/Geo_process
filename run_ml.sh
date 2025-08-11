#!/bin/bash
# ML Regression Pipeline Launcher - Hypothesis Testing for Biodiversity Analysis
# Implements the 3 methods from regression_analysis_decisions.md

echo "🤖 Machine Learning Regression Pipeline for Biodiversity Analysis"
echo "=================================================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

cd "$PROJECT_ROOT"
echo "📍 Project root: $PROJECT_ROOT"

# Default values
INPUT_DATA=""
MONITOR_MODE=false
DAEMON_MODE=false
PROCESS_NAME=""
CONFIG_PATH=""
OUTPUT_DIR=""
METHOD="all"
DIAGNOSTICS_ONLY=false
SKIP_RESOURCE_CHECK=false
LOG_LEVEL="INFO"
SPATIAL_GRID_STEP=1
CLEAR_CACHE=false
SHOW_CACHE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_DATA="$2"
            shift 2
            ;;
        --method|-m)
            METHOD="$2"
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
        --diagnostics-only)
            DIAGNOSTICS_ONLY=true
            shift
            ;;
        --skip-resource-check)
            SKIP_RESOURCE_CHECK=true
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --spatial-grid-step)
            SPATIAL_GRID_STEP="$2"
            shift 2
            ;;
        --clear-cache)
            CLEAR_CACHE=true
            shift
            ;;
        --show-cache)
            SHOW_CACHE=true
            shift
            ;;
        --show-config)
            # Show configuration and exit
            echo "Current configuration:"
            echo "  Config path: ${CONFIG_PATH:-"src/config/ml/ml_config.yml (default)"}"
            python -c "
from src.machine_learning.config_loader import MLRegressionConfig
config = MLRegressionConfig(validate=False)
print('  Preprocessing:', config.get_preprocessing_config().get('missing_data', {}).get('strategy', 'Not found'))
print('  GLM distribution:', config.get_glm_config().get('distribution', 'Not found'))
print('  DML n_folds:', config.get_dml_config().get('n_folds', 'Not found'))
print('  H2O memory:', config.get_compute_config().get('h2o_max_memory', 'Not found'))
"
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 --input INPUT_DATA [OPTIONS]"
            echo ""
            echo "Machine Learning Regression Pipeline - Tests 3 Hypotheses"
            echo ""
            echo "Required:"
            echo "  -i, --input PATH         Input data file (parquet/csv)"
            echo ""
            echo "Options:"
            echo "  -m, --method METHOD      Method to run:"
            echo "                          - all: Run all 3 methods (default)"
            echo "                          - glm-continuous: Method 1 - GLM with continuous interaction (H1)"
            echo "                          - glm-categorical: Method 2 - GLM with categorical interaction (H2)"
            echo "                          - dml: Method 3 - Double Machine Learning (causal inference)"
            echo ""
            echo "Sampling & Cache Options:"
            echo "  --spatial-grid-step N    Spatial grid sampling: take every Nth point (default: 1=no sampling)"
            echo "                          Example: 10 = take every 10th point in x and y coordinates"
            echo "  --clear-cache           Clear MESF and preprocessing cache before running"
            echo "  --show-cache            Show cache information and exit"
            echo ""
            echo "Cache Behavior:"
            echo "  • MESF computations are automatically cached based on:"
            echo "    - Input file (path, size, modification time)"
            echo "    - Spatial grid step value"
            echo "    - MESF parameters (eigenvectors, thresholds)"
            echo "  • Cache is reused when running with same parameters"
            echo "  • Cache persists between runs (stored in /tmp/ml_cache/)"
            echo "  • Cache is NOT auto-cleaned (use --clear-cache when needed)"
            echo ""
            echo "  --monitor                Run with tmux monitoring (4-pane layout)"
            echo "  -d, --daemon             Run as daemon process"
            echo "  -p, --process-name NAME  Process name for daemon mode"
            echo "  --config PATH            Configuration file path"
            echo "  --output PATH            Output directory"
            echo "  --diagnostics-only       Run only diagnostic checks"
            echo "  --skip-resource-check    Skip resource availability check"
            echo "  --log-level LEVEL        Logging level (DEBUG|INFO|WARNING|ERROR)"
            echo "  --show-config            Show current configuration and exit"
            echo "  -h, --help               Show this help"
            echo ""
            echo "The 3 Methods (from regression_analysis_decisions.md):"
            echo ""
            echo "  Method 1 - Continuous GLM (H1: Biodiversity begets biodiversity)"
            echo "    Tests if biodiversity-biodiversity relationship varies"
            echo "    continuously with temperature seasonality (bio4)"
            echo ""
            echo "  Method 2 - Categorical GLM (H2: Temperate mismatch hypothesis)"
            echo "    Tests if biodiversity-biodiversity relationship is stronger"
            echo "    in temperate regions with high plant diversity"
            echo ""
            echo "  Method 3 - Double Machine Learning (Causal inference)"
            echo "    Tests all hypotheses using causal inference with"
            echo "    cross-fitting and bootstrap inference"
            echo ""
            echo "Examples:"
            echo "  # Run all 3 methods"
            echo "  $0 --input outputs/biodiversity_data.parquet"
            echo ""
            echo "  # Run specific method with monitoring"
            echo "  $0 --input data.parquet --method dml --monitor"
            echo ""
            echo "  # Run with spatial grid sampling (every 10th point)"
            echo "  $0 --input large_data.parquet --spatial-grid-step 10"
            echo ""
            echo "  # Run Method 1 only"
            echo "  $0 --input data.parquet --method glm-continuous"
            echo ""
            echo "  # Run with custom config"
            echo "  $0 --input data.parquet --config src/config/ml/ml_config_unified.yml"
            echo ""
            echo "Monitor mode (--monitor):"
            echo "  Creates a tmux session with 4 panes:"
            echo "  - Main: Pipeline execution"
            echo "  - Logs: Real-time log output"
            echo "  - Status: Progress monitoring"
            echo "  - Resources: System resource usage"
            echo ""
            echo "Configuration:"
            echo "  Default: src/config/ml/ml_config.yml"
            echo "  Custom: Use --config PATH to specify alternative config"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Handle cache operations
if [ "$SHOW_CACHE" = true ]; then
    echo "📊 MESF Cache Information"
    echo "========================="
    python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.machine_learning.spatial.mesf_cache_manager import MESFCacheManager

cache_mgr = MESFCacheManager()
info = cache_mgr.get_cache_info()
print(f'Cache directory: {info[\"cache_dir\"]}')
print(f'Cached files: {info[\"num_cached\"]}')
print(f'Total size: {info[\"total_size_mb\"]:.2f} MB')
if info['cache_files']:
    print('\\nCached entries:')
    for f in info['cache_files']:
        print(f'  - {f}')
else:
    print('\\nNo cached files')
print('\\nCache is persistent between runs.')
print('Use --clear-cache to remove all cached MESF computations.')
"
    exit 0
fi

if [ "$CLEAR_CACHE" = true ]; then
    echo "🧹 Clearing MESF cache..."
    python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from src.machine_learning.spatial.mesf_cache_manager import MESFCacheManager

cache_mgr = MESFCacheManager()
info_before = cache_mgr.get_cache_info()
print(f'Found {info_before[\"num_cached\"]} cached files ({info_before[\"total_size_mb\"]:.2f} MB)')

cache_mgr.clear_cache()
print('✅ Cache cleared successfully')

# Also clear standardized files in /scratch if they exist
import os
from pathlib import Path
scratch_dir = Path('/scratch/yl998')
if scratch_dir.exists():
    standardized_files = list(scratch_dir.glob('*_standardized.parquet'))
    if standardized_files:
        print(f'\\nFound {len(standardized_files)} standardized files in /scratch/yl998')
        for f in standardized_files:
            print(f'  Removing: {f.name}')
            f.unlink()
        print('✅ Standardized files cleared')
"
    echo ""
    echo "Cache has been cleared. Next run will recompute MESF."
    
    # If only clearing cache, exit
    if [ -z "$INPUT_DATA" ]; then
        exit 0
    fi
fi

# Validate required inputs
if [ -z "$INPUT_DATA" ]; then
    echo "❌ Error: --input is required"
    echo "Usage: $0 --input INPUT_DATA [OPTIONS]"
    echo "Use --help for more information"
    exit 1
fi

if [ ! -f "$INPUT_DATA" ]; then
    echo "❌ Error: Input file not found: $INPUT_DATA"
    exit 1
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

# Check resource availability unless skipped
if [ "$SKIP_RESOURCE_CHECK" = false ]; then
    echo "🔍 Checking resource availability..."
    
    # Check memory
    available_mem=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$available_mem" -lt 8 ]; then
        echo "⚠️  Warning: Low memory available (${available_mem}GB). Consider using --daemon mode."
    fi
    
    # Check H2O requirements for regression
    h2o_mem_required=8  # GB
    if [ "$available_mem" -lt "$h2o_mem_required" ]; then
        echo "❌ Error: Insufficient memory for H2O (need ${h2o_mem_required}GB, have ${available_mem}GB)"
        echo "   Use --skip-resource-check to override"
        exit 1
    fi
fi

echo "📊 Running regression analysis for hypothesis testing"
echo "   Method: $METHOD"
echo "   Input: $INPUT_DATA"

# Build regression command
ML_CMD="$PYTHON_CMD $PROJECT_ROOT/run_ml_regression.py $INPUT_DATA"

# Add configuration
if [ -n "$CONFIG_PATH" ]; then
    ML_CMD="$ML_CMD --config $CONFIG_PATH"
fi

# Always use legacy config (the only one we have now)
ML_CMD="$ML_CMD --use-legacy-config"

# Add output directory
if [ -n "$OUTPUT_DIR" ]; then
    ML_CMD="$ML_CMD --output-dir $OUTPUT_DIR"
else
    ML_CMD="$ML_CMD --output-dir outputs/regression_results"
fi

# Add method
if [ -n "$METHOD" ]; then
    ML_CMD="$ML_CMD --method $METHOD"
fi

# Add spatial sampling parameter
if [ "$SPATIAL_GRID_STEP" -gt 1 ]; then
    ML_CMD="$ML_CMD --spatial-grid-step $SPATIAL_GRID_STEP"
fi

# Add diagnostics flag
if [ "$DIAGNOSTICS_ONLY" = true ]; then
    ML_CMD="$ML_CMD --diagnostics-only"
fi

# Auto-generate process name
if [ -z "$PROCESS_NAME" ]; then
    PROCESS_NAME="ml_regression_$(date +%Y%m%d_%H%M%S)"
fi

# Handle different execution modes
if [ "$DAEMON_MODE" = true ]; then
    echo "🌙 Starting in daemon mode..."
    echo "Process name: $PROCESS_NAME"
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    LOG_FILE="$PROJECT_ROOT/logs/${PROCESS_NAME}.log"
    
    # Run as background process
    nohup $ML_CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    
    echo "🌙 Pipeline started as daemon (PID: $PID)"
    echo "📝 Logs: $LOG_FILE"
    echo ""
    echo "Commands:"
    echo "  View logs:  tail -f $LOG_FILE"
    echo "  Check status: ps -p $PID"
    echo "  Stop process: kill $PID"
    
elif [ "$MONITOR_MODE" = true ]; then
    echo "📊 Starting with tmux monitoring..."
    
    # Create unique session name
    SESSION_NAME="ml_regression_${PROCESS_NAME}"
    
    # Check if session already exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "⚠️  tmux session '$SESSION_NAME' already exists"
        echo "Attaching to existing session..."
        tmux attach-session -t "$SESSION_NAME"
        exit 0
    fi
    
    # Create new tmux session with 4-pane layout
    tmux new-session -d -s "$SESSION_NAME" -n "ML Regression"
    
    # Pane 0: Main pipeline execution
    tmux send-keys -t "$SESSION_NAME:0.0" "$ML_CMD" C-m
    
    # Split horizontally for logs (Pane 1)
    tmux split-window -h -t "$SESSION_NAME:0"
    LOG_FILE="$PROJECT_ROOT/regression_analysis.log"
    tmux send-keys -t "$SESSION_NAME:0.1" "echo 'Waiting for logs...' && sleep 2 && tail -f $LOG_FILE 2>/dev/null" C-m
    
    # Split vertically for status monitoring (Pane 2)
    tmux split-window -v -t "$SESSION_NAME:0.0"
    # Show hypothesis test progress
    tmux send-keys -t "$SESSION_NAME:0.2" "watch -n 2 'echo \"=== Regression Analysis Progress ===\"; echo \"\"; tail -20 $LOG_FILE 2>/dev/null | grep -E \"(Method|H[012]|significant|COMPLETE|ERROR)\" | tail -10; echo \"\"; echo \"Latest activity:\"; tail -1 $LOG_FILE 2>/dev/null'" C-m
    
    # Split for resource monitoring (Pane 3)
    tmux split-window -v -t "$SESSION_NAME:0.1"
    tmux send-keys -t "$SESSION_NAME:0.3" "htop || top" C-m
    
    # Set pane titles and layout
    tmux select-pane -t "$SESSION_NAME:0.0" -T "Main Execution"
    tmux select-pane -t "$SESSION_NAME:0.1" -T "Logs"
    tmux select-pane -t "$SESSION_NAME:0.2" -T "Progress"
    tmux select-pane -t "$SESSION_NAME:0.3" -T "Resources"
    
    # Balance panes
    tmux select-layout -t "$SESSION_NAME:0" tiled
    
    # Focus on main pane
    tmux select-pane -t "$SESSION_NAME:0.0"
    
    # Attach to session
    echo "📺 Attaching to tmux session '$SESSION_NAME'..."
    echo ""
    echo "💡 Tmux Tips:"
    echo "   • Navigate panes: Ctrl+B then arrow keys"
    echo "   • Detach session: Ctrl+B then D"
    echo "   • Reattach later: tmux attach -t $SESSION_NAME"
    echo "   • Kill session:   tmux kill-session -t $SESSION_NAME"
    echo ""
    
    tmux attach-session -t "$SESSION_NAME"
    
else
    echo "🚀 Starting pipeline in foreground mode..."
    echo "Command: $ML_CMD"
    echo ""
    
    # Run directly
    exec $ML_CMD
fi