#!/bin/bash
# scripts/production/run_unified_resampling.sh
# Production launcher with process control support

echo "🚀 Unified Resampling Pipeline - Production Launcher"
echo "=================================================="

# Get script directory and find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
echo "📍 Project root: $PROJECT_ROOT"

# Default values
DAEMON_MODE=false
PROCESS_NAME=""
RESUME_MODE=false
EXPERIMENT_NAME=""
SIGNAL_FORWARD=""
ANALYSIS_METHOD="som"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --daemon)
            DAEMON_MODE=true
            shift
            ;;
        --process-name)
            PROCESS_NAME="$2"
            shift 2
            ;;
        --resume)
            RESUME_MODE=true
            shift
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --analysis-method)
            ANALYSIS_METHOD="$2"
            shift 2
            ;;
        --signal)
            SIGNAL_FORWARD="$2"
            shift 2
            ;;
        *)
            # Pass through other arguments
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# Auto-generate process name if not provided
if [ -z "$PROCESS_NAME" ]; then
    PROCESS_NAME="unified_resampling_$(date +%Y%m%d_%H%M%S)"
fi

# Check if process manager exists
PROCESS_MANAGER="$PROJECT_ROOT/scripts/process_manager.py"
if [ ! -f "$PROCESS_MANAGER" ]; then
    echo "❌ Error: Process manager not found at $PROCESS_MANAGER"
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
echo "📝 Process name: $PROCESS_NAME"

# Handle signal forwarding
if [ ! -z "$SIGNAL_FORWARD" ]; then
    case $SIGNAL_FORWARD in
        pause)
            echo "⏸️  Pausing process: $PROCESS_NAME"
            exec $PYTHON_ENV "$PROCESS_MANAGER" pause "$PROCESS_NAME"
            ;;
        resume)
            echo "▶️  Resuming process: $PROCESS_NAME"
            exec $PYTHON_ENV "$PROCESS_MANAGER" resume "$PROCESS_NAME"
            ;;
        stop)
            echo "⏹️  Stopping process: $PROCESS_NAME"
            exec $PYTHON_ENV "$PROCESS_MANAGER" stop "$PROCESS_NAME"
            ;;
        status)
            echo "📊 Process status: $PROCESS_NAME"
            exec $PYTHON_ENV "$PROCESS_MANAGER" status "$PROCESS_NAME"
            ;;
        *)
            echo "❌ Unknown signal: $SIGNAL_FORWARD"
            exit 1
            ;;
    esac
fi

# Build start command
START_CMD=("$PYTHON_ENV" "$PROCESS_MANAGER" "start" "--name" "$PROCESS_NAME")

if [ "$DAEMON_MODE" = true ]; then
    START_CMD+=("--daemon")
    echo "🌙 Running in daemon mode"
fi

if [ "$RESUME_MODE" = true ]; then
    START_CMD+=("--resume")
    echo "♻️  Resume mode enabled"
fi

if [ ! -z "$EXPERIMENT_NAME" ]; then
    START_CMD+=("--experiment-name" "$EXPERIMENT_NAME")
fi

# Always add analysis method
START_CMD+=("--analysis-method" "$ANALYSIS_METHOD")

# Add pass-through arguments
if [ ${#PASS_THROUGH_ARGS[@]} -gt 0 ]; then
    # The first pass-through arg is typically the experiment name
    if [ -z "$EXPERIMENT_NAME" ] && [ ${#PASS_THROUGH_ARGS[@]} -gt 0 ]; then
        EXPERIMENT_NAME="${PASS_THROUGH_ARGS[0]}"
        START_CMD+=("--experiment-name" "$EXPERIMENT_NAME")
    fi
fi

# Show what we're executing
echo ""
echo "🎬 Starting pipeline with process controller..."
echo "Command: ${START_CMD[@]}"
echo ""

# Function to handle signals
handle_signal() {
    local sig=$1
    echo ""
    echo "Received signal: $sig"
    
    case $sig in
        INT|TERM)
            echo "Checking if process is registered..."
            # Check if process exists before trying to stop it
            if $PYTHON_ENV "$PROCESS_MANAGER" status "$PROCESS_NAME" 2>/dev/null | grep -q "$PROCESS_NAME"; then
                echo "Forwarding shutdown signal to process..."
                $PYTHON_ENV "$PROCESS_MANAGER" stop "$PROCESS_NAME" --timeout 30
            else
                echo "Process not yet registered, killing directly..."
                if [ ! -z "$PIPELINE_PID" ]; then
                    kill -TERM $PIPELINE_PID 2>/dev/null || true
                fi
            fi
            exit 0
            ;;
        USR1)
            echo "Forwarding pause signal to process..."
            $PYTHON_ENV "$PROCESS_MANAGER" pause "$PROCESS_NAME"
            ;;
        USR2)
            echo "Forwarding resume signal to process..."
            $PYTHON_ENV "$PROCESS_MANAGER" resume "$PROCESS_NAME"
            ;;
    esac
}

# Set up signal handlers
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap 'handle_signal USR1' USR1
trap 'handle_signal USR2' USR2

# Execute the pipeline
if [ "$DAEMON_MODE" = true ]; then
    # Start and detach
    "${START_CMD[@]}"
    echo ""
    echo "✅ Pipeline started in daemon mode"
    echo ""
    echo "📋 Useful commands:"
    echo "  Status:  $0 --process-name $PROCESS_NAME --signal status"
    echo "  Logs:    $PYTHON_ENV $PROCESS_MANAGER logs $PROCESS_NAME -f"
    echo "  Pause:   $0 --process-name $PROCESS_NAME --signal pause"
    echo "  Resume:  $0 --process-name $PROCESS_NAME --signal resume"
    echo "  Stop:    $0 --process-name $PROCESS_NAME --signal stop"
else
    # Run in foreground
    "${START_CMD[@]}" &
    PIPELINE_PID=$!
    
    echo "Pipeline PID: $PIPELINE_PID"
    echo "Press Ctrl+C to stop gracefully"
    
    # Wait for pipeline to complete or be interrupted
    wait $PIPELINE_PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Pipeline completed successfully"
    else
        echo "❌ Pipeline exited with code: $EXIT_CODE"
    fi
    
    exit $EXIT_CODE
fi