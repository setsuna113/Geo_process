#!/bin/bash
# Consolidated SOM analysis runner with monitoring
# This replaces all the redundant SOM scripts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="run"  # run, monitor, or both
CONFIG_TYPE="optimized"  # optimized or default
DETACHED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --monitor)
            MODE="monitor"
            shift
            ;;
        --both)
            MODE="both"
            shift
            ;;
        --default-config)
            CONFIG_TYPE="default"
            shift
            ;;
        --detached|-d)
            DETACHED=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --monitor        Only monitor existing SOM run"
            echo "  --both           Run analysis and monitor in tmux"
            echo "  --default-config Use default SOM config instead of optimized"
            echo "  --detached, -d   Run in detached tmux session"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Project settings
PROJECT_DIR="/home/yl998/dev/geo"
cd "$PROJECT_DIR"

# Set configuration
if [ "$CONFIG_TYPE" = "optimized" ]; then
    export SOM_CONFIG_PATH="$PROJECT_DIR/src/config/som/som_config_optimized.yml"
    CONFIG_DESC="50x50 grid, spatial_weight=0.4, equal feature weights"
else
    unset SOM_CONFIG_PATH
    CONFIG_DESC="Default configuration"
fi

# Function to find parquet file
find_parquet_file() {
    # First try the specific file we know exists
    PARQUET_FILE="outputs/45d78409-3818-4eb6-960f-c00216110460/merged_data_45d78409-3818-4eb6-960f-c00216110460_20250801_001157.parquet"
    
    if [ ! -f "$PARQUET_FILE" ]; then
        # Try alternative
        PARQUET_FILE="outputs/biodiversity_data.parquet"
        if [ ! -f "$PARQUET_FILE" ]; then
            # Find any parquet file
            PARQUET_FILE=$(find outputs -name "*.parquet" -type f | grep -v checkpoint | head -1)
            if [ -z "$PARQUET_FILE" ]; then
                echo -e "${RED}ERROR: No parquet file found!${NC}"
                exit 1
            fi
        fi
    fi
    echo "$PARQUET_FILE"
}

# Function to run analysis
run_analysis() {
    echo -e "${GREEN}=== Starting SOM Analysis ===${NC}"
    echo "Configuration: $CONFIG_DESC"
    echo "Timestamp: $(date)"
    
    PARQUET_FILE=$(find_parquet_file)
    echo "Using parquet file: $PARQUET_FILE"
    
    if [ ! -z "$SOM_CONFIG_PATH" ]; then
        echo "SOM config: $SOM_CONFIG_PATH"
    fi
    echo ""
    
    # Create log directory
    mkdir -p logs
    LOG_FILE="logs/som_${CONFIG_TYPE}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to: $LOG_FILE"
    echo ""
    
    # Activate conda environment if needed
    if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "geo" ]; then
        echo "Activating geo conda environment..."
        eval "$(conda shell.bash hook)"
        conda activate geo
    fi
    
    # Run the analysis
    export PYTHONUNBUFFERED=1
    python scripts/run_analysis.py \
        --experiment test_som \
        --input "$PARQUET_FILE" \
        --method som \
        2>&1 | tee "$LOG_FILE"
}

# Function to monitor
monitor_som() {
    echo -e "${GREEN}=== SOM Live Monitor ===${NC}"
    echo "Looking for active SOM process..."
    
    # Find the latest log file
    LOG_FILE=$(ls -t logs/som_*.log logs/analysis_standalone_*.log 2>/dev/null | head -1)
    
    if [ -z "$LOG_FILE" ]; then
        echo -e "${YELLOW}No log file found. Waiting...${NC}"
        WAIT_COUNT=0
        while [ -z "$LOG_FILE" ] && [ $WAIT_COUNT -lt 30 ]; do
            sleep 1
            LOG_FILE=$(ls -t logs/som_*.log logs/analysis_standalone_*.log 2>/dev/null | head -1)
            ((WAIT_COUNT++))
        done
    fi
    
    if [ -z "$LOG_FILE" ]; then
        echo -e "${RED}ERROR: No log file found after 30 seconds${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Monitoring: $LOG_FILE${NC}"
    echo "Press Ctrl+C to stop"
    echo "====================================="
    echo ""
    
    # Monitor with color coding
    tail -f "$LOG_FILE" | while read line; do
        if [[ "$line" == *"ERROR"* ]] || [[ "$line" == *"Failed"* ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ "$line" == *"WARNING"* ]]; then
            echo -e "${YELLOW}$line${NC}"
        elif [[ "$line" == *"Starting"* ]] || [[ "$line" == *"Completed"* ]] || [[ "$line" == *"✅"* ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ "$line" == *"Epoch"* ]] || [[ "$line" == *"QE="* ]] || [[ "$line" == *"fold"* ]]; then
            echo -e "${BLUE}$line${NC}"
        elif [[ "$line" == *"rows"* ]] || [[ "$line" == *"samples"* ]] || [[ "$line" == *"Memory"* ]] || [[ "$line" == *"workers"* ]]; then
            echo "$line"
        fi
    done
}

# Function to run in tmux with monitoring
run_with_tmux() {
    SESSION_NAME="som_analysis"
    
    # Kill existing session
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    
    # Create new session
    tmux new-session -d -s "$SESSION_NAME" -n "SOM"
    
    # Main pane: Run analysis
    tmux send-keys -t "$SESSION_NAME:0.0" "cd $PROJECT_DIR" C-m
    tmux send-keys -t "$SESSION_NAME:0.0" "./run_som_analysis.sh" C-m
    
    # Split for monitoring
    tmux split-window -h -t "$SESSION_NAME:0"
    tmux send-keys -t "$SESSION_NAME:0.1" "./run_som_analysis.sh --monitor" C-m
    
    # Split for system resources
    tmux split-window -v -t "$SESSION_NAME:0.0"
    tmux send-keys -t "$SESSION_NAME:0.2" "htop -d 2" C-m
    
    echo -e "${GREEN}Created tmux session: $SESSION_NAME${NC}"
    echo "Commands:"
    echo "  tmux attach -t $SESSION_NAME  # Attach to session"
    echo "  tmux kill-session -t $SESSION_NAME  # Kill session"
    echo ""
    
    if [ "$DETACHED" = false ]; then
        tmux attach-session -t "$SESSION_NAME"
    fi
}

# Main execution
case $MODE in
    run)
        run_analysis
        ;;
    monitor)
        monitor_som
        ;;
    both)
        run_with_tmux
        ;;
esac