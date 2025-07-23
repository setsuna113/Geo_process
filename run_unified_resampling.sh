#!/bin/bash
# Unified resampling pipeline runner with comprehensive error handling

echo "🚀 Unified Resampling Pipeline - Complete Runner"
echo "==============================================="

# Check if we're in the right directory
if [ ! -f "src/pipelines/unified_resampling/scripts/run_unified_resampling.py" ]; then
    echo "❌ Error: Unified resampling script not found. Are you in the right directory?"
    exit 1
fi

# Step 1: Clean up first (optional)
echo "🧹 Step 1: Cleanup check..."
if [ -f "./cleanup.sh" ]; then
    read -p "🗑️  Run cleanup first? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if ./cleanup.sh; then
            echo "✅ Cleanup completed"
        else
            echo "⚠️ Cleanup had issues, but continuing..."
        fi
    fi
else
    echo "ℹ️  No cleanup script found, continuing..."
fi

echo ""

# Step 2: Check dependencies
echo "🔍 Step 2: Checking system dependencies..."

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ Error: tmux is not installed."
    echo "   Install with: sudo apt install tmux"
    exit 1
else
    echo "✅ tmux is available"
fi

# Auto-detect Python environment (same logic as original)
detect_python_env() {
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
        if python3 -c "import sys; sys.path.insert(0, 'src'); from src.pipelines.unified_resampling import UnifiedResamplingPipeline" 2>/dev/null; then
            echo "python3"
            return 0
        fi
    fi
    
    return 1
}

PYTHON_ENV=$(detect_python_env)
if [ $? -ne 0 ] || [ -z "$PYTHON_ENV" ]; then
    echo "❌ Error: Python environment with required dependencies not found"
    echo "   Please ensure you have:"
    echo "   - A conda environment named 'geo_py311' with required packages, OR"
    echo "   - System python3 with the unified resampling dependencies installed"
    exit 1
else
    echo "✅ Python environment found: $PYTHON_ENV"
fi

# Read data directory from config.yml
if [ ! -f "config.yml" ]; then
    echo "❌ Error: config.yml not found in current directory"
    exit 1
fi

# Extract data_dir from config.yml
DATA_DIR=$(grep -E "^\s*data_dir:" config.yml | grep -v "^#" | head -1 | sed 's/.*data_dir:\s*["\x27]\?\([^"\x27]*\)["\x27]\?\s*#*.*/\1/' | xargs)

if [ -z "$DATA_DIR" ]; then
    echo "❌ Error: data_dir not found in config.yml"
    echo "   Please ensure data_dir is properly configured"
    exit 1
fi

echo "📁 Using data directory: $DATA_DIR"

# Check if data files exist (read from config.yml)
echo "🔍 Checking configured datasets..."

# Extract dataset files from config.yml
DATASET_FILES=$(grep -A 10 "^data_files:" config.yml | grep -E "^\s*[a-zA-Z_]+:" | sed 's/.*:\s*["\x27]\?\([^"\x27]*\)["\x27]\?\s*#*.*/\1/')
MISSING_FILES=0

for FILE in $DATASET_FILES; do
    FULL_PATH="$DATA_DIR/$FILE"
    if [ ! -f "$FULL_PATH" ]; then
        echo "❌ Missing: $FULL_PATH"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        FILE_SIZE=$(du -h "$FULL_PATH" | cut -f1)
        echo "✅ Found: $FILE ($FILE_SIZE)"
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo "❌ Error: $MISSING_FILES dataset file(s) missing"
    exit 1
fi

echo ""

# Step 3: System resource check (enhanced for resampling)
echo "🖥️  Step 3: Checking system resources..."

# Check available memory
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
AVAIL_MEM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
AVAIL_MEM_GB=$((AVAIL_MEM_KB / 1024 / 1024))

echo "   Memory: ${AVAIL_MEM_GB}GB available / ${TOTAL_MEM_GB}GB total"

# More stringent memory requirements for resampling
if [ "$AVAIL_MEM_GB" -lt 8 ]; then
    echo "❌ Error: Insufficient memory (${AVAIL_MEM_GB}GB). Resampling requires at least 8GB."
    echo "   Recommended: 16GB+ for optimal performance"
    exit 1
elif [ "$AVAIL_MEM_GB" -lt 16 ]; then
    echo "⚠️ Warning: Limited memory (${AVAIL_MEM_GB}GB). Consider using batch processing options."
fi

# Check available disk space (higher requirements for resampling)
AVAIL_DISK_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
echo "   Disk space: ${AVAIL_DISK_GB}GB available"

if [ "$AVAIL_DISK_GB" -lt 20 ]; then
    echo "❌ Error: Insufficient disk space (${AVAIL_DISK_GB}GB)."
    echo "   Resampling pipeline needs at least 20GB for intermediate data and results"
    exit 1
fi

echo ""

# Step 4: Pipeline-specific validation
echo "📋 Step 4: Pipeline validation..."

# Run dry-run validation
echo "   Running configuration validation..."
$PYTHON_ENV src/pipelines/unified_resampling/scripts/run_unified_resampling.py --dry-run --validate-inputs

if [ $? -ne 0 ]; then
    echo "❌ Error: Pipeline validation failed"
    echo "   Check your configuration and data files"
    exit 1
fi

echo "   ✅ Pipeline validation passed"
echo ""

# Step 5: Show pipeline information
echo "⏱️  Step 5: Pipeline information..."
echo "   📊 Processing Mode: Multi-dataset resampling with SOM analysis"
echo "   🎯 Target: Uniform resolution resampling → database storage → SOM"
echo "   ⏰ Estimated phases:"
echo "     - Dataset resampling: 30-60 minutes per dataset"
echo "     - Dataset merging: 5-15 minutes" 
echo "     - SOM analysis: 1-4 hours (depending on resolution)"
echo "     - Results generation: 5-10 minutes"
echo ""

# Step 6: Advanced options
echo "🔧 Step 6: Processing options..."

# Ask about processing preferences
echo "Select processing options:"
echo "1) Standard processing (recommended)"
echo "2) High-memory processing (faster, requires 32GB+ RAM)"
echo "3) Conservative processing (slower, works with limited resources)"
echo "4) Custom options"

read -p "Choose option (1-4): " -n 1 -r
echo ""

PYTHON_ARGS=""

case $REPLY in
    2)
        echo "   Using high-memory processing options..."
        PYTHON_ARGS="--memory-limit 24 --max-samples 2000000 --batch-processing"
        ;;
    3)
        echo "   Using conservative processing options..."
        PYTHON_ARGS="--memory-limit 4 --max-samples 100000 --batch-processing --chunk-size 10000"
        ;;
    4)
        echo "   Available custom options:"
        echo "     --target-resolution FLOAT    Override target resolution (degrees)"
        echo "     --resampling-engine {numpy,gdal}  Choose resampling engine"
        echo "     --skip-existing              Skip already resampled datasets"
        echo "     --skip-som                   Skip SOM analysis"
        echo "     --memory-limit GB            Set memory limit"
        echo "     --experiment-name NAME       Set experiment name"
        echo ""
        read -p "   Enter custom arguments (or press Enter for defaults): " CUSTOM_ARGS
        PYTHON_ARGS="$CUSTOM_ARGS"
        ;;
    *)
        echo "   Using standard processing options..."
        PYTHON_ARGS="--memory-limit 8 --max-samples 500000"
        ;;
esac

echo ""

# Step 7: Final confirmation
echo "🎬 Ready to start unified resampling pipeline!"
echo ""
echo "📱 The pipeline will run with:"
echo "   - Multi-dataset resampling to uniform resolution"
echo "   - Database storage of intermediate results"
echo "   - Real-time progress monitoring" 
echo "   - Automatic checkpoint saving"
echo "   - Comprehensive validation and error handling"
echo ""
echo "Processing arguments: $PYTHON_ARGS"
echo ""

read -p "🚀 Start the pipeline now? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "📋 Pipeline cancelled. Run this script again when ready."
    exit 0
fi

echo ""
echo "🎬 Starting unified resampling pipeline..."
sleep 2

# Execute the pipeline
echo "Executing: $PYTHON_ENV src/pipelines/unified_resampling/scripts/run_unified_resampling.py $PYTHON_ARGS"
exec $PYTHON_ENV src/pipelines/unified_resampling/scripts/run_unified_resampling.py $PYTHON_ARGS