#!/bin/bash
# Unified resampling pipeline launcher - Minimal wrapper
# All validation and logic is handled by the Python script

echo "🚀 Unified Resampling Pipeline - Smart Launcher"
echo "==============================================="

# Get script directory and find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find project root (look for src directory)
if [ -d "$SCRIPT_DIR/../../src" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
elif [ -d "$SCRIPT_DIR/src" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    echo "❌ Error: Cannot find project root with src/ directory"
    exit 1
fi

cd "$PROJECT_ROOT"
echo "📍 Project root: $PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "src/pipelines/unified_resampling/scripts/run_unified_resampling.py" ]; then
    echo "❌ Error: Unified resampling script not found. Are you in the right directory?"
    exit 1
fi

# Auto-detect Python environment
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
        if python3 -c "import sys; sys.path.insert(0, 'src'); from src.config.config import Config" 2>/dev/null; then
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

echo ""
echo "🎬 The Python script will handle:"
echo "   - Automatic test/production mode detection"
echo "   - Configuration validation (defaults.py + config.yml)"
echo "   - Data file validation"
echo "   - Database connectivity check"
echo "   - System resource validation"
echo "   - All pipeline execution"
echo ""

# Show usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "📚 Usage Examples:"
    echo ""
    echo "  # Small test run (auto-detects test mode):"
    echo "  $0 --target-resolution 0.2 --max-samples 10000 --som-grid-size 4 4 --som-iterations 100"
    echo ""
    echo "  # Production run with custom settings:"
    echo "  $0 --target-resolution 0.05 --max-samples 500000 --experiment-name my_experiment"
    echo ""
    echo "  # Dry run to validate configuration:"
    echo "  $0 --dry-run --validate-inputs"
    echo ""
    echo "  # See all options:"
    echo "  $0 --help"
    echo ""
    read -p "Continue with default small test parameters? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        set -- --target-resolution 0.2 --max-samples 10000 --som-grid-size 4 4 --som-iterations 100 --experiment-name "interactive_test"
    else
        echo "📋 Cancelled. Run with specific parameters or --help for options."
        exit 0
    fi
fi

echo "🎬 Executing pipeline..."
echo "Command: $PYTHON_ENV src/pipelines/unified_resampling/scripts/run_unified_resampling.py $@"
echo ""

# Execute the pipeline - let Python handle everything
exec $PYTHON_ENV src/pipelines/unified_resampling/scripts/run_unified_resampling.py "$@"