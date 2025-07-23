#!/bin/bash
# Robust cleanup script with proper error handling

echo "🧹 Cleaning up database and outputs..."

# Auto-detect Python environment
detect_python_env() {
    # Try common conda environment paths
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
        if python3 -c "import sys; sys.path.insert(0, 'src'); from src.database.connection import DatabaseManager" 2>/dev/null; then
            echo "python3"
            return 0
        fi
    fi
    
    return 1
}

PYTHON_ENV=$(detect_python_env)
if [ $? -ne 0 ] || [ -z "$PYTHON_ENV" ]; then
    echo "⚠️ Warning: Python environment with geo dependencies not found"
    echo "   Database cleanup will be skipped"
    PYTHON_ENV=""
else
    echo "✅ Python environment found: $PYTHON_ENV"
fi

# Clean database entries (only if Python environment is available)
if [ ! -z "$PYTHON_ENV" ]; then
    $PYTHON_ENV -c "
import sys
sys.path.insert(0, 'src')
try:
    from src.database.connection import DatabaseManager
    
    db = DatabaseManager()
    with db.get_connection() as conn:
        cur = conn.cursor()
        
        # Clear raster entries
        cur.execute('DELETE FROM raster_sources')
        raster_count = cur.rowcount
        
        # Clear merge log entries  
        cur.execute('DELETE FROM raster_merge_log')
        merge_count = cur.rowcount
        
        conn.commit()
        print(f'✅ Cleared {raster_count} raster entries and {merge_count} merge log entries')

except Exception as e:
    print(f'⚠️ Database cleanup error: {e}')
"
else
    echo "⚠️ Skipping database cleanup (no Python environment)"
fi

# Clean outputs properly (handle empty directories and missing files)
if [ -d "outputs/spatial_analysis" ]; then
    # Remove files if they exist
    find outputs/spatial_analysis -type f -delete 2>/dev/null
    # Remove empty directories if they exist
    find outputs/spatial_analysis -type d -empty -delete 2>/dev/null
    echo "✅ Cleared analysis outputs"
else
    echo "✅ No outputs directory to clear"
fi

# Create necessary directories
mkdir -p outputs/spatial_analysis
mkdir -p logs
echo "✅ Created directories"

# Clean up checkpoint files (if any exist)
if ls checkpoint_*.json 1> /dev/null 2>&1; then
    rm checkpoint_*.json
    echo "✅ Removed checkpoint files"
else
    echo "✅ No checkpoint files to remove"
fi

# Kill any hanging Python processes
if pgrep -f "run_richness_analysis.py" > /dev/null; then
    pkill -f "run_richness_analysis.py"
    echo "🛑 Killed hanging analysis processes"
else
    echo "✅ No hanging processes found"
fi

# Kill any existing tmux sessions
if tmux has-session -t "richness_analysis" 2>/dev/null; then
    tmux kill-session -t "richness_analysis"
    echo "🛑 Killed existing tmux session"
else
    echo "✅ No tmux session to kill"
fi

echo "✅ Cleanup complete - system ready!"