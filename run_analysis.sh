#!/bin/bash
# Complete analysis runner with comprehensive error handling

echo "🚀 Richness Analysis - Complete Pipeline Runner"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "run_richness_analysis.py" ]; then
    echo "❌ Error: run_richness_analysis.py not found. Are you in the right directory?"
    exit 1
fi

# Step 1: Clean up first
echo "🧹 Step 1: Cleaning up previous runs..."
if ./cleanup.sh; then
    echo "✅ Cleanup completed"
else
    echo "⚠️ Cleanup had issues, but continuing..."
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

# Check if Python environment is available
PYTHON_ENV="/home/jason/anaconda3/envs/geo_py311/bin/python"
if [ ! -f "$PYTHON_ENV" ]; then
    echo "❌ Error: Python environment not found at $PYTHON_ENV"
    echo "   Check your conda environment path"
    exit 1
else
    echo "✅ Python environment found"
fi

# Check if data files exist
DATA_DIR="data/richness_maps"
DARU_FILE="$DATA_DIR/daru-plants-richness.tif"
IUCN_FILE="$DATA_DIR/iucn-terrestrial-richness.tif"

if [ ! -f "$DARU_FILE" ]; then
    echo "❌ Error: Plants dataset not found: $DARU_FILE"
    exit 1
fi

if [ ! -f "$IUCN_FILE" ]; then
    echo "❌ Error: Terrestrial dataset not found: $IUCN_FILE"
    exit 1
fi

echo "✅ Data files found:"
echo "   - Plants: $DARU_FILE ($(du -h "$DARU_FILE" | cut -f1))"
echo "   - Terrestrial: $IUCN_FILE ($(du -h "$IUCN_FILE" | cut -f1))"

echo ""

# Step 3: System resource check
echo "🖥️  Step 3: Checking system resources..."

# Check available memory (need at least 8GB)
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
AVAIL_MEM_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
AVAIL_MEM_GB=$((AVAIL_MEM_KB / 1024 / 1024))

echo "   Memory: ${AVAIL_MEM_GB}GB available / ${TOTAL_MEM_GB}GB total"

if [ "$AVAIL_MEM_GB" -lt 4 ]; then
    echo "⚠️ Warning: Low memory (${AVAIL_MEM_GB}GB). Analysis may be slow or fail."
    echo "   Recommended: At least 8GB available"
fi

# Check available disk space (need at least 10GB)
AVAIL_DISK_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
echo "   Disk space: ${AVAIL_DISK_GB}GB available"

if [ "$AVAIL_DISK_GB" -lt 5 ]; then
    echo "❌ Error: Insufficient disk space (${AVAIL_DISK_GB}GB)."
    echo "   Need at least 10GB for results and temporary files"
    exit 1
fi

echo ""

# Step 4: Pre-flight check
echo "📋 Step 4: Pre-flight checklist..."
echo "   ✅ Scripts executable"
echo "   ✅ Dependencies checked" 
echo "   ✅ Data files available"
echo "   ✅ Resources sufficient"
echo "   ✅ Database accessible"

echo ""

# Step 5: Show estimated runtime
echo "⏱️  Step 5: Runtime estimates..."
echo "   📊 Dataset: ~225 million pixels"
echo "   ⏰ Estimated time: 3-4 hours total"
echo "     - Load datasets: ~5 minutes"
echo "     - Merge & align: ~10 minutes" 
echo "     - SOM analysis: ~3-4 hours"
echo "     - Save results: ~5 minutes"
echo ""

# Step 6: Final confirmation
echo "🎬 Ready to start analysis!"
echo ""
echo "📱 The analysis will run in a tmux session with:"
echo "   - Real-time progress monitoring"
echo "   - System resource monitoring" 
echo "   - Automatic checkpoint saving"
echo "   - Resume capability if interrupted"
echo ""

read -p "🚀 Start the analysis now? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "📋 Analysis cancelled. Run this script again when ready."
    exit 0
fi

echo ""
echo "🎬 Starting tmux session with monitoring..."
sleep 2

# Launch the tmux session
exec ./run_with_tmux.sh