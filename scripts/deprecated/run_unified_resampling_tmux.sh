#!/bin/bash
# Unified resampling pipeline with tmux session management

echo "🚀 Unified Resampling Pipeline - Tmux Session Manager"
echo "===================================================="

# Check dependencies
if ! command -v tmux &> /dev/null; then
    echo "❌ Error: tmux is not installed."
    echo "   Install with: sudo apt install tmux"
    exit 1
fi

# Session configuration
SESSION_NAME="unified_resampling"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/unified_resampling"
mkdir -p "$LOG_DIR"

# Kill existing session if it exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Existing session '$SESSION_NAME' found."
    read -p "Kill existing session and start new one? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t $SESSION_NAME
        echo "✅ Killed existing session"
    else
        echo "📋 Cancelled. Use 'tmux attach -t $SESSION_NAME' to rejoin existing session."
        exit 0
    fi
fi

echo "🎬 Creating tmux session: $SESSION_NAME"

# Create new session with main pipeline window
tmux new-session -d -s $SESSION_NAME -n "pipeline" \
    "echo '🚀 Starting Unified Resampling Pipeline...'; \
     echo 'Session: $SESSION_NAME'; \
     echo 'Timestamp: $TIMESTAMP'; \
     echo ''; \
     ./run_unified_resampling.sh 2>&1 | tee $LOG_DIR/pipeline_$TIMESTAMP.log"

# Create monitoring window
tmux new-window -t $SESSION_NAME -n "monitor" \
    "echo '📊 System Monitoring for Unified Resampling Pipeline'; \
     echo 'Press Ctrl+C to stop monitoring'; \
     echo ''; \
     while true; do \
         clear; \
         echo '=== System Resources $(date) ==='; \
         echo 'Memory Usage:'; \
         free -h; \
         echo ''; \
         echo 'Disk Usage:'; \
         df -h . | head -2; \
         echo ''; \
         echo 'CPU Usage:'; \
         top -b -n1 | head -5; \
         echo ''; \
         echo 'Python Processes:'; \
         ps aux | grep python | grep -v grep | head -5; \
         echo ''; \
         echo 'Database Connections:'; \
         ss -tulpn | grep :51051 | wc -l | xargs echo 'Active connections:'; \
         echo ''; \
         echo 'Latest Log Entries:'; \
         tail -n 10 $LOG_DIR/pipeline_$TIMESTAMP.log 2>/dev/null || echo 'Log not yet available'; \
         sleep 5; \
     done"

# Create database monitoring window
tmux new-window -t $SESSION_NAME -n "database" \
    "echo '🗄️  Database Monitoring'; \
     echo 'This window shows database activity and statistics'; \
     echo 'Connect to your database CLI or monitoring tools here'; \
     echo ''; \
     echo 'Useful commands:'; \
     echo '  - psql -h localhost -p 51051 -U jason geoprocess_db'; \
     echo '  - SELECT * FROM resampled_datasets ORDER BY created_at DESC LIMIT 10;'; \
     echo '  - SELECT COUNT(*) FROM resampled_datasets;'; \
     echo ''; \
     bash"

# Create log viewer window
tmux new-window -t $SESSION_NAME -n "logs" \
    "echo '📄 Live Log Viewer'; \
     echo 'Pipeline log: $LOG_DIR/pipeline_$TIMESTAMP.log'; \
     echo ''; \
     tail -f $LOG_DIR/pipeline_$TIMESTAMP.log 2>/dev/null || \
     (echo 'Waiting for log file to be created...'; \
      while [ ! -f $LOG_DIR/pipeline_$TIMESTAMP.log ]; do sleep 1; done; \
      tail -f $LOG_DIR/pipeline_$TIMESTAMP.log)"

# Create results browser window
tmux new-window -t $SESSION_NAME -n "results" \
    "echo '📊 Results Browser'; \
     echo 'This window will show pipeline results when available'; \
     echo ''; \
     echo 'Output directory: outputs/unified_resampling/'; \
     echo ''; \
     cd outputs/unified_resampling 2>/dev/null || mkdir -p outputs/unified_resampling; \
     while true; do \
         echo '=== Results Status $(date) ==='; \
         if [ -d 'outputs/unified_resampling' ]; then \
             echo 'Available results:'; \
             ls -la outputs/unified_resampling/ 2>/dev/null | head -20; \
         else \
             echo 'Results directory not yet created'; \
         fi; \
         echo ''; \
         echo 'Press Ctrl+C to stop auto-refresh'; \
         sleep 30; \
         clear; \
     done"

# Select the main pipeline window
tmux select-window -t $SESSION_NAME:pipeline

# Create status bar configuration
tmux set-option -t $SESSION_NAME status on
tmux set-option -t $SESSION_NAME status-bg blue
tmux set-option -t $SESSION_NAME status-fg white
tmux set-option -t $SESSION_NAME status-left "#[fg=green]#S #[fg=white]| "
tmux set-option -t $SESSION_NAME status-right "#[fg=yellow]%Y-%m-%d %H:%M:%S"
tmux set-option -t $SESSION_NAME status-interval 1

echo ""
echo "✅ Tmux session '$SESSION_NAME' created successfully!"
echo ""
echo "📱 Session Layout:"
echo "   - Window 0 (pipeline): Main pipeline execution"
echo "   - Window 1 (monitor): System resource monitoring"
echo "   - Window 2 (database): Database monitoring and queries"
echo "   - Window 3 (logs): Live log viewer"
echo "   - Window 4 (results): Results browser"
echo ""
echo "🔧 Tmux Commands:"
echo "   - Attach to session: tmux attach -t $SESSION_NAME"
echo "   - Switch windows: Ctrl+b then 0-4"
echo "   - Detach session: Ctrl+b then d"
echo "   - Kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "📁 Logs will be saved to: $LOG_DIR/pipeline_$TIMESTAMP.log"
echo ""

# Attach to the session
echo "🎬 Attaching to session..."
sleep 2
tmux attach -t $SESSION_NAME