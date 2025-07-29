# Monitoring and Logging System Usage Guide

## Overview

The enhanced monitoring and logging system provides comprehensive visibility into pipeline execution, especially for daemon processes that previously died silently. This guide covers how to use the new monitoring features.

## Quick Start

### Running a Monitored Pipeline

```bash
# Basic usage with tmux monitoring
./run_monitored.sh --experiment-name my_experiment

# Run in daemon mode
./run_monitored.sh --experiment-name my_experiment --daemon

# Resume from checkpoint
./run_monitored.sh --experiment-name my_experiment --resume

# Specify analysis method
./run_monitored.sh --experiment-name my_experiment --analysis-method gwpca
```

### Monitoring Commands

```bash
# View experiment status and progress tree
python scripts/monitor.py status my_experiment

# View logs in real-time
python scripts/monitor.py logs my_experiment -f

# Watch live monitoring dashboard
python scripts/monitor.py watch my_experiment

# View performance metrics
python scripts/monitor.py metrics my_experiment

# View error summary with tracebacks
python scripts/monitor.py errors my_experiment --traceback

# List all experiments
python scripts/monitor.py list
```

## tmux Session Layout

When using `run_monitored.sh`, a tmux session is created with 5 windows:

1. **pipeline** (Window 0): Main pipeline execution
2. **monitoring** (Window 1): Live monitoring dashboard
   - Pane 0: Progress tracking
   - Pane 1: Memory metrics
   - Pane 2: Error monitoring
3. **logs** (Window 2): Log viewing
   - Pane 0: All logs
   - Pane 1: Error logs with tracebacks
4. **system** (Window 3): System monitoring
   - Pane 0: htop/top
   - Pane 1: Disk usage
5. **control** (Window 4): Control panel with command reference

### tmux Navigation

- **Attach to session**: `tmux attach -t <session_name>`
- **Detach from session**: `Ctrl+b d`
- **Switch windows**: `Ctrl+b [0-4]`
- **Switch panes**: `Ctrl+b arrow_keys`
- **Scroll in pane**: `Ctrl+b [` then use arrow keys, `q` to exit

## Process Control

### Managing Daemon Processes

```bash
# Start a daemon process
python scripts/process_manager.py start --name my_daemon --daemon --experiment-name unique_exp_name

# Check daemon status
python scripts/process_manager.py status my_daemon

# View daemon logs
python scripts/process_manager.py logs my_daemon -f

# Pause/resume daemon
python scripts/process_manager.py pause my_daemon
python scripts/process_manager.py resume my_daemon

# Stop daemon gracefully
python scripts/process_manager.py stop my_daemon

# List all daemons
python scripts/process_manager.py status
```

## Debugging Features

### Error Capture

The system now captures full tracebacks when processes crash:

```bash
# View all errors with tracebacks
python scripts/monitor.py errors my_experiment --traceback

# Filter logs by level
python scripts/monitor.py logs my_experiment --level ERROR --traceback

# Search in logs
python scripts/monitor.py logs my_experiment --search "specific error"
```

### Signal Handling

The enhanced signal handler captures context when processes die:
- SIGTERM: Graceful shutdown with cleanup
- SIGINT: User interruption (Ctrl+C)
- SIGUSR1: Custom signal (triggers status report)
- Uncaught exceptions: Full traceback capture

### Performance Monitoring

```bash
# View real-time metrics
python scripts/monitor.py watch my_experiment

# Get specific metric types
python scripts/monitor.py metrics my_experiment --type memory
python scripts/monitor.py metrics my_experiment --type cpu
python scripts/monitor.py metrics my_experiment --type throughput
```

## Database Schema

All monitoring data is stored in PostgreSQL:

- `pipeline_logs`: Structured log entries with context
- `pipeline_events`: Significant events (start, stop, error)
- `pipeline_progress`: Hierarchical progress tracking
- `pipeline_metrics`: Performance metrics

### Querying Logs Directly

```sql
-- Recent errors for an experiment
SELECT timestamp, level, message, traceback
FROM pipeline_logs
WHERE experiment_id = 'my_experiment' 
  AND level IN ('ERROR', 'CRITICAL')
ORDER BY timestamp DESC
LIMIT 10;

-- Progress summary
SELECT * FROM pipeline_progress_summary
WHERE experiment_id = 'my_experiment';
```

## Best Practices

1. **Always use unique experiment names** to avoid database conflicts
2. **Monitor resource usage** - the system tracks memory and CPU
3. **Check logs regularly** - especially for long-running daemons
4. **Use tmux sessions** for interactive monitoring
5. **Enable resume mode** for long pipelines to handle interruptions

## Troubleshooting

### Process Won't Start
- Check if another process with same name exists: `python scripts/process_manager.py status`
- Verify experiment name is unique
- Check Python environment detection

### Can't See Logs
- Ensure database is running: `python scripts/monitor.py list`
- Check log level filter
- Verify experiment name

### tmux Session Issues
- List sessions: `tmux ls`
- Kill old session: `tmux kill-session -t <name>`
- Check if tmux is installed: `which tmux`

## Configuration

The monitoring system respects these config settings:

```yaml
# config.yml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  database:
    batch_size: 100
    flush_interval: 5.0
  
monitoring:
  metrics_interval: 10.0
  progress_update_interval: 5.0
```

## Migration from Old System

The new system is backward compatible:
- Old checkpoints still work
- Existing progress tracking continues
- Database logging is optional (falls back to file)

To migrate:
1. Use `run_monitored.sh` instead of `run_pipeline.sh`
2. Update daemon launches to use process manager
3. Replace print statements with structured logging