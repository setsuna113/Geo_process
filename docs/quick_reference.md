# Monitoring & Logging Quick Reference Card

## 🚀 Starting Pipelines

```bash
# Run with monitoring dashboard (recommended)
./run_monitored.sh --experiment-name my_experiment

# Run in background (daemon mode)
./run_monitored.sh --experiment-name my_experiment --daemon

# Resume from checkpoint
./run_monitored.sh --experiment-name my_experiment --resume
```

## 📊 Monitoring Commands

### Check Status
```bash
# View experiment status and progress
python scripts/monitor.py status my_experiment

# List all experiments
python scripts/monitor.py list
```

### View Logs
```bash
# View recent logs
python scripts/monitor.py logs my_experiment

# Filter by error level
python scripts/monitor.py logs my_experiment --level ERROR

# Search in logs
python scripts/monitor.py logs my_experiment --search "keyword"

# Show full tracebacks
python scripts/monitor.py logs my_experiment --level ERROR --traceback
```

### Live Monitoring
```bash
# Watch live progress (Ctrl+C to exit)
python scripts/monitor.py watch my_experiment
```

### Performance Metrics
```bash
# View resource usage
python scripts/monitor.py metrics my_experiment

# Filter by metric type
python scripts/monitor.py metrics my_experiment --type memory
```

### Error Summary
```bash
# Get error overview
python scripts/monitor.py errors my_experiment

# With tracebacks
python scripts/monitor.py errors my_experiment --traceback
```

## 🖥️ tmux Dashboard Navigation

When using `run_monitored.sh`, navigate with:

- **Switch windows**: `Ctrl+b` then `0-4`
- **Switch panes**: `Ctrl+b` then arrow keys
- **Detach session**: `Ctrl+b` then `d`
- **Reattach**: `tmux attach -t session_name`
- **Scroll in pane**: `Ctrl+b [` then arrows, `q` to exit

### Window Layout:
- **0**: Pipeline execution
- **1**: Live monitoring (progress, metrics, errors)
- **2**: Log viewers
- **3**: System resources (htop, disk usage)
- **4**: Control panel with commands

## 🔧 Process Management

```bash
# Check daemon status
python scripts/process_manager.py status my_daemon

# View daemon logs
python scripts/process_manager.py logs my_daemon -f

# Stop daemon gracefully
python scripts/process_manager.py stop my_daemon

# List all daemons
python scripts/process_manager.py status
```

## 🐍 Code Integration

### Basic Logging
```python
from src.infrastructure.logging import get_logger
logger = get_logger(__name__)

# Log with context
logger.info("Processing data", extra={
    'context': {'file_count': 100, 'size_mb': 1024}
})

# Log errors with traceback
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", exc_info=True)
```

### Performance Logging
```python
import time
start = time.time()
result = process_data()
duration = time.time() - start

logger.log_performance(
    "data_processing",
    duration,
    records_processed=len(result)
)
```

### Pipeline Context
```python
from src.pipelines.enhanced_context import EnhancedPipelineContext

# Use enhanced context for automatic monitoring
context = EnhancedPipelineContext(
    config=config,
    db=db,
    experiment_id="my_experiment",
    checkpoint_dir=checkpoint_dir,
    output_dir=output_dir
)

# Monitoring starts automatically
context.start_monitoring()
```

## 🚨 Common Issues

### No Logs Appearing
```bash
# Check monitoring is enabled
grep -A5 "monitoring:" config.yml

# Validate setup
python scripts/validate_monitoring_setup.py
```

### Database Connection Failed
```bash
# Check PostgreSQL is running
pg_isready

# Test connection
python -c "from src.database.connection import DatabaseManager; DatabaseManager()"
```

### High Memory Usage
```yaml
# Reduce in config.yml:
logging:
  database:
    batch_size: 50  # Smaller batches
    max_queue_size: 1000  # Limit queue
```

## 📝 SQL Queries

### Recent Errors
```sql
SELECT timestamp, level, message, traceback
FROM pipeline_logs
WHERE experiment_id = 'my_experiment'
  AND level IN ('ERROR', 'CRITICAL')
ORDER BY timestamp DESC
LIMIT 10;
```

### Experiment Progress
```sql
SELECT node_name, status, progress_percent
FROM pipeline_progress
WHERE experiment_id = 'my_experiment'
ORDER BY node_id;
```

### Resource Usage
```sql
SELECT timestamp, memory_mb, cpu_percent
FROM pipeline_metrics
WHERE experiment_id = 'my_experiment'
ORDER BY timestamp DESC
LIMIT 20;
```

## 🔗 Useful Links

- **Full Documentation**: `docs/monitoring_usage.md`
- **Migration Guide**: `MIGRATION_GUIDE.md`
- **Performance Tuning**: `docs/performance_tuning.md`
- **Dashboard Guide**: `docs/monitoring_dashboard.md`

## 📞 Support

- **Slack Channel**: #monitoring-support
- **Wiki**: [Internal Wiki Link]
- **On-Call**: [Phone/Contact]

---
*Keep this card handy for quick reference during pipeline operations!*