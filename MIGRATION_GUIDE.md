# Migration Guide: Unified Monitoring and Logging System

## Overview

This guide helps you migrate from the old logging approach (print statements, basic logging) to the new unified monitoring and logging system that provides structured logging, error capture, and real-time monitoring.

## Quick Start Migration

### 1. Database Setup

First, apply the monitoring schema to your database:

```bash
# Run the migration script
python scripts/migrate_monitoring_schema.py
```

This creates the following tables:
- `pipeline_logs` - Structured log storage
- `pipeline_events` - Significant events
- `pipeline_progress` - Hierarchical progress tracking
- `pipeline_metrics` - Performance metrics

### 2. Configuration Update

Add monitoring settings to your `config.yml`:

```yaml
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  database:
    enabled: true
    batch_size: 100
    flush_interval: 5.0
  file:
    enabled: true
    max_size_mb: 100
    backup_count: 5

monitoring:
  enabled: true
  metrics_interval: 10.0
  progress_update_interval: 5.0
  enable_resource_tracking: true
```

## Code Migration Patterns

### Replace Basic Logging

**Old approach:**
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing started")
print(f"Loading {file_count} files...")
```

**New approach:**
```python
from src.infrastructure.logging import get_logger
logger = get_logger(__name__)

logger.info("Processing started")
logger.info(f"Loading files", extra={'context': {'file_count': file_count}})
```

### Add Context to Operations

**Old approach:**
```python
def process_data(self, context):
    print("Starting data processing")
    try:
        # processing logic
        pass
    except Exception as e:
        print(f"Error: {e}")
        raise
```

**New approach:**
```python
from src.infrastructure.logging.decorators import log_stage

@log_stage("data_processing")
def process_data(self, context):
    logger.info("Starting data processing")
    # Context and errors are automatically captured
    # processing logic
```

### Replace Manual Timing

**Old approach:**
```python
import time

start = time.time()
result = expensive_operation()
duration = time.time() - start
print(f"Operation took {duration:.2f} seconds")
```

**New approach:**
```python
start = time.time()
result = expensive_operation()
duration = time.time() - start

logger.log_performance(
    "expensive_operation",
    duration,
    items_processed=len(result),
    status="success"
)
```

### Error Handling with Traceback

**Old approach:**
```python
try:
    risky_operation()
except Exception as e:
    print(f"Operation failed: {e}")
    # Traceback lost in daemon mode!
    raise
```

**New approach:**
```python
try:
    risky_operation()
except Exception as e:
    logger.error(
        "Operation failed",
        exc_info=True,  # Captures full traceback
        extra={
            'context': {
                'operation': 'risky_operation',
                'input_size': data_size
            }
        }
    )
    raise
```

## Pipeline Migration

### Update Pipeline Context

**Old approach:**
```python
from src.pipelines.context import PipelineContext

context = PipelineContext(
    config=config,
    db=db,
    experiment_id=exp_id,
    checkpoint_dir=checkpoint_dir,
    output_dir=output_dir
)
```

**New approach:**
```python
from src.pipelines.enhanced_context import EnhancedPipelineContext

context = EnhancedPipelineContext(
    config=config,
    db=db,
    experiment_id=exp_id,
    checkpoint_dir=checkpoint_dir,
    output_dir=output_dir
)

# Start monitoring
context.start_monitoring()

try:
    # Pipeline execution
    pass
finally:
    context.stop_monitoring()
```

### Update Pipeline Orchestrator

**Old approach:**
```python
from src.pipelines.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(config, db)
orchestrator.run_pipeline(experiment_name)
```

**New approach:**
```python
from src.pipelines.orchestrator_enhanced import EnhancedPipelineOrchestrator

orchestrator = EnhancedPipelineOrchestrator(config, db)
# Monitoring and logging are automatically integrated
orchestrator.run_pipeline(experiment_name)
```

## Process Management Migration

### Update Process Controller

**Old approach:**
```python
from src.core.process_controller import ProcessController

controller = ProcessController()
pid = controller.start_process(
    name="my_process",
    command=["python", "script.py"],
    daemon_mode=True
)
```

**New approach:**
```python
from src.core.process_controller_enhanced import EnhancedProcessController

controller = EnhancedProcessController(
    experiment_id="my_experiment"  # Links logs to experiment
)
pid = controller.start_process(
    name="my_process",
    command=["python", "script.py"],
    daemon_mode=True
)
# Daemon crashes now captured with full traceback!
```

### Update Signal Handlers

**Old approach:**
```python
from src.core.signal_handler import SignalHandler

handler = SignalHandler(cleanup_func)
```

**New approach:**
```python
from src.core.signal_handler_enhanced import EnhancedSignalHandler

handler = EnhancedSignalHandler(cleanup_func)
# Uncaught exceptions now logged with full context
```

## Running Pipelines

### Use New Run Scripts

**Old approach:**
```bash
./run_pipeline.sh my_experiment
```

**New approach with monitoring:**
```bash
# Run with integrated tmux monitoring
./run_monitored.sh --experiment-name my_experiment

# Run in daemon mode with monitoring
./run_monitored.sh --experiment-name my_experiment --daemon
```

### Monitor Running Pipelines

```bash
# View real-time status
python scripts/monitor.py status my_experiment

# Watch live progress
python scripts/monitor.py watch my_experiment

# Query logs
python scripts/monitor.py logs my_experiment --level ERROR --traceback

# View performance metrics
python scripts/monitor.py metrics my_experiment

# Get error summary
python scripts/monitor.py errors my_experiment
```

## Migration Checklist

### Phase 1: Infrastructure (Day 1)
- [ ] Run database migration script
- [ ] Update config.yml with monitoring settings
- [ ] Verify database tables created
- [ ] Test monitor.py CLI tool

### Phase 2: Core Components (Days 2-3)
- [ ] Replace ProcessController with EnhancedProcessController
- [ ] Replace SignalHandler with EnhancedSignalHandler
- [ ] Update PipelineOrchestrator to EnhancedPipelineOrchestrator
- [ ] Replace PipelineContext with EnhancedPipelineContext

### Phase 3: Logging Updates (Days 4-5)
- [ ] Replace `import logging` with `from src.infrastructure.logging import get_logger`
- [ ] Add `@log_stage` decorators to pipeline stages
- [ ] Replace print statements with structured logging
- [ ] Add `exc_info=True` to error logging

### Phase 4: Testing (Day 6)
- [ ] Run test suite: `pytest tests/infrastructure/`
- [ ] Test daemon process monitoring
- [ ] Verify error capture in tmux/daemon mode
- [ ] Test monitoring dashboard

### Phase 5: Deployment (Day 7)
- [ ] Update production configuration
- [ ] Deploy enhanced components
- [ ] Update documentation
- [ ] Train team on new monitoring tools

## Common Issues and Solutions

### Issue: "Database connection refused"
**Solution:** Ensure PostgreSQL is running and monitoring tables are created:
```bash
python scripts/migrate_monitoring_schema.py
```

### Issue: "No logs appearing in monitor"
**Solution:** Check that database logging is enabled in config:
```yaml
logging:
  database:
    enabled: true
```

### Issue: "Context not propagating"
**Solution:** Ensure you're using EnhancedPipelineContext and logging within context managers:
```python
with context.logging_context.stage("my_stage"):
    logger.info("This will have stage context")
```

### Issue: "Daemon logs not captured"
**Solution:** Use EnhancedProcessController with experiment_id:
```python
controller = EnhancedProcessController(experiment_id="my_exp")
```

## Best Practices

1. **Always use structured logging**
   - Include context in extra parameter
   - Use exc_info=True for exceptions
   - Log performance metrics for key operations

2. **Use context managers**
   - Wrap stages in logging contexts
   - Use monitor.track_stage() for progress

3. **Monitor resource usage**
   - Record metrics at key points
   - Use context.record_metrics()

4. **Handle errors properly**
   - Always log with traceback
   - Include relevant context
   - Continue gracefully when possible

5. **Use unique experiment names**
   - Prevents database conflicts
   - Makes monitoring easier

## Rollback Plan

If issues arise, you can temporarily disable the new system:

1. Set in config.yml:
   ```yaml
   logging:
     database:
       enabled: false
   monitoring:
     enabled: false
   ```

2. Use original components:
   - ProcessController instead of EnhancedProcessController
   - PipelineOrchestrator instead of EnhancedPipelineOrchestrator

3. Continue using old run scripts:
   ```bash
   ./run_pipeline.sh experiment_name
   ```

## Support

For issues or questions:
1. Check logs: `python scripts/monitor.py logs <experiment> --level ERROR`
2. Review this guide
3. Run tests: `pytest tests/infrastructure/ -v`
4. Check database: `psql -d <database> -c "SELECT * FROM pipeline_logs ORDER BY timestamp DESC LIMIT 10;"`

## Next Steps

After migration:
1. Remove old print statements
2. Add more detailed context to logs
3. Set up alerts based on error patterns
4. Create custom monitoring dashboards
5. Archive old log files

The new system provides the debugging capabilities needed for complex distributed pipelines while maintaining backward compatibility during the transition.