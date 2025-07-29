# Unified Monitoring and Logging System Implementation Plan

## Executive Summary

This plan integrates structured logging with the monitoring reconstruction to create a unified system for tracking pipeline execution, capturing errors, and debugging issues in daemon/tmux environments. The system enhances (not replaces) existing components while adding database persistence and structured logging.

## Current State Assessment

### Working Components to Keep
1. **ProcessController** (`src/core/process_controller.py`)
   - Daemon management without sudo (cluster-friendly)
   - PID file tracking
   - Log rotation
   - Auto-restart logic
   - **Status**: Keep and enhance with structured logging

2. **ProcessRegistry** (`src/core/process_registry.py`)
   - JSON-based persistent process tracking
   - Cross-invocation daemon management
   - **Status**: Keep as-is

3. **Base Abstractions** (`src/base/`)
   - Define contracts for processors, checkpoints, memory tracking
   - **Status**: Respect these interfaces in new implementations

4. **Database Schema**
   - `experiments` table with status tracking
   - `processing_jobs` with log_messages[] array
   - `pipeline_checkpoints` for checkpoint tracking
   - **Status**: Extend with new tables

### Problems to Solve
1. **Silent Deaths**: No traceback when daemons die
2. **Hanging Processes**: No visibility into stuck operations
3. **Lost Logs**: Daemon logs scattered, no correlation
4. **No Context**: Logs lack experiment_id, stage, node_id
5. **Debugging Difficulty**: Can't query logs by experiment/error

## Implementation Plan

### Phase 1: Database Schema Extensions (Week 1, Days 1-2)

#### 1.1 Create Logging Tables
```sql
-- Structured logs table
CREATE TABLE pipeline_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id),
    job_id UUID REFERENCES processing_jobs(id),
    node_id VARCHAR(255),  -- Links to pipeline_progress.node_id
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(20) CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    logger_name VARCHAR(255),
    message TEXT,
    context JSONB DEFAULT '{}',  -- stage, step, iteration, etc.
    traceback TEXT,  -- Full traceback for errors
    performance JSONB DEFAULT '{}',  -- timing, memory, throughput
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast querying
CREATE INDEX idx_logs_experiment_time ON pipeline_logs(experiment_id, timestamp DESC);
CREATE INDEX idx_logs_level ON pipeline_logs(level) WHERE level IN ('ERROR', 'CRITICAL');
CREATE INDEX idx_logs_node ON pipeline_logs(node_id);
CREATE INDEX idx_logs_context ON pipeline_logs USING gin(context);

-- Events table for significant occurrences
CREATE TABLE pipeline_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id),
    event_type VARCHAR(50) NOT NULL,  -- 'stage_start', 'stage_complete', 'error', 'warning'
    source VARCHAR(255) NOT NULL,  -- Component that generated event
    severity VARCHAR(20) DEFAULT 'info',
    title VARCHAR(500) NOT NULL,
    details JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Monitoring tables from original plan
CREATE TABLE pipeline_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id),
    node_id VARCHAR(255) NOT NULL,
    node_level VARCHAR(50) NOT NULL CHECK (node_level IN ('pipeline', 'phase', 'step', 'substep')),
    parent_id VARCHAR(255),
    node_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress_percent FLOAT DEFAULT 0,
    completed_units INTEGER DEFAULT 0,
    total_units INTEGER DEFAULT 100,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, node_id)
);

CREATE TABLE pipeline_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id),
    node_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    memory_mb FLOAT,
    cpu_percent FLOAT,
    disk_usage_mb FLOAT,
    throughput_per_sec FLOAT,
    custom_metrics JSONB DEFAULT '{}'
);
```

#### 1.2 Migration Script
Create `scripts/migrate_monitoring_schema.py`:
```python
#!/usr/bin/env python3
"""Add monitoring and logging tables to database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import DatabaseManager
from src.database.schema import DatabaseSchema

def main():
    db = DatabaseManager()
    schema = DatabaseSchema(db)
    
    # Read and execute the new schema
    schema_file = Path(__file__).parent.parent / "src/database/monitoring_schema.sql"
    db.execute_sql_file(schema_file)
    
    print("✅ Monitoring schema created successfully")

if __name__ == "__main__":
    main()
```

### Phase 2: Structured Logging Infrastructure (Week 1, Days 3-5)

#### 2.1 Create Logging Module Structure
```
src/infrastructure/logging/
├── __init__.py
├── structured_logger.py      # Main logger class
├── context.py               # Logging context management
├── handlers/
│   ├── __init__.py
│   ├── database_handler.py  # PostgreSQL handler
│   ├── file_handler.py      # Enhanced file handler
│   └── console_handler.py   # Colored console output
├── formatters/
│   ├── __init__.py
│   ├── json_formatter.py    # JSON for files/database
│   └── human_formatter.py   # Human-readable for console
└── decorators.py            # Logging decorators
```

#### 2.2 Implement StructuredLogger
`src/infrastructure/logging/structured_logger.py`:
```python
"""Structured logging with context propagation."""
import logging
import json
import traceback
from typing import Dict, Any, Optional
from contextvars import ContextVar
from datetime import datetime

# Context variables for correlation
experiment_context: ContextVar[Optional[str]] = ContextVar('experiment_id', default=None)
node_context: ContextVar[Optional[str]] = ContextVar('node_id', default=None)
stage_context: ContextVar[Optional[str]] = ContextVar('stage', default=None)

class StructuredLogger(logging.Logger):
    """Enhanced logger with structured output and context."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._context_fields: Dict[str, Any] = {}
    
    def _log(self, level, msg, args, exc_info=None, extra=None, **kwargs):
        """Override to add context and structure."""
        # Gather context
        context = {
            'experiment_id': experiment_context.get(),
            'node_id': node_context.get(),
            'stage': stage_context.get(),
            'logger_name': self.name,
            'timestamp': datetime.utcnow().isoformat(),
            **self._context_fields
        }
        
        # Add any extra context
        if extra:
            context.update(extra)
        
        # Capture traceback if error
        tb_text = None
        if exc_info:
            if isinstance(exc_info, bool):
                exc_info = sys.exc_info()
            if exc_info[0] is not None:
                tb_text = ''.join(traceback.format_exception(*exc_info))
        
        # Create structured record
        structured_extra = {
            'context': context,
            'traceback': tb_text
        }
        
        super()._log(level, msg, args, exc_info=False, extra=structured_extra, **kwargs)
    
    def add_context(self, **fields):
        """Add persistent context fields."""
        self._context_fields.update(fields)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        self.info(f"Performance: {operation}", extra={
            'performance': {
                'operation': operation,
                'duration_seconds': duration,
                **metrics
            }
        })

# Factory function
def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    logging.setLoggerClass(StructuredLogger)
    logger = logging.getLogger(name)
    logging.setLoggerClass(logging.Logger)  # Reset for other loggers
    return logger
```

#### 2.3 Implement Database Handler
`src/infrastructure/logging/handlers/database_handler.py`:
```python
"""Database handler for structured logging."""
import logging
import json
import queue
import threading
from typing import Optional
from datetime import datetime

class DatabaseLogHandler(logging.Handler):
    """Async database handler with batching."""
    
    def __init__(self, db_manager, batch_size: int = 100, flush_interval: float = 5.0):
        super().__init__()
        self.db = db_manager
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Async queue
        self.queue = queue.Queue(maxsize=10000)
        self.batch = []
        
        # Background thread
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    
    def emit(self, record: logging.LogRecord):
        """Queue log record for database insertion."""
        try:
            # Don't block if queue is full
            self.queue.put_nowait(record)
        except queue.Full:
            # Fallback to stderr
            sys.stderr.write(f"Log queue full, dropping: {record.getMessage()}\n")
    
    def _worker(self):
        """Background worker to batch insert logs."""
        while not self._stop_event.is_set():
            try:
                # Collect batch
                timeout = self.flush_interval
                deadline = time.time() + timeout
                
                while len(self.batch) < self.batch_size and time.time() < deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                        
                    try:
                        record = self.queue.get(timeout=min(remaining, 1.0))
                        self.batch.append(record)
                    except queue.Empty:
                        break
                
                # Flush batch if any records
                if self.batch:
                    self._flush_batch()
                    
            except Exception as e:
                sys.stderr.write(f"Database log handler error: {e}\n")
    
    def _flush_batch(self):
        """Insert batch of logs to database."""
        if not self.batch:
            return
            
        try:
            with self.db.get_cursor() as cursor:
                # Prepare batch data
                values = []
                for record in self.batch:
                    context = getattr(record, 'context', {})
                    tb = getattr(record, 'traceback', None)
                    perf = getattr(record, 'performance', {})
                    
                    values.append({
                        'experiment_id': context.get('experiment_id'),
                        'node_id': context.get('node_id'),
                        'timestamp': datetime.fromtimestamp(record.created),
                        'level': record.levelname,
                        'logger_name': record.name,
                        'message': record.getMessage(),
                        'context': json.dumps(context),
                        'traceback': tb,
                        'performance': json.dumps(perf) if perf else None
                    })
                
                # Bulk insert
                cursor.executemany("""
                    INSERT INTO pipeline_logs 
                    (experiment_id, node_id, timestamp, level, logger_name, message, context, traceback, performance)
                    VALUES (%(experiment_id)s, %(node_id)s, %(timestamp)s, %(level)s, 
                            %(logger_name)s, %(message)s, %(context)s::jsonb, %(traceback)s, %(performance)s::jsonb)
                """, values)
                
            self.batch.clear()
            
        except Exception as e:
            sys.stderr.write(f"Failed to write logs to database: {e}\n")
            self.batch.clear()  # Prevent memory leak
    
    def close(self):
        """Cleanup handler."""
        self._stop_event.set()
        self._worker_thread.join(timeout=10)
        self._flush_batch()  # Final flush
        super().close()
```

### Phase 3: Context Management (Week 2, Days 1-2)

#### 3.1 Implement Context Manager
`src/infrastructure/logging/context.py`:
```python
"""Logging context management for pipeline execution."""
from contextlib import contextmanager
from typing import Optional, Dict, Any
import uuid

from .structured_logger import experiment_context, node_context, stage_context

class LoggingContext:
    """Manages logging context throughout pipeline execution."""
    
    def __init__(self, experiment_id: Optional[str] = None):
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.node_stack = []
        self.stage_stack = []
    
    @contextmanager
    def pipeline(self, name: str):
        """Context for pipeline execution."""
        node_id = f"pipeline_{name}"
        experiment_context.set(self.experiment_id)
        node_context.set(node_id)
        self.node_stack.append(node_id)
        
        try:
            yield self
        finally:
            self.node_stack.pop()
            if self.node_stack:
                node_context.set(self.node_stack[-1])
            else:
                node_context.set(None)
    
    @contextmanager
    def stage(self, name: str):
        """Context for stage execution."""
        stage_context.set(name)
        self.stage_stack.append(name)
        
        # Create node ID
        parent = self.node_stack[-1] if self.node_stack else "unknown"
        node_id = f"{parent}/{name}"
        node_context.set(node_id)
        self.node_stack.append(node_id)
        
        try:
            yield self
        finally:
            self.stage_stack.pop()
            self.node_stack.pop()
            
            if self.stage_stack:
                stage_context.set(self.stage_stack[-1])
            else:
                stage_context.set(None)
                
            if self.node_stack:
                node_context.set(self.node_stack[-1])
    
    @contextmanager
    def operation(self, name: str, **metadata):
        """Context for specific operations."""
        parent = self.node_stack[-1] if self.node_stack else "unknown"
        node_id = f"{parent}/{name}"
        node_context.set(node_id)
        self.node_stack.append(node_id)
        
        try:
            yield self
        finally:
            self.node_stack.pop()
            if self.node_stack:
                node_context.set(self.node_stack[-1])
```

#### 3.2 Create Logging Decorators
`src/infrastructure/logging/decorators.py`:
```python
"""Decorators for automatic logging and error capture."""
import functools
import time
from typing import Callable, Any

from .structured_logger import get_logger

def log_operation(operation_name: Optional[str] = None):
    """Decorator to log operation execution and capture errors."""
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                logger.info(f"Starting {name}")
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger.log_performance(name, duration, status="success")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Failed {name}: {str(e)}", 
                    exc_info=True,
                    extra={'performance': {'duration': duration, 'status': 'failed'}}
                )
                raise
        
        return wrapper
    return decorator

def log_stage(stage_name: str):
    """Decorator for pipeline stages with automatic context."""
    def decorator(func: Callable) -> Callable:
        logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(self, context, *args, **kwargs) -> Any:
            # Use logging context if available
            if hasattr(context, 'logging_context'):
                with context.logging_context.stage(stage_name):
                    return func(self, context, *args, **kwargs)
            else:
                # Fallback without context
                logger.warning(f"No logging context for stage {stage_name}")
                return func(self, context, *args, **kwargs)
        
        return wrapper
    return decorator
```

### Phase 4: Integration with Existing Systems (Week 2, Days 3-5)

#### 4.1 Enhanced Progress Manager
Create `src/base/monitoring/progress_backend.py`:
```python
"""Progress tracking backend abstraction."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class ProgressBackend(ABC):
    """Abstract backend for progress storage."""
    
    @abstractmethod
    def create_node(self, node_id: str, parent_id: Optional[str], 
                   level: str, name: str, total_units: int) -> None:
        """Create a progress node."""
        pass
    
    @abstractmethod
    def update_progress(self, node_id: str, completed_units: int,
                       status: str, metadata: Optional[Dict] = None) -> None:
        """Update node progress."""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node information."""
        pass
    
    @abstractmethod
    def get_children(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get child nodes."""
        pass
```

Create `src/infrastructure/monitoring/database_progress_backend.py`:
```python
"""Database backend for progress tracking."""
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.base.monitoring.progress_backend import ProgressBackend

class DatabaseProgressBackend(ProgressBackend):
    """PostgreSQL backend for progress tracking."""
    
    def __init__(self, db_manager, experiment_id: str):
        self.db = db_manager
        self.experiment_id = experiment_id
    
    def create_node(self, node_id: str, parent_id: Optional[str], 
                   level: str, name: str, total_units: int) -> None:
        """Create a progress node in database."""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO pipeline_progress 
                (experiment_id, node_id, parent_id, node_level, node_name, total_units, status)
                VALUES (%(exp_id)s, %(node_id)s, %(parent_id)s, %(level)s, %(name)s, %(total)s, 'pending')
                ON CONFLICT (experiment_id, node_id) DO UPDATE
                SET parent_id = EXCLUDED.parent_id,
                    node_name = EXCLUDED.node_name,
                    total_units = EXCLUDED.total_units
            """, {
                'exp_id': self.experiment_id,
                'node_id': node_id,
                'parent_id': parent_id,
                'level': level,
                'name': name,
                'total': total_units
            })
    
    def update_progress(self, node_id: str, completed_units: int,
                       status: str, metadata: Optional[Dict] = None) -> None:
        """Update node progress in database."""
        with self.db.get_cursor() as cursor:
            # Calculate percentage
            cursor.execute("""
                UPDATE pipeline_progress
                SET completed_units = %(completed)s,
                    progress_percent = (%(completed)s::float / NULLIF(total_units, 0)) * 100,
                    status = %(status)s,
                    metadata = metadata || %(metadata)s::jsonb,
                    updated_at = CURRENT_TIMESTAMP,
                    start_time = CASE 
                        WHEN start_time IS NULL AND %(status)s = 'running' 
                        THEN CURRENT_TIMESTAMP 
                        ELSE start_time 
                    END,
                    end_time = CASE 
                        WHEN %(status)s IN ('completed', 'failed', 'cancelled') 
                        THEN CURRENT_TIMESTAMP 
                        ELSE end_time 
                    END
                WHERE experiment_id = %(exp_id)s AND node_id = %(node_id)s
            """, {
                'exp_id': self.experiment_id,
                'node_id': node_id,
                'completed': completed_units,
                'status': status,
                'metadata': json.dumps(metadata or {})
            })
```

#### 4.2 Enhance ProcessController
Modify `src/core/process_controller.py` to add structured logging:
```python
# Add to imports
from src.infrastructure.logging import get_logger, LoggingContext

# In ProcessController.__init__
self.logger = get_logger(__name__)

# In _start_daemon_process method, after fork
# Add structured logging setup
if pid == 0:
    # Child process - setup logging
    from src.infrastructure.logging import setup_daemon_logging
    setup_daemon_logging(name, log_file, self.experiment_id)
    
    # Rest of daemon setup...

# In _monitoring_loop method
# Replace basic logging with structured
if process_info.status == "stopped" and self._auto_restart.get(name, False):
    self.logger.error(
        f"Process '{name}' crashed unexpectedly",
        extra={
            'event_type': 'process_crash',
            'process_name': name,
            'pid': process_info.pid,
            'restart_attempts': attempts
        }
    )
```

#### 4.3 Update PipelineOrchestrator
Modify `src/pipelines/orchestrator.py`:
```python
# Add imports
from src.infrastructure.logging import get_logger, LoggingContext
from src.infrastructure.monitoring import UnifiedMonitor

class PipelineOrchestrator:
    def __init__(self, config, db, signal_handler=None):
        # Existing init...
        self.logger = get_logger(__name__)
        self.monitor = UnifiedMonitor(config, db)
    
    def run_pipeline(self, experiment_name: str, **kwargs):
        """Run pipeline with integrated monitoring and logging."""
        # Create logging context
        logging_ctx = LoggingContext(experiment_id)
        
        with logging_ctx.pipeline(experiment_name):
            try:
                # Start monitoring
                self.monitor.start(experiment_id)
                
                # Existing pipeline logic...
                for stage in self.stages:
                    with logging_ctx.stage(stage.name):
                        self._run_stage(stage, context)
                        
            except Exception as e:
                self.logger.error("Pipeline failed", exc_info=True)
                raise
            finally:
                self.monitor.stop()
```

### Phase 5: CLI Tools and Utilities (Week 3, Days 1-3)

#### 5.1 Enhanced Monitor Script
Create `scripts/monitor.py`:
```python
#!/usr/bin/env python3
"""Unified monitoring and logging CLI."""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import DatabaseManager
from src.infrastructure.monitoring import MonitoringClient
from tabulate import tabulate

class MonitorCLI:
    """CLI for monitoring and log access."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.monitor = MonitoringClient(self.db)
    
    def status(self, args):
        """Show experiment status with progress."""
        status = self.monitor.get_experiment_status(args.experiment)
        
        # Display progress tree
        print(f"\nExperiment: {status['name']} ({status['id'][:8]})")
        print(f"Status: {status['status']}")
        print(f"Started: {status['started_at']}")
        
        print("\nProgress:")
        self._display_progress_tree(status['progress_tree'])
        
        # Show recent logs if errors
        if status['error_count'] > 0:
            print(f"\n⚠️  {status['error_count']} errors found. Use 'monitor logs --level ERROR' to view")
    
    def logs(self, args):
        """Query and display logs."""
        logs = self.monitor.query_logs(
            experiment_id=args.experiment,
            level=args.level,
            search=args.search,
            start_time=args.since,
            limit=args.limit
        )
        
        # Format based on output type
        if args.json:
            import json
            for log in logs:
                print(json.dumps(log))
        else:
            for log in logs:
                # Color based on level
                level_colors = {
                    'ERROR': '\033[91m',
                    'WARNING': '\033[93m',
                    'INFO': '\033[0m',
                    'DEBUG': '\033[90m'
                }
                color = level_colors.get(log['level'], '')
                reset = '\033[0m' if color else ''
                
                print(f"{log['timestamp']} {color}{log['level']:8}{reset} "
                      f"[{log['stage']}] {log['message']}")
                
                if log['traceback'] and args.traceback:
                    print(f"{color}{log['traceback']}{reset}")
    
    def watch(self, args):
        """Live monitoring of experiment."""
        import time
        
        print(f"Watching experiment {args.experiment} (Ctrl+C to stop)")
        
        last_log_id = None
        while True:
            try:
                # Get status
                status = self.monitor.get_experiment_status(args.experiment)
                
                # Clear screen and show status
                print("\033[2J\033[H")  # Clear screen
                self.status(args)
                
                # Show recent logs
                logs = self.monitor.query_logs(
                    experiment_id=args.experiment,
                    after_id=last_log_id,
                    limit=10
                )
                
                if logs:
                    print("\nRecent logs:")
                    for log in logs:
                        print(f"{log['timestamp']} {log['level']:8} {log['message'][:100]}")
                    last_log_id = logs[-1]['id']
                
                # Check if complete
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    print(f"\nExperiment {status['status']}")
                    break
                
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\nStopped watching")
                break
    
    def metrics(self, args):
        """Show performance metrics."""
        metrics = self.monitor.get_metrics(
            experiment_id=args.experiment,
            node_id=args.node,
            metric_type=args.type
        )
        
        # Display as table
        data = []
        for m in metrics:
            data.append([
                m['timestamp'].strftime('%H:%M:%S'),
                m['node_id'].split('/')[-1],
                f"{m['memory_mb']:.1f}",
                f"{m['cpu_percent']:.1f}",
                f"{m.get('throughput_per_sec', 0):.2f}"
            ])
        
        headers = ['Time', 'Node', 'Memory(MB)', 'CPU%', 'Items/sec']
        print(tabulate(data, headers=headers, tablefmt='grid'))
    
    def errors(self, args):
        """Show error summary."""
        errors = self.monitor.get_error_summary(args.experiment)
        
        print(f"\nError Summary for {args.experiment}:")
        print(f"Total Errors: {errors['total_count']}")
        
        if errors['by_stage']:
            print("\nErrors by Stage:")
            for stage, count in errors['by_stage'].items():
                print(f"  {stage}: {count}")
        
        if errors['recent_errors']:
            print("\nRecent Errors:")
            for err in errors['recent_errors'][:5]:
                print(f"\n{err['timestamp']} [{err['stage']}]")
                print(f"  {err['message']}")
                if args.traceback and err['traceback']:
                    print(f"  Traceback: {err['traceback'][:200]}...")

def main():
    parser = argparse.ArgumentParser(description="Unified monitoring and logging")
    subparsers = parser.add_subparsers(dest='command')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show experiment status')
    status_parser.add_argument('experiment', help='Experiment name or ID')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Query logs')
    logs_parser.add_argument('experiment', help='Experiment name or ID')
    logs_parser.add_argument('--level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    logs_parser.add_argument('--search', help='Search in messages')
    logs_parser.add_argument('--since', help='Time filter (e.g., "1h", "30m")')
    logs_parser.add_argument('--limit', type=int, default=100)
    logs_parser.add_argument('--json', action='store_true', help='JSON output')
    logs_parser.add_argument('--traceback', action='store_true', help='Show tracebacks')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Live monitoring')
    watch_parser.add_argument('experiment', help='Experiment name or ID')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show metrics')
    metrics_parser.add_argument('experiment', help='Experiment name or ID')
    metrics_parser.add_argument('--node', help='Filter by node')
    metrics_parser.add_argument('--type', choices=['memory', 'cpu', 'throughput'])
    
    # Errors command
    errors_parser = subparsers.add_parser('errors', help='Error summary')
    errors_parser.add_argument('experiment', help='Experiment name or ID')
    errors_parser.add_argument('--traceback', action='store_true')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = MonitorCLI()
    commands = {
        'status': cli.status,
        'logs': cli.logs,
        'watch': cli.watch,
        'metrics': cli.metrics,
        'errors': cli.errors
    }
    
    try:
        commands[args.command](args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#### 5.2 Update tmux Scripts
Enhance `run_analysis.sh` to include monitoring pane:
```bash
#!/bin/bash
# Add monitoring pane to tmux layout

# Create new window for monitoring
tmux new-window -t $SESSION:2 -n "monitoring"

# Split for different monitoring views
tmux split-window -h -t $SESSION:2
tmux split-window -v -t $SESSION:2.0

# Pane 2.0: Live status
tmux send-keys -t $SESSION:2.0 "python scripts/monitor.py watch $EXPERIMENT_NAME" C-m

# Pane 2.1: Error logs
tmux send-keys -t $SESSION:2.1 "python scripts/monitor.py logs $EXPERIMENT_NAME --level ERROR -f" C-m

# Pane 2.2: Metrics
tmux send-keys -t $SESSION:2.2 "watch -n 5 'python scripts/monitor.py metrics $EXPERIMENT_NAME --type memory | tail -20'" C-m
```

### Phase 6: Testing and Migration (Week 3, Days 4-5)

#### 6.1 Create Test Suite
`tests/test_unified_monitoring.py`:
```python
"""Tests for unified monitoring and logging system."""
import pytest
import time
from unittest.mock import Mock, patch

from src.infrastructure.logging import get_logger, LoggingContext
from src.infrastructure.monitoring import UnifiedMonitor

class TestStructuredLogging:
    """Test structured logging functionality."""
    
    def test_context_propagation(self, db_mock):
        """Test that context propagates through operations."""
        logger = get_logger("test")
        ctx = LoggingContext("test-exp-123")
        
        with ctx.pipeline("test_pipeline"):
            with ctx.stage("test_stage"):
                # Log should have context
                with patch.object(logger, '_log') as mock_log:
                    logger.info("Test message")
                    
                    call_args = mock_log.call_args
                    extra = call_args[1]['extra']
                    
                    assert extra['context']['experiment_id'] == "test-exp-123"
                    assert extra['context']['stage'] == "test_stage"
                    assert 'pipeline' in extra['context']['node_id']
    
    def test_error_capture(self, db_mock):
        """Test error and traceback capture."""
        logger = get_logger("test")
        
        try:
            raise ValueError("Test error")
        except ValueError:
            with patch.object(logger, '_log') as mock_log:
                logger.error("Caught error", exc_info=True)
                
                call_args = mock_log.call_args
                extra = call_args[1]['extra']
                
                assert extra['traceback'] is not None
                assert "ValueError: Test error" in extra['traceback']
    
    def test_performance_logging(self):
        """Test performance metric logging."""
        logger = get_logger("test")
        
        with patch.object(logger, '_log') as mock_log:
            logger.log_performance("test_operation", 1.23, items_processed=100)
            
            call_args = mock_log.call_args
            extra = call_args[1]['extra']
            
            assert extra['performance']['operation'] == "test_operation"
            assert extra['performance']['duration_seconds'] == 1.23
            assert extra['performance']['items_processed'] == 100

class TestDatabaseBackend:
    """Test database backend for monitoring."""
    
    @pytest.fixture
    def backend(self, test_db):
        """Create test backend."""
        from src.infrastructure.monitoring import DatabaseProgressBackend
        return DatabaseProgressBackend(test_db, "test-exp-123")
    
    def test_create_and_update_node(self, backend):
        """Test node creation and updates."""
        # Create node
        backend.create_node(
            node_id="pipeline/stage1",
            parent_id="pipeline",
            level="stage",
            name="Stage 1",
            total_units=100
        )
        
        # Update progress
        backend.update_progress(
            node_id="pipeline/stage1",
            completed_units=50,
            status="running"
        )
        
        # Verify
        node = backend.get_node("pipeline/stage1")
        assert node['progress_percent'] == 50.0
        assert node['status'] == "running"
    
    def test_hierarchical_progress(self, backend):
        """Test hierarchical progress aggregation."""
        # Create parent and children
        backend.create_node("pipeline", None, "pipeline", "Test Pipeline", 100)
        backend.create_node("pipeline/s1", "pipeline", "stage", "Stage 1", 50)
        backend.create_node("pipeline/s2", "pipeline", "stage", "Stage 2", 50)
        
        # Update children
        backend.update_progress("pipeline/s1", 50, "completed")
        backend.update_progress("pipeline/s2", 25, "running")
        
        # Parent should aggregate
        parent = backend.get_node("pipeline")
        assert parent['progress_percent'] == 75.0  # (50 + 25) / 100

class TestProcessIntegration:
    """Test integration with process management."""
    
    def test_daemon_logging(self, tmp_path):
        """Test that daemon processes use structured logging."""
        from src.core.process_controller import ProcessController
        
        controller = ProcessController(
            pid_dir=tmp_path / "pid",
            log_dir=tmp_path / "logs"
        )
        
        # Start test daemon
        pid = controller.start_process(
            name="test_daemon",
            command=["python", "-c", "import time; time.sleep(1)"],
            daemon_mode=True
        )
        
        # Check log file has structured format
        log_file = tmp_path / "logs" / "test_daemon.log"
        assert log_file.exists()
        
        # Logs should be JSON formatted
        time.sleep(0.5)
        content = log_file.read_text()
        if content:
            import json
            # Each line should be valid JSON
            for line in content.strip().split('\n'):
                if line:
                    data = json.loads(line)
                    assert 'timestamp' in data
                    assert 'level' in data
```

#### 6.2 Migration Guide
Create `MIGRATION_GUIDE.md`:
```markdown
# Migration Guide: Unified Monitoring and Logging

## For Existing Code

### 1. Replace Basic Logging
```python
# Old
import logging
logger = logging.getLogger(__name__)

# New
from src.infrastructure.logging import get_logger
logger = get_logger(__name__)
```

### 2. Add Context to Operations
```python
# In pipeline stages
@log_stage("data_loading")
def execute(self, context):
    # Automatic context and error handling
    pass
```

### 3. Log Performance Metrics
```python
# Replace manual timing
start = time.time()
# ... operation ...
logger.info(f"Took {time.time() - start}s")

# With
logger.log_performance("operation_name", duration, items=count)
```

## For New Development

1. Always use structured logger
2. Use decorators for automatic instrumentation
3. Include context in error messages
4. Log performance metrics for key operations

## Viewing Logs

```bash
# Live monitoring
python scripts/monitor.py watch my_experiment

# Query specific errors
python scripts/monitor.py logs my_experiment --level ERROR --since 1h

# View performance metrics
python scripts/monitor.py metrics my_experiment

# Error summary
python scripts/monitor.py errors my_experiment --traceback
```
```

### Phase 7: Deployment and Rollout (Week 4)

#### 7.1 Deployment Steps

1. **Database Migration**
   ```bash
   python scripts/migrate_monitoring_schema.py
   ```

2. **Update Configuration**
   Add to `config.yml`:
   ```yaml
   monitoring:
     enable_database_logging: true
     log_batch_size: 100
     log_flush_interval: 5
     enable_metrics: true
     metrics_interval: 10
   ```

3. **Gradual Rollout**
   - Week 1: Deploy infrastructure, test with new experiments
   - Week 2: Update critical paths (error handling)
   - Week 3: Migrate existing components
   - Week 4: Full deployment, deprecate old logging

4. **Monitoring the Monitoring**
   - Set up alerts for log queue overflow
   - Monitor database table growth
   - Track query performance

## Benefits Summary

1. **Debugging Improvements**
   - Full traceback capture for daemon crashes
   - Context-aware error messages
   - Queryable logs by experiment/stage/time

2. **Performance Visibility**
   - Automatic operation timing
   - Resource usage correlation
   - Bottleneck identification

3. **Operational Excellence**
   - Live experiment monitoring
   - Historical analysis
   - Error pattern detection

4. **Developer Experience**
   - Simple decorators for instrumentation
   - Automatic context propagation
   - Rich CLI tools

## Success Metrics

1. **Reduced Debug Time**: 80% reduction in time to identify crash causes
2. **Error Detection**: 100% of errors captured with full context
3. **Performance**: <1% overhead from logging
4. **Adoption**: 100% of pipeline operations instrumented

## Next Steps

1. Implement Phase 1 (Database Schema)
2. Deploy basic structured logging
3. Test with one pipeline stage
4. Gradually expand coverage
5. Deprecate print() debugging

This unified system provides the observability needed for complex distributed pipeline execution while maintaining compatibility with existing code.