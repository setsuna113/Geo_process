# Performance Tuning Guide for Monitoring System

## Overview

This guide helps optimize the monitoring and logging system for different workloads and environments. The system is designed to have minimal overhead while providing comprehensive visibility.

## Performance Baseline

Expected overhead with default settings:
- **CPU**: < 1% additional usage
- **Memory**: 50-100 MB for logging buffers
- **Disk I/O**: Batched writes every 5 seconds
- **Network**: Minimal (database connection pooling)
- **Overall Impact**: < 2% on pipeline performance

## Key Performance Parameters

### 1. Logging Configuration

```yaml
# config.yml
logging:
  level: INFO  # Use WARNING or ERROR for less overhead
  
  database:
    enabled: true
    batch_size: 100  # Increase for high-volume logging
    flush_interval: 5.0  # Increase to reduce write frequency
    max_queue_size: 10000  # Prevent memory overflow
    
  file:
    enabled: true
    max_size_mb: 100  # Rotate before files get too large
    backup_count: 5  # Limit disk usage
    
  # Performance optimizations
  async_mode: true  # Non-blocking logging
  compression: true  # Compress rotated logs
```

### 2. Monitoring Configuration

```yaml
monitoring:
  enabled: true
  
  # Sampling intervals (seconds)
  metrics_interval: 10.0  # Increase for less frequent sampling
  progress_update_interval: 5.0  # Balance between freshness and overhead
  
  # Resource tracking
  enable_resource_tracking: true
  resource_sample_interval: 30.0  # CPU/memory sampling
  
  # Aggregation settings
  aggregate_metrics: true  # Reduce data volume
  aggregation_window: 60  # Seconds
```

### 3. Database Optimization

```yaml
monitoring:
  database:
    # Connection pooling
    connection_pool_size: 10  # Adjust based on concurrency
    connection_timeout: 30
    
    # Batch operations
    max_batch_size: 1000  # Larger batches for bulk inserts
    batch_timeout: 10.0  # Maximum wait before flush
    
    # Query optimization
    use_prepared_statements: true
    enable_query_cache: true
```

## Optimization Strategies

### 1. High-Volume Logging Scenarios

For pipelines generating >1000 logs/second:

```python
# Increase batch sizes
config.set('logging.database.batch_size', 500)
config.set('logging.database.flush_interval', 10.0)

# Use sampling for debug logs
from src.infrastructure.logging import get_logger
logger = get_logger(__name__)

# Sample debug logs
import random
if random.random() < 0.1:  # Log 10% of debug messages
    logger.debug("Detailed debug info")
```

### 2. Memory-Constrained Environments

For systems with limited memory:

```yaml
logging:
  database:
    batch_size: 50  # Smaller batches
    max_queue_size: 1000  # Limit queue size
    
monitoring:
  # Disable resource tracking to save memory
  enable_resource_tracking: false
  
  # Longer intervals
  metrics_interval: 30.0
  progress_update_interval: 15.0
```

### 3. Network-Constrained Environments

For slow database connections:

```python
# Use local file logging with periodic sync
config.set('logging.database.enabled', False)
config.set('logging.file.enabled', True)

# Implement periodic sync to database
from src.infrastructure.logging.sync import LogSynchronizer
synchronizer = LogSynchronizer(
    sync_interval=300,  # 5 minutes
    batch_size=1000
)
synchronizer.start()
```

### 4. CPU-Intensive Pipelines

Minimize monitoring overhead:

```yaml
# Reduce sampling frequency
monitoring:
  metrics_interval: 60.0  # Once per minute
  enable_resource_tracking: false  # Disable CPU sampling
  
logging:
  # Use higher log levels
  level: WARNING  # Skip INFO and DEBUG
  
  # Disable synchronous writes
  database:
    async_mode: true
    wait_for_flush: false
```

## Database Tuning

### 1. Indexes

Ensure optimal query performance:

```sql
-- Already created by migration, but verify:
CREATE INDEX IF NOT EXISTS idx_logs_experiment_time 
ON pipeline_logs(experiment_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_logs_level_time 
ON pipeline_logs(level, timestamp DESC) 
WHERE level IN ('ERROR', 'CRITICAL');

CREATE INDEX IF NOT EXISTS idx_progress_experiment_node 
ON pipeline_progress(experiment_id, node_id);

CREATE INDEX IF NOT EXISTS idx_metrics_experiment_time 
ON pipeline_metrics(experiment_id, timestamp DESC);

-- Additional indexes for common queries
CREATE INDEX IF NOT EXISTS idx_logs_logger_time 
ON pipeline_logs(logger_name, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_logs_search 
ON pipeline_logs USING gin(to_tsvector('english', message));
```

### 2. Partitioning

For very large deployments, partition tables by time:

```sql
-- Convert pipeline_logs to partitioned table
CREATE TABLE pipeline_logs_partitioned (
    LIKE pipeline_logs INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE pipeline_logs_2024_01 
PARTITION OF pipeline_logs_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Automated partition creation
CREATE OR REPLACE FUNCTION create_monthly_partition()
RETURNS void AS $$
DECLARE
    start_date date;
    end_date date;
    partition_name text;
BEGIN
    start_date := date_trunc('month', CURRENT_DATE);
    end_date := start_date + interval '1 month';
    partition_name := 'pipeline_logs_' || to_char(start_date, 'YYYY_MM');
    
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF pipeline_logs_partitioned FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date
    );
END;
$$ LANGUAGE plpgsql;

-- Schedule monthly
CREATE EXTENSION IF NOT EXISTS pg_cron;
SELECT cron.schedule('create-partitions', '0 0 1 * *', 'SELECT create_monthly_partition()');
```

### 3. Vacuum and Maintenance

```sql
-- Aggressive autovacuum for monitoring tables
ALTER TABLE pipeline_logs SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

ALTER TABLE pipeline_metrics SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

-- Regular maintenance
CREATE OR REPLACE FUNCTION monitoring_maintenance()
RETURNS void AS $$
BEGIN
    -- Delete old logs
    DELETE FROM pipeline_logs 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    -- Archive old metrics
    INSERT INTO pipeline_metrics_archive
    SELECT * FROM pipeline_metrics
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '7 days';
    
    DELETE FROM pipeline_metrics
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '7 days';
    
    -- Update statistics
    ANALYZE pipeline_logs;
    ANALYZE pipeline_metrics;
END;
$$ LANGUAGE plpgsql;

-- Schedule daily
SELECT cron.schedule('monitoring-maintenance', '0 2 * * *', 'SELECT monitoring_maintenance()');
```

## Application-Level Optimizations

### 1. Lazy Logging

```python
# Use lazy evaluation for expensive operations
logger.debug("Processing %s", expensive_function)  # Bad
logger.debug("Processing %s", lambda: expensive_function())  # Good

# Or check level first
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Detailed info: %s", expensive_debug_info())
```

### 2. Context Caching

```python
# Cache frequently used context
from src.infrastructure.logging import get_logger, LoggingContext

class OptimizedStage:
    def __init__(self, context):
        self.context = context
        self.logger = get_logger(__name__)
        # Pre-compute context
        self.logger.add_context(
            stage=self.__class__.__name__,
            version=self.version
        )
    
    def process_batch(self, items):
        # Reuse context for all items
        for item in items:
            self.logger.info("Processing item", extra={'item_id': item.id})
```

### 3. Bulk Operations

```python
# Instead of logging each item
for item in large_dataset:
    logger.info(f"Processed {item}")  # Bad

# Log summaries
processed = 0
for item in large_dataset:
    process(item)
    processed += 1
    if processed % 1000 == 0:
        logger.info(f"Processed {processed} items")  # Good
```

### 4. Conditional Monitoring

```python
# Disable monitoring for specific operations
from src.infrastructure.monitoring import monitoring_context

# Temporarily disable
with monitoring_context.disabled():
    # CPU-intensive operation without monitoring overhead
    result = expensive_computation()

# Or conditionally enable
if config.get('monitoring.detailed_metrics'):
    monitor.record_metrics(detailed=True)
else:
    monitor.record_metrics(basic=True)
```

## Monitoring the Monitoring System

### 1. Performance Metrics

```sql
-- Monitor logging performance
SELECT 
    DATE_TRUNC('minute', created_at) as minute,
    COUNT(*) as logs_per_minute,
    AVG(LENGTH(message)) as avg_message_size,
    MAX(LENGTH(traceback)) as max_traceback_size
FROM pipeline_logs
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY minute
ORDER BY minute DESC;

-- Check batch efficiency
WITH batch_stats AS (
    SELECT 
        DATE_TRUNC('minute', created_at) as minute,
        COUNT(*) as records,
        COUNT(DISTINCT created_at) as unique_timestamps
    FROM pipeline_logs
    WHERE created_at > NOW() - INTERVAL '1 hour'
    GROUP BY minute
)
SELECT 
    minute,
    records,
    unique_timestamps,
    ROUND(records::numeric / NULLIF(unique_timestamps, 0), 1) as avg_batch_size
FROM batch_stats
ORDER BY minute DESC;
```

### 2. Resource Usage

```python
# Monitor monitoring overhead
import psutil
import time

class MonitoringProfiler:
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_cpu = None
        self.baseline_memory = None
    
    def start_profiling(self):
        """Establish baseline without monitoring."""
        self.baseline_cpu = self.process.cpu_percent(interval=1)
        self.baseline_memory = self.process.memory_info().rss
    
    def measure_overhead(self):
        """Measure monitoring overhead."""
        current_cpu = self.process.cpu_percent(interval=1)
        current_memory = self.process.memory_info().rss
        
        cpu_overhead = current_cpu - self.baseline_cpu
        memory_overhead = (current_memory - self.baseline_memory) / 1024 / 1024
        
        return {
            'cpu_overhead_percent': cpu_overhead,
            'memory_overhead_mb': memory_overhead
        }
```

## Troubleshooting Performance Issues

### 1. High CPU Usage

**Symptoms**: Monitoring threads consuming significant CPU

**Solutions**:
- Increase sampling intervals
- Disable resource tracking
- Reduce log level
- Check for log loops

### 2. Memory Growth

**Symptoms**: Increasing memory usage over time

**Solutions**:
- Reduce queue sizes
- Enable log rotation
- Check for memory leaks in context
- Limit batch sizes

### 3. Database Bottlenecks

**Symptoms**: Slow queries, connection timeouts

**Solutions**:
- Add missing indexes
- Increase connection pool
- Enable prepared statements
- Implement partitioning

### 4. Disk I/O Spikes

**Symptoms**: High disk write activity

**Solutions**:
- Increase batch intervals
- Enable compression
- Rotate logs more frequently
- Use separate disk for logs

## Best Practices Summary

1. **Start with defaults** - Only tune when needed
2. **Monitor the monitors** - Track overhead
3. **Use appropriate log levels** - INFO for production
4. **Batch operations** - Reduce database load
5. **Archive old data** - Keep tables manageable
6. **Index strategically** - Based on query patterns
7. **Profile before optimizing** - Measure impact
8. **Document changes** - Track tuning decisions

## Configuration Templates

### Minimal Overhead Configuration

```yaml
# For maximum performance, minimal monitoring
logging:
  level: WARNING
  database:
    enabled: false
  file:
    enabled: true
    
monitoring:
  enabled: false
```

### Balanced Configuration

```yaml
# Good balance of visibility and performance
logging:
  level: INFO
  database:
    batch_size: 100
    flush_interval: 5.0
    
monitoring:
  metrics_interval: 30.0
  progress_update_interval: 10.0
```

### Full Monitoring Configuration

```yaml
# Maximum visibility for debugging
logging:
  level: DEBUG
  database:
    batch_size: 50
    flush_interval: 2.0
    
monitoring:
  metrics_interval: 5.0
  progress_update_interval: 2.0
  enable_resource_tracking: true
```

Remember: The best configuration depends on your specific use case, infrastructure, and requirements. Always test changes in a non-production environment first.