# Monitoring Dashboard Guide

## Overview

The monitoring system provides multiple ways to visualize and track pipeline execution. This guide covers the available dashboards and monitoring interfaces.

## 1. tmux Monitoring Dashboard

When running pipelines with `./run_monitored.sh`, a comprehensive tmux dashboard is created:

### Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Window 0: pipeline - Main pipeline execution                 │
├─────────────────────────────────────────────────────────────┤
│ Window 1: monitoring                                         │
│ ┌─────────────────────┬─────────────────────┐              │
│ │ Live Progress       │ Resource Metrics     │              │
│ │ (watch status)      │ (memory usage)       │              │
│ ├─────────────────────┴─────────────────────┤              │
│ │ Error Monitoring                           │              │
│ │ (error count and recent errors)           │              │
│ └───────────────────────────────────────────┘              │
├─────────────────────────────────────────────────────────────┤
│ Window 2: logs                                               │
│ ┌─────────────────────┬─────────────────────┐              │
│ │ All Logs           │ Error Logs Only      │              │
│ │                    │ (with tracebacks)    │              │
│ └─────────────────────┴─────────────────────┘              │
├─────────────────────────────────────────────────────────────┤
│ Window 3: system - System resource monitoring                │
├─────────────────────────────────────────────────────────────┤
│ Window 4: control - Command reference panel                  │
└─────────────────────────────────────────────────────────────┘
```

### Navigation

- **Switch windows**: `Ctrl+b [0-4]`
- **Switch panes**: `Ctrl+b arrow_keys`
- **Detach session**: `Ctrl+b d`
- **Scroll in pane**: `Ctrl+b [` then use arrow keys, `q` to exit

## 2. CLI Monitoring Commands

### Real-time Status Dashboard

```bash
python scripts/monitor.py watch my_experiment
```

Shows:
- Experiment status and runtime
- Progress tree with completion percentages
- Recent log entries
- Resource metrics
- Error counts

**Example output:**
```
Experiment: my_experiment - Status: RUNNING
Running for: 0:05:23
================================================================================
Overall Progress: 45.2% (✅ 23 | 🔄 2 | ❌ 0 | ⏳ 26)
================================================================================

Recent logs:
14:23:45 ✓  Starting stage: data_processing
14:23:46 ✓  Loaded 1000 records from database
14:23:47 ⚠️  Warning: Missing values in column 'species_count'
14:23:48 ✓  Data validation completed

Current Resources:
  Memory: 1024.5 MB
  CPU: 75.3%
  Throughput: 150.2 items/sec
```

### Progress Tree View

```bash
python scripts/monitor.py status my_experiment
```

Shows hierarchical progress:
```
Experiment: biodiversity_analysis (a1b2c3d4)
Status: RUNNING
Started: 2024-01-20 14:20:00
==============================================================

Progress:
✅ Pipeline: [████████████████░░░░] 80.0%
  ✅ data_loading: [████████████████████] 100.0%
  🔄 data_processing: [████████░░░░░░░░░░░░] 40.0%
    ✅ validation: [████████████████████] 100.0%
    🔄 normalization: [████░░░░░░░░░░░░░░░░] 20.0%
    ⏳ aggregation: [░░░░░░░░░░░░░░░░░░░░] 0.0%
  ⏳ analysis: [░░░░░░░░░░░░░░░░░░░░] 0.0%
  ⏳ output_generation: [░░░░░░░░░░░░░░░░░░░░] 0.0%
```

### Log Viewer

```bash
# All logs
python scripts/monitor.py logs my_experiment

# Filter by level
python scripts/monitor.py logs my_experiment --level ERROR

# Search in logs
python scripts/monitor.py logs my_experiment --search "species_data"

# Recent logs (last hour)
python scripts/monitor.py logs my_experiment --since 1h

# With full tracebacks
python scripts/monitor.py logs my_experiment --level ERROR --traceback
```

### Performance Metrics

```bash
python scripts/monitor.py metrics my_experiment
```

Shows tabulated metrics:
```
┌──────────┬─────────────┬────────────┬──────┬────────────┐
│ Time     │ Node        │ Memory(MB) │ CPU% │ Items/sec  │
├──────────┼─────────────┼────────────┼──────┼────────────┤
│ 14:20:15 │ data_load   │ 512.3      │ 45.2 │ 1250.5     │
│ 14:20:25 │ data_load   │ 623.1      │ 52.1 │ 1180.3     │
│ 14:20:35 │ processing  │ 1024.5     │ 78.9 │ 523.7      │
│ 14:20:45 │ processing  │ 1156.2     │ 81.3 │ 498.2      │
└──────────┴─────────────┴────────────┴──────┴────────────┘
```

### Error Summary

```bash
python scripts/monitor.py errors my_experiment --traceback
```

Shows:
```
Error Summary for biodiversity_analysis
============================================================
Total Errors: 3

Errors by Level:
  ERROR: 2
  CRITICAL: 1

Errors by Stage:
  data_processing: 2
  analysis: 1

Recent Errors:
------------------------------------------------------------

1. [2024-01-20 14:25:33] Stage: data_processing
   Database connection timeout during batch insert

   Traceback:
   File "processors/data_loader.py", line 145, in batch_insert
     cursor.executemany(sql, values)
   psycopg2.OperationalError: connection timeout
   ...

2. [2024-01-20 14:26:45] Stage: analysis
   Memory allocation failed for large matrix operation
   ...
```

## 3. Database Queries for Custom Views

### Active Experiments

```sql
-- List active experiments with progress
SELECT 
    e.name,
    e.status,
    e.started_at,
    COUNT(DISTINCT p.node_id) as total_nodes,
    COUNT(DISTINCT p.node_id) FILTER (WHERE p.status = 'completed') as completed_nodes,
    ROUND(AVG(p.progress_percent), 1) as avg_progress
FROM experiments e
LEFT JOIN pipeline_progress p ON e.id = p.experiment_id
WHERE e.status = 'running'
GROUP BY e.id, e.name, e.status, e.started_at
ORDER BY e.started_at DESC;
```

### Error Analysis

```sql
-- Error frequency by hour
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as error_count,
    COUNT(DISTINCT experiment_id) as affected_experiments
FROM pipeline_logs
WHERE level IN ('ERROR', 'CRITICAL')
    AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;
```

### Performance Trends

```sql
-- Average memory usage by stage
SELECT 
    SUBSTRING(node_id FROM '[^/]+/([^/]+)') as stage,
    ROUND(AVG(memory_mb), 1) as avg_memory_mb,
    ROUND(MAX(memory_mb), 1) as max_memory_mb,
    ROUND(AVG(cpu_percent), 1) as avg_cpu_percent
FROM pipeline_metrics
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY stage
ORDER BY avg_memory_mb DESC;
```

## 4. Grafana Dashboard (Optional)

For production environments, you can set up Grafana dashboards:

### PostgreSQL Data Source

Add PostgreSQL as a data source in Grafana with connection to your monitoring tables.

### Example Dashboard Panels

1. **Experiment Status Overview**
   - Pie chart of experiment statuses
   - Running experiments list
   - Success/failure rates

2. **Performance Metrics**
   - Time series of memory usage
   - CPU utilization graphs
   - Throughput trends

3. **Error Tracking**
   - Error rate over time
   - Error heatmap by stage
   - Recent errors table

4. **Progress Tracking**
   - Progress bars for active experiments
   - Stage completion times
   - Estimated time remaining

### Sample Grafana Query

```sql
-- For time series memory usage
SELECT
  timestamp AS time,
  node_id AS metric,
  memory_mb AS value
FROM pipeline_metrics
WHERE 
  experiment_id = '$experiment_id'
  AND timestamp BETWEEN $__timeFrom() AND $__timeTo()
ORDER BY time
```

## 5. Alerting Setup

### Database Triggers

Create triggers for critical events:

```sql
-- Alert on high error rate
CREATE OR REPLACE FUNCTION check_error_rate()
RETURNS trigger AS $$
DECLARE
    error_count INTEGER;
BEGIN
    -- Count errors in last 5 minutes
    SELECT COUNT(*) INTO error_count
    FROM pipeline_logs
    WHERE level IN ('ERROR', 'CRITICAL')
        AND timestamp > NOW() - INTERVAL '5 minutes';
    
    IF error_count > 50 THEN
        -- Log alert
        INSERT INTO pipeline_events (
            experiment_id, event_type, severity, 
            title, details
        ) VALUES (
            NEW.experiment_id, 'alert', 'critical',
            'High error rate detected',
            jsonb_build_object('error_count', error_count)
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER error_rate_monitor
AFTER INSERT ON pipeline_logs
FOR EACH ROW
WHEN (NEW.level IN ('ERROR', 'CRITICAL'))
EXECUTE FUNCTION check_error_rate();
```

### Email Alerts (Future Enhancement)

```python
# Example alert configuration
alerts:
  email:
    enabled: true
    smtp_server: smtp.example.com
    recipients:
      - ops-team@example.com
    
  rules:
    - name: high_error_rate
      condition: "error_count > 10 per minute"
      severity: critical
      
    - name: memory_threshold
      condition: "memory_mb > 8192"
      severity: warning
      
    - name: pipeline_stuck
      condition: "no progress for 30 minutes"
      severity: warning
```

## 6. Best Practices

1. **Regular Monitoring**
   - Check active experiments daily
   - Review error summaries
   - Monitor resource trends

2. **Dashboard Usage**
   - Use tmux for active monitoring
   - Use CLI for investigations
   - Set up Grafana for trends

3. **Alerting**
   - Configure alerts for critical issues
   - Review and tune thresholds
   - Document response procedures

4. **Performance**
   - Monitor database table growth
   - Archive old logs periodically
   - Index frequently queried fields

## 7. Troubleshooting Dashboard Issues

### tmux Session Not Created
```bash
# Check if tmux is installed
which tmux

# List existing sessions
tmux ls

# Kill stuck session
tmux kill-session -t <session_name>
```

### Monitor Command Not Working
```bash
# Validate setup
python scripts/validate_monitoring_setup.py

# Check database connection
python -c "from src.database.connection import DatabaseManager; DatabaseManager()"
```

### Missing Data in Dashboard
```sql
-- Check if logs are being written
SELECT COUNT(*), MAX(timestamp) 
FROM pipeline_logs 
WHERE experiment_id = 'your_experiment_id';

-- Check monitoring is enabled
SELECT * FROM experiments 
WHERE name = 'your_experiment_name';
```

The monitoring dashboard provides comprehensive visibility into pipeline execution, making it easy to identify issues, track progress, and optimize performance.