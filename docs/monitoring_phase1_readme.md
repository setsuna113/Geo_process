# Phase 1: Database Schema Extensions - Implementation Guide

## Overview
This phase implements the database schema for the unified monitoring and logging system. It adds structured logging tables, progress tracking, and metrics collection to enable better debugging of daemon processes.

## Files Created

1. **`src/database/monitoring_schema.sql`**
   - Defines all monitoring tables, views, and functions
   - Tables: `pipeline_logs`, `pipeline_events`, `pipeline_progress`, `pipeline_metrics`
   - Views for easy querying: recent errors, progress summary, active monitoring
   - Functions for log queries, progress trees, and error summaries

2. **`scripts/migrate_monitoring_schema.py`**
   - Migration script to apply schema changes
   - Checks existing tables before migration
   - Options for force migration, test data creation, and sample queries

3. **`scripts/test_monitoring_schema.py`**
   - Test script to verify schema functionality
   - Creates test data and runs sample queries
   - Demonstrates all monitoring features

## How to Deploy

### 1. First, check what exists:
```bash
python scripts/migrate_monitoring_schema.py --check-only
```

### 2. Run the migration:
```bash
python scripts/migrate_monitoring_schema.py
```

### 3. (Optional) Run with test data:
```bash
# Get an experiment ID first
psql -h localhost -p 51051 -U jason -d geo_cluster_db -c "SELECT id, name FROM experiments LIMIT 5;"

# Run migration with test data
python scripts/migrate_monitoring_schema.py --test-data <experiment_id>
```

### 4. Verify the schema:
```bash
python scripts/test_monitoring_schema.py
```

### 5. View sample queries:
```bash
python scripts/migrate_monitoring_schema.py --show-queries
```

## Key Features

### 1. Structured Logging (`pipeline_logs`)
- JSON context with experiment_id, node_id, stage
- Full traceback capture for errors
- Performance metrics in logs
- Indexed for fast queries

### 2. Progress Tracking (`pipeline_progress`)
- Hierarchical structure (pipeline → phase → step → substep)
- Automatic parent progress aggregation
- Real-time status updates
- Unique constraint on (experiment_id, node_id)

### 3. Resource Metrics (`pipeline_metrics`)
- CPU, memory, disk usage tracking
- Throughput measurements
- Custom metrics support
- Time-series data for analysis

### 4. Event Tracking (`pipeline_events`)
- Significant pipeline events
- Severity levels
- Structured details in JSONB

## Query Examples

### Get recent errors:
```sql
SELECT * FROM recent_experiment_errors 
WHERE experiment_name = 'your_experiment' 
LIMIT 10;
```

### Get progress tree:
```sql
SELECT * FROM get_progress_tree('experiment_uuid'::uuid);
```

### Query logs with filters:
```sql
SELECT * FROM get_experiment_logs(
    'experiment_uuid'::uuid,
    p_level := 'ERROR',
    p_search := 'memory',
    p_limit := 50
);
```

### View active monitoring:
```sql
SELECT * FROM active_pipeline_monitoring;
```

## Next Steps

After Phase 1 is deployed and tested:

1. **Phase 2**: Implement structured logging infrastructure
   - Create `src/infrastructure/logging/` module
   - Implement StructuredLogger with context propagation
   - Add database handler for async log writes

2. **Phase 3**: Context management
   - Implement LoggingContext for correlation
   - Add decorators for automatic instrumentation

3. **Phase 4**: Integration with existing systems
   - Update ProcessController for structured logging
   - Enhance PipelineOrchestrator with monitoring

## Troubleshooting

### Migration fails with permission errors:
- Ensure database user has CREATE TABLE permissions
- Check PostGIS extension is installed

### Tables already exist:
- Use `--force` flag to drop and recreate (DANGEROUS in production!)
- Or manually drop specific tables and re-run

### Test script fails:
- Ensure an experiment exists in the database
- Check database connection settings in config.yml

## Database Maintenance

### Clean up old logs (after 30 days):
```sql
SELECT cleanup_old_logs(30);
```

### Check table sizes:
```sql
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE tablename LIKE 'pipeline_%'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## Success Metrics

- ✅ All 4 monitoring tables created
- ✅ All 3 monitoring views created  
- ✅ All 5 utility functions created
- ✅ Test data insertion working
- ✅ Progress aggregation trigger working
- ✅ Query functions returning results

Phase 1 provides the foundation for the unified monitoring and logging system. Once deployed, you'll have persistent storage for all pipeline execution data, enabling better debugging of daemon processes and hanging operations.