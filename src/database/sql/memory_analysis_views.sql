-- Memory Analysis Views for Pipeline Monitoring
-- These views provide insights into memory usage patterns during pipeline execution

-- Drop existing views if they exist
DROP VIEW IF EXISTS v_memory_timeline CASCADE;
DROP VIEW IF EXISTS v_memory_by_stage CASCADE;
DROP VIEW IF EXISTS v_memory_peaks CASCADE;
DROP VIEW IF EXISTS v_memory_pressure_events CASCADE;

-- Memory usage by stage
-- Shows average, peak, and min memory usage for each stage in each experiment
CREATE OR REPLACE VIEW v_memory_by_stage AS
SELECT 
    experiment_id,
    custom_metrics->>'stage' as stage,
    AVG(memory_mb / 1024.0) as avg_memory_gb,
    MAX(memory_mb / 1024.0) as peak_memory_gb,
    MIN(memory_mb / 1024.0) as min_memory_gb,
    STDDEV(memory_mb / 1024.0) as stddev_memory_gb,
    COUNT(*) as samples,
    MIN(timestamp) as stage_start,
    MAX(timestamp) as stage_end,
    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) as duration_seconds
FROM pipeline_metrics
WHERE custom_metrics->>'memory_type' = 'process_rss'
    AND custom_metrics->>'stage' IS NOT NULL
GROUP BY experiment_id, custom_metrics->>'stage'
ORDER BY experiment_id, MIN(timestamp);

-- Memory usage timeline
-- Provides a time-series view of memory usage with stage and operation context
CREATE OR REPLACE VIEW v_memory_timeline AS
SELECT 
    experiment_id,
    created_at,
    metric_value as memory_gb,
    metadata->>'stage' as stage,
    metadata->>'operation' as operation,
    (metadata->>'system_percent')::float as system_percent,
    (metadata->>'system_available_gb')::float as system_available_gb,
    (metadata->>'peak_memory_gb')::float as peak_memory_gb
FROM pipeline_metrics
WHERE metric_type = 'memory'
ORDER BY experiment_id, created_at;

-- Memory peaks per experiment
-- Identifies peak memory usage points and their context
CREATE OR REPLACE VIEW v_memory_peaks AS
WITH ranked_memory AS (
    SELECT 
        experiment_id,
        created_at,
        metric_value as memory_gb,
        metadata->>'stage' as stage,
        metadata->>'operation' as operation,
        ROW_NUMBER() OVER (PARTITION BY experiment_id ORDER BY metric_value DESC) as rank
    FROM pipeline_metrics
    WHERE metric_type = 'memory' AND metric_name = 'process_rss_gb'
)
SELECT 
    experiment_id,
    memory_gb as peak_memory_gb,
    stage,
    operation,
    created_at as peak_time
FROM ranked_memory
WHERE rank = 1;

-- Memory pressure events
-- Identifies when memory usage exceeded warning (80%) or critical (90%) thresholds
CREATE OR REPLACE VIEW v_memory_pressure_events AS
WITH memory_with_limits AS (
    SELECT 
        pm.experiment_id,
        pm.created_at,
        pm.metric_value as memory_gb,
        pm.metadata->>'stage' as stage,
        pm.metadata->>'operation' as operation,
        e.config->>'memory_limit_gb' as memory_limit_str,
        COALESCE((e.config->>'memory_limit_gb')::float, 16.0) as memory_limit_gb
    FROM pipeline_metrics pm
    JOIN experiments e ON pm.experiment_id = e.experiment_id
    WHERE pm.metric_type = 'memory' AND pm.metric_name = 'process_rss_gb'
)
SELECT 
    experiment_id,
    created_at,
    memory_gb,
    memory_limit_gb,
    (memory_gb / memory_limit_gb * 100) as usage_percent,
    stage,
    operation,
    CASE 
        WHEN memory_gb / memory_limit_gb >= 0.9 THEN 'critical'
        WHEN memory_gb / memory_limit_gb >= 0.8 THEN 'warning'
        ELSE 'normal'
    END as pressure_level
FROM memory_with_limits
WHERE memory_gb / memory_limit_gb >= 0.8
ORDER BY experiment_id, created_at;

-- Memory usage summary by experiment
-- Provides a high-level summary of memory usage for each experiment
CREATE OR REPLACE VIEW v_memory_summary AS
SELECT 
    e.experiment_id,
    e.name as experiment_name,
    e.status,
    e.start_time,
    e.end_time,
    COUNT(DISTINCT pm.metadata->>'stage') as stages_tracked,
    AVG(pm.metric_value) as avg_memory_gb,
    MAX(pm.metric_value) as peak_memory_gb,
    MIN(pm.metric_value) as min_memory_gb,
    COUNT(pm.*) as total_samples,
    COALESCE((e.config->>'memory_limit_gb')::float, 16.0) as memory_limit_gb,
    MAX(pm.metric_value) / COALESCE((e.config->>'memory_limit_gb')::float, 16.0) * 100 as peak_usage_percent
FROM experiments e
LEFT JOIN pipeline_metrics pm ON e.experiment_id = pm.experiment_id 
    AND pm.metric_type = 'memory' 
    AND pm.metric_name = 'process_rss_gb'
GROUP BY e.experiment_id, e.name, e.status, e.start_time, e.end_time, e.config
ORDER BY e.start_time DESC;

-- Stage duration and memory efficiency
-- Compares stage duration with memory usage to identify inefficient stages
CREATE OR REPLACE VIEW v_stage_memory_efficiency AS
WITH stage_stats AS (
    SELECT 
        experiment_id,
        metadata->>'stage' as stage,
        AVG(metric_value) as avg_memory_gb,
        MAX(metric_value) as peak_memory_gb,
        MIN(created_at) as stage_start,
        MAX(created_at) as stage_end,
        EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at))) as duration_seconds
    FROM pipeline_metrics
    WHERE metric_type = 'memory' 
        AND metric_name = 'process_rss_gb'
        AND metadata->>'stage' IS NOT NULL
    GROUP BY experiment_id, metadata->>'stage'
)
SELECT 
    experiment_id,
    stage,
    duration_seconds,
    avg_memory_gb,
    peak_memory_gb,
    CASE 
        WHEN duration_seconds > 0 THEN peak_memory_gb * duration_seconds / 60 
        ELSE 0 
    END as memory_time_product,  -- GB-minutes
    CASE 
        WHEN duration_seconds > 0 THEN avg_memory_gb / (duration_seconds / 60)
        ELSE 0
    END as gb_per_minute
FROM stage_stats
ORDER BY memory_time_product DESC;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_memory ON pipeline_metrics(experiment_id, created_at) 
WHERE metric_type = 'memory';

CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_stage ON pipeline_metrics((metadata->>'stage')) 
WHERE metric_type = 'memory';