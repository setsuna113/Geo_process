-- Monitoring and Logging Schema Extensions
-- For unified monitoring and structured logging system

-- ==============================================================================
-- STRUCTURED LOGGING TABLES
-- ==============================================================================

-- Structured logs table for pipeline execution
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    job_id UUID REFERENCES processing_jobs(id) ON DELETE CASCADE,
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
CREATE INDEX idx_logs_job ON pipeline_logs(job_id);

-- Events table for significant occurrences
CREATE TABLE IF NOT EXISTS pipeline_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,  -- 'stage_start', 'stage_complete', 'error', 'warning', 'checkpoint_saved'
    source VARCHAR(255) NOT NULL,  -- Component that generated event
    severity VARCHAR(20) DEFAULT 'info' CHECK (severity IN ('debug', 'info', 'warning', 'error', 'critical')),
    title VARCHAR(500) NOT NULL,
    details JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_experiment ON pipeline_events(experiment_id, timestamp DESC);
CREATE INDEX idx_events_type ON pipeline_events(event_type);
CREATE INDEX idx_events_severity ON pipeline_events(severity) WHERE severity IN ('error', 'critical');

-- ==============================================================================
-- MONITORING TABLES
-- ==============================================================================

-- Progress tracking table with hierarchy support
CREATE TABLE IF NOT EXISTS pipeline_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    node_id VARCHAR(255) NOT NULL,
    node_level VARCHAR(50) NOT NULL CHECK (node_level IN ('pipeline', 'phase', 'step', 'substep')),
    parent_id VARCHAR(255),
    node_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress_percent FLOAT DEFAULT 0 CHECK (progress_percent >= 0 AND progress_percent <= 100),
    completed_units INTEGER DEFAULT 0,
    total_units INTEGER DEFAULT 100,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, node_id)
);

CREATE INDEX idx_progress_experiment_node ON pipeline_progress(experiment_id, node_id);
CREATE INDEX idx_progress_status ON pipeline_progress(status) WHERE status = 'running';
CREATE INDEX idx_progress_parent ON pipeline_progress(experiment_id, parent_id);

-- Resource metrics table
CREATE TABLE IF NOT EXISTS pipeline_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    node_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    memory_mb FLOAT,
    cpu_percent FLOAT,
    disk_usage_mb FLOAT,
    throughput_per_sec FLOAT,
    custom_metrics JSONB DEFAULT '{}'
);

CREATE INDEX idx_metrics_experiment_time ON pipeline_metrics(experiment_id, timestamp DESC);
CREATE INDEX idx_metrics_node ON pipeline_metrics(experiment_id, node_id);

-- ==============================================================================
-- VIEWS FOR EASY QUERYING
-- ==============================================================================

-- View for recent errors by experiment
CREATE OR REPLACE VIEW recent_experiment_errors AS
SELECT 
    e.name as experiment_name,
    e.id as experiment_id,
    l.timestamp,
    l.level,
    l.node_id,
    l.message,
    l.traceback,
    l.context
FROM pipeline_logs l
JOIN experiments e ON l.experiment_id = e.id
WHERE l.level IN ('ERROR', 'CRITICAL')
ORDER BY l.timestamp DESC;

-- View for experiment progress summary
CREATE OR REPLACE VIEW experiment_progress_summary AS
SELECT 
    e.name as experiment_name,
    e.id as experiment_id,
    p.node_level,
    COUNT(*) as total_nodes,
    COUNT(CASE WHEN p.status = 'completed' THEN 1 END) as completed_nodes,
    COUNT(CASE WHEN p.status = 'failed' THEN 1 END) as failed_nodes,
    COUNT(CASE WHEN p.status = 'running' THEN 1 END) as running_nodes,
    AVG(p.progress_percent) as avg_progress
FROM experiments e
JOIN pipeline_progress p ON e.id = p.experiment_id
GROUP BY e.id, e.name, p.node_level;

-- View for active pipeline monitoring
CREATE OR REPLACE VIEW active_pipeline_monitoring AS
SELECT 
    e.name as experiment_name,
    e.id as experiment_id,
    e.status as experiment_status,
    p.node_id,
    p.node_name,
    p.node_level,
    p.status as node_status,
    p.progress_percent,
    p.start_time,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - p.start_time)) as elapsed_seconds,
    m.memory_mb as latest_memory_mb,
    m.cpu_percent as latest_cpu_percent
FROM experiments e
JOIN pipeline_progress p ON e.id = p.experiment_id
LEFT JOIN LATERAL (
    SELECT memory_mb, cpu_percent 
    FROM pipeline_metrics 
    WHERE experiment_id = e.id AND node_id = p.node_id 
    ORDER BY timestamp DESC 
    LIMIT 1
) m ON true
WHERE e.status = 'running' AND p.status = 'running';

-- ==============================================================================
-- FUNCTIONS FOR MONITORING
-- ==============================================================================

-- Function to get experiment logs with filters
CREATE OR REPLACE FUNCTION get_experiment_logs(
    p_experiment_id UUID,
    p_level VARCHAR DEFAULT NULL,
    p_search TEXT DEFAULT NULL,
    p_start_time TIMESTAMP DEFAULT NULL,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE(
    id UUID,
    timestamp TIMESTAMP WITH TIME ZONE,
    level VARCHAR,
    node_id VARCHAR,
    message TEXT,
    context JSONB,
    traceback TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        l.id,
        l.timestamp,
        l.level,
        l.node_id,
        l.message,
        l.context,
        l.traceback
    FROM pipeline_logs l
    WHERE l.experiment_id = p_experiment_id
        AND (p_level IS NULL OR l.level = p_level)
        AND (p_search IS NULL OR l.message ILIKE '%' || p_search || '%')
        AND (p_start_time IS NULL OR l.timestamp >= p_start_time)
    ORDER BY l.timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to get progress tree for an experiment
CREATE OR REPLACE FUNCTION get_progress_tree(p_experiment_id UUID)
RETURNS TABLE(
    node_id VARCHAR,
    parent_id VARCHAR,
    node_level VARCHAR,
    node_name VARCHAR,
    status VARCHAR,
    progress_percent FLOAT,
    path TEXT[]
) AS $$
WITH RECURSIVE progress_tree AS (
    -- Base case: root nodes
    SELECT 
        node_id,
        parent_id,
        node_level,
        node_name,
        status,
        progress_percent,
        ARRAY[node_id] as path
    FROM pipeline_progress
    WHERE experiment_id = p_experiment_id AND parent_id IS NULL
    
    UNION ALL
    
    -- Recursive case: child nodes
    SELECT 
        p.node_id,
        p.parent_id,
        p.node_level,
        p.node_name,
        p.status,
        p.progress_percent,
        pt.path || p.node_id
    FROM pipeline_progress p
    JOIN progress_tree pt ON p.parent_id = pt.node_id
    WHERE p.experiment_id = p_experiment_id
)
SELECT * FROM progress_tree
ORDER BY path;
$$ LANGUAGE plpgsql;

-- Function to aggregate child progress to parent
CREATE OR REPLACE FUNCTION update_parent_progress()
RETURNS TRIGGER AS $$
DECLARE
    v_parent_id VARCHAR;
    v_experiment_id UUID;
    v_total_progress FLOAT;
    v_child_count INTEGER;
BEGIN
    -- Get parent info
    v_parent_id := NEW.parent_id;
    v_experiment_id := NEW.experiment_id;
    
    -- If has parent, update it
    IF v_parent_id IS NOT NULL THEN
        -- Calculate average progress of all children
        SELECT 
            AVG(progress_percent),
            COUNT(*)
        INTO v_total_progress, v_child_count
        FROM pipeline_progress
        WHERE experiment_id = v_experiment_id 
        AND parent_id = v_parent_id;
        
        -- Update parent progress
        UPDATE pipeline_progress
        SET 
            progress_percent = COALESCE(v_total_progress, 0),
            updated_at = CURRENT_TIMESTAMP
        WHERE experiment_id = v_experiment_id 
        AND node_id = v_parent_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update parent progress when child changes
-- Only trigger on significant changes to reduce load
CREATE TRIGGER update_parent_progress_trigger
AFTER INSERT OR UPDATE OF progress_percent, status ON pipeline_progress
FOR EACH ROW
WHEN (
    NEW.status IS DISTINCT FROM OLD.status OR 
    ABS(COALESCE(NEW.progress_percent, 0) - COALESCE(OLD.progress_percent, 0)) > 5 OR
    OLD.progress_percent IS NULL
)
EXECUTE FUNCTION update_parent_progress();

-- ==============================================================================
-- MAINTENANCE FUNCTIONS
-- ==============================================================================

-- Function to clean up old logs
CREATE OR REPLACE FUNCTION cleanup_old_logs(p_days_to_keep INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    v_deleted_count INTEGER;
BEGIN
    DELETE FROM pipeline_logs
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * p_days_to_keep;
    
    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    RETURN v_deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get error summary for experiment
CREATE OR REPLACE FUNCTION get_error_summary(p_experiment_id UUID)
RETURNS TABLE(
    total_errors BIGINT,
    error_by_level JSONB,
    error_by_stage JSONB,
    recent_errors JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH error_stats AS (
        SELECT 
            COUNT(*) as total,
            jsonb_object_agg(level, level_count) as by_level
        FROM (
            SELECT level, COUNT(*) as level_count
            FROM pipeline_logs
            WHERE experiment_id = p_experiment_id 
            AND level IN ('ERROR', 'CRITICAL')
            GROUP BY level
        ) l
    ),
    stage_errors AS (
        SELECT jsonb_object_agg(
            COALESCE(context->>'stage', 'unknown'), 
            stage_count
        ) as by_stage
        FROM (
            SELECT 
                context->>'stage' as stage,
                COUNT(*) as stage_count
            FROM pipeline_logs
            WHERE experiment_id = p_experiment_id 
            AND level IN ('ERROR', 'CRITICAL')
            GROUP BY context->>'stage'
        ) s
    ),
    recent AS (
        SELECT jsonb_agg(
            jsonb_build_object(
                'timestamp', timestamp,
                'level', level,
                'message', message,
                'stage', context->>'stage'
            ) ORDER BY timestamp DESC
        ) as recent_list
        FROM (
            SELECT timestamp, level, message, context
            FROM pipeline_logs
            WHERE experiment_id = p_experiment_id 
            AND level IN ('ERROR', 'CRITICAL')
            ORDER BY timestamp DESC
            LIMIT 10
        ) r
    )
    SELECT 
        COALESCE(e.total, 0),
        COALESCE(e.by_level, '{}'::jsonb),
        COALESCE(s.by_stage, '{}'::jsonb),
        COALESCE(r.recent_list, '[]'::jsonb)
    FROM error_stats e
    CROSS JOIN stage_errors s
    CROSS JOIN recent r;
END;
$$ LANGUAGE plpgsql;

-- Add comments for documentation
COMMENT ON TABLE pipeline_logs IS 'Structured logs for pipeline execution with full context and tracebacks';
COMMENT ON TABLE pipeline_events IS 'Significant events during pipeline execution';
COMMENT ON TABLE pipeline_progress IS 'Hierarchical progress tracking for pipeline stages';
COMMENT ON TABLE pipeline_metrics IS 'Resource usage metrics during pipeline execution';
COMMENT ON FUNCTION get_experiment_logs IS 'Query logs with filtering options';
COMMENT ON FUNCTION get_progress_tree IS 'Get hierarchical progress tree for an experiment';
COMMENT ON FUNCTION get_error_summary IS 'Get error statistics for an experiment';