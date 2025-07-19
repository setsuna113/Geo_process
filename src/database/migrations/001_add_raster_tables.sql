-- Migration: Add raster compatibility tables
-- Date: 2025-07-19
-- Description: Adds tables for raster data sources, tiling, resampling cache, and processing queue

-- Create raster status enum for processing tracking
DO $$ BEGIN
    CREATE TYPE raster_status_enum AS ENUM (
        'pending',
        'validating',
        'tiling',
        'indexing',
        'ready',
        'error',
        'archived'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create tile processing status enum
DO $$ BEGIN
    CREATE TYPE tile_status_enum AS ENUM (
        'pending',
        'processing',
        'completed',
        'failed',
        'skipped'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Raster sources table - metadata for each raster file
CREATE TABLE raster_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    file_path VARCHAR(1000) NOT NULL,
    data_type VARCHAR(50) NOT NULL, -- 'Int32', 'UInt16', 'Float32', etc.
    pixel_size_degrees FLOAT NOT NULL DEFAULT 0.016666666666667, -- ~1.85km at equator
    spatial_extent GEOMETRY(POLYGON, 4326) NOT NULL,
    nodata_value FLOAT,
    band_count INTEGER DEFAULT 1,
    file_size_mb FLOAT,
    checksum VARCHAR(64),
    last_modified TIMESTAMP,
    processing_status raster_status_enum DEFAULT 'pending',
    
    -- Metadata
    source_dataset VARCHAR(200),
    variable_name VARCHAR(100),
    units VARCHAR(50),
    description TEXT,
    temporal_info JSONB DEFAULT '{}', -- start_date, end_date, temporal_resolution
    
    -- Indexing and management
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    
    CHECK(pixel_size_degrees > 0),
    CHECK(band_count > 0),
    CHECK(file_size_mb >= 0)
);

-- Raster tiles table - divide large rasters into manageable chunks
CREATE TABLE raster_tiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    raster_source_id UUID REFERENCES raster_sources(id) ON DELETE CASCADE,
    tile_x INTEGER NOT NULL,
    tile_y INTEGER NOT NULL,
    tile_size_pixels INTEGER NOT NULL DEFAULT 1000,
    
    -- Spatial bounds of the tile
    tile_bounds GEOMETRY(POLYGON, 4326) NOT NULL,
    
    -- File access information for random access
    file_byte_offset BIGINT,
    file_byte_length BIGINT,
    
    -- Pre-computed statistics per tile for quick filtering
    tile_stats JSONB DEFAULT '{}', -- {min, max, mean, std, nodata_count, valid_count}
    
    -- Processing information
    processing_status tile_status_enum DEFAULT 'pending',
    processed_at TIMESTAMP,
    
    -- Optimization
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(raster_source_id, tile_x, tile_y),
    CHECK(tile_x >= 0),
    CHECK(tile_y >= 0),
    CHECK(tile_size_pixels > 0)
);

-- Resampling cache table - cache resampled values to avoid recomputation
CREATE TABLE resampling_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_raster_id UUID REFERENCES raster_sources(id) ON DELETE CASCADE,
    target_grid_id UUID REFERENCES grids(id) ON DELETE CASCADE,
    cell_id VARCHAR(100) NOT NULL,
    
    -- Resampling configuration
    method VARCHAR(50) NOT NULL DEFAULT 'bilinear', -- 'nearest', 'bilinear', 'cubic', 'average'
    band_number INTEGER DEFAULT 1,
    
    -- Results
    value FLOAT,
    confidence_score FLOAT DEFAULT 1.0, -- Quality metric for resampling (0-1)
    
    -- Metadata
    source_tiles_used INTEGER[], -- Array of tile IDs used in computation
    computation_metadata JSONB DEFAULT '{}',
    
    -- Cache management
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    
    UNIQUE(source_raster_id, target_grid_id, cell_id, method, band_number),
    CHECK(confidence_score BETWEEN 0 AND 1),
    CHECK(band_number > 0),
    CHECK(access_count > 0)
);

-- Processing queue table - track which tiles/regions need processing
CREATE TABLE processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    queue_type VARCHAR(50) NOT NULL, -- 'raster_tiling', 'resampling', 'validation'
    
    -- Task identification
    raster_source_id UUID REFERENCES raster_sources(id) ON DELETE CASCADE,
    grid_id UUID REFERENCES grids(id) ON DELETE CASCADE,
    tile_id UUID REFERENCES raster_tiles(id) ON DELETE CASCADE,
    
    -- Processing parameters
    parameters JSONB DEFAULT '{}',
    priority INTEGER DEFAULT 0, -- Higher number = higher priority
    
    -- Status tracking
    status tile_status_enum DEFAULT 'pending',
    worker_id VARCHAR(100), -- ID of worker processing this task
    
    -- Timing
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Error handling and retry
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    
    -- Checkpoint/resume support
    checkpoint_data JSONB DEFAULT '{}',
    
    CHECK(priority >= 0),
    CHECK(retry_count >= 0),
    CHECK(max_retries >= 0)
);

-- Create spatial indexes for performance
CREATE INDEX idx_raster_sources_extent ON raster_sources USING GIST (spatial_extent);
CREATE INDEX idx_raster_sources_active ON raster_sources (active) WHERE active = TRUE;
CREATE INDEX idx_raster_sources_status ON raster_sources (processing_status);
CREATE INDEX idx_raster_sources_name ON raster_sources (name);

CREATE INDEX idx_raster_tiles_bounds ON raster_tiles USING GIST (tile_bounds);
CREATE INDEX idx_raster_tiles_source ON raster_tiles (raster_source_id);
CREATE INDEX idx_raster_tiles_coords ON raster_tiles (raster_source_id, tile_x, tile_y);
CREATE INDEX idx_raster_tiles_status ON raster_tiles (processing_status);

CREATE INDEX idx_resampling_cache_lookup ON resampling_cache (source_raster_id, target_grid_id, cell_id);
CREATE INDEX idx_resampling_cache_method ON resampling_cache (method);
CREATE INDEX idx_resampling_cache_accessed ON resampling_cache (last_accessed);
CREATE INDEX idx_resampling_cache_grid ON resampling_cache (target_grid_id);

CREATE INDEX idx_processing_queue_type_status ON processing_queue (queue_type, status);
CREATE INDEX idx_processing_queue_priority ON processing_queue (priority DESC, created_at ASC);
CREATE INDEX idx_processing_queue_worker ON processing_queue (worker_id) WHERE worker_id IS NOT NULL;
CREATE INDEX idx_processing_queue_raster ON processing_queue (raster_source_id);
CREATE INDEX idx_processing_queue_retry ON processing_queue (retry_count, max_retries);

-- Create utility views for monitoring and management

-- Raster processing status overview
CREATE VIEW raster_processing_status AS
SELECT 
    rs.id as raster_id,
    rs.name as raster_name,
    rs.processing_status,
    rs.file_size_mb,
    COUNT(rt.id) as total_tiles,
    COUNT(CASE WHEN rt.processing_status = 'completed' THEN 1 END) as completed_tiles,
    COUNT(CASE WHEN rt.processing_status = 'failed' THEN 1 END) as failed_tiles,
    COUNT(CASE WHEN rt.processing_status = 'pending' THEN 1 END) as pending_tiles,
    ROUND(
        (COUNT(CASE WHEN rt.processing_status = 'completed' THEN 1 END)::FLOAT / 
         NULLIF(COUNT(rt.id), 0) * 100)::NUMERIC, 2
    ) as completion_percent,
    rs.created_at,
    rs.updated_at
FROM raster_sources rs
LEFT JOIN raster_tiles rt ON rs.id = rt.raster_source_id
GROUP BY rs.id, rs.name, rs.processing_status, rs.file_size_mb, rs.created_at, rs.updated_at;

-- Cache efficiency view
CREATE VIEW cache_efficiency_summary AS
SELECT 
    rc.source_raster_id,
    rs.name as raster_name,
    rc.target_grid_id,
    g.name as grid_name,
    rc.method,
    COUNT(*) as cached_cells,
    AVG(rc.confidence_score) as avg_confidence,
    AVG(rc.access_count) as avg_access_count,
    MIN(rc.created_at) as first_cached,
    MAX(rc.last_accessed) as last_accessed
FROM resampling_cache rc
JOIN raster_sources rs ON rc.source_raster_id = rs.id
JOIN grids g ON rc.target_grid_id = g.id
GROUP BY rc.source_raster_id, rs.name, rc.target_grid_id, g.name, rc.method;

-- Processing queue summary
CREATE VIEW processing_queue_summary AS
SELECT 
    queue_type,
    status,
    COUNT(*) as task_count,
    AVG(priority) as avg_priority,
    MIN(created_at) as oldest_task,
    COUNT(CASE WHEN retry_count > 0 THEN 1 END) as retried_tasks,
    COUNT(CASE WHEN worker_id IS NOT NULL THEN 1 END) as assigned_tasks
FROM processing_queue
GROUP BY queue_type, status
ORDER BY queue_type, status;

-- Functions for cache management

-- Function to clean up old cache entries
CREATE OR REPLACE FUNCTION cleanup_resampling_cache(
    days_old INTEGER DEFAULT 30,
    min_access_count INTEGER DEFAULT 1
) RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM resampling_cache 
    WHERE last_accessed < (NOW() - INTERVAL '1 day' * days_old)
    AND access_count <= min_access_count;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update tile statistics
CREATE OR REPLACE FUNCTION update_tile_statistics(
    p_tile_id UUID,
    p_stats JSONB
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE raster_tiles 
    SET tile_stats = p_stats,
        processing_status = 'completed',
        processed_at = CURRENT_TIMESTAMP
    WHERE id = p_tile_id;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to get next processing task
CREATE OR REPLACE FUNCTION get_next_processing_task(
    p_queue_type VARCHAR(50),
    p_worker_id VARCHAR(100)
) RETURNS UUID AS $$
DECLARE
    task_id UUID;
BEGIN
    -- Get highest priority pending task
    UPDATE processing_queue 
    SET status = 'processing',
        worker_id = p_worker_id,
        started_at = CURRENT_TIMESTAMP
    WHERE id = (
        SELECT id FROM processing_queue 
        WHERE queue_type = p_queue_type 
        AND status = 'pending'
        AND retry_count < max_retries
        ORDER BY priority DESC, created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    RETURNING id INTO task_id;
    
    RETURN task_id;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_raster_sources_updated_at
    BEFORE UPDATE ON raster_sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE raster_sources IS 'Metadata for raster data files used in analysis';
COMMENT ON TABLE raster_tiles IS 'Spatial tiles for efficient raster processing and access';
COMMENT ON TABLE resampling_cache IS 'Cache for expensive raster resampling operations';
COMMENT ON TABLE processing_queue IS 'Queue for managing distributed raster processing tasks';

COMMENT ON FUNCTION cleanup_resampling_cache IS 'Remove old and rarely accessed cache entries';
COMMENT ON FUNCTION update_tile_statistics IS 'Update tile statistics after processing';
COMMENT ON FUNCTION get_next_processing_task IS 'Get next task for worker with locking';
