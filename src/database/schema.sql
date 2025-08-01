-- Storage layer for spatial biodiversity analysis pipeline

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- ENUM TYPES
-- ==============================================================================

-- Processing status enum for grid computation stages
DO $$ BEGIN
    CREATE TYPE processing_status_enum AS ENUM (
        'pending',
        'initializing', 
        'grid_generation',
        'species_intersection',
        'feature_calculation',
        'aggregation',
        'optimization',
        'validation',
        'finalization',
        'completed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Raster processing status enum
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

-- Tile processing status enum
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

-- ==============================================================================
-- CORE TABLES
-- ==============================================================================

-- Grid definitions table
CREATE TABLE IF NOT EXISTS grids (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    grid_type VARCHAR(20) NOT NULL,
    resolution INTEGER NOT NULL,
    crs VARCHAR(20) NOT NULL,
    bounds GEOMETRY(POLYGON, 4326),
    processing_status processing_status_enum DEFAULT 'pending',
    total_cells INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    metadata JSONB DEFAULT '{}',
    
    UNIQUE(name),
    CHECK(grid_type IN ('cubic', 'hexagonal')),
    CHECK(crs IN ('EPSG:3857', 'EPSG:4326'))
);

-- Grid cells table
CREATE TABLE IF NOT EXISTS grid_cells (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grid_id UUID REFERENCES grids(id) ON DELETE CASCADE,
    cell_id VARCHAR(100) NOT NULL,
    geometry GEOMETRY(POLYGON, 4326) NOT NULL,
    area_km2 FLOAT,
    centroid GEOMETRY(POINT, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    
    UNIQUE(grid_id, cell_id)
);

-- Species range data table
CREATE TABLE IF NOT EXISTS species_ranges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    species_name VARCHAR(200) NOT NULL,
    scientific_name VARCHAR(200),
    genus VARCHAR(100),
    family VARCHAR(100),
    order_name VARCHAR(100),
    class_name VARCHAR(100),
    phylum VARCHAR(100),
    kingdom VARCHAR(100),
    category VARCHAR(50) DEFAULT 'unknown',
    range_type VARCHAR(50) DEFAULT 'distribution',
    geometry GEOMETRY(GEOMETRY, 4326) NOT NULL,
    source_file VARCHAR(500) NOT NULL,
    source_dataset VARCHAR(200),
    confidence FLOAT DEFAULT 1.0,
    area_km2 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    metadata JSONB DEFAULT '{}',
    
    CHECK(category IN ('plant', 'animal', 'fungi', 'unknown')),
    CHECK(confidence BETWEEN 0 AND 1)
);

-- Species-grid intersections table
CREATE TABLE IF NOT EXISTS species_grid_intersections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grid_id UUID REFERENCES grids(id) ON DELETE CASCADE,
    cell_id VARCHAR(100) NOT NULL,
    species_range_id UUID REFERENCES species_ranges(id) ON DELETE CASCADE,
    species_name VARCHAR(200) NOT NULL,
    category VARCHAR(50) NOT NULL,
    range_type VARCHAR(50) NOT NULL,
    intersection_area_km2 FLOAT,
    coverage_percent FLOAT,
    presence_score FLOAT DEFAULT 1.0,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    computation_metadata JSONB DEFAULT '{}',
    
    UNIQUE(grid_id, cell_id, species_range_id),
    CHECK(category IN ('plant', 'animal', 'fungi', 'unknown')),
    CHECK(coverage_percent BETWEEN 0 AND 100),
    CHECK(presence_score BETWEEN 0 AND 1)
);

-- Features table
CREATE TABLE IF NOT EXISTS features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grid_id UUID REFERENCES grids(id) ON DELETE CASCADE,
    cell_id VARCHAR(100) NOT NULL,
    feature_type VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value FLOAT NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    computation_metadata JSONB DEFAULT '{}',
    
    UNIQUE(grid_id, cell_id, feature_type, feature_name)
);

-- Climate data table
CREATE TABLE IF NOT EXISTS climate_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grid_id UUID REFERENCES grids(id) ON DELETE CASCADE,
    cell_id VARCHAR(100) NOT NULL,
    variable VARCHAR(20) NOT NULL,
    value FLOAT NOT NULL,
    source VARCHAR(100),
    resolution VARCHAR(20),
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    
    UNIQUE(grid_id, cell_id, variable, source, resolution)
);

-- ==============================================================================
-- EXPERIMENT AND JOB TRACKING TABLES
-- ==============================================================================

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    results JSONB DEFAULT '{}',
    error_message TEXT,
    created_by VARCHAR(100),
    last_checkpoint_id VARCHAR(255),
    
    UNIQUE(name),
    CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

-- Processing jobs table
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR(50) NOT NULL,
    job_name VARCHAR(200),
    status VARCHAR(20) DEFAULT 'pending',
    parameters JSONB NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    progress_percent FLOAT DEFAULT 0,
    log_messages TEXT[] DEFAULT '{}',
    error_message TEXT,
    parent_experiment_id UUID REFERENCES experiments(id),
    last_checkpoint_id VARCHAR(255),
    resume_metadata JSONB DEFAULT '{}',
    
    CHECK(status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CHECK(progress_percent BETWEEN 0 AND 100)
);

-- ==============================================================================
-- CHECKPOINT AND PROGRESS TRACKING TABLES
-- ==============================================================================

-- Pipeline checkpoints table
CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checkpoint_id VARCHAR(255) UNIQUE NOT NULL,
    level VARCHAR(50) NOT NULL CHECK (level IN ('pipeline', 'phase', 'step', 'substep')),
    parent_id VARCHAR(255),
    processor_name VARCHAR(255) NOT NULL,
    data_summary JSONB NOT NULL DEFAULT '{}',
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    compression_type VARCHAR(50),
    status VARCHAR(50) NOT NULL DEFAULT 'created' CHECK (status IN ('created', 'valid', 'corrupted', 'deleted')),
    validation_checksum VARCHAR(64),
    validation_result JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES pipeline_checkpoints(checkpoint_id) ON DELETE CASCADE
);

-- Processing steps table for fine-grained progress
CREATE TABLE IF NOT EXISTS processing_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    step_name VARCHAR(255) NOT NULL,
    processor_name VARCHAR(255) NOT NULL,
    parent_job_id UUID REFERENCES processing_jobs(id) ON DELETE CASCADE,
    total_items INTEGER NOT NULL,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    parameters JSONB DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_messages TEXT[],
    last_checkpoint_id VARCHAR(255) REFERENCES pipeline_checkpoints(checkpoint_id),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- File processing status for individual file progress
CREATE TABLE IF NOT EXISTS file_processing_status (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    bytes_processed BIGINT DEFAULT 0,
    processor_name VARCHAR(255) NOT NULL,
    parent_job_id UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    chunks_completed INTEGER DEFAULT 0,
    total_chunks INTEGER,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    last_checkpoint_id VARCHAR(255) REFERENCES pipeline_checkpoints(checkpoint_id),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================================================
-- RESAMPLING AND DATA PREPARATION TABLES
-- ==============================================================================

-- Resampled datasets table
CREATE TABLE IF NOT EXISTS resampled_datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    source_path TEXT NOT NULL,
    target_resolution FLOAT NOT NULL,
    target_crs VARCHAR(50) NOT NULL,
    bounds FLOAT[] NOT NULL,
    shape_height INTEGER NOT NULL,
    shape_width INTEGER NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    resampling_method VARCHAR(50) NOT NULL,
    band_name VARCHAR(100) NOT NULL,
    data_table_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Data cleaning operations log
CREATE TABLE IF NOT EXISTS data_cleaning_log (
    id SERIAL PRIMARY KEY,
    raster_name VARCHAR(255) NOT NULL,
    dataset_type VARCHAR(50),
    total_pixels BIGINT,
    pixels_cleaned BIGINT,
    cleaning_ratio FLOAT,
    value_range FLOAT[],
    operations JSONB,
    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system'
);

-- Raster merge operations log
CREATE TABLE IF NOT EXISTS raster_merge_log (
    id SERIAL PRIMARY KEY,
    source_rasters JSONB NOT NULL,
    band_names TEXT[],
    output_shape INTEGER[],
    merge_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    metadata JSONB
);

-- Normalization parameters storage
CREATE TABLE IF NOT EXISTS normalization_parameters (
    id SERIAL PRIMARY KEY,
    method VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system'
);

-- ==============================================================================
-- RASTER PROCESSING TABLES
-- ==============================================================================

-- Raster sources table
CREATE TABLE IF NOT EXISTS raster_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    file_path VARCHAR(1000) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    pixel_size_degrees FLOAT NOT NULL DEFAULT 0.016666666666667,
    spatial_extent GEOMETRY(POLYGON, 4326) NOT NULL,
    nodata_value FLOAT,
    band_count INTEGER DEFAULT 1,
    file_size_mb FLOAT,
    checksum VARCHAR(64),
    last_modified TIMESTAMP,
    processing_status raster_status_enum DEFAULT 'pending',
    source_dataset VARCHAR(200),
    variable_name VARCHAR(100),
    units VARCHAR(50),
    description TEXT,
    temporal_info JSONB DEFAULT '{}',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    metadata JSONB DEFAULT '{}',
    
    CHECK(pixel_size_degrees > 0),
    CHECK(band_count > 0),
    CHECK(file_size_mb >= 0)
);

-- Raster tiles table
CREATE TABLE IF NOT EXISTS raster_tiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raster_source_id UUID REFERENCES raster_sources(id) ON DELETE CASCADE,
    tile_x INTEGER NOT NULL,
    tile_y INTEGER NOT NULL,
    tile_size_pixels INTEGER NOT NULL DEFAULT 1000,
    tile_bounds GEOMETRY(POLYGON, 4326) NOT NULL,
    file_byte_offset BIGINT,
    file_byte_length BIGINT,
    tile_stats JSONB DEFAULT '{}',
    processing_status tile_status_enum DEFAULT 'pending',
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    
    UNIQUE(raster_source_id, tile_x, tile_y),
    CHECK(tile_x >= 0),
    CHECK(tile_y >= 0),
    CHECK(tile_size_pixels > 0)
);

-- Resampling cache table
CREATE TABLE IF NOT EXISTS resampling_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_raster_id UUID REFERENCES raster_sources(id) ON DELETE CASCADE,
    target_grid_id UUID REFERENCES grids(id) ON DELETE CASCADE,
    cell_id VARCHAR(100) NOT NULL,
    method VARCHAR(50) NOT NULL DEFAULT 'bilinear',
    band_number INTEGER DEFAULT 1,
    value FLOAT,
    confidence_score FLOAT DEFAULT 1.0,
    source_tiles_used INTEGER[],
    computation_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1,
    created_by VARCHAR(100) DEFAULT 'system',
    
    UNIQUE(source_raster_id, target_grid_id, cell_id, method, band_number),
    CHECK(confidence_score BETWEEN 0 AND 1),
    CHECK(band_number > 0),
    CHECK(access_count > 0)
);

-- Processing queue table
CREATE TABLE IF NOT EXISTS processing_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_type VARCHAR(50) NOT NULL,
    raster_source_id UUID REFERENCES raster_sources(id) ON DELETE CASCADE,
    grid_id UUID REFERENCES grids(id) ON DELETE CASCADE,
    tile_id UUID REFERENCES raster_tiles(id) ON DELETE CASCADE,
    parameters JSONB DEFAULT '{}',
    priority INTEGER DEFAULT 0,
    status tile_status_enum DEFAULT 'pending',
    worker_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    checkpoint_data JSONB DEFAULT '{}',
    created_by VARCHAR(100) DEFAULT 'system',
    
    CHECK(priority >= 0),
    CHECK(retry_count >= 0),
    CHECK(max_retries >= 0)
);

-- ==============================================================================
-- EXPORT TRACKING TABLES
-- ==============================================================================

-- Export metadata table
CREATE TABLE IF NOT EXISTS export_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    grid_id UUID REFERENCES grids(id) ON DELETE CASCADE,
    export_type VARCHAR(50) NOT NULL,
    feature_types TEXT[] DEFAULT '{}',
    file_path VARCHAR(1000) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_size_bytes BIGINT,
    format_version VARCHAR(20),
    compression VARCHAR(20),
    spatial_extent GEOMETRY(POLYGON, 4326),
    temporal_range JSONB,
    metadata JSONB DEFAULT '{}',
    checksum VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    created_by VARCHAR(100),
    
    CHECK(export_type IN ('geotiff', 'netcdf', 'csv', 'gpkg', 'shapefile', 'parquet', 'json')),
    CHECK(compression IN ('none', 'gzip', 'lzw', 'deflate', 'zstd'))
);

-- ==============================================================================
-- INDEXES
-- ==============================================================================

-- Core table indexes
CREATE INDEX idx_grids_type_resolution ON grids (grid_type, resolution);
CREATE INDEX idx_grids_name ON grids (name);
CREATE INDEX idx_grids_bounds ON grids USING GIST(bounds);
CREATE INDEX idx_grids_processing_status ON grids (processing_status);

CREATE INDEX idx_grid_cells_geometry ON grid_cells USING GIST (geometry);
CREATE INDEX idx_grid_cells_centroid ON grid_cells USING GIST (centroid);
CREATE INDEX idx_grid_cells_grid_id ON grid_cells (grid_id);
CREATE INDEX idx_grid_cells_cell_id ON grid_cells (grid_id, cell_id);

CREATE INDEX idx_species_ranges_geometry ON species_ranges USING GIST (geometry);
CREATE INDEX idx_species_ranges_category ON species_ranges (category);
CREATE INDEX idx_species_ranges_species ON species_ranges (species_name);
CREATE INDEX idx_species_ranges_source ON species_ranges (source_file);

CREATE INDEX idx_species_intersections_grid_cell ON species_grid_intersections (grid_id, cell_id);
CREATE INDEX idx_species_intersections_species ON species_grid_intersections (species_name);
CREATE INDEX idx_species_intersections_category ON species_grid_intersections (category);

CREATE INDEX idx_features_grid_cell ON features (grid_id, cell_id);
CREATE INDEX idx_features_type_name ON features (feature_type, feature_name);

CREATE INDEX idx_climate_grid_cell ON climate_data (grid_id, cell_id);
CREATE INDEX idx_climate_variable ON climate_data (variable);

-- Experiment and job indexes
CREATE INDEX idx_experiments_status ON experiments (status);
CREATE INDEX idx_experiments_created ON experiments (started_at);

CREATE INDEX idx_jobs_type_status ON processing_jobs (job_type, status);
CREATE INDEX idx_jobs_experiment ON processing_jobs (parent_experiment_id);

-- Checkpoint and progress indexes
CREATE INDEX idx_pipeline_checkpoints_processor ON pipeline_checkpoints(processor_name);
CREATE INDEX idx_pipeline_checkpoints_level ON pipeline_checkpoints(level);
CREATE INDEX idx_pipeline_checkpoints_status ON pipeline_checkpoints(status);
CREATE INDEX idx_pipeline_checkpoints_created ON pipeline_checkpoints(created_at);

CREATE INDEX idx_processing_steps_job ON processing_steps(parent_job_id);
CREATE INDEX idx_processing_steps_status ON processing_steps(status);
CREATE INDEX idx_processing_steps_processor ON processing_steps(processor_name);

CREATE INDEX idx_file_processing_path ON file_processing_status(file_path);
CREATE INDEX idx_file_processing_status ON file_processing_status(status);
CREATE INDEX idx_file_processing_job ON file_processing_status(parent_job_id);

-- Resampling and data preparation indexes
CREATE INDEX idx_resampled_datasets_name ON resampled_datasets(name);
CREATE INDEX idx_resampled_datasets_type ON resampled_datasets(data_type);
CREATE INDEX idx_resampled_datasets_resolution ON resampled_datasets(target_resolution);

CREATE INDEX idx_cleaning_log_raster ON data_cleaning_log(raster_name);
CREATE INDEX idx_merge_log_date ON raster_merge_log(merge_date);

-- Raster processing indexes
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

-- Export metadata indexes
CREATE INDEX idx_export_metadata_grid_id ON export_metadata (grid_id);
CREATE INDEX idx_export_metadata_type ON export_metadata (export_type);
CREATE INDEX idx_export_metadata_created ON export_metadata (created_at);
CREATE INDEX idx_export_metadata_expires ON export_metadata (expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_export_metadata_extent ON export_metadata USING GIST (spatial_extent);
CREATE INDEX idx_export_metadata_features ON export_metadata USING GIN (feature_types);

-- Test data tracking indexes
CREATE INDEX idx_grids_created_by ON grids(created_by) WHERE created_by LIKE 'test_%';
CREATE INDEX idx_grid_cells_created_by ON grid_cells(created_by) WHERE created_by LIKE 'test_%';
CREATE INDEX idx_species_ranges_created_by ON species_ranges(created_by) WHERE created_by LIKE 'test_%';
CREATE INDEX idx_experiments_test_metadata ON experiments((config->>'__test_data__')) WHERE config->>'__test_data__' IS NOT NULL;

-- ==============================================================================
-- VIEWS
-- ==============================================================================

-- Species richness summary view
CREATE OR REPLACE VIEW species_richness_summary AS
SELECT 
    sgi.grid_id,
    sgi.cell_id,
    sgi.category,
    sgi.range_type,
    COUNT(DISTINCT sgi.species_name) as species_count,
    AVG(sgi.coverage_percent) as avg_coverage,
    SUM(sgi.intersection_area_km2) as total_intersection_area,
    AVG(sgi.presence_score) as avg_presence_score
FROM species_grid_intersections sgi
GROUP BY sgi.grid_id, sgi.cell_id, sgi.category, sgi.range_type;

-- Grid processing status view
CREATE OR REPLACE VIEW grid_processing_status AS
SELECT 
    g.id as grid_id,
    g.name as grid_name,
    g.grid_type,
    g.resolution,
    g.processing_status,
    g.total_cells,
    COUNT(DISTINCT gc.id) as cells_generated,
    COUNT(DISTINCT sgi.cell_id) as cells_with_species,
    COUNT(DISTINCT f.cell_id) as cells_with_features,
    COUNT(DISTINCT cd.cell_id) as cells_with_climate,
    COUNT(DISTINCT em.id) as export_files,
    ROUND(
        ((COUNT(DISTINCT gc.id)::FLOAT / NULLIF(g.total_cells, 0)) * 100)::NUMERIC, 2
    ) as generation_progress_percent,
    ST_AsText(g.bounds) as bounds_wkt,
    CASE 
        WHEN g.bounds IS NOT NULL THEN ST_Area(g.bounds::geography) / 1000000.0
        ELSE NULL 
    END as bounds_area_km2
FROM grids g
LEFT JOIN grid_cells gc ON g.id = gc.grid_id
LEFT JOIN species_grid_intersections sgi ON g.id = sgi.grid_id
LEFT JOIN features f ON g.id = f.grid_id
LEFT JOIN climate_data cd ON g.id = cd.grid_id
LEFT JOIN export_metadata em ON g.id = em.grid_id
GROUP BY g.id, g.name, g.grid_type, g.resolution, g.processing_status, g.total_cells, g.bounds;

-- Experiment summary view
CREATE OR REPLACE VIEW experiment_summary AS
SELECT 
    e.id,
    e.name,
    e.description,
    e.status,
    e.started_at,
    e.completed_at,
    EXTRACT(EPOCH FROM (COALESCE(e.completed_at, NOW()) - e.started_at))/60 as duration_minutes,
    COUNT(pj.id) as total_jobs,
    COUNT(CASE WHEN pj.status = 'completed' THEN 1 END) as completed_jobs,
    COUNT(CASE WHEN pj.status = 'failed' THEN 1 END) as failed_jobs,
    ROUND(AVG(pj.progress_percent)::NUMERIC, 2) as avg_progress
FROM experiments e
LEFT JOIN processing_jobs pj ON e.id = pj.parent_experiment_id
GROUP BY e.id, e.name, e.description, e.status, e.started_at, e.completed_at;

-- Checkpoint summary view
CREATE OR REPLACE VIEW checkpoint_summary AS
SELECT 
    level,
    processor_name,
    COUNT(*) as checkpoint_count,
    SUM(file_size_bytes) / (1024*1024*1024) as total_size_gb,
    COUNT(CASE WHEN status = 'valid' THEN 1 END) as valid_count,
    COUNT(CASE WHEN status = 'corrupted' THEN 1 END) as corrupted_count,
    MAX(created_at) as latest_checkpoint
FROM pipeline_checkpoints
GROUP BY level, processor_name
ORDER BY level, processor_name;

-- Processing progress detail view
CREATE OR REPLACE VIEW processing_progress_detail AS
SELECT 
    pj.job_name,
    pj.job_type,
    pj.status as job_status,
    ps.step_name,
    ps.processor_name,
    ps.total_items,
    ps.processed_items,
    ps.failed_items,
    CASE 
        WHEN ps.total_items > 0 
        THEN ROUND((ps.processed_items::NUMERIC / ps.total_items) * 100, 2)
        ELSE 0 
    END as progress_percent,
    ps.status as step_status,
    ps.started_at,
    ps.completed_at,
    CASE 
        WHEN ps.completed_at IS NOT NULL AND ps.started_at IS NOT NULL
        THEN EXTRACT(EPOCH FROM (ps.completed_at - ps.started_at))
        ELSE NULL
    END as duration_seconds
FROM processing_jobs pj
JOIN processing_steps ps ON pj.id = ps.parent_job_id
ORDER BY pj.id DESC, ps.created_at;

-- Export summary view
CREATE OR REPLACE VIEW export_summary AS
SELECT 
    em.grid_id,
    g.name as grid_name,
    em.export_type,
    COUNT(*) as file_count,
    SUM(em.file_size_bytes) as total_size_bytes,
    pg_size_pretty(SUM(em.file_size_bytes)) as total_size_human,
    MIN(em.created_at) as first_export,
    MAX(em.created_at) as latest_export,
    COUNT(CASE WHEN em.expires_at IS NULL OR em.expires_at > NOW() THEN 1 END) as active_files,
    COUNT(CASE WHEN em.expires_at IS NOT NULL AND em.expires_at <= NOW() THEN 1 END) as expired_files
FROM export_metadata em
JOIN grids g ON em.grid_id = g.id
GROUP BY em.grid_id, g.name, em.export_type;

-- Raster processing status view
CREATE OR REPLACE VIEW raster_processing_status AS
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

-- Cache efficiency summary view
CREATE OR REPLACE VIEW cache_efficiency_summary AS
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

-- Processing queue summary view
CREATE OR REPLACE VIEW processing_queue_summary AS
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

-- ==============================================================================
-- FUNCTIONS
-- ==============================================================================

-- Update grid cell count function
CREATE OR REPLACE FUNCTION update_grid_cell_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE grids 
        SET total_cells = (
            SELECT COUNT(*) 
            FROM grid_cells 
            WHERE grid_id = NEW.grid_id
        )
        WHERE id = NEW.grid_id;
        RETURN NEW;
    END IF;
    
    IF TG_OP = 'DELETE' THEN
        UPDATE grids 
        SET total_cells = (
            SELECT COUNT(*) 
            FROM grid_cells 
            WHERE grid_id = OLD.grid_id
        )
        WHERE id = OLD.grid_id;
        RETURN OLD;
    END IF;
    
    IF TG_OP = 'UPDATE' THEN
        IF OLD.grid_id != NEW.grid_id THEN
            UPDATE grids 
            SET total_cells = (
                SELECT COUNT(*) 
                FROM grid_cells 
                WHERE grid_id = OLD.grid_id
            )
            WHERE id = OLD.grid_id;
        END IF;
        
        UPDATE grids 
        SET total_cells = (
            SELECT COUNT(*) 
            FROM grid_cells 
            WHERE grid_id = NEW.grid_id
        )
        WHERE id = NEW.grid_id;
        RETURN NEW;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Update grid processing status with validation
CREATE OR REPLACE FUNCTION update_grid_processing_status(
    p_grid_id UUID,
    p_new_status processing_status_enum,
    p_metadata JSONB DEFAULT NULL
) RETURNS BOOLEAN AS $$
DECLARE
    current_status processing_status_enum;
    transition_allowed BOOLEAN := FALSE;
BEGIN
    SELECT processing_status INTO current_status 
    FROM grids WHERE id = p_grid_id;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Grid not found: %', p_grid_id;
    END IF;
    
    transition_allowed := CASE 
        WHEN current_status = 'pending' AND p_new_status IN ('initializing', 'failed') THEN TRUE
        WHEN current_status = 'initializing' AND p_new_status IN ('grid_generation', 'failed') THEN TRUE
        WHEN current_status = 'grid_generation' AND p_new_status IN ('species_intersection', 'failed', 'partial') THEN TRUE
        WHEN current_status = 'species_intersection' AND p_new_status IN ('feature_computation', 'failed', 'partial') THEN TRUE
        WHEN current_status = 'feature_computation' AND p_new_status IN ('climate_extraction', 'analysis_ready', 'failed', 'partial') THEN TRUE
        WHEN current_status = 'climate_extraction' AND p_new_status IN ('analysis_ready', 'failed', 'partial') THEN TRUE
        WHEN current_status = 'analysis_ready' AND p_new_status IN ('completed', 'failed') THEN TRUE
        WHEN current_status = 'partial' AND p_new_status IN ('feature_computation', 'climate_extraction', 'analysis_ready', 'completed', 'failed') THEN TRUE
        WHEN current_status = 'failed' AND p_new_status IN ('pending', 'initializing') THEN TRUE
        WHEN current_status = p_new_status THEN TRUE
        ELSE FALSE
    END;
    
    IF NOT transition_allowed THEN
        RAISE EXCEPTION 'Invalid status transition from % to %', current_status, p_new_status;
    END IF;
    
    UPDATE grids 
    SET processing_status = p_new_status,
        metadata = CASE 
            WHEN p_metadata IS NOT NULL THEN metadata || p_metadata 
            ELSE metadata 
        END
    WHERE id = p_grid_id;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Cleanup expired exports function
CREATE OR REPLACE FUNCTION cleanup_expired_exports() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM export_metadata 
    WHERE expires_at IS NOT NULL AND expires_at <= NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Cleanup resampling cache function
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

-- Update tile statistics function
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

-- Get next processing task function
CREATE OR REPLACE FUNCTION get_next_processing_task(
    p_queue_type VARCHAR(50),
    p_worker_id VARCHAR(100)
) RETURNS UUID AS $$
DECLARE
    task_id UUID;
BEGIN
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

-- Update updated_at column function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ==============================================================================
-- TRIGGERS
-- ==============================================================================

-- Grid cell count trigger
CREATE TRIGGER trigger_update_grid_cell_count
    AFTER INSERT OR UPDATE OR DELETE ON grid_cells
    FOR EACH ROW
    EXECUTE FUNCTION update_grid_cell_count();

-- Updated_at triggers
CREATE TRIGGER trigger_update_raster_sources_updated_at
    BEFORE UPDATE ON raster_sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ==============================================================================
-- COMMENTS
-- ==============================================================================

-- Type comments
COMMENT ON TYPE processing_status_enum IS 'Processing stages for grid computation pipeline';
COMMENT ON TYPE raster_status_enum IS 'Processing status for raster data sources';
COMMENT ON TYPE tile_status_enum IS 'Processing status for individual raster tiles';

-- Table comments
COMMENT ON TABLE grids IS 'Grid system definitions and metadata';
COMMENT ON TABLE grid_cells IS 'Individual cells within grid systems';
COMMENT ON TABLE species_ranges IS 'Species distribution data from various sources';
COMMENT ON TABLE species_grid_intersections IS 'Pre-computed species presence in grid cells';
COMMENT ON TABLE features IS 'Computed features for each grid cell';
COMMENT ON TABLE climate_data IS 'Climate variables extracted for grid cells';
COMMENT ON TABLE experiments IS 'Experimental runs and their configurations';
COMMENT ON TABLE processing_jobs IS 'Long-running processing job tracking';
COMMENT ON TABLE pipeline_checkpoints IS 'Checkpoint data for pipeline resume functionality';
COMMENT ON TABLE processing_steps IS 'Fine-grained progress tracking for processing jobs';
COMMENT ON TABLE file_processing_status IS 'Individual file processing progress tracking';
COMMENT ON TABLE resampled_datasets IS 'Metadata for resampled raster datasets';
COMMENT ON TABLE data_cleaning_log IS 'Log of data cleaning operations performed on rasters';
COMMENT ON TABLE raster_merge_log IS 'Log of raster merge operations and their metadata';
COMMENT ON TABLE normalization_parameters IS 'Storage for normalization parameters used in data preparation';
COMMENT ON TABLE raster_sources IS 'Metadata for raster data files used in analysis';
COMMENT ON TABLE raster_tiles IS 'Spatial tiles for efficient raster processing and access';
COMMENT ON TABLE resampling_cache IS 'Cache for expensive raster resampling operations';
COMMENT ON TABLE processing_queue IS 'Queue for managing distributed raster processing tasks';
COMMENT ON TABLE export_metadata IS 'Tracks generated output files and export metadata';

-- Function comments
COMMENT ON FUNCTION update_grid_processing_status IS 'Updates grid processing status with validation of allowed transitions';
COMMENT ON FUNCTION cleanup_expired_exports IS 'Removes expired export files from metadata tracking';
COMMENT ON FUNCTION cleanup_resampling_cache IS 'Removes old and rarely accessed cache entries';
COMMENT ON FUNCTION get_next_processing_task IS 'Atomically assigns next available task to worker';

-- Column comments for test data tracking
COMMENT ON COLUMN grids.created_by IS 'Identifies who/what created this record - use test_ prefix for test data';
COMMENT ON COLUMN grid_cells.created_by IS 'Identifies who/what created this record - use test_ prefix for test data';
COMMENT ON COLUMN species_ranges.created_by IS 'Identifies who/what created this record - use test_ prefix for test data';
COMMENT ON COLUMN features.created_by IS 'Identifies who/what created this record - use test_ prefix for test data';
COMMENT ON COLUMN experiments.last_checkpoint_id IS 'Reference to last checkpoint for resume functionality';
COMMENT ON COLUMN processing_jobs.last_checkpoint_id IS 'Reference to last checkpoint for resume functionality';
COMMENT ON COLUMN processing_jobs.resume_metadata IS 'Additional metadata needed for job resume';