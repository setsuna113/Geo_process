# Geo Pipeline Execution Guide

## Table of Contents
1. [Overview](#overview)
2. [Running the Full Pipeline](#running-the-full-pipeline)
3. [Pause and Resume](#pause-and-resume)
4. [Process Tracking and Monitoring](#process-tracking-and-monitoring)
5. [Common Command Arguments](#common-command-arguments)
6. [Configuration Fields](#configuration-fields)
7. [Troubleshooting](#troubleshooting)

## Overview

The Geo pipeline is a geospatial biodiversity analysis system that processes large raster datasets through multiple stages. It uses checkpoint-based execution for resumability and tmux for process management.

### Pipeline Stages
1. **Data Preparation** - Loads and normalizes input data
2. **Grid Processing** - Creates spatial grids at multiple resolutions
3. **Feature Extraction** - Computes features for each grid cell
4. **Analysis** - Runs spatial analysis (GWPCA, SOM)
5. **Export** - Exports results in various formats

## Running the Full Pipeline

### Method 1: Using run_analysis.sh (Recommended)

```bash
# Basic run
./run_analysis.sh

# With custom config
./run_analysis.sh --config config_custom.yml

# Dry run (shows what would be executed)
./run_analysis.sh --dry-run

# Resume from checkpoint
./run_analysis.sh --resume
```

### Method 2: Direct Python Execution

```bash
# Basic run
python run_pipeline.py

# With custom config
python run_pipeline.py --config config_custom.yml

# Resume from specific stage
python run_pipeline.py --resume --stage analysis

# Debug mode
python run_pipeline.py --debug
```

### Method 3: Using tmux (for long-running processes)

```bash
# Start new tmux session
tmux new-session -s geo-pipeline

# Run pipeline
python run_pipeline.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t geo-pipeline
```

## Pause and Resume

### Automatic Checkpointing

The pipeline automatically saves progress at:
- End of each stage
- Every N processed items (configurable)
- Before any error occurs

Checkpoint files are stored in: `checkpoints/pipeline_state.json`

### Manual Pause

1. **Graceful shutdown** (saves checkpoint):
   ```bash
   # In tmux session
   Ctrl+C  # Sends SIGINT, triggers checkpoint save
   ```

2. **Check saved state**:
   ```bash
   cat checkpoints/pipeline_state.json | jq .
   ```

### Resume Operations

1. **Resume from last checkpoint**:
   ```bash
   python run_pipeline.py --resume
   ```

2. **Resume from specific stage**:
   ```bash
   python run_pipeline.py --resume --stage feature_extraction
   ```

3. **Resume with modified config**:
   ```bash
   python run_pipeline.py --resume --config config_modified.yml
   ```

## Process Tracking and Monitoring

### Real-time Progress

1. **Console output**:
   - Progress bars for each stage
   - Current item being processed
   - ETA and processing rate
   - Memory usage

2. **Log files**:
   ```bash
   # Main log
   tail -f logs/pipeline_$(date +%Y%m%d).log
   
   # Stage-specific logs
   tail -f logs/stage_analysis_$(date +%Y%m%d).log
   ```

3. **Database monitoring**:
   ```sql
   -- Current experiment status
   SELECT * FROM experiments WHERE status = 'running' ORDER BY created_at DESC;
   
   -- Stage progress
   SELECT stage_name, items_processed, total_items, 
          ROUND(items_processed::numeric / total_items * 100, 2) as percent_complete
   FROM experiment_progress 
   WHERE experiment_id = (SELECT id FROM experiments ORDER BY created_at DESC LIMIT 1);
   ```

### Using process_manager.py

```bash
# Check pipeline status
python scripts/process_manager.py status

# Monitor in real-time
python scripts/process_manager.py monitor

# View stage details
python scripts/process_manager.py stage-info --stage analysis
```

### tmux Monitoring

```bash
# List tmux sessions
tmux ls

# Attach to running session
tmux attach -t geo-pipeline

# Split panes for monitoring
# Ctrl+B, then %  (vertical split)
# Ctrl+B, then "  (horizontal split)
```

## Common Command Arguments

### run_pipeline.py Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--config` | Path to config file | `--config config_custom.yml` |
| `--resume` | Resume from checkpoint | `--resume` |
| `--stage` | Specific stage to run | `--stage analysis` |
| `--debug` | Enable debug logging | `--debug` |
| `--dry-run` | Show execution plan | `--dry-run` |
| `--force` | Override safety checks | `--force` |
| `--workers` | Number of parallel workers | `--workers 8` |
| `--memory-limit` | Max memory usage (GB) | `--memory-limit 32` |

### Stage-specific Arguments

**Data Preparation**:
- `--input-dir`: Directory with input data
- `--validate`: Run data validation
- `--skip-missing`: Continue if files missing

**Analysis**:
- `--algorithm`: Analysis algorithm (gwpca, som)
- `--parameters`: Algorithm parameters JSON
- `--output-format`: Output format (csv, netcdf, geotiff)

## Configuration Fields

### Core Configuration (config.yml)

```yaml
# Database settings
database:
  host: localhost
  port: 5432
  name: geo_biodiversity
  user: geo_user
  password: ${DB_PASSWORD}  # Environment variable
  pool_size: 20
  test_mode: false

# Pipeline control
pipeline:
  checkpoint_interval: 100  # Save every N items
  memory_limit_gb: 32
  parallel_workers: 8
  resume_on_error: true
  
# Stages configuration  
stages:
  data_preparation:
    enabled: true
    batch_size: 1000
    validation_mode: strict
    
  grid_processing:
    enabled: true
    resolutions: [5, 10, 25, 50, 100]  # km
    grid_type: cubic  # or hexagonal
    
  feature_extraction:
    enabled: true
    features:
      - type: climate
        variables: ["temperature", "precipitation"]
      - type: species_richness
        min_occurrences: 5
        
  analysis:
    enabled: true
    algorithms:
      gwpca:
        components: 10
        kernel_type: gaussian
        adaptive: true
      som:
        grid_size: [20, 20]
        iterations: 1000
        
  export:
    enabled: true
    formats: ["csv", "geotiff"]
    output_dir: outputs/
```

### Data Source Configuration

```yaml
data_sources:
  rasters:
    - name: climate_temperature
      path: data/climate/temperature/*.tif
      type: geotiff
      temporal: monthly
      resampling: bilinear
      
    - name: land_cover
      path: data/landcover/lc_*.tif
      type: geotiff
      categorical: true
      
  species:
    source: gbif
    filters:
      min_occurrences: 10
      spatial_uncertainty_max: 1000  # meters
      date_range: [2010, 2023]
```

### Performance Tuning

```yaml
performance:
  cache:
    enabled: true
    size_gb: 8
    ttl_hours: 24
    
  processing:
    chunk_size: 10000
    prefetch_queue: 50
    compression: lz4
    
  memory:
    max_usage_percent: 80
    gc_threshold_gb: 4
    swap_warning: true
```

## Troubleshooting

### Common Issues

1. **Pipeline hangs or crashes**:
   ```bash
   # Check logs
   tail -n 100 logs/pipeline_*.log | grep ERROR
   
   # Check database locks
   psql -d geo_biodiversity -c "SELECT * FROM pg_locks WHERE granted = false;"
   
   # Resume with debug
   python run_pipeline.py --resume --debug
   ```

2. **Out of memory**:
   ```bash
   # Reduce workers and batch size
   python run_pipeline.py --workers 4 --config config_low_memory.yml
   ```

3. **Checkpoint corruption**:
   ```bash
   # Backup corrupted checkpoint
   mv checkpoints/pipeline_state.json checkpoints/pipeline_state.json.bak
   
   # Restore from previous checkpoint
   cp checkpoints/pipeline_state.json.1 checkpoints/pipeline_state.json
   ```

### Performance Monitoring

```bash
# System resources
htop  # CPU and memory usage
iotop  # Disk I/O
nvidia-smi  # GPU usage (if applicable)

# Database performance
psql -d geo_biodiversity -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Pipeline metrics
python scripts/process_manager.py metrics --last-hour
```

### Recovery Procedures

1. **From database failure**:
   ```bash
   # Restart PostgreSQL
   sudo systemctl restart postgresql
   
   # Resume pipeline
   python run_pipeline.py --resume
   ```

2. **From incomplete stage**:
   ```bash
   # Check stage status
   python scripts/process_manager.py stage-info --stage feature_extraction
   
   # Resume from stage start
   python run_pipeline.py --resume --stage feature_extraction --force-restart
   ```

3. **Full reset** (caution):
   ```bash
   # Clear checkpoints
   rm -rf checkpoints/*
   
   # Clear database state
   psql -d geo_biodiversity -c "UPDATE experiments SET status = 'failed' WHERE status = 'running';"
   
   # Start fresh
   python run_pipeline.py
   ```

## Best Practices

1. **Always use tmux** for production runs
2. **Monitor logs** in separate pane/terminal
3. **Set appropriate memory limits** based on system
4. **Regular checkpoint backups** for critical runs
5. **Test configuration** with small dataset first
6. **Use --dry-run** to verify execution plan