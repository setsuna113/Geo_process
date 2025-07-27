# Geo Biodiversity Analysis Pipeline - User Guide

## Overview

This pipeline processes large raster datasets for biodiversity analysis with comprehensive features:
- **Skip-resampling**: Automatically skips when resolution already matches
- **Database storage**: All results stored for querying and analysis
- **CSV export**: Chunked export with optional compression
- **SOM analysis**: Self-Organizing Maps for spatial pattern analysis
- **Checkpoints**: Automatic saving and resuming at each stage
- **Process control**: Full pause/resume/stop functionality

## Quick Start

### Basic Pipeline Run
```bash
# Start the pipeline (runs in foreground)
./run_pipeline.sh

# Start with custom experiment name
./run_pipeline.sh --experiment-name "my_analysis_2025"
```

### Daemon Mode (Background Process)
```bash
# Start in background (detached)
./run_pipeline.sh --daemon

# Start daemon with custom process name
./run_pipeline.sh --daemon --process-name "biodiversity_analysis_main"
```

## Process Management

### Starting Processes
```bash
# Foreground (attached to terminal)
./run_pipeline.sh --experiment-name "test_run"

# Background daemon (detached, survives terminal close)
./run_pipeline.sh --daemon --process-name "main_analysis"

# Resume from checkpoint
./run_pipeline.sh --resume --process-name "main_analysis"
```

### Process Control Commands
```bash
# Check process status
./run_pipeline.sh --process-name <name> --signal status

# Pause running process (SIGUSR1)
./run_pipeline.sh --process-name <name> --signal pause

# Resume paused process (SIGUSR2) 
./run_pipeline.sh --process-name <name> --signal resume

# Stop process gracefully (SIGTERM)
./run_pipeline.sh --process-name <name> --signal stop
```

### Process Monitoring
```bash
# List all managed processes
python scripts/process_manager.py status

# Show specific process details
python scripts/process_manager.py status <process_name>

# View live logs (follow mode)
python scripts/process_manager.py logs <process_name> --follow

# View last 100 lines of logs
python scripts/process_manager.py logs <process_name> --lines 100
```

## Advanced Usage

### Process Manager Direct Commands
```bash
# Start process with custom settings
python scripts/process_manager.py start \
    --name "custom_analysis" \
    --experiment-name "experiment_2025" \
    --analysis-method som \
    --daemon

# List all processes with details
python scripts/process_manager.py status

# Show resource usage
python scripts/process_manager.py resources
```

### Checkpoint Management
```bash
# List available checkpoints
python scripts/process_manager.py checkpoints list

# Show checkpoint details
python scripts/process_manager.py checkpoints info <checkpoint_id>

# Clean up old checkpoints (keep last 7 days)
python scripts/process_manager.py checkpoints cleanup --days 7
```

### Experiment Management
```bash
# List recent experiments
python scripts/process_manager.py experiments --limit 10

# Show experiment details
python scripts/process_manager.py experiments <experiment_id>
```

## Configuration

### Main Configuration File: `config.yml`

Key settings for pipeline behavior:

```yaml
# Target resolution for resampling
resampling:
  target_resolution: 0.016667  # ~5km at equator
  
# Skip-resampling when resolution matches
resampling:
  cache_resampled: true  # Cache results to avoid reprocessing

# SOM analysis settings
som_analysis:
  default_grid_size: [8, 8]  # 8x8 SOM grid
  iterations: 1000
  subsample_ratio: 1.0  # Use full dataset

# Export settings  
export:
  compress: false  # Set to true for gzip compression
  chunk_size: 10000  # Rows per chunk
  include_metadata: true
```

### Dataset Configuration
```yaml
datasets:
  target_datasets:
    - name: "plants-richness"
      path: "/maps/mwd24/richness/daru-plants-richness.tif"
      enabled: true
    - name: "terrestrial-richness"
      path: "/maps/mwd24/richness/iucn-terrestrial-richness.tif" 
      enabled: true
```

## Pipeline Stages

The pipeline consists of 5 stages that run sequentially:

1. **Data Load**: Validate and load dataset configurations
2. **Resample**: Resample to target resolution (or skip if matching)
3. **Merge**: Merge all datasets into single NetCDF file
4. **Export**: Export merged data to CSV format
5. **Analysis**: Run SOM analysis and generate reports

Each stage automatically saves a checkpoint upon completion.

## Debugging and Troubleshooting

### Log Files
```bash
# View real-time logs
python scripts/process_manager.py logs <process_name> --follow

# Search logs for errors
python scripts/process_manager.py logs <process_name> | grep -i error

# View debug information
python scripts/process_manager.py logs <process_name> --lines 200
```

### Debug Mode
```bash
# Run with debug logging
export GEO_LOG_LEVEL=DEBUG
./run_pipeline.sh

# Check process manager logs
tail -f logs/process_manager.log
```

### Common Issues

**Pipeline hangs during resampling:**
```bash
# Check memory usage
python scripts/process_manager.py resources

# Pause and inspect
./run_pipeline.sh --process-name <name> --signal pause
python scripts/process_manager.py status <name>
```

**Process killed unexpectedly:**
```bash
# Always use daemon mode for long runs
./run_pipeline.sh --daemon --process-name "safe_run"

# Check system resources
python scripts/process_manager.py resources
```

**Resume from failure:**
```bash
# Resume automatically finds latest checkpoint
./run_pipeline.sh --resume --process-name <name>

# Or start new with same experiment name (auto-resume)
./run_pipeline.sh --experiment-name <existing_experiment>
```

## Output Structure

```
outputs/<experiment_id>/
├── merged_dataset.nc              # NetCDF merged data
├── merged_data_<id>_<time>.csv    # Exported CSV
├── merged_data_<id>_<time>.meta.json  # Export metadata
├── som_Analysis_<id>/             # SOM analysis results
│   ├── som_report.txt            # Analysis report
│   ├── som_statistics.json       # Statistics
│   ├── cluster_assignments.csv   # Cell clusters
│   └── visualizations/           # Maps and plots
└── pipeline_report.json          # Full pipeline report
```

## Process Safety

### Avoiding Process Termination
- **Always use `--daemon` for long-running processes**
- Daemon processes survive terminal disconnection
- Use `screen` or `tmux` for additional safety:
  ```bash
  tmux new-session -d -s geo_pipeline './run_pipeline.sh --daemon'
  ```

### Resource Management
- Monitor with: `python scripts/process_manager.py resources`
- Set memory limits in config: `processing.memory_limit_gb`
- Use chunking for large datasets: `processing.chunk_size`

### Graceful Shutdown
```bash
# Always stop gracefully (saves checkpoint)
./run_pipeline.sh --process-name <name> --signal stop

# Emergency kill (loses progress)
kill -9 <pid>  # Only if absolutely necessary
```

## Examples

### Complete Analysis Workflow
```bash
# 1. Start background analysis
./run_pipeline.sh --daemon --process-name "analysis_2025" --experiment-name "biodiversity_study"

# 2. Monitor progress
python scripts/process_manager.py status analysis_2025

# 3. Check logs if needed
python scripts/process_manager.py logs analysis_2025 --follow

# 4. Results available in outputs/ when complete
```

### Pause and Resume Workflow
```bash
# Start process
./run_pipeline.sh --daemon --process-name "long_analysis"

# Pause during execution (saves checkpoint)
./run_pipeline.sh --process-name "long_analysis" --signal pause

# Check status
python scripts/process_manager.py status long_analysis

# Resume from where it left off
./run_pipeline.sh --process-name "long_analysis" --signal resume
```

### Recovery from Interruption
```bash
# If process was killed or system restarted
./run_pipeline.sh --resume --process-name "recovery_run" --experiment-name "original_experiment"

# The pipeline will automatically resume from the latest checkpoint
```

### Multiple Parallel Analyses
```bash
# Start multiple analyses with different datasets
./run_pipeline.sh --daemon --process-name "analysis_1" --experiment-name "dataset_A"
./run_pipeline.sh --daemon --process-name "analysis_2" --experiment-name "dataset_B"

# Monitor all processes
python scripts/process_manager.py status

# Track specific process
python scripts/process_manager.py logs analysis_1 --follow
```

## Performance Tips

1. **Use daemon mode** for long analyses to avoid interruption
2. **Monitor resources** regularly with `python scripts/process_manager.py resources`
3. **Adjust chunk sizes** in config based on available memory
4. **Use compression** for CSV exports to save disk space
5. **Clean up checkpoints** periodically to free disk space
6. **Use tmux/screen** for additional process safety

## Getting Help

```bash
# Show all available commands
./run_pipeline.sh --help

# Process manager help
python scripts/process_manager.py --help

# Command-specific help
python scripts/process_manager.py start --help
python scripts/process_manager.py logs --help
python scripts/process_manager.py checkpoints --help
```

## Process Lifecycle

### Normal Flow
1. `./run_pipeline.sh --daemon` → Process starts in background
2. Pipeline runs through all 5 stages with checkpoints
3. Results saved to `outputs/<experiment_id>/`
4. Process completes and exits cleanly

### With Interruption
1. Process paused/stopped → Checkpoint automatically saved
2. `./run_pipeline.sh --resume` → Resumes from last checkpoint
3. Pipeline continues from exact point of interruption
4. Normal completion

### Monitoring Commands
```bash
# Essential monitoring commands
python scripts/process_manager.py status          # Process overview
python scripts/process_manager.py resources       # System resources
python scripts/process_manager.py logs <name> -f  # Live logs
```