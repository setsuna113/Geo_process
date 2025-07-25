 Based on the production script architecture I integrated, here's how to run the skip-resampling pipeline on a cluster:

  Running on Cluster

  1. Basic Execution

  # Start pipeline in daemon mode (recommended for cluster)
  ./scripts/production/run_unified_resampling.sh \
    --daemon \
    --experiment-name "cluster_skip_resampling_run" \
    --process-name "biodiversity_pipeline_$(date +%Y%m%d_%H%M%S)"

  2. With Custom Configuration

  # For production cluster run (not test mode)
  DB_NAME=production_geoprocess_db \
  ./scripts/production/run_unified_resampling.sh \
    --daemon \
    --experiment-name "production_skip_resampling" \
    --process-name "prod_pipeline_001"

  3. Resume from Checkpoint

  # Resume interrupted pipeline
  ./scripts/production/run_unified_resampling.sh \
    --daemon \
    --resume \
    --experiment-name "production_skip_resampling" \
    --process-name "prod_pipeline_001"

  Process Control

  Check Status

  # Check specific process status
  ./scripts/production/run_unified_resampling.sh \
    --process-name "prod_pipeline_001" \
    --signal status

  # Alternative: use process manager directly
  python scripts/process_manager.py status prod_pipeline_001

  Pause Pipeline

  # Pause running pipeline (graceful)
  ./scripts/production/run_unified_resampling.sh \
    --process-name "prod_pipeline_001" \
    --signal pause

  # Or send SIGUSR1 signal
  kill -USR1 <pipeline_pid>

  Resume Pipeline

  # Resume paused pipeline
  ./scripts/production/run_unified_resampling.sh \
    --process-name "prod_pipeline_001" \
    --signal resume

  # Or send SIGUSR2 signal
  kill -USR2 <pipeline_pid>

  Stop Pipeline

  # Graceful stop (saves checkpoint)
  ./scripts/production/run_unified_resampling.sh \
    --process-name "prod_pipeline_001" \
    --signal stop

  # Emergency stop
  kill -TERM <pipeline_pid>

  Monitoring

  View Logs

  # Follow logs in real-time
  python scripts/process_manager.py logs prod_pipeline_001 -f

  # View recent logs
  python scripts/process_manager.py logs prod_pipeline_001 --lines 100

  Check Resource Usage

  # Monitor memory/CPU usage
  python scripts/process_manager.py resources

  List Experiments

  # View experiment history
  python scripts/process_manager.py experiments --limit 20

  Cluster-Specific Recommendations

  1. SLURM Integration

  #!/bin/bash
  #SBATCH --job-name=biodiversity_pipeline
  #SBATCH --time=24:00:00
  #SBATCH --mem=64G
  #SBATCH --cpus-per-task=16

  # Load environment
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate geo

  # Set production configuration
  export DB_NAME=production_geoprocess_db
  export PIPELINE_MEMORY_LIMIT_GB=60
  export PROCESSING_MAX_WORKERS=16

  # Run pipeline
  ./scripts/production/run_unified_resampling.sh \
    --daemon \
    --experiment-name "slurm_${SLURM_JOB_ID}" \
    --process-name "pipeline_${SLURM_JOB_ID}"

  2. Production Configuration (config.yml)

  # Place in project root for production
  pipeline:
    memory_limit_gb: 60.0
    enable_memory_monitoring: true
    memory_check_interval: 30.0

  processing:
    batch_size: 10000
    max_workers: 16
    chunk_size: 50000
    memory_limit_mb: 32768  # 32GB
    enable_chunking: true

  resampling:
    target_resolution: 0.008333  # Production resolution
    allow_skip_resampling: true
    resolution_tolerance: 0.000001

  som_analysis:
    max_pixels_in_memory: 1000000
    default_grid_size: [10, 10]
    iterations: 1000

  3. Checkpoint Management

  # List available checkpoints
  python scripts/process_manager.py checkpoints list --limit 10

  # Get checkpoint details
  python scripts/process_manager.py checkpoints info <checkpoint_id>

  # Clean old checkpoints
  python scripts/process_manager.py checkpoints cleanup --days 7

  Skip-Resampling Verification

  Check Skip-Resampling Status

  The pipeline will automatically log when datasets are skipped:
  ✓ Using existing passthrough dataset: plants-richness
    Resolution: 0.016667° (matches target)
  ✓ Skipped resampling via passthrough for: terrestrial-richness

  Monitor Database Results

  -- Check passthrough datasets
  SELECT name, metadata->'passthrough' as is_passthrough,
         target_resolution, created_at
  FROM resampled_datasets
  WHERE metadata->>'passthrough' = 'true';

  -- View experiment progress
  SELECT name, status, config->'target_resolution' as resolution
  FROM experiments
  ORDER BY created_at DESC LIMIT 5;

  Error Recovery

  Common Issues & Solutions

  1. Memory errors: Adjust pipeline.memory_limit_gb in config.yml
  2. Checkpoint corruption: Use checkpoints cleanup and restart
  3. Database issues: Check DB_NAME and connection settings
  4. Process hanging: Use --signal stop then restart with --resume

  The production script is now fully integrated with the skip-resampling functionality and ready for cluster deployment with proper process control, monitoring, and checkpoint/resume capabilities.

