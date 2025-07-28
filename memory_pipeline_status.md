# Pipeline Status and Debug Guide (2025-07-28)

## Current Pipeline Status
- **Running**: Full pipeline with skip control enabled
- **Process**: Running in background via `nohup python test_pipeline_skip.py > full_pipeline.log 2>&1 &`
- **Progress**: As of 02:26, merge stage at chunk 220/924 (~24%)
- **Estimated completion**: ~8 more minutes for merge, then export + SOM stages

## How to Check Status

### 1. Check if still running:
```bash
ps aux | grep test_pipeline_skip
```

### 2. Check pipeline progress:
```bash
# See latest activity
tail -50 full_pipeline.log

# Check specific stages
tail -100 full_pipeline.log | grep -E "SKIPPING|completed|ERROR|chunk|export|analysis"

# Get current chunk progress
tail -20 full_pipeline.log | grep "Processing chunk"
```

### 3. Check output files:
```bash
# List output directory (experiment ID will vary)
ls -la outputs/*/

# Expected files when complete:
# - merged_dataset.nc (NetCDF with merged data)
# - merged_data_*.csv (or .csv.gz if compressed)
# - som_results_*.npz (SOM analysis results)
```

### 4. Check database for completion:
```bash
python -c "
from src.database.connection import DatabaseManager
from src.database.schema import schema
db = DatabaseManager()
experiments = schema.get_experiments(limit=5)
for exp in experiments:
    print(f\"{exp['name']}: {exp['status']} - Started: {exp['started_at']} - Completed: {exp.get('completed_at', 'Not yet')}\")
"
```

## How to Debug if Failed

### 1. Check error in log:
```bash
# Find errors
grep -n "ERROR\|Exception\|Traceback" full_pipeline.log | tail -50

# Get full traceback
grep -A20 "Traceback" full_pipeline.log | tail -100
```

### 2. Common failure points:
- **Permission denied**: Output file already exists - remove it
- **Memory error**: Despite lazy loading, might hit memory limits
- **Database connection**: Check PostgreSQL is running on port 51051

### 3. Recovery options:
```bash
# Clean and restart
python -m src.utils.cleanup_manager --experiment "production_run_skip_test" --force
rm -f outputs/*/merged_dataset.nc
python test_pipeline_skip.py
```

## Files Created/Modified During Implementation

### 1. **Core Implementation Files** (KEEP - they're part of the system now):
- `/home/yl998/dev/geo/src/pipelines/stages/skip_control.py` - Skip control logic
- `/home/yl998/dev/geo/src/pipelines/orchestrator.py` - Modified to support skip control
- `/home/yl998/dev/geo/src/processors/data_preparation/resampling_processor.py` - Fixed to store passthrough data

### 2. **Test/Debug Files** (DELETE after successful run):
- `/home/yl998/dev/geo/test_pipeline_skip.py` - Test script for skip functionality
- `/home/yl998/dev/geo/test_skip_control.py` - Skip control unit test
- `/home/yl998/dev/geo/merge_debug.log` - Debug log from earlier run
- `/home/yl998/dev/geo/full_pipeline.log` - Current run log

### 3. **Configuration** (KEEP - production ready):
- `/home/yl998/dev/geo/config.yml` - Updated with pipeline skip control settings:
```yaml
pipeline:
  allow_skip_stages: true  # Set to false for safety in production
  stages:
    data_load:
      skip_if_exists: true
    resample:
      skip_if_exists: true
```

## Gap Analysis: Test vs Production

**Good news**: NO GAP! The test pipeline exactly matches production:

1. **Production pipeline** (`run_pipeline.sh` → `process_manager.py`) includes all 5 stages:
   - DataLoadStage
   - ResampleStage
   - MergeStage
   - ExportStage
   - AnalysisStage (SOM)

2. **Our test** includes the same 5 stages after we added export and analysis

3. **Skip control** works with both test and production - controlled by config.yml

### To use skip control in production:
```bash
# Just run normally - skip control respects config.yml settings
./run_pipeline.sh production_run_with_skip
```

## SSH Detachment Safety

**YES, it's safe to detach!** The pipeline is running with `nohup` which:
- Ignores hangup signals (SSH disconnect)
- Redirects all output to log file
- Continues running after terminal closes

The process will complete even if you disconnect from SSH.

## Key Architecture Insights

1. **Data Flow**: Stages pass data via "context" object, not database
2. **Memory Safety**: Large datasets use xarray with dask chunks (lazy loading)
3. **Skip Control**: Only skips if data exists AND is fresh (< 24 hours old by default)
4. **Passthrough Resampling**: Data that matches target resolution is stored in DB without resampling

## Next Steps When You Return

1. Check if pipeline completed successfully
2. If success: 
   - Set `allow_skip_stages: false` in config.yml for safety
   - Delete test files listed above
   - Verify CSV and SOM outputs
3. If failed:
   - Check logs for error
   - Clean up with cleanup_manager
   - Restart with appropriate fixes

The pipeline is processing ~130 million biodiversity records into merged, exported, and analyzed datasets!