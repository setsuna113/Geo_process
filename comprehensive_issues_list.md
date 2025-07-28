# Comprehensive List of Issues to Fix

## Critical Issues (Data Corruption/Pipeline Failures)

### 1. **Coordinate Mapping Bug in Merge Stage** 🔴
- **Issue**: Terrestrial data has 40-pixel Y-offset due to different bounds
- **Impact**: Wrong values in output (e.g., 152 instead of 211)
- **Fix**: Correct the offset calculation in `_align_data_to_common_grid()`
- **File**: `src/pipelines/stages/merge_stage.py`

### 2. **Merge Ignores DB, Re-reads TIF Files** 🔴
- **Issue**: `load_passthrough_data()` reads from TIF instead of DB passthrough tables
- **Impact**: Redundant I/O and coordinate misalignment
- **Fix**: Modify to read from `passthrough_*_richness` tables
- **File**: `src/processors/data_preparation/resampling_processor.py`

### 3. **Memory Check Too Strict** 🔴
- **Issue**: Fails when 797.8GB available but 800GB required (2.2GB difference)
- **Impact**: Pipeline fails unnecessarily
- **Fix**: Add 5% tolerance margin
- **File**: `src/pipelines/orchestrator.py` line 643

### 4. **Permission Denied on Existing Files** 🔴
- **Issue**: Cannot overwrite `merged_dataset.nc` when it exists
- **Impact**: Pipeline fails on retry
- **Fix**: Check and remove existing file before writing
- **Location**: Multiple stages when writing outputs

### 5. **Export Stage Failure** 🔴
- **Issue**: `run_pipeline.sh` fails at export stage (mentioned in memory_pipeline_status.md)
- **Impact**: Cannot complete full pipeline
- **Root Cause**: Likely related to file permissions or memory

## Missing Features (Robustness Issues)

### 6. **No DB Status Detection** 🟡
- **Issue**: Pipeline doesn't detect if DB data is corrupted or complete
- **Impact**: May process bad data or re-process completed data
- **Fix**: Add checks for:
  - Table existence and row counts
  - Data integrity (checksums)
  - Completion markers

### 7. **No Data Validation** 🟡
- **Issue**: No verification that merged values match source
- **Impact**: Silent data corruption goes undetected
- **Fix**: Add spot checks comparing output to source

## Architectural Improvements

### 8. **Remove NetCDF Intermediate Step** 🟢
- **Current**: DB → Merge → NetCDF → Export → CSV/Parquet
- **Improved**: DB → Export → Parquet (direct)
- **Benefit**: Faster, simpler, less disk I/O

### 9. **Implement Direct DB → Parquet Export** 🟢
- **Current**: Must go through NetCDF first
- **Improved**: Direct SQL → Parquet using pandas
- **Benefit**: 4x faster for ML workflows

## Cleanup Tasks

### 10. **Clean Up Test Files** 🔵
From memory_pipeline_status.md:
- `test_pipeline_skip.py`
- `test_skip_control.py`
- `merge_debug.log`
- `full_pipeline.log`

## Additional Issues from Analysis

### 11. **Skip Control Configuration**
- Currently uses 24-hour freshness check
- Should be configurable per dataset
- Should check source file timestamps

### 12. **Experiment Tracking**
- Multiple experiments show as "running" in DB
- No cleanup of failed experiments
- No way to resume from specific stage

### 13. **Memory Estimation**
- Merge stage thinks it needs 800GB for 77MB output
- Overestimates by ~10,000x
- Should base on chunk size, not full dataset

## Priority Order for Fixes

1. **First**: Fix data corruption issues (1, 2, 8)
2. **Second**: Fix pipeline failures (3, 4, 5)
3. **Third**: Add robustness (6, 7)
4. **Fourth**: Architectural improvements (8, 9)
5. **Last**: Cleanup and optimization (10, 11, 12, 13)

## New Workflow After Fixes

```
TIF → ResampleStage → DB passthrough tables
                           ↓
                      ExportStage → Parquet (required for ML)
                           ↓        ↘ CSV (optional)
                      AnalysisStage (SOM/PCA)
```

Note: MergeStage eliminated - ExportStage handles joining datasets directly from DB.