# Pipeline Issues Summary (2025-07-28)

## Critical Issues Requiring Systematic Updates

### 1. Storage Permission Problems
**Issue**: PermissionError when writing merged_dataset.nc
- **Location**: `/src/pipelines/stages/merge_stage.py:328`
- **Error**: `PermissionError: [Errno 13] Permission denied: 'outputs/.../merged_dataset.nc'`
- **Cause**: File already exists from previous failed attempt, xarray/netCDF4 cannot overwrite
- **Temporary Fix**: Manually delete file before retry
- **Proper Fix Needed**: 
  - Add file existence check before writing
  - Use mode='w' with proper overwrite handling
  - Implement atomic write with temp file + rename

### 2. Overly Strict Memory Check
**Issue**: Pipeline fails when available memory is 2.2GB less than required
- **Location**: `/src/pipelines/orchestrator.py:643` in `_pre_stage_checks()`
- **Error**: `MemoryError: Insufficient memory for stage 'merge': required 800.0GB, available 797.8GB`
- **Impact**: Pipeline fails even though 797GB is more than sufficient
- **Proper Fix Needed**:
  - Add tolerance margin (e.g., 5% or 10GB)
  - Make memory requirements more realistic
  - Allow override flag for memory checks

### 3. Merge Stage Memory Calculation
**Issue**: Merge stage reports needing 800GB for a dataset that produces only 77MB output
- **Location**: Stage memory estimation logic
- **Problem**: Overestimates memory by ~10,000x
- **Proper Fix Needed**:
  - Fix memory calculation to account for lazy loading/chunking
  - Base estimate on chunk size * concurrent chunks, not full dataset

### 4. Pipeline Recovery Issues
**Issue**: Recovery attempt hits same memory check even after successful merge
- **Location**: `/src/pipelines/orchestrator.py:921` in `_attempt_recovery()`
- **Problem**: Doesn't skip completed stages during recovery
- **Proper Fix Needed**:
  - Check for existing outputs before re-running stages
  - Implement proper stage completion detection

### 5. Context Object Creation
**Issue**: PipelineContext requires too many parameters for standalone usage
- **Location**: Used in orchestrator but hard to instantiate manually
- **Problem**: Can't easily run individual stages
- **Proper Fix Needed**:
  - Add factory method or builder pattern
  - Allow minimal context creation for single-stage runs

## Successful Implementations

### ✅ Skip Control System
- Works perfectly for data_load and resample stages
- Correctly detects existing data and skips reprocessing
- Saved significant time on retry

### ✅ Lazy Chunk-Based Merging
- Successfully processes 924 chunks
- Memory efficient (doesn't load full dataset)
- Produces correct output

## Immediate Action Items

1. **Manual CSV Export** (doing now)
2. **Update memory check tolerance** in orchestrator.py
3. **Add file overwrite handling** in merge_stage.py
4. **Fix memory estimation** for merge stage
5. **Improve recovery logic** to skip completed stages

## Code Locations for Updates

```python
# 1. Memory check tolerance (orchestrator.py:643)
if required_memory > available_memory * 0.95:  # Add 5% tolerance

# 2. File overwrite (merge_stage.py:328)
if os.path.exists(output_path):
    os.remove(output_path)
    
# 3. Memory estimation (merge_stage.py)
# Update estimate_memory() to use chunk-based calculation

# 4. Recovery logic (orchestrator.py:921)
# Check for stage outputs before re-execution
```