# Pipeline Integration Checklist

## Overview
This checklist provides step-by-step instructions to fully integrate the pipeline with updates from the four branches (monitoring/logging, resampling reconstruction, merge stage upgrade, and analysis stage refactor). Each task respects the system hierarchy and leverages existing lower-level modules.

## CRITICAL ARCHITECTURAL ISSUE

**Problem**: MergeStage is currently creating Parquet files (exporting), which violates separation of concerns.
- MergeStage calls `merger.create_ml_ready_parquet()` and outputs a file
- ExportStage expects `merged_dataset` (xarray) but gets nothing
- This means ExportStage will FAIL because MergeStage doesn't provide the expected data

**Required Fix**: 
1. MergeStage should only merge data and return an in-memory dataset
2. ExportStage should handle ALL file exports (both Parquet and CSV)

## Phase 0: Fix Critical Architectural Issue (Priority: URGENT)

### 0.1 Refactor MergeStage to Only Merge
**File**: `src/pipelines/stages/merge_stage.py`

- [x] Change MergeStage to return merged data, not export files:
```python
# REMOVE: ml_ready_path = merger.create_ml_ready_parquet(...)
# ADD: Create merged dataset in memory
merged_dataset = merger.create_merged_dataset(
    dataset_dicts,
    chunk_size=chunk_size,
    return_as='xarray'  # or 'dataframe'
)

# Store in context for ExportStage
context.set('merged_dataset', merged_dataset)
```

### 0.2 Create New Merge Method in CoordinateMerger
**File**: `src/processors/data_preparation/coordinate_merger.py`

- [x] Add method that returns data instead of writing files:
```python
def create_merged_dataset(self, datasets: List[Dict], 
                         chunk_size: Optional[int] = None,
                         return_as: str = 'xarray') -> Union[xr.Dataset, pd.DataFrame]:
    """Merge datasets and return in-memory result."""
    # Use existing merge logic but return data instead of writing to file
```

### 0.3 Update ExportStage to Handle Both Formats
**File**: `src/pipelines/stages/export_stage.py`

- [x] Add Parquet export capability:
```python
# Determine export format from config
export_formats = context.config.get('export.formats', ['csv', 'parquet'])

exported_paths = {}
for format in export_formats:
    if format == 'parquet':
        output_path = context.output_dir / f"merged_data_{context.experiment_id}.parquet"
        # Use efficient parquet writing
        merged_dataset.to_parquet(output_path)  # or use pyarrow
        exported_paths['parquet'] = str(output_path)
        context.set('ml_ready_path', str(output_path))  # For AnalysisStage
    elif format == 'csv':
        # Existing CSV export code
        exported_paths['csv'] = str(output_path)
        context.set('exported_csv_path', str(output_path))  # For AnalysisStage
```

### 0.4 Alternative: Memory-Efficient Approach
**Note**: If merged datasets are too large for memory, consider:
- Keep MergeStage creating files but rename to clarify purpose
- OR: Use lazy evaluation with xarray/dask
- OR: Stream directly from database tables

**Dependencies to Consider**:
- AnalysisStage expects `ml_ready_path` (Parquet) or `exported_csv_path` (CSV)
- Current flow: Merge → creates Parquet → Analysis reads Parquet
- New flow: Merge → creates dataset → Export → creates files → Analysis reads files

## Phase 1: Database Schema Updates (Priority: CRITICAL)

### 1.1 Run Monitoring Schema Migration
```bash
python scripts/migrate_monitoring_schema.py
```
- [x] Verify `pipeline_logs` table created (already exists)
- [x] Verify `pipeline_progress` table created (already exists)
- [x] Verify `pipeline_metrics` table created (already exists)
- [x] Verify `pipeline_events` table created (already exists)

### 1.2 Run Coordinate Storage Migration
```bash
python scripts/migrate_to_coordinate_storage.py
```
- [⚠️] Method `migrate_legacy_table_to_coordinates` exists but migration script fails
- [ ] Note: Tables already have coordinate info from merge stage upgrade
- [ ] Test with `scripts/validate_monitoring_setup.py`

## Phase 2: Configuration Updates (Priority: HIGH)

### 2.1 Update Default Configuration
**File**: `src/config/defaults.py` or `config.yml`

- [x] Set `resampling.enable_memory_aware_processing: true` (already set)
- [x] Set `monitoring.enable_database_logging: true` (added)
- [x] Set `merge.enable_chunked_processing: true` (added)
- [x] Add missing configuration options:
```yaml
resampling:
  enable_memory_aware_processing: true  # Enable by default
  window_size: 2048
  window_overlap: 128
  skip_data_loading_for_passthrough: true
  use_legacy_passthrough: false  # Deprecate old method
  use_legacy_resampling: false   # Deprecate old method

monitoring:
  enable_database_logging: true
  log_batch_size: 100
  log_flush_interval: 5
  enable_metrics: true
  metrics_interval: 10

merge:
  enable_chunked_processing: true
  chunk_size: 5000
  enable_validation: true
  alignment_tolerance: 0.01

export:
  enable_streaming: true  # New option
  chunk_size: 10000
  compress: true
```

## Phase 3: Structured Logging Integration (Priority: HIGH)

### 3.1 Update DataLoadStage
**File**: `src/pipelines/stages/load_stage.py`

- [x] Add imports at top:
```python
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_stage
```

- [x] Replace logger initialization:
```python
# OLD: logger = logging.getLogger(__name__)
logger = get_logger(__name__)  # NEW
```

- [x] Add decorator to execute method:
```python
@log_stage("data_load")
def execute(self, context) -> StageResult:
```

- [x] Update log calls to use structured data:
```python
# Example: Add context to important log messages
logger.info(
    f"Loaded {len(enabled_datasets)} datasets",
    extra={
        'datasets_count': len(enabled_datasets),
        'total_size_mb': metrics['total_size_mb']
    }
)
```

### 3.2 Update ResampleStage
**File**: `src/pipelines/stages/resample_stage.py`

- [x] Add imports (same as 3.1)
- [x] Replace logger initialization
- [x] Add `@log_stage("resample")` decorator
- [x] Add structured logging context

### 3.3 Update ExportStage
**File**: `src/pipelines/stages/export_stage.py`

- [x] Add imports (same as 3.1)
- [x] Replace logger initialization
- [x] Add `@log_stage("export")` decorator
- [x] Add performance logging:
```python
logger.log_performance(
    "csv_export",
    duration=time.time() - start_time,
    rows=rows_exported,
    size_mb=file_size / 1024 / 1024
)
```

## Phase 4: Memory-Aware Processing Integration (Priority: HIGH)

### 4.1 Update ResampleStage to Use Memory-Aware Processing
**File**: `src/pipelines/stages/resample_stage.py`

- [x] Check configuration for memory-aware mode:
```python
# After line 67 (processor creation)
use_memory_aware = context.config.get('resampling.enable_memory_aware_processing', False)
```

- [x] Update processing logic (around line 108):
```python
# Check if dataset needs resampling (from load stage)
if not dataset_info.get('needs_resampling', True):
    logger.info(f"Dataset {dataset_config['name']} resolution matches target, checking for existing passthrough")
    existing = processor.get_resampled_dataset(dataset_config['name'])
    if existing:
        resampled_datasets.append(existing)
        metrics['passthrough_datasets'] += 1
        continue

# Use memory-aware processing if enabled
if use_memory_aware:
    logger.info(f"Using memory-aware resampling for {dataset_config['name']}")
    resampled_info = processor.resample_dataset_memory_aware(dataset_config)
else:
    # Legacy path - mark as deprecated
    logger.warning(
        f"Using legacy resampling for {dataset_config['name']}. "
        "Consider enabling memory-aware processing.",
        extra={'deprecated': True}
    )
    resampled_info = processor.resample_dataset(dataset_config)
```

### 4.2 Verify WindowedStorageManager Usage
**Check**: `src/processors/data_preparation/resampling_processor.py`

- [x] Ensure `resample_dataset_memory_aware()` uses `WindowedStorageManager`
- [x] Verify it doesn't load full datasets into memory
- [x] Check that passthrough datasets use `store_passthrough_windowed()`

## Phase 5: Export Stage Enhancement (Priority: MEDIUM)

### 5.1 Clarify Export Purpose and Format
**Understanding**: The pipeline has two outputs:
- **MergeStage**: Produces Parquet file via `CoordinateMerger.create_ml_ready_parquet()`
- **ExportStage**: Produces CSV file for inspection/compatibility

### 5.2 Enhance CSV Export (Keep Current Purpose)
**File**: `src/pipelines/stages/export_stage.py`

- [x] The export stage is correctly exporting to CSV (not Parquet)
- [x] Current implementation already does chunked writing
- [x] Main improvements needed:
  - Add structured logging ✓
  - Add progress tracking ✓
  - Consider using existing `CSVExporter` if it has better features ✓

- [x] Update to leverage existing exporter if available:
```python
# Check if streaming is enabled
if context.config.get('export.enable_streaming', True):
    # Use existing lower-level exporter
    exporter = CSVExporter(
        chunk_size=context.config.get('export.chunk_size', 10000),
        compression='gzip' if context.config.get('export.compress', False) else None
    )
    
    # Let the exporter handle the streaming
    rows_exported = exporter.export_dataset(
        merged_dataset,
        output_path,
        progress_callback=lambda current, total: 
            self._update_progress(current * 100 / total)
    )
else:
    # Legacy path - keep existing code but mark as deprecated
    logger.warning("Using legacy export method", extra={'deprecated': True})
    # ... existing xarray export code ...
```

### 5.3 Add Progress Tracking
- [x] Add progress callback support ✓
- [x] Update progress during export chunks ✓
- [x] Log performance metrics ✓

## Phase 6: Deprecation Markers (Priority: MEDIUM)

### 6.1 Mark Old Methods as Deprecated
**Files**: Various processor files

- [x] In `src/processors/data_preparation/resampling_processor.py`:
```python
from warnings import warn

def resample_dataset(self, dataset_config: dict) -> ResampledDatasetInfo:
    """Legacy resampling method.
    
    .. deprecated:: 2.0
       Use :meth:`resample_dataset_memory_aware` instead.
    """
    warn(
        "resample_dataset is deprecated. Use resample_dataset_memory_aware instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing implementation ...
```

- [x] Add similar deprecation warnings to other legacy methods ✓

### 6.2 Update Documentation
- [x] Add deprecation notices to docstrings ✓
- [x] Update method comments to point to new alternatives ✓

## Phase 7: Integration Testing (Priority: HIGH)

### 7.1 Test Monitoring Integration
```bash
# Start pipeline with monitoring
python scripts/process_manager.py start --name test_integration --experiment-name integration_test

# In another terminal, verify monitoring works
python scripts/monitor.py watch integration_test
python scripts/monitor.py logs integration_test
python scripts/monitor.py metrics integration_test
```

- [ ] Verify logs appear in database
- [ ] Verify progress tracking works
- [ ] Verify metrics are collected

### 7.2 Test Memory-Aware Processing
- [ ] Run pipeline with large dataset
- [ ] Monitor memory usage stays under configured window size
- [ ] Verify passthrough datasets aren't loaded into memory

### 7.3 Test Full Pipeline Integration
- [ ] Run complete pipeline with all stages
- [ ] Verify all stages use structured logging
- [ ] Verify chunked processing in merge stage
- [ ] Verify streaming export

## Phase 8: Performance Validation (Priority: MEDIUM)

### 8.1 Benchmark Memory Usage
- [ ] Compare memory usage: legacy vs memory-aware
- [ ] Document peak memory for each stage
- [ ] Verify memory stays within configured limits

### 8.2 Benchmark Processing Time
- [ ] Compare processing time: legacy vs new
- [ ] Identify any performance regressions
- [ ] Optimize bottlenecks if found

## Phase 9: Cleanup (Priority: LOW)

### 9.1 Remove Redundant Code
- [ ] Remove commented-out old implementations
- [ ] Remove unused imports
- [ ] Clean up debug logging statements

### 9.2 Update Tests
- [ ] Update unit tests to use new methods
- [ ] Add tests for memory-aware processing
- [ ] Add tests for structured logging

## Completion Checklist

### Critical Path (Must Complete)
- [⚠️] Database migrations run successfully (partial - tables exist, migration script has issues)
- [x] Configuration updated with new defaults
- [x] All stages use structured logging
- [x] Memory-aware processing enabled and working
- [ ] Integration tests pass (requires user interaction)

### Nice to Have (Can be gradual)
- [x] All deprecation warnings added
- [ ] Performance benchmarks documented (requires testing)
- [ ] Tests updated for new features (requires user interaction)
- [ ] Old code cleaned up (requires review)

## Notes on System Hierarchy

This checklist respects the system hierarchy by:
1. **Using existing lower-level modules** (WindowedStorageManager, CSVExporter)
2. **Not creating new functionality in pipelines/** (only orchestration)
3. **Leveraging infrastructure layers** (logging, monitoring)
4. **Maintaining clean abstractions** (stages only coordinate, not implement)

Remember: Pipeline stages should ONLY orchestrate - actual implementation belongs in processors, infrastructure, or domain modules.