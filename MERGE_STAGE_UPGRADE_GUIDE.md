# Comprehensive Merge Stage Upgrade Guide

## Overview
This guide provides a systematic approach to upgrading the merge stage to handle the new windowed storage format from the resampling stage, fix alignment issues, and integrate with the unified monitoring system.

## Guiding Principles
1. **Respect Architecture Hierarchy**: High → Mid → Low level abstractions
2. **No Bypass**: Use existing abstractions, don't create parallel implementations
3. **Preserve Working Code**: Enhance, don't replace functioning components
4. **Database-First**: All data flows through proper schema methods

## Phase 1: Mid-Level Infrastructure Updates (Days 1-3)

### 1.1 Extend RasterAligner for Database Support
**File**: `src/processors/data_preparation/raster_alignment.py`

- [ ] Add `GridAlignment` dataclass for shift vectors
- [ ] Add `calculate_grid_shifts()` method that works with ResampledDatasetInfo
- [ ] Add `create_aligned_coordinate_query()` for SQL with shifts
- [ ] Add unit tests for grid shift calculations
- [ ] Ensure no file I/O - only metadata operations

### 1.2 Update Database Schema for Unified Storage
**File**: `src/database/schema.py`

- [ ] Add migration to update legacy tables to include coordinates:
  ```sql
  ALTER TABLE passthrough_* ADD COLUMN x_coord DOUBLE PRECISION;
  ALTER TABLE passthrough_* ADD COLUMN y_coord DOUBLE PRECISION;
  ALTER TABLE resampled_* ADD COLUMN x_coord DOUBLE PRECISION;
  ALTER TABLE resampled_* ADD COLUMN y_coord DOUBLE PRECISION;
  ```
- [ ] Add method `migrate_legacy_table_to_coordinates()` 
- [ ] Add method `ensure_table_has_coordinates()` for compatibility checks
- [ ] Update indexes for coordinate columns

### 1.3 Fix CoordinateMerger Storage Format Issues
**File**: `src/processors/data_preparation/coordinate_merger.py`

- [ ] Fix `_load_resampled_coordinates()` to handle both formats:
  ```python
  def _load_resampled_coordinates(self, name, table_name):
      # Check if table has coordinate columns
      if self._table_has_coordinates(table_name):
          # Use direct coordinate query
      else:
          # Use index-to-coordinate conversion (legacy)
  ```
- [ ] Add `_table_has_coordinates()` helper method
- [ ] Update `_load_passthrough_coordinates()` to check for coordinates first
- [ ] Add chunked merge support:
  - [ ] Add `_get_chunk_bounds()` method
  - [ ] Add `_merge_chunk_with_alignment()` method
  - [ ] Modify main merge to process chunks

### 1.4 Integrate Monitoring Hooks
**Files**: Various processors

- [ ] Add structured logging context to RasterAligner
- [ ] Add progress tracking to CoordinateMerger chunks
- [ ] Ensure all errors include traceback in extra context

## Phase 2: High-Level Pipeline Updates (Days 4-5)

### 2.1 Update MergeStage for New Architecture
**File**: `src/pipelines/stages/merge_stage.py`

- [ ] Update to handle windowed storage table names:
  ```python
  # Detect storage type from metadata
  if info.metadata.get('memory_aware', False):
      table_name = info.metadata.get('storage_table')
  else:
      # Legacy naming
  ```
- [ ] Add alignment checking before merge:
  ```python
  # Check alignment using RasterAligner
  aligner = RasterAligner()
  dataset_infos = [info for info in resampled_datasets]
  alignment_report = aligner.analyze_mixed_dataset_alignment(dataset_infos)
  ```
- [ ] Pass alignment info to CoordinateMerger
- [ ] Add progress tracking with new monitoring system
- [ ] Handle validation results properly

### 2.2 Enhance Pipeline Orchestrator Integration
**File**: `src/pipelines/orchestrator_enhanced.py`

- [ ] Ensure MergeStage has access to monitoring context
- [ ] Add structured logging for merge operations:
  ```python
  with enhanced_context.logging_context.operation("merge_datasets"):
      # Merge operations
  ```
- [ ] Track merge metrics (rows merged, alignment shifts applied)
- [ ] Ensure proper error capture with tracebacks

## Phase 3: Storage Format Migration (Days 6-7)

### 3.1 Create Migration Script
**File**: `scripts/migrate_to_coordinate_storage.py`

- [ ] Script to add coordinate columns to existing tables
- [ ] Calculate coordinates from indices for legacy data
- [ ] Verify data integrity after migration
- [ ] Add rollback capability

### 3.2 Update Windowed Storage for Consistency
**File**: `src/processors/data_preparation/windowed_storage.py`

- [ ] Ensure passthrough storage includes coordinates
- [ ] Verify resampled storage includes coordinates
- [ ] Add validation for coordinate accuracy

## Phase 4: Testing and Validation (Days 8-10)

### 4.1 Unit Tests
- [ ] Test RasterAligner grid shift calculations
- [ ] Test CoordinateMerger with mixed storage formats
- [ ] Test alignment SQL query generation
- [ ] Test chunked merge operations

### 4.2 Integration Tests
- [ ] Test full pipeline with mixed passthrough/resampled data
- [ ] Test alignment correction during merge
- [ ] Test memory usage stays within bounds
- [ ] Test monitoring integration

### 4.3 Performance Tests
- [ ] Benchmark chunked merge vs full merge
- [ ] Measure memory usage with large datasets
- [ ] Verify alignment doesn't slow merge significantly

## Phase 5: Documentation and Cleanup (Days 11-12)

### 5.1 Update Documentation
- [ ] Update CLAUDE.md with new merge architecture
- [ ] Document coordinate storage format
- [ ] Add examples of alignment correction

### 5.2 Remove Deprecated Code
- [ ] Mark old merge methods as deprecated
- [ ] Remove GeoTIFF-based alignment code paths
- [ ] Clean up unused imports

## Implementation Checklist

### Week 1: Infrastructure
- [ ] Day 1-2: RasterAligner database support
- [ ] Day 3: Database schema updates
- [ ] Day 4: CoordinateMerger fixes
- [ ] Day 5: Monitoring integration

### Week 2: Pipeline & Testing  
- [ ] Day 6: MergeStage updates
- [ ] Day 7: Orchestrator integration
- [ ] Day 8: Storage migration
- [ ] Day 9-10: Testing suite
- [ ] Day 11-12: Documentation

## Rollback Plan
1. Keep legacy code paths with feature flags
2. Add `use_legacy_merge` configuration option
3. Maintain backward compatibility for 2 releases
4. Log deprecation warnings for old methods

## Success Metrics
1. **Correctness**: All 4 datasets merge without gaps or duplicates
2. **Performance**: Memory usage < 2GB per dataset during merge
3. **Alignment**: Sub-pixel alignment accuracy (<0.1 pixel offset)
4. **Monitoring**: All merge operations logged with context
5. **Architecture**: No direct database queries in high-level code

## Code Examples

### Example 1: RasterAligner Extension
```python
# In raster_alignment.py
def calculate_grid_shifts(self, dataset_infos: List[ResampledDatasetInfo]) -> List[GridAlignment]:
    """Calculate shifts without loading data."""
    alignments = []
    reference = dataset_infos[0]
    
    for dataset in dataset_infos[1:]:
        # Calculate grid origins
        ref_origin_x = reference.bounds[0]
        ds_origin_x = dataset.bounds[0]
        
        # Check pixel alignment
        pixel_offset = (ds_origin_x - ref_origin_x) / reference.target_resolution
        if abs(pixel_offset % 1.0) > 0.01:
            # Need shift
            shift = -(pixel_offset % 1.0) * reference.target_resolution
            alignments.append(GridAlignment(
                reference_dataset=reference.name,
                aligned_dataset=dataset.name,
                x_shift=shift,
                requires_shift=True
            ))
    return alignments
```

### Example 2: CoordinateMerger Chunked Processing
```python
# In coordinate_merger.py
def create_ml_ready_parquet(self, resampled_datasets, output_dir):
    # Get alignment info
    aligner = RasterAligner()
    alignments = aligner.calculate_grid_shifts(dataset_infos)
    
    # Process in chunks
    output_dfs = []
    for chunk_bounds in self._get_chunk_bounds():
        chunk_df = self._process_chunk_with_alignment(
            resampled_datasets, 
            alignments, 
            chunk_bounds
        )
        output_dfs.append(chunk_df)
    
    # Combine and save
    final_df = pd.concat(output_dfs)
    final_df.to_parquet(output_path)
```

### Example 3: MergeStage with Monitoring
```python
# In merge_stage.py
def execute(self, context) -> StageResult:
    with context.monitor.track_stage("merge") as progress:
        # Setup
        progress.update(10, "Checking alignment")
        
        # Check alignment
        alignment_report = self._check_alignment(resampled_datasets)
        
        # Merge with monitoring
        progress.update(30, "Starting coordinate merge")
        merger = CoordinateMerger(context.config, context.db)
        
        # Pass alignment info
        merger.set_alignment_info(alignment_report)
        
        # Execute merge
        ml_ready_path = merger.create_ml_ready_parquet(
            dataset_dicts,
            context.output_dir,
            progress_callback=lambda pct: progress.update(30 + pct * 0.6)
        )
        
        progress.update(100, "Merge complete")
```

## Notes
- This plan maintains all architectural boundaries
- No new abstractions are created - only extensions
- All database operations go through schema methods
- Monitoring is integrated at appropriate levels
- The solution is incremental and can be rolled back