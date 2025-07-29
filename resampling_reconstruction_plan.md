# Resampling Pipeline Reconstruction Plan

## Executive Summary

This plan addresses the critical memory bottleneck in the current resampling pipeline where entire datasets are loaded into memory, even for passthrough cases. The solution implements a unified, memory-aware processing strategy that handles both passthrough and resampling scenarios without loading full datasets into memory.

## Current Architecture Issues

### 1. Memory Bottlenecks
- **ResamplingProcessor** (line 308): Loads entire dataset for passthrough: `data = src.read(1).astype(np.float32)`
- **Chunked resampling** still creates full output array: `output_data = np.zeros(output_shape)`
- **Storage function** expects full array: `_store_resampled_dataset(resampled_info, result_data)`

### 2. Processing Inefficiencies
- Sequential processing of 4 large datasets
- No early exit for passthrough datasets
- Redundant data loading for skip-resampling cases

### 3. Design Issues
- Inconsistent handling between passthrough and resampling
- Complex decision logic buried deep in implementation
- No streaming/windowed processing support

## Proposed Solution: Unified Memory-Aware Pipeline

### Core Principle
**Never load full dataset into memory** - use windowed/streaming processing for both passthrough and resampling operations.

### Unified Logic Flow
```
IF resolution matches target:
    Stream copy metadata + register source path for windowed access
ELSE:
    Stream resample using chunked processing + store chunks progressively
```

## Implementation Plan

### Phase 1: Add Windowed Storage Infrastructure

#### 1.1 Create Windowed Storage Manager
**New file**: `src/processors/data_preparation/windowed_storage.py`
```python
class WindowedStorageManager:
    """Handles chunked storage operations for large rasters."""
    
    def store_passthrough_windowed(self, raster_path, table_name, db):
        """Stream copy raster data using windows without loading all."""
        
    def store_resampled_windowed(self, table_name, db):
        """Accept resampled chunks and store progressively."""
```

**Status**: NEW MODULE - Creates foundation for memory-aware storage

#### 1.2 Update Database Schema Utils
**Update**: `src/database/schema.py` - Add windowed insert methods
```python
def insert_raster_chunk(self, table_name, chunk_data, row_offset, col_offset):
    """Insert a chunk of raster data with proper indexing."""
```

**Status**: ADD METHOD - Extends existing schema class

### Phase 2: Refactor ResamplingProcessor for Streaming

#### 2.1 Extract Passthrough Logic
**Update**: `src/processors/data_preparation/resampling_processor.py`
- **Line 267-342**: Replace full data loading with metadata-only registration
```python
def _handle_passthrough_dataset(self, raster_entry, dataset_config):
    """Handle passthrough without loading data."""
    # Create metadata entry
    passthrough_info = self._create_passthrough_dataset_info(raster_entry, dataset_config)
    
    # Register in database WITHOUT loading data
    self._register_passthrough_dataset(passthrough_info)
    
    # Let downstream processes use windowed reads
    return passthrough_info
```

**Status**: REPLACE METHOD - Eliminates memory bottleneck

#### 2.2 Implement Streaming Resampler Interface
**Update**: `src/domain/resampling/engines/base_resampler.py`
- Add streaming interface:
```python
@abstractmethod
def resample_windowed(self, source_path, window, progress_callback=None):
    """Resample a window of data without loading full raster."""
```

**Status**: ADD METHOD - Extends base class interface

#### 2.3 Update NumpyResampler for Windowed Processing
**Update**: `src/domain/resampling/engines/numpy_resampler.py`
- **Line 165-241**: Refactor `_resample_chunked` to yield chunks instead of accumulating
```python
def resample_windowed(self, source_path, window, progress_callback=None):
    """Resample and yield chunks progressively."""
    # Open source with rasterio
    with rasterio.open(source_path) as src:
        # Read window
        data = src.read(1, window=window)
        # Resample this chunk
        result = self._resample_chunk(data, ...)
        yield result, window_bounds
```

**Status**: MODIFY METHOD - Enables streaming processing

### Phase 3: Update Pipeline Stages

#### 3.1 Enhance Load Stage with Resolution Check
**Update**: `src/pipelines/stages/load_stage.py`
- **Line 73-90**: Add resolution matching info
```python
# Add resolution check during loading
raster_entry = catalog.get_raster(dataset_name)
needs_resampling = not self._check_resolution_match(
    raster_entry.resolution_degrees,
    context.config.get('resampling.target_resolution')
)

dataset_info = {
    'config': normalized_config,
    'needs_resampling': needs_resampling,
    'source_resolution': raster_entry.resolution_degrees
}
```

**Status**: ENHANCE - Adds early resolution detection

#### 3.2 Optimize Resample Stage
**Update**: `src/pipelines/stages/resample_stage.py`
- **Line 80-122**: Use windowed processing
```python
# Fast path for passthrough
if not dataset_info.get('needs_resampling'):
    result = processor.handle_passthrough(dataset_config)
else:
    # Use windowed resampling
    result = processor.resample_dataset_windowed(dataset_config)
```

**Status**: MODIFY - Implements fast path

### Phase 4: Update Merge Stage for Windowed Access

#### 4.1 Enhance CoordinateMerger
**Update**: `src/processors/data_preparation/coordinate_merger.py`
- Already supports SQL-based coordinate extraction
- Ensure it never loads full rasters

**Status**: VERIFY - Ensure memory-safe operation

### Phase 5: Configuration Updates

#### 5.1 Add Windowed Processing Config
**Update**: `config.yml`
```yaml
resampling:
  # Existing settings...
  
  # New windowed processing settings
  window_size: 2048  # Process in 2048x2048 windows
  enable_streaming: true
  skip_data_loading_for_passthrough: true
```

**Status**: ADD SECTION - Enables configuration control

### Phase 6: Deprecation and Cleanup

#### 6.1 Mark Old Methods as Deprecated
**Update**: Throughout codebase
```python
@deprecated("Use resample_dataset_windowed instead")
def resample_dataset(self, ...):
    # Existing implementation
```

**Status**: ANNOTATE - Guides migration

#### 6.2 Remove Full Array Storage
**Update**: `src/processors/data_preparation/resampling_processor.py`
- **Line 770-871**: Replace `_store_resampled_dataset` with windowed version
- Keep interface but change implementation

**Status**: REFACTOR - Internal implementation change

## Implementation Order

1. **Week 1**: Implement windowed storage infrastructure (Phase 1)
2. **Week 2**: Update resampling processor for streaming (Phase 2.1-2.2)
3. **Week 3**: Update resampling engines (Phase 2.3)
4. **Week 4**: Update pipeline stages (Phase 3)
5. **Week 5**: Testing and optimization
6. **Week 6**: Deprecation and cleanup

## Testing Strategy

### Unit Tests
- Test windowed storage manager
- Test streaming resampler
- Test passthrough without data loading

### Integration Tests
- Test full pipeline with memory monitoring
- Verify output consistency with current implementation
- Test mixed passthrough/resample scenarios

### Performance Tests
- Measure memory usage: Should never exceed window_size memory
- Measure processing time: Should be similar or better
- Test with actual large datasets

## Rollback Plan

1. Keep old methods with @deprecated annotation
2. Add feature flag: `enable_windowed_processing`
3. Gradual migration with A/B testing
4. Full rollback possible by flipping feature flag

## Success Metrics

1. **Memory Usage**: Peak memory < 2GB per dataset (vs current full dataset size)
2. **Processing Time**: ≤ current processing time
3. **Output Quality**: Identical results to current implementation
4. **Code Maintainability**: Cleaner separation of concerns

## Risk Mitigation

1. **Risk**: Window boundary artifacts
   - **Mitigation**: Use overlapping windows with proper edge handling

2. **Risk**: Performance degradation from many small DB inserts
   - **Mitigation**: Batch inserts within windows

3. **Risk**: Compatibility with downstream processes
   - **Mitigation**: Maintain same output table structure

## Conclusion

This plan provides a path to eliminate memory bottlenecks while maintaining compatibility with the existing system. The unified approach simplifies the codebase and makes the distinction between passthrough and resampling transparent to downstream processes.

---

## Appendix: Concrete Implementation Steps

### Step 1: Create Windowed Storage Manager (Day 1-2)

Create `src/processors/data_preparation/windowed_storage.py`:

```python
import rasterio
from rasterio.windows import Window
import numpy as np
from typing import Iterator, Tuple
import logging

logger = logging.getLogger(__name__)

class WindowedStorageManager:
    """Handles chunked storage operations for large rasters."""
    
    def __init__(self, window_size: int = 2048):
        self.window_size = window_size
    
    def iter_windows(self, raster_path: str) -> Iterator[Tuple[Window, Tuple[int, int]]]:
        """Iterate over windows for a raster file."""
        with rasterio.open(raster_path) as src:
            height, width = src.shape
            
            for row_off in range(0, height, self.window_size):
                row_size = min(self.window_size, height - row_off)
                
                for col_off in range(0, width, self.window_size):
                    col_size = min(self.window_size, width - col_off)
                    
                    window = Window(col_off, row_off, col_size, row_size)
                    yield window, (row_off, col_off)
    
    def store_passthrough_windowed(self, raster_path: str, table_name: str, 
                                  db_connection, bounds: Tuple[float, float, float, float]):
        """Stream copy raster data using windows without loading all."""
        logger.info(f"Starting windowed passthrough storage for {table_name}")
        
        with rasterio.open(raster_path) as src:
            total_windows = ((src.height + self.window_size - 1) // self.window_size) * \
                          ((src.width + self.window_size - 1) // self.window_size)
            processed = 0
            
            for window, (row_off, col_off) in self.iter_windows(raster_path):
                # Read window data
                data = src.read(1, window=window)
                
                # Store this chunk
                self._store_chunk(db_connection, table_name, data, row_off, col_off)
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_windows} windows")
    
    def _store_chunk(self, db_connection, table_name: str, 
                    data: np.ndarray, row_offset: int, col_offset: int):
        """Store a single chunk to database."""
        # Extract non-NaN values
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return
        
        rows, cols = np.where(valid_mask)
        values = data[valid_mask]
        
        # Adjust indices by offset
        global_rows = rows + row_offset
        global_cols = cols + col_offset
        
        # Prepare batch insert
        data_to_insert = [
            (int(r), int(c), float(v))
            for r, c, v in zip(global_rows, global_cols, values)
        ]
        
        # Insert using psycopg2
        from psycopg2.extras import execute_values
        
        with db_connection.get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"INSERT INTO {table_name} (row_idx, col_idx, value) VALUES %s "
                    f"ON CONFLICT (row_idx, col_idx) DO UPDATE SET value = EXCLUDED.value",
                    data_to_insert,
                    page_size=1000
                )
            conn.commit()
```

### Step 2: Update ResamplingProcessor (Day 3-4)

Add new methods to `src/processors/data_preparation/resampling_processor.py`:

```python
def resample_dataset_memory_aware(self, dataset_config: dict) -> ResampledDatasetInfo:
    """Memory-aware resampling that never loads full dataset."""
    # ... existing setup code ...
    
    # Check if skip-resampling
    if self._check_resolution_match(raster_entry):
        return self._handle_passthrough_memory_aware(raster_entry, dataset_config)
    else:
        return self._handle_resampling_memory_aware(raster_entry, dataset_config)

def _handle_passthrough_memory_aware(self, raster_entry, dataset_config):
    """Handle passthrough without loading data."""
    logger.info(f"Processing passthrough dataset: {dataset_config['name']}")
    
    # Create metadata
    passthrough_info = self._create_passthrough_dataset_info(raster_entry, dataset_config)
    
    # Register metadata in database
    self._register_dataset_metadata(passthrough_info)
    
    # Stream copy data using windowed storage
    table_name = f"passthrough_{passthrough_info.name.replace('-', '_')}"
    
    # Create table
    self._create_data_table(table_name)
    
    # Use windowed storage manager
    storage_manager = WindowedStorageManager(
        window_size=self.config.get('resampling.window_size', 2048)
    )
    storage_manager.store_passthrough_windowed(
        str(raster_entry.path),
        table_name,
        self.db,
        raster_entry.bounds
    )
    
    return passthrough_info

def _handle_resampling_memory_aware(self, raster_entry, dataset_config):
    """Handle resampling without loading full dataset."""
    logger.info(f"Processing resampling dataset: {dataset_config['name']}")
    
    # Create output table first
    table_name = f"resampled_{dataset_config['name'].replace('-', '_')}"
    self._create_data_table(table_name)
    
    # Process in windows
    with rasterio.open(raster_entry.path) as src:
        # Calculate output shape
        output_shape = self._calculate_output_shape(raster_entry.bounds)
        
        # Process each window
        for window, (row_off, col_off) in self._iter_resampling_windows(src):
            # Read source window
            source_data = src.read(1, window=window)
            
            # Calculate bounds for this window
            window_bounds = self._get_window_bounds(src, window)
            
            # Resample this window
            resampled_chunk = self._resample_window(
                source_data, window_bounds, dataset_config['method']
            )
            
            # Store immediately
            self._store_resampled_chunk(
                table_name, resampled_chunk, 
                row_off, col_off, output_shape
            )
    
    # Create and return info
    return self._create_resampled_info(raster_entry, dataset_config, table_name)
```

### Step 3: Update Configuration (Day 5)

Add to `config.yml`:

```yaml
resampling:
  # ... existing config ...
  
  # Memory-aware processing
  enable_memory_aware_processing: true
  window_size: 2048  # Process in 2048x2048 windows
  window_overlap: 128  # Overlap for avoiding edge artifacts
  
  # Feature flags for gradual rollout
  use_legacy_passthrough: false
  use_legacy_resampling: false
```

### Step 4: Update Pipeline Stages (Day 6-7)

Update `src/pipelines/stages/resample_stage.py`:

```python
def execute(self, context) -> StageResult:
    """Execute resampling with memory-aware processing."""
    # ... existing setup ...
    
    # Check if memory-aware processing is enabled
    use_memory_aware = context.config.get('resampling.enable_memory_aware_processing', False)
    
    for idx, dataset_info in enumerate(loaded_datasets):
        try:
            dataset_config = dataset_info['config']
            
            # Use appropriate method based on config
            if use_memory_aware:
                resampled_info = processor.resample_dataset_memory_aware(dataset_config)
            else:
                # Legacy path
                resampled_info = processor.resample_dataset(dataset_config)
            
            resampled_datasets.append(resampled_info)
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_config.get('name')}: {e}")
            continue
```

### Step 5: Testing and Validation (Day 8-10)

Create test script `test_memory_aware_resampling.py`:

```python
import psutil
import time
from src.processors.data_preparation.resampling_processor import ResamplingProcessor

def test_memory_usage():
    """Test that memory usage stays within bounds."""
    # Monitor memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process a large dataset
    processor = ResamplingProcessor(config, db)
    result = processor.resample_dataset_memory_aware({
        'name': 'test-large-dataset',
        'path': '/maps/mwd24/richness/daru-plants-richness.tif'
    })
    
    # Check peak memory
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    
    print(f"Memory used: {mem_used:.1f} MB")
    assert mem_used < 2048, f"Memory usage {mem_used} exceeds limit"
    
    # Verify output
    # ... validation code ...
```

This concrete implementation provides a clear path forward with specific code changes that maintain backward compatibility while solving the memory issues.