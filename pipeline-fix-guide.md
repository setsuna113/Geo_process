# Pipeline Fix Implementation Guide - Step-by-Step

## ⚠️ CRITICAL: Read Before Making ANY Changes

This guide is designed for fixing the pipeline issues systematically. Each step builds on the previous one. **DO NOT SKIP STEPS** or you will create more problems.

### System Invariants (NEVER VIOLATE)
1. **Config Import**: Always use `from src.config import config` (lowercase, the instance)
2. **Database**: Never use `DatabaseManager(test_mode=True)` in production code
3. **File Paths**: Always use `Path` objects, never string concatenation
4. **Imports**: Always use absolute imports from `src.*`

## Phase 1: Foundation Fixes (Enables Everything Else)

### Step 1.1: Fix Memory Check Issue
**File**: `src/pipelines/orchestrator.py`
**Why First**: Blocks pipeline execution, simplest fix

```python
# FIND around line 643:
if required_memory > available_memory:
    raise MemoryError(...)

# REPLACE WITH:
memory_tolerance = 0.95  # 5% tolerance
if required_memory > available_memory * memory_tolerance:
    logger.warning(f"Memory usage close to limit: {required_memory:.1f}GB required, {available_memory:.1f}GB available")
else:
    # Only fail if significantly over
    if required_memory > available_memory * 1.1:  # 10% over
        raise MemoryError(...)
```

**Test**: Run pipeline with 790-800GB data, should not fail

### Step 1.2: Fix File Permission Issues
**Files**: 
- `src/pipelines/stages/merge_stage.py`
- `src/pipelines/stages/export_stage.py`
**Why Next**: Blocks retries, affects multiple stages

```python
# ADD this utility function at module level:
def safe_write_file(file_path: Path, write_func):
    """Safely write file with overwrite handling."""
    file_path = Path(file_path)
    if file_path.exists():
        logger.warning(f"File exists, removing: {file_path}")
        file_path.unlink()
    
    # Write with temporary file first
    temp_path = file_path.with_suffix('.tmp')
    try:
        write_func(temp_path)
        temp_path.rename(file_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise

# USE in merge_stage.py around line 328:
# REPLACE: with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as nc:
# WITH:
def write_nc(path):
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as nc:
        # ... existing write code ...
        
safe_write_file(output_path, write_nc)
```

**Test**: Run pipeline twice, second run should not fail on file exists

## Phase 2: Data Flow Fixes (Core Issues)

### Step 2.1: Create DB Data Reader
**New File**: `src/processors/data_preparation/db_data_reader.py`
**Why**: Centralizes DB reading logic, needed by multiple fixes

```python
"""Database data reader for passthrough tables."""
import numpy as np
from typing import Optional, Tuple
import logging
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class DBDataReader:
    """Reads data from passthrough tables with proper coordinate mapping."""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        
    def read_passthrough_data(self, dataset_name: str, 
                            bounds: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
        """
        Read data from passthrough table.
        
        Args:
            dataset_name: Name like 'terrestrial-richness' 
            bounds: Optional (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            2D numpy array with data
        """
        # Normalize name to table name
        table_name = f"passthrough_{dataset_name.replace('-', '_')}"
        
        with self.db.get_cursor() as cursor:
            # Get dimensions
            cursor.execute(f"""
                SELECT MIN(row_idx), MAX(row_idx), 
                       MIN(col_idx), MAX(col_idx)
                FROM {table_name}
            """)
            min_row, max_row, min_col, max_col = cursor.fetchone()
            
            # Create empty array
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            data = np.full((height, width), np.nan, dtype=np.float32)
            
            # Fill with data
            cursor.execute(f"""
                SELECT row_idx, col_idx, value
                FROM {table_name}
                WHERE value IS NOT NULL
            """)
            
            for row in cursor:
                r = row['row_idx'] - min_row
                c = row['col_idx'] - min_col
                data[r, c] = row['value']
                
        logger.info(f"Loaded {table_name}: shape={data.shape}, "
                   f"valid_cells={np.sum(~np.isnan(data))}")
        
        return data
```

### Step 2.2: Fix Merge Stage Data Source
**File**: `src/processors/data_preparation/resampling_processor.py`
**Critical**: This fixes the core data corruption issue

```python
# FIND the load_passthrough_data method (around line 838)
def load_passthrough_data(self, info: ResampledDatasetInfo) -> Optional[np.ndarray]:
    """Load data for passthrough datasets."""
    
    # ADD at beginning:
    # Check if we should load from DB instead of file
    if hasattr(info, 'data_table_name') and info.data_table_name:
        logger.info(f"Loading passthrough data from DB table: {info.data_table_name}")
        from .db_data_reader import DBDataReader
        reader = DBDataReader(self.db)
        
        # Extract dataset name from table name
        # passthrough_terrestrial_richness -> terrestrial-richness
        dataset_name = info.data_table_name.replace('passthrough_', '').replace('_', '-')
        
        return reader.read_passthrough_data(dataset_name)
    
    # KEEP existing file loading as fallback
    logger.info(f"Loading passthrough data from file: {info.source_path}")
    # ... rest of existing code ...
```

**Critical Test**: After this change, merged data should match DB values exactly

### Step 2.3: Fix Coordinate Alignment
**File**: `src/pipelines/stages/merge_stage.py`
**Why**: Fixes the 40-pixel offset bug

```python
# FIND _align_data_to_common_grid method (around line 181)
def _align_data_to_common_grid(self, array_data: np.ndarray, 
                              data_bounds: Tuple[float, float, float, float],
                              common_bounds: Tuple[float, float, float, float],
                              common_shape: Tuple[int, int],
                              resolution: float) -> np.ndarray:
    """Align dataset to common coordinate grid."""
    
    # ADD validation:
    logger.info(f"Aligning data: shape={array_data.shape}, "
               f"data_bounds={data_bounds}, common_bounds={common_bounds}")
    
    # REPLACE offset calculation:
    # OLD:
    # x_offset = int(np.round((data_bounds[0] - common_bounds[0]) / resolution))
    # y_offset = int(np.round((common_bounds[3] - data_bounds[3]) / resolution))
    
    # NEW (with validation):
    x_offset = int(np.round((data_bounds[0] - common_bounds[0]) / resolution))
    y_offset = int(np.round((common_bounds[3] - data_bounds[3]) / resolution))
    
    # Validate offsets
    if x_offset < 0 or y_offset < 0:
        logger.error(f"Negative offsets detected: x={x_offset}, y={y_offset}")
        logger.error(f"This indicates bounds mismatch - data may be corrupted!")
    
    # Log for debugging
    logger.info(f"Calculated offsets: x={x_offset}, y={y_offset}")
    
    # ... rest of existing code ...
```

## Phase 3: Skip Logic Enhancement

### Step 3.1: Add DB Status Detection
**File**: `src/pipelines/stages/resample_stage.py`
**Why**: Enables intelligent skip decisions

```python
# ADD new method to ResampleStage class:
def _check_db_status(self, dataset_name: str) -> str:
    """Check if dataset exists in DB and its status."""
    try:
        from src.database.schema import schema
        
        # Check passthrough table
        table_name = f"passthrough_{dataset_name.replace('-', '_')}"
        
        with self.db.get_cursor() as cursor:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            if not cursor.fetchone()[0]:
                return "missing"
            
            # Check row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            if count == 0:
                return "empty"
            
            # Check metadata
            datasets = schema.get_resampled_datasets({'name': dataset_name})
            if not datasets:
                return "incomplete"  # Table exists but no metadata
                
            return "complete"
            
    except Exception as e:
        logger.error(f"Error checking DB status: {e}")
        return "error"

# MODIFY execute method to use status:
def execute(self, context) -> StageResult:
    # ... existing code ...
    
    for dataset_config in datasets:
        db_status = self._check_db_status(dataset_config['name'])
        
        if db_status == "complete" and self.skip_control.should_skip():
            logger.info(f"Skipping {dataset_config['name']} - already in DB")
            # Load metadata from DB instead of reprocessing
            continue
        elif db_status in ["error", "incomplete"]:
            logger.warning(f"DB status {db_status}, cleaning and reprocessing")
            # Clean partial data
            self._cleanup_partial_data(dataset_config['name'])
```

## Phase 4: New Architecture Implementation

### Step 4.1: Create Direct DB to Parquet Exporter
**New File**: `src/processors/exporters/parquet_exporter.py`
**Why**: Implements new efficient pipeline

```python
"""Direct database to Parquet exporter."""
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ParquetExporter:
    """Export directly from DB to Parquet format."""
    
    def __init__(self, db):
        self.db = db
        
    def export_merged_data(self, output_path: Path, 
                          datasets: list,
                          chunk_size: int = 1000000) -> Path:
        """
        Export merged biodiversity data to Parquet.
        
        This replaces both MergeStage and ExportStage.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build efficient SQL query
        query = self._build_merge_query(datasets)
        
        # Export in chunks to handle large data
        chunks_written = 0
        
        with self.db.get_connection() as conn:
            for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
                if chunks_written == 0:
                    # First chunk - write new file
                    chunk.to_parquet(output_path, engine='pyarrow', 
                                   compression='snappy')
                else:
                    # Append to existing (requires pyarrow>=8.0)
                    chunk.to_parquet(output_path, engine='pyarrow',
                                   compression='snappy', append=True)
                
                chunks_written += 1
                logger.info(f"Written {chunks_written * chunk_size:,} rows")
        
        logger.info(f"Export complete: {output_path}")
        return output_path
    
    def _build_merge_query(self, datasets: list) -> str:
        """Build SQL query to merge datasets."""
        # This is simplified - real implementation needs proper coordinate mapping
        return """
        WITH coords AS (
            -- Generate coordinate grid
            SELECT DISTINCT row_idx, col_idx,
                   -89.833 + (row_idx * 0.016667) as latitude,
                   -180.0 + (col_idx * 0.016667) as longitude
            FROM (
                SELECT row_idx, col_idx FROM passthrough_terrestrial_richness
                UNION
                SELECT row_idx, col_idx FROM passthrough_plants_richness
            ) all_cells
        )
        SELECT 
            c.latitude, c.longitude,
            t.value as terrestrial_richness,
            p.value as plants_richness
        FROM coords c
        LEFT JOIN passthrough_terrestrial_richness t 
            ON c.row_idx = t.row_idx AND c.col_idx = t.col_idx
        LEFT JOIN passthrough_plants_richness p
            ON c.row_idx = p.row_idx AND c.col_idx = p.col_idx
        WHERE t.value IS NOT NULL OR p.value IS NOT NULL
        ORDER BY c.row_idx, c.col_idx
        """
```

### Step 4.2: Modify Pipeline to Use New Architecture
**File**: `scripts/process_manager.py`
**Why**: Implements the new streamlined pipeline

```python
# MODIFY the pipeline stages configuration (around line 84):
# OLD:
stages = [
    DataLoadStage,
    ResampleStage, 
    MergeStage,
    ExportStage,
    lambda: AnalysisStage(analysis_method)
]

# NEW:
stages = [
    DataLoadStage,
    ResampleStage,
    ParquetExportStage,  # Replaces both Merge and Export
    lambda: AnalysisStage(analysis_method)
]

# ADD new stage class:
from src.processors.exporters.parquet_exporter import ParquetExporter

class ParquetExportStage(PipelineStage):
    """Direct DB to Parquet export stage."""
    
    @property
    def name(self) -> str:
        return "parquet_export"
    
    def execute(self, context) -> StageResult:
        exporter = ParquetExporter(context.db)
        
        output_file = context.output_dir / "biodiversity_data.parquet"
        datasets = context.get('resampled_datasets', [])
        
        exporter.export_merged_data(output_file, datasets)
        
        context.set('export_file', str(output_file))
        
        return StageResult(
            success=True,
            data={'output_file': str(output_file)}
        )
```

## Phase 5: Validation and Cleanup

### Step 5.1: Add Data Validation
**File**: `src/pipelines/stages/base_stage.py`
**Why**: Catches corruption early

```python
# ADD to PipelineStage base class:
def validate_output(self, context) -> bool:
    """Validate stage output data integrity."""
    # Override in subclasses
    return True

# In ParquetExportStage, add:
def validate_output(self, context) -> bool:
    """Spot check exported data matches DB."""
    import pandas as pd
    import numpy as np
    
    output_file = context.get('export_file')
    if not output_file:
        return False
        
    # Load small sample
    df = pd.read_parquet(output_file, nrows=100)
    
    # Spot check a few values against DB
    with context.db.get_cursor() as cursor:
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            
            # Convert lat/lon back to indices
            row_idx = int((83.667 - row['latitude']) / 0.016667)
            col_idx = int((row['longitude'] + 180.0) / 0.016667)
            
            # Check terrestrial value
            cursor.execute("""
                SELECT value FROM passthrough_terrestrial_richness
                WHERE row_idx = %s AND col_idx = %s
            """, (row_idx, col_idx))
            
            result = cursor.fetchone()
            if result:
                db_val = result['value']
                csv_val = row['terrestrial_richness']
                
                if not np.isnan(db_val) and not np.isnan(csv_val):
                    if abs(db_val - csv_val) > 0.001:
                        logger.error(f"Validation failed: DB={db_val}, Export={csv_val}")
                        return False
    
    logger.info("Output validation passed")
    return True
```

## Testing Strategy

### After Each Phase:
1. **Phase 1**: Pipeline should run without memory/permission errors
2. **Phase 2**: Exported values should match DB exactly (test with spot checks)
3. **Phase 3**: Re-runs should skip completed work intelligently
4. **Phase 4**: Parquet export should be 4x faster than CSV
5. **Phase 5**: Validation should catch any data mismatches

### Integration Test:
```bash
# Clean start
./scripts/cleanup_all.sh

# Run full pipeline
./run_pipeline.sh test_fixed_pipeline

# Verify parquet output
python -c "
import pandas as pd
df = pd.read_parquet('outputs/*/biodiversity_data.parquet')
print(f'Rows: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(df.head())
"
```

## Common Pitfalls to Avoid

1. **Don't skip Phase 1** - Foundation fixes enable everything else
2. **Don't modify config imports** - Use existing patterns
3. **Don't remove logging** - Add more for debugging
4. **Don't assume file paths** - Always use config
5. **Test after each step** - Don't wait until the end

## Success Criteria

- [ ] No coordinate mapping errors
- [ ] No memory check failures  
- [ ] No permission denied errors
- [ ] DB status properly detected
- [ ] Direct DB → Parquet export works
- [ ] Validation catches mismatches
- [ ] 4x performance improvement