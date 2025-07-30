# Memory Optimization Implementation Checklist

## Golden Rules

1. **Respect Current System Hierarchy**
   - `base/` → Pure abstractions, no business logic
   - `infrastructure/` → Technical services (monitoring, logging)
   - `processors/` → Business logic implementation
   - `pipelines/` → Orchestration only, no implementation

2. **Respect Abstractions**
   - Don't modify interfaces in `abstractions/`
   - Extend through inheritance, not modification
   - Use dependency injection patterns already established

3. **Search Before Adding**
   - ALWAYS search for existing similar functionality first
   - Update/extend existing code rather than duplicate
   - If adding new functionality, carefully consider module placement
   - New additions should be rare - the system is comprehensive

## Phase 1: Memory Pressure Callbacks (EASIEST)

### 1.1 Analyze Existing Infrastructure
**Existing Components Found**:
- ✅ `src/pipelines/monitors/memory_monitor.py` - Has MemoryMonitor with callbacks
- ✅ `warning_callbacks` and `critical_callbacks` lists already exist
- ✅ `register_warning_callback()` and `register_critical_callback()` methods ready
- ✅ Thresholds already defined: warning=80%, critical=90%

### 1.2 Connect MemoryMonitor to Pipeline Context
**File**: `src/pipelines/orchestrator.py` or `src/pipelines/enhanced_context.py`

```python
# In PipelineContext initialization, the memory monitor is already created:
self.memory_monitor = MemoryMonitor(config)

# What's missing: Start it and make it available to processors
# Add after line where memory_monitor is created:
self.memory_monitor.start()
```

### 1.3 Add Adaptive Behavior to ResamplingProcessor
**File**: `src/processors/data_preparation/resampling_processor.py`

**Step 1**: In `__init__`, get memory monitor from context
```python
# After line 72 (self.memory_manager = get_memory_manager())
# Note: We need to pass context or get monitor differently since __init__ doesn't have context
# Alternative: Add to resample_dataset_memory_aware method instead
```

**Step 2**: Add adaptive callbacks (in resample_dataset_memory_aware method)
```python
def resample_dataset_memory_aware(self, dataset_config: dict, 
                                progress_callback: Optional[Callable[[str, float], None]] = None,
                                context: Optional[Any] = None) -> ResampledDatasetInfo:
    """Memory-aware resampling that adapts to memory pressure."""
    
    # Get memory monitor if available
    memory_monitor = None
    if context and hasattr(context, 'memory_monitor'):
        memory_monitor = context.memory_monitor
    
    # Store original window size
    original_window_size = self.config.get('resampling.window_size', 2048)
    current_window_size = original_window_size
    
    # Define pressure callbacks
    def on_memory_warning(usage):
        nonlocal current_window_size
        current_window_size = max(512, current_window_size // 2)
        logger.warning(f"Memory pressure: reducing window size to {current_window_size}")
        # Update config temporarily
        self.config.set('resampling.window_size', current_window_size)
    
    def on_memory_critical(usage):
        nonlocal current_window_size
        current_window_size = 256  # Minimum viable size
        logger.error(f"Critical memory: window size set to minimum {current_window_size}")
        self.config.set('resampling.window_size', current_window_size)
        # Force garbage collection
        import gc
        gc.collect()
    
    # Register callbacks
    if memory_monitor:
        memory_monitor.register_warning_callback(on_memory_warning)
        memory_monitor.register_critical_callback(on_memory_critical)
    
    try:
        # Existing method logic here...
        result = self._existing_resample_logic(dataset_config, progress_callback)
        return result
    finally:
        # Restore original window size
        self.config.set('resampling.window_size', original_window_size)
```

### 1.4 Add Adaptive Behavior to CoordinateMerger
**File**: `src/processors/data_preparation/coordinate_merger.py`

**Similar approach** - Add adaptive chunk sizing in `create_merged_dataset`:
```python
def create_merged_dataset(self, resampled_datasets: List[Dict],
                        chunk_size: Optional[int] = None,
                        return_as: str = 'xarray',
                        context: Optional[Any] = None) -> any:
    # Get memory monitor
    memory_monitor = context.memory_monitor if context and hasattr(context, 'memory_monitor') else None
    
    # Adaptive chunk size
    base_chunk_size = chunk_size or self.config.get('merge.chunk_size', 5000)
    current_chunk_size = base_chunk_size
    
    def on_memory_warning(usage):
        nonlocal current_chunk_size
        current_chunk_size = max(1000, current_chunk_size // 2)
        logger.warning(f"Memory pressure: reducing chunk size to {current_chunk_size}")
    
    # Register callback
    if memory_monitor:
        memory_monitor.register_warning_callback(on_memory_warning)
    
    # Continue with existing logic using current_chunk_size
```

### 1.5 Update Stage Calls to Pass Context
**Files**: `src/pipelines/stages/resample_stage.py`, `src/pipelines/stages/merge_stage.py`

```python
# In ResampleStage.execute(), the processor already has access to context
# We just need to ensure memory_monitor is available in context
# This is already done if using PipelineContext!

# No changes needed - context is already available
```

### 1.6 Test Memory Pressure Callbacks
- Create test that triggers high memory usage
- Verify window/chunk sizes reduce under pressure
- Confirm processing continues with degraded performance

## Phase 2: Enhanced Memory Tracking

### 2.1 Analyze Existing Infrastructure
**Existing Components Found**:
- ✅ `src/pipelines/monitors/memory_monitor.py` - Tracks memory every second
- ✅ `src/infrastructure/monitoring/unified_monitor.py` - Central monitoring
- ✅ `src/infrastructure/monitoring/database_metrics_backend.py` - Stores metrics
- ✅ Database tables: `pipeline_metrics` already exists

### 2.2 Add Database Storage to MemoryMonitor
**File**: `src/pipelines/monitors/memory_monitor.py`

**Step 1**: Add database backend reference
```python
def __init__(self, config, db_manager=None):  # Add db_manager parameter
    self.config = config
    self.db = db_manager  # Store reference
    # ... existing init code ...
```

**Step 2**: Modify `_monitor_loop` to store measurements
```python
def _monitor_loop(self):
    """Main monitoring loop."""
    while self._monitoring:
        try:
            # ... existing measurement code ...
            
            # Store to database if available
            if self.db and hasattr(self, 'experiment_id'):
                self._store_memory_snapshot(current_usage)
            
            # ... rest of existing code ...

def _store_memory_snapshot(self, usage: Dict[str, Any]):
    """Store memory snapshot to database."""
    try:
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO pipeline_metrics 
                    (experiment_id, metric_type, metric_name, metric_value, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    self.experiment_id,
                    'memory',
                    'process_rss_gb',
                    usage['process_rss_gb'],
                    json.dumps({
                        'system_percent': usage['system_percent'],
                        'system_available_gb': usage['system_available_gb'],
                        'stage': self.current_stage,
                        'operation': self.current_operation
                    })
                ))
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to store memory snapshot: {e}")
```

### 2.3 Add Stage/Operation Tracking
**File**: `src/pipelines/monitors/memory_monitor.py`

```python
# Add methods to track current context
def set_stage(self, stage: str):
    """Set current pipeline stage."""
    self.current_stage = stage
    logger.debug(f"Memory monitor tracking stage: {stage}")

def set_operation(self, operation: str):
    """Set current operation within stage."""
    self.current_operation = operation
```

### 2.4 Integrate with Pipeline Stages
**File**: `src/pipelines/stages/base_stage.py`

```python
# In PipelineStage base class, add memory tracking hooks
def execute(self, context) -> StageResult:
    """Execute stage with memory tracking."""
    # Get memory monitor from context
    memory_monitor = getattr(context, 'memory_monitor', None)
    
    if memory_monitor:
        memory_monitor.set_stage(self.name)
    
    try:
        # Call actual implementation
        result = self._execute_impl(context)
        
        # Add memory metrics to result
        if memory_monitor:
            summary = memory_monitor.get_summary()
            result.metrics.update({
                'memory_peak_gb': summary.get('peak_usage_gb', 0),
                'memory_avg_gb': summary.get('average_usage_gb', 0)
            })
        
        return result
    finally:
        if memory_monitor:
            memory_monitor.set_stage(None)
```

### 2.5 Add Memory Queries/Views
**File**: Create `src/database/sql/memory_analysis_views.sql`

```sql
-- Memory usage by stage
CREATE OR REPLACE VIEW v_memory_by_stage AS
SELECT 
    experiment_id,
    metadata->>'stage' as stage,
    AVG(metric_value) as avg_memory_gb,
    MAX(metric_value) as peak_memory_gb,
    MIN(metric_value) as min_memory_gb,
    COUNT(*) as samples
FROM pipeline_metrics
WHERE metric_type = 'memory' AND metric_name = 'process_rss_gb'
GROUP BY experiment_id, metadata->>'stage';

-- Memory usage timeline
CREATE OR REPLACE VIEW v_memory_timeline AS
SELECT 
    experiment_id,
    created_at,
    metric_value as memory_gb,
    metadata->>'stage' as stage,
    metadata->>'operation' as operation,
    (metadata->>'system_percent')::float as system_percent
FROM pipeline_metrics
WHERE metric_type = 'memory'
ORDER BY created_at;
```

### 2.6 Update UnifiedMonitor Integration
**File**: `src/infrastructure/monitoring/unified_monitor.py`

```python
# The UnifiedMonitor should already create MemoryMonitor
# Just ensure it passes db_manager:
self.memory_monitor = MemoryMonitor(config, db_manager=db_manager)
```

## Phase 3: Streaming Export (MEDIUM COMPLEXITY)

### 3.1 Analyze Existing Infrastructure
**Existing Components Found**:
- ✅ `src/processors/data_preparation/coordinate_merger.py` - Has chunked processing
- ✅ Database tables store resampled data
- ✅ `create_merged_dataset` already supports chunks

### 3.2 Add Streaming Interface to CoordinateMerger
**File**: `src/processors/data_preparation/coordinate_merger.py`

```python
def iter_merged_chunks(self, 
                      resampled_datasets: List[Dict],
                      chunk_size: int = 5000) -> Iterator[pd.DataFrame]:
    """
    Iterate over merged data chunks without loading full dataset.
    
    Yields:
        DataFrame chunks of merged data
    """
    # Determine coordinate ranges
    common_bounds = self._determine_common_bounds(resampled_datasets)
    
    # Create coordinate grid
    x_coords = np.arange(common_bounds[0], common_bounds[2], self.target_resolution)
    y_coords = np.arange(common_bounds[1], common_bounds[3], self.target_resolution)
    
    # Process in coordinate chunks
    for y_start in range(0, len(y_coords), chunk_size):
        y_end = min(y_start + chunk_size, len(y_coords))
        y_chunk = y_coords[y_start:y_end]
        
        # Query each dataset for this coordinate range
        chunk_data = {}
        for dataset in resampled_datasets:
            table_name = dataset['table_name']
            
            # Build query for coordinate range
            query = f"""
                SELECT x, y, value 
                FROM {table_name}
                WHERE y >= %s AND y < %s
                ORDER BY y, x
            """
            
            with self.db.get_connection() as conn:
                df = pd.read_sql_query(
                    query, 
                    conn,
                    params=(float(y_chunk[0]), float(y_chunk[-1]))
                )
            
            chunk_data[dataset['name']] = df
        
        # Merge this chunk
        if chunk_data:
            merged_chunk = self._merge_coordinate_chunk(chunk_data, y_chunk)
            yield merged_chunk
```

### 3.3 Update MergeStage for Streaming Mode
**File**: `src/pipelines/stages/merge_stage.py`

```python
def execute(self, context) -> StageResult:
    # ... existing validation code ...
    
    # Check if streaming mode is enabled
    enable_streaming = context.config.get('merge.enable_streaming', False)
    
    if enable_streaming and context.config.get('export.formats', ['csv']) == ['csv']:
        # For streaming, we don't create merged dataset
        # Instead, store configuration for ExportStage
        context.set('merge_mode', 'streaming')
        context.set('merge_config', {
            'dataset_dicts': dataset_dicts,
            'chunk_size': chunk_size,
            'merger': merger  # Store merger instance
        })
        
        logger.info("Merge configured for streaming mode")
        
        return StageResult(
            success=True,
            data={'mode': 'streaming'},
            metrics={'streaming_enabled': True}
        )
    else:
        # Existing in-memory merge
        merged_dataset = merger.create_merged_dataset(...)
        context.set('merged_dataset', merged_dataset)
        # ... rest of existing code ...
```

### 3.4 Update ExportStage for Streaming
**File**: `src/pipelines/stages/export_stage.py`

```python
def execute(self, context) -> StageResult:
    # Check merge mode
    merge_mode = context.get('merge_mode', 'in_memory')
    
    if merge_mode == 'streaming':
        return self._execute_streaming(context)
    else:
        return self._execute_in_memory(context)

def _execute_streaming(self, context) -> StageResult:
    """Export using streaming to avoid loading full dataset."""
    merge_config = context.get('merge_config')
    merger = merge_config['merger']
    
    # Only supports CSV for streaming
    output_path = context.output_dir / f"merged_data_{context.experiment_id}.csv"
    
    rows_exported = 0
    first_chunk = True
    
    # Stream chunks and write
    for chunk_df in merger.iter_merged_chunks(
        merge_config['dataset_dicts'],
        merge_config['chunk_size']
    ):
        # Write chunk
        chunk_df.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        
        first_chunk = False
        rows_exported += len(chunk_df)
        
        # Log progress
        if rows_exported % 10000 == 0:
            logger.info(f"Streaming export progress: {rows_exported:,} rows")
    
    # ... create result ...
```

### 3.5 Add Configuration Options
**File**: `config.yml`

```yaml
merge:
  enable_streaming: false  # Set to true for memory-efficient export
  streaming_chunk_size: 5000  # Rows per chunk in streaming mode
```

## Testing Strategy

### Phase 1 Tests (Memory Pressure)
1. Create script that allocates large arrays to trigger pressure
2. Monitor window size changes in logs
3. Verify processing completes successfully

### Phase 2 Tests (Memory Tracking)
1. Run pipeline with memory tracking enabled
2. Query database for memory metrics
3. Verify stage-level memory attribution

### Phase 3 Tests (Streaming)
1. Enable streaming in config
2. Compare memory usage: streaming vs in-memory
3. Verify output file identical

## Success Criteria

1. **No modifications to**:
   - Any files in `base/` (except using existing interfaces)
   - Any files in `abstractions/`
   - Core infrastructure behavior

2. **All changes isolated to**:
   - `processors/` - business logic
   - `pipelines/` - orchestration updates
   - `config.yml` - new options

3. **Backward compatibility**:
   - All existing pipelines work unchanged
   - New features are opt-in via configuration