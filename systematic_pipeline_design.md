# Systematic Pipeline Design

## Recommended Architecture: Direct Export with Bypass Options

```
┌─────┐    ┌────┐    ┌─────────────────────┐    ┌────┐
│ TIF ├───→│ DB ├───→│   Export Stage      ├───→│ ML │
└─────┘    └────┘    │                     │    └────┘
                     │ ┌─────────────────┐ │
                     │ │ Format Router   │ │
                     │ └────────┬────────┘ │
                     │          ↓          │
                     │ ┌─────────────────┐ │
                     │ │ parquet → ML    │ │
                     │ ├─────────────────┤ │
                     │ │ csv → external  │ │
                     │ ├─────────────────┤ │
                     │ │ none → direct   │ │
                     │ └─────────────────┘ │
                     └─────────────────────┘
```

## Why This Design?

### 1. **Performance**
- **DB → Parquet → ML**: 63 seconds (fastest for large data)
- **DB → ML (direct)**: 50 seconds (fastest for small data)
- **DB → CSV**: 170 seconds (only when needed)
- **Avoid**: DB → CSV → Parquet (273 seconds, wasteful)

### 2. **Flexibility**
```yaml
# config.yml
export:
  format: parquet  # Options: parquet, csv, none
  
  # Format-specific options
  parquet:
    engine: pyarrow
    compression: snappy
    
  csv:
    include_index: false
    compression: none  # or gzip
    
  bypass:
    max_rows_for_direct: 1000000  # Skip export if smaller
```

### 3. **Storage Efficiency**
- Parquet: ~800MB (compressed, columnar)
- CSV: ~3.4GB (uncompressed text)
- CSV.gz: ~400MB (compressed but slow to read)

## Implementation Plan

### Phase 1: Fix Current Issues
1. **Fix MergeStage** to read from DB instead of TIF
2. **Fix coordinate alignment** bug
3. **Remove NetCDF** intermediate step

### Phase 2: Implement New Architecture
```python
# Simplified pipeline
stages = [
    DataLoadStage,      # TIF → DB metadata
    ResampleStage,      # TIF → DB passthrough tables
    ExportStage,        # DB → Parquet/CSV/None
    AnalysisStage       # Parquet/DB → ML
]
```

### Phase 3: Optimize Queries
```sql
-- Efficient export query with proper joins
SELECT 
    t.latitude, t.longitude,
    t.value as terrestrial_richness,
    p.value as plants_richness
FROM (
    SELECT 
        lat_from_idx(row_idx) as latitude,
        lon_from_idx(col_idx) as longitude,
        value
    FROM passthrough_terrestrial_richness
) t
FULL OUTER JOIN (
    SELECT 
        lat_from_idx(row_idx) as latitude,
        lon_from_idx(col_idx) as longitude,
        value
    FROM passthrough_plants_richness
) p
ON t.latitude = p.latitude AND t.longitude = p.longitude
WHERE t.value IS NOT NULL OR p.value IS NOT NULL;
```

## Benefits

1. **Simplicity**: Remove redundant NetCDF step
2. **Speed**: 4.3x faster than CSV intermediate
3. **Flexibility**: Choose format based on use case
4. **Efficiency**: Parquet for ML, CSV for compatibility
5. **Scalability**: Can handle datasets larger than RAM