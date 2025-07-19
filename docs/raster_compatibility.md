# Raster Data Compatibility Guide

This document explains the new raster data compatibility features added to the geo processing system.

## Overview

The raster compatibility update adds support for:
- Raster data source management
- Efficient spatial tiling for large rasters
- Resampling cache for performance optimization
- Distributed processing queue for scalable operations

## New Database Tables

### 1. `raster_sources`
Stores metadata for each raster file:
- **Purpose**: Track raster data files and their properties
- **Key fields**: 
  - `pixel_size_degrees`: Resolution (default 0.0167° ≈ 1.85km)
  - `data_type`: Data format (Int32, UInt16, Float32, etc.)
  - `nodata_value`: Value representing missing data
  - `spatial_extent`: Geographic bounds
  - `processing_status`: Current processing state

### 2. `raster_tiles`
Divides large rasters into manageable chunks:
- **Purpose**: Enable efficient random access to raster data
- **Key fields**:
  - `tile_size_pixels`: Size of each tile (default 1000x1000)
  - `tile_bounds`: Geographic bounds of the tile
  - `file_byte_offset`: Position in source file for direct access
  - `tile_stats`: Pre-computed statistics (min, max, mean, std)

### 3. `resampling_cache`
Caches resampled values to avoid recomputation:
- **Purpose**: Performance optimization for repeated resampling operations
- **Key fields**:
  - `method`: Resampling algorithm (nearest, bilinear, cubic, average)
  - `confidence_score`: Quality metric (0-1)
  - `last_accessed`: For cache management
  - `access_count`: Usage tracking

### 4. `processing_queue`
Manages distributed processing tasks:
- **Purpose**: Track and coordinate raster processing workflows
- **Key fields**:
  - `queue_type`: Task type (raster_tiling, resampling, validation)
  - `priority`: Task priority for scheduling
  - `worker_id`: Assigned processing worker
  - `retry_count`: Error recovery tracking

## Configuration

New configuration section in `defaults.py`:

```python
RASTER_PROCESSING = {
    'tile_size': 1000,           # pixels per tile
    'cache_ttl_days': 30,        # cache retention period
    'memory_limit_mb': 4096,     # memory limit for operations
    'parallel_workers': 4,       # concurrent processing workers
    'lazy_loading': {
        'chunk_size_mb': 100,    # data chunk size
        'prefetch_tiles': 2      # tiles to prefetch
    },
    'resampling_methods': {
        'default': 'bilinear',
        'categorical': 'nearest',
        'continuous': 'bilinear'
    }
}
```

## Python API

### RasterManager Class

```python
from src.database.raster import raster_manager

# Register a new raster source
raster_id = raster_manager.register_raster_source(
    file_path='data/richness_maps/daru-plants-richness.tif',
    name='daru_plants',
    metadata={
        'source_dataset': 'DARU',
        'variable_name': 'plant_richness',
        'units': 'species_count',
        'description': 'Plant species richness'
    }
)

# Create spatial tiles
tile_count = raster_manager.create_raster_tiles(raster_id)

# Resample to grid cells
results = raster_manager.resample_to_grid(
    raster_id=raster_id,
    grid_id='grid-123',
    method='bilinear',
    use_cache=True
)

# Check processing status
status = raster_manager.get_processing_status(raster_id)

# Clean up old cache
deleted = raster_manager.cleanup_cache(days_old=30, min_access_count=2)
```

### Database Schema Operations

```python
from src.database.schema import schema

# Store raster source
raster_data = {
    'name': 'bio_1_annual',
    'file_path': '/data/worldclim/bio_1.tif',
    'data_type': 'Float32',
    'pixel_size_degrees': 0.016666666666667,
    'spatial_extent_wkt': 'POLYGON(...)',
    # ... other metadata
}
raster_id = schema.store_raster_source(raster_data)

# Get raster sources
sources = schema.get_raster_sources(active_only=True)

# Cache resampling results
cache_data = [{
    'source_raster_id': raster_id,
    'target_grid_id': grid_id,
    'cell_id': 'cell_123',
    'method': 'bilinear',
    'value': 25.5
}]
schema.store_resampling_cache_batch(cache_data)

# Add processing task
task_id = schema.add_processing_task(
    queue_type='raster_tiling',
    raster_id=raster_id,
    priority=1
)
```

## Command Line Interface

The `raster_cli.py` tool provides command-line access to raster operations:

```bash
# Register a raster source
./raster_cli.py register data/richness_maps/daru-plants-richness.tif \
    --name daru_plants \
    --dataset DARU \
    --variable plant_richness \
    --description "Plant species richness"

# List raster sources
./raster_cli.py list --status ready

# Create tiles manually
./raster_cli.py tile <raster-id>

# Resample to grid
./raster_cli.py resample <raster-id> <grid-id> \
    --method bilinear \
    --output results.json

# Check processing status
./raster_cli.py status --raster-id <raster-id>

# View cache statistics
./raster_cli.py cache-stats

# Clean up old cache
./raster_cli.py cache-cleanup --days-old 30 --min-access 2

# Process queued tasks
./raster_cli.py process-tasks raster_tiling --max-tasks 5

# Show configuration
./raster_cli.py config-info
```

## Usage Examples

### 1. Processing Plant Richness Data

```python
# Register the DARU plant richness raster
raster_id = raster_manager.register_raster_source(
    file_path='data/richness_maps/daru-plants-richness.tif',
    name='daru_plants_richness',
    metadata={
        'source_dataset': 'DARU',
        'variable_name': 'plant_richness',
        'units': 'species_count',
        'description': 'Vascular plant species richness',
        'temporal_info': {
            'reference_period': 'current',
            'data_year': 2020
        }
    }
)

# Create tiles for efficient access
tile_count = raster_manager.create_raster_tiles(raster_id)
print(f"Created {tile_count} tiles")

# Resample to your grid
grid_id = 'your_analysis_grid'
plant_richness = raster_manager.resample_to_grid(
    raster_id=raster_id,
    grid_id=grid_id,
    method='bilinear'  # Smooth interpolation for continuous data
)

# Store results in your analysis
for cell_id, richness_value in plant_richness.items():
    # Store in features table or use directly
    schema.store_feature(
        grid_id=grid_id,
        cell_id=cell_id,
        feature_type='richness',
        feature_name='plant_richness_daru',
        feature_value=richness_value
    )
```

### 2. Processing IUCN Terrestrial Data

```python
# Register IUCN data
iucn_raster_id = raster_manager.register_raster_source(
    file_path='data/richness_maps/iucn-terrestrial-richness.tif',
    name='iucn_terrestrial_richness',
    metadata={
        'source_dataset': 'IUCN Red List',
        'variable_name': 'terrestrial_species_richness',
        'units': 'species_count',
        'description': 'Terrestrial vertebrate species richness'
    }
)

# Use nearest neighbor for species count data (discrete values)
iucn_richness = raster_manager.resample_to_grid(
    raster_id=iucn_raster_id,
    grid_id=grid_id,
    method='nearest'  # Preserve discrete counts
)
```

### 3. Batch Processing with Queue

```python
# Add multiple resampling tasks to queue
raster_ids = ['raster1', 'raster2', 'raster3']
grid_ids = ['grid_a', 'grid_b']

for raster_id in raster_ids:
    for grid_id in grid_ids:
        schema.add_processing_task(
            queue_type='resampling',
            raster_id=raster_id,
            grid_id=grid_id,
            parameters={
                'method': 'bilinear',
                'band_number': 1
            },
            priority=1
        )

# Process tasks with multiple workers
import threading

def worker(worker_id):
    while True:
        success = raster_manager.process_queue_task('resampling', worker_id)
        if not success:
            break  # No more tasks

# Start 4 workers
threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=[f'worker_{i}'])
    t.start()
    threads.append(t)

# Wait for completion
for t in threads:
    t.join()
```

## Performance Considerations

### Tiling Strategy
- Default tile size: 1000x1000 pixels (balance between memory and I/O)
- Larger tiles: Better for sequential access, more memory usage
- Smaller tiles: Better for random access, more overhead

### Caching Strategy
- Cache frequently accessed resampling results
- Clean up cache based on age and access patterns
- Monitor cache hit rate with `cache-stats` command

### Memory Management
- Configure `memory_limit_mb` based on available RAM
- Use `chunk_size_mb` for lazy loading of large rasters
- Prefetch tiles to reduce I/O latency

### Parallel Processing
- Set `parallel_workers` based on CPU cores
- Use processing queue for distributed workflows
- Monitor queue status to identify bottlenecks

## Integration with Existing Modules

### Grid Systems
Raster resampling integrates seamlessly with existing grid systems:

```python
from src.grid_systems import CubicGrid, HexagonalGrid

# Create grid
grid = CubicGrid(resolution=5000, bounds=study_area)
grid_id = grid.store_to_database()

# Resample raster to grid
raster_values = raster_manager.resample_to_grid(
    raster_id=raster_id,
    grid_id=grid_id
)
```

### Feature Processing
Raster values can be stored as features:

```python
# Store raster-derived features
for cell_id, value in raster_values.items():
    schema.store_feature(
        grid_id=grid_id,
        cell_id=cell_id,
        feature_type='environmental',
        feature_name='temperature_annual_mean',
        feature_value=value,
        metadata={
            'source_raster': raster_id,
            'resampling_method': 'bilinear',
            'extraction_date': datetime.now().isoformat()
        }
    )
```

## Migration and Compatibility

### Applying the Migration
The raster tables are automatically created when you run `setup_database()`. For existing databases:

```python
from src.database.schema import schema

# Apply the raster migration
success = schema.run_migration('001_add_raster_tables.sql')
```

### Backward Compatibility
- All existing functionality remains unchanged
- New raster features are optional
- Existing grid and feature workflows continue to work

## Monitoring and Maintenance

### Regular Maintenance Tasks

```bash
# Weekly cache cleanup
./raster_cli.py cache-cleanup --days-old 7 --min-access 1

# Check processing status
./raster_cli.py queue-status

# Monitor cache efficiency
./raster_cli.py cache-stats
```

### Performance Monitoring

```python
# Check cache hit rates
cache_stats = raster_manager.get_cache_statistics()
for stat in cache_stats:
    hit_rate = stat['avg_access_count']
    print(f"Raster {stat['raster_name']}: {hit_rate:.1f} avg accesses")

# Monitor processing queue
queue_summary = schema.get_processing_queue_summary()
for qs in queue_summary:
    pending_tasks = qs.get('task_count', 0)
    if pending_tasks > 100:
        print(f"Warning: {pending_tasks} pending {qs['queue_type']} tasks")
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `tile_size` or `memory_limit_mb`
   - Process rasters in smaller chunks

2. **Slow Resampling**
   - Check cache hit rates
   - Ensure proper indexing on database tables
   - Consider using coarser resampling methods

3. **Queue Backlog**
   - Increase number of workers
   - Check for failed tasks with high retry counts
   - Monitor worker performance

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('src.database.raster').setLevel(logging.DEBUG)

# Check task failures
failed_tasks = schema.get_processing_queue_summary()
failed_tasks = [t for t in failed_tasks if t['status'] == 'failed']
```

This completes the raster compatibility implementation for your geo processing system!
