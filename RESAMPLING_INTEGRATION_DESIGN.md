# Resampling Integration Design Document

## Overview
Upgrade the current richness processing pipeline to support:
1. **Multi-dataset resampling** to uniform target resolution
2. **Database storage** of resampled datasets  
3. **Coordinate alignment and merging** of resampled datasets
4. **Scalable pipeline** for N datasets → uniform database → SOM analysis

## Required Component Updates

### 1. Config.yml Extensions

```yaml
# NEW: Resampling Configuration Section
resampling:
  # Target resolution for all datasets (in degrees for geographic data)
  target_resolution: 0.05  # ~5km at equator
  target_crs: 'EPSG:4326'
  
  # Resampling strategy per data type
  strategies:
    richness_data: 'sum'          # Sum for count data
    continuous_data: 'bilinear'   # Bilinear for continuous
    categorical_data: 'majority'  # Majority for categories
  
  # Processing options
  chunk_size: 1000
  validate_output: true
  preserve_sum: true  # Important for richness data
  cache_resampled: true

# NEW: Multiple Dataset Definitions
datasets:
  target_datasets:
    - name: "plants-richness"
      path_key: "plants_richness"  # References data_files section
      data_type: "richness_data"
      band_name: "plants_richness"
      
    - name: "terrestrial-richness" 
      path_key: "terrestrial_richness"
      data_type: "richness_data"
      band_name: "terrestrial_richness"
      
    # Future extensibility
    - name: "climate-temperature"
      path_key: "climate_temp"
      data_type: "continuous_data"
      band_name: "temperature"
      enabled: false  # Can disable datasets

# ENHANCED: Database configuration for resampled data storage
database:
  # ... existing config ...
  
  # New tables for resampled data
  resampled_data_storage:
    table_prefix: "resampled_"
    store_intermediate: true    # Store individual resampled datasets
    store_merged: true         # Store final merged dataset
    cleanup_intermediate: false # Keep for debugging/analysis
```

### 2. Processors/ Updates

#### New: ResamplingProcessor (`src/processors/data_preparation/resampling_processor.py`)

```python
class ResamplingProcessor(BaseProcessor):
    """Handles resampling of datasets to target resolution with database storage."""
    
    def __init__(self, config: Config, db: DatabaseManager):
        # Integration with existing resampling engines
        self.resampler_engine = self._create_resampler_engine(config)
        self.catalog = RasterCatalog(db, config)
        
    def resample_dataset(self, dataset_config: dict) -> ResampledDatasetEntry:
        """Resample single dataset and store in database."""
        
    def resample_all_datasets(self, dataset_configs: List[dict]) -> List[ResampledDatasetEntry]:
        """Resample all configured datasets."""
        
    def get_resampled_dataset(self, name: str) -> Optional[ResampledDatasetEntry]:
        """Retrieve resampled dataset from database."""
```

#### Enhanced: RasterMerger Updates
- Integration with `ResamplingProcessor`
- Support for pre-resampled datasets
- Enhanced coordinate alignment validation

### 3. Database/ Updates

#### New Schema Extensions (`src/database/schema.py`)

```python
class DatabaseSchema:
    # ... existing methods ...
    
    # NEW: Resampled data management
    def store_resampled_dataset(self, dataset_info: ResampledDatasetInfo) -> int:
        """Store resampled dataset metadata and optionally data."""
        
    def get_resampled_datasets(self, filters: dict = None) -> List[ResampledDatasetEntry]:
        """Retrieve resampled datasets."""
        
    def create_resampled_data_table(self, dataset_name: str, columns: dict):
        """Create table for storing resampled dataset values."""
        
    def store_resampled_values_batch(self, table_name: str, values: np.ndarray):
        """Batch store resampled values."""
```

#### New Tables
```sql
-- Resampled dataset registry
CREATE TABLE resampled_datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    source_raster_id INTEGER REFERENCES raster_sources(id),
    target_resolution FLOAT NOT NULL,
    target_crs VARCHAR(50) NOT NULL,
    resampling_method VARCHAR(50) NOT NULL,
    bounds GEOMETRY(POLYGON, 4326),
    shape_height INTEGER NOT NULL,
    shape_width INTEGER NOT NULL,
    data_table_name VARCHAR(255),  -- Reference to actual data table
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Dynamic data tables (created per dataset)
-- Example: resampled_plants_richness
CREATE TABLE resampled_{dataset_name} (
    row_idx INTEGER NOT NULL,
    col_idx INTEGER NOT NULL,
    value FLOAT,
    PRIMARY KEY (row_idx, col_idx)
);
```

### 4. New Pipeline Folder Structure

```
src/pipelines/
├── __init__.py
├── unified_resampling/
│   ├── __init__.py
│   ├── pipeline_orchestrator.py     # Main orchestration logic
│   ├── dataset_processor.py         # Dataset-specific processing
│   ├── resampling_workflow.py       # Resampling workflow management
│   └── validation_checks.py         # Data validation utilities
├── configs/
│   ├── default_resampling.yml       # Default resampling parameters  
│   └── dataset_definitions.yml      # Dataset configuration templates
└── scripts/
    ├── run_unified_resampling.py    # Main execution script
    ├── run_unified_resampling.sh    # Shell wrapper
    └── monitor_resampling.py        # Progress monitoring
```

## Implementation Flow

### Phase 1: Core Infrastructure
1. **Config Extensions**: Add resampling and datasets sections
2. **Database Schema**: Create resampled dataset tables
3. **ResamplingProcessor**: Core resampling logic with database integration

### Phase 2: Pipeline Integration  
1. **Enhanced RasterMerger**: Integration with resampled datasets
2. **Pipeline Orchestrator**: Workflow management (resample → store → merge → SOM)
3. **Validation Systems**: Data quality checks throughout pipeline

### Phase 3: Execution Scripts
1. **Unified Processing Script**: Inherits from `process_richness_datasets.py`
2. **Shell Scripts**: Enhanced monitoring and execution
3. **Progress Tracking**: Real-time monitoring of multi-step pipeline

## Key Design Principles

1. **Abstraction**: Resampling logic abstracted in processors/, orchestration in pipelines/
2. **Extensibility**: Easy addition of new datasets via config
3. **Database Integration**: Persistent storage of intermediate results
4. **Validation**: Comprehensive checks at each pipeline stage
5. **Monitoring**: Progress tracking for long-running processes
6. **Recovery**: Checkpoint-based resumability

## Integration with Existing System

- **Maintains Compatibility**: Existing `process_richness_datasets.py` continues to work
- **Gradual Migration**: New pipeline can run alongside existing workflows  
- **Shared Components**: Leverages existing `RasterAligner`, `SOMAnalyzer`, etc.
- **Enhanced Capabilities**: Adds resampling without breaking current functionality

This design provides a robust foundation for scaling from 2 datasets to N datasets with uniform resampling, database storage, and SOM analysis.