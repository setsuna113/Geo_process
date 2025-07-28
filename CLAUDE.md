# Geo Project Architecture Summary

## Project Overview
This is a geospatial biodiversity analysis system that processes large raster datasets and species distribution data. The system is built around a PostgreSQL/PostGIS database with a comprehensive caching system for efficient spatial operations.

## Critical Implementation Notes (2025-07-26)

### Recent Architectural Analysis Results
- **Unified Checkpoint System**: Successfully implemented, replaced 5 separate checkpoint systems with 1 clean implementation
- **Major Issues Discovered**: 
  - Complete module duplication (raster/ vs raster_data/): 8,000-10,000 lines of duplicate code
  - Dependency inversion violations in base/ layer (imports from core/, checkpoints/, database/)
  - Config import pattern inconsistency causing multiple config instances
  - 3 separate memory management systems with 70% overlap
  - Circular dependencies between spatial_analysis/ and processors/

### Critical Configuration Rules
- **ALWAYS** use `from src.config import config` (singleton instance)
- **NEVER** use `from src.config import Config` then `config = Config()` (creates new instance!)
- **NEVER** modify `defaults.py` directly - use `config.yml` for overrides
- **NEVER** use `DatabaseManager(test_mode=True)` in production code

### Previous Implementation Notes (2025-07-22)
- **RasterAligner** (`src/processors/data_preparation/raster_alignment.py`): Handles misaligned datasets automatically, fixes transform pixel size extraction bug (use transform[0] not transform[1])
- **DatabaseSchemaUtils** (`src/database/utils.py`): Adapts to different column names (spatial_extent vs bounds) via configuration-driven schema mapping
- **RasterMerger**: Refactored to use RasterAligner, manages temporary alignment files properly to prevent file deletion before merge completion
- **Long-running processes**: Use `./run_analysis.sh` with tmux for checkpoint-based resumability and multi-pane monitoring

## Core Architecture

### Database Layer (`/src/database/`)
**Primary Components:**
- `connection.py`: `DatabaseManager` class with connection pooling, test mode detection, and PostgreSQL service management
- `schema.py`: `DatabaseSchema` class providing comprehensive data access layer with methods for:
  - Grid operations (store/retrieve grid definitions and cells)
  - Species range operations (store species data and grid intersections)
  - Feature storage (computed features, climate data)
  - Experiment tracking (analysis jobs and results)
  - Raster processing (tile management, resampling cache)
- `setup.py`: Database initialization and validation utilities

**Key Database Features:**
- Connection pooling with retry logic and exponential backoff
- Test mode safety checks with data cleanup capabilities  
- Bulk insert operations for performance
- Sophisticated SQL parsing for schema management
- Comprehensive caching system for raster resampling

### Configuration System (`/src/config/`)
- `config.py`: `Config` class with YAML override support and dot notation access
- `defaults.py`: Default configuration values for all system components
- Auto-discovery of `config.yml` in project root
- Deep merging of configuration hierarchies

### Component Registry (`/src/core/`)
- `registry.py`: Enhanced registry system for dynamic component management
- Metadata-driven component registration with capability flags
- Format compatibility matrix for data sources
- Memory usage tracking and optimal component selection
- Validation system for registered components

### Grid Systems (`/src/grid_systems/`)
- `grid_factory.py`: Factory pattern for creating multi-resolution grids
- `bounds_manager.py`: Geographic boundary management
- Support for cubic and hexagonal grid systems
- Standard resolution hierarchies (5km to 100km)
- Grid compatibility validation and data upscaling/downscaling

### Base Classes (`/src/base/`)
- `dataset.py`: `BaseDataset` abstract class with enhanced features:
  - Multiple data types (raster, vector, tabular, etc.)
  - Tile-based access for large datasets
  - Memory estimation for operations
  - Lazy loading and chunked reading
- Abstract classes for processors, grids, and features

### Raster Processing (`/src/raster_data/`)
- `catalog.py`: `RasterCatalog` for managing raster data sources
- Loader system (`loaders/`) with format-specific implementations
- Validation system (`validators/`) for coverage and value validation
- Metadata extraction and database storage

### Spatial Analysis (`/src/spatial_analysis/`)
**Key Analyzers:**
- `gwpca/gwpca_analyzer.py`: Geographically Weighted PCA with block aggregation
- `som/som_trainer.py`: Self-Organizing Maps for biodiversity pattern clustering
- `base_analyzer.py`: Common analysis framework with progress tracking

### Data Processors (`/src/processors/`)
- `data_preparation/data_normalizer.py`: Spatial data normalization with metadata preservation
- Array conversion and cleaning utilities
- Raster merging capabilities

## Module Interactions

### Primary Data Flow:
1. **Configuration Loading**: `config.Config` loads settings from YAML and defaults
2. **Database Connection**: `DatabaseManager` establishes pooled connections with safety checks
3. **Component Registration**: `ComponentRegistry` registers processors, grids, and data sources
4. **Data Ingestion**: Raster catalog scans and validates data sources
5. **Grid Creation**: `GridFactory` creates multi-resolution spatial grids
6. **Processing Pipeline**: Data processors normalize and prepare datasets
7. **Spatial Analysis**: GWPCA/SOM analyzers process gridded biodiversity data
8. **Result Storage**: Analysis results stored in database with full provenance

### Key Integration Points:
- **Database Schema**: Central hub for all data storage and retrieval
- **Component Registry**: Enables dynamic selection of optimal processors
- **Grid Factory**: Provides standardized spatial frameworks
- **Config System**: Drives behavior across all components

### Database Call Patterns:
Most modules interact with the database through `schema` methods:
- `schema.store_grid_definition()` and `schema.store_grid_cells_batch()`
- `schema.store_species_range()` and `schema.store_species_intersections_batch()`
- `schema.store_features_batch()` and `schema.store_climate_data_batch()`
- `schema.create_experiment()` and `schema.update_experiment_status()`

The caching system is heavily used for raster operations:
- `schema.store_resampling_cache_batch()` and `schema.get_cached_resampling_values()`
- Processing queue management for distributed processing

## Notable Features:
- **Test Mode Safety**: Comprehensive test data isolation and cleanup
- **Performance Optimization**: Extensive caching, bulk operations, and memory management
- **Spatial Efficiency**: Multi-resolution grids with intelligent aggregation
- **Extensibility**: Plugin architecture via component registry
- **Data Provenance**: Full experiment tracking and metadata preservation

## Architectural Refactoring Plan (2025-07-26)

### Target Architecture (system_update.md active)
```
src/
├── foundations/     # Pure abstractions only (no src imports allowed)
├── infrastructure/  # Technical services (memory, progress, registry)
├── domain/         # Business logic (unified raster, spatial, analysis)
├── application/    # Workflows and processors
└── orchestration/  # Pipeline coordination
```

### Lessons from Checkpoint System Success:
- Clean abstractions in base/ with implementations in separate module
- Comprehensive testing at each layer
- Gradual migration with compatibility layers
- Feature flags for safe rollout

### Daemon Process Management

The system uses a persistent process registry for tracking daemon processes across invocations:

- **Registry location**: `~/.biodiversity/pid/registry/`
- **Process state**: Each daemon has a JSON file with full metadata (PID, status, logs, etc.)
- **Automatic cleanup**: Stale processes cleaned up on startup
- **File locking**: Atomic operations prevent race conditions
- **Cross-invocation tracking**: Different process manager invocations can manage the same daemons

#### Common Daemon Commands:
```bash
# Start daemon with unique experiment name
./run_pipeline.sh --daemon --process-name "production_run"
python scripts/process_manager.py start --name "my_daemon" --daemon --experiment-name "unique_name"

# Check daemon status (works from any terminal)
python scripts/process_manager.py status production_run
python scripts/process_manager.py status  # List all daemons

# View daemon logs in real-time
python scripts/process_manager.py logs production_run -f
python scripts/process_manager.py logs production_run --lines 100

# Stop daemon gracefully
python scripts/process_manager.py stop production_run

# Monitor system resources
python scripts/process_manager.py resources
```

#### Critical Implementation Notes:
- **ALWAYS** use unique experiment names to avoid database constraint violations
- **NEVER** assume daemons persist after system reboot (they don't)
- **ALWAYS** check daemon status before starting new ones with same name
- Registry handles PID file cleanup automatically

This architecture supports large-scale biodiversity analysis workflows with robust data management, efficient spatial processing, comprehensive analysis capabilities, and reliable daemon process management. The ongoing refactoring (system_update.md) will eliminate technical debt while maintaining system stability.
