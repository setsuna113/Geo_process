# Geo Pipeline Refactoring Plan

## 1. Critical Data Integrity Issue Analysis

### Current Issue
All sampled values from the parquet file are mismatched with the original raster values.

### Mismatch Pattern Analysis
- **Parquet values are consistently different** from raster values (e.g., Parquet=171, Raster=105)
- **Not a simple offset** - the differences vary (66, 20, 60, 406, etc.)
- **Some zero values in raster show as non-zero in parquet** (e.g., Raster=0, Parquet=7.0)
- **Some non-zero raster values show as NaN in parquet** (e.g., Raster=528, Parquet=nan)

### Suspected Root Causes

1. **Coordinate System Mismatch**: The bounds stored in database metadata don't match actual raster bounds
   - IUCN bounds in DB: `[-180, -90, 180, 90]` (generic world bounds)
   - IUCN actual bounds: `[-180.0, -54.996..., 180.000..., 83.004...]`
   - This causes row/col to coordinate conversion to be incorrect

2. **Row/Column Indexing Issue**: The passthrough tables use row_idx/col_idx, but the conversion formula might be using wrong origin
   - Current formula: `y = max_y - (row_idx + 0.5) * resolution`
   - Should verify if row_idx=0 is at top or bottom of raster

3. **Data Loading Mismatch**: The values stored in passthrough tables might be from different array positions than expected

## 2. Immediate Fixes Needed

- [ ] **Fix coordinate conversion in `fast_parquet_export.py`**:
  - Use actual raster bounds from raster files, not generic bounds
  - Verify row indexing direction (top-to-bottom vs bottom-to-top)
  - Test conversion with known pixel values

- [ ] **Update `resampling_processor.py`** to store correct bounds:
  - When creating passthrough tables, store actual raster bounds in metadata
  - Currently stores generic bounds which causes misalignment

- [ ] **Fix merge stage** to properly delegate to processors:
  - Currently reimplements coordinate conversion
  - Should use processors for data access

## 3. Systematic Workflow Changes - Detailed Implementation

### A. Architecture Overview

```
STAGES (Orchestration Layer)          PROCESSORS (Implementation Layer)
│                                    │
├── Load Stage                       ├── DataLoader (new)
│   ├── validate inputs              │   ├── scan_datasets()
│   ├── register datasets            │   ├── validate_paths()
│   └── store metadata               │   └── register_in_catalog()
│                                    │
├── Resample Stage                   ├── ResamplingProcessor (existing)
│   ├── get datasets list            │   ├── check_resolution_match()
│   ├── delegate to processor        │   ├── resample_dataset()
│   └── update context               │   └── store_passthrough_data()
│                                    │
├── Merge Stage                      ├── CoordinateMerger (new)
│   ├── get resampled list           │   ├── load_passthrough_tables()
│   ├── delegate to merger           │   ├── convert_coordinates()
│   └── store output path            │   └── merge_to_parquet()
│                                    │
├── Export Stage                     ├── ParquetExporter (new)
│   ├── get merge output             │   ├── validate_parquet()
│   └── delegate export              │   └── add_metadata()
│                                    │
└── Analysis Stage                   └── Analyzers (existing)
    ├── get ML-ready data                ├── SOMAnalyzer
    └── run selected analyzer            └── GWPCAAnalyzer
```

### B. File Changes Required

#### 1. **Create New Processor: `/src/processors/data_preparation/coordinate_merger.py`**

```python
# src/processors/data_preparation/coordinate_merger.py
"""Coordinate-based merger for passthrough and resampled data."""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.base.processor import BaseProcessor
from src.database.connection import DatabaseManager
import rasterio

logger = logging.getLogger(__name__)

class CoordinateMerger(BaseProcessor):
    """Merges datasets from passthrough/resampled tables into ML-ready format."""
    
    def __init__(self, config, db: DatabaseManager):
        super().__init__(config=config)
        self.db = db
        
    def create_ml_ready_parquet(self, 
                               resampled_datasets: List[Dict], 
                               output_dir: Path) -> Path:
        """Main entry point - creates parquet from database tables."""
        
        # Step 1: Load coordinate data from each dataset
        all_dfs = []
        for dataset_info in resampled_datasets:
            df = self._load_dataset_coordinates(dataset_info)
            if df is not None:
                all_dfs.append(df)
        
        # Step 2: Merge all datasets
        merged_df = self._merge_coordinate_datasets(all_dfs)
        
        # Step 3: Save as parquet
        output_path = output_dir / 'ml_ready_aligned_data.parquet'
        merged_df.to_parquet(output_path, index=False)
        
        return output_path
    
    def _load_dataset_coordinates(self, dataset_info: Dict) -> Optional[pd.DataFrame]:
        """Load coordinates from passthrough or resampled table."""
        
        # Get actual bounds from the original raster file
        actual_bounds = self._get_raster_bounds(dataset_info['source_path'])
        
        if dataset_info.get('passthrough', False):
            return self._load_passthrough_coordinates(
                dataset_info['name'],
                dataset_info['table_name'],
                actual_bounds,
                dataset_info['resolution']
            )
        else:
            return self._load_resampled_coordinates(
                dataset_info['name'],
                dataset_info['table_name']
            )
    
    def _get_raster_bounds(self, raster_path: str) -> Tuple[float, float, float, float]:
        """Get actual bounds from raster file."""
        with rasterio.open(raster_path) as src:
            return src.bounds
    
    def _load_passthrough_coordinates(self, name: str, table_name: str, 
                                    bounds: Tuple, resolution: float) -> pd.DataFrame:
        """Convert row/col indices to coordinates using actual raster bounds."""
        
        min_x, min_y, max_x, max_y = bounds
        
        # SQL query that does coordinate conversion
        query = f"""
            SELECT 
                {min_x} + (col_idx + 0.5) * {resolution} AS x,
                {max_y} - (row_idx + 0.5) * {resolution} AS y,
                value AS {name.replace('-', '_')}
            FROM {table_name}
            WHERE value IS NOT NULL AND value != 0
        """
        
        with self.db.get_connection() as conn:
            return pd.read_sql(query, conn)
    
    def _load_resampled_coordinates(self, name: str, table_name: str) -> pd.DataFrame:
        """Load already converted coordinates from resampled table."""
        
        query = f"""
            SELECT x, y, value AS {name.replace('-', '_')}
            FROM {table_name}
            WHERE value IS NOT NULL AND value != 0
        """
        
        with self.db.get_connection() as conn:
            return pd.read_sql(query, conn)
    
    def _merge_coordinate_datasets(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple coordinate datasets."""
        
        if not dfs:
            raise ValueError("No datasets to merge")
        
        # Start with first dataset
        merged = dfs[0]
        
        # Merge others
        for df in dfs[1:]:
            # Round coordinates to avoid floating point issues
            for col in ['x', 'y']:
                merged[col] = merged[col].round(6)
                df[col] = df[col].round(6)
            
            merged = merged.merge(df, on=['x', 'y'], how='outer')
        
        # Sort by coordinates
        merged = merged.sort_values(['y', 'x']).reset_index(drop=True)
        
        return merged
```

#### 2. **Update `/src/pipelines/stages/merge_stage.py`**

```python
# src/pipelines/stages/merge_stage.py
"""Dataset merging stage - orchestration only."""

from typing import List, Tuple
import logging
from pathlib import Path
from .base_stage import PipelineStage, StageResult
from src.processors.data_preparation.coordinate_merger import CoordinateMerger

logger = logging.getLogger(__name__)

class MergeStage(PipelineStage):
    """Orchestrates dataset merging using CoordinateMerger processor."""
    
    @property
    def name(self) -> str:
        return "merge"
    
    @property
    def dependencies(self) -> List[str]:
        return ["resample"]
    
    def execute(self, context) -> StageResult:
        """Orchestrate merge process."""
        logger.info("Starting merge stage orchestration")
        
        try:
            # Get resampled datasets from context
            resampled_datasets = context.get('resampled_datasets', [])
            
            if len(resampled_datasets) < 2:
                return StageResult(
                    success=False,
                    warnings=['Need at least 2 datasets for merging']
                )
            
            # Delegate all work to processor
            merger = CoordinateMerger(context.config, context.db)
            ml_ready_path = merger.create_ml_ready_parquet(
                resampled_datasets,
                context.output_dir
            )
            
            # Update context with results
            context.set('ml_ready_path', str(ml_ready_path))
            
            # Return success metrics
            return StageResult(
                success=True,
                data={
                    'ml_ready_path': str(ml_ready_path),
                    'file_size_mb': ml_ready_path.stat().st_size / (1024**2)
                },
                metrics={
                    'datasets_merged': len(resampled_datasets),
                    'output_format': 'parquet'
                }
            )
            
        except Exception as e:
            logger.error(f"Merge stage failed: {e}")
            return StageResult(
                success=False,
                error=str(e)
            )
```

#### 3. **Update `/src/processors/data_preparation/resampling_processor.py`**

Key changes needed:
```python
# Around line 170 in _store_resampled_dataset()
def _store_resampled_dataset(self, info: ResampledDatasetInfo) -> bool:
    """Store resampled dataset metadata with ACTUAL bounds."""
    
    # ... existing code ...
    
    # CRITICAL FIX: Get actual bounds from raster
    actual_bounds = self._get_actual_raster_bounds(info.source_path)
    
    # Update metadata with actual bounds
    info.metadata['bounds'] = list(actual_bounds)
    info.metadata['actual_shape'] = info.shape
    
    # ... rest of function ...

def _get_actual_raster_bounds(self, raster_path: Path) -> Tuple[float, float, float, float]:
    """Get actual bounds from raster file."""
    import rasterio
    with rasterio.open(raster_path) as src:
        return src.bounds

# Around line 350 in create_passthrough_info()
def create_passthrough_info(self, dataset_name: str, ...) -> ResampledDatasetInfo:
    """Create dataset info for passthrough with ACTUAL bounds."""
    
    # Get actual bounds from raster
    actual_bounds = self._get_actual_raster_bounds(source_path)
    
    return ResampledDatasetInfo(
        name=dataset_name,
        source_path=source_path,
        table_name=f"passthrough_{dataset_name.replace('-', '_')}",
        bounds=actual_bounds,  # Use actual bounds, not default!
        shape=shape,
        target_resolution=resolution,
        actual_resolution=resolution,
        metadata={
            'passthrough': True,
            'bounds': list(actual_bounds),  # Store in metadata too
            'resolution': resolution,
            'shape': shape
        }
    )
```

#### 4. **Create New Processor: `/src/processors/data_preparation/data_loader.py`**

```python
# src/processors/data_preparation/data_loader.py
"""Data loading processor for pipeline."""

from typing import List, Dict
import logging
from pathlib import Path
from src.base.processor import BaseProcessor
from src.domain.raster.catalog import RasterCatalog

logger = logging.getLogger(__name__)

class DataLoader(BaseProcessor):
    """Handles dataset discovery and registration."""
    
    def __init__(self, config, db):
        super().__init__(config=config)
        self.db = db
        self.catalog = RasterCatalog(db, config)
    
    def load_datasets(self, dataset_configs: List[Dict]) -> List[Dict]:
        """Load and validate datasets."""
        loaded_datasets = []
        
        for config in dataset_configs:
            if self._validate_dataset(config):
                # Register in catalog
                self.catalog.add_raster(
                    name=config['name'],
                    path=config['path'],
                    data_type=config.get('data_type', 'richness_data')
                )
                loaded_datasets.append(config)
        
        return loaded_datasets
    
    def _validate_dataset(self, config: Dict) -> bool:
        """Validate dataset configuration and file existence."""
        path = Path(config['path'])
        if not path.exists():
            logger.error(f"Dataset file not found: {path}")
            return False
        
        return True
```

#### 5. **Update `/src/pipelines/stages/load_stage.py`**

```python
# src/pipelines/stages/load_stage.py
"""Data loading stage - orchestration only."""

from typing import List, Tuple
import logging
from .base_stage import PipelineStage, StageResult
from src.processors.data_preparation.data_loader import DataLoader

logger = logging.getLogger(__name__)

class DataLoadStage(PipelineStage):
    """Orchestrates data loading using DataLoader processor."""
    
    @property
    def name(self) -> str:
        return "data_load"
    
    def execute(self, context) -> StageResult:
        """Orchestrate data loading."""
        logger.info("Starting data load orchestration")
        
        try:
            # Get dataset configurations
            dataset_configs = context.config.get('datasets.target_datasets', [])
            enabled_datasets = [d for d in dataset_configs if d.get('enabled', True)]
            
            # Delegate to processor
            loader = DataLoader(context.config, context.db)
            loaded_datasets = loader.load_datasets(enabled_datasets)
            
            # Update context
            context.set('loaded_datasets', loaded_datasets)
            
            return StageResult(
                success=True,
                data={'datasets_loaded': len(loaded_datasets)},
                metrics={'total_datasets': len(loaded_datasets)}
            )
            
        except Exception as e:
            logger.error(f"Load stage failed: {e}")
            return StageResult(success=False, error=str(e))
```

#### 6. **Update `/src/pipelines/stages/resample_stage.py`**

```python
# src/pipelines/stages/resample_stage.py
"""Resample stage - orchestration only."""

def execute(self, context) -> StageResult:
    """Orchestrate resampling - delegate all work to processor."""
    logger.info("Starting resample orchestration")
    
    try:
        loaded_datasets = context.get('loaded_datasets', [])
        processor = ResamplingProcessor(context.config, context.db)
        
        resampled_datasets = []
        for dataset_config in loaded_datasets:
            # Delegate entirely to processor
            resampled_info = processor.resample_dataset(dataset_config)
            if resampled_info:
                resampled_datasets.append({
                    'name': resampled_info.name,
                    'table_name': resampled_info.table_name,
                    'source_path': str(resampled_info.source_path),
                    'bounds': resampled_info.bounds,
                    'resolution': resampled_info.target_resolution,
                    'passthrough': resampled_info.metadata.get('passthrough', False)
                })
        
        context.set('resampled_datasets', resampled_datasets)
        
        return StageResult(
            success=True,
            data={'datasets_resampled': len(resampled_datasets)},
            metrics={'passthrough_count': sum(1 for d in resampled_datasets if d['passthrough'])}
        )
        
    except Exception as e:
        logger.error(f"Resample stage failed: {e}")
        return StageResult(success=False, error=str(e))
```

### C. Testing Strategy

1. **Unit Tests for Coordinate Conversion**:
   ```python
   # tests/test_coordinate_merger.py
   def test_coordinate_conversion():
       # Test with known values
       bounds = (-180, -55, 180, 83)
       resolution = 0.016667
       row_idx, col_idx = 0, 0
       
       x = bounds[0] + (col_idx + 0.5) * resolution
       y = bounds[3] - (row_idx + 0.5) * resolution
       
       assert x == -179.9916665  # Top-left pixel center
       assert y == 82.9916665
   ```

2. **Integration Test**:
   - Create small test rasters with known values
   - Run through pipeline
   - Verify output matches input

### D. Migration Path

1. **Phase 1**: Fix coordinate conversion (immediate)
   - Update resampling_processor to store actual bounds
   - Create CoordinateMerger with correct conversion
   - Test with production data

2. **Phase 2**: Refactor stages (short-term)
   - Move all logic to processors
   - Stages only orchestrate
   - Remove duplicate code

3. **Phase 3**: Add validation (long-term)
   - Add coordinate verification
   - Create data integrity checks
   - Implement proper testing

### E. Configuration Changes

No configuration schema changes needed, but ensure:
- `config.yml` continues to work as-is
- Bounds are read from raster files, not config
- Resolution matching uses actual raster metadata

## 4. Summary

The key issue is using generic bounds `[-180, -90, 180, 90]` instead of actual raster bounds. This must be fixed at multiple levels:

1. **Data storage**: Store actual bounds when creating passthrough tables
2. **Coordinate conversion**: Use actual bounds for row/col to x/y conversion
3. **Architecture**: Move all implementation to processors, stages only orchestrate

This plan provides a clear path from the current broken state to a properly abstracted, working pipeline.