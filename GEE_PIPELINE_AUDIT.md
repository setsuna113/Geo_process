# GEE Pipeline Comprehensive Audit Report

## ✅ FIXED: ~~Multiple Orchestrators~~ 

### ~~Orchestrator Redundancy~~ RESOLVED
~~We have **TWO separate orchestrator implementations**:~~

1. **`orchestrator.py`** (39KB, 857 lines) - **KEPT**
   - Full-featured orchestrator with logging context
   - Uses infrastructure.logging
   - Complex state management
   - Handles GeoTIFF exports (CSV support removed)

2. ~~**`simple_orchestrator.py`** (10KB, 245 lines)~~ - **DELETED**
   - ~~"Simplified" version without logging contexts~~
   - ~~Duplicates most of orchestrator.py logic~~
   - ~~Created to avoid logging issues~~
   - ~~Nearly identical workflow steps~~

**Duplication Eliminated: 245 lines removed**

### Detailed Comparison

| Feature | orchestrator.py | simple_orchestrator.py | Duplication |
|---------|----------------|----------------------|-------------|
| Authentication | Lines 175-196 | Lines 64-78 | ✅ Duplicate |
| Asset Upload | Lines 198-254 | Lines 80-125 | ✅ Duplicate |
| Processing Logic | Lines 256-438 | Lines 127-195 | ✅ Duplicate |
| Export Logic | Lines 440-738 | Lines 197-240 | ✅ Duplicate |
| State Management | Full WorkflowState | Same WorkflowState | ✅ Same |
| Logging | LoggingContext | Basic logger | Different |

## 📊 Complete Hardcoded Parameters Inventory

### 1. Authentication & Credentials
```python
# Hardcoded across multiple files:
SERVICE_ACCOUNT_KEY = '.gee/service-account-key.json'  # 5 occurrences
PROJECT_ID = 'triple-carrier-441622-t1'  # 3 occurrences
EARTHENGINE_CLI = '/home/jason/anaconda3/envs/geo/bin/earthengine'  # 12 occurrences
ASSET_PREFIX = 'projects/earthengine-legacy/assets/users/geebiodiversity'  # 4 occurrences
```

### 2. Geographic & Scale Constants
```python
# Inconsistent across files:
BASE_RESOLUTION = 0.016666666666667  # 8 occurrences (config.py, orchestrators, resampler)
METERS_PER_DEGREE = 111320  # 6 occurrences
METERS_PER_DEGREE_ALT = 111319.5  # 2 occurrences (export_manager.py)

# Scale thresholds:
COARSE_MASK_THRESHOLD = 5000  # merger.py:238, 277
MAX_RESOLUTION_FACTOR = 16  # recursive_resampler.py:64
DOWNSAMPLE_FACTOR = 2  # recursive_resampler.py (implicit in logic)
```

### 3. Cloud Storage & Paths
```python
GCS_BUCKET = 'biodiv-gee-yl998'  # 8 occurrences
STATE_DIR = '~/.biodiversity'  # 3 occurrences
TEMP_DIR = '/tmp/biodiversity'  # config.py:111
PARQUET_DIR = 'data/merged'  # config.py:113
EXPORT_FOLDER = 'merged_outputs'  # config.py:99
```

### 4. Processing Parameters
```python
# Scattered across files:
MAX_PIXELS = 1e13  # 6 occurrences
TILE_SCALE = 16  # 3 occurrences
CHUNK_SIZE = 100 * 1024 * 1024  # config.py:89 (100MB)
PARQUET_CHUNK_SIZE = 100000  # config.py:115
PARALLEL_UPLOADS = 4  # config.py:90

# Ocean mask parameters:
WATER_MASK_ASSET = "MODIS/006/MOD44W/2015_01_01"  # merger.py:230
WATER_VALUE = 1  # merger.py:254
LAND_VALUE = 0  # merger.py:241
MAX_PIXELS_REDUCE = 1024  # merger.py:245, 281
```

### 5. Timeouts & Intervals
```python
EXPORT_TIMEOUT = 7200  # config.py:101 (2 hours)
TASK_TIMEOUT = 3600  # asset_uploader.py:243 (1 hour)
CHECK_INTERVAL = 30  # config.py:102
RETRY_DELAY = 2.0  # authenticator.py:47
GSUTIL_RETRY_DELAY = 5.0  # fetcher.py:33
TASK_INIT_DELAY = 10  # orchestrator.py:627
```

### 6. Dataset Definitions
```python
# Hardcoded in simple_orchestrator.py:95-102
BIODIVERSITY_DATASETS = [
    ('animals', 'animal_richness', 'continuous', 'max'),
    ('plants', 'plant_richness', 'continuous', 'max'),
    ('am_fungi', 'am_fungi_richness', 'continuous', 'max'),
    ('am_fungi_cv', 'am_fungi_cv', 'cv', 'mean'),
    ('ecm_fungi', 'ecm_fungi_richness', 'continuous', 'max'),
    ('ecm_fungi_cv', 'ecm_fungi_cv', 'cv', 'mean')
]

# Climate bands hardcoded in band_config.py:85-90
CLIMATE_BANDS = ['bio01', 'bio04', 'bio12', 'bio15']
```

### 7. Geographic Bounds
```python
# Default latitude bounds (repeated in multiple files):
LAT_MIN = -60  # config.py:57, merger.py:144
LAT_MAX = 75   # config.py:57, merger.py:144
LON_MIN = -180  # export_manager.py, orchestrator.py
LON_MAX = 180   # export_manager.py, orchestrator.py
```

## 🔄 Additional Code Redundancy Patterns

### 1. Resolution Processing (3 implementations)
- **recursive_resampler.py**: Full recursive implementation
- **resampler.py**: Alternative resampling logic
- **Both orchestrators**: Duplicate resolution handling

### 2. Export Management (Multiple approaches)
- **export_manager.py**: `_export_as_geotiff()` and `_export_as_csv()`
- **orchestrator.py**: Lines 600-656 - inline export logic
- **simple_orchestrator.py**: Lines 197-240 - simplified export

### 3. State Management (Redundant usage)
- **state_manager.py**: WorkflowState class
- **orchestrator.py**: Complex state tracking
- **simple_orchestrator.py**: Same state tracking (duplicate)

### 4. Asset Management
- **batch_manager.py**: BatchUploadManager
- **asset_uploader.py**: AssetUploader
- Both handle similar upload logic with different approaches

### 5. Task Monitoring (4 implementations!)
- **task_monitor.py**: GEETaskMonitor class
- **export_manager.py**: Lines 315-385 - inline monitoring
- **orchestrator.py**: Lines 624-679 - inline monitoring
- **asset_uploader.py**: Lines 240-280 - inline monitoring

### 6. GCS Operations (Scattered)
- **fetcher.py**: GCS download operations
- **export_manager.py**: GCS path construction
- **batch_manager.py**: GCS upload operations
- All with similar gsutil subprocess calls

### 7. Band Ordering Logic (Multiple definitions)
- **merger.py**: Band ordering in merge operation
- **band_config.py**: Band configuration
- **orchestrator.py**: Lines 740-755 - band name building
- **converter.py**: Band expectations

## 📈 Metrics Summary

### Hardcoded Parameters
- **Total hardcoded values**: 75+ instances
- **Repeated constants**: 30+ (same value in multiple places)
- **System-specific paths**: 15+ instances
- **Magic numbers**: 25+ instances

### Code Duplication
- **Orchestrator duplication**: ~500 lines
- **Task monitoring**: ~400 lines (4 implementations)
- **Export logic**: ~300 lines (3 implementations)
- **Authentication**: ~200 lines (3 implementations)
- **File processing**: ~400 lines (4 implementations)
- **State management**: ~150 lines (2 implementations)

### Total Redundant Code: **~2000 lines** (approximately 25% of the codebase)

## 🎯 Priority Refactoring Targets

### Critical (Breaks on environment change)
1. **Merge the two orchestrators** - eliminate simple_orchestrator.py
2. **Extract ALL hardcoded paths** to environment variables
3. **Fix meters-per-degree inconsistency** (111320 vs 111319.5)

### High Priority (Major redundancy)
1. **Create single TaskMonitor base class** - eliminate 4 implementations
2. **Consolidate export logic** into ExportManager
3. **Unify authentication** into single module

### Medium Priority
1. **Create constants.py** for all magic numbers
2. **Build shared GCS utilities** module
3. **Standardize band configuration** usage

### Low Priority
1. **Merge resampler implementations**
2. **Consolidate state management** patterns
3. **Unify file processing** utilities

## 🔴 Most Problematic Files

1. **orchestrator.py** - 857 lines, heavy duplication with simple version
2. **simple_orchestrator.py** - 245 lines, should not exist
3. **export_manager.py** - Inline monitoring, hardcoded values
4. **asset_uploader.py** - Duplicate auth, monitoring, hardcoded CLI paths
5. **merger.py** - Hardcoded thresholds, band ordering issues

## 💡 Recommendations

### Immediate Actions
1. **DELETE simple_orchestrator.py** after fixing logging issues in main orchestrator
2. **Create .env file** for all system-specific paths
3. **Create constants.py** with all numeric constants

### Refactoring Strategy
1. **Week 1**: Merge orchestrators, extract configurations
2. **Week 2**: Create base classes for common patterns
3. **Week 3**: Consolidate duplicate implementations
4. **Week 4**: Testing and validation

### Expected Benefits
- **Code reduction**: ~2000 lines (25% less code)
- **Maintenance**: Single point of change for each feature
- **Reliability**: Consistent behavior across modules
- **Portability**: Environment-agnostic configuration

---
*Generated: 2025-01-07*
*Total Files Analyzed: 25*
*Total Lines of Code: ~8000*
*Redundancy Factor: 25%*