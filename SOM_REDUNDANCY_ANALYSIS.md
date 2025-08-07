# SOM Pipeline Redundancy Analysis

## Overview
This document catalogs unintentional redundancies that have accumulated in the SOM pipeline over time. These appear to be the result of iterative development, refactoring attempts, and parallel implementations rather than deliberate design choices.

## 1. Configuration Redundancy (RESOLVED)

### The Critical Issue: Dual `max_epochs` Settings
**Original Problem:** `/src/config/som/som_config_unified.yml` had max_epochs in TWO places:
- `architecture_config.convergence.max_epochs: 15` (WAS BEING USED)
- `training_config.max_epochs: 100` (WAS IGNORED)

**Root Cause:** analyzer.py line 373 read from wrong location:
```python
# BEFORE (BUG):
max_epochs = kwargs.get('iterations', conv_config['max_epochs'])  # Read from architecture_config.convergence

# AFTER (FIXED):
max_epochs = kwargs.get('iterations', training_config['max_epochs'])  # Read from training_config
```

### Resolution Implemented
1. **Moved ALL training params to `training_config`**:
   - max_epochs, chunk_size, n_workers (training params)
   - initial_learning_rate, min/max_learning_rate, lr factors (learning params)
   - patience, qe_improvement_threshold, convergence criteria
   
2. **Kept ONLY structural params in `architecture_config`**:
   - grid_size, topology, spatial_weight (architecture)
   - neighborhood_function, initial/final_radius (structure)

3. **Fixed analyzer.py** to read from correct locations

4. **Deleted redundant `som_config_loader.py`** - using only `som_config.py` now

## 2. File Redundancy

### Old/Backup Files Still Present
```
src/biodiversity_analysis/methods/som/
├── analyzer.py              # Current
├── analyzer_old.py          # Old backup (redundant)
├── geo_som_core.py          # Current
├── som_core_old.py          # Old backup (redundant)
├── geo_som_core_vectorized.py  # Alternative implementation
├── spatial_utils.py         # Current
└── spatial_utils_optimized.py  # Alternative implementation
```

### Config Loading Files
```
src/config/som/
├── som_config.py            # Main config class
├── som_config_loader.py     # Separate loader (redundant?)
└── som_config_unified.yml   # Actual config file
```

## 3. Function Redundancy

### Distance Calculations (16+ implementations across files)
| File | Count | Functions |
|------|-------|-----------|
| geo_som_core.py | 4 | `_calculate_distances`, `_combined_distance`, etc. |
| spatial_utils.py | 5 | `haversine_distance`, `euclidean_distance`, etc. |
| minibatch_distance_calculator.py | 2 | Distance calculation methods |
| parallel_batch_processor.py | 2 | Distance calculation methods |
| analyzer.py | 1 | Distance calculation |
| geo_som_core_vectorized.py | 1 | Vectorized distance |
| partial_metrics.py | 1 | Distance for metrics |

### Training Methods (RESOLVED)
**Status:** NO CONFLICT - Clean separation of concerns!
- `analyzer.py`: Orchestrates analysis, calls `som.train_batch()`
- `geo_som_core.py`: Contains actual training implementation
- `analyzer_old.py`: DELETED (old backup)
- `som_core_old.py`: DELETED (old backup)

### Preprocessing (Multiple paths)
1. `preprocessing.py::fit_transform()` - Dedicated preprocessing
2. `analyzer.py` - Inline preprocessing
3. Normalization logic duplicated in multiple places

## 4. Parallel Processing Redundancy

Two different parallel processing approaches:
1. **ParallelBatchProcessor** (`parallel_batch_processor.py`)
   - Uses joblib
   - Chunk-based parallelization
   - Currently active

2. **Vectorized Implementation** (`geo_som_core_vectorized.py`)
   - NumPy vectorization approach
   - Alternative implementation
   - Status unclear

## 5. Progress Tracking Redundancy

Three separate progress tracking systems:
1. **JSON-based** (`progress_tracker.py`)
   - Writes to JSON files
   - Used by enhanced monitor

2. **Log-based**
   - Standard Python logging
   - Writes to log files

3. **Callback-based**
   - Progress callbacks in training loop
   - Used for real-time updates

## 6. Metrics Calculation Redundancy

Quantization Error (QE) calculated in:
- `geo_som_core.py::calculate_quantization_error()`
- `partial_metrics.py::calculate_qe()`
- Inline calculations in various places

Topographic Error (TE) calculated in:
- `geo_som_core.py::calculate_topographic_error()`
- `partial_metrics.py`
- Different implementations with different approaches

## 7. Data Loading Redundancy

Multiple data loading paths:
- `analyzer.py` loads data
- `parquet_loader.py` loads data
- Some scripts have their own loading logic

## 8. Checkpoint System Redundancy

Checkpoint saving/loading appears in:
- `geo_som_core.py` - Main checkpoint logic
- `analyzer.py` - Checkpoint management
- Different checkpoint formats/locations

## Impact Assessment

### Critical Issues
1. **Wrong epoch count**: Using 15 instead of 100 epochs
2. **Memory inefficiency**: Multiple implementations loaded
3. **Maintenance nightmare**: Changes need to be made in multiple places

### Performance Impact
- Redundant imports increase memory usage
- Multiple distance calculations slow down training
- Confusing code paths make debugging difficult

### Development Impact
- Hard to know which implementation is canonical
- Risk of fixing bugs in wrong place
- New developers confused by multiple options

## Recommended Actions

### Immediate (High Priority)
1. Fix `max_epochs` configuration conflict
2. Delete `*_old.py` files
3. Consolidate distance calculations

### Short Term
1. Choose between vectorized vs batch processor
2. Unify preprocessing pipeline
3. Single progress tracking system

### Long Term
1. Refactor config structure
2. Create clear module boundaries
3. Document canonical implementations

## Questions to Resolve

1. Which parallel processing approach should be canonical?
2. Should vectorized implementation be kept as alternative?
3. How to consolidate config without breaking existing runs?
4. Which distance calculation implementation is most efficient?

## Files Deleted (Cleanup Complete)
- ✅ `analyzer_old.py` (DELETED)
- ✅ `som_core_old.py` (DELETED)
- ✅ Old monitoring scripts (DELETED)
- ✅ `som/preprocessing.py` (DELETED - using SharedPreprocessor now)
- ✅ `som_config_loader.py` (DELETED - using som_config.py only)

## Files to Keep (Verified Useful)
- ✅ `analyzer_adapter.py` - Used by analyzer_factory.py for pipeline integration
- ✅ `spatial_utils_optimized.py` - Imported by spatial_utils.py for large datasets
- ❓ `geo_som_core_vectorized.py` - Need to decide (alternative implementation)

## Preprocessing Consolidation (COMPLETED)

### What We Did
1. **Created SharedPreprocessor** in `/src/biodiversity_analysis/preprocessing/pipeline.py`
   - Simple, method-aware preprocessor
   - Auto-configures based on method name (som, cpwopt, kmeans)
   - Replaces all method-specific preprocessing

2. **Removed Bloat**:
   - Removed imports of "advanced" features (spatial, zero-inflation, diversity)
   - These were over-engineered and not actually useful:
     - SpatialFeatureEngineering: Moran's I, spatial lags (we already have coordinates!)
     - ZeroInflationPreprocessor: CLR, ILR transforms (over-engineering for simple counts)
     - BiodiversityFeatureEngineering: Shannon/Simpson indices (circular reasoning!)

3. **Updated SOM**:
   - Removed all preprocessing logic from `analyzer.py`
   - Now uses `SharedPreprocessor(method_name='som')`
   - Deleted `som/preprocessing.py` entirely

### Benefits
- Single preprocessing module for all methods
- No redundancy
- Simple configuration (minmax for SOM, standard for kmeans, etc.)
- Much cleaner code

---
*Generated: 2025-08-07*
*Branch: fix/som-redundancy-cleanup*