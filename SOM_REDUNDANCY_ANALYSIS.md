# SOM Pipeline Redundancy Analysis

## Overview
This document catalogs unintentional redundancies that have accumulated in the SOM pipeline over time. These appear to be the result of iterative development, refactoring attempts, and parallel implementations rather than deliberate design choices.

## 1. Configuration Redundancy

### The Critical Issue: Dual `max_epochs` Settings
**Location:** `/src/config/som/som_config_unified.yml`

```yaml
# Line 74 - ACTUALLY USED
architecture_config:
  convergence:
    max_epochs: 15  # This value is being used!

# Line 84 - IGNORED  
training_config:
  max_epochs: 100  # This value is ignored!
```

**Impact:** Production runs use 15 epochs instead of intended 100 epochs.

### Other Config Duplications
- Learning rate settings appear in multiple sections
- Convergence criteria split between `convergence` and `early_stopping` sections
- Memory settings scattered across `memory_config` and `training_config`

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

### Training Methods (3 separate implementations)
1. `geo_som_core.py::train()` - Current implementation
2. `analyzer_old.py::train()` - Old version
3. `som_core_old.py::train()` - Another old version

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

## Files to Delete (Confirmed Redundant)
- `analyzer_old.py`
- `som_core_old.py`
- Old monitoring scripts (already deleted)

## Files to Investigate Further
- `analyzer_adapter.py` - Check if still needed
- `geo_som_core_vectorized.py` - Performance comparison needed
- `spatial_utils_optimized.py` - Check which is actually used
- `som_config_loader.py` - Might be redundant with `som_config.py`

---
*Generated: 2025-08-07*
*Branch: fix/som-redundancy-cleanup*