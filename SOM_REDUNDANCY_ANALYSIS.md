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

## 2. File Redundancy (RESOLVED ✅)

### Old/Backup Files Were Present:
```
src/biodiversity_analysis/methods/som/
├── analyzer.py              # Current
├── analyzer_old.py          # DELETED ✅
├── geo_som_core.py          # Current  
├── som_core_old.py          # DELETED ✅
├── geo_som_core_vectorized.py  # DELETED ✅
├── spatial_utils.py         # Current
└── spatial_utils_optimized.py  # KEPT - Actually used for large datasets!
```

**Status: All old backup files have been deleted!**
- `analyzer_old.py` - DELETED
- `som_core_old.py` - DELETED  
- `geo_som_core_vectorized.py` - DELETED
- `spatial_utils_optimized.py` - KEPT (imported by spatial_utils.py for n_samples > 50,000)

### Config Loading Files (RESOLVED ✅)
```
src/config/som/
├── som_config.py            # Main config class (KEPT)
├── som_config_loader.py     # DELETED ✅
└── som_config_unified.yml   # Actual config file (KEPT)
```

**Status: Redundant config loader deleted!**
- Using only `som_config.py` for loading configuration
- Clean separation: Python class + YAML config file

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

## 5. Progress Tracking Redundancy (INVESTIGATED)

### Three Interconnected Progress Systems Found:

1. **JSON-based** (`progress_tracker.py` - 122 lines)
   - **Purpose**: External monitoring (e.g., by som_progress_monitor.py)
   - **Location**: Writes to `outputs/analysis_results/som/som_progress_*.json`
   - **Updates**: Every epoch with QE, LR, radius, time estimates
   - **Used by**: `analyzer.py` line 188 creates tracker
   - **Atomic writes**: Uses temp file + rename for safety

2. **Log-based** (Standard Python logging)
   - **Purpose**: Debugging and audit trail
   - **Location**: Log files via Python logging config
   - **What's logged**: 20+ different progress messages in geo_som_core.py
   - **Examples**: "Starting epoch X/Y", "QE calculated", "Converged at epoch"
   - **Cannot be removed**: Essential for debugging

3. **Callback-based** (Function callbacks)
   - **Purpose**: Connects the training loop to external trackers
   - **How it works**: 
     - geo_som_core.py calls `progress_callback(progress_info)` at line 641
     - This callback is created by `create_progress_callback()` at line 96
     - Which updates the JSON tracker
   - **Flow**: Training → Callback → JSON tracker → File

### How They Work Together:

```
geo_som_core.py (training)
    ↓
    ├── logger.info() → Log files (for debugging)
    ↓
    └── progress_callback() → SOMProgressTracker → JSON file (for monitoring)
```

### The Real Issue:

**These aren't actually redundant - they serve different purposes!**

1. **Logs**: For debugging/audit - MUST KEEP
2. **JSON**: For external monitoring tools - USEFUL
3. **Callbacks**: The glue between training and JSON tracker - NECESSARY

### However, there IS redundancy:

1. **Duplicate information**: Same epoch/QE info goes to both logs AND JSON
2. **Multiple update points**: Progress updated in 3+ places for same event
3. **Inconsistent data**: JSON has more fields than actually used

### Recommendation:

**Keep all three but streamline:**

1. **Logs**: Keep for debugging (no change)
2. **JSON tracker**: Simplify - remove unused fields
3. **Callbacks**: Keep as the connection mechanism

The "redundancy" is more about **duplicate information** than duplicate systems. Each serves a valid purpose!


## 8. Checkpoint System Redundancy (INVESTIGATED)

### Multiple Checkpoint Systems Found:

1. **SOM-specific checkpoint system (ACTUALLY USED):**
   - `geo_som_core.py::save_checkpoint()` (line 1449) - Saves to .npz files
   - `geo_som_core.py::load_checkpoint()` (line 1498) - Loads from .npz files
   - Location: `checkpoints/som/som_checkpoint_epoch_*.npz`
   - Features: Saves weights, learning rate, radius, training history
   - Auto-cleanup: Keeps only last 3 checkpoints
   - Creates symlink: `som_checkpoint_latest.npz`

2. **Analyzer checkpoint interface (PARTIALLY USED):**
   - `analyzer.py::save_checkpoint_state()` (line 857) - Returns dict
   - `analyzer.py::load_checkpoint_state()` (line 884) - Loads from dict
   - Used for: Cross-validation state
   - NOT actually saving to disk - just returns state dict!

3. **Base analyzer stub methods (NOT USED):**
   - `base_analyzer.py::save_checkpoint_state()` - Empty stub
   - `base_analyzer.py::load_checkpoint_state()` - Empty stub
   - Just placeholders, not implemented

4. **Unified checkpoint system (EXISTS BUT NOT USED BY SOM):**
   - `analysis_checkpoint.py::SimpleAnalysisCheckpointer` - File-based system
   - `AnalysisCheckpoint` dataclass - Unified format
   - Supports method_state storage
   - **SOM doesn't use this at all!**

### How Checkpointing Actually Works in SOM:

1. **During training (geo_som_core.py):**
   - Every N epochs (checkpoint_frequency=10)
   - Calls `save_checkpoint()` directly
   - Saves to `checkpoints/som/` directory
   - Creates .npz files with numpy arrays

2. **Resume logic (analyzer.py + geo_som_core.py):**
   - analyzer.py checks for `som_checkpoint_latest.npz`
   - Sets `resume_checkpoint` in config
   - geo_som_core.py loads it if exists

3. **The analyzer methods are misleading:**
   - `save_checkpoint_state()` doesn't save to disk!
   - It just returns a dict (for potential external saving)
   - But nothing calls it to actually save!

### Issues:

1. **Confusing dual system** - analyzer methods suggest checkpointing but don't actually save
2. **Unified system ignored** - SimpleAnalysisCheckpointer exists but SOM doesn't use it
3. **No integration** - Each method has its own checkpoint format
4. **Misleading method names** - `save_checkpoint_state()` doesn't save anything

### Analysis: Are SOM checkpoints useful for other methods?

**NO - Each method needs completely different checkpoint data:**
- **SOM**: weights, learning_rate, radius, grid_size (topology-specific)
- **CP-WOPT**: factor matrices, tensor shapes (completely different structure)
- **KMeans**: cluster centers, labels, n_clusters (centroid-based)

The checkpoint data structures are fundamentally incompatible!

### Resolution (COMPLETED ✅):

**Keep checkpointing method-specific, removed misleading methods:**

1. **✅ Removed misleading methods from analyzer.py:**
   - Deleted `save_checkpoint_state()` - was just returning dict, not saving
   - Deleted `load_checkpoint_state()` - was never called
   - Added clear comments explaining checkpoint is in geo_som_core.py

2. **Decision: Keep SOM checkpoint in SOM:**
   - The .npz format is SOM-specific (weights, LR, radius)
   - Not useful for CP-WOPT or KMeans
   - Each method should handle its own checkpointing

3. **Benefits of this approach:**
   - Simple and clear - no forced abstraction
   - Each method optimizes its checkpoint format
   - No confusion about what's being saved

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
- ✅ `geo_som_core_vectorized.py` (DELETED - alternative implementation, already integrated useful parts)
- ✅ Old monitoring scripts (DELETED)
- ✅ `som/preprocessing.py` (DELETED - using SharedPreprocessor now)
- ✅ `som_config_loader.py` (DELETED - using som_config.py only)

## Files to Keep (Verified Useful)
- ✅ `analyzer_adapter.py` - Used by analyzer_factory.py for pipeline integration
- ✅ `spatial_utils_optimized.py` - Imported by spatial_utils.py for large datasets

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

## QE/TE Calculation Verification (NO REDUNDANCY ✅)

**Double-checked as requested - Results:**

Quantization Error (QE):
- ✅ ONLY defined in `geo_som_core.py::calculate_quantization_error()`
- `analyzer.py` just calls `som.calculate_quantization_error()` (no duplicate)

Topographic Error (TE):
- ✅ ONLY defined in `geo_som_core.py::calculate_topographic_error()`
- No duplicate implementations found anywhere

Note: `partial_metrics.py` only contains distance functions (Bray-Curtis, Euclidean), NOT QE/TE calculations.

**Conclusion:** QE and TE are calculated in exactly ONE place each - no redundancy!

## 7. Data Loading Redundancy (INVESTIGATED)

### Multiple Data Loading Implementations Found:

1. **Primary ParquetLoader path (CANONICAL):**
   - `/src/biodiversity_analysis/shared/data/parquet_loader.py` - Main loader with spatial grid sampling
   - Used by: BaseBiodiversityAnalyzer → SOM, KMeans analyzers
   - Features: Spatial grid sampling, coordinate detection, PyArrow optimization
   - **This is the GOOD path - sophisticated with memory-efficient sampling**

2. **Direct pd.read_parquet() calls (19 instances):**
   - `coordinate_merger.py:770` - Verification read after saving
   - `standalone_analysis_datasets.py:150` - Direct loading in _load_data()
   - `standalone_analysis_datasets.py:279` - Direct loading
   - `gee_integration/download/converter.py` (3 instances) - GEE data conversion
   - `machine_learning/pipelines/regression_pipeline.py:105` - ML pipeline loading
   - `scripts/run_ml.py` (2 instances) - ML script loading
   - `security_utils.py` (2 instances) - Security validation
   - `mesf_preprocessor.py:140` - Coordinate loading
   - `memory_utils.py:258` - Chunked reading
   - `parquet_converter.py:289` - Climate data conversion
   - `inspect_parquet.py:14` - Inspection tool
   - `ml_compare_models.py:279` - Model comparison
   - `ml_predict.py:93` - Prediction loading
   - `ml_train.py:72` - Training data loading

3. **Dataset abstraction layers:**
   - `ParquetAnalysisDataset` - Uses PyArrow for streaming
   - `DatabaseAnalysisDataset` - Database loading path
   - `StandaloneParquetDataset` - Uses pd.read_parquet directly

4. **Adapter patterns:**
   - `cpwopt/analyzer_adapter.py:239` - Direct pd.read_parquet for CP-WOPT

### Which Paths Are Actually Used:

**For Biodiversity Analysis (SOM, KMeans):**
- ✅ Uses ParquetLoader via BaseBiodiversityAnalyzer.load_data()
- This is the CORRECT path with spatial sampling capabilities

**For CP-WOPT:**
- ⚠️ Uses direct pd.read_parquet() in analyzer_adapter
- Missing spatial sampling features!

**For ML Pipeline:**
- ⚠️ Uses direct pd.read_parquet() in regression_pipeline.py
- No spatial sampling, basic loading only

**For Pipeline Stages:**
- Uses Dataset abstractions (ParquetAnalysisDataset, etc.)
- Different from ParquetLoader, creates another layer

### Key Issues:

1. **CP-WOPT doesn't use ParquetLoader** - Missing spatial grid sampling feature
2. **ML pipeline has its own loading** - Duplicates logic, no sampling
3. **Dataset abstractions vs ParquetLoader** - Two parallel systems
4. **19 direct pd.read_parquet calls** - No error handling, no sampling

### Recommendation:

**CONSOLIDATE to ParquetLoader as the single source of truth:**

1. **Immediate fixes (COMPLETED ✅):**
   - ✅ Updated CP-WOPT analyzer_adapter to use ParquetLoader
   - ✅ Added data_processing config section to cpwopt_config.yml
   - ✅ CP-WOPT now has spatial grid sampling capability (spatial_grid_step: 10)

2. **Direct pd.read_parquet in biodiversity_analysis/ (INVESTIGATED ✅):**
   - Only 2 instances found:
     - `cpwopt/analyzer_adapter.py:239` - **FIXED** to use ParquetLoader
     - `kmeans/analyzer.py:661` - Validation only, acceptable to keep

3. **ML Pipeline:**
   - Ignoring as requested - switching to H2O

4. **Long term:**
   - Decide between Dataset abstractions vs ParquetLoader
   - Probably keep ParquetLoader for its sophisticated sampling

---
*Generated: 2025-08-07*
*Branch: fix/som-redundancy-cleanup*