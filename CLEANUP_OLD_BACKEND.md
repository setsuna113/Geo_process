# Cleanup Plan: Remove Old Joblib Backend

## Current Architecture Chain
```
analyzer.py 
  → GeoSOMVLRSOM (geo_som_core.py)
    → ParallelBatchProcessor (parallel_batch_processor.py)
      → joblib.Parallel with process_single_chunk
        → OptimizedDistanceCalculator (13% Numba, 87% NumPy)
```

## Files to Modify/Remove

### 1. **parallel_batch_processor.py** ❌ REMOVE
- Contains all joblib logic
- 466 lines of complex chunking code
- Already deprecated with warning

### 2. **geo_som_core.py** ✏️ MODIFY
- Line 29: Remove `ParallelBatchProcessor` import
- Line 205: Replace with `PureNumbaSOM` initialization
- Line 700: Replace `process_chunks_parallel` with `train_epoch`
- Remove all chunking logic

### 3. **analyzer.py** ✏️ MODIFY  
- Update to use modified GeoSOMVLRSOM
- Or create new analyzer that directly uses PureNumbaSOM

### 4. **optimized_distance_calculator.py** ⚠️ PARTIAL REMOVE
- Remove chunking methods
- Keep basic distance calculations for other uses
- Remove mini-batch logic that's not needed

### 5. **__init__.py files** ✏️ UPDATE
- Remove ParallelBatchProcessor exports
- Add PureNumbaSOM exports

## Dependencies to Check

```bash
# Files importing parallel_batch_processor
src/biodiversity_analysis/methods/som/core/geo_som_core.py
src/biodiversity_analysis/methods/som/trainers/__init__.py

# Files importing geo_som_core  
src/biodiversity_analysis/methods/som/analyzer.py
src/biodiversity_analysis/methods/som/__init__.py
```

## Proposed New Architecture
```
analyzer.py
  → GeoSOMPure (new simplified core)
    → PureNumbaSOM.train_epoch()
      → 100% Numba parallel processing
```

## Benefits of Full Cleanup
1. **Remove 1000+ lines of unnecessary code**
2. **Eliminate entire joblib dependency**
3. **Simplify configuration (no chunk_size, n_workers, pre_dispatch, etc.)**
4. **Direct data → Numba flow**
5. **No serialization, no IPC, no memory copies**

## Migration Steps

### Option A: Conservative (Keep compatibility)
1. Create `geo_som_core_pure.py` alongside old one
2. Update it to use PureNumbaSOM
3. Create `analyzer_pure.py` that uses new core
4. Test thoroughly
5. Switch production to new version
6. Remove old files after stable

### Option B: Direct Replacement (Breaking change)
1. Backup current files
2. Modify geo_som_core.py to use PureNumbaSOM
3. Remove all chunking/joblib code
4. Update analyzer.py
5. Test with production data
6. Delete parallel_batch_processor.py

## Code Reduction Estimate
- `parallel_batch_processor.py`: -466 lines
- `geo_som_core.py` chunking logic: ~-200 lines  
- Config parameters: ~-50 lines
- **Total: ~700+ lines removed!**

## Next Action
Should we go with Option A (conservative) or Option B (direct replacement)?