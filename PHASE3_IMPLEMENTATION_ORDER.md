# Phase 3 Implementation Order: Building a Robust System

## ✅ Already Completed:

### Phase 1: Foundation Cleanup
1. Renamed foundations/ → abstractions/ for clarity
2. Created proper type organization in abstractions/types/
3. Removed ALL duplicate type definitions from base/
4. Fixed all type imports to use abstractions/types/
5. Migrated analyzers to use base/analyzer.py
6. Removed redundant analyzer files (base_analyzer.py, enhanced_analyzer.py, base_analyzer_compat.py)

### Phase 2: Architectural Violations Fixed
1. Fixed base/grid.py - removed database imports, now uses dependency injection
2. Fixed base/processor.py - removed imports from core/, checkpoints/
3. All base/ layer files now follow proper dependency hierarchy

### Phase 3: Domain Layer Restructured
1. Moved src/grid_systems/ → src/domain/grid_systems/
2. Moved src/resampling/ → src/domain/resampling/
3. Updated all imports throughout codebase to use new domain paths
4. Fixed relative imports in domain modules (.. → ...)

### Phase 4: Data Integrity Framework Created
1. Created src/domain/validators/ directory structure
2. Implemented base validator interface in abstractions/interfaces/validator.py
3. Created validation types in abstractions/types/validation_types.py
4. Implemented three coordinate integrity validators:
   - BoundsConsistencyValidator: Validates geographic bounds
   - CoordinateTransformValidator: Validates CRS transformations
   - ParquetValueValidator: Validates data values and ranges

## 📋 Implementation Order (Chronological):

### ✅ Step 1: Fix Architectural Violations (Foundation) - COMPLETED
**Why first**: Can't build robust features on broken dependencies

1. ✅ Fix base/grid.py - remove database imports, use dependency injection
2. ✅ Fix base/processor.py - remove imports from core/, checkpoints/
3. ✅ Review all base/ files for upward dependencies
4. ✅ Implement proper dependency injection pattern

### ✅ Step 2: Complete Domain Layer Structure - COMPLETED
**Why second**: Establishes proper architectural boundaries before adding features

1. ✅ Move src/grid_systems/ → src/domain/grid_systems/
2. ✅ Move src/resampling/ → src/domain/resampling/
3. ✅ Update all imports throughout codebase
4. ✅ Verify no broken imports or circular dependencies

### ✅ Step 3: Create Data Integrity Framework - COMPLETED
**Why third**: Now we have clean architecture to build validation on

1. ✅ Create src/domain/validators/ directory
2. ✅ Implement coordinate_integrity.py with:
   - BoundsConsistencyValidator
   - CoordinateTransformValidator
   - ParquetValueValidator
3. ✅ Create base validator interface in abstractions/interfaces/validator.py
4. ✅ Implement validation error types in abstractions/types/validation_types.py

### Step 4: Integrate Validation into Processing Pipeline - NEXT
**Why fourth**: Apply validation to fix the actual data integrity issue

1. Update CoordinateMerger to use validators
2. Add validation to ResamplingProcessor
3. Update merge_stage.py to handle validation errors
4. Add validation monitoring to pipeline orchestrator

### Step 5: Update Higher-Level Components
**Why last**: Ensure changes propagate correctly to pipeline layer

1. Update pipeline stages to handle new domain structure
2. Update monitors to track validation metrics
3. Update orchestrator for validation checkpoints
4. Add validation results to experiment tracking

## Key Principles:
1. **Bottom-up approach**: Fix foundation before building features
2. **Clean architecture**: Each layer only depends on layers below
3. **Incremental validation**: Test after each step
4. **No breaking changes**: Maintain compatibility during migration

## Success Metrics:
- Zero architectural violations (no upward dependencies)
- All imports follow hierarchy: abstractions → base → domain → processors → pipelines
- Data integrity validation catches coordinate/bounds mismatches
- All existing tests continue to pass
- Pipeline correctly reports validation failures

This order ensures we build a robust system by:
1. First fixing the foundation (architectural violations)
2. Then organizing the structure (domain modules)
3. Finally adding the features (validation) on solid ground