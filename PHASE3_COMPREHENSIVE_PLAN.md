# Phase 3 Comprehensive Plan: Architectural Refinement and Data Integrity

## Current State Assessment

### ✅ Completed:
1. **abstractions/** - Pure interfaces, types, and mixins (no src imports)
2. **base/** - Contains implementations of abstractions
3. **Analyzers migrated** - Now use base/analyzer.py and proper imports
4. **Type migration started** - Created abstractions/types/ with proper organization
5. **infrastructure/** - Removed (was redundant)

### 🚧 In Progress:
1. **Type migration incomplete** - Still have duplicate type definitions in base/
2. **Domain layer incomplete** - Missing grid_systems and resampling domains

### ❌ Issues Remaining:
1. **base/ architectural violations** - Imports from higher layers (core/, checkpoints/, database/)
2. **Scattered domain logic** - grid_systems/ and resampling/ at root level
3. **Data integrity concerns** - No validation framework for coordinate conversions
4. **Import inconsistencies** - Some modules still use old patterns

## Implementation Plan

### Step 1: Complete Type Migration (High Priority)

Remove duplicate type definitions and update imports:

```bash
# Files to update:
- base/processor.py - Remove ProcessorConfig, ProcessingResult
- base/feature.py - Remove SourceType, FeatureResult  
- base/resampler.py - Remove ResamplingMethod, AggregationMethod, ResamplingConfidence
- base/memory_manager.py - Remove MemoryPressureLevel, MemoryAllocation
- base/services/memory_tracker_service.py - Remove duplicate types
- base/tile_processor.py - Remove ProcessingStatus, TileProgress
```

### Step 2: Fix Architectural Violations in base/ (High Priority)

Current violations to fix:
```python
# base/grid.py imports from:
- from src.config import config  # OK - config is foundational
- from src.database import schema  # VIOLATION - base shouldn't know about database

# base/processor.py imports from:
- from src.core import registry  # VIOLATION - base shouldn't import from core
- from src.checkpoints import manager  # VIOLATION - base shouldn't import from checkpoints
```

Solution: Use dependency injection instead of direct imports.

### Step 3: Complete Domain Layer (Medium Priority)

Move domain logic to proper location:
```bash
# Move existing modules
mv src/grid_systems/ src/domain/grid_systems/
mv src/resampling/ src/domain/resampling/

# Update imports throughout codebase
```

### Step 4: Implement Data Integrity Validation (High Priority)

Create validation framework for the critical data integrity issue:

```python
# src/domain/validators/coordinate_integrity.py
class CoordinateIntegrityValidator:
    """Validates coordinate transformations and bounds consistency."""
    
    def validate_bounds_consistency(self, stored_bounds, actual_bounds):
        """Ensure stored bounds match actual raster bounds."""
        pass
    
    def validate_coordinate_conversion(self, row_col, xy_coords, transform):
        """Verify row/col to x/y conversion is correct."""
        pass
    
    def validate_parquet_values(self, source_raster, output_parquet):
        """Ensure values in parquet match original raster."""
        pass

# Integrate into CoordinateMerger
class ValidatedCoordinateMerger(CoordinateMerger):
    def __init__(self, config, db):
        super().__init__(config, db)
        self.validator = CoordinateIntegrityValidator()
```

### Step 5: Clean Import Hierarchy (Medium Priority)

Establish clear import rules:
```
abstractions/ → no imports from src.*
base/ → imports only from abstractions/
domain/ → imports from abstractions/, base/
processors/ → imports from abstractions/, base/, domain/
pipelines/ → imports from processors/ (and transitively lower layers)
```

### Step 6: Create Compatibility Layer (Low Priority)

For smooth migration, create temporary compatibility imports:
```python
# src/base/compat.py
# Temporary compatibility layer - will be removed in future
from src.abstractions.types.processing_types import ProcessingResult
from src.abstractions.types.memory_types import MemoryPressureLevel
# ... other common imports

# This allows existing code to work while we migrate
```

## Implementation Order

### Phase 3.1 (Immediate - Data Integrity)
1. Create coordinate integrity validator
2. Integrate validation into CoordinateMerger
3. Add validation tests with known test cases

### Phase 3.2 (Short Term - Type Cleanup)
1. Complete type migration (remove duplicates from base/)
2. Update all imports to use abstractions/types/
3. Run tests to ensure nothing breaks

### Phase 3.3 (Medium Term - Architecture)
1. Fix base/ layer violations using dependency injection
2. Move grid_systems/ and resampling/ to domain/
3. Update import paths

### Phase 3.4 (Long Term - Refinement)
1. Add comprehensive validation throughout pipeline
2. Create architecture validation tests
3. Document import hierarchy rules

## Success Criteria

1. **No duplicate type definitions** - Single source of truth in abstractions/types/
2. **Clean architecture** - No upward dependencies (base → core violations)
3. **Data integrity assured** - Validation catches coordinate/bounds mismatches
4. **All tests pass** - No functionality regression
5. **Clear import hierarchy** - Documented and enforced

## Notes on Current Structure

### What's Working Well:
- abstractions/ provides clean interfaces
- base/ provides reusable implementations
- Analyzer migration shows the pattern works
- Type organization in abstractions/types/ is clean

### Key Principles:
- **Don't break working code** - Use compatibility layers during migration
- **Focus on real problems** - Data integrity is the priority
- **Incremental migration** - Small, testable changes
- **Maintain functionality** - All tests must continue passing

This plan respects the current system while addressing critical issues, particularly the data integrity concerns that motivated this refactoring.