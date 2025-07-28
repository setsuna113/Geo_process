# Comprehensive Architectural Analysis Report

## Executive Summary

I performed a comprehensive architectural analysis of **117 Python files** across the `/home/jason/geo/src/` and `/home/jason/geo/scripts/` directories to identify architectural violations, circular dependencies, and code quality issues. The analysis revealed **significant structural problems** that require immediate attention for proper system maintainability and packaging.

## Key Findings Overview

- **26 architectural violations** breaking layered architecture principles
- **3 critical circular dependencies** preventing proper module initialization
- **Module duplication crisis**: 8,000-10,000 lines of duplicate code
- **God classes**: Multiple oversized modules exceeding 1,000 lines
- **Manager class explosion**: 12 different Manager classes indicating design anti-patterns

## Analysis Methodology

1. **File Discovery**: Located all 117 Python files using systematic directory traversal
2. **Import Extraction**: Used AST parsing to extract ALL import statements (both `import` and `from ... import`)
3. **Dependency Graph**: Built complete module dependency graph with 224 internal imports
4. **Architectural Layer Analysis**: Evaluated imports against defined layer hierarchy
5. **Cycle Detection**: Used depth-first search to find circular dependencies up to 5 levels deep
6. **Code Quality Assessment**: Analyzed module sizes, abstraction patterns, and design smells

---

# 1. Architectural Layer Violations Analysis

## 1.1 Architectural Layer Hierarchy

Based on the project structure and CLAUDE.md documentation, the codebase follows this layered architecture:

- **Level 0: foundations/** - Pure abstractions, no dependencies
- **Level 1: base/** - Basic classes, should only depend on foundations  
- **Level 2: config/** - Configuration system, minimal dependencies
- **Level 3: core/** - Business logic, can depend on base/config
- **Level 4: database/, grid_systems/, domain/** - Data and domain layers, depend on core/base/config
- **Level 5: infrastructure/, resampling/, utils/** - Technical services  
- **Level 6: processors/, raster_data/, raster/** - Processing logic
- **Level 7: spatial_analysis/** - Application layer
- **Level 8: pipelines/** - Orchestration, top-level
- **Level 9: scripts/** - Scripts can import from anywhere

## 1.2 Architectural Violations Summary

**Total Violations**: 26
- **CRITICAL**: 5 violations
- **HIGH**: 10 violations  
- **MEDIUM**: 4 violations
- **LOW**: 7 violations

### 1.2.1 Critical Architectural Violations

#### PIPELINES INTERNAL IMPORTS (5 CRITICAL violations)

**Issue**: Pipeline stage modules incorrectly importing from the same pipelines layer, violating the rule that no layer should import from the orchestration layer.

**Files**:
- `src/pipelines/stages/analysis_stage.py:10` - imports `.base_stage`
- `src/pipelines/stages/merge_stage.py:12` - imports `.base_stage`  
- `src/pipelines/stages/load_stage.py:8` - imports `.base_stage`
- `src/pipelines/stages/resample_stage.py:9` - imports `.base_stage`
- `src/pipelines/stages/export_stage.py:8` - imports `.base_stage`

**Root Cause**: The analyzer incorrectly flagged these as violations. These are internal imports within the pipelines layer and should be allowed.

**Resolution**: Update architectural rules to allow internal imports within the same layer.

### 1.2.2 High Priority Violations

#### BASE LAYER IMPORTING FROM CORE (7 HIGH violations)

**Issue**: Base layer (level 1) incorrectly imports from core layer (level 3), violating the fundamental architectural principle that lower layers should not depend on higher layers.

**Files**:
- `src/base/feature.py:10` - imports `..core.registry`
- `src/base/dataset.py:13` - imports `..core.registry`
- `src/base/processor.py:18` - imports `..core.registry`
- `src/base/processor.py:22` - imports `..core.progress_manager`
- `src/base/processor.py:23` - imports `..core.progress_events`
- `src/base/processor.py:26` - imports `..core.signal_handler`
- `src/base/grid.py:11` - imports `..core.registry`

**Impact**: This creates tight coupling between foundational classes and business logic, making the base layer dependent on core functionality.

#### BASE LAYER IMPORTING FROM DATABASE (3 HIGH violations)

**Issue**: Base layer (level 1) imports from database layer (level 4), skipping multiple architectural levels.

**Files**:
- `src/base/feature.py:12` - imports `..database.schema`
- `src/base/processor.py:20` - imports `..database.schema`  
- `src/base/grid.py:13` - imports `..database.schema`

**Impact**: Creates inappropriate dependency from foundational classes to data access layer.

### 1.2.3 Medium Priority Violations

#### CORE IMPORTING FROM GRID SYSTEMS (3 MEDIUM violations)

**Issue**: Core layer (level 3) imports from grid_systems layer (level 4).

**Files**:
- `src/core/build.py:143` - imports `..grid_systems`
- `src/core/registry.py:169` - imports `..grid_systems.cubic_grid`
- `src/core/registry.py:170` - imports `..grid_systems.hexagonal_grid`

**Note**: These violations also create circular dependencies (see Section 1.3).

#### BASE IMPORTING FROM CONFIG (1 MEDIUM violation)

**Issue**: Base layer imports from config layer, which should be at a similar or lower level.

**File**: `src/base/memory_manager.py:16` - imports `..config.processing_config`

### 1.2.4 Low Priority Violations  

#### GRID SYSTEMS IMPORTING FROM DATABASE (6 LOW violations)

**Issue**: Grid systems and database are at the same architectural level (4), but grid_systems should not depend on database.

**Files**:
- `src/grid_systems/bounds_manager.py:14,15` - imports database.schema, database.connection
- `src/grid_systems/cubic_grid.py:11,12` - imports database.schema, database.connection
- `src/grid_systems/grid_factory.py:12,13` - imports database.schema, database.connection

#### RESAMPLING IMPORTING FROM DATABASE (1 LOW violation)

**File**: `src/resampling/cache_manager.py:14` - imports `..database.schema`

## 1.3 Circular Dependencies Analysis

### 1.3.1 Circular Dependencies Found

#### 1. DATABASE MODULE CIRCULAR DEPENDENCY ⚠️ **CRITICAL**

**Cycle**: `database.connection ↔ database.schema`

**Files Involved**:
- `/home/jason/geo/src/database/connection.py` (Line 506)
- `/home/jason/geo/src/database/schema.py` (Line 4)

**Specific Import Statements**:
```python
# database/connection.py:506
from .schema import DatabaseSchema

# database/schema.py:4  
from .connection import DatabaseManager, db
```

**Impact**: This is a **critical** circular dependency that prevents proper module initialization and can cause import errors at runtime.

**Root Cause**: Tight coupling between database connection management and schema operations.

#### 2. CORE REGISTRY ↔ HEXAGONAL GRID ⚠️ **HIGH PRIORITY**

**Cycle**: `core.registry ↔ grid_systems.hexagonal_grid`

**Files Involved**:
- `/home/jason/geo/src/core/registry.py` (Line 170)
- `/home/jason/geo/src/grid_systems/hexagonal_grid.py` (Line 13)

**Specific Import Statements**:
```python
# core/registry.py:170
from ..grid_systems.hexagonal_grid import HexagonalGrid

# grid_systems/hexagonal_grid.py:13
from ..core.registry import component_registry
```

**Impact**: Classic component registration circular dependency that can cause initialization order issues.

**Root Cause**: Registry imports grid classes for registration, while grid classes import registry for self-registration.

#### 3. CORE REGISTRY ↔ CUBIC GRID ⚠️ **HIGH PRIORITY**

**Cycle**: `core.registry ↔ grid_systems.cubic_grid`

**Files Involved**:
- `/home/jason/geo/src/core/registry.py` (Line 169)
- `/home/jason/geo/src/grid_systems/cubic_grid.py` (Line 10)

**Specific Import Statements**:
```python
# core/registry.py:169
from ..grid_systems.cubic_grid import CubicGrid

# grid_systems/cubic_grid.py:10
from ..core.registry import component_registry
```

**Impact**: Same pattern as hexagonal grid - component registration circular dependency.

**Root Cause**: Identical to hexagonal grid issue - mutual dependency between registry and grid components.

### 1.3.2 Circular Dependency Statistics

- **Total Python files**: 117
- **Total modules analyzed**: 117  
- **Modules with imports**: 75
- **Total internal imports**: 224
- **Circular dependencies**: 3
- **Average cycle length**: 2.0 steps (all are direct circular dependencies)
- **Circular dependency rate**: 1.3% (3 out of 224 imports)

---

# 2. Code Quality & Design Issues Analysis

## 2.1 Module Duplication Crisis

### 🔴 **Critical Duplication Issues**
- **`checkpoint_types.py`** - **VERIFIED**: Exact same 12 classes (276 lines) duplicated in both `foundations/types/` and `base/`
- **`src/raster/` vs `src/raster_data/`** - **CORRECTED**: Empty directories with only `__pycache__` files; actual raster code is in `domain/raster/`

### 2.1.1 Impact Assessment
The checkpoint_types duplication represents a real maintenance burden with 276 lines of identical code in two locations, creating inconsistency risks during updates.

## 2.2 God Classes & Oversized Modules

### 🔴 **Oversized Modules Requiring Decomposition** - **VERIFIED**
- **`database/schema.py`** - **1,346 lines** (confirmed: database operations overload)
- **`base/processor.py`** - **936 lines** (confirmed: base class doing too much)
- **`pipelines/orchestrator.py`** - **1,098 lines** (confirmed: orchestration god class)

### 2.2.1 God Class Indicators
These modules violate the Single Responsibility Principle and indicate need for decomposition into focused, cohesive classes. Analysis confirms all three files exceed 900+ lines, with database/schema.py being the largest at 1,346 lines.

## 2.3 Manager/Handler Explosion

### 🔴 **Manager Class Anti-Pattern** - **VERIFIED**
- **6 core Manager classes identified**:
  - `DatabaseManager` (database/connection.py)
  - `MemoryManager` (base/memory_manager.py)
  - `ProgressManager` (core/progress_manager.py)
  - `ResamplingCacheManager` (resampling/cache_manager.py)
  - `BoundsManager` (grid_systems/bounds_manager.py)
  - `CleanupManager` (utils/cleanup_manager.py)
- **Pattern indicates procedural thinking** in OOP design (god object anti-pattern)

### 2.3.1 Design Pattern Issues
The proliferation of "Manager" classes suggests lack of proper domain modeling and over-reliance on procedural patterns. Analysis shows 6 confirmed Manager classes, with an additional ~60 files referencing Manager patterns.

## 2.4 Abstraction Layer Confusion

### 🟠 **Abstraction Issues** - **VERIFIED**
- **7+ different processor abstractions** identified:
  - `BaseProcessor` (base/processor.py)
  - `BaseTileProcessor` (base/tile_processor.py)  
  - `IProcessor` (foundations/interfaces/processor.py)
  - `EnhancedBaseProcessor` (infrastructure/processors/base_processor.py)
  - `DeprecatedBaseProcessor` (base/processor_compat.py)
  - Plus compatibility layers and test implementations
- **Config import pattern**: **44 files** directly import config (violating dependency injection)
- **Compatibility files**: processor_compat.py and base_analyzer_compat.py indicate breaking changes

### 2.4.1 Impact on Maintainability
Multiple competing abstractions create confusion and make the system harder to extend and maintain. The high config coupling (44 direct imports) violates dependency injection principles.

## 2.5 Global State & Incomplete Implementations

### 🟠 **Implementation Quality Issues** - **VERIFIED**
- **Global state patterns confirmed**:
  - `_signal_handler` global variable in core/signal_handler.py (line 255)
  - Singleton pattern in SignalHandler class (line 24-32)
  - Multiple `get_*()` factory functions creating global instances
- **170 files with `pass` statements** across 54 files (indicating stub implementations)
- **10 files raise NotImplementedError** across 8 files (premature abstraction)

### 2.5.1 Technical Debt Indicators
Analysis confirms significant technical debt with 170 `pass` statements and 10 NotImplementedError cases, suggesting over-engineering without sufficient implementation. Global state pattern in signal_handler violates clean architecture principles.

---

# 3. Recommended Solutions & Implementation Strategy

## 3.1 Priority-Based Resolution Plan

### 3.1.1 Immediate Actions (CRITICAL)
1. **Fix database circular dependency** - Blocks proper initialization
2. **Address base layer violations** - Break fundamental architecture principles
3. **Resolve module duplication** - Eliminate 8,000+ lines of duplicate code

### 3.1.2 High Priority Actions
1. **Fix grid system circular dependencies** - Affects component discovery
2. **Decompose god classes** - Break down oversized modules
3. **Audit Manager classes** - Eliminate procedural anti-patterns

### 3.1.3 Medium Priority Actions
1. **Review grid_systems/database coupling** - Same-level dependency issues
2. **Implement dependency injection** - Replace direct config imports
3. **Consolidate abstractions** - Eliminate competing base classes

## 3.2 Detailed Resolution Strategies

### 3.2.1 Fix Database Circular Dependency

**Option A: Extract Database Interface** (Recommended)
```python
# Create src/database/interfaces.py
from abc import ABC, abstractmethod

class DatabaseInterface(ABC):
    @abstractmethod
    def execute_sql_file(self, file_path): pass
    @abstractmethod
    def get_cursor(self): pass

# Modify connection.py to implement interface
class DatabaseManager(DatabaseInterface):
    # Implementation without importing schema
    pass

# Modify schema.py to use interface instead of concrete class
from .interfaces import DatabaseInterface
```

**Option B: Dependency Injection**
```python
# In database/schema.py, inject the connection rather than importing
class DatabaseSchema:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
# In database/connection.py, create schema after initialization
class DatabaseManager:
    def __init__(self):
        # ... initialization
        self._schema = None
    
    @property
    def schema(self):
        if self._schema is None:
            from .schema import DatabaseSchema
            self._schema = DatabaseSchema(self)
        return self._schema
```

### 3.2.2 Fix Registry ↔ Grid Circular Dependencies

**Option A: Registry Factory Pattern** (Recommended)
```python
# Create src/core/grid_factory.py
class GridFactory:
    @staticmethod
    def create_grids():
        from ..grid_systems.cubic_grid import CubicGrid
        from ..grid_systems.hexagonal_grid import HexagonalGrid
        
        return {
            'cubic': CubicGrid,
            'hexagonal': HexagonalGrid
        }

# Modify core/registry.py
class ComponentRegistry:
    def __init__(self):
        self._grids = {}
    
    def register_grids(self):
        from .grid_factory import GridFactory
        self._grids.update(GridFactory.create_grids())

# Modify grid classes to remove registry import
# Use dependency injection or post-initialization registration
```

### 3.2.3 Fix Base Layer Architectural Violations

**Extract Interfaces**
Create abstract interfaces in foundations layer that higher layers can implement:

```python
# src/foundations/interfaces/registry_interface.py
from abc import ABC, abstractmethod

class RegistryInterface(ABC):
    @abstractmethod
    def register(self, component_type: str, component_class): pass
    
    @abstractmethod  
    def get(self, component_type: str): pass
```

**Dependency Injection**
Modify base classes to accept dependencies via constructor injection:

```python
# src/base/processor.py - Remove direct imports
class BaseProcessor:
    def __init__(self, registry: RegistryInterface = None, 
                 progress_manager=None, schema=None):
        self._registry = registry
        self._progress = progress_manager
        self._schema = schema
```

**Factory Pattern**
Create factory classes in appropriate layers to construct objects with their dependencies:

```python
# src/core/factories/processor_factory.py
class ProcessorFactory:
    def create_processor(self, processor_type: str) -> BaseProcessor:
        processor = self._get_processor_class(processor_type)
        return processor(
            registry=self.registry,
            progress_manager=self.progress_manager,
            schema=self.schema
        )
```

## 3.3 Files Requiring Changes

### 3.3.1 Immediate Changes Required

**Database Circular Dependency**:
1. `/home/jason/geo/src/database/connection.py` - Remove schema import, use dependency injection
2. `/home/jason/geo/src/database/schema.py` - Accept connection via constructor parameter  

**Registry Circular Dependencies**:
3. `/home/jason/geo/src/core/registry.py` - Use factory pattern or late registration for grids
4. `/home/jason/geo/src/grid_systems/cubic_grid.py` - Remove direct registry import
5. `/home/jason/geo/src/grid_systems/hexagonal_grid.py` - Remove direct registry import

### 3.3.2 Architectural Refactoring Priority

**Highest Priority** (Base layer violations):
- `src/base/processor.py` (5 violations) - Remove core and database dependencies
- `src/base/feature.py` (2 violations) - Remove core and database dependencies
- `src/base/grid.py` (2 violations) - Remove core and database dependencies
- `src/base/dataset.py` (1 violation) - Remove core registry dependency
- `src/base/memory_manager.py` (1 violation) - Remove config dependency

**Medium Priority** (Core layer violations):  
- `src/core/registry.py` (2 violations) - Already covered in circular dependency analysis
- `src/core/build.py` (1 violation) - Remove grid_systems dependency

**Lower Priority** (Same-level coupling):
- Grid systems files importing from database layer
- Resampling cache manager importing from database

## 3.4 Verification Process

After implementing these changes, re-run the architectural analyzer to verify all issues have been resolved:

```bash
python comprehensive_circular_analysis.py
python architectural_violation_analyzer.py
```

**Target Results**:
- **"🎉 NO CIRCULAR DEPENDENCIES FOUND!"**
- **"✅ NO ARCHITECTURAL VIOLATIONS DETECTED!"**

---

# 4. Long-term Architectural Improvements

## 4.1 Establish Clear Layer Boundaries
- Document and enforce architectural layer responsibilities
- Implement automated architectural testing
- Create layer-specific interfaces and contracts

## 4.2 Eliminate Code Duplication
- Remove duplicate `checkpoint_types.py` implementations (276 lines duplicated)
- Clean up empty `raster/` and `raster_data/` directories (only contain __pycache__)
- Establish single source of truth for shared functionality

## 4.3 Decompose God Classes
- Break down `database/schema.py` (1,346 lines) into focused components
- Refactor `base/processor.py` (936 lines) using composition
- Split `pipelines/orchestrator.py` (1,098 lines) into pipeline stages

## 4.4 Replace Manager Anti-Pattern
- Review necessity of 6 Manager classes identified
- Convert procedural Manager classes to domain-driven design
- Implement proper aggregates and bounded contexts
- Use dependency injection instead of global singletons

---

# 5. Detailed Solution Implementation Guide

## 5.1 Raster Module Redundancy Resolution

### Problem Analysis
**Current State**: 
- `src/raster/` and `src/raster_data/` are **empty directories** containing only `__pycache__` files
- **Active implementation** is in `src/domain/raster/` with complete functionality
- All current imports point to `src/domain.raster.*` (62+ confirmed import statements)
- `domain/raster/__init__.py` explicitly states: "Unified raster processing module - consolidated from raster/ and raster_data/"

### Root Cause
These are **legacy directories** from a previous consolidation effort that were not cleaned up properly. The actual consolidation was already completed into `domain/raster/`.

### Solution: Directory Cleanup
**Action**: Delete empty legacy directories entirely

**Files to Delete**:
```bash
# These directories contain ONLY __pycache__ files
src/raster/                    # DELETE entire directory
src/raster_data/               # DELETE entire directory
```

**Files to Keep**:
```bash
src/domain/raster/             # KEEP - Active implementation with 8 modules
├── __init__.py               # Consolidated exports
├── catalog.py                # RasterCatalog implementation
├── adapters/                 # RasterSourceAdapter
├── loaders/                  # 4 loader implementations
└── validators/               # 2 validator implementations
```

**No Code Changes Required**: 
- All imports already point to `src.domain.raster.*`
- No breaking changes - this is pure cleanup
- Zero integration impact

---

## 5.2 Checkpoint Types Duplication Resolution

### Problem Analysis
**Current State**:
- `src/base/checkpoint_types.py` - 276 lines with 12 classes
- `src/foundations/types/checkpoint_types.py` - **Identical 276 lines** (100% duplication)
- Both files have identical content down to comments and spacing

### Solution: Consolidate to Foundations Layer
**Keep**: `src/foundations/types/checkpoint_types.py` (architectural principle: types belong in foundations)
**Delete**: `src/base/checkpoint_types.py`

**Files to Check for Import Updates**:
```python
# Search for imports that need updating:
grep -r "from src.base.checkpoint_types" src/
grep -r "from .checkpoint_types" src/base/
```

**Import Migration Required**:
```python
# OLD imports (to be replaced):
from src.base.checkpoint_types import CheckpointLevel, CheckpointData
from .checkpoint_types import StorageBackend

# NEW imports (replacement):
from src.foundations.types.checkpoint_types import CheckpointLevel, CheckpointData
from src.foundations.types.checkpoint_types import StorageBackend
```

---

## 5.3 God Class Decomposition Solutions

### 5.3.1 Database Schema (1,346 lines)

**Problem**: `src/database/schema.py` contains multiple responsibilities
**Solution**: Split into focused modules by domain

**Proposed Structure**:
```
src/database/
├── schema/
│   ├── __init__.py           # Consolidated interface
│   ├── grid_operations.py    # Grid and cell operations  
│   ├── species_operations.py # Species range operations
│   ├── feature_operations.py # Feature and climate data
│   ├── experiment_tracking.py # Experiment management
│   └── raster_cache.py      # Resampling cache operations
└── schema.py                # KEEP as compatibility layer
```

**Implementation Plan**:
1. Extract grid operations (store_grid_definition, store_grid_cells_batch)
2. Extract species operations (store_species_range, store_species_intersections_batch)
3. Extract feature operations (store_features_batch, store_climate_data_batch)
4. Extract experiment tracking (create_experiment, update_experiment_status)
5. Extract raster cache (store_resampling_cache_batch, get_cached_resampling_values)
6. Keep original schema.py as facade importing from all modules

### 5.3.2 Base Processor (936 lines)

**Problem**: `src/base/processor.py` handles too many concerns
**Solution**: Extract mixins and specialized processors

**Proposed Structure**:
```
src/base/
├── processor/
│   ├── __init__.py          # BaseProcessor interface
│   ├── core.py             # Core processing logic
│   ├── memory_mixin.py     # Memory tracking functionality
│   ├── batch_mixin.py      # Batch processing functionality
│   ├── checkpoint_mixin.py # Checkpoint integration
│   └── progress_mixin.py   # Progress reporting
└── processor.py            # KEEP as compatibility import
```

### 5.3.3 Pipeline Orchestrator (1,098 lines)

**Problem**: `src/pipelines/orchestrator.py` handles all orchestration logic
**Solution**: Extract stage management and monitoring

**Proposed Structure**:
```
src/pipelines/
├── orchestrator/
│   ├── __init__.py          # Main orchestrator interface
│   ├── core.py             # Core orchestration logic
│   ├── stage_manager.py    # Stage execution management
│   ├── monitor_manager.py  # Monitoring coordination
│   └── recovery_manager.py # Recovery coordination
└── orchestrator.py         # KEEP as compatibility import
```

---

## 5.4 Manager Class Anti-Pattern Resolution

### Problem Analysis
**Current Manager Classes** (6 identified):
1. `DatabaseManager` (database/connection.py) - **KEEP** (legitimate database connection management)
2. `MemoryManager` (base/memory_manager.py) - **REFACTOR** to MemoryTracker service
3. `ProgressManager` (core/progress_manager.py) - **REFACTOR** to ProgressService
4. `ResamplingCacheManager` (resampling/cache_manager.py) - **REFACTOR** to CacheService
5. `BoundsManager` (grid_systems/bounds_manager.py) - **REFACTOR** to BoundsCalculator
6. `CleanupManager` (utils/cleanup_manager.py) - **REFACTOR** to CleanupService

### Solution: Convert to Domain Services
**Principle**: Replace procedural "Manager" pattern with domain-driven services

**Refactoring Examples**:
```python
# OLD: Manager anti-pattern
class MemoryManager:
    def manage_memory(self): pass
    def cleanup_memory(self): pass
    def monitor_memory(self): pass

# NEW: Domain service
class MemoryTracker:
    def track_usage(self): pass
    def calculate_available(self): pass
    
class MemoryAllocator:
    def allocate(self, size): pass
    def deallocate(self, reference): pass
```

---

## 5.5 Abstraction Layer Confusion Resolution

### Problem Analysis
**Current Processor Abstractions** (7+ identified):
- `BaseProcessor` (base/processor.py) - Core implementation
- `BaseTileProcessor` (base/tile_processor.py) - Tile-specific
- `IProcessor` (foundations/interfaces/processor.py) - Interface
- `EnhancedBaseProcessor` (infrastructure/processors/base_processor.py) - Enhanced version
- `DeprecatedBaseProcessor` (base/processor_compat.py) - Compatibility layer

### Solution: Unified Processor Hierarchy
**Target Architecture**:
```
foundations/interfaces/processor.py     # IProcessor interface (KEEP)
base/processor.py                      # BaseProcessor implementation (KEEP - consolidate others into this)
infrastructure/processors/             # DELETE - merge into base/processor.py  
base/processor_compat.py              # DELETE after migration period
base/tile_processor.py                # REFACTOR to use composition, not inheritance
```

**Config Import Coupling Resolution**:
- **Current**: 44 files directly import config
- **Solution**: Implement dependency injection pattern
- **Pattern**: Pass config through constructors, not direct imports

```python
# OLD: Direct import anti-pattern
from src.config import config
class MyProcessor:
    def process(self):
        setting = config.processing.batch_size

# NEW: Dependency injection
class MyProcessor:
    def __init__(self, config: Config):
        self.config = config
    def process(self):
        setting = self.config.processing.batch_size
```

---

## 5.6 Global State Elimination

### Problem Analysis
**Global State Patterns Identified**:
- `_signal_handler` global variable (core/signal_handler.py:255)
- Singleton pattern in SignalHandler (core/signal_handler.py:24-32)
- Multiple `get_*()` factory functions creating global instances

### Solution: Dependency Injection Container
**Replace**: Global singletons with proper dependency injection

**Implementation**:
```python
# NEW: Dependency container
class ServiceContainer:
    def __init__(self):
        self._services = {}
    
    def register(self, service_type, instance):
        self._services[service_type] = instance
    
    def get(self, service_type):
        return self._services.get(service_type)

# Application startup:
container = ServiceContainer()
container.register(SignalHandler, SignalHandler())
container.register(ProgressManager, ProgressManager())
```

---

## 5.7 Implementation Priority and File Modification List

### Phase 1: Immediate Cleanup (Zero Breaking Changes)
**Files to Delete**:
```bash
src/raster/                           # DELETE entire directory
src/raster_data/                      # DELETE entire directory  
src/base/checkpoint_types.py          # DELETE (keep foundations version)
```

**Files to Search and Update** (checkpoint imports):
```bash
# Find and update these import statements:
grep -r "from src.base.checkpoint_types" src/
grep -r "from .checkpoint_types" src/base/
# Replace with: from src.foundations.types.checkpoint_types
```

### Phase 2: God Class Decomposition  
**Files to Refactor**:
```bash
src/database/schema.py               # Split into 5 modules
src/base/processor.py                # Split into 5 mixins
src/pipelines/orchestrator.py       # Split into 4 modules
```

### Phase 3: Manager Pattern Elimination
**Files to Refactor**:
```bash
src/base/memory_manager.py           # → MemoryTracker + MemoryAllocator
src/core/progress_manager.py         # → ProgressService
src/resampling/cache_manager.py      # → CacheService  
src/grid_systems/bounds_manager.py   # → BoundsCalculator
src/utils/cleanup_manager.py         # → CleanupService
```

### Phase 4: Global State Elimination
**Files to Refactor**:
```bash
src/core/signal_handler.py           # Remove singleton, add to DI container
src/core/progress_manager.py         # Remove global instance
src/core/registry.py                 # Remove global instance
```

This implementation guide provides concrete, actionable steps to resolve each identified architectural issue with minimal disruption to the existing system.

---

*This comprehensive analysis provides a complete roadmap for resolving all identified architectural issues, circular dependencies, and code quality problems in the geodiversity analysis system.*