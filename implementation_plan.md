# Unified Checkpoint System Implementation Plan

## ⚡ Quick Start
```bash
# Activate environment
conda activate geo

# After each phase completion:
# 1. Check the box [x] in this file
# 2. Test functionality with provided test commands
# 3. Clear temporary test files and old redundant files
# 4. Commit changes before moving to next phase
```

## 🎯 Objective
Replace 5 chaotic checkpoint systems with 1 clean, unified system distributed across `base/` (core abstractions) and `checkpoints/` (implementation).

## 📊 Current State Audit
- ❌ **Core Checkpoint Manager**: Metadata-based, registry.json format
- ❌ **Pipeline Checkpoint Manager**: JSON files per stage format  
- ❌ **Tile Processor Checkpoints**: Pickled dataclass format
- ❌ **Database Checkpoints**: PostgreSQL pipeline_checkpoints table
- ❌ **Processor Memory Checkpoints**: Progress + state via core manager

## 🏗️ Target Architecture
```
src/base/
├── checkpoint.py         # Core abstractions & interfaces
├── checkpointable.py     # Mixin for checkpointable processes  
└── checkpoint_types.py   # Data structures & enums

src/checkpoints/
├── __init__.py          # Unified public API
├── manager.py           # UnifiedCheckpointManager
├── storage.py           # Storage backend abstraction
├── backends/            # Concrete storage implementations
│   ├── file_backend.py  # File system storage
│   ├── db_backend.py    # Database storage
│   └── memory_backend.py # In-memory storage
├── registry.py          # Central checkpoint registry
└── validator.py         # Integrity validation
```

---

## 📋 Implementation Checklist

### **Phase 1: Core Abstractions (base/)**
- [x] **1.1** Create `src/base/checkpoint_types.py` ✅ **COMPLETED & TESTED**
  - [x] Define `CheckpointLevel` enum (PIPELINE, STAGE, STEP, SUBSTEP)
  - [x] Define `CheckpointStatus` enum (CREATED, VALID, CORRUPTED, ARCHIVED)
  - [x] Define `CheckpointData` dataclass (universal data structure)
  - [x] Define `CheckpointMetadata` dataclass
  - [x] Define `StorageConfig` dataclass

- [x] **1.2** Create `src/base/checkpoint.py` ✅ **COMPLETED & TESTED**
  - [x] Define `CheckpointStorage` abstract base class
    - [x] Abstract methods: `save()`, `load()`, `list()`, `delete()`, `exists()`
  - [x] Define `CheckpointValidator` abstract base class
    - [x] Abstract methods: `validate()`, `calculate_checksum()`, `verify_integrity()`
  - [x] Define `CheckpointError` exception hierarchy
  - [x] Define helper functions: `generate_checkpoint_id()`, `parse_checkpoint_id()`

- [x] **1.3** Create `src/base/checkpointable.py` ✅ **COMPLETED & TESTED**
  - [x] Define `CheckpointableProcess` mixin class
    - [x] Methods: `save_checkpoint()`, `load_checkpoint()`, `should_checkpoint()`
    - [x] Properties: `checkpoint_interval`, `process_id`, `checkpoint_metadata`
  - [x] Define `CheckpointPolicy` class for timing/retention rules
  - [x] Define `ResumableProcess` interface
    - [x] Abstract methods: `get_state()`, `restore_state()`, `get_progress()`

- [x] **1.4** Update `src/base/__init__.py` ✅ **COMPLETED & TESTED**
  - [x] Export all new checkpoint abstractions
  - [x] Ensure clean import structure

### **Phase 2: Storage Backends (checkpoints/backends/)**
- [x] **2.1** Create `src/checkpoints/backends/file_backend.py` ✅ **COMPLETED & TESTED**
  - [x] Implement `FileCheckpointStorage` class
  - [x] Support JSON format (for pipeline compatibility)
  - [x] Support binary/pickle format (for tile processor compatibility) 
  - [x] Hierarchical directory structure: `{base_dir}/{process_id}/{level}/{checkpoint_id}`
  - [x] Atomic write operations (write to temp, then rename)
  - [x] Compression support for large checkpoints

- [x] **2.2** Create `src/checkpoints/backends/db_backend.py` ✅ **COMPLETED & TESTED**
  - [x] Implement `DatabaseCheckpointStorage` class
  - [x] Use existing `pipeline_checkpoints` table from schema.sql
  - [x] Support JSONB data storage for PostgreSQL
  - [x] Implement efficient querying (by process_id, level, timestamp)
  - [x] Transaction support for atomic operations

- [ ] **2.3** Create `src/checkpoints/backends/memory_backend.py`
  - [ ] Implement `MemoryCheckpointStorage` class
  - [ ] In-memory dictionary storage
  - [ ] Optional persistence to disk on shutdown
  - [ ] Useful for testing and temporary checkpoints

- [ ] **2.4** Create `src/checkpoints/storage.py`
  - [ ] Implement `StorageBackendFactory` 
  - [ ] Auto-detection of storage type from config
  - [ ] Storage backend pooling/caching
  - [ ] Unified error handling and retry logic

### **Phase 3: Core Manager (checkpoints/)**
- [ ] **3.1** Create `src/checkpoints/registry.py`
  - [ ] Implement `CheckpointRegistry` class
  - [ ] Track all active checkpoints across storage backends
  - [ ] Hierarchical checkpoint relationships (parent-child)
  - [ ] Efficient querying and filtering
  - [ ] Cleanup policies (age-based, count-based)

- [ ] **3.2** Create `src/checkpoints/validator.py`
  - [ ] Implement `DefaultCheckpointValidator` class
  - [ ] Checksum validation (SHA256)
  - [ ] Data integrity verification
  - [ ] Schema validation for checkpoint data
  - [ ] Corruption detection and reporting

- [ ] **3.3** Create `src/checkpoints/manager.py`
  - [ ] Implement `UnifiedCheckpointManager` class
  - [ ] Core methods: `save()`, `load()`, `load_latest()`, `resume_from()`
  - [ ] Process lifecycle management
  - [ ] Automatic cleanup based on policies
  - [ ] Progress tracking integration
  - [ ] Thread-safe operations

- [ ] **3.4** Create `src/checkpoints/__init__.py`
  - [ ] Export unified public API
  - [ ] Singleton `get_checkpoint_manager()` function
  - [ ] Configuration loading from `src.config`
  - [ ] Backward-compatible imports (temporary)

### **Phase 4: Migration & Integration**
- [ ] **4.1** Migrate Core Checkpoint Manager
  - [ ] Update `src/base/processor.py` to use unified system
  - [ ] Migrate checkpoint format (registry.json → unified)
  - [ ] Update process_manager.py to use unified API
  - [ ] Test processor checkpoint functionality

- [ ] **4.2** Migrate Pipeline Checkpoint Manager  
  - [ ] Update `src/pipelines/orchestrator.py` to use unified system
  - [ ] Migrate existing stage-based JSON checkpoints
  - [ ] Update resume functionality in orchestrator
  - [ ] Test pipeline resumability

- [ ] **4.3** Migrate Tile Processor Checkpoints
  - [ ] Update `src/base/tile_processor.py` to use unified system
  - [ ] Migrate pickle format to unified format
  - [ ] Update tile-based processing resume logic
  - [ ] Test tile processor resumability

- [ ] **4.4** Migrate Database Checkpoints
  - [ ] Update `src/database/schema.py` to use unified system
  - [ ] Ensure database backend uses existing table structure
  - [ ] Update database checkpoint queries
  - [ ] Test database checkpoint operations

- [ ] **4.5** Migrate Processor Memory Checkpoints
  - [ ] Already handled via processor.py migration in 4.1
  - [ ] Verify memory checkpoint functionality
  - [ ] Test checkpoint timing and policies

### **Phase 5: Testing & Validation**
- [ ] **5.1** Unit Tests
  - [ ] Test all base abstractions
  - [ ] Test all storage backends independently
  - [ ] Test checkpoint manager functionality
  - [ ] Test migration compatibility

- [ ] **5.2** Integration Tests
  - [ ] Test full pipeline resumability
  - [ ] Test processor checkpointing
  - [ ] Test tile processor resumability
  - [ ] Test database integration

- [ ] **5.3** Migration Testing
  - [ ] Test existing checkpoint format migration
  - [ ] Test backward compatibility during transition
  - [ ] Test rollback procedures if needed

- [ ] **5.4** Performance Testing
  - [ ] Benchmark checkpoint save/load operations
  - [ ] Test with large checkpoint data
  - [ ] Test concurrent checkpoint operations
  - [ ] Memory usage validation

### **Phase 6: Cleanup & Documentation**
- [ ] **6.1** Remove Old Systems
  - [ ] Delete `src/core/checkpoint_manager.py`
  - [ ] Delete `src/pipelines/recovery/checkpoint_manager.py`  
  - [ ] Remove old checkpoint logic from `src/base/tile_processor.py`
  - [ ] Clean up database schema methods (keep table, remove old methods)

- [ ] **6.2** Update Dependencies
  - [ ] Update all import statements across codebase
  - [ ] Remove legacy checkpoint imports
  - [ ] Update configuration references
  - [ ] Update process_manager.py CLI

- [ ] **6.3** Documentation
  - [ ] Update CLAUDE.md architecture documentation
  - [ ] Document unified checkpoint API
  - [ ] Create migration guide for developers
  - [ ] Update configuration documentation

---

## 🎯 Success Criteria
- ✅ **Single source of truth**: Only one checkpoint system exists
- ✅ **Unified API**: All checkpoint operations use the same interface
- ✅ **Full resumability**: All existing checkpoint use cases work
- ✅ **Clean architecture**: Clear separation between base abstractions and implementation
- ✅ **Backward compatibility**: Existing checkpoints can be migrated
- ✅ **Performance**: No regression in checkpoint save/load times
- ✅ **Reliability**: No data loss during migration

## 🚀 Estimated Timeline
- **Phase 1**: 2-3 days (base abstractions)
- **Phase 2**: 3-4 days (storage backends)  
- **Phase 3**: 2-3 days (core manager)
- **Phase 4**: 4-5 days (migration)
- **Phase 5**: 2-3 days (testing)
- **Phase 6**: 1-2 days (cleanup)

**Total**: ~14-20 days for complete unified checkpoint system

## 🔄 Rollback Plan
If issues arise during migration:
1. Keep old checkpoint systems until migration is complete
2. Use feature flags to switch between old/new systems
3. Maintain old checkpoint files during transition period
4. Have automated rollback scripts ready

---

**Next Step**: Begin Phase 1 - Create base abstractions starting with `src/base/checkpoint_types.py`