# Unified System Architecture Update Plan - DETAILED IMPLEMENTATION GUIDE

## ⚠️ CRITICAL SYSTEM INVARIANTS - READ FIRST

### Configuration System Rules (NEVER VIOLATE)
1. **Config vs config**:
   - import as `from src.config import config`
   - `config = Config()` - The config INSTANCE (lowercase c)
   - **NEVER**  - (this breaks cluster deployments) `from src.config import Config` - The Config CLASS (capital C)
   - **NEVER** use `src/config/defaults.py` parameters if it conflicts with `config.yml` - default values for fallback values only

2. **Configuration Loading Hierarchy**:
   ```python
   # CORRECT configuration usage:
   from src.config import config  # Import the instance
   # config.yml (if exists) overrides defaults.py
   # Environment variables override config.yml
   # Runtime parameters override environment variables
   ```

3. **Database Connection Rules**:
   - **NEVER** use `DatabaseManager(test_mode=True)` because that is associated with test on a remote client
   - **NEVER** hardcode database credentials
   - **NEVER** use production database in test mode
   - Test mode adds '_test' suffix to database name automatically

4. **Import Path Rules**:
   - **ALWAYS** use absolute imports: `from src.module import Component`
   - **NEVER** use relative imports across modules: `from ..other_module import X`
   - **NEVER** assume current working directory - code runs from different locations

5. **File System Rules**:
   - **ALWAYS** use `Path` objects for file operations
   - **NEVER** use string concatenation for paths
   - **ALWAYS** check file existence before operations
   - Cluster paths differ from local paths - use config for base directories

## Executive Summary

This plan addresses critical architectural issues through a systematic, foundation-first refactoring approach. It consolidates the insights from our architectural analysis, checkpoint implementation experience, and testing strategies into a single, comprehensive guide designed for step-by-step execution without requiring deep system knowledge.

## Critical Issues Requiring Resolution

### 1. **Dependency Inversion Violations** (SYSTEM-BREAKING)
**Impact**: Causes circular imports, breaks modularity, prevents testing
**Example of violation**:
```python
# THIS IS WRONG - base/ should NEVER import from higher layers
# base/processor.py
from src.core import registry  # VIOLATION!
from src.checkpoints import get_checkpoint_manager  # VIOLATION!
from src.database import schema  # VIOLATION!
```

### 2. **Complete Module Duplication** (8,000+ LINES)
**Impact**: Maintenance nightmare, import confusion, resource waste
**Evidence**: `raster/` and `raster_data/` are 95% identical
**Import chaos**: 200+ files import from one or the other randomly

**Detailed Duplication Analysis** (from extended_architecture_analysis.md):
| File Pair | Similarity | Lines | Critical Issues |
|-----------|------------|-------|-----------------|
| `raster/catalog.py` vs `raster_data/catalog.py` | 95% | ~500 each | Same class names, methods |
| `raster/loaders/base_loader.py` vs `raster_data/loaders/base_loader.py` | 98% | ~300 each | Nearly identical implementations |
| `raster/loaders/geotiff_loader.py` vs `raster_data/loaders/geotiff_loader.py` | 95% | ~400 each | Same GDAL operations |
| `raster/validators/coverage_validator.py` vs `raster_data/validators/coverage_validator.py` | 98% | ~200 each | Identical validation rules |

**Total Impact**: 8,000-10,000 lines of duplicate code across ~20 files

### 3. **Circular Dependencies** (IMPORT FAILURES)
**Impact**: Random import failures, unpredictable behavior
**Example**: `spatial_analysis/` ↔ `processors/spatial_analysis/`

### 4. **Infrastructure Redundancy** (3X MEMORY SYSTEMS)
**Impact**: Memory leaks, inconsistent behavior, performance issues

**Memory Management Systems Analysis**:
| Component | Location | Features | Issues |
|-----------|----------|----------|--------|
| MemoryManager | `base/memory_manager.py` | Allocation tracking, pressure detection | In wrong layer (should be infrastructure) |
| MemoryTracker | `base/memory_tracker.py` | Detailed snapshots, statistics | Duplicates manager functionality |
| MemoryMonitor | `pipelines/monitors/memory_monitor.py` | Pipeline-specific monitoring | Reimplements core features |

**Progress Tracking Systems**:
| Component | Location | Features | Overlap |
|-----------|----------|----------|---------|
| ProgressManager | `core/progress_manager.py` | Hierarchical progress | 70% overlap with tracker |
| ProgressTracker | `pipelines/monitors/progress_tracker.py` | Stage-based tracking | Different hierarchy model |

### 5. **Checkpoint System Success** (COMPLETED BUT TEACHES LESSONS)
**What worked**: Clean abstractions, proper layering, comprehensive testing
**Lesson learned**: This is the pattern to follow for all refactoring

### 6. **Config Import Pattern Inconsistency** (CONFUSION RISK)
**Impact**: Different import patterns cause subtle bugs
**Evidence**: 
- `config/__init__.py` exports both `Config` class and `config` instance
- Some files: `from src.config import config` (instance)
- Other files: `from src.config import Config` then `config = Config()` (creates new instance!)
- This creates MULTIPLE config instances with different values

### 7. **Analyzer Duplication Beyond Raster** (MEDIUM)
**Impact**: Confusion about which analyzer to use
**Evidence**:
- `spatial_analysis/gwpca_analyzer.py` vs `spatial_analysis/gwpca/gwpca_analyzer.py`
- Different parameter names (Config vs config in constructor)
- Different default values (bandwidth_method: 'cv' vs 'AICc')

### 8. **BaseAnalyzer Misplacement** (ARCHITECTURAL VIOLATION)
**Impact**: Base class in wrong location
**Current**: `spatial_analysis/base_analyzer.py`
**Should be**: `base/analyzer.py` or `foundations/interfaces/analyzer.py`

### 9. **Backup Files in Version Control** (HOUSEKEEPING)
**Impact**: Confusion, potential security risk
**Files**: `lazy_loadable.py.bak`, `merge_stage.py.backup`

## Refactoring Phases - Detailed Implementation

### Phase 0: Fix Config Import Pattern (URGENT - Before Any Refactoring)

**CRITICAL**: This MUST be fixed first as it affects ALL other changes.

#### Step 0.1: Standardize Config Import Pattern
```python
# DECISION: Use singleton instance pattern consistently
# src/config/__init__.py should ONLY export:
from .config import config  # The singleton instance
__all__ = ['config']  # Do NOT export Config class

# WHY: Exporting both causes multiple config instances with different values
```

#### Step 0.2: Update All Imports
```bash
# Find all files importing Config class
grep -r "from src.config import Config" src/ --include="*.py" > config_class_imports.txt

# Update each file to use instance instead:
# WRONG:
from src.config import Config
config = Config()  # Creates NEW instance!

# CORRECT:
from src.config import config  # Use singleton instance
```

#### Step 0.3: Fix Analyzer Constructor Patterns
```python
# Find analyzers taking Config as parameter
grep -r "def __init__.*Config" src/spatial_analysis/ --include="*.py"

# These should take config instance or no config parameter
# WRONG:
def __init__(self, config: Config, db_connection=None):
    
# CORRECT:
def __init__(self, db_connection=None):
    # Use global config instance
    from src.config import config
```

### Phase 1: Foundation Layer Creation (Week 1)

#### Day 1-2: Create Pure Abstractions

**CRITICAL**: This phase creates the foundation. ANY mistakes here cascade throughout the system.

**Step 1.1: Create Directory Structure**
```bash
# DO NOT create these in src/base/ - that's being replaced!
mkdir -p src/foundations/interfaces
mkdir -p src/foundations/types  
mkdir -p src/foundations/mixins

# Verify structure:
ls -la src/foundations/
# Should show: interfaces/ types/ mixins/
```

**Step 1.2: Create IProcessor Interface**

⚠️ **DO NOT**:
- Import ANYTHING from src/ in these files
- Include ANY implementation code
- Reference ANY concrete classes

✅ **DO**:
```python
# src/foundations/interfaces/processor.py
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Dict

class IProcessor(ABC):
    """Pure processor interface - NO IMPLEMENTATIONS!"""
    
    @abstractmethod
    def process_single(self, item: Any) -> Any:
        """Process a single item."""
        pass
    
    @abstractmethod
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input before processing."""
        pass
    
    @abstractmethod
    def get_config_requirements(self) -> Dict[str, Any]:
        """Declare configuration requirements."""
        pass
```

**Step 1.3: Create IGrid Interface**
```python
# src/foundations/interfaces/grid.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GridCell:
    """Pure data class - no behavior."""
    id: str
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    level: int
    parent_id: Optional[str] = None

class IGrid(ABC):
    """Pure grid interface."""
    
    @abstractmethod
    def create_cells(self) -> List[GridCell]:
        """Create grid cells."""
        pass
    
    @abstractmethod
    def get_resolution(self) -> float:
        """Get grid resolution in meters."""
        pass
```

**Step 1.4: Create Compatibility Adapters**

⚠️ **CRITICAL**: These maintain backward compatibility during migration

```python
# src/base/processor.py - TEMPORARY COMPATIBILITY LAYER
import warnings
from src.foundations.interfaces.processor import IProcessor

# This will be replaced with infrastructure.processors.enhanced_processor
# For now, keep existing implementation but add warning
class BaseProcessor:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BaseProcessor is deprecated. Use foundations.interfaces.IProcessor "
            "for interfaces or infrastructure.processors.EnhancedProcessor for "
            "implementation. This compatibility layer will be removed in Phase 6.",
            DeprecationWarning,
            stacklevel=2
        )
        # Existing implementation continues...
```

**Verification Steps**:
```bash
# Test that foundations has NO dependencies on other src modules
python -c "
import ast
import os

def check_imports(filepath):
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name.startswith('src.'):
                    print(f'ERROR: {filepath} imports from {name.name}')
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('src.'):
                print(f'ERROR: {filepath} imports from {node.module}')
                return False
    return True

# Check all files in foundations/
for root, dirs, files in os.walk('src/foundations'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            if not check_imports(filepath):
                exit(1)
                
print('✅ All foundation files are clean - no src imports')
"
```

#### Day 3-4: Move Clean Components

**Step 1.5: Identify and Move Clean Components**

✅ **Safe to move** (no src dependencies):
- `base/checkpoint_types.py` → `foundations/types/checkpoint_types.py`
- `base/cacheable.py` → `foundations/mixins/cacheable.py`
- `base/lazy_loadable.py` → `foundations/mixins/lazy_loadable.py`
- `base/tileable.py` → `foundations/mixins/tileable.py`

⚠️ **DO NOT move** (has dependencies):
- `base/processor.py` - imports from core, checkpoints, database
- `base/tile_processor.py` - imports from checkpoints
- `base/memory_manager.py` - infrastructure, not abstraction
- `base/dataset.py` - has database dependencies

**Move Script**:
```python
# move_clean_components.py
import shutil
import os
from pathlib import Path

moves = [
    ('src/base/checkpoint_types.py', 'src/foundations/types/checkpoint_types.py'),
    ('src/base/cacheable.py', 'src/foundations/mixins/cacheable.py'),
    ('src/base/lazy_loadable.py', 'src/foundations/mixins/lazy_loadable.py'),
    ('src/base/tileable.py', 'src/foundations/mixins/tileable.py'),
]

for src, dst in moves:
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"✅ Moved {src} to {dst}")
        
        # Update imports in the moved file
        with open(dst, 'r') as f:
            content = f.read()
        
        # Fix any relative imports
        content = content.replace('from .', 'from src.foundations.')
        
        with open(dst, 'w') as f:
            f.write(content)
    else:
        print(f"⚠️  Source file not found: {src}")
```

**Step 1.6: Update Imports in Moved Files**

Each moved file needs import updates:

```python
# Example for src/foundations/types/checkpoint_types.py
# BEFORE:
from src.base import SomeOtherType  # If any

# AFTER:
# Remove imports from src.base
# Add any needed standard library imports
```

#### Day 5: Create Infrastructure Layer

**Step 1.7: Create Infrastructure Directory**
```bash
mkdir -p src/infrastructure/memory
mkdir -p src/infrastructure/progress
mkdir -p src/infrastructure/processors
mkdir -p src/infrastructure/monitoring
```

**Step 1.8: Create Enhanced Processor**

This replaces the violation-filled BaseProcessor:

```python
# src/infrastructure/processors/enhanced_processor.py
from typing import Any, Optional, Dict, Tuple
from src.foundations.interfaces.processor import IProcessor
from src.foundations.mixins.checkpointable import CheckpointableProcess
from src.infrastructure.memory.tracker import MemoryTracker
from src.infrastructure.progress.tracker import ProgressTracker

class EnhancedProcessor(IProcessor, CheckpointableProcess):
    """Full-featured processor implementation."""
    
    def __init__(self, config: Optional[Dict] = None):
        # NO imports from src.base!
        self._config = config or {}
        self._memory_tracker = MemoryTracker()
        self._progress_tracker = ProgressTracker()
        
        # Initialize from config properly
        if 'checkpoint_interval' in self._config:
            self.checkpoint_interval = self._config['checkpoint_interval']
    
    def process_single(self, item: Any) -> Any:
        """Implement abstract method."""
        # Base implementation that subclasses override
        raise NotImplementedError("Subclasses must implement process_single")
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Implement abstract method."""
        # Basic validation - subclasses can extend
        if item is None:
            return False, "Input cannot be None"
        return True, None
    
    def get_config_requirements(self) -> Dict[str, Any]:
        """Declare what config keys this processor needs."""
        return {
            'checkpoint_interval': 100,
            'memory_limit_mb': 1024,
            'enable_progress_tracking': True
        }
```

### Phase 2: Critical Raster Module Consolidation (Week 2)

**CRITICAL**: This affects 200+ imports across the codebase. One mistake can break everything.

#### Day 1: Complete Analysis

**Step 2.1: Generate Import Inventory**
```bash
# Find ALL imports of raster modules
echo "=== Direct imports from raster/ ===" > raster_import_inventory.txt
grep -r "from src\.raster" src/ --include="*.py" >> raster_import_inventory.txt
grep -r "import src\.raster" src/ --include="*.py" >> raster_import_inventory.txt

echo -e "\n=== Direct imports from raster_data/ ===" >> raster_import_inventory.txt  
grep -r "from src\.raster_data" src/ --include="*.py" >> raster_import_inventory.txt
grep -r "import src\.raster_data" src/ --include="*.py" >> raster_import_inventory.txt

echo -e "\n=== Total affected files ===" >> raster_import_inventory.txt
grep -r "raster\|raster_data" src/ --include="*.py" -l | wc -l >> raster_import_inventory.txt

# Review the inventory
cat raster_import_inventory.txt
```

**Step 2.2: Create Feature Comparison Matrix**
```python
# analyze_raster_modules.py
import os
import ast
from collections import defaultdict

def analyze_module(module_path):
    """Extract classes, functions, and imports from module."""
    features = {
        'classes': [],
        'functions': [],
        'imports': [],
        'size': 0
    }
    
    for root, dirs, files in os.walk(module_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                features['size'] += os.path.getsize(filepath)
                
                with open(filepath, 'r') as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                features['classes'].append(f"{file}:{node.name}")
                            elif isinstance(node, ast.FunctionDef):
                                features['functions'].append(f"{file}:{node.name}")
                    except:
                        print(f"Error parsing {filepath}")
    
    return features

# Analyze both modules
raster_features = analyze_module('src/raster')
raster_data_features = analyze_module('src/raster_data')

print("=== Feature Comparison ===")
print(f"raster/: {len(raster_features['classes'])} classes, "
      f"{len(raster_features['functions'])} functions, "
      f"{raster_features['size']} bytes")
print(f"raster_data/: {len(raster_data_features['classes'])} classes, "
      f"{len(raster_data_features['functions'])} functions, "
      f"{raster_data_features['size']} bytes")

# Find differences
only_in_raster = set(raster_features['classes']) - set(raster_data_features['classes'])
only_in_raster_data = set(raster_data_features['classes']) - set(raster_features['classes'])

if only_in_raster:
    print(f"\nUnique to raster/: {only_in_raster}")
if only_in_raster_data:
    print(f"\nUnique to raster_data/: {only_in_raster_data}")
```

#### Day 2-3: Implement Unified Module

**Step 2.3: Create Unified Raster Module Structure**
```bash
# Create new structure in domain layer
mkdir -p src/domain/raster/sources
mkdir -p src/domain/raster/loaders
mkdir -p src/domain/raster/validators
mkdir -p src/domain/raster/processors
```

**Step 2.4: Merge Best Features**

⚠️ **CRITICAL DECISION MATRIX**:
```python
# merge_decision_matrix.py
decisions = {
    'catalog.py': {
        'keep_from': 'raster_data',  # Has enhanced metadata handling
        'merge_features': ['scan_directory method from raster/'],
        'new_location': 'src/domain/raster/catalog.py'
    },
    'loaders/geotiff_loader.py': {
        'keep_from': 'raster_data',  # Better error handling
        'merge_features': ['memory_efficient_load from raster/'],
        'new_location': 'src/domain/raster/loaders/geotiff_loader.py'
    },
    'validators/coverage_validator.py': {
        'keep_from': 'both',  # Merge validation rules
        'merge_features': ['union of all validation rules'],
        'new_location': 'src/domain/raster/validators/coverage_validator.py'
    }
}
```

**Step 2.5: Create Import Migration Script**

This is CRITICAL - it updates 200+ files:

```python
# migrate_raster_imports.py
import os
import re
from pathlib import Path

# Define import mappings
import_mappings = {
    # Old import -> New import
    'from src.raster.catalog import': 'from src.domain.raster.catalog import',
    'from src.raster_data.catalog import': 'from src.domain.raster.catalog import',
    'from src.raster.loaders.': 'from src.domain.raster.loaders.',
    'from src.raster_data.loaders.': 'from src.domain.raster.loaders.',
    'import src.raster': 'import src.domain.raster',
    'import src.raster_data': 'import src.domain.raster',
}

def update_imports_in_file(filepath, dry_run=True):
    """Update imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    for old_import, new_import in import_mappings.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            changes.append(f"  {old_import} -> {new_import}")
    
    if changes and not dry_run:
        # Create backup
        backup_path = f"{filepath}.backup"
        with open(backup_path, 'w') as f:
            f.write(original_content)
        
        # Write updated content
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {filepath}")
        for change in changes:
            print(change)
    elif changes:
        print(f"Would update {filepath}:")
        for change in changes:
            print(change)
    
    return len(changes) > 0

# First run in dry-run mode
print("=== DRY RUN - No files will be modified ===")
total_files = 0
for root, dirs, files in os.walk('src'):
    # Skip the old raster directories
    if 'raster' in root or 'raster_data' in root:
        continue
        
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            if update_imports_in_file(filepath, dry_run=True):
                total_files += 1

print(f"\nTotal files to update: {total_files}")
print("\nRun with dry_run=False to actually update files")
```

#### Day 4: Validation and Cleanup

**Step 2.6: Validate Functionality Preserved**
```python
# test_raster_merge.py
import unittest
from src.domain.raster.catalog import RasterCatalog
from src.domain.raster.loaders.geotiff_loader import GeoTiffLoader

class TestRasterMerge(unittest.TestCase):
    """Ensure all functionality is preserved after merge."""
    
    def test_catalog_functionality(self):
        """Test that catalog works as before."""
        catalog = RasterCatalog()
        # Add specific tests based on your needs
        
    def test_loader_functionality(self):
        """Test that loaders work as before."""
        loader = GeoTiffLoader()
        # Add specific tests
        
    def test_no_old_imports(self):
        """Ensure no code imports from old modules."""
        import subprocess
        result = subprocess.run(
            ['grep', '-r', 'from src.raster', 'src/', '--include=*.py'],
            capture_output=True
        )
        self.assertEqual(result.returncode, 1, "Found imports from old raster module")

if __name__ == '__main__':
    unittest.main()
```

**Step 2.7: Remove Old Modules**

⚠️ **ONLY DO THIS AFTER ALL TESTS PASS!**

```bash
# Create archive first
tar -czf raster_modules_backup.tar.gz src/raster src/raster_data

# Verify archive
tar -tzf raster_modules_backup.tar.gz | head -20

# Remove old modules
rm -rf src/raster
rm -rf src/raster_data

echo "✅ Old raster modules removed. Backup at raster_modules_backup.tar.gz"
```

### Phase 2.5: Fix Analyzer Duplication (Week 2 - Day 5)

#### Step 2.5.1: Identify All Analyzer Duplicates
```bash
# Find duplicate analyzer patterns
find src/spatial_analysis -name "*_analyzer.py" -type f | sort

# Check for duplicates
# gwpca_analyzer.py exists in TWO locations:
# - src/spatial_analysis/gwpca_analyzer.py
# - src/spatial_analysis/gwpca/gwpca_analyzer.py
```

#### Step 2.5.2: Compare and Merge Analyzers
```python
# compare_analyzers.py
import difflib

files = [
    'src/spatial_analysis/gwpca_analyzer.py',
    'src/spatial_analysis/gwpca/gwpca_analyzer.py'
]

# Key differences to preserve:
# 1. Block aggregation support (in gwpca/ version)
# 2. Different default parameters
# 3. xarray support (in gwpca/ version)

# DECISION: Keep gwpca/ version as it has more features
# Move to: src/domain/analysis/gwpca/gwpca_analyzer.py
```

#### Step 2.5.3: Move BaseAnalyzer to Proper Location
```bash
# BaseAnalyzer is currently misplaced
# FROM: src/spatial_analysis/base_analyzer.py
# TO: src/foundations/interfaces/analyzer.py

# Update all imports:
# OLD: from src.spatial_analysis.base_analyzer import BaseAnalyzer
# NEW: from src.foundations.interfaces.analyzer import IAnalyzer
```

### Phase 3: Fix Dependency Violations (Week 3)

#### Day 1-2: Extract Concrete Implementations

**Step 3.1: Analyze BaseProcessor Dependencies**
```python
# analyze_base_processor.py
import ast

with open('src/base/processor.py', 'r') as f:
    tree = ast.parse(f.read())

imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        imports.append(f"from {node.module} import {', '.join(n.name for n in node.names)}")

print("=== BaseProcessor imports (VIOLATIONS) ===")
for imp in imports:
    if 'src.' in imp and 'src.base' not in imp:
        print(f"❌ {imp}")
```

**Step 3.2: Create Processor Hierarchy**

```python
# src/foundations/interfaces/processor_advanced.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ICheckpointableProcessor(IProcessor):
    """Processor that can be checkpointed."""
    
    @abstractmethod
    def should_checkpoint(self) -> bool:
        """Determine if checkpoint is needed."""
        pass
    
    @abstractmethod
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data to checkpoint."""
        pass

class IMemoryAwareProcessor(IProcessor):
    """Processor that tracks memory usage."""
    
    @abstractmethod
    def estimate_memory_usage(self, input_size: int) -> int:
        """Estimate memory needs."""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> int:
        """Get current memory usage."""
        pass

# Concrete implementations go in infrastructure/processors/
```

### Phase 4: Testing Strategy Integration (Week 4)

Based on checkpoint_test_plan.md, here's the comprehensive testing approach:

#### Day 1: Test Environment Setup

**Step 4.1: Create Test Infrastructure**
```bash
#!/bin/bash
# setup_test_env.sh

# Ensure test database
python -c "
from src.database.connection import DatabaseManager
db = DatabaseManager(test_mode=True)
db.create_test_database()
print('✅ Test database ready')
"

# Create test directories
mkdir -p test_outputs/{foundations,infrastructure,domain,application}
mkdir -p test_data/{raster,vector,configs}

# Verify environment
python -c "
import sys
import os
print(f'Python: {sys.version}')
print(f'Working directory: {os.getcwd()}')
print(f'PYTHONPATH: {os.environ.get(\"PYTHONPATH\", \"Not set\")}')

# Check critical imports work
try:
    from src.config import Config
    print('✅ Config import works')
except Exception as e:
    print(f'❌ Config import failed: {e}')
    
try:
    from src.database.connection import DatabaseManager
    print('✅ Database import works')
except Exception as e:
    print(f'❌ Database import failed: {e}')
"
```

#### Day 2-3: Layer-by-Layer Testing

**Checkpoint System Testing** (from checkpoint_test_plan.md):

**Test Environment Setup**:
```bash
# Create test checkpoint directories
mkdir -p test_checkpoints/{file,memory,db}

# Verify checkpoint backends
python -c "
from src.checkpoints.backends import FileCheckpointStorage, DatabaseCheckpointStorage, MemoryCheckpointStorage
print('✅ All checkpoint backends importable')
"
```

**Checkpoint Storage Backend Tests**:
```python
# test_checkpoint_backends.py
from src.checkpoints.backends.file_backend import FileCheckpointStorage
from src.checkpoints.backends.db_backend import DatabaseCheckpointStorage
from src.checkpoints.backends.memory_backend import MemoryCheckpointStorage
from src.base.checkpoint_types import CheckpointData, CheckpointLevel
import json

def test_file_backend():
    storage = FileCheckpointStorage(base_dir='test_checkpoints/file')
    data = CheckpointData(
        process_id='test_process',
        level=CheckpointLevel.STAGE,
        data={'test': 'data', 'count': 42},
        metadata={'stage': 'processing'}
    )
    checkpoint_id = storage.save(data)
    loaded = storage.load(checkpoint_id)
    assert loaded.data['count'] == 42
    print("✅ File backend test passed")

def test_db_backend():
    from src.database.connection import DatabaseManager
    db_manager = DatabaseManager(test_mode=True)
    storage = DatabaseCheckpointStorage(db_manager)
    data = CheckpointData(
        process_id='test_db_process',
        level=CheckpointLevel.PIPELINE,
        data={'pipeline': 'test', 'progress': 0.75},
        metadata={'experiment_id': 'exp_123'}
    )
    checkpoint_id = storage.save(data)
    loaded = storage.load(checkpoint_id)
    assert loaded.data['progress'] == 0.75
    print("✅ Database backend test passed")

def test_memory_backend():
    storage = MemoryCheckpointStorage()
    for i in range(100):
        data = CheckpointData(
            process_id=f'mem_proc_{i % 10}',
            level=CheckpointLevel.SUBSTEP,
            data={'iteration': i},
            metadata={}
        )
        checkpoint_id = storage.save(data)
    print(f"✅ Memory backend test passed - {len(storage._storage)} checkpoints")

# Run all tests
test_file_backend()
test_db_backend()
test_memory_backend()
```

**Step 4.2: Foundation Layer Tests**
```python
# test_foundations.py
import unittest
import importlib
import ast

class TestFoundationLayer(unittest.TestCase):
    """Ensure foundation layer has no dependencies."""
    
    def test_no_src_imports(self):
        """Foundation should not import from src.*"""
        import os
        
        for root, dirs, files in os.walk('src/foundations'):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        tree = ast.parse(f.read())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom):
                            if node.module and node.module.startswith('src.'):
                                self.fail(f"{filepath} imports from {node.module}")
    
    def test_interfaces_are_abstract(self):
        """All interface methods should be abstract."""
        from src.foundations.interfaces import processor
        
        # Check that IProcessor cannot be instantiated
        with self.assertRaises(TypeError):
            processor.IProcessor()
```

**Step 4.3: Integration Tests**
```python
# test_integration.py
class TestSystemIntegration(unittest.TestCase):
    """Test that refactored components work together."""
    
    def test_processor_with_checkpoints(self):
        """Enhanced processor should support checkpointing."""
        from src.infrastructure.processors.enhanced_processor import EnhancedProcessor
        from src.checkpoints import get_checkpoint_manager
        
        # This tests that the new architecture works with existing checkpoint system
        processor = EnhancedProcessor({'checkpoint_interval': 10})
        manager = get_checkpoint_manager()
        
        # Process some items
        for i in range(20):
            processor.process_single(f"item_{i}")
            
        # Verify checkpoints were created
        checkpoints = manager.list_checkpoints(processor.process_id)
        self.assertGreater(len(checkpoints), 0)
```

### Phase 5: Risk Mitigation Implementation (Week 5)

#### Risk Assessment Matrix (from extended_architecture_analysis.md)

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|-------------------|
| Import Chaos (200+ files) | CRITICAL | HIGH | Automated migration script, phased approach |
| Performance Regression | HIGH | MEDIUM | Continuous benchmarking, 5% tolerance |
| Functionality Loss | HIGH | MEDIUM | Feature parity matrix, comprehensive testing |
| Circular Dependencies | MEDIUM | HIGH | Dependency validation tools, CI/CD checks |
| Memory Usage Increase | MEDIUM | LOW | Memory profiling, optimization focus |

#### Import Chaos Mitigation

**Step 5.1: Create Import Validator**
```python
# validate_imports.py
#!/usr/bin/env python3
"""
Run this after ANY refactoring to ensure imports are valid.
"""
import subprocess
import sys
from pathlib import Path

def check_all_imports():
    """Try to import every Python file to catch issues early."""
    errors = []
    src_path = Path('src')
    
    for py_file in src_path.rglob('*.py'):
        # Convert path to module name
        module_path = py_file.relative_to(src_path.parent)
        module_name = str(module_path).replace('/', '.').replace('.py', '')
        
        try:
            # Try to import the module
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except Exception as e:
            errors.append(f"❌ {module_name}: {e}")
            print(f"❌ {module_name}: {e}")
    
    return errors

if __name__ == '__main__':
    print("Validating all imports...")
    errors = check_all_imports()
    
    if errors:
        print(f"\n❌ Found {len(errors)} import errors")
        sys.exit(1)
    else:
        print("\n✅ All imports valid")
        sys.exit(0)
```

#### Performance Regression Prevention

**Step 5.2: Create Performance Baselines**
```python
# benchmark_baseline.py
import time
import psutil
import json
from pathlib import Path

def benchmark_operation(operation_name, func, *args, **kwargs):
    """Benchmark a single operation."""
    process = psutil.Process()
    
    # Memory before
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Time operation
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'operation': operation_name,
        'duration_seconds': end_time - start_time,
        'memory_used_mb': mem_after - mem_before,
        'timestamp': time.time()
    }

# Benchmark current system before refactoring
baselines = []

# Example: Benchmark raster loading
from src.raster.loaders.geotiff_loader import GeoTiffLoader  # OLD module
loader = GeoTiffLoader()
baseline = benchmark_operation(
    'geotiff_load',
    loader.load,
    'test_data/sample.tif'
)
baselines.append(baseline)

# Save baselines
with open('performance_baselines.json', 'w') as f:
    json.dump(baselines, f, indent=2)

print("✅ Performance baselines saved")
```

### Phase 6: Cleanup and Documentation (Week 6)

#### Day 0: Remove Backup Files
**Step 6.0: Clean Version Control**
```bash
# Find all backup files
find src/ -name "*.bak" -o -name "*.backup" -o -name "*~" | tee backup_files.txt

# Review the list
cat backup_files.txt

# Remove backup files
while read file; do
    echo "Removing: $file"
    rm "$file"
done < backup_files.txt

# Add to .gitignore to prevent future issues
echo "*.bak" >> .gitignore
echo "*.backup" >> .gitignore
echo "*~" >> .gitignore
```

#### Day 1: Remove Deprecated Code

**Step 6.1: Safe Deprecation Removal**
```python
# remove_deprecated.py
import os
import ast

# List of deprecated items to remove
deprecated = [
    'src/base/processor.py',  # After all imports updated
    'src/base/tile_processor.py',
    # Add more as identified
]

# First, verify nothing imports these
for module in deprecated:
    result = subprocess.run(
        ['grep', '-r', f'from {module.replace("/", ".").replace(".py", "")}', 'src/'],
        capture_output=True
    )
    if result.returncode == 0:
        print(f"❌ Still has imports: {module}")
        print(result.stdout.decode())
    else:
        print(f"✅ Safe to remove: {module}")
```

## Architecture Validation Checklist

### Pre-Refactoring Checklist
- [ ] Current system backed up
- [ ] All tests passing
- [ ] Performance baselines recorded
- [ ] Import inventory created
- [ ] Team notified of changes

### Per-Phase Validation
- [ ] No circular imports introduced
- [ ] All tests still passing
- [ ] Performance within 5% of baseline
- [ ] Import validator passes
- [ ] Documentation updated

### Post-Refactoring Validation
- [ ] All deprecated code removed
- [ ] No references to old modules
- [ ] Architecture rules enforced
- [ ] Documentation complete
- [ ] Team trained on new structure

## Common Pitfalls and How to Avoid Them

### 1. **Config vs config Confusion**
```python
# ❌ WRONG - This breaks on cluster
from src.config import config  # lowercase config instance

# ✅ CORRECT
from src.config import Config  # Capital Config class
config = Config()  # Create instance
```

### 2. **Modifying defaults.py**
```python
# ❌ WRONG - Never modify defaults.py directly
# src/config/defaults.py
DATABASE_HOST = 'production.server.com'  # NO!

# ✅ CORRECT - Use config.yml or environment variables
# config.yml
database:
  host: production.server.com
```

### 3. **Relative Imports Across Modules**
```python
# ❌ WRONG - Breaks when run from different locations
from ..other_module import Something

# ✅ CORRECT - Always use absolute imports
from src.other_module import Something
```

### 4. **Direct Database Access**
```python
# ❌ WRONG - Bypasses connection pooling and safety
import psycopg2
conn = psycopg2.connect("dbname=geo")

# ✅ CORRECT - Use DatabaseManager
from src.database.connection import DatabaseManager
db = DatabaseManager()
```

### 5. **Hardcoded Paths**
```python
# ❌ WRONG - Breaks on different systems
checkpoint_dir = '/home/user/checkpoints'

# ✅ CORRECT - Use configuration
from src.config import Config
config = Config()
checkpoint_dir = config.checkpoints.base_dir
```

## Success Metrics and Monitoring

### Quantitative Metrics
- **Code Reduction**: Target 8,000-10,000 lines removed
- **Import Statements**: 200+ simplified to <50
- **Circular Dependencies**: 0 (down from 3+)
- **Performance**: No regression (±5%)
- **Memory Usage**: 15% reduction target

### Automated Validation Script
```bash
#!/bin/bash
# validate_architecture.sh

echo "=== Architecture Validation ==="

# Check for circular dependencies
echo -n "Circular dependencies: "
python -m pycycle src/ --here . 2>/dev/null | wc -l

# Check for old module references
echo -n "References to old raster module: "
grep -r "src\.raster" src/ --include="*.py" 2>/dev/null | wc -l

# Check foundation layer purity
echo -n "Foundation layer violations: "
grep -r "from src\." src/foundations/ --include="*.py" 2>/dev/null | wc -l

# Run import validator
echo "Import validation:"
python validate_imports.py

# Check performance
echo "Performance check:"
python benchmark_check.py
```

## Rollback Strategy (from implementation_plan.md)

If issues arise during migration:

### 1. **Feature Flags for Gradual Rollout**
```python
# feature_flags.py
import os

class FeatureFlags:
    # Phase-specific flags
    USE_NEW_FOUNDATIONS = os.getenv('USE_NEW_FOUNDATIONS', 'false').lower() == 'true'
    USE_UNIFIED_RASTER = os.getenv('USE_UNIFIED_RASTER', 'false').lower() == 'true'
    USE_UNIFIED_MEMORY = os.getenv('USE_UNIFIED_MEMORY', 'false').lower() == 'true'
    
    @staticmethod
    def is_enabled(feature: str) -> bool:
        return os.getenv(f'FEATURE_{feature}', 'false').lower() == 'true'

# Usage in code
if FeatureFlags.USE_UNIFIED_RASTER:
    from src.domain.raster import RasterCatalog  # New
else:
    from src.raster import RasterCatalog  # Old
```

### 2. **Phased Rollback Plan**
1. Keep old modules until migration complete
2. Maintain compatibility shims during transition
3. Archive old checkpoint/config files
4. Have automated rollback scripts ready:

```bash
#!/bin/bash
# rollback_phase.sh
PHASE=$1

case $PHASE in
  "raster")
    echo "Rolling back raster consolidation..."
    rm -rf src/domain/raster
    tar -xzf raster_modules_backup.tar.gz
    ./restore_raster_imports.sh
    ;;
  "foundations")
    echo "Rolling back foundation layer..."
    rm -rf src/foundations
    ./restore_base_imports.sh
    ;;
  *)
    echo "Unknown phase: $PHASE"
    exit 1
    ;;
esac
```

## Architectural Rules Enforcement

### Automated Dependency Check
```python
# check_architecture.py
import ast
import os
from collections import defaultdict

def check_layer_dependencies():
    """Ensure no upward dependencies in architecture."""
    
    layers = {
        'foundations': 0,
        'infrastructure': 1,
        'domain': 2,
        'application': 3,
        'orchestration': 4
    }
    
    violations = []
    
    for root, dirs, files in os.walk('src'):
        for file in files:
            if not file.endswith('.py'):
                continue
                
            filepath = os.path.join(root, file)
            
            # Determine layer of current file
            current_layer = None
            for layer_name, level in layers.items():
                if f'/{layer_name}/' in filepath:
                    current_layer = level
                    break
            
            if current_layer is None:
                continue
            
            # Check imports
            with open(filepath, 'r') as f:
                try:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom):
                            if node.module and node.module.startswith('src.'):
                                # Check if importing from lower layer
                                for layer_name, level in layers.items():
                                    if f'.{layer_name}.' in node.module:
                                        if level < current_layer:
                                            violations.append({
                                                'file': filepath,
                                                'imports': node.module,
                                                'violation': f'Layer {current_layer} importing from layer {level}'
                                            })
                except:
                    pass
    
    return violations

# Run check
violations = check_layer_dependencies()
if violations:
    print("❌ Architecture violations found:")
    for v in violations:
        print(f"  {v['file']}: {v['violation']}")
else:
    print("✅ No architecture violations")
```

### Common Architectural Patterns to Enforce

1. **Interface Segregation**
   - Interfaces should be small and focused
   - No "god interfaces" with 20+ methods
   - Use composition over inheritance

2. **Dependency Injection**
   - Components should receive dependencies, not create them
   - Use factory pattern or registry for creation

3. **Single Source of Truth**
   - One implementation per functionality
   - One configuration system
   - One way to do each thing

## Conclusion

This comprehensive plan provides the detailed, step-by-step guidance needed to refactor the system without requiring deep contextual knowledge. By following these explicit instructions and heeding the warnings about system-breaking changes, even models without holistic understanding can execute the refactoring safely.

The key principles:
1. **Fix config pattern FIRST** (Phase 0) before any other changes
2. **Never violate system invariants** (config singleton, paths, imports)
3. **Test at every step** before proceeding
4. **Maintain backwards compatibility** during migration
5. **Validate continuously** using provided scripts
6. **Document everything** for future reference
7. **Remove duplicates systematically** (raster, analyzers, memory systems)
8. **Enforce architecture rules** with automated checks

Success depends on methodical execution and strict adherence to the guidelines provided. The addition of Phase 0 (config pattern fix) and Phase 2.5 (analyzer deduplication) ensures that all critical issues identified in the codebase are addressed.