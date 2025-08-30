# Repetitive Patterns Analysis - Full Codebase Scan

## Executive Summary
After comprehensive analysis of 210+ Python files across all modules, I've identified significant repetitive patterns that could benefit from abstraction. The codebase has ~63,791 lines of Python code with many common patterns repeated across modules.

## 1. CRITICAL REPETITION PATTERNS

### 1.1 File I/O Operations (50+ occurrences)
**Pattern**: save_results, save_figure, save_output, load_data, load_config
```python
# Repeated in 20+ files with slight variations:
def save_results(self, results: Dict, output_path: Path) -> Path
def load_data(self, source: Union[str, Path]) -> Any
def save_figure(self, fig: plt.Figure, filename: str) -> None
```

**Files affected**:
- src/biodiversity_analysis/methods/cpwopt/analyzer_adapter.py
- src/biodiversity_analysis/methods/som/analyzer.py
- src/biodiversity_analysis/methods/kmeans_hierarchical/hierarchical_analyzer.py
- src/machine_learning/doubleML/refactored_pipeline.py
- src/biodiversity_analysis/statistical/data_utils.py
- And 15+ more files

**Recommendation**: Create `abstractions/io.py` with:
```python
class IDataPersistence(ABC):
    @abstractmethod
    def save_results(self, results: Dict, output_path: Path) -> Path
    @abstractmethod
    def load_results(self, input_path: Path) -> Dict

class IFigureSaver(ABC):
    @abstractmethod
    def save_figure(self, fig, output_path: Path, dpi: int = 150) -> Path
```

### 1.2 Validation Methods (40+ occurrences)
**Pattern**: validate_parameters, validate_input_data, check_*, verify_*
```python
# Repeated pattern across analyzers:
def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]
def validate_input_data(self, data: Any) -> Tuple[bool, List[str]]
```

**Files affected**:
- Every analyzer class (CPWOPT, SOM, KMeans)
- Config loaders (7 instances)
- ML pipelines
- Security utils

**Recommendation**: Create `abstractions/validation.py`:
```python
class IValidator(ABC):
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]
    @abstractmethod
    def validate_input_data(self, data: Any) -> Tuple[bool, List[str]]
```

### 1.3 Logger Initialization (50+ files)
**Pattern**: Every module initializes logger identically
```python
logger = logging.getLogger(__name__)
```

**Recommendation**: Create `abstractions/mixins.py`:
```python
class LoggerMixin:
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__module__)
        return self._logger
```

### 1.4 Configuration Management (7+ implementations)
**Pattern**: Config loaders all implement similar patterns
```python
def load_config(self, preset: Optional[str] = None) -> Dict[str, Any]
def validate_config(self, config: Dict[str, Any]) -> None
def save_config(self, output_path: Path) -> None
```

**Already abstracted**: IMethodConfig exists but could be expanded

### 1.5 Visualization Base (30+ files)
**Pattern**: Matplotlib figure creation and saving
```python
fig, ax = plt.subplots(figsize=(...))
# plotting code
save_figure(fig, output_path)
```

**Recommendation**: Create `abstractions/visualization.py`:
```python
class IVisualizer(ABC):
    @abstractmethod
    def create_figure(self, **kwargs) -> Tuple[Figure, Axes]
    @abstractmethod
    def save_figure(self, fig: Figure, output_path: Path) -> None
```

## 2. MODULE-SPECIFIC PATTERNS

### 2.1 Machine Learning Module
- **Cross-validators**: 3 classes (SpatialBlockCV, SpatialBufferCV, EnvironmentalBlockCV) inherit from BaseCrossValidator
- **Diagnostics**: Multiple diagnostic classes with similar structure
- **Plotting**: 10+ visualization modules with repeated figure setup code

### 2.2 Biodiversity Analysis Module
- **Reporters**: 8 reporter classes with similar base functionality
- **Preprocessors**: Multiple transform() methods with similar signatures
- **Analyzers**: All follow analyze() -> AnalysisResult pattern (already abstracted)

### 2.3 Config Module
- **Dataset utilities**: Repeated validation patterns
- **Path handling**: Similar path validation across modules

## 3. ERROR HANDLING PATTERNS

### 3.1 Custom Exceptions (13+ definitions)
Multiple modules define their own exception hierarchies:
- CPWOPTError and 8 subclasses
- SOMConfigError
- DataLoadingError, ConfigurationError, SecurityValidationError

**Recommendation**: Create `abstractions/exceptions.py`:
```python
class AnalysisError(Exception):
    """Base for all analysis errors"""

class ValidationError(AnalysisError):
    """Data or parameter validation failed"""

class ConfigurationError(AnalysisError):
    """Configuration is invalid"""

class ConvergenceError(AnalysisError):
    """Algorithm failed to converge"""
```

## 4. DATA TRANSFORMATION PATTERNS

### 4.1 Transform Methods (9+ implementations)
```python
def transform(self, data: np.ndarray) -> np.ndarray
```
Found in preprocessing, SOM, CPWOPT modules

**Recommendation**: Create `abstractions/transformers.py`:
```python
class IDataTransformer(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'IDataTransformer'
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray
    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray
```

## 5. QUANTIFIED REPETITION IMPACT

### Lines of Code That Could Be Eliminated:
- File I/O patterns: ~500 lines (50 instances × 10 lines each)
- Validation methods: ~400 lines (40 instances × 10 lines)
- Logger initialization: ~100 lines
- Visualization setup: ~300 lines
- Error handling: ~200 lines
- **Total potential reduction: ~1,500 lines**

### Files That Would Be Affected:
- Direct changes needed: ~80 files
- Import updates needed: ~120 files

## 6. PRIORITY RECOMMENDATIONS

### High Priority (Most Impact, Least Risk):
1. **Create abstractions/io.py** - Consolidate all file I/O operations
2. **Create abstractions/validation.py** - Standardize validation
3. **Expand abstractions/exceptions.py** - Unified error hierarchy

### Medium Priority (Good Impact, Some Refactoring):
4. **Create abstractions/mixins.py** - LoggerMixin, CacheableMixin
5. **Create abstractions/visualization.py** - Plotting base classes
6. **Create abstractions/transformers.py** - Data transformation interface

### Low Priority (Nice to Have):
7. Consider consolidating ML diagnostic classes
8. Unify reporter base functionality
9. Standardize checkpoint/state management

## 7. ANTI-PATTERNS FOUND

1. **God Objects**: Some classes > 1500 lines (geo_som_core.py, refactored_pipeline.py)
2. **Duplicate Code**: vrt_merger.py appears twice (src/raster_merge/ and src/raster_merge/merger_raster/)
3. **Inconsistent Validation**: Some return Tuple[bool, List[str]], others raise exceptions
4. **Mixed Concerns**: Many classes handle both computation and visualization

## 8. IMPLEMENTATION STRATEGY

Following Linus's principle: "Abstract if used multiple times, don't if used once"

**Phase 1**: Create new abstractions (interfaces + mixins)
**Phase 2**: Gradually migrate high-use patterns
**Phase 3**: Remove redundant code
**Phase 4**: Standardize remaining modules

## Summary

The codebase has significant opportunities for abstraction, particularly around:
- **I/O operations** (50+ instances)
- **Validation** (40+ instances)  
- **Logging** (50+ files)
- **Visualization** (30+ files)
- **Error handling** (13+ custom exceptions)

Implementing these abstractions could reduce codebase by ~1,500 lines and significantly improve maintainability.