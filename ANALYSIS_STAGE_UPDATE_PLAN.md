# Analysis Stage Systematic Update Plan

## Golden Rules

### 1. Always Respect Hierarchy and System Structure
- **Base layer** (`src/base/`) contains only abstract implementations, no imports from higher layers
- **Abstractions layer** (`src/abstractions/`) defines pure interfaces and types
- **Mid-level layer** (`src/spatial_analysis/`) implements specific algorithms using base abstractions
- **High-level layer** (`src/pipelines/`) orchestrates and coordinates lower layers
- **Infrastructure layer** (`src/infrastructure/`) provides cross-cutting services (logging, monitoring)

### 2. Abstraction
- Every concrete implementation must have an abstract interface
- Depend on abstractions, not concretions
- Use factory patterns to decouple creation from usage
- Data flows through well-defined interfaces

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Pipeline Orchestrator                   │
│                 (High-level Control)                     │
├─────────────────────────────────────────────────────────┤
│                   Analysis Stage                         │
│              (Orchestration & Config)                    │
├─────────────────────────────────────────────────────────┤
│                  Analyzer Factory                        │
│               (Decoupled Creation)                       │
├─────────────────────────────────────────────────────────┤
│   SOMAnalyzer │ GWPCAAnalyzer │ MaxPRegionsAnalyzer    │
│              (Concrete Implementations)                  │
├─────────────────────────────────────────────────────────┤
│                   BaseAnalyzer                          │
│              (Common Implementation)                     │
├─────────────────────────────────────────────────────────┤
│                    IAnalyzer                            │
│                (Pure Interface)                         │
├─────────────────────────────────────────────────────────┤
│     BaseDataset │ ProgressBackend │ StructuredLogger    │
│              (Infrastructure Services)                   │
└─────────────────────────────────────────────────────────┘
```

## Implementation Checklist

### Phase 1: Interface & Base Layer Updates (Foundation)

#### 1.1 Update Analyzer Interface
**File**: `src/abstractions/interfaces/analyzer.py`
```python
# Add to IAnalyzer interface:
@abstractmethod
def save_results(self, result: AnalysisResult, output_name: str, 
                 output_dir: Path = None) -> Path:
    """Save analysis results to disk"""
    pass

@abstractmethod
def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
    """Set callback for progress updates"""
    pass
```
**Notes**: 
- Pure interface, no implementation
- Maintains abstraction principle
- All analyzers must implement these methods

#### 1.2 Fix BaseAnalyzer Fatal Bug
**File**: `src/base/analyzer.py`
```python
# Add missing method that SOM calls:
def _update_progress(self, current: int, total: int, message: str):
    """Update progress through callback if set"""
    if hasattr(self, '_progress_callback') and self._progress_callback:
        self._progress_callback(current, total, message)

# Add progress callback setter
def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
    """Set callback for progress updates"""
    self._progress_callback = callback

# Add default save_results implementation
def save_results(self, result: AnalysisResult, output_name: str, 
                 output_dir: Path = None) -> Path:
    """Default implementation for saving results"""
    # Implementation details...
```
**Notes**:
- Fixes the AttributeError in SOM
- Provides sensible defaults
- Subclasses can override if needed

### Phase 2: Data Source Abstraction

#### 2.1 Create Analysis Dataset Classes
**File**: `src/processors/data_preparation/analysis_data_source.py`
```python
from src.base.dataset import BaseDataset
from src.abstractions.types import DataType, DatasetInfo

class ParquetAnalysisDataset(BaseDataset):
    """Dataset for streaming parquet files"""
    # Implementation following BaseDataset contract
    
class DatabaseAnalysisDataset(BaseDataset):
    """Dataset for streaming from database"""
    # Implementation following BaseDataset contract
```
**Notes**:
- Extends existing BaseDataset abstraction
- Respects established patterns
- Provides streaming capabilities

#### 2.2 Create Data Iterators
**File**: `src/spatial_analysis/data_iterators.py`
```python
class AnalysisDataIterator(ABC):
    """Base class for analysis-specific data iteration"""
    @abstractmethod
    def iterate(self, dataset: BaseDataset, **kwargs) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        pass

class SOMDataIterator(AnalysisDataIterator):
    """Random sampling for SOM"""
    
class GWPCADataIterator(AnalysisDataIterator):
    """Spatial tiles for GWPCA"""
    
class MaxPDataIterator(AnalysisDataIterator):
    """Connected regions for MaxP"""
```
**Notes**:
- Each algorithm gets optimal data access pattern
- Maintains abstraction with common interface
- Lives in spatial_analysis layer (algorithm-specific)

### Phase 3: Configuration Integration

#### 3.1 Update Configuration Schema
**File**: `config.yml`
```yaml
# Add new configuration sections
analysis:
  data_source: 'parquet'  # or 'database'
  chunk_size: 10000
  save_results:
    enabled: true
    output_dir: 'outputs/analysis'
    formats: ['pkl', 'json', 'csv']
  keep_results_in_memory: false
  memory:
    max_memory_gb: 8
    reduce_on_pressure: true
    cleanup_after_stage: true

# Update existing sections
som_analysis:
  max_pixels_in_memory: 1000000
  memory_overhead_factor: 3.0
  use_subsampling: true
  # ... keep existing config

gwpca_analysis:
  tile_size: 1000
  spatial_overlap: 0.1
  # ... keep existing config

maxp_regions:
  max_region_size: 10000
  connectivity: 'queen'
  # ... keep existing config
```
**Notes**:
- Centralized configuration
- No hard-coded values in code
- Follows existing config patterns

### Phase 4: Analyzer Standardization

#### 4.1 Standardize SOMAnalyzer
**File**: `src/spatial_analysis/som/som_trainer.py`
```python
class SOMAnalyzer(BaseAnalyzer):
    def __init__(self, config: Config, db_connection: Optional[DatabaseManager] = None):
        """Standardized constructor"""
        super().__init__(config, db_connection)
        # Remove duplicate config loading
        # Use self.safe_get_config() for all config access
```
**Changes**:
- Remove direct config import
- Use config passed through constructor
- Ensure save_results is implemented

#### 4.2 Standardize GWPCAAnalyzer
**File**: `src/spatial_analysis/gwpca/gwpca_analyzer.py`
```python
# Same pattern as SOM
```

#### 4.3 Standardize MaxPRegionsAnalyzer
**File**: `src/spatial_analysis/maxp_regions/region_optimizer.py`
```python
# Same pattern as SOM
```

**Notes**:
- Consistent initialization across all analyzers
- Respect dependency injection principle
- Remove direct config imports

### Phase 5: Factory Pattern Implementation

#### 5.1 Create Analyzer Factory
**File**: `src/pipelines/stages/analyzer_factory.py`
```python
from typing import Dict, Tuple, Type
from importlib import import_module
from src.abstractions.interfaces.analyzer import IAnalyzer
from src.config import Config
from src.database.connection import DatabaseManager

class AnalyzerFactory:
    """Factory for creating analyzer instances"""
    
    # Registry of available analyzers
    _analyzers: Dict[str, Tuple[str, str]] = {
        'som': ('src.spatial_analysis.som.som_trainer', 'SOMAnalyzer'),
        'gwpca': ('src.spatial_analysis.gwpca.gwpca_analyzer', 'GWPCAAnalyzer'),
        'maxp_regions': ('src.spatial_analysis.maxp_regions.region_optimizer', 'MaxPRegionsAnalyzer')
    }
    
    @classmethod
    def create(cls, method: str, config: Config, db: DatabaseManager) -> IAnalyzer:
        """Create analyzer instance"""
        if method not in cls._analyzers:
            available = ', '.join(cls._analyzers.keys())
            raise ValueError(f"Unknown analysis method: {method}. Available: {available}")
        
        module_path, class_name = cls._analyzers[method]
        
        try:
            module = import_module(module_path)
            analyzer_class = getattr(module, class_name)
            
            # Verify it implements IAnalyzer
            if not issubclass(analyzer_class, IAnalyzer):
                raise TypeError(f"{class_name} does not implement IAnalyzer interface")
            
            return analyzer_class(config, db)
            
        except ImportError as e:
            raise ImportError(f"Failed to import {method} analyzer: {e}")
        except AttributeError as e:
            raise AttributeError(f"Analyzer class {class_name} not found in {module_path}: {e}")
    
    @classmethod
    def register(cls, method: str, module_path: str, class_name: str):
        """Register new analyzer type"""
        cls._analyzers[method] = (module_path, class_name)
    
    @classmethod
    def available_methods(cls) -> List[str]:
        """Get list of available analysis methods"""
        return list(cls._analyzers.keys())
```
**Notes**:
- Decouples stage from concrete implementations
- Allows easy addition of new analyzers
- Validates interface implementation

### Phase 6: Analysis Stage Refactoring

#### 6.1 Update Analysis Stage
**File**: `src/pipelines/stages/analysis_stage.py`
```python
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path

from .base_stage import PipelineStage, StageResult
from .analyzer_factory import AnalyzerFactory
from src.processors.data_preparation.analysis_data_source import (
    ParquetAnalysisDataset, DatabaseAnalysisDataset
)
from src.infrastructure.logging import get_logger
from src.base.dataset import BaseDataset

class AnalysisStage(PipelineStage):
    """Refactored analysis stage with proper abstractions"""
    
    def __init__(self, analysis_method: str = 'som'):
        super().__init__()
        self.analysis_method = analysis_method.lower()
        self.logger = get_logger(__name__)  # Structured logging
        self._analyzer = None
        self._validate_method()
    
    def _validate_method(self):
        """Validate analysis method is supported"""
        available = AnalyzerFactory.available_methods()
        if self.analysis_method not in available:
            raise ValueError(
                f"Unsupported analysis method: {self.analysis_method}. "
                f"Available: {available}"
            )
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate stage configuration"""
        errors = []
        
        # Validate factory can create analyzer
        try:
            # Don't actually create it yet, just check
            available = AnalyzerFactory.available_methods()
            if self.analysis_method not in available:
                errors.append(f"Analysis method '{self.analysis_method}' not available")
        except Exception as e:
            errors.append(f"Factory validation failed: {e}")
        
        return len(errors) == 0, errors
    
    def _load_dataset(self, context) -> BaseDataset:
        """Load dataset using appropriate source"""
        data_source = context.config.get('analysis.data_source', 'parquet')
        
        if data_source == 'parquet':
            parquet_path = Path(context.get('ml_ready_path'))
            if not parquet_path or not parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            
            return ParquetAnalysisDataset(
                parquet_path,
                chunk_size=context.config.get('analysis.chunk_size', 10000)
            )
        
        elif data_source == 'database':
            return DatabaseAnalysisDataset(
                context.db,
                experiment_id=context.experiment_id,
                chunk_size=context.config.get('analysis.chunk_size', 10000)
            )
        
        else:
            raise ValueError(f"Unknown data source: {data_source}")
    
    def _setup_progress_tracking(self, context):
        """Set up progress tracking integration"""
        if hasattr(context, 'progress_tracker') and context.progress_tracker:
            # Create progress node
            context.progress_tracker.create_node(
                node_id=f"analysis/{self.analysis_method}",
                parent_id="analysis",
                level="step",
                name=f"{self.analysis_method} analysis",
                total_units=100
            )
            
            # Create callback
            def progress_callback(current: int, total: int, message: str):
                context.progress_tracker.update_progress(
                    node_id=f"analysis/{self.analysis_method}",
                    completed_units=int((current / total) * 100),
                    status="running",
                    metadata={"message": message, "current": current, "total": total}
                )
                
                # Also log for debugging
                self.logger.debug(f"Analysis progress: {message} ({current}/{total})")
            
            self._analyzer.set_progress_callback(progress_callback)
    
    def _get_analysis_params(self, context) -> Dict[str, Any]:
        """Get analysis parameters from config"""
        # Base parameters from analysis config
        base_config = context.config.get(f'{self.analysis_method}_analysis', {})
        
        # Method-specific parameter extraction
        if self.analysis_method == 'som':
            return {
                'grid_size': base_config.get('default_grid_size', [8, 8]),
                'iterations': base_config.get('iterations', 1000),
                'sigma': base_config.get('sigma', 1.5),
                'learning_rate': base_config.get('learning_rate', 0.5),
                'neighborhood_function': base_config.get('neighborhood_function', 'gaussian'),
                'random_seed': base_config.get('random_seed', 42)
            }
        
        elif self.analysis_method == 'maxp_regions':
            return {
                'n_regions': base_config.get('n_regions', 10),
                'min_region_size': base_config.get('min_region_size', 5),
                'method': base_config.get('method', 'ward'),
                'spatial_weights': base_config.get('spatial_weights', 'queen'),
                'random_seed': base_config.get('random_seed', 42)
            }
        
        elif self.analysis_method == 'gwpca':
            return {
                'n_components': base_config.get('n_components', 3),
                'bandwidth': base_config.get('bandwidth', 'adaptive'),
                'kernel': base_config.get('kernel', 'gaussian'),
                'adaptive_bw': base_config.get('adaptive_bw', 50)
            }
        
        return {}
    
    def execute(self, context) -> StageResult:
        """Execute analysis with full integration"""
        self.logger.info(
            f"Starting {self.analysis_method} analysis",
            extra={
                'experiment_id': context.experiment_id,
                'stage': self.name,
                'method': self.analysis_method
            }
        )
        
        try:
            # Step 1: Load dataset
            self.logger.debug("Loading dataset")
            dataset = self._load_dataset(context)
            dataset_info = dataset.load_info()
            
            self.logger.info(
                f"Dataset loaded: {dataset_info.record_count:,} records, "
                f"{dataset_info.size_mb:.2f} MB"
            )
            
            # Step 2: Create analyzer
            self.logger.debug("Creating analyzer via factory")
            self._analyzer = AnalyzerFactory.create(
                self.analysis_method,
                context.config,
                context.db
            )
            
            # Step 3: Set up progress tracking
            self._setup_progress_tracking(context)
            
            # Step 4: Get parameters
            params = self._get_analysis_params(context)
            self.logger.debug(f"Analysis parameters: {params}")
            
            # Step 5: Perform analysis
            self.logger.info("Starting analysis computation")
            results = self._analyzer.analyze(dataset, **params)
            
            # Step 6: Save results if configured
            saved_path = None
            if context.config.get('analysis.save_results.enabled', True):
                output_dir = context.output_dir / 'analysis' / self.analysis_method
                saved_path = self._analyzer.save_results(
                    results,
                    f"{self.analysis_method}_{context.experiment_id}",
                    output_dir
                )
                context.set(f'{self.analysis_method}_output_path', str(saved_path))
                
                self.logger.info(f"Results saved to: {saved_path}")
            
            # Step 7: Store in context if configured
            if context.config.get('analysis.keep_results_in_memory', False):
                context.set(f'{self.analysis_method}_results', results)
            
            # Step 8: Extract metrics
            metrics = self._extract_metrics(results, params)
            if saved_path:
                metrics['output_path'] = str(saved_path)
            
            # Step 9: Update progress to complete
            if hasattr(context, 'progress_tracker') and context.progress_tracker:
                context.progress_tracker.update_progress(
                    node_id=f"analysis/{self.analysis_method}",
                    completed_units=100,
                    status="completed"
                )
            
            self.logger.info(
                f"Analysis completed successfully",
                extra={'metrics': metrics}
            )
            
            return StageResult(
                success=True,
                data={
                    'analysis_method': self.analysis_method,
                    'output_path': str(saved_path) if saved_path else None
                },
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(
                f"{self.analysis_method} analysis failed: {e}",
                exc_info=True,
                extra={
                    'experiment_id': context.experiment_id,
                    'stage': self.name,
                    'error_type': type(e).__name__
                }
            )
            
            # Update progress to failed
            if hasattr(context, 'progress_tracker') and context.progress_tracker:
                context.progress_tracker.update_progress(
                    node_id=f"analysis/{self.analysis_method}",
                    completed_units=0,
                    status="failed",
                    metadata={"error": str(e)}
                )
            
            raise
    
    def cleanup(self, context):
        """Clean up resources after execution"""
        self.logger.debug("Starting cleanup")
        
        # Clean up analyzer
        if self._analyzer:
            if hasattr(self._analyzer, 'cleanup'):
                try:
                    self._analyzer.cleanup()
                except Exception as e:
                    self.logger.warning(f"Analyzer cleanup failed: {e}")
            self._analyzer = None
        
        # Remove large data from context
        if context.config.get('analysis.memory.cleanup_after_stage', True):
            keys_to_remove = [
                f'{self.analysis_method}_results',
                'merged_dataset',
                'spatial_coordinates',
                'exported_csv_path'  # If loaded from CSV
            ]
            
            removed_count = 0
            for key in keys_to_remove:
                if key in context.shared_data:
                    del context.shared_data[key]
                    removed_count += 1
            
            if removed_count > 0:
                self.logger.debug(f"Removed {removed_count} items from context")
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        
        self.logger.info(
            f"Cleanup completed",
            extra={
                'stage': self.name,
                'gc_collected': collected
            }
        )
    
    def _extract_metrics(self, results: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from results"""
        # Import here to avoid circular dependency
        from src.pipelines.unified_resampling.pipeline_orchestrator import clean_nan_for_json
        
        metrics = {
            'analysis_method': self.analysis_method,
            'parameters': params
        }
        
        # Method-specific metrics extraction
        if hasattr(results, 'metadata'):
            metrics['metadata'] = {
                'processing_time': results.metadata.processing_time,
                'input_shape': results.metadata.input_shape,
                'timestamp': results.metadata.timestamp
            }
        
        if hasattr(results, 'statistics'):
            metrics['statistics'] = clean_nan_for_json(results.statistics)
        
        return metrics
```
**Notes**:
- Complete rewrite with proper abstractions
- Full monitoring integration
- Proper error handling and cleanup
- Configuration-driven behavior

### Phase 7: Monitoring Integration

#### 7.1 Update Analyzers for Progress
**File**: Each analyzer in `src/spatial_analysis/`
```python
# In each analyzer's analyze method:
self._update_progress(1, 5, "Loading data")
# ... operation ...
self._update_progress(2, 5, "Initializing algorithm")
# ... operation ...
self._update_progress(3, 5, "Training model")
# ... operation ...
self._update_progress(4, 5, "Computing results")
# ... operation ...
self._update_progress(5, 5, "Finalizing")
```
**Notes**:
- Use inherited _update_progress method
- Consistent progress points across analyzers
- Meaningful progress messages

### Phase 8: Memory Management

#### 8.1 Add Memory Pressure Response
**File**: `src/pipelines/stages/analysis_stage.py`
```python
# Add to AnalysisStage.__init__:
self._original_chunk_size = None

# Add method:
def _handle_memory_pressure(self, context, pressure_level: str):
    """Respond to memory pressure"""
    if pressure_level == 'warning':
        # Reduce chunk size
        current_chunk = context.config.get('analysis.chunk_size', 10000)
        new_chunk = max(1000, current_chunk // 2)
        context.config.set('analysis.chunk_size', new_chunk)
        self.logger.warning(f"Memory pressure: reduced chunk size to {new_chunk}")
    
    elif pressure_level == 'critical':
        # Trigger cleanup and further reduction
        self.cleanup(context)
        context.config.set('analysis.chunk_size', 1000)
        self.logger.error("Critical memory pressure: forced cleanup and minimum chunk size")
```

#### 8.2 Register Memory Callbacks
**File**: `src/pipelines/stages/analysis_stage.py`
```python
# In execute method, after creating analyzer:
if hasattr(context, 'memory_monitor'):
    context.memory_monitor.register_warning_callback(
        lambda usage: self._handle_memory_pressure(context, 'warning')
    )
    context.memory_monitor.register_critical_callback(
        lambda usage: self._handle_memory_pressure(context, 'critical')
    )
```

### Phase 9: Testing Strategy

#### 9.1 Unit Tests
**File**: `tests/test_analysis_stage.py`
```python
# Test each component in isolation:
- Test AnalyzerFactory creation
- Test dataset loading
- Test progress callbacks
- Test cleanup
```

#### 9.2 Integration Tests
**File**: `tests/integration/test_analysis_pipeline.py`
```python
# Test full pipeline flow:
- Test parquet → analysis → results
- Test database → analysis → results
- Test memory pressure handling
- Test monitoring integration
```

### Phase 10: Documentation

#### 10.1 Update Developer Documentation
**File**: `docs/analysis_stage.md`
- Document new architecture
- Explain data flow
- Provide examples

#### 10.2 Update User Configuration Guide
**File**: `docs/configuration.md`
- Document new config options
- Explain memory settings
- Provide tuning guidelines

## Validation Checklist

Before considering the update complete:

- [ ] All analyzers use consistent constructors
- [ ] Factory successfully creates all analyzer types
- [ ] Progress tracking works in UI/monitoring tools
- [ ] Memory cleanup reduces memory usage
- [ ] Large datasets can be processed without OOM
- [ ] Configuration changes are reflected in behavior
- [ ] Structured logs appear in database
- [ ] No direct imports between layers (respect hierarchy)
- [ ] All methods respect their abstractions
- [ ] Tests pass for all components

## Rollback Plan

If issues arise:

1. **Stage 1**: Revert analysis_stage.py to previous version
2. **Stage 2**: Revert analyzer changes, keep interface updates
3. **Stage 3**: Full revert, document issues for next iteration

## Success Metrics

- **Memory usage**: Peak memory < configured limit
- **Performance**: < 10% overhead from new abstractions
- **Reliability**: Zero OOM errors on test datasets
- **Maintainability**: New analyzers can be added without modifying stage
- **Monitoring**: 100% of operations tracked in database

## Notes

- This plan respects the two golden rules throughout
- Each phase builds on the previous one
- Testing happens continuously, not just at the end
- Rollback is possible at each phase
- Documentation is updated alongside code