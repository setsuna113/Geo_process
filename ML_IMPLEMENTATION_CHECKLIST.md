# Comprehensive ML Module Implementation Checklist (Revised)

## Phase 1: Foundation Layer Integration

### 1.1 Abstractions Layer Setup
- [ ] Create `src/abstractions/interfaces/ml_analyzer.py`:
  - `IMLAnalyzer` interface extending `IAnalyzer`
  - Add `fit()`, `predict()`, `evaluate()` methods
  - Define model persistence contract
- [ ] Create `src/abstractions/interfaces/feature_builder.py`:
  - `IFeatureBuilder` interface
  - `build_features()` method contract
  - `get_feature_names()` method contract
  - `get_required_columns()` method contract
- [ ] Create `src/abstractions/types/ml_types.py`:
  - `MLResult` dataclass extending `AnalysisResult`
  - `ModelMetadata` dataclass
  - `FeatureMetadata` dataclass
  - `CVFold` and `CVResult` types
  - `ImputationStrategy` enum

### 1.2 Core Registry Extensions
- [ ] Extend `src/core/registry.py`:
  - Add `ml_models` registry to `ComponentRegistry.__init__`
  - Add `feature_builders` registry
  - Add `imputation_strategies` registry
  - Add `cv_strategies` registry
- [ ] Create ML-specific validators:
  - `_validate_ml_model_class()` - check for fit/predict methods
  - `_validate_feature_builder_class()` - check for build_features method
  - `_validate_cv_strategy_class()` - check for split method
- [ ] Add ML convenience registration functions:
  - `register_ml_model()` with ML-specific metadata
  - `register_feature_builder()` with feature categories
  - `register_cv_strategy()` with spatial awareness flags
- [ ] Add ML discovery methods:
  - `find_models_for_task()` - find models by task type
  - `find_feature_builders_by_category()` - find by feature type
  - `find_spatial_cv_strategies()` - find spatial-aware CV

### 1.3 Base Layer Implementation
- [ ] Create `src/base/ml_analyzer.py`:
  - `BaseMLAnalyzer` extending `BaseAnalyzer`
  - Implement `analyze()` to wrap fit/predict workflow
  - Add `fit()`, `predict()`, `fit_predict()` abstract methods
  - Add model persistence methods (`save_model()`, `load_model()`)
  - Include feature importance extraction
  - Support for both database and parquet I/O
- [ ] Create `src/base/feature_builder.py`:
  - `BaseFeatureBuilder` abstract class
  - `build_features()` abstract method
  - `get_feature_names()` abstract method
  - `get_required_columns()` abstract method
  - Metadata tracking for feature lineage
  - Integration with progress tracking

## Phase 2: ML Module Implementation

### 2.1 Module Structure (Already Created)
- [x] Create `src/machine_learning/` directory structure:
  ```
  src/machine_learning/
  ├── __init__.py
  ├── models/
  ├── preprocessing/
  ├── validation/
  └── utils/
  ```

### 2.2 ML Models Implementation
- [ ] Implement `machine_learning/models/linear_regression.py`:
  - `LinearRegressionAnalyzer` extending `BaseMLAnalyzer`
  - Ridge regression by default (with alpha parameter)
  - Automatic feature scaling integration
  - Register with `@ml_model` decorator
  - Coefficient interpretation helpers
- [ ] Implement `machine_learning/models/lightgbm_regressor.py`:
  - `LightGBMAnalyzer` extending `BaseMLAnalyzer`
  - Native missing value handling
  - Regularization parameters (reg_alpha, reg_lambda)
  - Early stopping with validation set
  - Feature importance extraction
  - Register with `@ml_model` decorator

### 2.3 Feature Engineering System
- [ ] Create `machine_learning/preprocessing/feature_engineering/`:
  - Directory for feature builders
- [ ] Implement `richness_features.py`:
  - `RichnessFeatureBuilder` extending `BaseFeatureBuilder`
  - Total richness calculation
  - Richness ratios (plant/vertebrate, etc.)
  - Richness diversity metrics
  - Log transformations for skewed distributions
  - Register with `@feature_builder` decorator
- [ ] Implement `spatial_features.py`:
  - `SpatialFeatureBuilder` extending `BaseFeatureBuilder`
  - Distance to equator
  - Latitude squared/cubed (polynomial features)
  - Distance to tropics
  - Spatial binning features
  - Grid cell area adjustments
  - Register with `@feature_builder` decorator
- [ ] Implement `ecological_features.py`:
  - `EcologicalFeatureBuilder` extending `BaseFeatureBuilder`
  - Temperature × Precipitation interactions (when available)
  - Seasonality metrics
  - Environmental gradients
  - Placeholder for future NDVI, climate data
  - Register with `@feature_builder` decorator
- [ ] Create `composite_feature_builder.py`:
  - `CompositeFeatureBuilder` that combines multiple builders
  - Handle feature name conflicts
  - Track feature provenance

### 2.4 Imputation Strategies
- [ ] Create `machine_learning/preprocessing/imputation/`:
  - Directory for imputation strategies
- [ ] Implement `knn_imputer.py`:
  - `SpatialKNNImputer` wrapper
  - Spatial-aware KNN (uses lat/lon distance)
  - Configurable neighbor count
  - Missing indicator generation
  - Register with `@imputation_strategy` decorator
- [ ] Implement `iterative_imputer.py`:
  - `SpatialIterativeImputer` wrapper (MICE)
  - Multiple imputation rounds
  - Convergence monitoring
  - Uncertainty estimation
  - Register with `@imputation_strategy` decorator
- [ ] Create `imputation_selector.py`:
  - Auto-select based on missing data percentage
  - Different strategies for different feature types
  - Never use zero-fill for ecological data

### 2.5 Data Validation
- [ ] Implement `machine_learning/preprocessing/data_validation.py`:
  - Collinearity checks (VIF calculation)
  - Correlation matrix analysis
  - Auto-removal of highly correlated features
  - Scale validation and warnings
  - Outlier detection (spatial and statistical)

### 2.6 Spatial Cross-Validation
- [ ] Implement `machine_learning/validation/spatial_cv.py`:
  - `SpatialBlockCV` class
  - Configurable block size (default 100km)
  - Checkerboard pattern for fold assignment
  - Buffer zones between train/test blocks
  - Register with `@cv_strategy` decorator
- [ ] Implement `machine_learning/validation/environmental_cv.py`:
  - `EnvironmentalBlockCV` class
  - Stratification by latitude bands
  - Environmental gradient blocking
  - Register with `@cv_strategy` decorator
- [ ] Implement `machine_learning/validation/buffer_cv.py`:
  - `SpatialBufferCV` class
  - Create buffer zones around test points
  - Configurable buffer distance
  - Register with `@cv_strategy` decorator

### 2.7 Model Evaluation
- [ ] Implement `machine_learning/validation/model_evaluation.py`:
  - Spatial performance metrics
  - Per-fold performance tracking
  - Confidence interval estimation
  - Spatial autocorrelation testing
  - Visualization utilities

### 2.8 ML Utilities
- [ ] Create `machine_learning/utils/ml_config.py`:
  - ML-specific configuration handling
  - Parameter validation
  - Default parameter management
- [ ] Create `machine_learning/utils/data_loader.py`:
  - Parquet file loader (primary)
  - Database query support (secondary)
  - Memory-efficient streaming
  - Integration with existing data sources

## Phase 3: Integration & Configuration

### 3.1 Configuration Extensions
- [ ] Add ML configuration section to `src/config/defaults.py`:
  ```python
  MACHINE_LEARNING = {
      'models': {
          'linear_regression': {
              'alpha': 1.0,
              'fit_intercept': True,
              'normalize': False
          },
          'lightgbm': {
              'num_leaves': 31,
              'learning_rate': 0.1,
              'n_estimators': 100,
              'reg_alpha': 0.0,
              'reg_lambda': 0.0
          }
      },
      'preprocessing': {
          'imputation': {
              'strategy': 'auto',  # 'auto', 'knn', 'iterative'
              'knn_neighbors': 5,
              'missing_threshold': 0.3
          },
          'scaling': {
              'method': 'standard',  # 'standard', 'robust', 'minmax'
              'clip_outliers': True
          }
      },
      'validation': {
          'cv_folds': 5,
          'spatial_block_size_km': 100,
          'buffer_distance_km': 50,
          'stratify_by': 'latitude'
      },
      'output': {
          'save_predictions': True,
          'save_model': True,
          'formats': ['parquet', 'pickle']
      }
  }
  ```

### 3.2 Pipeline Integration
- [ ] Create `src/pipelines/stages/ml_stage.py`:
  - `MLStage` extending `BaseStage`
  - Read from export stage parquet files
  - Configurable model selection via registry
  - Automatic feature engineering pipeline
  - Spatial CV integration
  - Results to parquet/database
- [ ] Update `src/pipelines/orchestrator.py`:
  - Add ML stage to pipeline options
  - Configure stage dependencies
  - Add ML-specific skip conditions

### 3.3 Experiment Tracking Integration
- [ ] Extend database schema for ML experiments:
  - Add ML-specific metadata fields
  - Track hyperparameters
  - Store performance metrics
  - Link to input data versions
- [ ] Create ML experiment tracking utilities:
  - Log model parameters
  - Track feature importance
  - Store CV fold results
  - Compare model performance

## Phase 4: Testing & Documentation

### 4.1 Unit Tests
- [ ] Test ML interfaces and types
- [ ] Test base ML classes
- [ ] Test registry integration
- [ ] Test feature builders with mock data
- [ ] Test imputation strategies
- [ ] Test spatial CV fold generation
- [ ] Test model training/prediction
- [ ] Test data validation and safeguards

### 4.2 Integration Tests
- [ ] End-to-end ML pipeline test
- [ ] Test with real biodiversity data subset
- [ ] Verify parquet I/O integration
- [ ] Test memory limits and chunking
- [ ] Test checkpoint/resume functionality
- [ ] Test registry discovery functions

### 4.3 Documentation
- [ ] API documentation for all ML classes
- [ ] Usage examples for common workflows
- [ ] Configuration guide for ML settings
- [ ] Best practices for biodiversity ML
- [ ] Integration guide with existing pipeline

## Implementation Order

### Critical Path (Must have for basic functionality):
1. Phase 1.1-1.3: Foundation setup (interfaces, registry, base classes)
2. Phase 2.2: Basic models (Linear Regression, LightGBM)
3. Phase 2.4: Basic imputation
4. Phase 2.6: Basic spatial CV
5. Phase 3.2: Pipeline integration

### Important Enhancements:
1. Phase 2.3: Feature engineering
2. Phase 2.5: Data validation
3. Phase 2.7: Model evaluation
4. Phase 3.1: Configuration
5. Phase 3.3: Experiment tracking

### Nice to Have:
1. Phase 2.8: Advanced utilities
2. Phase 4: Testing and documentation

## Key Architecture Decisions

### Module Structure
- **Hybrid approach**: Low-level integration (base, abstractions) with standalone high-level module
- **Registry-driven**: All components discovered via registry
- **Consistent patterns**: ML analyzers follow same patterns as spatial analyzers

### Integration Points
- **BaseMLAnalyzer** extends `BaseAnalyzer` for consistency
- **Registry** in `core/registry.py` manages all ML components
- **Pipeline** integration via standard `BaseStage`
- **Configuration** extends existing `defaults.py`

### Data Flow
- **Primary input**: Parquet files from export stage
- **Feature engineering**: Modular, composable builders
- **Validation**: Always spatial-aware CV
- **Output**: Parquet files and model artifacts

This revised plan emphasizes the hybrid architecture with proper integration sequence.