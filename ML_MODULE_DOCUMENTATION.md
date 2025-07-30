# Machine Learning Module Documentation

## Overview

The Machine Learning (ML) module provides a comprehensive framework for biodiversity prediction and analysis, with special emphasis on spatial-aware techniques. It integrates seamlessly with the existing geospatial processing pipeline while maintaining modularity and extensibility.

## Architecture

### Hybrid Design Pattern

The ML module follows a hybrid architecture that balances integration with standalone functionality:

- **Low-level Integration**: Core interfaces in `abstractions/`, base classes in `base/`, and registry integration in `core/`
- **High-level Standalone**: ML-specific implementations in the dedicated `src/machine_learning/` module

This design ensures compatibility with the existing system while allowing ML components to evolve independently.

## Core Components

### 1. Models (`src/machine_learning/models/`)

#### LinearRegressionAnalyzer
Ridge regression with L2 regularization, suitable for baseline modeling and interpretability.

```python
from src.machine_learning import LinearRegressionAnalyzer
from src.config import config

model = LinearRegressionAnalyzer(config)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Key Features:**
- Automatic feature scaling
- Ridge regularization (alpha=1.0 default)
- Feature importance via coefficients
- Model persistence support

#### LightGBMAnalyzer (Optional)
Gradient boosting model for high-performance predictions.

```python
from src.machine_learning import LightGBMAnalyzer

model = LightGBMAnalyzer(config)
model.fit(X_train, y_train)
```

**Key Features:**
- Native missing value handling
- Feature importance calculation
- Early stopping support
- Optimized for large datasets

### 2. Feature Engineering (`src/machine_learning/preprocessing/feature_engineering/`)

#### RichnessFeatureBuilder
Creates biodiversity-specific features from species richness data.

```python
from src.machine_learning import RichnessFeatureBuilder

builder = RichnessFeatureBuilder(config)
features = builder.fit_transform(data)
```

**Generated Features:**
- Total richness (sum of all richness columns)
- Richness ratios (e.g., plants/total)
- Diversity indices (Shannon, Simpson)
- Log-transformed richness values
- Richness anomalies (z-scores)

#### SpatialFeatureBuilder
Generates geographic and spatial features.

```python
from src.machine_learning import SpatialFeatureBuilder

builder = SpatialFeatureBuilder(config)
features = builder.fit_transform(data)
```

**Generated Features:**
- Distance to equator
- Coordinate polynomials (lat², lon², lat×lon)
- Spatial bins (configurable grid size)
- Hemisphere indicators
- Coordinate interactions

#### EcologicalFeatureBuilder
Extensible framework for environmental features (placeholder for future data sources).

```python
from src.machine_learning import EcologicalFeatureBuilder

builder = EcologicalFeatureBuilder(config)
# Add custom data source
builder.add_data_source('climate', climate_config)
features = builder.fit_transform(data)
```

#### CompositeFeatureBuilder
Automatically combines all registered feature builders.

```python
from src.machine_learning import CompositeFeatureBuilder

builder = CompositeFeatureBuilder(config)
all_features = builder.fit_transform(data)

# Get feature summary
summary = builder.get_feature_summary()
```

### 3. Missing Value Imputation (`src/machine_learning/preprocessing/imputation/`)

#### SpatialKNNImputer
Geographic distance-weighted K-nearest neighbors imputation.

```python
from src.machine_learning import SpatialKNNImputer

imputer = SpatialKNNImputer(
    n_neighbors=10,
    spatial_weight=0.6  # Weight for spatial distance vs feature distance
)
data_imputed = imputer.fit_transform(data)
```

**Key Features:**
- Haversine distance for geographic coordinates
- Configurable spatial weighting
- Preserves spatial autocorrelation
- Handles multiple missing patterns

### 4. Cross-Validation Strategies (`src/machine_learning/validation/`)

#### SpatialBlockCV
Checkerboard pattern blocking for spatial independence.

```python
from src.machine_learning import SpatialBlockCV

cv = SpatialBlockCV(
    n_splits=5,
    block_size=100,  # km
    random_state=42
)

for train_idx, test_idx in cv.split(data):
    # Train and evaluate
    pass
```

#### SpatialBufferCV
Buffer zones around test points to ensure spatial separation.

```python
from src.machine_learning import SpatialBufferCV

cv = SpatialBufferCV(
    n_splits=5,
    buffer_distance=50,  # km
    random_state=42
)
```

#### EnvironmentalBlockCV
Stratification based on environmental gradients.

```python
from src.machine_learning import EnvironmentalBlockCV

cv = EnvironmentalBlockCV(
    n_splits=5,
    stratify_by='latitude'  # or custom environmental variable
)
```

## Pipeline Integration

### MLStage (Standalone)
The ML module operates as a standalone stage that reads from parquet files, making it independent of the data processing pipeline.

```python
from src.pipelines.stages.ml_stage import MLStage

ml_stage = MLStage(ml_config={
    'input_parquet': '/path/to/biodiversity_data.parquet',  # Required
    'model_type': 'lightgbm',
    'target_column': 'total_richness',
    'cv_strategy': {
        'type': 'spatial_block',
        'n_splits': 5,
        'block_size': 100
    },
    'imputation_strategy': {
        'type': 'spatial_knn',
        'n_neighbors': 10
    }
})
```

**Key Points:**
- No dependencies on other pipeline stages
- Explicitly configured with input parquet path
- Can run on any biodiversity parquet file
- Enables multiple ML experiments on same data

## Configuration

ML settings are configured in `src/config/defaults.py`:

```python
MACHINE_LEARNING = {
    'models': {
        'linear_regression': {
            'alpha': 1.0,
            'fit_intercept': True,
            'max_iter': 1000
        },
        'lightgbm': {
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'min_child_samples': 20
        }
    },
    'feature_engineering': {
        'richness': {
            'calculate_ratios': True,
            'calculate_diversity': True,
            'log_transform': True
        },
        'spatial': {
            'polynomial_degree': 2,
            'include_bins': True,
            'bin_size': 10
        }
    },
    'validation': {
        'spatial_block_size_km': 100,
        'buffer_distance_km': 50,
        'cross_validation_folds': 5
    }
}
```

## Usage Examples

### Basic Workflow

```python
import pandas as pd
from src.machine_learning import (
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    LinearRegressionAnalyzer,
    SpatialBlockCV
)
from src.config import config

# Load data
data = pd.read_parquet('biodiversity_data.parquet')

# Handle missing values
imputer = SpatialKNNImputer(n_neighbors=10)
data_clean = imputer.fit_transform(data)

# Engineer features
feature_builder = CompositeFeatureBuilder(config)
features = feature_builder.fit_transform(data_clean)

# Prepare for modeling
target = features['total_richness']
X = features.drop(['total_richness'], axis=1)

# Cross-validation
cv = SpatialBlockCV(n_splits=5)
model = LinearRegressionAnalyzer(config)

cv_results = model.cross_validate(X, target, cv_strategy=cv)
print(f"CV R²: {cv_results.mean_metrics['r2']:.3f}")

# Train final model
model.fit(X, target)
model.save_model('biodiversity_model.pkl')
```

### Advanced Model Comparison

```python
from src.machine_learning import LinearRegressionAnalyzer, LightGBMAnalyzer

models = {
    'Ridge': LinearRegressionAnalyzer(config),
    'LightGBM': LightGBMAnalyzer(config)
}

results = {}
for name, model in models.items():
    cv_result = model.cross_validate(X, target, cv_strategy=cv)
    results[name] = cv_result.mean_metrics['r2']
    
best_model = max(results.items(), key=lambda x: x[1])
print(f"Best model: {best_model[0]} (R²: {best_model[1]:.3f})")
```

## Registry System

The ML module uses a decorator-based registry system for dynamic component discovery:

```python
from src.core.registry import ml_model, feature_builder

@ml_model(model_type='my_custom_model', requires_scaling=True)
class MyCustomModel:
    # Implementation
    pass

@feature_builder('environmental', required_columns={'temperature', 'precipitation'})
class ClimateFeatureBuilder:
    # Implementation
    pass
```

## Best Practices

1. **Always use spatial CV** for geographic data to avoid overfitting to spatial autocorrelation
2. **Impute missing values** before feature engineering to maximize data utilization
3. **Use CompositeFeatureBuilder** to automatically leverage all available feature builders
4. **Monitor feature importance** to understand model behavior and identify key predictors
5. **Save models with metadata** for reproducibility and deployment

## Extending the Module

### Adding New Models

1. Create a new model class inheriting from `BaseMLAnalyzer`
2. Implement required methods: `_fit()`, `_predict()`, `get_feature_importance()`
3. Register with `@ml_model` decorator

```python
from src.base.ml_analyzer import BaseMLAnalyzer
from src.core.registry import ml_model

@ml_model(model_type='random_forest', requires_scaling=False)
class RandomForestAnalyzer(BaseMLAnalyzer):
    def _fit(self, X, y, sample_weight=None):
        # Implementation
        pass
```

### Adding New Feature Builders

1. Create a new builder inheriting from `BaseFeatureBuilder`
2. Implement `_engineer_features()` method
3. Register with `@feature_builder` decorator

```python
from src.base.feature_builder import BaseFeatureBuilder
from src.core.registry import feature_builder

@feature_builder('climate', required_columns={'temperature'})
class ClimateFeatureBuilder(BaseFeatureBuilder):
    def _engineer_features(self, data):
        # Implementation
        pass
```

## Performance Considerations

1. **Memory Management**: The ML stage automatically handles memory constraints through the pipeline's memory monitoring
2. **Chunked Processing**: For predictions on large datasets, the ML stage supports chunked processing
3. **Feature Selection**: Use feature importance to reduce dimensionality for large feature sets
4. **Model Persistence**: Models are saved in pickle format with metadata for fast loading

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed, especially optional ones like LightGBM
2. **Memory Errors**: Reduce batch size or use chunked processing for large datasets
3. **CV Errors**: Ensure sufficient spatial coverage for spatial CV strategies
4. **Feature Mismatch**: Use the same feature builder configuration for training and prediction

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Deep Learning Models**: Integration with PyTorch/TensorFlow for neural networks
2. **AutoML Capabilities**: Automated hyperparameter tuning and model selection
3. **Ensemble Methods**: Voting and stacking ensembles
4. **Online Learning**: Incremental model updates with new data
5. **Model Monitoring**: Drift detection and performance tracking in production