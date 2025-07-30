# Machine Learning Module

## Overview

The Machine Learning module provides spatial-aware ML capabilities for biodiversity prediction and analysis. It integrates seamlessly with the geospatial processing pipeline while maintaining modularity.

## Quick Start

```python
from src.machine_learning import (
    LinearRegressionAnalyzer,
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    SpatialBlockCV
)

# Load your data
data = pd.read_parquet('biodiversity_data.parquet')

# Handle missing values
imputer = SpatialKNNImputer(n_neighbors=10)
data_clean = imputer.fit_transform(data)

# Engineer features automatically
feature_builder = CompositeFeatureBuilder(config)
features = feature_builder.fit_transform(data_clean)

# Prepare for modeling
target = features['total_richness']
X = features.drop(['total_richness'], axis=1)

# Train with spatial cross-validation
cv = SpatialBlockCV(n_splits=5, block_size=100)
model = LinearRegressionAnalyzer(config)

cv_results = model.cross_validate(X, target, cv_strategy=cv)
print(f"CV R²: {cv_results.mean_metrics['r2']:.3f}")

# Train final model
model.fit(X, target)
model.save_model('biodiversity_model.pkl')
```

## Components

### Models
- **LinearRegressionAnalyzer**: Ridge regression with automatic scaling
- **LightGBMAnalyzer**: Gradient boosting (optional, if LightGBM installed)

### Feature Engineering
- **RichnessFeatureBuilder**: Biodiversity-specific features (diversity indices, ratios)
- **SpatialFeatureBuilder**: Geographic features (distance to equator, spatial bins)
- **EcologicalFeatureBuilder**: Extensible framework for environmental data
- **CompositeFeatureBuilder**: Automatically combines all registered builders

### Imputation
- **SpatialKNNImputer**: Geographic distance-weighted imputation

### Cross-Validation
- **SpatialBlockCV**: Checkerboard blocking for spatial independence
- **SpatialBufferCV**: Buffer zones around test points
- **EnvironmentalBlockCV**: Stratification by environmental gradients

## Pipeline Integration

The ML stage operates independently from the data processing pipeline:

```python
from src.pipelines.stages.ml_stage import MLStage

# ML stage reads directly from parquet files
ml_stage = MLStage(ml_config={
    'input_parquet': '/path/to/biodiversity_data.parquet',  # Required
    'model_type': 'lightgbm',
    'target_column': 'total_richness',
    'cv_strategy': {
        'type': 'spatial_block',
        'n_splits': 5,
        'block_size': 100
    }
})

# Run as standalone pipeline
ml_orchestrator = PipelineOrchestrator(config, db)
ml_orchestrator.register_stage(ml_stage)
ml_orchestrator.run(experiment_name="ml_analysis")
```

### Standalone Usage

Run ML analysis on any parquet file:

```bash
python examples/ml_standalone.py /path/to/data.parquet --model lightgbm --target total_richness
```

## Utility Scripts

- `scripts/ml_train.py`: Train models with various configurations
- `scripts/ml_predict.py`: Make predictions with trained models
- `scripts/ml_compare_models.py`: Compare multiple models
- `scripts/ml_benchmark.py`: Performance benchmarking

## Examples

See the `examples/` directory for complete examples:
- `ml_example.py`: Basic ML workflow
- `ml_advanced_example.py`: Advanced features with visualization
- `pipeline_with_ml.py`: ML stage in full pipeline

## Configuration

Configure ML settings in `src/config/defaults.py`:

```python
MACHINE_LEARNING = {
    'models': {
        'linear_regression': {'alpha': 1.0},
        'lightgbm': {'num_leaves': 31, 'learning_rate': 0.05}
    },
    'feature_engineering': {
        'richness': {'calculate_diversity': True},
        'spatial': {'polynomial_degree': 2}
    }
}
```

## Testing

Run tests with pytest:

```bash
pytest tests/machine_learning/
```

## Documentation

See [ML_MODULE_DOCUMENTATION.md](../../ML_MODULE_DOCUMENTATION.md) for comprehensive documentation.