# Machine Learning Implementation Summary

## Overview
This document summarizes the complete ML implementation for the biodiversity analysis system. The ML module provides both standard predictive modeling and research-oriented analysis capabilities.

## Architecture

### 1. Standalone ML Pipeline
- **Entry Point**: `scripts/run_ml.py`
- **Shell Wrapper**: `run_ml.sh`
- **Design**: Completely independent from main data processing pipeline
- **Data Input**: Reads directly from parquet files (no database dependency)

### 2. Module Structure
```
src/machine_learning/
├── models/               # ML algorithms
│   ├── linear_regression.py
│   ├── lightgbm_regressor.py
│   └── formula_model.py  # R-style formula specification
├── preprocessing/        # Feature engineering & imputation
│   ├── feature_builder.py
│   └── imputers.py
├── evaluation/          # Model evaluation & statistical tests
│   ├── cross_validation.py
│   ├── metrics.py
│   ├── statistical_tests.py
│   └── permutation_importance.py
├── interpretation/      # Model interpretation tools
│   └── pdp.py          # Partial Dependence Plots
└── research/           # Research-oriented workflows
    └── biodiversity_analysis.py
```

## Key Features

### Standard ML Pipeline
1. **Multiple Model Types**: Linear Regression, LightGBM
2. **Spatial-Aware Cross-Validation**: Block, Buffer, Environmental strategies
3. **Feature Engineering**: Composite features, spatial features
4. **Missing Value Imputation**: Spatial KNN imputation
5. **Model Persistence**: Save/load trained models

### Research Extensions
1. **Formula-Based Models**: R-style syntax (e.g., `Y ~ X1 + X2 + X1:X2`)
2. **Nested Model Comparison**: Likelihood Ratio Tests
3. **Permutation Feature Importance**: Robust feature evaluation
4. **2D Partial Dependence Plots**: Interaction visualization
5. **Hypothesis Testing Framework**: Academic research workflows

## Usage Examples

### Standard ML Analysis
```bash
# Run with named experiment from config.yml
python scripts/run_ml.py --experiment test_generated

# Run with custom parameters
python scripts/run_ml.py --input data.parquet --model-type linear_regression --target total_richness
```

### Research Analysis
```bash
# Test biodiversity hypotheses with nested models
python scripts/run_ml.py --analysis-type research --experiment temperate_mismatch

# Custom research analysis
python scripts/run_ml.py --analysis-type research --input biodiversity.parquet
```

## Configuration

### config.yml Structure
```yaml
machine_learning:
  defaults:
    model_type: 'linear_regression'
    target_column: 'total_richness'
    cv_strategy:
      type: 'spatial_block'
      n_splits: 5
  
  experiments:
    test_generated:
      input_parquet: 'outputs/test_biodiversity.parquet'
      model_type: 'linear_regression'
  
  research:
    default_formulas:
      - 'F ~ avg_temp + avg_precip + seasonal_temp'
      - 'F ~ avg_temp + avg_precip + seasonal_temp + P + A'
      - 'F ~ avg_temp + avg_precip + seasonal_temp + P + A + P:seasonal_temp'
    
    experiments:
      temperate_mismatch:
        input_parquet: 'outputs/test_biodiversity.parquet'
        nested_formulas: [...]
```

## Research Capabilities

### Biodiversity Hypothesis Testing
The research pipeline tests two main hypotheses:
1. **Non-proxy relationships**: Do biodiversity components (plants, animals) improve predictions beyond climate?
2. **Temperate mismatch hypothesis**: Is there an interaction between plant richness and temperature seasonality?

### Output Files
- `hypothesis_tests.json`: Statistical test results
- `model_comparison_summary.csv`: Nested model comparisons
- `permutation_importance.csv/png`: Feature importance analysis
- `pdp_2d_*.png`: 2D interaction visualizations
- `analysis_report.txt`: Complete analysis summary

## Technical Notes

### Database Independence
- Uses `SKIP_DB_INIT=true` environment variable
- Reads data directly from parquet files
- No dependency on PostgreSQL/PostGIS for ML operations

### Extensibility
- Registry-based component discovery
- Abstract base classes for models and preprocessing
- Plugin architecture for new algorithms

## Future Enhancements
1. Deep learning models (neural networks)
2. Ensemble methods beyond gradient boosting
3. Spatial autoregressive models
4. Time series forecasting capabilities
5. Automated hyperparameter tuning
6. Model deployment utilities

## Dependencies
- Core: numpy, pandas, scikit-learn, scipy
- Optional: lightgbm, matplotlib
- Spatial: already included in geo environment

This implementation provides a complete ML framework for biodiversity analysis, supporting both practical predictive modeling and academic research workflows.