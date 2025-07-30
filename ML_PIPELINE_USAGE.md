# ML Pipeline Usage Guide

## Overview

The ML pipeline is a standalone system for running machine learning experiments on biodiversity data. It operates independently from the data processing pipeline, reading from parquet files and producing model artifacts and predictions.

## Quick Start

### 1. Basic Usage

Run a predefined experiment from config.yml:
```bash
./run_ml.sh --experiment test_linear
```

Run with a custom parquet file:
```bash
./run_ml.sh --input outputs/biodiversity_data.parquet --experiment test_linear
```

### 2. With Monitoring

Run with tmux monitoring (recommended for long runs):
```bash
./run_ml.sh --experiment production_lgb --monitor
```

This creates a tmux session with:
- Main pane: ML pipeline execution
- Log pane: Real-time logs
- Status pane: Pipeline status
- Resource pane: System resources (htop)

### 3. Daemon Mode

Run as a background daemon:
```bash
./run_ml.sh --experiment production_lgb --daemon --process-name ml_prod_v1
```

Check status:
```bash
python scripts/process_manager.py status ml_prod_v1
```

View logs:
```bash
python scripts/process_manager.py logs ml_prod_v1 -f
```

## Configuration

### Setting Up Experiments

Add ML experiments to your `config.yml`:

```yaml
machine_learning:
  # Default settings for all experiments
  defaults:
    model_type: 'linear_regression'
    target_column: 'total_richness'
    cv_strategy:
      type: 'spatial_block'
      n_splits: 5
      block_size: 100
  
  # Named experiments
  experiments:
    my_experiment:
      input_parquet: 'outputs/my_data.parquet'
      model_type: 'lightgbm'
      # ... other settings
```

See `config_ml_examples.yml` for comprehensive examples.

### Key Configuration Options

- **input_parquet**: Path to input data (required)
- **model_type**: 'linear_regression' or 'lightgbm'
- **target_column**: Column to predict
- **cv_strategy**: Cross-validation configuration
  - type: 'spatial_block', 'spatial_buffer', or 'environmental'
  - n_splits: Number of CV folds
  - block_size/buffer_distance: Spatial parameters
- **imputation_strategy**: Missing value handling
  - type: 'spatial_knn'
  - n_neighbors: Number of neighbors
  - spatial_weight: Weight for geographic distance

## Input Data Requirements

The input parquet file should contain:
- **Coordinate columns**: `latitude`, `longitude`
- **Richness columns**: e.g., `plants_richness`, `animals_richness`
- **Target column**: As specified in config (e.g., `total_richness`)

## Output Structure

ML pipeline outputs are saved to:
```
outputs/
└── ml_results/
    └── <experiment_id>/
        ├── model.pkl              # Trained model
        ├── predictions.parquet    # Predictions with residuals
        ├── feature_importance.csv # Feature importance scores
        └── ml_metrics.json       # Performance metrics
```

## Advanced Usage

### Override Configuration

Override specific settings from command line:
```bash
./run_ml.sh --experiment test_linear --config model_type=lightgbm,cv_folds=3
```

### Resume from Checkpoint

Resume interrupted ML pipeline:
```bash
./run_ml.sh --experiment production_lgb --resume
```

### Custom ML Pipeline

Run ML directly with Python:
```python
from scripts.ml_pipeline_runner import run_ml_pipeline

ml_config = {
    'input_parquet': 'outputs/data.parquet',
    'model_type': 'lightgbm',
    'target_column': 'total_richness'
}

run_ml_pipeline(ml_config)
```

## Process Management

### Using process_manager.py

Start ML pipeline:
```bash
python scripts/process_manager.py start \
  --name ml_analysis_1 \
  --pipeline-type ml \
  --ml-config '{"experiment_name": "production_lgb"}' \
  --daemon
```

List all processes:
```bash
python scripts/process_manager.py status
```

Stop ML pipeline:
```bash
python scripts/process_manager.py stop ml_analysis_1
```

## Monitoring and Debugging

### Log Files

ML pipeline logs are saved to:
- `logs/ml_pipeline.log` - Main ML pipeline log
- `logs/<process_name>.log` - Process-specific logs (daemon mode)

### Real-time Monitoring

For tmux sessions:
- `Ctrl+B` then arrow keys to navigate panes
- `Ctrl+B` then `D` to detach
- `tmux attach -t ml_<session_name>` to reattach

### Performance Metrics

After completion, check metrics in:
- `outputs/ml_results/<experiment_id>/ml_metrics.json`

Example metrics:
```json
{
  "model_type": "lightgbm",
  "n_features": 25,
  "n_samples": 100000,
  "training_metrics": {
    "r2": 0.85,
    "rmse": 12.5,
    "mae": 8.3
  },
  "cv_results": {
    "mean_metrics": {
      "r2": 0.82,
      "rmse": 13.1
    },
    "std_metrics": {
      "r2": 0.03,
      "rmse": 0.8
    }
  }
}
```

## Best Practices

1. **Always use spatial CV** for geographic data
2. **Start with test_linear** experiment for quick validation
3. **Use monitoring mode** for long-running experiments
4. **Save experiment configs** in config.yml for reproducibility
5. **Check data quality** before running ML:
   ```bash
   # Verify parquet file
   python -c "import pandas as pd; df = pd.read_parquet('outputs/data.parquet'); print(df.info())"
   ```

## Troubleshooting

### Common Issues

1. **Missing input file**:
   - Ensure parquet path in config.yml is correct
   - Check if data pipeline has completed export

2. **Memory errors**:
   - Use lighter model (linear_regression instead of lightgbm)
   - Reduce CV folds
   - Process smaller regions

3. **Import errors**:
   - Ensure conda environment is activated
   - Install optional dependencies: `pip install lightgbm`

### Debug Mode

Run with debug logging:
```bash
python scripts/ml_pipeline_runner.py --experiment test_linear --log-level DEBUG
```

## Integration with Data Pipeline

### Sequential Workflow

1. Run data pipeline to create parquet:
   ```bash
   ./run_pipeline.sh --experiment-name biodiversity_prep
   ```

2. Run ML on the output:
   ```bash
   ./run_ml.sh --input outputs/<experiment_id>/merged_biodiversity_data.parquet
   ```

### Parallel Experiments

Run multiple ML experiments on same data:
```bash
# Terminal 1
./run_ml.sh --experiment test_linear --process-name ml_linear

# Terminal 2  
./run_ml.sh --experiment production_lgb --process-name ml_lgb
```

## Example Workflows

### 1. Quick Test
```bash
# Create small test data
./run_pipeline.sh --experiment-name test_small --bounds test_small

# Run quick ML test
./run_ml.sh --experiment test_linear
```

### 2. Production Run
```bash
# Run with monitoring
./run_ml.sh --experiment production_lgb --monitor

# Or as daemon
./run_ml.sh --experiment production_lgb --daemon --process-name ml_prod_$(date +%Y%m%d)
```

### 3. Model Comparison
```bash
# Run multiple models
for model in test_linear production_lgb; do
  ./run_ml.sh --experiment $model --daemon --process-name ml_$model
done

# Check all statuses
python scripts/process_manager.py status
```

## Next Steps

- See `ML_MODULE_DOCUMENTATION.md` for detailed API documentation
- Check `config_ml_examples.yml` for more configuration examples
- Use `scripts/ml_compare_models.py` for model comparison
- Run `scripts/ml_benchmark.py` for performance analysis