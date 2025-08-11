# Machine Learning Regression Pipeline - Single Entry Point

## Overview
The ML regression pipeline implements the **3 methods from regression_analysis_decisions.md** through a single script: `run_ml.sh`

This is the ONLY script you need to run for ML regression hypothesis testing.

## Architecture

```
run_ml.sh (single entry point)
    └── run_ml_regression.py
        ├── Method1ContinuousGLM (H1: Biodiversity begets biodiversity)
        ├── Method2CategoricalGLM (H2: Temperate mismatch hypothesis)
        └── Method3DML (Causal inference for all hypotheses)
```

## Usage

### Required Input
The pipeline requires an input data file (parquet format):

```bash
./run_ml.sh --input outputs/biodiversity_data.parquet
```

### Running the 3 Methods

#### Run all methods (default):
```bash
./run_ml.sh --input outputs/biodiversity_data.parquet
```

#### Run specific method:
```bash
# Method 1: Continuous GLM (H1 hypothesis)
./run_ml.sh --input data.parquet --method glm-continuous

# Method 2: Categorical GLM (H2 hypothesis)  
./run_ml.sh --input data.parquet --method glm-categorical

# Method 3: Double Machine Learning (causal inference)
./run_ml.sh --input data.parquet --method dml
```

#### Additional Options:

```bash
# Show current configuration
./run_ml.sh --show-config

# Run diagnostics only
./run_ml.sh --input data.parquet --diagnostics-only

# Use legacy config system
./run_ml.sh --use-legacy-config --input data.parquet

# Use custom config file
./run_ml.sh --config src/config/ml/ml_config_unified.yml --input data.parquet
```

### Execution Modes

```bash
# Foreground (default) - see output directly
./run_ml.sh --input data.parquet

# Daemon mode - run in background
./run_ml.sh --input data.parquet --daemon

# Monitor mode - 4-pane tmux layout
./run_ml.sh --input data.parquet --monitor
```

## Configuration

### Location
All ML configurations are in `src/config/ml/`:
- `ml_config.yml` - Legacy configuration
- `ml_config_unified.yml` - New unified configuration (recommended)
- `unified_config.py` - Configuration dataclasses

### System Selection
- **Default**: Unified configuration system (clean, type-safe)
- **Legacy**: Use `--use-legacy-config` flag

## The 3 Methods (from regression_analysis_decisions.md)

- **Method 1 - glm-continuous**: GLM with continuous interaction (H1: Biodiversity begets biodiversity)
  - Tests if biodiversity-biodiversity relationship varies continuously with temperature seasonality (bio4)
  
- **Method 2 - glm-categorical**: GLM with categorical interaction (H2: Temperate mismatch hypothesis)
  - Tests if biodiversity-biodiversity relationship is stronger in temperate regions with high plant diversity
  
- **Method 3 - dml**: Double Machine Learning for causal inference
  - Tests all hypotheses using causal inference with cross-fitting and bootstrap inference

## Monitor Mode Layout

When using `--monitor`, you get a 4-pane tmux view:
```
┌─────────────────┬─────────────────┐
│                 │                 │
│  Main Execution │      Logs       │
│                 │                 │
├─────────────────┼─────────────────┤
│                 │                 │
│    Progress     │    Resources    │
│                 │                 │
└─────────────────┴─────────────────┘
```

- **Main Execution**: Pipeline output
- **Logs**: Real-time log tail
- **Progress**: Status monitoring
- **Resources**: System resources (htop/top)

## Quick Start Examples

```bash
# 1. Run all 3 methods
./run_ml.sh --input outputs/biodiversity_data.parquet

# 2. Run Method 3 (DML) only
./run_ml.sh --input outputs/biodiversity_data.parquet --method dml

# 3. Run with monitoring (4-pane tmux view)
./run_ml.sh --input outputs/biodiversity_data.parquet --monitor

# 4. Check configuration
./run_ml.sh --show-config

# 5. Run in background
./run_ml.sh --input data.parquet --daemon --process-name ml_background_job
```

## File Structure

**Removed redundant scripts:**
- ❌ `run_ml_v2.sh` - Redundant, functionality merged into `run_ml.sh`
- ❌ `scripts/run_ml.py` - Traditional ML experiments (not your 3 methods)

**Essential scripts:**
- ✅ `run_ml.sh` - Single entry point for running your 3 methods
- ✅ `run_ml_regression.py` - Implements calls to your 3 methods
- ✅ `src/machine_learning/research/methods/method1_continuous.py` - Method 1 implementation
- ✅ `src/machine_learning/research/methods/method2_categorical.py` - Method 2 implementation
- ✅ `src/machine_learning/research/methods/method3_dml.py` - Method 3 implementation

## Why Single Entry Point?

1. **Simplicity**: One script (`run_ml.sh`) to run all 3 methods
2. **Consistency**: Same interface for all regression methods
3. **Focused**: Only implements your 3 methods from regression_analysis_decisions.md
4. **Clear**: No confusion with traditional ML experiments

## Help

For comprehensive help:
```bash
./run_ml.sh --help
```

This shows all options, examples, and available methods.