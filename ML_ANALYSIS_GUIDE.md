# ML Analysis Guide - Method 1 and Method 2

This guide shows you exactly how to run Method 1 and Method 2 analysis from scratch.

## Important Security Notes
- All paths use environment variables to avoid hardcoding
- Variables are properly quoted to handle spaces and special characters
- Julia installation provides both user-local and system-wide options
- Always verify data paths before running commands

## Prerequisites Installation

### Step 1: Install Julia
Copy and run these commands:

```bash
cd ~
# Option 1: User-local installation (no sudo required)
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
tar -xzf julia-1.9.4-linux-x86_64.tar.gz
mkdir -p ~/bin
ln -sf ~/julia-1.9.4/bin/julia ~/bin/julia
export PATH="$HOME/bin:$PATH"
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
julia --version

# Option 2: System-wide installation (requires sudo)
# sudo mv julia-1.9.4 /opt/
# sudo ln -sf /opt/julia-1.9.4/bin/julia /usr/local/bin/julia
```

### Step 2: Install Julia Packages
Copy and run this command:

```bash
julia -e 'using Pkg; Pkg.add(["DataFrames", "Arrow", "Statistics", "Distributions", "JSON", "StatsBase", "LinearAlgebra", "YAML", "JLD2", "Dates", "NPZ", "Random", "StatsModels", "HypothesisTests", "GLM", "Combinatorics"])'
```

### Step 3: Install R (for diagnostics)
Copy and run these commands:

```bash
sudo apt update
sudo apt install -y r-base r-base-dev
R -e 'install.packages(c("DHARMa", "jsonlite", "ggplot2", "dplyr"), repos="https://cloud.r-project.org/")'
```

### Step 4: Setup Python Environment
Copy and run these commands:

```bash
# Create conda environment with required packages
conda create -n geo python=3.9 -y
conda activate geo
pip install --upgrade jax jaxlib numpy pandas pyarrow scipy
```

## Data Preparation

First, set your data path as a variable:
```bash
# Replace with YOUR actual data file path
export DATA_PATH="/path/to/your/biodiversity_data.arrow"

# For example:
# export DATA_PATH="/scratch/username/biodiversity_climate_production_1x_corrected.arrow"
```

## Running Method 1 Analysis

### Step 1: Activate Environment and Set Thread Count
Copy and run:

```bash
conda activate geo
export JULIA_NUM_THREADS=120
```

### Step 2: Compute MESF (Spatial Components)
Copy and run:

```bash
cd src/machine_learning/spatial
python compute_mesf.py "$DATA_PATH" 1000
```

Note: This will compute MESF for sampled data (every 1000th point). The results are cached.

### Step 3: Run Julia Method 1
Copy and run:

```bash
cd src/machine_learning/pipelines
julia julia_hybrid_pipeline.jl method1 "$DATA_PATH" 1000
```

### Step 4: Check Results
Copy and run:

```bash
ls -la "/scratch/$USER/julia/method1"*
```

## Running Method 2 Analysis

### Step 1: Ensure MESF is Computed
If you already ran Method 1, skip this. Otherwise run Step 2 from Method 1 above.

### Step 2: Run Julia Method 2
Copy and run:

```bash
cd src/machine_learning/pipelines
julia julia_hybrid_pipeline.jl method2 "$DATA_PATH" 1000
```

### Step 3: Check Results
Copy and run:

```bash
ls -la "/scratch/$USER/julia/method2"*
```

## View Results

### View Method 1 Results
Copy and run:

```bash
python3 -c "
import json, os
with open(f'/scratch/{os.environ["USER"]}/julia/method1_results.json', 'r') as f:
    results = json.load(f)
    if 'method1_lr_test' in results:
        print('Method 1 LRT p-value:', results['method1_lr_test']['p_value'])
    if 'method1_full' in results:
        print('Method 1 Full Model AIC:', results['method1_full']['aic'])
"
```

### View Method 2 Results
Copy and run:

```bash
python3 -c "
import json, os
with open(f'/scratch/{os.environ["USER"]}/julia/method2_results.json', 'r') as f:
    results = json.load(f)
    if 'method2_continuous' in results:
        print('Method 2 Continuous AIC:', results['method2_continuous']['aic'])
    if 'method2_categorical' in results:
        print('Method 2 Categorical AIC:', results['method2_categorical']['aic'])
"
```

## Configuration Changes (Optional)

If you need to change settings, edit this file:
```
src/config/ml/ml_config.yml
```

### To Enable Ridge Regression:
Change line 12 from:
```yaml
  enabled: false
```
to:
```yaml
  enabled: true
```

### To Change Sampling:
The `1000` in the Julia commands means "use every 1000th data point".
- For full data: Change `1000` to `1`
- For faster test: Change `1000` to `10000`

## Troubleshooting

### Error: "MESF cache not found"
Run Step 2 from Method 1 to compute MESF.

### Error: "Julia not found"
Run Step 1 from Prerequisites to install Julia.

### Error: "Package not found" in Julia
Run Step 2 from Prerequisites to install Julia packages.

### Out of Memory
Use larger sampling step (change `1000` to `10000` in Julia commands).

## Full Dataset Analysis

To run on the complete dataset (warning: very slow, needs 64GB+ RAM):

```bash
# First compute MESF for full data (this will take hours)
cd src/machine_learning/spatial
python compute_mesf.py "$DATA_PATH" 1

# Then run Julia with no sampling
cd src/machine_learning/pipelines
julia julia_hybrid_pipeline.jl method1 "$DATA_PATH" 1
```