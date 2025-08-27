# ML Pipeline Installation Guide

## System Requirements

This ML pipeline requires R, Julia, and Python with specific packages. Follow the installation steps below for your system.

## R Installation

### Ubuntu/Debian
```bash
# Update system
sudo apt update
sudo apt install -y software-properties-common dirmngr

# Add R repository
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

# Install R
sudo apt update
sudo apt install -y r-base r-base-dev
```

### CentOS/RHEL/Rocky Linux
```bash
# Install EPEL
sudo dnf install -y epel-release

# Install R
sudo dnf install -y R R-devel
```

### macOS
```bash
# Using Homebrew
brew install r

# Or download from: https://cran.r-project.org/bin/macosx/
```

### R Package Installation
```bash
# Install required R packages
R -e "install.packages(c('emmeans', 'MASS', 'arrow', 'jsonlite', 'car'), repos='https://cloud.r-project.org/')"

# Set custom library path if needed
R -e "install.packages(c('emmeans', 'MASS', 'arrow', 'jsonlite', 'car'), repos='https://cloud.r-project.org/', lib='~/R/libs')"
```

## Julia Installation

### Download and Install Julia
```bash
# Download Julia 1.9+ (adjust version as needed)
cd ~/
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.4-linux-x86_64.tar.gz
tar -xzf julia-1.9.4-linux-x86_64.tar.gz
mv julia-1.9.4 julia

# Add to PATH
export PATH="$HOME/julia/bin:$PATH"
echo 'export PATH="$HOME/julia/bin:$PATH"' >> ~/.bashrc
```

### macOS Julia Installation
```bash
# Using Homebrew
brew install julia

# Or download from: https://julialang.org/downloads/
```

### Julia Package Installation
```bash
# Install required Julia packages
julia -e 'import Pkg; Pkg.add(["DataFrames", "Arrow", "GLM", "Distributions", "JSON", "StatsBase", "Distances", "LinearAlgebra", "YAML", "JLD2", "NPZ", "Random", "StatsModels", "HypothesisTests", "LightGBM"])'

# Alternative: Install packages interactively
julia
```

Then in Julia REPL:
```julia
import Pkg
Pkg.add([
    "DataFrames",
    "Arrow", 
    "GLM",
    "Distributions",
    "JSON",
    "StatsBase", 
    "Distances",
    "LinearAlgebra",
    "YAML",
    "JLD2",
    "NPZ",
    "Random",
    "StatsModels",
    "HypothesisTests",
    "LightGBM"
])
```

## Python Environment Setup

### Conda Environment (Recommended)
```bash
# Install Anaconda/Miniconda if not already installed
# Download from: https://www.anaconda.com/download

# Create conda environment
conda create -n geo python=3.9 -y
conda activate geo

# Install core packages
conda install -y numpy pandas scipy scikit-learn matplotlib seaborn jupyter

# Install additional packages with pip
pip install h2o lightgbm polars fastdist pynndescent econml jsonschema rpy2
```

### System Python Installation
```bash
# Ubuntu/Debian
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv ~/geo_env
source ~/geo_env/bin/activate

# Install packages
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter h2o lightgbm polars fastdist pynndescent econml jsonschema rpy2
```

## Verification

### Test R Installation
```bash
R -e "library(emmeans); cat('R packages installed successfully\n')"
```

### Test Julia Installation
```bash
julia -e "using DataFrames, GLM, LightGBM; println(\"Julia packages installed successfully\")"
```

### Test Python Installation
```bash
# Activate conda environment
conda activate geo

# Test imports
python -c "import h2o, lightgbm, pandas, numpy; print('Python packages installed successfully')"
```

## Environment Variables

Set these environment variables for optimal performance:

```bash
# Julia threading
export JULIA_NUM_THREADS=8

# Python multiprocessing  
export NUMBA_NUM_THREADS=120
export MESF_N_JOBS=120
export NUMEXPR_MAX_THREADS=120
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=120"
```

Add to your `~/.bashrc` or `~/.zshrc` to make permanent.

## Running the Pipeline

```bash
# Make sure you're in the project directory
cd /home/yl998/Geo_process

# Activate Python environment
conda activate geo  # or source ~/geo_env/bin/activate

# Run Julia pipeline
./run_julia_pipeline.sh

# Run ML analysis with monitoring
./run_ml_analysis.sh --data your_data.parquet --monitor
```

## Troubleshooting

### R Package Installation Issues
```bash
# If installation fails, try installing system dependencies first
sudo apt install -y libcurl4-openssl-dev libssl-dev libxml2-dev

# For arrow package specifically
sudo apt install -y libarrow-dev
```

### Julia Package Issues
```bash
# Update Julia packages
julia -e 'import Pkg; Pkg.update()'

# Resolve package conflicts
julia -e 'import Pkg; Pkg.resolve()'
```

### Python Package Issues
```bash
# Update pip
pip install --upgrade pip

# Install specific versions if needed
pip install h2o==3.44.0.3
```

### Memory Issues
- For large datasets, ensure sufficient RAM (32GB+ recommended)
- Use spatial sampling: `--spatial-grid-step 10`
- Monitor memory usage with `htop` or `top`

## File Paths

Update these paths in scripts if your installation differs:

- Julia: `$HOME/julia/bin/julia`
- Python: `$HOME/anaconda3/envs/geo/bin/python`
- R: `R` (should be in PATH)

## Support

For installation issues:
1. Check system requirements
2. Verify PATH variables
3. Test each component individually
4. Check log files in `logs/` directory