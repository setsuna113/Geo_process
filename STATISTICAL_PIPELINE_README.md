# Statistical Analysis Pipeline - Complete Guide

## Overview

This guide provides step-by-step instructions to reproduce all statistical analysis results from scratch. The pipeline analyzes biodiversity patterns using multiple statistical methods on geospatial data.

## Prerequisites

### System Requirements
- **Python**: 3.9+ 
- **RAM**: 32GB minimum (64GB recommended for full dataset)
- **Storage**: 100GB free space for data and results
- **OS**: Linux/macOS (Windows via WSL2)

## Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd geo
```

### Step 2: Create Python Environment
```bash
# Using conda (recommended)
conda create -n geo_stats python=3.11
conda activate geo_stats

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Python Dependencies
```bash
# Install core dependencies
pip install -r requirements_statistical.txt

# Verify installation
python -c "import numpy, pandas, rasterio, geopandas; print('Core packages OK')"
```

### Step 4: Install Additional Dependencies (if needed)
```bash
# For visualization support
pip install matplotlib seaborn cartopy

# For parallel processing
pip install dask[complete] ray

# For geospatial operations
pip install shapely fiona pyproj
```

## Data Preparation

### Step 1: Prepare Input Data

The pipeline expects data in specific formats:

1. **Tabular Data** (Parquet/Arrow format):
   - Located at: `data/preprocessed/merged_climate_biodiversity.parquet`
   - Required columns: latitude, longitude, plants_richness, vertebrates_richness, am_fungi_richness, ecm_fungi_richness

2. **Raster Data** (GeoTIFF format):
   - Multiband rasters in: `data/rasters/`
   - Individual species rasters: plants.tif, vertebrates.tif, fungi.tif

```bash
# Create data directories
mkdir -p data/preprocessed
mkdir -p data/rasters
mkdir -p outputs/statistical

# Copy or link your data files
cp /path/to/your/merged_data.parquet data/preprocessed/merged_climate_biodiversity.parquet
cp /path/to/your/*.tif data/rasters/
```

### Step 2: Verify Data Structure
```python
# verify_data.py
import pandas as pd
from pathlib import Path

# Check parquet file
data_path = Path("data/preprocessed/merged_climate_biodiversity.parquet")
if data_path.exists():
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
else:
    print(f"ERROR: Data file not found at {data_path}")
```

## Configuration

### Step 1: Review Configuration File
```bash
# Check configuration
cat src/config/stat/stat_methods.yml
```

### Step 2: Adjust Configuration (if needed)
```yaml
# src/config/stat/stat_methods.yml
datasets:
  climate_parquet: "data/preprocessed/merged_climate_biodiversity.parquet"
  
  # Raster paths
  plants_raster: "data/rasters/plants.tif"
  vertebrates_raster: "data/rasters/vertebrates.tif"
  am_fungi_raster: "data/rasters/am_fungi.tif"
  ecm_fungi_raster: "data/rasters/ecm_fungi.tif"

# Sampling configuration
spatial_grid_step: 1  # 1 = use all data, 10 = every 10th point

# Output configuration
output_dir: "outputs/statistical"

# Method-specific settings
methods:
  correlation:
    enabled: true
    compute_p_values: true
  coverage:
    enabled: true
  overlap:
    enabled: true
    threshold_percentile: 90
  latitudinal:
    enabled: true
    n_latitude_bins: 180
  global_maps:
    enabled: true
  trust_maps:
    enabled: true
  biome_analysis:
    enabled: true
  partial_correlation:
    enabled: true
```

## Running the Pipeline

### Step 1: Test Single Method
```bash
# Test correlation analysis only
python run_stats.py --methods correlation

# Check output
ls outputs/statistical/correlation/
```

### Step 2: Run All Statistical Analyses
```bash
# Run all methods with 4 features (AM/ECM fungi separate)
python run_stats.py --variant 4_features

# Run all methods with 3 features (fungi merged)
python run_stats.py --variant 3_features_merged

# Run both variants automatically
python run_stats.py --both-variants
```

### Step 3: Run with Spatial Sampling (for testing)
```bash
# Use every 10th spatial point (faster for testing)
# Edit src/config/stat/stat_methods.yml:
# spatial_grid_step: 10

python run_stats.py --variant 4_features
```

### Step 4: Run Specific Methods Only
```bash
# Run only correlation and coverage analyses
python run_stats.py --methods correlation coverage

# Run latitudinal and global maps
python run_stats.py --methods latitudinal global_maps
```

## Expected Outputs

After successful execution, you should see:

```
outputs/statistical/
├── correlation/
│   ├── correlation_matrix.csv
│   ├── correlation_heatmap.png
│   └── correlation_results.json
├── coverage/
│   ├── coverage_stats.csv
│   ├── coverage_maps.png
│   └── coverage_summary.json
├── overlap/
│   ├── overlap_matrix.csv
│   ├── overlap_venn.png
│   └── overlap_results.json
├── latitudinal/
│   ├── latitudinal_gradients.csv
│   ├── latitude_plots.png
│   └── gradient_stats.json
├── global_maps/
│   ├── richness_maps.png
│   ├── hotspot_analysis.csv
│   └── global_stats.json
├── trust_maps/
│   ├── cv_maps.png
│   ├── trust_scores.csv
│   └── uncertainty_analysis.json
├── biome_analysis/
│   ├── biome_richness.csv
│   ├── biome_plots.png
│   └── biome_stats.json
└── partial_correlation/
    ├── partial_corr_matrix.csv
    ├── partial_corr_network.png
    └── partial_results.json
```

## Validation

### Step 1: Check Results
```python
# check_results.py
import json
from pathlib import Path

output_dir = Path("outputs/statistical")

# Check each method's output
for method_dir in output_dir.iterdir():
    if method_dir.is_dir():
        json_files = list(method_dir.glob("*.json"))
        if json_files:
            with open(json_files[0]) as f:
                result = json.load(f)
                success = result.get('success', False)
                print(f"{method_dir.name}: {'✓' if success else '✗'}")
```

### Step 2: Verify Reproducibility
```bash
# Run analysis twice and compare
python run_stats.py --methods correlation --variant 4_features
mv outputs/statistical/correlation outputs/statistical/correlation_run1

python run_stats.py --methods correlation --variant 4_features
mv outputs/statistical/correlation outputs/statistical/correlation_run2

# Compare results
diff outputs/statistical/correlation_run1/correlation_matrix.csv \
     outputs/statistical/correlation_run2/correlation_matrix.csv
```

## Troubleshooting

### Common Issues and Solutions

1. **Memory Error**
```bash
# Reduce data size using spatial sampling
# Edit config: spatial_grid_step: 10
```

2. **Missing Dependencies**
```bash
# Install missing packages
pip install <missing_package>

# Update all packages
pip install --upgrade -r requirements_statistical.txt
```

3. **Data Not Found**
```bash
# Check data paths in config
grep -E "parquet|raster" src/config/stat/stat_methods.yml

# Verify files exist
ls -la data/preprocessed/*.parquet
ls -la data/rasters/*.tif
```

4. **Permission Errors**
```bash
# Ensure write permissions
chmod -R u+w outputs/
```

## Advanced Usage

### Custom Configuration
```python
# run_custom.py
from src.biodiversity_analysis.statistical.orchestrator import run_analysis

# Custom config
config = {
    'datasets': {
        'climate_parquet': 'path/to/data.parquet'
    },
    'spatial_grid_step': 5,
    'methods': {
        'correlation': {'enabled': True}
    }
}

# Run with custom config
results = run_analysis(
    methods=['correlation'],
    config=config,
    output_dir='custom_outputs'
)
```

### Parallel Processing
```bash
# Set number of threads for numpy/scipy
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Run pipeline
python run_stats.py --variant 4_features
```

## Citation

If you use this pipeline in your research, please cite:
```
[Citation information to be added]
```

## Support

For issues or questions:
1. Check existing issues: [GitHub Issues]
2. Create new issue with:
   - Error message
   - Config file
   - Data sample info
   - System specs

## License

[License information]