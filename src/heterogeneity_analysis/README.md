# Heterogeneity Analysis Module

A reusable, config-driven module for analyzing biodiversity heterogeneity across biomes using WWF ecoregions classification. All scripts read from a central configuration file with no hardcoded fallbacks.

## Purpose
Tests whether biological relationships (correlations between taxa) vary across environmental contexts (biomes). This analysis examines the **context-dependency** of ecological relationships.

## Key Question
**Do correlations between fungi, plants, and vertebrates differ across biomes?**

## Module Structure

```
heterogeneity_analysis/
├── config.yaml                        # Central configuration file (REQUIRED)
├── assign_wwf_biomes.py              # Step 1: Assign WWF biomes to biodiversity data
├── spearman_correlation_analysis.R    # Step 2a: Spearman correlation analysis
├── partial_spearman_analysis.R        # Step 2b: Partial Spearman (controlling for abiotic)
├── permanova_analysis.R               # Step 2c: PERMANOVA and beta dispersion analysis
└── README.md                          # This file
```

## Configuration

All settings are controlled via `config.yaml`. Key sections:

- **data**: Input/output paths for biodiversity data
- **parameters**: Sample sizes, thresholds, random seeds
- **visualization**: Figure dimensions and styles
- **columns**: Column name mappings for flexibility

## Workflow

### Step 1: Configure the Module

Edit `config.yaml` to set:
- Input data path (`data.raw_biodiversity`)
- Output directory (`output.base_dir`)
- Sample sizes for each analysis type
- Other parameters as needed

### Step 2: Assign WWF Biomes (if needed)

```bash
python src/heterogeneity_analysis/assign_wwf_biomes.py
```

This reads from `data.raw_biodiversity` and outputs to `data.biodiversity_with_biomes`.

### Step 3: Run Analyses

Each analysis script is independent and reads from config:

```bash
# Standard Spearman correlations
cd src/heterogeneity_analysis
Rscript spearman_correlation_analysis.R

# Partial Spearman (controlling for environment)
Rscript partial_spearman_analysis.R

# PERMANOVA and beta dispersion
Rscript permanova_analysis.R
```

## Analyses Performed

### 1. Spearman Correlation Analysis
- Calculates Spearman correlations between taxa for each biome
- Handles up to 1M samples efficiently
- Generates 4 visualizations showing correlation patterns

### 2. Partial Spearman Analysis
- Controls for abiotic factors (temperature, precipitation, soil, etc.)
- Reveals biological relationships independent of environment
- Shows how much correlation is explained by shared environmental response

### 3. PERMANOVA & Beta Dispersion
- **PERMANOVA**: Tests if biodiversity composition differs between biomes
- **Beta Dispersion**: Tests if variance in biodiversity differs between biomes
- **NMDS**: Visualizes biodiversity patterns in ordination space
- Limited to 46,340 samples (Fortran constraint)

## Output Structure

Results are organized under `output.base_dir`:

```
results/heterogeneity_analysis/
├── biome_assignment/                  # WWF biome assignment logs
├── spearman_correlations/            # Standard Spearman results
│   ├── spearman_correlations_by_biome.csv
│   ├── spearman_correlations_all.png
│   ├── spearman_fungi_plants.png
│   ├── spearman_fungi_vertebrates.png
│   └── spearman_plants_vertebrates.png
├── partial_spearman/                  # Partial correlation results
│   ├── partial_spearman_correlations.csv
│   ├── controlled_abiotic_variables.txt
│   ├── partial_spearman_all.png
│   ├── partial_spearman_fungi_plants.png
│   ├── partial_spearman_fungi_vertebrates.png
│   └── partial_spearman_plants_vertebrates.png
└── permanova_beta_dispersion/        # PERMANOVA results
    ├── permanova_results.rds
    ├── permanova_summary.txt
    ├── beta_dispersion.png
    └── nmds_by_biome.png
```

## Interpretation

### Significant PERMANOVA
- Biomes have different biodiversity compositions
- Justifies examining relationships separately by biome

### Significant Beta Dispersion Test
- Variance differs between biomes (heteroscedasticity)
- Indicates heterogeneity in biodiversity variability

### Different Correlations Between Biomes
- Biological relationships are context-dependent
- Example: Fungi-plant correlations may be positive in grasslands but negative in forests
- Suggests different ecological processes operating in different biomes

### Partial vs Standard Correlations
- Large difference: Environmental factors drive much of the correlation
- Small difference: Direct biological interactions dominate

## Key Features

- **Config-driven**: All settings in one YAML file
- **No fallbacks**: Scripts fail cleanly if config/data missing
- **Independent scripts**: Each analysis runs standalone
- **Large-scale ready**: Handles 1M+ samples for correlations
- **Memory-aware**: PERMANOVA limited to 46K samples

## Requirements

### Python
```bash
pip install pandas geopandas pyarrow shapely requests
```

### R
```r
install.packages(c("vegan", "arrow", "dplyr", "ggplot2", "reshape2", "yaml", "ppcor"))
```

## Error Handling

Scripts will stop with clear error messages if:
- `config.yaml` is missing
- Required data files don't exist  
- Required columns are missing
- Insufficient samples for analysis

## Sample Limits

- **Spearman**: 1M samples (configurable)
- **Partial Spearman**: 1M samples (configurable)
- **PERMANOVA**: 46,340 samples maximum (hard limit)

## Required Data Columns

- `fungi_richness`: Fungal species richness
- `plants_richness`: Plant species richness
- `vertebrates_richness`: Vertebrate species richness
- `longitude` or `x`: Site longitude
- `latitude` or `y`: Site latitude
- Plus abiotic variables for partial correlations

## Citation

If using this module, please cite:
- WWF Ecoregions: Olson et al. (2001) BioScience
- vegan package: Oksanen et al. (2022)
- Your analysis publication