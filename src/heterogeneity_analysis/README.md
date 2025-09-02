# Heterogeneity Analysis Module

## Purpose
Tests whether biological relationships (correlations between taxa) vary across environmental contexts (biomes). This analysis examines the **context-dependency** of ecological relationships.

## Key Question
**Do correlations between fungi, plants, and vertebrates differ across biomes?**

## Analyses Performed

### 1. PERMANOVA
- Tests if biodiversity **composition** differs between biomes
- Establishes that biomes represent different ecological contexts
- Uses `adonis2()` from vegan package

### 2. Beta Dispersion (Variance Heterogeneity)
- Tests if **variance** in biodiversity differs between biomes
- Multivariate analog of Levene's test
- Uses `betadisper()` and `permutest()` from vegan

### 3. Correlation Analysis by Biome (Core Analysis)
- Calculates correlation matrices for each biome separately
- Tests if correlations differ between biomes using Fisher's Z transformation
- This is the main analysis for understanding heterogeneity of biological relationships

### 4. NMDS Visualization
- Visualizes biodiversity patterns in ordination space
- Colors points by biome to show clustering patterns
- Uses `metaMDS()` from vegan

## Usage

### For Raw Biodiversity Data
```r
source("src/heterogeneity_analysis/heterogeneity_analysis.R")

results <- heterogeneity_analysis(
  data_path = "path/to/biodiversity_data.arrow",
  output_dir = "./results/heterogeneity",
  min_samples_per_biome = 100,
  permutations = 999
)
```

### Required Data Columns
- `fungi_richness`: Fungal species richness
- `plants_richness`: Plant species richness
- `vertebrates_richness`: Vertebrate species richness
- `longitude`: Site longitude
- `latitude`: Site latitude

## Outputs

### Files Generated
1. `correlation_summary_by_biome.csv` - Correlations for each biome
2. `beta_dispersion.pdf` - Variance heterogeneity visualization
3. `nmds_by_biome.pdf` - NMDS ordination colored by biome
4. `correlation_heterogeneity.pdf` - Heatmap of correlations across biomes
5. `heterogeneity_analysis_results.rds` - Complete R results object

### Interpretation

#### Significant PERMANOVA
- Biomes have different biodiversity compositions
- Justifies examining relationships separately by biome

#### Significant Beta Dispersion Test
- Variance differs between biomes (heteroscedasticity)
- Indicates heterogeneity in biodiversity variability

#### Different Correlations Between Biomes
- Biological relationships are context-dependent
- Example: Fungi-plant correlations may be positive in grasslands but negative in forests
- Suggests different ecological processes operating in different biomes

## Note on Residual Data
This analysis should be performed on **raw biodiversity data**, not residuals from environmental correction. If environmental effects have been removed (e.g., via DML), biome differences should be minimal and this analysis would not be meaningful.

## Dependencies
- `vegan`: For PERMANOVA, beta dispersion, NMDS
- `cocor`: For comparing correlations (optional, can use Fisher's Z manually)
- `speciesgeocodeR`: For WWF biome assignment
- `ggplot2`: For visualizations
- `dplyr`: For data manipulation