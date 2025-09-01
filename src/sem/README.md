# Mycorrhizal SEM Analysis

## Overview
Implementation of REAL piecewise Structural Equation Modeling (SEM) using R's piecewiseSEM package.
This performs proper SEM with Fisher's C statistic, tests of directed separation, and correlated errors.

**WARNING**: Python cannot do real SEM - only path analysis (separate regressions).

## Model Structure

```
Environmental Variables (Climate, Soil, Topography)
                ↓
        Mycorrhizal Fungi (AM + ECM)
                ↓ ↔ (bidirectional: nutrient-carbon exchange)
            Plant Richness
                ↓
            Animal Richness
                ↑ ↔ (bidirectional with fungi: consumption-dispersal)
```

## Data Source
- File: `/scratch/yl998/biodiv_1x_renamed.arrow` (~4GB)
- Variables:
  - **Biodiversity**: plant_richness, animal_richness, am_fungi_richness, ecm_fungi_richness
  - **Climate**: bio01 (temp), bio04 (temp seasonality), bio12 (precip), bio15 (precip seasonality)
  - **Topography**: elevation, slope, tpi
  - **Soil**: soil_organic_carbon, soil_ph
  - **Other**: pet_mean, pet_range

## Files
- `mycorrhizal_sem.R` - Main SEM analysis with full and simplified models
- `explore_data.py` - Data exploration script
- `install_packages.R` - R package installation

## Running the Analysis

1. Install required R packages:
```bash
Rscript src/sem/install_packages.R
```

2. Explore the data:
```bash
python src/sem/explore_data.py
```

3. Run SEM analysis:
```bash
Rscript src/sem/mycorrhizal_sem.R
```

## Outputs
All outputs are saved to `src/sem/outputs/`:
- `mycorrhizal_sem_summary_*.txt` - Model summaries
- `mycorrhizal_sem_coefficients.csv` - Path coefficients
- `mycorrhizal_sem_rsquared.csv` - R-squared values
- `mycorrhizal_sem_path_diagram.pdf` - Path diagram visualization
- `mycorrhizal_relationships.pdf` - Scatter plots of key relationships
- `am_fungi_coefficients.csv` - AM fungi specific effects
- `ecm_fungi_coefficients.csv` - ECM fungi specific effects

## Key Findings
The analysis tests:
1. Direct environmental effects on mycorrhizal richness
2. Mycorrhizal effects on plant richness (mutualism)
3. Bottom-up trophic effects (plants → animals)
4. Direct mycorrhizal-animal interactions (mycophagy)
5. Differences between AM and ECM fungi effects