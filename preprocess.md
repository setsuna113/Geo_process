# Data Preprocessing Pipeline Documentation

## Data Loading Steps

1. **Load Arrow/Parquet Data**
   - Path: `/scratch/yl998/biodiversity_climate_production_1x_corrected.parquet`
   - Use Arrow.Table for memory efficiency (no copy by default)
   - Convert to DataFrame with `copycols=false` to save memory

2. **Spatial Sampling** 
   - If `spatial_step > 1`: Sample every nth row using `df[1:spatial_step:end, :]`
   - Default `spatial_step=1` keeps all data (for MESF caching)

3. **Data Type Conversion**
   - Convert important numeric columns to Float32 for memory efficiency:
     - Climate: bio01, bio04, bio12, bio15
     - Elevation: elevation
     - Fungi: am_fungi_richness, ecm_fungi_richness
     - Plants: plants_richness, plant_richness  
     - Animals: vertebrates_richness, animal_richness, terrestrial_richness

## MESF (Moran's Eigenvector Spatial Filtering) Integration

1. **Cache Directory**: `/scratch/yl998/.cache/mesf`
2. **Cache File Pattern**: `mesf_step{spatial_step}_n10.npy`
3. **MESF Loading Process**:
   - Check if cached MESF exists for given spatial_step
   - If exists: Load from NPZ file and add as `mesf_1` to `mesf_n` columns
   - If not exists: Call Python MESF processor to compute
4. **MESF Computation** (when needed):
   - Export data to temporary Arrow file
   - Call `mesf_processor.py` with parameters:
     - `--spatial-step`: Sampling rate
     - `--n-eigenvectors 10`: Number of spatial eigenvectors
     - `--n-jobs 120`: Parallel processing
   - Load computed MESF and add to DataFrame

## Core Preprocessing Transformations

### 1. Missing Data Handling
- Remove all rows with missing values using `dropmissing(df)`

### 2. Feature Creation
- **Fungi Richness**: `fungi_richness = am_fungi_richness + ecm_fungi_richness`
- **Standardize Naming**:
  - Map `plant_richness` → `plants_richness`
  - Map `animal_richness` or `terrestrial_richness` → `vertebrates_richness`

### 3. Climate Variable Standardization (2SD Method)
For each climate variable (bio01, bio04, bio12, bio15, elevation):
```julia
μ = mean(variable)
σ = std(variable)
variable_std = (variable - μ) / (2σ)
```

### 4. Elevation Squared Term
- Add `elevation_std_sq = elevation_std^2`

### 5. Method-Specific Transformations

#### For GLM Methods (method1, method2):
- **Fungi**: Round to integers for count regression
- **Plants**: Add log transform `plants_richness_log = log(plants_richness + 1)`
- **Vertebrates**: Add log transform `vertebrates_richness_log = log(vertebrates_richness + 1)`

#### For DML Methods (method3, dml):
- Keep raw counts (no log transformation)
- Add interaction term: `plants_x_vertebrates = plants_richness × vertebrates_richness`
- This follows the paper notation: A = [a, b], A° = [a×b]

#### Method2 Specific:
- Create categorical variables:
  - `temperate = 1 if bio04_std > median(bio04_std) else 0`
  - `plant_high = 1 if plants_richness > median(plants_richness) else 0`
- Add interactions:
  - `temperate_x_plant_high = temperate × plant_high`
  - `bio04_std_x_plants_log = bio04_std × plants_richness_log`

## Final Feature Sets

### Base Features (all methods):
- Climate: bio01_std, bio04_std, bio12_std, bio15_std
- Elevation: elevation_std, elevation_std_sq
- Spatial: mesf_1, ..., mesf_n

### Method1 Features:
- Base features + plants_richness_log, vertebrates_richness_log

### Method2 Features:
- Base features + biodiversity + interaction terms

### Method3/DML Features:
- Base features (confounders)
- Treatments: plants_richness, vertebrates_richness, plants_x_vertebrates
- Outcome: fungi_richness (raw counts)

## Memory Optimization Notes
- Use Float32 for all numeric columns
- Arrow backend with `copycols=false` to avoid duplication
- Cache MESF computations to avoid recalculation
- Process data in chunks when possible