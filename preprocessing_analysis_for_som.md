# Analysis of Preprocessing Methods for SOM

## Available Preprocessing Methods

### 1. **Spatial Lag Features** ✅ HIGHLY SUITABLE
- **Implementation**: `_create_spatial_lags()` in spatial_features.py
- **What it does**: Calculates average richness values of spatial neighbors
- **Why suitable for SOM**: 
  - Provides explicit spatial context beyond coordinate-based distance
  - Helps SOM understand regional patterns
  - Reduces noise by incorporating neighborhood information
  - Compatible with partial Bray-Curtis distance

### 2. **CLR Transform** ⚠️ CONDITIONALLY SUITABLE
- **Implementation**: `_clr_transform()` in zero_inflation.py
- **What it does**: Centered log-ratio transformation for compositional data
- **Considerations for SOM**:
  - PROS: Handles compositional nature of species assemblages
  - CONS: Assumes data sums to a constant (not true for richness)
  - RECOMMENDATION: Use **hurdle transform** instead for richness data

### 3. **Hurdle Transform** ✅ HIGHLY SUITABLE
- **Implementation**: `_hurdle_transform()` in zero_inflation.py
- **What it does**: Two-stage transformation:
  - Stage 1: Binary presence/absence
  - Stage 2: Log abundance conditional on presence
- **Why suitable for SOM**:
  - Perfectly handles 95% zero-inflation
  - Separates "no species" from "low richness"
  - Maintains interpretability
  - Works well with partial Bray-Curtis

### 4. **Multi-scale Spatial Statistics** ✅ HIGHLY SUITABLE
- **Implementation**: `_multiscale_statistics()` in spatial_features.py
- **What it does**: Calculates mean, std, range at 5km, 10km, 25km, 50km, 100km scales
- **Why suitable for SOM**:
  - Captures patterns at different ecological scales
  - Provides hierarchical spatial context
  - Helps identify regional vs local patterns

### 5. **Local Moran's I** ✅ SUITABLE
- **Implementation**: `_calculate_local_morans()` in spatial_features.py
- **What it does**: Identifies spatial clusters and outliers for each location
- **Why suitable for SOM**:
  - Identifies hotspots and coldspots
  - Adds spatial autocorrelation information
  - Helps SOM preserve spatial structure

### 6. **Moran Eigenvectors (MEM/pMEM)** ⚠️ USE WITH CAUTION
- **Implementation**: `_moran_eigenvector_maps()` and `_predictive_moran_eigenvectors()`
- **What it does**: Decomposes spatial structure into orthogonal components
- **Considerations**:
  - PROS: Captures multi-scale spatial patterns
  - CONS: May add too many features (20 eigenvectors)
  - RECOMMENDATION: Use only top 5-10 eigenvectors

## Recommended Preprocessing Pipeline for SOM

```python
# 1. Handle zero-inflation with hurdle transform
preprocessor = ZeroInflationPreprocessor(
    compositional_transform='hurdle',  # NOT 'clr' for richness data
    pseudo_count_method='adaptive'
)

# 2. Add spatial features
spatial_engineer = SpatialFeatureEngineering(
    n_neighbors=8,
    scales=[5, 10, 25, 50, 100],  # Multi-scale
    use_pmem=True,  # Predictive MEM
    pmem_dwf='exponential'
)

# 3. Feature selection
# Original: [plants, terrestrial, am_fungi, ecm_fungi] = 4 features
# After preprocessing: ~50-60 features (need to be selective)
```

## Key Differences from Current SOM Preprocessing

Current SOM uses only:
- Log(x+1) transform
- MinMax scaling

New preprocessing would add:
- Explicit handling of 95% zeros (hurdle transform)
- Spatial context at multiple scales
- Neighborhood information
- Spatial autocorrelation patterns

## Evidence for Suitability

1. **From literature**: Hurdle models are standard for zero-inflated ecological data (Zuur et al., 2009)
2. **From data characteristics**: 95% zeros require special handling beyond log transform
3. **From SOM theory**: Additional spatial features provide context without changing distance metric
4. **From ecological theory**: Multi-scale patterns are fundamental in biodiversity

## Implementation Priority

1. **HIGH**: Hurdle transform (replaces current log1p)
2. **HIGH**: Spatial lag features 
3. **HIGH**: Multi-scale spatial statistics
4. **MEDIUM**: Local Moran's I
5. **LOW**: MEM eigenvectors (experimental)