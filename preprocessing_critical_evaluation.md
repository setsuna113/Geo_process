# Critical Evaluation of Preprocessing Methods for 95% Zero-Inflated Biodiversity Richness Data

## Executive Summary

After extensive web research and analysis, I provide this critical evaluation of preprocessing methods for your specific data characteristics:
- **95% zero-inflated** species richness counts (not relative abundances)
- **70% missing data** (spatial coverage gaps)
- **4 features**: plants, terrestrial, AM fungi, ECM fungi richness

## 1. Zero-Inflation Handling: Hurdle Transform vs CLR

### ✅ RECOMMENDED: Hurdle Transform
**Evidence:**
- Research shows hurdle models "performed best, with the highest correlations between observed and predicted abundances" for zero-inflated species data
- Explicitly separates presence/absence from abundance - aligns with ecological reality
- Preserves true zeros as meaningful information (species absence)
- Creates interpretable features: presence indicators + conditional abundance

**Implementation in your pipeline:**
```python
compositional_transform='hurdle'  # Correct choice
```

### ❌ NOT RECOMMENDED: CLR Transform
**Critical Issues:**
1. **Destroys presence/absence information**: CLR requires replacing all zeros with pseudocounts, making all species appear "present"
2. **Assumes compositionality**: CLR is designed for data that sums to a constant (like microbiome relative abundances), NOT for species richness counts
3. **Incompatible with diversity metrics**: After CLR, you cannot calculate richness or other presence/absence-based metrics
4. **Inappropriate for count data**: Species richness is absolute counts, not compositional data

**Evidence quote**: "There are no 'absent' species after CLR transformation; samples would have the same number of species 'present', rendering this type of metric useless"

## 2. Spatial Features: Beneficial but Context-Dependent

### ✅ RECOMMENDED: Spatial Lag Features
**Benefits:**
- Provides explicit neighborhood context to SOM
- Helps smooth measurement errors through spatial averaging
- Captures ecological processes that operate across space
- Research shows SOMs successfully use spatial features for "pattern recognition" and "variable relationship assessment"

### ✅ RECOMMENDED: Multi-scale Spatial Statistics
**Benefits:**
- Captures patterns at different ecological scales (5-100km)
- Aligns with known ecological processes operating at multiple scales
- Helps SOM distinguish between local vs regional patterns

### ⚠️ USE WITH CAUTION: Local Moran's I as Features
**Considerations:**
- You already use global Moran's I for convergence checking
- Adding local Moran's I as features could create circular dependencies
- However, local indicators provide different information than global metrics
- Recommend: Include but monitor for overfitting

### ✅ RECOMMENDED: Predictive Moran Eigenvectors (pMEM)
**Benefits:**
- Captures spatial structure orthogonally
- Limited to 10 eigenvectors prevents overfitting
- Research shows spatial self-organization is key in biodiversity patterns
- Quote: "Spatial self-organization...driven by intraspecific competition...is widely observed in stressed environments"

## 3. Critical Warnings and Recommendations

### ⚠️ Feature Explosion Risk
Your current setup creates ~90 features from 4 original features:
- Hurdle: 4 → 8 features (presence + abundance)
- Spatial lags: +8 features
- Multi-scale: +60 features (4 × 3 stats × 5 scales)
- Local Moran's: +4 features
- pMEM: +10 eigenvectors

**Recommendation**: Start with fewer scales (e.g., [10, 50, 100] instead of [5, 10, 25, 50, 100])

### ⚠️ Missing Data Interaction
With 70% missing data, spatial features may propagate missingness:
- Spatial lags of missing neighbors → more missing values
- Multi-scale statistics with sparse coverage → unreliable estimates

**Recommendation**: Use your existing partial distance metrics that handle missing data properly

### ✅ Standardization Method
Your choice of `robust` scaler is appropriate for zero-inflated data with outliers.

## 4. Revised Configuration Recommendation

```yaml
preprocessing_config:
  use_advanced: true
  
  # Zero-inflation - KEEP HURDLE, NOT CLR
  compositional_transform: 'hurdle'
  structural_zero_threshold: 0.95
  pseudo_count_method: 'adaptive'
  
  # Spatial features - REDUCE COMPLEXITY
  spatial_neighbors: 8
  spatial_scales: [10, 50, 100]  # Reduced from 5 scales
  n_moran_eigenvectors: 10      # Good choice
  use_pmem: true
  
  # Keep these as configured
  calculate_diversity_indices: false  # Correct - avoid circular reasoning
  standardize: true
  standardization_method: 'robust'
```

## 5. Evidence Summary

1. **Hurdle models are proven superior** for zero-inflated species data in multiple studies
2. **CLR is fundamentally inappropriate** for species richness (absolute counts, not compositional)
3. **Spatial features enhance SOM performance** but should be used judiciously with missing data
4. **Feature reduction is critical** to avoid curse of dimensionality with only ~5000 samples

## 6. Testing Strategy

1. **Baseline**: Run with `use_advanced: false` (current simple preprocessing)
2. **Hurdle only**: Enable hurdle transform without spatial features
3. **Hurdle + reduced spatial**: Use recommended configuration above
4. **Full spatial**: Test with all spatial features if reduced version improves results

Monitor:
- Convergence speed
- Geographic coherence
- Cluster interpretability
- Computational time

## Conclusion

The advanced preprocessing methods are **suitable with modifications**:
- ✅ Use hurdle transform (NOT CLR) for zero-inflation
- ✅ Include spatial features but reduce complexity
- ✅ Keep robust standardization
- ❌ Avoid CLR transform completely
- ⚠️ Monitor feature explosion and missing data propagation

This evidence-based approach should improve SOM clustering while avoiding the pitfalls of inappropriate transformations for your specific data type.