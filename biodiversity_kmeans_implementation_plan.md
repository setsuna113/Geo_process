# Biodiversity K-means Implementation Plan

## Overview
This document outlines the implementation plan for an optimized k-means clustering algorithm specifically designed for global biodiversity data with high missing values (90%) and fine spatial resolution (18km grids).

## Final Architecture and Methods Summary

### 1. Distance Metric: Weighted Partial Bray-Curtis
- **Base Metric**: Bray-Curtis distance for ecological data
- **Partial Comparison**: Only compare non-missing feature pairs
- **Dynamic Weighting**: Calculate weights from data completeness/variance
- **Formula**: `sum(w * |u-v|) / sum(w * (u+v))` for valid pairs only

### 2. Adaptive Minimum Features
- **Latitude-based Thresholds**:
  - Arctic (|lat| > 66.5°): 1 feature minimum
  - Temperate (45° < |lat| < 66.5°): 2 features minimum
  - Tropical (|lat| < 45°): 2 features minimum
- **Rationale**: Accounts for systematic data scarcity in polar regions

### 3. Preprocessing Pipeline
- **Transformation**: Log(x+1) to handle zero-inflated species counts
- **Normalization**: Z-score standardization after transformation
- **Missing Data**: Preserve NaN values (no imputation)
- **Zero Handling**: Explicit handling of true zeros vs missing data

### 4. Model Structure: Hierarchical Sparse K-means
- **Stage 1**: Cluster high-quality grids (3-4 valid features)
- **Stage 2**: Assign moderate-quality grids (2 valid features)
- **Stage 3**: Assign sparse grids (1 valid feature) to nearest cluster
- **Optimization**: Prefilter completely empty grids before processing

### 5. Weight Calculation Methods
- **Completeness-based**: Weight by % non-missing values per feature
- **Variance-based**: Weight by information content (variance)
- **Spatial coverage**: Weight by geographic distribution quality
- **Auto mode**: Choose method based on overall data sparsity

### 6. Efficiency Optimizations
- **Sparse Distance Matrix**: Only compute where overlap exists
- **Parallel Processing**: Multi-core distance calculations
- **Chunk Processing**: Handle large datasets in memory-efficient chunks
- **Numba Integration**: JIT compilation for distance metrics

### 7. Configuration Parameters
```yaml
methods:
  kmeans:
    # Core parameters
    n_clusters: 20
    init: 'k-means++'
    n_init: 10
    max_iter: 300
    tol: 1e-4
    
    # Distance and weights
    distance_metric: 'bray_curtis'
    weight_method: 'auto'  # auto-calculate from data
    
    # Adaptive thresholds
    adaptive_mode: 'latitude'
    arctic_boundary: 66.5
    temperate_boundary: 45.0
    arctic_min_features: 1
    temperate_min_features: 2
    tropical_min_features: 2
    
    # Spatial parameters
    grid_size_km: 18
    neighborhood_radius_km: 100
    
    # Preprocessing
    transform: 'log1p'
    normalize: 'standardize'
    
    # Optimization
    prefilter_empty: true
    min_features_prefilter: 1
    use_sparse_distances: true
    n_jobs: -1
```

## Implementation Steps

### Phase 1: Configuration Setup
1. Create `biodiversity_config.yml` with k-means parameters
2. Define `KMeansConfig` dataclass
3. Integrate with existing `BiodiversityConfigManager`

### Phase 2: Core Components
1. **Adaptive Distance Module** (`adaptive_distance.py`)
   - Implement weighted partial Bray-Curtis
   - Add latitude-based threshold logic
   - Integrate with existing partial metrics

2. **Sparse Optimizer** (`sparse_optimizer.py`)
   - Prefiltering of empty grids
   - Sparse distance matrix computation
   - Hierarchical clustering strategy

3. **Core K-means** (`core.py`)
   - Main algorithm implementation
   - Integration of adaptive distances
   - Hierarchical fitting process

### Phase 3: Analyzer Integration
1. Create `KMeansAnalyzer` extending `BaseBiodiversityAnalyzer`
2. Leverage existing data loading and preprocessing
3. Implement k-means specific metrics
4. Integrate with progress tracking system

### Phase 4: Optimization Implementation
1. Parallel distance computation
2. Chunked data processing
3. Memory-efficient sparse matrices
4. Numba optimization for hot paths

### Phase 5: Validation and Visualization
1. Adapt existing validation framework
2. Implement biodiversity-aware silhouette score
3. Create cluster visualization tools
4. Geographic mapping of results

### Phase 6: Testing and Documentation
1. Unit tests for each component
2. Integration tests with sample data
3. Performance benchmarks
4. User documentation

## File Structure
```
src/biodiversity_analysis/methods/kmeans/
├── __init__.py
├── analyzer.py              # Main KMeansAnalyzer class
├── core.py                  # Core k-means algorithm
├── adaptive_distance.py     # Distance metric implementation
├── sparse_optimizer.py      # Sparse data optimizations
├── kmeans_config.py        # Configuration structures
├── weight_calculator.py     # Feature weight calculations
└── visualization.py        # Result visualization
```

## Key Design Decisions

### Why Weighted Partial Bray-Curtis?
- Bray-Curtis is standard for ecological abundance data
- Partial comparison handles 90% missing data
- Dynamic weights adapt to sampling biases

### Why Adaptive Minimum Features?
- Arctic regions systematically lack some taxa (e.g., reptiles)
- Tropical regions have complete sampling
- Fixed threshold would exclude important polar data

### Why Hierarchical Clustering?
- Leverages structure in data quality
- More efficient than treating all grids equally
- Better cluster centers from high-quality data

### Why No Imputation?
- Missing data has ecological meaning (species absence vs. not sampled)
- Imputation would create false patterns
- Partial metrics preserve data integrity

## Expected Outcomes
- Robust clustering despite 90% missing data
- Ecologically meaningful biodiversity groups
- Computationally efficient for millions of grids
- Seamless integration with existing pipeline

## References
- KMD Clustering paper (Zelig et al., 2023) for efficient algorithms
- Existing SOM implementation for infrastructure patterns
- Biodiversity analysis best practices for ecological validity