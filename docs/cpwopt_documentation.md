# CP-WOPT Tensor Factorization Documentation

## Overview

CP-WOPT (CANDECOMP/PARAFAC with Weighted Optimization) is a tensor factorization method designed for analyzing sparse, high-dimensional biodiversity data. It decomposes a 3-way tensor (latitude × longitude × species) into interpretable components that capture co-occurring spatial and species patterns.

## Key Features

- **Handles Extreme Sparsity**: Optimized for biodiversity data with >95% sparsity
- **Weighted Optimization**: Accounts for missing values vs. true zeros
- **Memory Efficient**: Mini-batch processing for large-scale datasets
- **Interpretable Components**: Reveals ecological patterns and biodiversity hotspots
- **Comprehensive Reporting**: Automated visualization and interpretation

## Usage

### Command Line

```bash
# Run CP-WOPT analysis
./run_analysis.sh --experiment test_cpwopt --method cpwopt

# With specific input file
./run_analysis.sh --experiment cpwopt_custom --method cpwopt --input outputs/biodiversity_data.parquet

# Run with monitoring
./run_analysis.sh --experiment production_cpwopt --method cpwopt --monitor

# Run as daemon process
./run_analysis.sh --experiment cpwopt_analysis --method cpwopt --daemon --process-name cpwopt_prod
```

### Configuration

Add CP-WOPT experiments to `config.yml`:

```yaml
experiments:
  test_cpwopt:
    input_path: data/test_biodiversity.parquet
    analysis_method: cpwopt
    rank: 5                      # Number of components
    max_iterations: 100          # Maximum ALS iterations
    convergence_threshold: 1e-4  # Convergence criterion
    use_minibatch: true         # Enable memory-efficient processing
    minibatch_size: 500         # Samples per mini-batch
    preprocessing:
      lat_bins: 20              # Latitude grid bins
      lon_bins: 20              # Longitude grid bins
      aggregation: sum          # How to aggregate multiple observations
      transform: log1p          # Data transformation (none|log1p|sqrt|asinh)
    generate_plots: true        # Generate visualization plots
    save_factors: true          # Save factor matrices
    output_dir: outputs/cpwopt/test
```

### Advanced Configuration

For production runs with large datasets:

```yaml
experiments:
  production_cpwopt:
    input_path: /data/biodiversity_full.parquet
    analysis_method: cpwopt
    rank: 10                    # More components for complex data
    max_iterations: 200
    convergence_threshold: 1e-5
    initialization: svd         # Use SVD-based initialization
    regularization: 0.01        # L2 regularization strength
    use_minibatch: true
    minibatch_size: 1000
    preprocessing:
      lat_bins: 50              # Finer spatial resolution
      lon_bins: 50
      aggregation: mean         # Average for dense sampling
      transform: asinh          # Better for heavy-tailed distributions
    random_state: 42            # For reproducibility
    verbose: true
    output_dir: /scratch/outputs/cpwopt/production
```

## Algorithm Details

### Tensor Construction

1. **Spatial Binning**: Observations are aggregated into spatial grid cells
2. **Species Dimension**: Each species becomes a tensor slice
3. **Weight Mask**: Distinguishes missing data from true zeros

### CP Decomposition

The algorithm decomposes the tensor X into rank-R components:

```
X ≈ Σ(r=1 to R) λr * (ar ⊗ br ⊗ cr)
```

Where:
- `ar`: Latitude factor for component r
- `br`: Longitude factor for component r  
- `cr`: Species factor for component r
- `λr`: Component weight
- `⊗`: Outer product

### Optimization

Uses Alternating Least Squares (ALS) with modifications for sparse data:
- Weighted updates only consider observed entries
- Mini-batch processing for memory efficiency
- Adaptive convergence criteria

## Output Interpretation

### Factor Matrices

1. **Latitude Factors** (`latitude_factors.npy`):
   - Shape: (n_lat_bins, rank)
   - Each column represents latitudinal loading for a component

2. **Longitude Factors** (`longitude_factors.npy`):
   - Shape: (n_lon_bins, rank)
   - Each column represents longitudinal loading for a component

3. **Species Factors** (`species_factors.npy`):
   - Shape: (n_species, rank)
   - Each column shows species contributions to components

### Visualizations

1. **Spatial Components** (`spatial_component_*.png`):
   - Geographic distribution of each component
   - Identifies biodiversity hotspots and gradients

2. **Species Loadings** (`component_*_loadings.png`):
   - Top contributing species for each component
   - Reveals species associations

3. **Reconstruction Quality** (`reconstruction_error.png`):
   - Model fit across space
   - Identifies well/poorly modeled regions

4. **Component Importance** (`component_magnitudes.png`):
   - Relative importance of each component
   - Guides interpretation focus

### Metrics

- **Fit**: Proportion of variance explained (0-1)
- **Relative Error**: Reconstruction error relative to data magnitude
- **Sparsity**: Proportion of zero entries
- **Convergence**: Number of iterations until convergence

## Interpretation Guide

### Component Types

1. **Latitudinal Gradients**: 
   - Smooth latitude factors indicate climate-driven patterns
   - Example: Tropical vs. temperate species assemblages

2. **Regional Hotspots**:
   - Peaked spatial factors show biodiversity centers
   - Example: Endemic species in isolated regions

3. **Habitat Associations**:
   - Correlated species in factors reveal habitat preferences
   - Example: Wetland vs. forest species groups

4. **Widespread Patterns**:
   - Uniform spatial factors with diverse species
   - Example: Generalist species distributions

## Best Practices

### Data Preparation

1. **Coordinate Standardization**: Ensure consistent lat/lon format
2. **Species Selection**: Include species with sufficient observations
3. **Outlier Handling**: Consider removing extreme outliers
4. **Missing Data**: Distinguish true absences from no sampling

### Parameter Selection

1. **Rank Selection**:
   - Start with 5-10 components
   - Increase until fit plateaus
   - Consider interpretability vs. fit trade-off

2. **Grid Resolution**:
   - Balance between detail and sparsity
   - 20×20 to 50×50 typically works well
   - Finer grids need more data

3. **Transformations**:
   - `log1p`: General purpose, handles zeros
   - `sqrt`: Reduces influence of high counts
   - `asinh`: Good for heavy-tailed distributions
   - `none`: When counts are already normalized

### Validation

1. **Cross-validation**: Use held-out data to assess generalization
2. **Core Consistency**: Check if model follows CP structure
3. **Stability**: Run with different initializations
4. **Ecological Validity**: Verify patterns make ecological sense

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce grid resolution
   - Enable mini-batching
   - Decrease minibatch_size

2. **Poor Convergence**:
   - Increase max_iterations
   - Try different initialization
   - Add regularization

3. **Uninterpretable Components**:
   - Reduce rank
   - Try different transformations
   - Check for outliers

4. **Very Low Fit**:
   - Data may be too sparse
   - Increase rank
   - Check preprocessing

## Integration with Pipeline

CP-WOPT integrates seamlessly with the biodiversity analysis pipeline:

1. **Input**: Accepts parquet files from data export stage
2. **Processing**: Runs as standalone analysis method
3. **Output**: Generates reports in specified directory
4. **Monitoring**: Supports progress tracking and resource monitoring

## References

- Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. SIAM review, 51(3), 455-500.
- Acar, E., et al. (2011). Scalable tensor factorizations for incomplete data. Chemometrics and Intelligent Laboratory Systems, 106(1), 41-56.

## Examples

### Example 1: Regional Biodiversity Analysis

```yaml
experiments:
  regional_biodiversity:
    input_path: data/amazon_species.parquet
    analysis_method: cpwopt
    rank: 8
    preprocessing:
      lat_bins: 30
      lon_bins: 30
      transform: log1p
    output_dir: outputs/amazon_cpwopt
```

### Example 2: Invasive Species Detection

```yaml
experiments:
  invasive_patterns:
    input_path: data/species_with_invasive.parquet
    analysis_method: cpwopt
    rank: 5
    preprocessing:
      transform: sqrt  # Reduce influence of abundant invasives
      aggregation: max # Capture presence
    output_dir: outputs/invasive_analysis
```

### Example 3: Temporal Analysis

```yaml
experiments:
  temporal_cpwopt:
    input_path: data/species_2020_2023.parquet
    analysis_method: cpwopt
    rank: 6
    # Additional temporal handling would be in preprocessing
    output_dir: outputs/temporal_patterns
```