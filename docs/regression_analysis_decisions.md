# Regression Analysis Decisions for Cross-Taxon Biodiversity Congruence

## Overview
This document outlines the analytical framework for testing the "biodiversity begets biodiversity" hypothesis using species richness data across fungi, plants, and vertebrates.


## Hypothesis Framework
- **H0 (Null)**: Fungi richness is independent of plant/vertebrate richness after controlling for environmental factors
- **H1 (Alternative)**: Positive correlation exists between fungi richness and plant/vertebrate richness (biodiversity begets biodiversity)
- **H2 (Temperate Mismatch)**: Temperate regions (high temperature seasonality) show decoupling between plant and fungi richness - areas with high plant richness have unexpectedly low fungi richness (in GLM this is represented by bio4 * plant interaction)

## Model Selection

### Distribution Choice

#### Simple Decision Metric: Dispersion Ratio
```r
dispersion_ratio <- var(fungi_richness) / mean(fungi_richness)
```

**Decision Rule**:
- If ratio ≈ 1 (typically 0.8-1.2): Use **Poisson GLM**
- If ratio >> 1 (typically > 2): Use **Negative Binomial GLM**

#### Implementation
- **Poisson GLM**: 
  ```r
  model <- glm(fungi_richness ~ predictors, family = poisson())
  ```
- **Negative Binomial GLM**: 
  ```r
  library(MASS)
  model <- glm.nb(fungi_richness ~ predictors)
  ```

**Note**: Species richness data typically shows overdispersion, but always check first!

### Model Structure

#### 1. Base Model (Abiotic Only)
```r
model_null <- glm.nb(fungi_richness ~ bio1 + bio12 + elevation + ME1 + ME2 + ME3...)
```

#### 2. Full Model (Abiotic + Biotic)
```r
model_full <- glm.nb(fungi_richness ~ bio1 + bio12 + elevation + ME1 + ME2 + ME3... + 
                     plants_richness + vertebrates_richness)
```

## Feature Engineering

### Categorical Variables for H2 Testing

For testing the temperate mismatch hypothesis (H2), create categorical variables:

```python
# Create categorical predictors for Method 2 (Threshold Regression with Contrasts)
# Temperate regions: top 25% of temperature seasonality
df['temperate'] = (df['bio4_std'] > df['bio4_std'].quantile(0.75)).astype(int)

# High plant richness: top 25% of plant species
df['plant_high'] = (df['plants_richness'] > df['plants_richness'].quantile(0.75)).astype(int)

# Alternative thresholds (sensitivity analysis)
# Can test with median (0.5) or tertiles (0.67) for robustness
```

### Spatial Autocorrelation Control: Moran Eigenvector Spatial Filtering (MESF)

#### Rationale
- Standard GLMs assume independence between observations
- Ecological data violates this due to spatial autocorrelation
- Raw latitude/longitude terms only capture simple gradients
- MESF captures complex spatial patterns at multiple scales

#### Implementation: Nystrom Approximation for Large Datasets

For datasets with >100k points, use Nystrom approximation with variance-based selection:

```r
# Efficient MESF using Nystrom approximation
nystrom_mesf <- function(coords, 
                        n_landmark = 5000,     # Landmark points for eigendecomposition
                        n_sample = 25000,      # Total sample points
                        variance_threshold = 0.75,  # Explain 75% of spatial variance
                        max_eigenvectors = 10) {     # Cap at 10 eigenvectors
  
  library(spdep)
  library(fields)
  
  n_total <- nrow(coords)
  
  # 1. Select landmark and sample points
  landmark_idx <- sample(1:n_total, min(n_landmark, n_total))
  sample_idx <- sample(1:n_total, min(n_sample, n_total))
  
  # 2. Compute spatial weights for landmarks only
  coords_landmark <- coords[landmark_idx, ]
  nb_landmark <- dnearneigh(coords_landmark, 0, 100, longlat = TRUE)
  W_landmark <- nb2listw(nb_landmark, style = "W", zero.policy = TRUE)
  
  # 3. Eigendecomposition on small landmark matrix
  W_mat <- listw2mat(W_landmark)
  eigen_result <- eigen(W_mat)
  
  # 4. Variance-based selection (5-10 eigenvectors)
  positive_idx <- which(eigen_result$values > 0)
  positive_values <- eigen_result$values[positive_idx]
  
  # Calculate cumulative variance
  cumvar <- cumsum(positive_values) / sum(positive_values)
  n_select <- which(cumvar >= variance_threshold)[1]
  
  # Enforce 5-10 eigenvector range
  n_select <- max(5, min(n_select, max_eigenvectors, length(positive_idx)))
  
  selected_idx <- positive_idx[1:n_select]
  
  # 5. Nystrom extension to sample points
  # Compute distances between samples and landmarks
  D_SL <- rdist.earth(coords[sample_idx, ], coords_landmark)
  W_SL <- exp(-D_SL / 100)  # Gaussian kernel with 100km bandwidth
  
  # Normalize weights
  W_SL <- W_SL / rowSums(W_SL)
  
  # Project eigenvectors to sample points
  eigenvectors_sample <- W_SL %*% eigen_result$vectors[, selected_idx]
  
  # 6. Interpolate to full dataset
  ME_full <- matrix(NA, n_total, n_select)
  
  # Use k-nearest neighbors for fast interpolation
  library(FNN)
  nn <- get.knnx(coords[sample_idx, ], coords, k = min(50, n_sample))
  
  for(i in 1:n_select) {
    # Distance-weighted average
    weights <- 1 / (nn$nn.dist + 0.01)
    weights <- weights / rowSums(weights)
    
    ME_full[, i] <- rowSums(weights * 
                           matrix(eigenvectors_sample[nn$nn.index, i], 
                                  ncol = ncol(nn$nn.index)))
  }
  
  cat("Selected", n_select, "eigenvectors explaining", 
      round(cumvar[n_select] * 100, 1), "% of positive spatial variance\n")
  
  return(ME_full)
}

# Usage example
coords <- cbind(data$longitude, data$latitude)
ME_selected <- nystrom_mesf(coords)
data[, paste0("ME", 1:ncol(ME_selected))] <- ME_selected
```

#### For Smaller Datasets (<100k points)

Use standard MESF with variance-based selection:

```r
# Standard MESF for smaller datasets
standard_mesf <- function(coords, variance_threshold = 0.75, max_eigenvectors = 10) {
  library(spdep)
  library(adespatial)
  
  # Create spatial weights
  nb <- dnearneigh(coords, 0, 100, longlat = TRUE)
  W <- nb2listw(nb, style = "W", zero.policy = TRUE)
  
  # Compute eigenvectors
  ME <- mem(W)
  
  # Variance-based selection
  positive_idx <- which(ME$values > 0)
  positive_values <- ME$values[positive_idx]
  cumvar <- cumsum(positive_values) / sum(positive_values)
  
  n_select <- which(cumvar >= variance_threshold)[1]
  n_select <- max(5, min(n_select, max_eigenvectors, length(positive_idx)))
  
  selected_idx <- positive_idx[1:n_select]
  
  cat("Selected", n_select, "eigenvectors explaining", 
      round(cumvar[n_select] * 100, 1), "% of positive spatial variance\n")
  
  return(ME$vectors[, selected_idx])
}
```

### Environmental Predictors
- **Climate**: Selected bioclim variables (e.g., bio1 = annual mean temperature, bio12 = annual precipitation)
- **Topography**: Elevation
- **Note**: Variable selection should be done to avoid multicollinearity among bioclim variables

## Data Preprocessing

### Preprocessing Decisions

1. **Missing Data Handling**
   - Use **complete case analysis** - drop rows with any missing values
   - Justification: <0.1% missing data where fungi exists (~162M complete rows available)
   - No imputation needed due to negligible data loss
   ```r
   # Filter to fungi rows and drop minimal missing cases
   analysis_data <- data %>%
     filter(!is.na(fungi_richness)) %>%
     drop_na(plants_richness, vertebrates_richness, 
             bio1, bio4, bio12, bio15, elevation)
   ```

2. **Response Variable (Species Richness)**
   - Keep as **raw counts** - no transformation needed
   - Negative binomial GLM is designed to handle count data naturally
   - DO NOT log-transform or normalize the response

3. **Climate Variables (bio1, bio4, bio12, bio15)**
   - **Standardize** using center and scale
   - Recommended: Gelman's approach (divide by 2 SDs for better interpretation)
   ```r
   standardize_2sd <- function(x) {
     (x - mean(x, na.rm = TRUE)) / (2 * sd(x, na.rm = TRUE))
   }
   ```
   - Benefits: Allows comparison of effect sizes, reduces multicollinearity

4. **Elevation**
   - **Standardize** same as climate variables
   - Keep consistent preprocessing across continuous predictors

5. **Biotic Predictors (when used as predictors)**
   - **Both log-transformed and raw counts are valid options**
   - Consider model interpretability vs fit trade-off:
     - **Raw counts**: 
       - Cleaner interaction interpretation
       - "Each additional species" effects
       - Linear relationships on count scale
     - **Log transform**: 
       - Better for multiplicative relationships
       - "Percent change" interpretation
       - May improve model fit for skewed data
   
   ```r
   # Option 1: Raw counts (better interpretability)
   # Use directly: plants_richness, vertebrates_richness
   
   # Option 2: Log transformation (if multiplicative relationships expected)
   data$plants_richness_log <- log(data$plants_richness + 1)
   data$vertebrates_richness_log <- log(data$vertebrates_richness + 1)
   ```
   
   - **Recommendation**: Fit models with both approaches
   - If ΔAIC < 4, choose based on interpretability needs
   - For interaction terms, raw counts often provide clearer ecological interpretation

6. **Spatial Eigenvectors (ME1, ME2, ...)**
   - Already standardized by construction - no preprocessing needed

### Implementation
```r
# Standardize continuous environmental predictors
data$bio1_std <- standardize_2sd(data$bio1)
data$bio4_std <- standardize_2sd(data$bio4)
data$bio12_std <- standardize_2sd(data$bio12)
data$bio15_std <- standardize_2sd(data$bio15)
data$elevation_std <- standardize_2sd(data$elevation)

# Transform biotic predictors when used as predictors
data$plants_richness_log <- log(data$plants_richness + 1)
data$vertebrates_richness_log <- log(data$vertebrates_richness + 1)

# Response remains as raw counts
# fungi_richness - no transformation
```

## Model Specification and Interaction Terms

### Core Model Progression

#### Models for H0 and H1 Testing

```r
# Model 1: Climate only (baseline for H0 test)
fungi_richness ~ bio1_std + bio4_std + bio12_std + bio15_std + 
                 elevation_std + I(elevation_std^2) + ME1 + ME2 + ... + MEn

# Model 2: Add biodiversity (test H1)
fungi_richness ~ bio1_std + bio4_std + bio12_std + bio15_std + 
                 elevation_std + I(elevation_std^2) + ME1 + ME2 + ... + MEn +
                 plants_richness_log + vertebrates_richness_log
```

#### Method 1 for H2 Testing: Continuous Interaction

```r
# Model 3a: Temperate Mismatch with continuous interaction
fungi_richness ~ bio1_std + bio4_std + bio12_std + bio15_std + 
                 elevation_std + I(elevation_std^2) + ME1 + ME2 + ... + MEn +
                 plants_richness_log + vertebrates_richness_log +
                 plants_richness_log:bio4_std  # THE KEY INTERACTION
```

#### Method 2 for H2 Testing: Threshold Regression with Contrasts

```python
# Model 3b: Categorical interaction for temperate mismatch (H2)
# Note: bio4_std may be dropped to avoid collinearity with temperate category

# Option A: Without continuous bio4 (cleaner, recommended)
fungi_richness ~ bio1_std + bio12_std + bio15_std + 
                 elevation_std + I(elevation_std^2) + ME1 + ME2 + ... + MEn +
                 plant_high * temperate  # Expands to: plant_high + temperate + plant_high:temperate

# Option B: With continuous bio4 (more complete control)
fungi_richness ~ bio1_std + bio4_std + bio12_std + bio15_std + 
                 elevation_std + I(elevation_std^2) + ME1 + ME2 + ... + MEn +
                 plant_high * temperate  # May have collinearity issues
```

The interaction term `plant_high:temperate` is the key test for H2:
- **Negative coefficient**: High-plant temperate regions have fewer fungi (supports H2)
- **Positive coefficient**: High-plant temperate regions have more fungi (rejects H2)

#### Testing H2 with emmeans Contrasts

```r
# After fitting Model 3b, use emmeans for proper contrast testing
library(emmeans)

# Get estimated marginal means for all combinations
emm <- emmeans(model, ~ plant_high * temperate)

# Test the specific H2 hypothesis
# Compare high plants in temperate vs non-temperate regions
contrast(emm, list(H2_mismatch = c(0, -1, 0, 1)))
# Interpretation: c(low_nontemp, high_nontemp, low_temp, high_temp)
#                 c(     0,          -1,         0,        1)

# If using Python with rpy2:
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()

r('''
library(MASS)
library(emmeans)

model <- glm.nb(fungi_richness ~ bio1_std + bio12_std + bio15_std + elevation_std +
                ME1 + ME2 + ... + plant_high * temperate, data = df)
                
emm <- emmeans(model, ~ plant_high * temperate)
contrast_result <- contrast(emm, list(H2 = c(0, -1, 0, 1)))
summary(contrast_result)
''')
```

### Optional Additional Interactions

Only add if theoretically justified or if Model 3 shows significant improvement:

1. **Climate-Elevation Interactions**
   ```r
   bio1_std:elevation_std   # Temperature × Elevation
   bio12_std:elevation_std  # Precipitation × Elevation
   ```
   - Tests if climate effects vary with elevation
   - Useful for mountainous regions with strong gradients

2. **Vertebrate-Climate Interaction**
   ```r
   vertebrates_richness_log:bio4_std  # Vertebrate × Temperature Seasonality
   ```
   - Tests if vertebrate effects also vary with seasonality
   - Parallel to plant interaction

3. **Three-way Interaction (Advanced)**
   ```r
   plants_richness_log:bio4_std:bio12_std  # Plant × Temp Seasonality × Precipitation
   ```
   - Only if two-way interaction is significant
   - Tests if the temperate mismatch depends on water availability

## Model Comparison Methods

### 1. Likelihood Ratio Test (LRT)
- **Purpose**: Test if biodiversity predictors significantly improve model fit
- **Implementation**:
```r
anova(model_null, model_full, test = "Chisq")
```
- **Interpretation**: p < 0.05 indicates biodiversity predictors improve model

### 2. Akaike Information Criterion (AIC)
- **Purpose**: Model selection accounting for complexity
- **Implementation**:
```r
AIC(model_null, model_full)
ΔAIC <- AIC(model_null) - AIC(model_full)
```
- **Interpretation**: 
  - ΔAIC > 2: Substantial support for full model
  - ΔAIC > 4: Strong support
  - ΔAIC > 10: Essentially no support for null model

## Complete Analysis Workflow

```r
# 1. Check overdispersion
mean(fungi_richness)
var(fungi_richness)
var(fungi_richness) / mean(fungi_richness)  # If >> 1, use negative binomial

# 2. Create spatial eigenvectors using appropriate method
coords <- cbind(data$longitude, data$latitude)
n_obs <- nrow(data)

if(n_obs > 100000) {
  # Use Nystrom approximation for large datasets
  ME_selected <- nystrom_mesf(coords)
} else {
  # Use standard MESF for smaller datasets
  ME_selected <- standard_mesf(coords)
}

# Add selected eigenvectors to data
n_ME <- ncol(ME_selected)
ME_names <- paste0("ME", 1:n_ME)
data[, ME_names] <- ME_selected

# 3. Fit models with selected climate variables
library(MASS)

# Build formula with selected eigenvectors
ME_formula <- paste(ME_names, collapse = " + ")
null_formula <- as.formula(paste("fungi_richness ~ bio1 + bio4 + bio12 + bio15 + elevation +", ME_formula))
full_formula <- as.formula(paste("fungi_richness ~ bio1 + bio4 + bio12 + bio15 + elevation +", 
                                ME_formula, "+ plants_richness + vertebrates_richness"))

model_null <- glm.nb(null_formula, data = data)
model_full <- glm.nb(full_formula, data = data)

# 4. Test hypothesis
# Likelihood Ratio Test
lrt_result <- anova(model_null, model_full, test = "Chisq")
print(lrt_result)

# AIC comparison
aic_comparison <- data.frame(
  Model = c("Abiotic only", "Abiotic + Biotic"),
  AIC = c(AIC(model_null), AIC(model_full)),
  Delta_AIC = c(0, AIC(model_null) - AIC(model_full))
)
print(aic_comparison)

# 5. Check residual spatial autocorrelation
library(spdep)
# Sample for testing if dataset is large
test_idx <- if(n_obs > 10000) sample(1:n_obs, 10000) else 1:n_obs
coords_test <- coords[test_idx, ]
nb_test <- dnearneigh(coords_test, 0, 100, longlat = TRUE)
W_test <- nb2listw(nb_test, zero.policy = TRUE)

moran_result <- moran.test(residuals(model_full)[test_idx], W_test)
print(moran_result)

# 6. Check final model
summary(model_full)
```

## Model Diagnostics (Optional but Recommended)

### Using DHARMa Package for GLM Validation

DHARMa provides simulation-based residual diagnostics that work for any GLM type:

```r
library(DHARMa)

# 1. Create simulated residuals
simulationOutput <- simulateResiduals(model_full, n = 1000)

# 2. Visual diagnostics
plot(simulationOutput)  # QQ plot and residuals vs predicted

# 3. Test for specific issues
# Overdispersion (even for NB models)
testDispersion(simulationOutput)

# Spatial autocorrelation (verify MESF effectiveness)
testSpatialAutocorrelation(simulationOutput, 
                          x = data$longitude, 
                          y = data$latitude)

# Zero-inflation
testZeroInflation(simulationOutput)

# 4. Interpret results
# - Uniform QQ plot = model assumptions met
# - No patterns in residuals = good fit
# - Failed tests suggest model refinement needed
```

### When to Be Concerned:
- **Overdispersion test p < 0.05**: Consider additional predictors or random effects
- **Spatial autocorrelation p < 0.05**: Add more spatial eigenvectors or check MESF
- **Zero-inflation p < 0.05**: Consider zero-inflated negative binomial (ZINB)

### Quick Alternative Checks:
```r
# Basic overdispersion check for NB
E2 <- resid(model_full, type = "pearson")
N <- nrow(data)
p <- length(coef(model_full)) + 1  # +1 for theta
sum(E2^2) / (N - p)  # Should be close to 1

# Check theta parameter
model_full$theta  # Very small theta (<1) suggests zero-inflation
```

## Model Validation Approach

### Using Full Dataset for Hypothesis Testing
For hypothesis-driven analyses (like testing "biodiversity begets biodiversity"), we use the full dataset without cross-validation:

- **Rationale**: The goal is inference, not prediction
- **Statistical power**: Maximized for detecting subtle interactions
- **Standard practice**: Common in biodiversity literature for hypothesis testing
- **Spatial validation**: Already addressed through MESF

Note: Cross-validation is more appropriate for predictive modeling. For hypothesis testing with large datasets (>100k observations), using the full dataset provides maximum power to detect ecological relationships.

## Model Interpretation and Visualization

### Complete Interpretation Pipeline (4 Methods, Non-Redundant)

Modern biodiversity studies use a comprehensive set of interpretation methods to understand model results. Here's the recommended non-redundant framework:

```r
# COMPLETE INTERPRETATION PIPELINE (4 methods, no redundancy)

# 1. VARIABLE IMPORTANCE - "What matters most?"
#    Output: Ranking table with % variance explained
library(hier.part)
importance <- hier.part(fungi_richness, predictors, 
                       family = "negbinom", gof = "Rsqu")
# Creates: Table showing bio4 explains 28%, plants 22%, etc.

# 2. MARGINAL EFFECTS - "How much do things change?"
#    Output: Table of effects at different climate conditions  
library(margins)
me <- margins(model_full, 
              variables = "plants_richness",
              at = list(bio4_std = c(-1, 0, 1)))
# Creates: "In high seasonality, +10 plant species → +3.2 fungi species"

# 3. PARTIAL DEPENDENCE - "What's the shape?"
#    Output: One plot per main predictor
library(pdp)
pd <- partial(model_full, 
              pred.var = c("plants_richness", "bio4_std", "bio1_std"),
              plot = TRUE)
# Creates: Reveals if relationships are linear, saturating, or threshold

# 4. INTERACTION PLOT - "How does context matter?"
#    Output: 2D plot or faceted plot
library(ggeffects)
interaction_pred <- predict_response(model_full,
                                    terms = c("plants_richness [10:200 by=10]",
                                             "bio4_std [-1, 0, 1]"))
plot(interaction_pred)

# Creates: THE key figure for your temperate mismatch hypothesis
```

### Visualization for H2 Mismatch Pattern (Method 2)

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualize_temperate_mismatch(df, emmeans_results):
    """
    Create comprehensive visualization of the temperate mismatch pattern
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Bar plot of emmeans with confidence intervals
    ax1 = axes[0, 0]
    emm_df = pd.DataFrame({
        'condition': ['Low Plants\nNon-temperate', 'High Plants\nNon-temperate',
                     'Low Plants\nTemperate', 'High Plants\nTemperate'],
        'fungi_mean': emmeans_results['emmeans'],
        'fungi_ci': emmeans_results['SE'] * 1.96
    })
    
    x = np.arange(4)
    colors = ['lightblue', 'darkblue', 'lightcoral', 'darkred']
    bars = ax1.bar(x, emm_df['fungi_mean'], yerr=emm_df['fungi_ci'], 
                   color=colors, capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(emm_df['condition'], rotation=45, ha='right')
    ax1.set_ylabel('Predicted Fungi Richness')
    ax1.set_title('Estimated Marginal Means')
    
    # Add contrast annotation
    if emmeans_results['p_value'] < 0.05:
        ax1.annotate('***', xy=(2.5, max(emm_df['fungi_mean'])), 
                    fontsize=16, ha='center')
    
    # 2. Interaction plot
    ax2 = axes[0, 1]
    plant_levels = ['Low', 'High']
    temperate_lines = {
        'Non-temperate': [emm_df['fungi_mean'][0], emm_df['fungi_mean'][1]],
        'Temperate': [emm_df['fungi_mean'][2], emm_df['fungi_mean'][3]]
    }
    
    for climate, values in temperate_lines.items():
        style = '-' if climate == 'Non-temperate' else '--'
        ax2.plot(plant_levels, values, marker='o', label=climate, 
                linewidth=2, markersize=8, linestyle=style)
    
    ax2.set_xlabel('Plant Richness')
    ax2.set_ylabel('Predicted Fungi Richness')
    ax2.set_title('Interaction: Plant × Climate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot with observed data
    ax3 = axes[1, 0]
    for temp in [0, 1]:
        for plant in [0, 1]:
            mask = (df['temperate'] == temp) & (df['plant_high'] == plant)
            label = f"{'Temp' if temp else 'Non-temp'}, {'High' if plant else 'Low'} plants"
            color = colors[temp * 2 + plant]
            ax3.scatter(df.loc[mask, 'plants_richness'], 
                       df.loc[mask, 'fungi_richness'],
                       alpha=0.3, s=1, label=label, c=color)
    
    ax3.set_xlabel('Plant Richness (continuous)')
    ax3.set_ylabel('Fungi Richness')
    ax3.set_title('Observed Data by Categories')
    ax3.legend(fontsize=8)
    
    # 4. Contrast visualization
    ax4 = axes[1, 1]
    contrast_val = emmeans_results['contrast_estimate']
    contrast_se = emmeans_results['standard_error']
    
    ax4.barh(0, contrast_val, xerr=contrast_se * 1.96, 
            color='red' if contrast_val < 0 else 'green',
            height=0.5, capsize=10)
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Contrast Effect Size')
    ax4.set_title(f'H2 Test: High Plants (Temp - NonTemp)\np = {emmeans_results["p_value"]:.4f}')
    ax4.set_yticks([])
    
    # Add interpretation text
    if contrast_val < 0 and emmeans_results['p_value'] < 0.05:
        ax4.text(0.5, 0.5, 'MISMATCH CONFIRMED\nTemperate regions have\nfewer fungi despite\nhigh plant richness',
                transform=ax4.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    return fig
```


### Interpretation Guidelines

For each method:

1. **Variable Importance**: "Temperature seasonality (bio4) explains 28% of deviance, making it the strongest predictor of fungi richness after controlling for other factors"

2. **Marginal Effects**: "The effect of plant diversity on fungi richness is context-dependent: in high-seasonality environments, each 10-species increase in plants corresponds to 3.2 additional fungi species"

3. **Partial Dependence**: "The relationship between plants and fungi shows [linear/saturating/threshold] pattern, with diminishing returns above 150 plant species"

4. **Interaction Plot**: "The positive plant-fungi relationship strengthens with increasing temperature seasonality, supporting the temperate mismatch hypothesis"

### Interpreting H2 Results (Method 2 with Contrasts)

```python
def interpret_h2_results(emmeans_results, model_summary):
    """
    Provide standardized interpretation of H2 contrast results
    """
    
    contrast_est = emmeans_results['contrast_estimate']
    p_value = emmeans_results['p_value']
    se = emmeans_results['standard_error']
    
    # Calculate effect size metrics
    ci_lower = contrast_est - 1.96 * se
    ci_upper = contrast_est + 1.96 * se
    
    # Determine support for H2
    if p_value < 0.05 and contrast_est < 0:
        conclusion = "SUPPORTS H2: Significant temperate mismatch detected"
        explanation = f"""
        High-plant temperate regions have {abs(contrast_est):.1f} fewer fungi species
        than high-plant non-temperate regions (95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]).
        This confirms the biodiversity decoupling pattern in temperate zones.
        """
    elif p_value < 0.05 and contrast_est > 0:
        conclusion = "REJECTS H2: Opposite pattern detected"
        explanation = f"""
        High-plant temperate regions have {contrast_est:.1f} MORE fungi species
        than high-plant non-temperate regions, contrary to the mismatch hypothesis.
        """
    else:
        conclusion = "NO SUPPORT for H2: No significant difference"
        explanation = f"""
        No significant difference in fungi richness between high-plant temperate
        and non-temperate regions (contrast = {contrast_est:.1f}, p = {p_value:.3f}).
        """
    
    # Report the emmeans for context
    emmeans_interpretation = f"""
    Estimated marginal means:
    - Low plants, non-temperate: {emmeans_results['emmeans'][0]:.1f} fungi species
    - High plants, non-temperate: {emmeans_results['emmeans'][1]:.1f} fungi species
    - Low plants, temperate: {emmeans_results['emmeans'][2]:.1f} fungi species
    - High plants, temperate: {emmeans_results['emmeans'][3]:.1f} fungi species
    
    Key comparison (H2 test):
    High plants in temperate ({emmeans_results['emmeans'][3]:.1f}) vs 
    High plants in non-temperate ({emmeans_results['emmeans'][1]:.1f})
    Difference: {contrast_est:.1f} species (p = {p_value:.4f})
    """
    
    return {
        'conclusion': conclusion,
        'explanation': explanation,
        'emmeans_table': emmeans_interpretation,
        'publication_statement': f"The contrast analysis revealed that high-plant temperate regions had {abs(contrast_est):.1f} {'fewer' if contrast_est < 0 else 'more'} fungi species than high-plant non-temperate regions (contrast = {contrast_est:.2f} ± {se:.2f} SE, p = {p_value:.3f})"
    }
```

### Key Differences in Interpretation: Interaction Coefficient vs Contrast

**Original Method (Interaction Coefficient):**
- Tests: `plants_richness:bio4_std` coefficient
- Interpretation: "The slope of plant-fungi relationship changes with seasonality"
- Limited: Only tells you the relationship changes, not the actual pattern

**Method 2 (emmeans Contrast):**
- Tests: Specific comparison of high-plant regions across climates
- Interpretation: "High-plant temperate regions have X fewer/more fungi than high-plant non-temperate"
- Direct: Explicitly tests the mismatch hypothesis
- Provides: Actual species difference with confidence intervals

### Publication Output

Standard presentation order:
- **Table 2**: Variable importance ranking with % deviance explained
- **Table 3**: Marginal effects at three seasonality levels
- **Figure 2**: Partial dependence plots for top 3 predictors
- **Figure 3**: Interaction plot (main result for hypothesis)
- **Supplement**: Full coefficient table with standardized and unstandardized estimates

## Expected Outcomes
- If LRT p-value < 0.05 AND ΔAIC > 2: Support for biodiversity begets biodiversity hypothesis
- Positive coefficients for plants_richness and vertebrates_richness indicate positive relationships
- Effect sizes can be interpreted as: exp(coefficient) = multiplicative effect on fungi richness

## Notes
- This framework controls for shared environmental responses through abiotic predictors
- Spatial autocorrelation is addressed through MESF
- No domain-specific ecological knowledge required for statistical interpretation
- Results are robust to model specification through dual testing approach (LRT + AIC)

## Alternative Method: Double Machine Learning (DML)

### Overview
Double Machine Learning provides causal effect estimation for H0-H1 testing (plants/vertebrates → fungi) using machine learning to control for confounders while maintaining valid inference.

### DML Methodology for Biodiversity Analysis

#### Variable Definitions
- **Y (Outcome)**: Fungi richness (raw counts)
- **A (Treatments)**: 
  - A₁ = Plants richness (raw counts)
  - A₂ = Vertebrates richness (raw counts)
- **X (Confounders)**: bio1, bio4, bio12, bio15, elevation, ME1...MEn

#### Cross-fitting Procedure (K=5 folds)

```python
# Step 1: Split data into 5 folds
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 2: For each fold k, train on other K-1 folds
for k, (train_idx, test_idx) in enumerate(kf.split(X)):
    # Train nuisance functions on 80% of data (129.6M rows)
    # Model 1: Predict E[fungi|climate,elevation,ME]
    model_Y = LGBMRegressor(**params).fit(X[train_idx], fungi[train_idx])
    
    # Model 2: Predict E[plants|climate,elevation,ME]
    model_A1 = LGBMRegressor(**params).fit(X[train_idx], plants[train_idx])
    
    # Model 3: Predict E[vertebrates|climate,elevation,ME]
    model_A2 = LGBMRegressor(**params).fit(X[train_idx], vertebrates[train_idx])
    
    # Get residuals on held-out fold (20% of data)
    residuals_Y[test_idx] = fungi[test_idx] - model_Y.predict(X[test_idx])
    residuals_A1[test_idx] = plants[test_idx] - model_A1.predict(X[test_idx])
    residuals_A2[test_idx] = vertebrates[test_idx] - model_A2.predict(X[test_idx])

# Step 3: Pool ALL residuals and run final regression
from statsmodels.regression.linear_model import OLS
final_model = OLS(residuals_Y, np.column_stack([residuals_A1, residuals_A2])).fit()

# Extract causal effects
θ₁ = final_model.params[0]  # Causal effect of plants on fungi
θ₂ = final_model.params[1]  # Causal effect of vertebrates on fungi
```

#### Key Points
- **No validation set**: All data contributes to final estimate
- **Pooling, not averaging**: Residuals from all folds are pooled for final regression
- **Each row used twice**: Once for training (in 4 models), once for residual calculation

### DML Configuration

```python
DML_CONFIG = {
    # Data
    'n_samples': 'all_162M',  # Use full dataset
    'missing_data': 'complete_case',  # Same as GLM approach
    
    # Variables (raw counts, no transformation)
    'outcome': 'fungi_richness',
    'treatments': ['plants_richness', 'vertebrates_richness'],
    'controls': ['bio1', 'bio4', 'bio12', 'bio15', 'elevation'] + ME_vars,
    
    # Machine Learning
    'ml_algorithm': 'LightGBM',
    'k_folds': 5,
    'threads_per_fold': 80,  # Optimal for 256-core system
    
    # LightGBM parameters (no tuning needed)
    'lgb_params': {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,  # Between Poisson and Gamma
        'num_leaves': 255,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'num_threads': 80,
    },
    
    # Analysis
    'interaction_term': False,  # Primary analysis without interaction
    'standard_errors': 'asymptotic',  # From DML theory
}
```

### Interpretation

The ML models answer:
- **μ_Y(X)**: "Given this climate/location, how many fungi species do we expect?"
- **μ_A1(X)**: "Given this climate/location, how many plant species do we expect?"
- **μ_A2(X)**: "Given this climate/location, how many vertebrate species do we expect?"

The residuals capture "unexpected" richness after accounting for shared environmental drivers, isolating causal biodiversity relationships:
- **θ₁**: Areas with more plants than environmentally expected have θ₁ more fungi
- **θ₂**: Areas with more vertebrates than environmentally expected have θ₂ more fungi

### Advantages over GLM
- **Flexible functional forms**: ML captures non-linear environmental relationships
- **No model specification**: Avoids assumptions about parametric forms
- **Robust to confounding**: Better handles complex environmental drivers

### Integration with GLM Results
- **Primary analysis**: GLM (more interpretable, standard in ecology)
- **Robustness check**: DML (confirms results with fewer assumptions)
- **Reporting**: "DML analysis confirmed positive effects of plants (θ₁=X, p<0.001) and vertebrates (θ₂=Y, p<0.001) on fungi richness, supporting the GLM findings"

## Implementation Checklist

### Phase 1: Data Preparation
- [ ] Load 162M rows with fungi richness data
- [ ] Apply complete case analysis (drop <0.1% missing)
- [ ] Generate spatial eigenvectors (MESF with Nystrom approximation)
- [ ] Standardize climate variables (2SD method)
- [ ] Create categorical variables for H2 testing (temperate, plant_high)

### Phase 2: GLM Analysis (Primary)
- [ ] Test H0 vs H1 with Negative Binomial GLM
- [ ] Generate variable importance rankings
- [ ] Create marginal effects tables
- [ ] Produce interaction visualizations
- [ ] Validate with DHARMa diagnostics

### Phase 3: DML Analysis (Robustness)
- [ ] Configure LightGBM with default parameters
- [ ] Run 5-fold cross-fitting (80 threads per fold)
- [ ] Train 3 nuisance models (fungi, plants, vertebrates)
- [ ] Calculate residuals for all 162M rows
- [ ] Pool residuals and estimate causal effects
- [ ] Compute asymptotic standard errors

### Phase 4: Results Integration
- [ ] Compare GLM and DML estimates
- [ ] Create publication tables (coefficients, effects, contrasts)
- [ ] Generate main figures (4-panel H2 visualization)
- [ ] Write results narrative with proper interpretation
- [ ] Document any divergence between methods

## Future Extensions

### Sensitivity Analysis
- Test robustness to methodological choices
- E.g., varying MESF variance threshold (75% vs 80%)
- Alternative spatial autocorrelation approaches

### Post-hoc Tests
- If interaction is significant, test specific contrasts
- E.g., "Is the plant-fungi relationship significantly different between tropical vs temperate regions?"
- Regional submodels for detailed patterns