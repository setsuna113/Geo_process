# Regression Analysis Decisions for Cross-Taxon Biodiversity Congruence

## Overview
This document outlines the analytical framework for testing the "biodiversity begets biodiversity" hypothesis using species richness data across fungi, plants, and vertebrates.

## Hypothesis Framework
- **H0 (Null)**: Fungi richness is independent of plant/vertebrate richness after controlling for environmental factors
- **H1 (Alternative)**: Positive correlation exists between fungi richness and plant/vertebrate richness (biodiversity begets biodiversity)

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

1. **Response Variable (Species Richness)**
   - Keep as **raw counts** - no transformation needed
   - Negative binomial GLM is designed to handle count data naturally
   - DO NOT log-transform or normalize the response

2. **Climate Variables (bio1, bio4, bio12, bio15)**
   - **Standardize** using center and scale
   - Recommended: Gelman's approach (divide by 2 SDs for better interpretation)
   ```r
   standardize_2sd <- function(x) {
     (x - mean(x, na.rm = TRUE)) / (2 * sd(x, na.rm = TRUE))
   }
   ```
   - Benefits: Allows comparison of effect sizes, reduces multicollinearity

3. **Elevation**
   - **Standardize** same as climate variables
   - Keep consistent preprocessing across continuous predictors

4. **Biotic Predictors (when used as predictors)**
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

5. **Spatial Eigenvectors (ME1, ME2, ...)**
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

Based on the "Temperate Mismatch" hypothesis testing whether plant-fungi relationships vary with climate:

```r
# Model 1: Climate only (baseline)
fungi_richness ~ bio1_std + bio4_std + bio12_std + bio15_std + 
                 elevation_std + I(elevation_std^2) + ME1 + ME2 + ... + MEn

# Model 2: Add biodiversity (main effects)
fungi_richness ~ bio1_std + bio4_std + bio12_std + bio15_std + 
                 elevation_std + I(elevation_std^2) + ME1 + ME2 + ... + MEn +
                 plants_richness_log + vertebrates_richness_log

# Model 3: Temperate Mismatch (key hypothesis)
fungi_richness ~ bio1_std + bio4_std + bio12_std + bio15_std + 
                 elevation_std + I(elevation_std^2) + ME1 + ME2 + ... + MEn +
                 plants_richness_log + vertebrates_richness_log +
                 plants_richness_log:bio4_std  # THE KEY INTERACTION
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

## Expected Outcomes
- If LRT p-value < 0.05 AND ΔAIC > 2: Support for biodiversity begets biodiversity hypothesis
- Positive coefficients for plants_richness and vertebrates_richness indicate positive relationships
- Effect sizes can be interpreted as: exp(coefficient) = multiplicative effect on fungi richness

## Notes
- This framework controls for shared environmental responses through abiotic predictors
- Spatial autocorrelation is addressed through MESF
- No domain-specific ecological knowledge required for statistical interpretation
- Results are robust to model specification through dual testing approach (LRT + AIC)