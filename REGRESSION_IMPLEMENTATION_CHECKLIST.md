# Biodiversity Regression Analysis Implementation Checklist

## Overview
Complete step-by-step implementation plan for the biodiversity regression analysis framework based on `/home/yl998/dev/geo/docs/regression_analysis_decisions.md`. This implementation extends the existing ML infrastructure to support GLM, DML, and comprehensive hypothesis testing.

## Architecture Overview

```
src/machine_learning/
├── models/
│   ├── glm/
│   │   ├── negative_binomial.py
│   │   ├── poisson.py
│   │   └── glm_diagnostics.py
│   ├── dml/
│   │   ├── double_ml.py
│   │   └── nuisance_models.py
│   └── (existing models)
├── spatial/
│   ├── mesf.py
│   └── nystrom_mesf.py
├── research/
│   ├── biodiversity_regression.py
│   ├── methods/
│   │   ├── method1_continuous.py
│   │   ├── method2_categorical.py
│   │   └── method3_dml.py
│   └── interpretation/
│       ├── hypothesis_testing.py
│       ├── glm_diagnostics.py
│       ├── visualization.py
│       └── reporting.py
└── (existing structure)
```

---

## Phase 1: Configuration System Setup

### 1.1 Create ML Configuration File
**File**: `config/ml/ml_config.yml`

```yaml
# config/ml/ml_config.yml
regression_analysis:
  # Data preprocessing (lines 199-277)
  preprocessing:
    missing_data:
      strategy: "complete_case"  # Drop rows with any missing (<0.1% data loss)
      log_message: "Applying complete case analysis"
    
    standardization:
      method: "gelman_2sd"  # Divide by 2 SDs for better interpretation (line 223)
      apply_to: ["bio1", "bio4", "bio12", "bio15", "elevation"]
      log_message: "Standardizing continuous predictors using 2SD method"
    
    response_variable:
      transformation: "none"  # Keep raw counts for GLM (line 217)
      log_message: "Using raw species richness counts"
    
    biotic_predictors:
      transformation: "both"  # Test both raw and log (lines 235-257)
      log_message: "Testing both raw and log-transformed biotic predictors"
  
  # Model specifications (lines 284-384)
  models:
    glm:
      distribution:
        decision_metric: "dispersion_ratio"  # var/mean (lines 16-24)
        poisson_threshold: [0.8, 1.2]
        use_negative_binomial: true  # Default for overdispersed data
        log_message: "Checking overdispersion: var/mean ratio"
      
      ridge_alpha: 1.0  # For potential Ridge GLM
      max_iter: 1000
      solver: "auto"
    
    lightgbm:  # For DML (lines 849-859)
      objective: "tweedie"
      tweedie_variance_power: 1.5
      num_leaves: 255
      learning_rate: 0.05
      feature_fraction: 0.8
      bagging_fraction: 0.8
      bagging_freq: 5
      min_data_in_leaf: 100
      num_threads: 80
      log_message: "Configuring LightGBM for DML nuisance models"
  
  # Spatial analysis (lines 69-192)
  spatial:
    mesf:
      variance_threshold: 0.75  # Explain 75% of spatial variance
      max_eigenvectors: 10
      min_eigenvectors: 5
      distance_threshold: 100  # km for spatial neighbors
      log_message: "Computing Moran Eigenvector Spatial Filtering"
    
    nystrom:
      enable_for_large_data: true
      size_threshold: 100000  # Use Nystrom if >100k points
      n_landmark: 5000
      n_sample: 25000
      log_message: "Using Nystrom approximation for large dataset"
  
  # Feature engineering (lines 53-67)
  feature_engineering:
    h2_testing:
      temperate_threshold: 0.75  # Top 25% temperature seasonality
      plant_high_threshold: 0.75  # Top 25% plant richness
      alternative_thresholds: [0.5, 0.67]  # For sensitivity analysis
      log_message: "Creating categorical variables for H2 testing"
  
  # Cross-validation (lines 792-828)
  dml:
    k_folds: 5
    threads_per_fold: 80
    save_residuals: true
    log_message: "Running 5-fold cross-fitting for DML"
  
  # Hypothesis testing thresholds
  hypothesis_testing:
    alpha: 0.05
    aic_threshold: 2  # ΔAIC > 2 for model selection
    log_messages:
      h0: "Testing H0: Independence of fungi from plants/vertebrates"
      h1: "Testing H1: Biodiversity begets biodiversity"
      h2: "Testing H2: Temperate mismatch hypothesis"
  
  # Output configuration
  output:
    save_diagnostics: true
    save_predictions: true
    save_visualizations: true
    formats: ["parquet", "csv", "pickle"]
    log_message: "Saving all outputs to {output_dir}"
```

### 1.2 Implementation Checklist
- [ ] Create `config/ml/ml_config.yml` with above structure
- [ ] Add config loader in `src/machine_learning/__init__.py`
- [ ] Create `MLConfig` class to parse and validate configuration
- [ ] Add logging statements for each configuration load

**Logger implementation**:
```python
logger.info(f"Loading ML configuration from {config_path}")
logger.info(f"Dispersion ratio: {dispersion_ratio:.2f}")
logger.info(f"Selected distribution: {'Negative Binomial' if overdispersed else 'Poisson'}")
```

---

## Phase 2: Spatial Analysis Implementation

### 2.1 MESF Implementation (lines 69-192)
**File**: `src/machine_learning/spatial/mesf.py`

- [ ] Implement `StandardMESF` class for datasets <100k points
  - [ ] Create spatial weights using `dnearneigh` (100km threshold)
  - [ ] Compute eigenvectors with variance-based selection
  - [ ] Log: `logger.info(f"Selected {n_select} eigenvectors explaining {cumvar:.1f}% variance")`
  
- [ ] Implement `NystromMESF` class for datasets >100k points
  - [ ] Sample landmark points (5000 default)
  - [ ] Compute spatial weights for landmarks only
  - [ ] Use Nystrom extension to full dataset
  - [ ] Log: `logger.info(f"Nystrom approximation: {n_landmark} landmarks, {n_sample} samples")`

### 2.2 Integration Checklist
- [ ] Add MESF selection logic based on dataset size
- [ ] Integrate with preprocessing pipeline
- [ ] Add to feature matrix as ME1, ME2, ..., MEn
- [ ] Validate spatial autocorrelation reduction

---

## Phase 3: GLM Implementation

### 3.1 Base GLM Classes (lines 25-35, 438-443)
**File**: `src/machine_learning/models/glm/negative_binomial.py`

- [ ] Create `NegativeBinomialGLM(BaseMLAnalyzer)` class
  - [ ] Implement using `statsmodels.genmod.families.NegativeBinomial`
  - [ ] Add dispersion parameter estimation
  - [ ] Log: `logger.info(f"Fitting Negative Binomial GLM, theta={theta:.2f}")`

- [ ] Create `PoissonGLM(BaseMLAnalyzer)` class
  - [ ] Implement using `statsmodels.genmod.families.Poisson`
  - [ ] Add overdispersion check
  - [ ] Log: `logger.info("Fitting Poisson GLM")`

### 3.2 GLM Diagnostics (lines 473-520)
**File**: `src/machine_learning/models/glm/glm_diagnostics.py`

- [ ] Implement `DHARMaDiagnostics` class
  - [ ] Simulated residuals (n=1000)
  - [ ] QQ plot generation
  - [ ] Overdispersion test
  - [ ] Spatial autocorrelation test
  - [ ] Zero-inflation test
  - [ ] Log: `logger.info(f"DHARMa diagnostics: dispersion p={p_disp:.3f}")`

### 3.3 Model Formulas (lines 284-384)
- [ ] Model 1 (Climate only): `fungi ~ bio1 + bio4 + bio12 + bio15 + elevation + ME1...MEn`
- [ ] Model 2 (Add biodiversity): `+ plants_richness + vertebrates_richness`
- [ ] Model 3a (Continuous interaction): `+ plants_richness:bio4`
- [ ] Model 3b (Categorical interaction): `+ plant_high * temperate`

---

## Phase 4: Research Methods Implementation

### 4.1 Method 1: Continuous Interaction GLM (lines 296-304)
**File**: `src/machine_learning/research/methods/method1_continuous.py`

```python
class Method1ContinuousGLM:
    def __init__(self, config):
        self.config = config
        logger.info("Initializing Method 1: Continuous Interaction GLM")
    
    def fit(self, X, y):
        # Add plants_richness:bio4 interaction
        logger.info("Adding continuous interaction term: plants × seasonality")
        # Implementation...
```

- [ ] Implement continuous interaction term
- [ ] Fit Negative Binomial GLM
- [ ] Extract interaction coefficient
- [ ] Log model summary

### 4.2 Method 2: Categorical GLM with Contrasts (lines 306-357)
**File**: `src/machine_learning/research/methods/method2_categorical.py`

```python
class Method2CategoricalGLM:
    def __init__(self, config):
        self.temperate_threshold = config['temperate_threshold']
        self.plant_threshold = config['plant_high_threshold']
        logger.info(f"Method 2: Categorical with thresholds {self.temperate_threshold}")
    
    def create_categories(self, data):
        logger.info("Creating categorical variables for H2 testing")
        # temperate = bio4 > quantile(0.75)
        # plant_high = plants > quantile(0.75)
```

- [ ] Create categorical variables (temperate, plant_high)
- [ ] Fit model with interaction
- [ ] Implement emmeans contrasts
- [ ] Log contrast results: `logger.info(f"H2 contrast: {contrast:.2f}, p={p:.4f}")`

### 4.3 Method 3: Double Machine Learning (lines 777-887)
**File**: `src/machine_learning/research/methods/method3_dml.py`

```python
class Method3DML:
    def __init__(self, config):
        self.k_folds = config['dml']['k_folds']
        self.lgb_params = config['models']['lightgbm']
        logger.info(f"Method 3: DML with {self.k_folds}-fold cross-fitting")
    
    def cross_fit(self, X, y, A):
        for fold in range(self.k_folds):
            logger.info(f"DML fold {fold+1}/{self.k_folds}: Training nuisance models")
            # Train 3 models: E[Y|X], E[A1|X], E[A2|X]
```

- [ ] Implement 5-fold cross-fitting procedure
- [ ] Train nuisance models with LightGBM
- [ ] Calculate residuals on held-out folds
- [ ] Pool residuals and estimate causal effects
- [ ] Log: `logger.info(f"Causal effect θ1={theta1:.3f}, θ2={theta2:.3f}")`

---

## Phase 5: Hypothesis Testing & Interpretation

### 5.1 Hypothesis Testing Suite (lines 388-406)
**File**: `src/machine_learning/research/interpretation/hypothesis_testing.py`

```python
class HypothesisTestSuite:
    def test_h0(self, model_null, model_full):
        logger.info("Testing H0: Independence hypothesis")
        # Likelihood Ratio Test
        lrt_stat, p_value = self.likelihood_ratio_test(model_null, model_full)
        logger.info(f"LRT: χ²={lrt_stat:.2f}, p={p_value:.4f}")
        
        # AIC comparison
        delta_aic = AIC(model_null) - AIC(model_full)
        logger.info(f"ΔAIC = {delta_aic:.2f}")
        
        return {'rejected': p_value < 0.05, 'delta_aic': delta_aic}
```

- [ ] Implement H0 test (LRT + AIC)
- [ ] Implement H1 test (positive coefficients)
- [ ] Implement H2 test (emmeans contrasts)
- [ ] Generate interpretation text

### 5.2 Visualization Suite (lines 535-671)
**File**: `src/machine_learning/research/interpretation/visualization.py`

- [ ] Variable importance plots (hierarchical partitioning)
- [ ] Marginal effects tables
- [ ] Partial dependence plots
- [ ] H2 4-panel visualization:
  - [ ] Estimated marginal means bar plot
  - [ ] Interaction plot
  - [ ] Observed data scatter
  - [ ] Contrast effect visualization
- [ ] Log each plot: `logger.info(f"Generated {plot_type} visualization")`

### 5.3 Reporting (lines 759-766)
**File**: `src/machine_learning/research/interpretation/reporting.py`

- [ ] Generate Table 2: Variable importance
- [ ] Generate Table 3: Marginal effects
- [ ] Generate Figure 2: Partial dependence
- [ ] Generate Figure 3: Interaction plot
- [ ] Create publication-ready summary

---

## Phase 6: Main Pipeline Integration

### 6.1 Main Regression Pipeline
**File**: `src/machine_learning/research/biodiversity_regression.py`

```python
class BiodiversityRegressionPipeline:
    def __init__(self, config_path='config/ml/ml_config.yml'):
        self.config = self.load_config(config_path)
        logger.info("Initializing Biodiversity Regression Pipeline")
        
    def run_complete_analysis(self, data_path):
        logger.info(f"Starting regression analysis on {data_path}")
        
        # Phase 1: Data Preparation
        logger.info("Phase 1: Data preparation")
        data = self.load_data(data_path)
        data = self.apply_complete_case_analysis(data)
        
        # Phase 2: Spatial Filtering
        logger.info("Phase 2: Computing spatial eigenvectors")
        mesf_features = self.compute_mesf(data)
        
        # Phase 3: Feature Engineering
        logger.info("Phase 3: Feature engineering")
        data = self.standardize_features(data)
        data = self.create_categorical_features(data)
        
        # Phase 4: Model Fitting
        logger.info("Phase 4: Fitting models")
        results = {
            'null': self.fit_null_model(data),
            'method1': self.method1.fit(data),
            'method2': self.method2.fit(data),
            'method3': self.method3.fit(data)
        }
        
        # Phase 5: Hypothesis Testing
        logger.info("Phase 5: Testing hypotheses")
        hypothesis_results = self.test_hypotheses(results)
        
        # Phase 6: Interpretation
        logger.info("Phase 6: Generating interpretations")
        self.generate_visualizations(results)
        self.generate_report(results, hypothesis_results)
        
        logger.info("Analysis complete!")
        return results
```

### 6.2 Implementation Checklist
- [ ] Implement data loading with 162M rows support
- [ ] Add progress tracking for long operations
- [ ] Implement checkpoint saving between phases
- [ ] Add error handling and recovery
- [ ] Validate results at each phase

---

## Phase 7: Script Updates

### 7.1 Update run_ml.py
**File**: `scripts/run_ml.py`

Add new analysis type:
```python
parser.add_argument(
    '--analysis-type', '-a',
    choices=['standard', 'research', 'regression'],  # Add 'regression'
    default='standard',
    help='Type of analysis to run'
)

# Add regression pipeline
if args.analysis_type == 'regression':
    from src.machine_learning.research.biodiversity_regression import BiodiversityRegressionPipeline
    pipeline = BiodiversityRegressionPipeline(config_path=args.config_file)
    success = pipeline.run_complete_analysis(args.input)
```

### 7.2 Update run_ml.sh
**File**: `run_ml.sh`

Add regression examples:
```bash
# Examples section
echo "  # Run regression analysis (GLM + DML)"
echo "  $0 --experiment regression_full --analysis-type regression"
echo ""
echo "  # Run with specific hypotheses"
echo "  $0 --experiment h2_temperate --analysis-type regression --config h2_testing=true"
```

---

## Phase 8: Testing & Validation

### 8.1 Unit Tests
- [ ] Test MESF computation
- [ ] Test GLM model fitting
- [ ] Test DML cross-fitting
- [ ] Test hypothesis tests
- [ ] Test visualization generation

### 8.2 Integration Tests
- [ ] Test complete pipeline on sample data (1000 rows)
- [ ] Test with missing data handling
- [ ] Test checkpoint recovery
- [ ] Test all three methods produce consistent results

### 8.3 Performance Tests
- [ ] Test MESF on 100k+ rows
- [ ] Test Nystrom approximation accuracy
- [ ] Test DML parallelization efficiency
- [ ] Benchmark against R implementation

---

## Phase 9: Documentation

### 9.1 Code Documentation
- [ ] Add docstrings to all classes and methods
- [ ] Include parameter descriptions
- [ ] Add usage examples
- [ ] Document assumptions and limitations

### 9.2 User Documentation
- [ ] Create README for regression module
- [ ] Add configuration guide
- [ ] Include troubleshooting section
- [ ] Provide example notebooks

---

## Logging Strategy

### Logging Levels
```python
# Debug level - detailed information
logger.debug(f"Matrix shape before MESF: {X.shape}")
logger.debug(f"Eigenvalue {i}: {eigenval:.4f}")

# Info level - key milestones
logger.info("Starting MESF computation")
logger.info(f"Selected {n_eigenvectors} spatial eigenvectors")

# Warning level - potential issues
logger.warning(f"High VIF detected for {feature}: {vif:.2f}")
logger.warning("Convergence not achieved in 1000 iterations")

# Error level - failures
logger.error(f"Model fitting failed: {str(e)}")
logger.error("Insufficient data for spatial filtering")
```

### Progress Tracking
```python
# For long operations
total_chunks = 100
for i, chunk in enumerate(data_chunks):
    if i % 10 == 0:
        logger.info(f"Processing chunk {i}/{total_chunks} ({i*100/total_chunks:.1f}%)")
```

---

## Implementation Timeline

### Week 1: Foundation
- Days 1-2: Configuration system and project structure
- Days 3-4: MESF implementation
- Day 5: GLM base classes

### Week 2: Core Methods
- Days 1-2: Method 1 (Continuous GLM)
- Days 3-4: Method 2 (Categorical GLM + emmeans)
- Day 5: Method 3 (DML) foundation

### Week 3: Integration
- Days 1-2: Complete DML implementation
- Days 3-4: Hypothesis testing suite
- Day 5: Visualization and reporting

### Week 4: Polish
- Days 1-2: Testing and debugging
- Days 3-4: Performance optimization
- Day 5: Documentation and examples

---

## Success Criteria

1. **Correctness**: Results match statistical expectations from regression_analysis_decisions.md
2. **Performance**: Handle 162M rows within reasonable time (< 2 hours)
3. **Robustness**: Graceful handling of missing data, convergence issues
4. **Reproducibility**: Same data → same results
5. **Interpretability**: Clear logging and reporting of all decisions

---

## Risk Mitigation

### Potential Issues and Solutions

1. **Memory overflow with 162M rows**
   - Solution: Implement chunked processing
   - Use Dask for out-of-core computation

2. **MESF computation too slow**
   - Solution: Use Nystrom approximation
   - Parallelize eigendecomposition

3. **GLM convergence issues**
   - Solution: Adjust solver parameters
   - Try different optimization algorithms

4. **DML taking too long**
   - Solution: Reduce number of folds
   - Use fewer hyperparameters in LightGBM

---

## Final Checklist

- [ ] All requirements from regression_analysis_decisions.md implemented
- [ ] Configuration system fully integrated
- [ ] Logging provides complete audit trail
- [ ] All three methods (GLM continuous, GLM categorical, DML) working
- [ ] All three hypotheses (H0, H1, H2) testable
- [ ] Visualization suite complete
- [ ] run_ml.sh updated and tested
- [ ] Documentation complete
- [ ] Code review passed
- [ ] Performance benchmarks met

---

## Notes

This implementation plan maps every requirement from the regression analysis decisions document to specific implementation tasks. The modular structure allows for incremental development and testing, while the comprehensive logging ensures full traceability of the analysis process.