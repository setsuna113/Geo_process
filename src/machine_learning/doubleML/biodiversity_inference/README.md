# Biodiversity Inference Module

Domain-specific statistical inference for testing biodiversity effects on fungi richness using Double Machine Learning.

## Purpose

Tests the hypothesis:
- **H₀**: Biodiversity does not provide additional information beyond abiotic features
- **H₁**: Biodiversity provides additional information for fungi richness

## Key Features

- ✅ **Correct EconML API usage** (uses built-in inference methods)
- ✅ **Domain-specific interpretations** for ecological context
- ✅ **Works with saved results or fitted models**
- ✅ **No reimplementation of statistics** (uses EconML's methods)

## Usage

### Quick Analysis from Saved Results

```python
from biodiversity_inference import analyze_saved_results

# Analyze existing EconML results
results = analyze_saved_results()
```

### Full Inference with Interpretation

```python
from biodiversity_inference import BiodiversityInference

# From saved results
bio_inf = BiodiversityInference(results_path="/path/to/econml_results.json")
results = bio_inf.test_biodiversity_hypothesis()

# Note: DML focuses on causal effects, not variance explained
# The model provides unbiased treatment effect estimates, not R²
```

### With Fitted EconML Model

```python
from econml.dml import LinearDML
from biodiversity_inference import BiodiversityInference

# Fit model with proper inference
est = LinearDML(model_y=model_y, model_t=model_t)
est.fit(Y, T, X=None, W=W, inference='statsmodels')

# Test biodiversity hypothesis
bio_inf = BiodiversityInference(estimator=est)
results = bio_inf.test_biodiversity_hypothesis()
```

## Files

- `inference.py` - Main BiodiversityInference class
- `analysis.py` - Quick analysis functions for saved results
- `examples.py` - Correct EconML usage examples

## Key Methods

### BiodiversityInference

- `test_biodiversity_hypothesis()` - Complete hypothesis test with interpretation
- `extract_inference_properly()` - Correct extraction of EconML inference
- `_assess_biodiversity_contribution()` - Calculate both Z-statistics and Cohen's d

## Results Interpretation

The module provides ecological interpretations:

- **Plants effect**: "Each additional plant species → +X fungi richness"
- **Vertebrates effect**: "Each additional vertebrate species → +Y fungi richness"  
- **Interaction**: "Competitive/Synergistic interaction between biodiversity components"

## Important Notes

1. **Uses EconML's built-in methods** - No custom statistics
2. **Double underscore is correct**: `intercept__inference()` ✅
3. **Domain interpretation only** - Statistics come from EconML
4. **DML provides causal effects, not variance explained**:
   - ✅ Unbiased treatment effect estimates (θ₀)
   - ✅ Valid confidence intervals and p-values
   - ✅ Nuisance model R² (first stage only)
   - ❌ Overall model R² or variance explained
   - ❌ Partial R² for treatments

## Example Output

```
BIODIVERSITY → FUNGI RICHNESS HYPOTHESIS TEST
======================================================================

1. INDIVIDUAL BIODIVERSITY EFFECTS:
--------------------------------------------------
Plant Species Richness:
  Effect size: 0.002348
  Interpretation: Each additional plant species → +0.0023 fungi richness
  ✓ SIGNIFICANT at 5% level

2. JOINT BIODIVERSITY SIGNIFICANCE:
--------------------------------------------------
Wald χ² = 434.38 (df=3)
✓ Biodiversity variables are JOINTLY HIGHLY SIGNIFICANT

DECISION: STRONGLY REJECT H0
Evidence: Overwhelming evidence for biodiversity effects
Recommendation: Essential to include biodiversity in fungi models
```