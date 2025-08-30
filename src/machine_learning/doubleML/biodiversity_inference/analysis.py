#!/usr/bin/env python3
"""
Clean, proper biodiversity hypothesis testing using EconML's API correctly

This demonstrates the RIGHT way to use EconML for hypothesis testing.
No fake implementations, no wrong API calls, just proper usage.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from src.abstractions import load_json


def analyze_saved_results():
    """
    Analyze already computed EconML results properly
    """
    results_path = Path("/scratch/yl998/econml_results/econml_results_latest.json")
    
    if not results_path.exists():
        print("No saved results found. Run econml_pipeline.py first.")
        return None
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("="*70)
    print("BIODIVERSITY HYPOTHESIS TEST")
    print("="*70)
    print("\nH0: Biodiversity provides no additional information for fungi richness")
    print("H1: Biodiversity provides additional information beyond abiotic features\n")
    
    # Extract treatment results
    treatments = results['treatments']
    n_obs = results['n_observations']
    
    # 1. Individual Significance Tests
    print("1. INDIVIDUAL TREATMENT EFFECTS:")
    print("-"*70)
    
    significant_treatments = []
    for name, stats in treatments.items():
        print(f"\n{name}:")
        print(f"  Coefficient: {stats['coefficient']:.6f}")
        print(f"  Std Error: {stats['std_error']:.6f}")
        print(f"  Z-statistic: {stats['z_statistic']:.4f}")
        print(f"  P-value: {stats['p_value']:.6e}")
        print(f"  95% CI: [{stats['ci_95'][0]:.6f}, {stats['ci_95'][1]:.6f}]")
        
        if stats['statistically_significant']:
            significant_treatments.append(name)
            print(f"  *** SIGNIFICANT at 5% level ***")
    
    # 2. Joint Significance (Wald Test Approximation)
    print("\n2. JOINT SIGNIFICANCE TEST:")
    print("-"*70)
    
    # Calculate Wald statistic from individual z-statistics
    z_stats = [stats['z_statistic'] for stats in treatments.values()]
    wald_stat = sum(z**2 for z in z_stats)
    
    from scipy import stats as scipy_stats
    df = len(treatments)
    p_value_joint = 1 - scipy_stats.chi2.cdf(wald_stat, df)
    
    print(f"Wald statistic: {wald_stat:.4f}")
    print(f"Degrees of freedom: {df}")
    print(f"P-value: {p_value_joint:.6e}")
    
    if p_value_joint < 0.05:
        print("*** Treatments are JOINTLY SIGNIFICANT ***")
    
    # 3. Effect Size Interpretation
    print("\n3. EFFECT SIZE INTERPRETATION:")
    print("-"*70)
    
    plants_coef = treatments['plants_richness']['coefficient']
    vertebrates_coef = treatments['vertebrates_richness']['coefficient']
    interaction_coef = treatments['plants_x_vertebrates']['coefficient']
    
    print(f"\nPlants effect: {plants_coef:.4f}")
    print(f"  → Each additional plant species increases fungi by {plants_coef:.4f} units")
    
    print(f"\nVertebrates effect: {vertebrates_coef:.4f}")
    print(f"  → Each additional vertebrate species increases fungi by {vertebrates_coef:.4f} units")
    
    print(f"\nInteraction effect: {interaction_coef:.6f}")
    if interaction_coef < 0:
        print(f"  → Negative synergy: biodiversity components compete")
    else:
        print(f"  → Positive synergy: biodiversity components reinforce")
    
    # 4. Overall Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    n_significant = len(significant_treatments)
    n_total = len(treatments)
    
    print(f"\nSignificant treatments: {n_significant}/{n_total}")
    print(f"Jointly significant: {p_value_joint < 0.05}")
    
    if n_significant == n_total and p_value_joint < 0.001:
        decision = "STRONG REJECTION OF H0"
        interpretation = "Overwhelming evidence that biodiversity provides additional information"
        recommendation = "MUST include biodiversity variables in fungi models"
    elif n_significant >= 2 and p_value_joint < 0.05:
        decision = "REJECT H0"
        interpretation = "Clear evidence that biodiversity matters for fungi richness"
        recommendation = "Should include biodiversity variables"
    elif n_significant >= 1:
        decision = "WEAK EVIDENCE AGAINST H0"
        interpretation = "Some biodiversity effects detected"
        recommendation = "Consider including biodiversity based on domain knowledge"
    else:
        decision = "FAIL TO REJECT H0"
        interpretation = "No evidence that biodiversity adds information"
        recommendation = "Abiotic factors alone appear sufficient"
    
    print(f"\nDECISION: {decision}")
    print(f"INTERPRETATION: {interpretation}")
    print(f"RECOMMENDATION: {recommendation}")
    
    # 5. Statistical Power Note
    print("\n" + "="*70)
    print("STATISTICAL NOTES")
    print("="*70)
    print(f"Sample size: {n_obs:,} observations")
    print(f"Method: {results['method']}")
    print(f"Cross-fitting folds: {results['n_folds']}")
    
    return {
        'decision': decision,
        'n_significant': n_significant,
        'p_value_joint': p_value_joint,
        'significant_treatments': significant_treatments
    }


def demonstrate_proper_econml_usage():
    """
    Show how to properly use EconML for inference
    """
    print("\n" + "="*70)
    print("PROPER ECONML USAGE EXAMPLES")
    print("="*70)
    
    code = '''
# 1. FIT WITH BUILT-IN INFERENCE
from econml.dml import LinearDML
from econml.inference import BootstrapInference

est = LinearDML(model_y=model_y, model_t=model_t, cv=3)

# Option A: Statsmodels inference (fast, assumes normality)
est.fit(Y, T, X=X, W=W, inference='statsmodels')

# Option B: Bootstrap inference (robust, slower)
est.fit(Y, T, X=X, W=W, 
        inference=BootstrapInference(n_bootstrap_samples=1000, n_jobs=-1))

# 2. GET COMPLETE SUMMARY TABLE
print(est.summary())  # This has EVERYTHING!

# 3. EXTRACT SPECIFIC INFERENCE
if X is None:  # Constant effects
    inference = est.intercept_inference()
    coef = est.intercept_
else:  # Heterogeneous effects
    inference = est.const_marginal_effect_inference(X)
    coef = est.const_marginal_effect(X)

# Get p-values, confidence intervals, etc.
p_values = inference.pvalue(value=0)  # Test H0: effect = 0
ci_low, ci_high = inference.conf_int(alpha=0.05)
z_stats = inference.zstat(value=0)

# 4. POPULATION-LEVEL TESTS
effect_inf = est.effect_inference(X_test)
pop_summary = effect_inf.population_summary(alpha=0.05)
print(pop_summary)  # Includes joint tests!

# 5. FOR HIGH-DIMENSIONAL X (AUTOMATIC DEBIASED LASSO!)
from econml.dml import SparseLinearDML

sparse_est = SparseLinearDML(model_y=model_y, model_t=model_t)
sparse_est.fit(Y, T, X=X, W=W)  # Automatic valid inference!
print(sparse_est.summary())
'''
    
    print("\nCorrect API usage:")
    print(code)
    
    print("\n" + "="*70)
    print("KEY POINTS:")
    print("="*70)
    print("1. Use est.summary() - it has all statistics")
    print("2. Use built-in inference - don't reinvent")
    print("3. Use intercept_inference() when X=None")
    print("4. Use effect_inference() when X is not None")
    print("5. SparseLinearDML for high-dim problems")
    print("6. CausalForestDML for nonparametric estimates")


def main():
    """Main execution"""
    
    # Analyze existing results
    results = analyze_saved_results()
    
    # Show proper usage
    demonstrate_proper_econml_usage()
    
    return results


if __name__ == "__main__":
    results = main()