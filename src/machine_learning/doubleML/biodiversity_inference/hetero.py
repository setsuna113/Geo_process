#!/usr/bin/env python3
"""
Heterogeneous Effects Inference for Biodiversity Analysis

For models where biodiversity effects vary by location/climate (X != None)
"""

import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from scipy import stats


def get_hetero_inference(estimator, X: np.ndarray, max_samples: int = 100000, max_effects: int = 1000000) -> Dict[str, Any]:
    """
    Get heterogeneous DML inference results with smart sampling for expensive operations
    
    Parameters:
    -----------
    estimator : fitted EconML estimator with X != None
    X : test points where to evaluate effects
    max_samples : max samples for expensive operations (default 100k)
    max_effects : max samples for effects computation (default 1M)
    
    Returns:
    --------
    Dict with coefficients, stderr, pvalues, CI, n_effects, n_inference
    """
    if X is None:
        raise ValueError("X required for heterogeneous effects")
    
    n_samples = X.shape[0]
    
    # 1. Effects - sample if too large (memory constraints)
    if n_samples > max_effects:
        print(f"Sampling {max_effects} from {n_samples} for effects computation...")
        idx_effects = np.random.choice(n_samples, max_effects, replace=False)
        X_effects = X[idx_effects]
        coef = estimator.const_marginal_effect(X_effects)
    else:
        print(f"Computing effects on full {n_samples} samples...")
        X_effects = X
        coef = estimator.const_marginal_effect(X)
    
    # 2. Inference - EXPENSIVE, sample from effects data if needed
    n_effects = X_effects.shape[0]
    if n_effects > max_samples:
        print(f"Sampling {max_samples} from {n_effects} for inference (stderr, pvalues, CI)...")
        idx = np.random.choice(n_effects, max_samples, replace=False)
        X_inference = X_effects[idx]
        coef_inference = coef[idx] if coef.ndim > 1 else coef
    else:
        X_inference = X_effects
        coef_inference = coef
    
    inf = estimator.const_marginal_effect_inference(X_inference)
    pvalues = inf.pvalue(value=0)
    
    # Rank top 20% regions by impact (magnitude × significance)
    significant_mask = pvalues < 0.05
    impact_scores = np.sum(np.abs(coef) * significant_mask, axis=1)
    ranked_indices = np.argsort(impact_scores)[::-1]
    n_top = int(len(ranked_indices) * 0.2)
    top_20_percent = ranked_indices[:n_top]
    
    return {
        'coefficients': coef,  # All effects computed
        'stderr': inf.stderr,  # Sampled inference
        'pvalues': pvalues,  # Sampled inference  
        'ci_95': inf.conf_int(alpha=0.05),  # Sampled inference
        'n_effects': n_effects,  # Number of effects computed
        'n_inference': X_inference.shape[0],  # Number sampled for inference
        'top_20_percent_indices': top_20_percent,  # Top impact regions
        'top_20_percent_scores': impact_scores[top_20_percent]  # Their scores
    }






def interpret_biodiversity_effects(results: Dict[str, Any],
                                  treatment_names: List[str] = None) -> str:
    """
    Generate domain-specific interpretation
    
    This is the ONLY domain-specific function needed
    """
    if treatment_names is None:
        treatment_names = ['plants_richness', 'vertebrates_richness', 'plants_x_vertebrates']
    
    interpretations = []
    
    coef = results['coefficients']
    pvals = results['pvalues']
    
    # Handle both constant and heterogeneous
    if coef.ndim == 1:
        # Constant effects
        for i, name in enumerate(treatment_names):
            if pvals[i] < 0.05:
                if name == 'plants_richness':
                    interpretations.append(f"Each plant species → +{coef[i]:.4f} fungi")
                elif name == 'vertebrates_richness':
                    interpretations.append(f"Each vertebrate → +{coef[i]:.4f} fungi")
                elif name == 'plants_x_vertebrates' and coef[i] < 0:
                    interpretations.append("Competitive interaction between biodiversity")
    else:
        # Heterogeneous effects - show range
        for i, name in enumerate(treatment_names):
            effect_range = [np.min(coef[:, i]), np.max(coef[:, i])]
            if name == 'plants_richness':
                interpretations.append(
                    f"Plant effects vary from {effect_range[0]:.4f} to {effect_range[1]:.4f}"
                )
    
    return " | ".join(interpretations) if interpretations else "No significant effects"






if __name__ == "__main__":
    import pickle
    import pandas as pd
    import os
    import sys
    
    # Add project root to path for pickle
    sys.path.insert(0, '/home/yl998/Geo_process')
    
    print("Loading modules...")
    from src.machine_learning.doubleML.refactored_pipeline import HeterogeneousEffectEstimator
    import time
    
    model_path = "/scratch/yl998/econml_results/hetero_results_step10.pkl"
    data_path = "/scratch/yl998/biodiversity_climate_production_1x_corrected.arrow"
    
    # Load model
    print("Loading model...")
    t0 = time.time()
    with open(model_path, 'rb') as f:
        wrapper = pickle.load(f)
        model = wrapper.estimator
    print(f"Model loaded in {time.time()-t0:.1f}s")
    
    # Check if heterogeneous
    if not hasattr(model, '_d_x') or model._d_x is None or model._d_x == ():
        print("ERROR: Model is homogeneous. Need heterogeneous model for this script.")
        sys.exit(1)
    
    # Load data
    print("Loading data from arrow file...")
    t0 = time.time()
    import pyarrow.feather as feather
    df = feather.read_feather(data_path)
    print(f"Data loaded in {time.time()-t0:.1f}s, shape: {df.shape}")
    
    print("Extracting effect modifiers...")
    X = df[['bio01', 'bio04', 'bio12']].values
    
    print("Removing NaN values...")
    X = X[~np.isnan(X).any(axis=1)]
    print(f"Clean data shape: {X.shape}")
    
    # TEMPORARY: Sample to 10k for testing
    if X.shape[0] > 10000:
        print(f"TEMP: Sampling to 10,000 for testing...")
        idx = np.random.choice(X.shape[0], 10000, replace=False)
        X = X[idx]
    
    print(f"Running inference on {X.shape[0]} samples...")
    results = get_hetero_inference(model, X, max_samples=100000)
    
    # Generate report  
    report = []
    report.append("HETEROGENEOUS TREATMENT EFFECTS REPORT")
    report.append("="*50)
    report.append(f"Date: {pd.Timestamp.now()}")
    report.append(f"Model: {model.__class__.__name__}")
    report.append(f"Samples analyzed: {results['n_effects']} locations")
    report.append(f"Samples for inference: {results['n_inference']}")
    report.append(f"Effect modifiers: bio01, bio04, bio12")
    
    # Add feature importances if available
    if hasattr(model, 'feature_importances_'):
        report.append(f"\nFeature Importances:")
        for i, name in enumerate(['bio01', 'bio04', 'bio12']):
            report.append(f"  {name}: {model.feature_importances_[i]:.3f}")
    report.append("")
    
    # Effect statistics
    report.append("TREATMENT EFFECT STATISTICS")
    report.append("-"*30)
    treatment_names = ['plants_richness', 'vertebrates_richness', 'plants_x_vertebrates']
    
    for i, name in enumerate(treatment_names):
        effects = results['coefficients'][:, i] if results['coefficients'].ndim > 1 else [results['coefficients'][i]]
        
        report.append(f"\n{name}:")
        report.append(f"  Mean effect: {np.mean(effects):.6f}")
        report.append(f"  Std dev: {np.std(effects):.6f}")
        report.append(f"  Min: {np.min(effects):.6f}")
        report.append(f"  25%: {np.percentile(effects, 25):.6f}")
        report.append(f"  Median: {np.median(effects):.6f}")
        report.append(f"  75%: {np.percentile(effects, 75):.6f}")
        report.append(f"  Max: {np.max(effects):.6f}")
        
        # Only add significance if we have pvalues
        if results['pvalues'] is not None and results['pvalues'].shape[0] > 0:
            report.append(f"  % Significant (p<0.05): {(results['pvalues'][:, i] < 0.05).mean()*100:.1f}%")
    
    # Top impact regions
    report.append("\n" + "="*50)
    report.append("TOP 20% IMPACT REGIONS")
    report.append("-"*30)
    if 'top_20_percent_indices' in results:
        report.append(f"Number of top regions: {len(results['top_20_percent_indices'])}")
        report.append(f"Impact score range: [{results['top_20_percent_scores'].min():.4f}, {results['top_20_percent_scores'].max():.4f}]")
    
    # Heterogeneity analysis
    report.append("\n" + "="*50)
    report.append("HETEROGENEITY ANALYSIS")
    report.append("-"*30)
    
    for i, name in enumerate(treatment_names):
        effects = results['coefficients'][:, i] if results['coefficients'].ndim > 1 else [results['coefficients'][i]]
        cv = np.std(effects) / abs(np.mean(effects)) if np.mean(effects) != 0 else 0
        report.append(f"{name}: CV={cv:.2f} (heterogeneity: {'High' if cv > 0.5 else 'Low'})")
    
    # Climate correlations
    report.append("\n" + "="*50)
    report.append("CLIMATE CORRELATIONS")
    report.append("-"*30)
    
    climate_names = ['bio01 (Temp)', 'bio04 (Temp Season)', 'bio12 (Precip)']
    for t_idx, treatment in enumerate(treatment_names):
        effects = results['coefficients'][:, t_idx] if results['coefficients'].ndim > 1 else [results['coefficients'][t_idx]]
        report.append(f"\n{treatment} correlations:")
        for c_idx, climate in enumerate(climate_names):
            corr = np.corrcoef(X[:len(effects), c_idx], effects)[0, 1]
            report.append(f"  vs {climate}: {corr:.3f}")
    
    # Save text report
    report_path = "/scratch/yl998/econml_results/hetero_inference_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    # Also save summary CSV for easy analysis
    summary_df = pd.DataFrame({
        'treatment': treatment_names * len(effects),
        'effect': results['coefficients'].flatten(),
        'stderr': results['stderr'].flatten() if results['stderr'].size > 1 else results['stderr'],
        'pvalue': results['pvalues'].flatten()
    })
    csv_path = "/scratch/yl998/econml_results/hetero_effects_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    
    print(f"✓ Report saved to: {report_path}")
    print(f"✓ CSV saved to: {csv_path}")
    print(f"✓ Effects shape: {results['coefficients'].shape}")