#!/usr/bin/env python3
"""
Plot treatment effect magnitudes across climate gradients
Shows actual treatment effects (not SHAP values) along each climate dimension
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import argparse

def plot_effect_magnitudes(model, X, effect_modifiers, treatment_names, 
                          n_points=100, figsize=(15, 5)):
    """
    Plot actual treatment effect magnitudes along each climate gradient
    
    Args:
        X: REQUIRED - Real data matrix, cannot be None or empty
    """
    if X is None or len(X) == 0:
        raise ValueError("X (real data) is required - cannot generate synthetic data")
    n_modifiers = len(effect_modifiers)
    fig, axes = plt.subplots(1, n_modifiers, figsize=figsize)
    if n_modifiers == 1:
        axes = [axes]
    
    for mod_idx, modifier in enumerate(effect_modifiers):
        ax = axes[mod_idx]
        
        # Create gradient for this modifier
        mod_min, mod_max = X[:, mod_idx].min(), X[:, mod_idx].max()
        mod_range = np.linspace(mod_min, mod_max, n_points)
        
        # Hold other modifiers at median
        X_test = np.zeros((n_points, n_modifiers))
        for j in range(n_modifiers):
            if j == mod_idx:
                X_test[:, j] = mod_range
            else:
                X_test[:, j] = np.median(X[:, j])
        
        # Get treatment effects - use const_marginal_effect for heterogeneous effects
        effects = model.const_marginal_effect(X_test)
        
        # Plot for each treatment
        for t_idx, treatment in enumerate(treatment_names[:2]):  # Skip interaction
            color = 'green' if 'plant' in treatment.lower() else 'blue'
            ax.plot(mod_range, effects[:, t_idx], label=treatment, 
                   linewidth=2, color=color)
            
            # Add confidence band if available
            if hasattr(model, 'const_marginal_effect_interval'):
                lower, upper = model.const_marginal_effect_interval(X_test, alpha=0.05)
                ax.fill_between(mod_range, lower[:, t_idx], upper[:, t_idx], 
                              alpha=0.2, color=color)
        
        ax.set_xlabel(modifier)
        ax.set_ylabel('Treatment Effect')
        ax.set_title(f'Effect along {modifier}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add broadleaf zone if bio04
        if 'bio04' in modifier:
            # High seasonality zone (broadleaf indicator)
            ax.axvspan(0.5, mod_max, alpha=0.1, color='orange', 
                      label='Broadleaf zone')
        
        ax.legend()
    
    plt.suptitle('Treatment Effect Magnitudes', fontsize=14)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot climate gradient effects')
    parser.add_argument('--model', required=True, help='Path to model pickle')
    parser.add_argument('--data', required=True, help='Path to .npz data file (REQUIRED)')
    parser.add_argument('--output', default='effect_magnitudes.png', 
                       help='Output filename')
    parser.add_argument('--n-samples', type=int, default=1000, 
                       help='Number of samples to use')
    
    args = parser.parse_args()
    
    # Import classes BEFORE unpickling to avoid AttributeError
    sys.path.insert(0, '/home/yl998/Geo_process')
    from src.machine_learning.doubleML.refactored_pipeline import HeterogeneousEffectEstimator
    from src.machine_learning.doubleML.optimized_lgbm import OptimizedLGBMRegressor
    
    # Make classes available in __main__ namespace for unpickling
    globals()['HeterogeneousEffectEstimator'] = HeterogeneousEffectEstimator
    globals()['OptimizedLGBMRegressor'] = OptimizedLGBMRegressor
    
    with open(args.model, 'rb') as f:
        wrapper = pickle.load(f)
        # Extract the actual model from the wrapper
        if hasattr(wrapper, 'estimator'):
            model = wrapper.estimator
        else:
            model = wrapper
    
    # Load REAL data - NO GENERATION
    if not args.data:
        raise ValueError("--data is REQUIRED. Cannot generate synthetic data.")
    
    data = np.load(args.data)
    if 'X' not in data:
        raise KeyError(f"'X' not found in {args.data}")
    
    X = data['X']
    if args.n_samples < len(X):
        # Sample from real data
        indices = np.random.choice(len(X), args.n_samples, replace=False)
        X = X[indices]
    
    effect_modifiers = ['bio01_std', 'bio04_std', 'bio12_std']
    treatment_names = ['plants_richness', 'vertebrates_richness']
    
    # Create plot
    fig = plot_effect_magnitudes(model, X, effect_modifiers, treatment_names)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()