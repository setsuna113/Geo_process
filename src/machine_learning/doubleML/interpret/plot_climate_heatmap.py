#!/usr/bin/env python3
"""
Plot 2D climate space heatmaps showing treatment effects
Creates heatmaps for all pairs of climate variables
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
from itertools import combinations

def plot_climate_slices_2d(model, X, effect_modifiers, treatment, 
                           n_grid=30, figsize=(18, 5)):
    """
    Create 2D heatmaps for all pairs of effect modifiers
    
    Args:
        X: REQUIRED - Real data matrix, cannot be None or empty
    """
    if X is None or len(X) == 0:
        raise ValueError("X (real data) is required - cannot generate synthetic data")
    n_modifiers = len(effect_modifiers)
    
    if n_modifiers < 2:
        raise ValueError("Need at least 2 effect modifiers for 2D slices")
    
    # Get treatment index
    treatment_names = ['plants_richness', 'vertebrates_richness', 'plants_x_vertebrates']
    t_idx = treatment_names.index(treatment)
    
    # Get all pairs
    pairs = list(combinations(range(n_modifiers), 2))
    n_pairs = len(pairs)
    
    fig, axes = plt.subplots(1, n_pairs, figsize=figsize)
    if n_pairs == 1:
        axes = [axes]
    
    for pair_idx, (i, j) in enumerate(pairs):
        ax = axes[pair_idx]
        
        # Create 2D grid
        x_min, x_max = X[:, i].min(), X[:, i].max()
        y_min, y_max = X[:, j].min(), X[:, j].max()
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, n_grid),
            np.linspace(y_min, y_max, n_grid)
        )
        
        # Create test data - hold other vars at median
        X_test = np.zeros((n_grid * n_grid, n_modifiers))
        X_test[:, i] = xx.ravel()
        X_test[:, j] = yy.ravel()
        
        for k in range(n_modifiers):
            if k not in [i, j]:
                X_test[:, k] = np.median(X[:, k])
        
        # Get effects - use const_marginal_effect for heterogeneous effects
        effects = model.const_marginal_effect(X_test)[:, t_idx]
        Z = effects.reshape(xx.shape)
        
        # Plot heatmap
        im = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r')
        ax.contour(xx, yy, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
        
        # Add broadleaf zone overlay
        if 'bio04' in effect_modifiers[i]:
            ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, 
                      label='High seasonality')
        if 'bio04' in effect_modifiers[j]:
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7,
                      label='High seasonality')
        
        if 'bio12' in effect_modifiers[i]:
            ax.axvspan(0, 1, alpha=0.1, color='blue')
        if 'bio12' in effect_modifiers[j]:
            ax.axhspan(0, 1, alpha=0.1, color='blue')
        
        ax.set_xlabel(effect_modifiers[i])
        ax.set_ylabel(effect_modifiers[j])
        ax.set_title(f'{effect_modifiers[i]} × {effect_modifiers[j]}')
        
        # Colorbar
        plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    
    plt.suptitle(f'2D Climate Space Slices: {treatment} Effect', fontsize=14)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Plot 2D climate heatmaps')
    parser.add_argument('--model', required=True, help='Path to model pickle')
    parser.add_argument('--data', required=True, help='Path to .npz data file (REQUIRED)')
    parser.add_argument('--treatment', default='plants_richness', 
                       choices=['plants_richness', 'vertebrates_richness'],
                       help='Treatment to plot')
    parser.add_argument('--output', default='climate_heatmap.png', 
                       help='Output filename')
    parser.add_argument('--n-grid', type=int, default=30, 
                       help='Grid resolution')
    
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
    
    X = data['X'][:500]  # Use subset for heatmap calculation
    effect_modifiers = ['bio01_std', 'bio04_std', 'bio12_std']
    
    # Create plot
    fig = plot_climate_slices_2d(model, X, effect_modifiers, args.treatment)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()