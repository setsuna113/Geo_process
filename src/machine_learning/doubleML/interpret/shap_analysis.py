#!/usr/bin/env python3
"""
SHAP analysis module for heterogeneous treatment effects
High-density visualizations with confidence intervals and interaction analysis
"""

import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
from scipy.stats import binned_statistic


def compute_shap_values(model, X: np.ndarray, batch_size: int = 2000, 
                        verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute SHAP values for a model using batch processing
    
    Args:
        model: Fitted EconML model with shap_values method
        X: Feature matrix (n_samples, n_features)
        batch_size: Size of batches for processing
        verbose: Whether to show progress
        
    Returns:
        Dictionary mapping treatment names to SHAP value arrays
    """
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    if verbose:
        print(f"Computing SHAP values for {n_samples:,} samples...")
        iterator = tqdm(range(n_batches), desc="SHAP batches")
    else:
        iterator = range(n_batches)
    
    all_shap_values = {}
    
    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        batch_X = X[start_idx:end_idx]
        
        # Get SHAP values for this batch
        batch_shap = model.shap_values(batch_X)['Y0']
        
        # Initialize accumulator on first batch
        if batch_idx == 0:
            for key in batch_shap.keys():
                all_shap_values[key] = []
        
        # Accumulate results
        for key in batch_shap.keys():
            if hasattr(batch_shap[key], 'values'):
                all_shap_values[key].append(batch_shap[key].values)
            else:
                all_shap_values[key].append(batch_shap[key])
    
    # Concatenate all batches
    shap_values = {}
    for key in all_shap_values:
        shap_values[key] = np.vstack(all_shap_values[key])
    
    return shap_values


def create_shap_visualizations(model, X: np.ndarray, effect_modifiers: list,
                              treatment_names: list, output_dir: Path,
                              shap_sample_size: int = 10000, seed: int = 42,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Create comprehensive SHAP visualizations for heterogeneous treatment effects
    This generates EXACTLY the same visualizations as the original run_visualization_10k.py
    
    Args:
        model: Fitted EconML model
        X: Feature matrix or None to generate synthetic data
        effect_modifiers: List of feature names
        treatment_names: List of treatment names
        output_dir: Directory to save plots
        shap_sample_size: Number of samples for SHAP analysis
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Dictionary with SHAP values and statistics
    """
    np.random.seed(seed)
    
    # Ensure we have enough data
    if X is None:
        if verbose:
            print("Generating synthetic test data...")
        X = np.random.normal(0, 0.5, (shap_sample_size + 2000, len(effect_modifiers)))
    
    # Sample if we have too much data
    if len(X) > shap_sample_size:
        sample_idx = np.random.choice(len(X), shap_sample_size, replace=False)
        X_shap = X[sample_idx]
    else:
        X_shap = X
    
    if verbose:
        print(f"Using {len(X_shap):,} samples for SHAP analysis")
        print("\n" + "="*60)
        print(f"COMPUTING SHAP VALUES FOR {shap_sample_size:,} SAMPLES")
        print("="*60)
    
    # Compute SHAP values
    start_time = time.time()
    shap_dict = compute_shap_values(model, X_shap, verbose=verbose)
    
    # Reorganize by treatment
    shap_by_treatment = {}
    for idx, treatment in enumerate(treatment_names):
        key = f'T{idx}'
        if key in shap_dict:
            shap_by_treatment[treatment] = shap_dict[key]
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\n✓ SHAP computation completed in {elapsed/60:.2f} minutes")
        for treatment, values in shap_by_treatment.items():
            print(f"  {treatment}: shape={values.shape}, range=[{values.min():.4f}, {values.max():.4f}]")
    
    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
    
    # Generate all visualizations (exactly as in original)
    
    # 1. Feature importance with confidence intervals
    if verbose:
        print("\n1. Creating feature importance with confidence intervals...")
    _create_feature_importance_plot(shap_by_treatment, effect_modifiers, treatment_names, 
                                   output_dir, X_shap, shap_sample_size, verbose)
    
    # 2. High-density dependence plots
    if verbose:
        print("\n2. Creating high-density dependence plots...")
    _create_dependence_plots(shap_by_treatment, X_shap, effect_modifiers, treatment_names,
                            output_dir, shap_sample_size, verbose)
    
    # 3. Fixed violin plots
    if verbose:
        print("\n3. Creating fixed SHAP distribution violin plots...")
    _create_violin_plots(shap_by_treatment, effect_modifiers, treatment_names, output_dir,
                       shap_sample_size, verbose)
    
    # 4. Comparative violin plot
    if verbose:
        print("\n4. Creating comparative violin plot...")
    _create_comparative_violin_plot(shap_by_treatment, effect_modifiers, treatment_names,
                                   output_dir, shap_sample_size, verbose)
    
    # 5. 2D interaction heatmaps
    if verbose:
        print("\n5. Creating 2D interaction heatmaps...")
    _create_interaction_heatmaps(shap_by_treatment, X_shap, effect_modifiers,
                                treatment_names, output_dir, shap_sample_size, verbose)
    
    # Compute statistics
    statistics = compute_shap_statistics(shap_by_treatment, effect_modifiers, verbose)
    
    if verbose:
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print(f"\n✓ All visualizations saved to: {output_dir}")
        print(f"✓ Total SHAP samples analyzed: {shap_sample_size:,}")
        print(f"✓ Total computation time: {elapsed/60:.2f} minutes")
    
    return {
        'shap_values': shap_by_treatment,
        'X': X_shap,
        'statistics': statistics,
        'computation_time': elapsed
    }


def compute_shap_statistics(shap_by_treatment: Dict[str, np.ndarray], 
                           effect_modifiers: list, verbose: bool = True) -> Dict[str, Any]:
    """
    Compute detailed statistics for SHAP values
    """
    statistics = {}
    
    if verbose:
        print("\nKEY INSIGHTS FROM SAMPLES:")
    
    for treatment, shap_vals_all in shap_by_treatment.items():
        if treatment == 'plants_x_vertebrates':  # Skip interaction term
            continue
            
        statistics[treatment] = {}
        if verbose:
            print(f"\n{treatment}:")
            
        for feat_idx, feature in enumerate(effect_modifiers):
            feat_shap = shap_vals_all[:, feat_idx]
            statistics[treatment][feature] = {
                'mean': float(np.mean(feat_shap)),
                'std': float(np.std(feat_shap)),
                'min': float(np.min(feat_shap)),
                'max': float(np.max(feat_shap)),
                'percentile_2.5': float(np.percentile(feat_shap, 2.5)),
                'percentile_97.5': float(np.percentile(feat_shap, 97.5)),
                'mean_abs': float(np.mean(np.abs(feat_shap)))
            }
            
            if verbose:
                print(f"  {feature}:")
                print(f"    Mean effect: {statistics[treatment][feature]['mean']:.6f}")
                print(f"    Std dev: {statistics[treatment][feature]['std']:.6f}")
                print(f"    95% range: [{statistics[treatment][feature]['percentile_2.5']:.6f}, "
                      f"{statistics[treatment][feature]['percentile_97.5']:.6f}]")
    
    if verbose:
        print("\n" + "="*60)
    
    return statistics


def _create_feature_importance_plot(shap_by_treatment, effect_modifiers, treatment_names,
                                   output_dir, X_shap, shap_sample_size, verbose):
    """Create feature importance plot with confidence intervals (EXACT copy from original)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (treatment, shap_vals) in enumerate(shap_by_treatment.items()):
        ax = axes[idx]
        
        # Calculate mean absolute SHAP values and confidence intervals
        abs_shap = np.abs(shap_vals)
        mean_importance = np.mean(abs_shap, axis=0)
        std_importance = np.std(abs_shap, axis=0)
        ci_95 = 1.96 * std_importance / np.sqrt(len(shap_vals))
        
        # Create bar plot with error bars
        y_pos = np.arange(len(effect_modifiers))
        ax.barh(y_pos, mean_importance, xerr=ci_95, color='steelblue', alpha=0.8, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(effect_modifiers)
        ax.set_xlabel('Mean |SHAP value| (±95% CI)')
        ax.set_title(f'{treatment.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'Feature Importance from {shap_sample_size:,} SHAP Values', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_feature_importance_10k.png', dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✓ Saved: shap_feature_importance_10k.png")


def _create_dependence_plots(shap_by_treatment, X_shap, effect_modifiers, treatment_names,
                            output_dir, shap_sample_size, verbose):
    """Create high-density dependence plots (EXACT copy from original)"""
    for treatment in ['plants_richness', 'vertebrates_richness']:
        for feat_idx, feature_name in enumerate(effect_modifiers):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Get SHAP and feature values
            shap_vals = shap_by_treatment[treatment][:, feat_idx]
            feature_vals = X_shap[:, feat_idx]
            
            # Left plot: Hexbin for density
            hexbin = ax1.hexbin(feature_vals, shap_vals, gridsize=40, cmap='YlOrRd', 
                               mincnt=1, alpha=0.8)
            
            # Add trend line using binned means
            bin_means, bin_edges, _ = binned_statistic(feature_vals, shap_vals, 
                                                       statistic='mean', bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Remove NaN values
            valid = ~np.isnan(bin_means)
            if valid.sum() > 10:
                ax1.plot(bin_centers[valid], bin_means[valid], 'b-', linewidth=2, 
                        label='Binned mean', alpha=0.9)
            
            ax1.set_xlabel(f'{feature_name}')
            ax1.set_ylabel('SHAP value')
            ax1.set_title(f'Density view (hexbin)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add colorbar for density
            cb1 = plt.colorbar(hexbin, ax=ax1)
            cb1.set_label('Count', rotation=270, labelpad=15)
            
            # Right plot: Scatter with interaction coloring
            color_idx = (feat_idx + 1) % len(effect_modifiers)
            color_vals = X_shap[:, color_idx]
            
            # Subsample for scatter to avoid overplotting
            subsample_idx = np.random.choice(len(feature_vals), 
                                           size=min(2000, len(feature_vals)), 
                                           replace=False)
            
            scatter = ax2.scatter(feature_vals[subsample_idx], shap_vals[subsample_idx], 
                                c=color_vals[subsample_idx], cmap='viridis', 
                                s=5, alpha=0.5, rasterized=True)
            
            # Add percentile bands
            percentiles = [25, 50, 75]
            for p in percentiles:
                pct_vals, pct_bin_edges, _ = binned_statistic(feature_vals, shap_vals, 
                                                 statistic=lambda x: np.percentile(x, p), 
                                                 bins=30)
                pct_bin_centers = (pct_bin_edges[:-1] + pct_bin_edges[1:]) / 2
                valid = ~np.isnan(pct_vals)
                if valid.sum() > 10:
                    alpha = 0.9 if p == 50 else 0.5
                    linewidth = 2 if p == 50 else 1
                    label = f'{p}th percentile' if p != 50 else 'Median'
                    ax2.plot(pct_bin_centers[valid], pct_vals[valid], 
                           linewidth=linewidth, alpha=alpha, label=label)
            
            ax2.set_xlabel(f'{feature_name}')
            ax2.set_ylabel('SHAP value')
            ax2.set_title(f'Interaction view (colored by {effect_modifiers[color_idx]})')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9, loc='best')
            
            # Add colorbar
            cb2 = plt.colorbar(scatter, ax=ax2)
            cb2.set_label(effect_modifiers[color_idx], rotation=270, labelpad=15)
            
            plt.suptitle(f'SHAP Dependence - {treatment.replace("_", " ").title()} '
                        f'({shap_sample_size:,} samples)', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / f'shap_dependence_{treatment}_{feature_name}_10k.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"  ✓ Saved: shap_dependence_{treatment}_{feature_name}_10k.png")


def _create_violin_plots(shap_by_treatment, effect_modifiers, treatment_names, output_dir,
                        shap_sample_size, verbose):
    """Create fixed violin plots (EXACT copy from original)"""
    # Create a single figure with all violins for comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for treat_idx, treatment in enumerate(['plants_richness', 'vertebrates_richness']):
        ax = axes[treat_idx]
        
        # Collect all SHAP values for each feature
        all_data = []
        positions = []
        labels = []
        
        for feat_idx, feature_name in enumerate(effect_modifiers):
            shap_vals = shap_by_treatment[treatment][:, feat_idx]
            all_data.append(shap_vals)
            positions.append(feat_idx)
            labels.append(feature_name)
        
        # Create violin plot with all features side by side
        parts = ax.violinplot(all_data, positions=positions, widths=0.8,
                             showmeans=True, showmedians=True, showextrema=True)
        
        # Color the violins
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Different color for each feature
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        
        # Customize the plot
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('SHAP value (impact on treatment effect)', fontsize=11)
        ax.set_title(f'{treatment.replace("_", " ").title()}', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Add statistics annotations
        for i, (data, label) in enumerate(zip(all_data, labels)):
            stats_text = f'μ={np.mean(data):.4f}\nσ={np.std(data):.4f}'
            ax.text(i, ax.get_ylim()[1] * 0.95, stats_text, 
                   ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'SHAP Value Distributions by Feature ({shap_sample_size:,} samples)', 
                fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_distributions_10k.png', dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✓ Saved: shap_distributions_10k.png")


def _create_comparative_violin_plot(shap_by_treatment, effect_modifiers, treatment_names,
                                   output_dir, shap_sample_size, verbose):
    """Create comparative violin plot (EXACT copy from original)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for comparison
    all_data = []
    positions = []
    colors_list = []
    labels_list = []
    
    for treat_idx, treatment in enumerate(['plants_richness', 'vertebrates_richness']):
        for feat_idx, feature_name in enumerate(effect_modifiers):
            shap_vals = shap_by_treatment[treatment][:, feat_idx]
            all_data.append(shap_vals)
            
            # Position: group by feature, separate by treatment
            pos = feat_idx * 2.5 + treat_idx * 0.9
            positions.append(pos)
            
            # Colors: different for each treatment
            colors_list.append('green' if treat_idx == 0 else 'blue')
            
            # Labels
            if treat_idx == 0:
                labels_list.append(feature_name)
    
    # Create violin plot
    parts = ax.violinplot(all_data, positions=positions, widths=0.8,
                         showmeans=True, showmedians=True, showextrema=False)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.6)
    
    # Set up x-axis
    ax.set_xticks([i * 2.5 + 0.45 for i in range(len(effect_modifiers))])
    ax.set_xticklabels(effect_modifiers)
    ax.set_xlabel('Climate Features', fontsize=12)
    ax.set_ylabel('SHAP value', fontsize=12)
    ax.set_title('Comparative SHAP Distributions: Plants (green) vs Vertebrates (blue)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.6, label='Plants richness'),
                      Patch(facecolor='blue', alpha=0.6, label='Vertebrates richness')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_distributions_comparative_10k.png', dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print("  ✓ Saved: shap_distributions_comparative_10k.png")


def _create_interaction_heatmaps(shap_by_treatment, X_shap, effect_modifiers,
                                treatment_names, output_dir, shap_sample_size, verbose):
    """Create 2D interaction heatmaps (EXACT copy from original)"""
    for treatment in ['plants_richness', 'vertebrates_richness']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Create heatmaps for each pair of features
        pairs = [(0, 1), (0, 2), (1, 2)]
        pair_names = [(effect_modifiers[i], effect_modifiers[j]) for i, j in pairs]
        
        for idx, ((i, j), (name_i, name_j)) in enumerate(zip(pairs, pair_names)):
            ax = axes[idx]
            
            # Get SHAP values for both features
            shap_i = shap_by_treatment[treatment][:, i]
            shap_j = shap_by_treatment[treatment][:, j]
            
            # Create 2D histogram
            h, xedges, yedges = np.histogram2d(X_shap[:, i], X_shap[:, j], 
                                              bins=25, weights=shap_i + shap_j)
            
            # Plot heatmap
            im = ax.imshow(h.T, origin='lower', aspect='auto', cmap='RdBu_r',
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            
            ax.set_xlabel(name_i)
            ax.set_ylabel(name_j)
            ax.set_title(f'{name_i} vs {name_j}')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Combined SHAP effect')
        
        plt.suptitle(f'Feature Interaction Effects - {treatment.replace("_", " ").title()}', 
                    fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'shap_interactions_{treatment}_10k.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        if verbose:
            print(f"  ✓ Saved: shap_interactions_{treatment}_10k.png")