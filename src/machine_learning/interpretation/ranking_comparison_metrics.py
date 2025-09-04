"""
Ranking Comparison Metrics for ML Feature Importance Methods

This module provides metrics to compare rankings from different feature importance methods
(e.g., coefficients, SHAP, PFI). All metrics treat features equally without weighting.

Usage:
    # Prepare your data as dictionaries
    coef_importance = {'feature1': 0.8, 'feature2': 0.3, ...}
    shap_importance = {'feature1': 0.7, 'feature2': 0.4, ...}
    pfi_importance = {'feature1': 0.9, 'feature2': 0.2, ...}
    
    # Run comparison
    results = compare_rankings(coef_importance, shap_importance, pfi_importance)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any


def prepare_ranking_df(importance_dicts: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Convert importance dictionaries to a DataFrame with values and ranks.
    
    Args:
        importance_dicts: {'method_name': {'feature': importance_value}}
    
    Returns:
        DataFrame with features as rows and methods as columns (values and ranks)
    """
    # Get all features
    all_features = set()
    for method_dict in importance_dicts.values():
        all_features.update(method_dict.keys())
    
    # Create DataFrame
    df = pd.DataFrame(index=sorted(all_features))
    
    # Add values and ranks for each method
    for method_name, importance_dict in importance_dicts.items():
        # Add absolute values
        df[f'{method_name}_value'] = df.index.map(lambda x: abs(importance_dict.get(x, 0)))
        # Add ranks (1 = most important)
        df[f'{method_name}_rank'] = df[f'{method_name}_value'].rank(ascending=False, method='min')
    
    return df


def correlation_metrics(df: pd.DataFrame, method1: str, method2: str) -> Dict[str, float]:
    """
    Calculate correlation-based metrics between two ranking methods.
    
    Returns:
        Dictionary with Spearman's rho and Kendall's tau
    """
    rank1 = df[f'{method1}_rank']
    rank2 = df[f'{method2}_rank']
    
    spearman_rho, spearman_p = spearmanr(rank1, rank2)
    kendall_tau, kendall_p = kendalltau(rank1, rank2)
    
    return {
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p
    }


def set_metrics(df: pd.DataFrame, method1: str, method2: str, k_values: List[int] = [5, 10]) -> Dict[str, float]:
    """
    Calculate set-based metrics for top-k features.
    
    Returns:
        Dictionary with Jaccard index, Precision@k, and Overlap coefficient
    """
    results = {}
    
    for k in k_values:
        # Get top-k features for each method
        top_k1 = set(df.nsmallest(k, f'{method1}_rank').index)
        top_k2 = set(df.nsmallest(k, f'{method2}_rank').index)
        
        # Calculate metrics
        intersection = len(top_k1 & top_k2)
        union = len(top_k1 | top_k2)
        
        results[f'jaccard_top{k}'] = intersection / union if union > 0 else 0
        results[f'precision_top{k}'] = intersection / k
        results[f'overlap_top{k}'] = intersection / min(len(top_k1), len(top_k2)) if min(len(top_k1), len(top_k2)) > 0 else 0
    
    return results


def distance_metrics(df: pd.DataFrame, method1: str, method2: str) -> Dict[str, float]:
    """
    Calculate distance-based metrics between rankings.
    
    Returns:
        Dictionary with Footrule distance, Kendall distance, and Hamming distance
    """
    rank1 = df[f'{method1}_rank'].values
    rank2 = df[f'{method2}_rank'].values
    n = len(rank1)
    
    # Spearman's Footrule Distance
    footrule = np.sum(np.abs(rank1 - rank2))
    max_footrule = n * (n**2 - 1) / 3  # Maximum possible footrule
    normalized_footrule = footrule / max_footrule if max_footrule > 0 else 0
    
    # Kendall Distance (number of discordant pairs)
    kendall_distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (rank1[i] < rank1[j] and rank2[i] > rank2[j]) or \
               (rank1[i] > rank1[j] and rank2[i] < rank2[j]):
                kendall_distance += 1
    max_kendall = n * (n - 1) / 2
    normalized_kendall = kendall_distance / max_kendall if max_kendall > 0 else 0
    
    # Hamming Distance for top-10
    k = min(10, n)
    top_k1 = rank1 <= k
    top_k2 = rank2 <= k
    hamming = np.mean(top_k1 != top_k2)
    
    return {
        'footrule_distance': footrule,
        'normalized_footrule': normalized_footrule,
        'kendall_distance': kendall_distance,
        'normalized_kendall': normalized_kendall,
        'hamming_top10': hamming
    }


def ml_specific_metrics(df: pd.DataFrame, method1: str, method2: str, 
                        importance_threshold: float = 0.1,
                        stability_tolerance: int = 2) -> Dict[str, Any]:
    """
    Calculate ML-specific metrics for feature importance comparison.
    
    Args:
        importance_threshold: Threshold for "critical" features (relative to max)
        stability_tolerance: Maximum rank change for a feature to be considered "stable"
    
    Returns:
        Dictionary with critical features agreement, stability metrics, and displacement stats
    """
    # Critical features agreement
    max1 = df[f'{method1}_value'].max()
    max2 = df[f'{method2}_value'].max()
    
    critical1 = set(df[df[f'{method1}_value'] > max1 * importance_threshold].index)
    critical2 = set(df[df[f'{method2}_value'] > max2 * importance_threshold].index)
    
    if len(critical1) > 0 or len(critical2) > 0:
        critical_agreement = len(critical1 & critical2) / len(critical1 | critical2)
    else:
        critical_agreement = 1.0
    
    # Stability metrics
    rank_changes = df[f'{method2}_rank'] - df[f'{method1}_rank']
    n_stable = np.sum(np.abs(rank_changes) <= stability_tolerance)
    stability_pct = n_stable / len(df) * 100
    
    # Displacement statistics
    displacement_stats = {
        'mean_displacement': rank_changes.mean(),
        'std_displacement': rank_changes.std(),
        'max_upward_jump': rank_changes.min(),  # Negative means moving up in rank
        'max_downward_jump': rank_changes.max(),
        'features_stable': n_stable,
        'stability_percentage': stability_pct
    }
    
    return {
        'critical_agreement': critical_agreement,
        'critical_features_1': list(critical1),
        'critical_features_2': list(critical2),
        **displacement_stats
    }


def create_visualizations(df: pd.DataFrame, method_names: List[str], output_prefix: str = 'ranking_comparison'):
    """
    Create visualization plots for ranking comparisons.
    
    Creates:
    1. Displacement histogram
    2. Bubble plot of rank changes
    3. Heatmap of top-10 features across methods
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Displacement histogram (comparing first two methods)
    if len(method_names) >= 2:
        rank_changes = df[f'{method_names[1]}_rank'] - df[f'{method_names[0]}_rank']
        axes[0, 0].hist(rank_changes, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(0, color='red', linestyle='--', label='No change')
        axes[0, 0].set_xlabel('Rank Change')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title(f'Rank Displacement: {method_names[0]} → {method_names[1]}')
        axes[0, 0].legend()
    
    # 2. Bubble plot showing agreement across all methods
    if len(method_names) >= 2:
        x = df[f'{method_names[0]}_rank']
        y = df[f'{method_names[1]}_rank']
        sizes = df[f'{method_names[0]}_value'] * 100  # Scale for visibility
        
        scatter = axes[0, 1].scatter(x, y, s=sizes, alpha=0.6, c=np.abs(x - y), cmap='coolwarm')
        axes[0, 1].plot([0, len(df)], [0, len(df)], 'k--', alpha=0.3)  # Identity line
        axes[0, 1].set_xlabel(f'{method_names[0]} Rank')
        axes[0, 1].set_ylabel(f'{method_names[1]} Rank')
        axes[0, 1].set_title('Rank Agreement Bubble Plot')
        plt.colorbar(scatter, ax=axes[0, 1], label='Rank Difference')
    
    # 3. Top-10 features heatmap
    top_features = set()
    for method in method_names:
        top_features.update(df.nsmallest(10, f'{method}_rank').index)
    
    heatmap_data = pd.DataFrame(index=sorted(top_features))
    for method in method_names:
        heatmap_data[method] = df.loc[heatmap_data.index, f'{method}_rank']
    
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                ax=axes[1, 0], cbar_kws={'label': 'Rank'})
    axes[1, 0].set_title('Top Features Rank Comparison')
    axes[1, 0].set_ylabel('Feature')
    
    # 4. Correlation matrix
    if len(method_names) >= 3:
        corr_matrix = pd.DataFrame(index=method_names, columns=method_names)
        for m1 in method_names:
            for m2 in method_names:
                if m1 == m2:
                    corr_matrix.loc[m1, m2] = 1.0
                else:
                    corr, _ = spearmanr(df[f'{m1}_rank'], df[f'{m2}_rank'])
                    corr_matrix.loc[m1, m2] = corr
        
        corr_matrix = corr_matrix.astype(float)
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                    vmin=-1, vmax=1, ax=axes[1, 1], square=True)
        axes[1, 1].set_title("Spearman's Rank Correlation Matrix")
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def compare_rankings(importance_dicts: Dict[str, Dict[str, float]], 
                     visualize: bool = True,
                     output_prefix: str = 'ranking_comparison') -> Dict[str, Any]:
    """
    Main function to compare rankings from multiple feature importance methods.
    
    Args:
        importance_dicts: {'method_name': {'feature': importance_value}}
        visualize: Whether to create visualization plots
        output_prefix: Prefix for saved plot files
    
    Returns:
        Comprehensive dictionary of all metrics
    """
    # Prepare data
    df = prepare_ranking_df(importance_dicts)
    method_names = list(importance_dicts.keys())
    
    results = {
        'data': df,
        'methods': method_names,
        'pairwise_comparisons': {}
    }
    
    # Pairwise comparisons
    for i, method1 in enumerate(method_names):
        for method2 in method_names[i+1:]:
            pair_key = f'{method1}_vs_{method2}'
            
            results['pairwise_comparisons'][pair_key] = {
                'correlation': correlation_metrics(df, method1, method2),
                'set_based': set_metrics(df, method1, method2),
                'distance': distance_metrics(df, method1, method2),
                'ml_specific': ml_specific_metrics(df, method1, method2)
            }
    
    # Create visualizations
    if visualize and len(method_names) >= 2:
        fig = create_visualizations(df, method_names, output_prefix)
        results['figure'] = fig
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: Dict[str, Any]):
    """
    Print a formatted summary of ranking comparison results.
    """
    print("=" * 80)
    print("RANKING COMPARISON SUMMARY")
    print("=" * 80)
    
    for pair_key, metrics in results['pairwise_comparisons'].items():
        print(f"\n{pair_key.replace('_', ' ').upper()}")
        print("-" * 40)
        
        # Correlation metrics
        print(f"Spearman's ρ: {metrics['correlation']['spearman_rho']:.4f}")
        print(f"Kendall's τ:  {metrics['correlation']['kendall_tau']:.4f}")
        
        # Set metrics
        for k in [5, 10]:
            if f'precision_top{k}' in metrics['set_based']:
                print(f"Top-{k} Precision: {metrics['set_based'][f'precision_top{k}']:.3f}")
        
        # Distance metrics
        print(f"Normalized Footrule: {metrics['distance']['normalized_footrule']:.4f}")
        
        # ML-specific
        print(f"Critical Features Agreement: {metrics['ml_specific']['critical_agreement']:.3f}")
        print(f"Stable Features: {metrics['ml_specific']['stability_percentage']:.1f}%")
        print(f"Mean Rank Change: {metrics['ml_specific']['mean_displacement']:.2f}")


if __name__ == "__main__":
    # Example usage
    example_coef = {
        'temp': 2.5, 'precip': -0.8, 'elevation': 0.9,
        'soil_ph': 0.1, 'vegetation': 0.4
    }
    example_shap = {
        'temp': 1.1, 'precip': 0.4, 'elevation': 0.3,
        'soil_ph': 0.05, 'vegetation': 0.2
    }
    example_pfi = {
        'temp': 1.3, 'precip': 0.5, 'elevation': 0.4,
        'soil_ph': 0.02, 'vegetation': 0.3
    }
    
    results = compare_rankings({
        'coefficient': example_coef,
        'shap': example_shap,
        'pfi': example_pfi
    })