from src.abstractions import get_logger
"""
Correlation analysis for biodiversity data - Linus style.

"Get your statistics right or get out."
No bullshit correlations that assume normality when your data is skewed to hell.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure
from src.biodiversity_analysis.statistical.result_types import MethodResult
from src.biodiversity_analysis.statistical.smart_visualization import smart_imshow

logger = get_logger(__name__)


def analyze_correlation(
    df: pd.DataFrame,
    output_dir: Path,
    **kwargs
) -> MethodResult:
    """
    PROPER correlation analysis - DataFrame ONLY.
    
    NO SAMPLING HERE. Data comes pre-processed from pipeline.
    
    Args:
        df: DataFrame with features to correlate
        output_dir: Where to save results
        **kwargs: Config including:
            - method: 'spearman', 'dcor', 'mi', 'rda', or 'all' (default: 'spearman')
            - columns: Specific columns to analyze (default: None)
            - plot: Generate visualizations (default: True)
        
    Returns:
        MethodResult with correlation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract config with defaults
    method = kwargs.get('method', 'spearman')
    columns = kwargs.get('columns', None)
    plot = kwargs.get('plot', True)
    
    # LINUS: Smart column selection - biodiversity + climate ONLY
    if columns:
        # User specified columns
        numeric_data = df[columns]
    else:
        # Auto-select biodiversity and key climate variables
        from src.biodiversity_analysis.statistical.data_schema import DataSchema
        biodiv_cols = DataSchema.get_biodiversity_columns(df)
        climate_cols = ['bio01', 'bio04', 'bio12', 'bio15']
        climate_cols = [c for c in climate_cols if c in df.columns]
        
        selected_cols = biodiv_cols + climate_cols
        # Filter out CV columns - they're NOT biodiversity metrics!
        selected_cols = [c for c in selected_cols if '_cv' not in c and 'coeff_var' not in c]
        
        numeric_data = df[selected_cols]
    
    results = {}
    
    if method in ['spearman', 'all']:
        results['spearman'] = _compute_spearman(numeric_data)
    
    # SKIP distance correlation and mutual information per user request
    # Only compute Spearman correlation for this run
    if method in ['dcor', 'all']:
        logger.info("SKIPPING distance correlation computation per user request")
        results['distance_correlation'] = {'skipped': True, 'reason': 'User requested to skip'}
    
    if method in ['mi', 'all']:
        logger.info("SKIPPING mutual information computation per user request")
        results['mutual_information'] = {'skipped': True, 'reason': 'User requested to skip'}
    
    # REMOVED partial correlation - it has its own dedicated module
    # Don't duplicate functionality!
    
    # SKIP RDA for now - has NaN issues
    if False and method in ['rda', 'all'] and len(numeric_data.columns) > 2:
        # Check data size and use optimized version for large data
        n_samples = len(numeric_data)
        if n_samples > 100000:  # 100K threshold for optimization
            logger.warning(f"Using OPTIMIZED RDA for {n_samples:,} points")
            from src.biodiversity_analysis.statistical.optimized_rda import compute_rda_rigorous
            # Split into response and explanatory variables
            # Assume first half are response (biodiversity), second half are explanatory (climate)
            n_vars = len(numeric_data.columns)
            mid = n_vars // 2
            Y_df = numeric_data.iloc[:, :mid]
            X_df = numeric_data.iloc[:, mid:]
            results['rda'] = compute_rda_rigorous(Y_df, X_df, n_components=2, max_memory_gb=100.0)
        else:
            # Original implementation for smaller data
            results['rda'] = _compute_rda(numeric_data)
    
    results['statistics'] = {
        'n_samples': len(df),
        'n_features': len(numeric_data.columns),
        'features': list(numeric_data.columns)
    }
    
    if plot and len(numeric_data.columns) > 1:
        _generate_plots(numeric_data, results, output_dir)
    
    save_results(results, output_dir / 'correlation_results.json')
    
    return MethodResult(
        success=True,
        method='correlation',
        output_dir=output_dir,
        metrics=results
    )


def _compute_spearman(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Spearman rank correlation - the MINIMUM acceptable method.
    At least it doesn't assume normality.
    """
    corr_matrix = df.corr(method='spearman')
    
    # Compute p-values
    n = len(df)
    cols = df.columns
    p_values = np.zeros((len(cols), len(cols)))
    
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i != j:
                r, p = stats.spearmanr(df[col1], df[col2])
                p_values[i, j] = p
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'p_values': pd.DataFrame(p_values, index=cols, columns=cols).to_dict(),
        'method': 'spearman_rank'
    }


# DELETED _compute_dcor - O(n²) is PHYSICALLY IMPOSSIBLE for 51M points
# Now using optimized_dcor.compute_dcor_matrix_fast with random projections


def _compute_mutual_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Mutual information - information theory FTW.
    How much does knowing one variable reduce uncertainty about another?
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import KBinsDiscretizer
    
    # Clean data first - MI can't handle NaN
    df_clean = df.dropna()
    if len(df_clean) < 100:
        logger.warning(f"Only {len(df_clean)} samples after removing NaN for MI")
        return {'error': 'Insufficient data after removing NaN'}
    
    cols = df_clean.columns
    n_cols = len(cols)
    mi_matrix = np.zeros((n_cols, n_cols))
    
    # Discretize for MI computation (required for continuous data)
    discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
    df_discrete = pd.DataFrame(
        discretizer.fit_transform(df_clean),
        columns=cols
    )
    
    # Compute pairwise MI
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i == j:
                # MI of variable with itself is its entropy
                # For normalized MI, set diagonal to 1
                mi_matrix[i, j] = 1.0
            else:
                mi = mutual_info_regression(
                    df_discrete[[col1]], 
                    df_discrete[col2],
                    discrete_features=True,
                    random_state=42
                )[0]
                # Normalize by min entropy for comparability
                h1 = mutual_info_regression(
                    df_discrete[[col1]], df_discrete[col1],
                    discrete_features=True, random_state=42
                )[0]
                h2 = mutual_info_regression(
                    df_discrete[[col2]], df_discrete[col2],
                    discrete_features=True, random_state=42
                )[0]
                mi_matrix[i, j] = mi / min(h1, h2) if min(h1, h2) > 0 else 0
    
    return {
        'mutual_info_matrix': pd.DataFrame(
            mi_matrix, index=cols, columns=cols
        ).to_dict(),
        'method': 'normalized_mutual_information',
        'interpretation': 'Values range [0,1]. Higher means more information shared'
    }


# DELETED _compute_partial_correlations - duplicate functionality
# Use the dedicated partial_correlation module instead!


def _compute_rda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Redundancy Analysis - multivariate relationships.
    Only on sampled data because this is expensive.
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler
    
    # Clean data first - RDA can't handle NaN
    df_clean = df.dropna()
    if len(df_clean) < 100:
        logger.warning(f"Only {len(df_clean)} samples after removing NaN for RDA")
        return {'error': 'Insufficient data after removing NaN'}
    
    # Use full data - no subsampling
    # If too large, pipeline should handle sampling
    
    # Standardize data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_clean),
        columns=df_clean.columns
    )
    
    # Split into groups for CCA (example: first half vs second half of variables)
    mid = len(df_scaled.columns) // 2
    if mid < 1:
        return {'error': 'Need at least 2 variables for RDA'}
    
    X = df_scaled.iloc[:, :mid]
    Y = df_scaled.iloc[:, mid:]
    
    # Fit CCA
    n_components = min(2, min(X.shape[1], Y.shape[1]))
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    
    # Get canonical correlations
    X_c, Y_c = cca.transform(X, Y)
    canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] 
                       for i in range(n_components)]
    
    return {
        'canonical_correlations': canonical_corrs,
        'n_components': n_components,
        'x_vars': list(X.columns),
        'y_vars': list(Y.columns),
        'method': 'redundancy_analysis',
        'interpretation': 'Canonical correlations show strength of multivariate relationships'
    }


def _generate_plots(df: pd.DataFrame, results: Dict[str, Any], output_dir: Path):
    """Generate PROPER visualization for correlation methods.
    
    LINUS: Generate TWO matrices per method:
    1. Biodiversity-only correlations
    2. Biodiversity-climate cross-correlations
    """
    
    # Get biodiversity and climate columns
    from src.biodiversity_analysis.statistical.data_schema import DataSchema
    biodiv_cols = DataSchema.get_biodiversity_columns(df)
    climate_cols = [c for c in ['bio01', 'bio04', 'bio12', 'bio15'] if c in df.columns]
    
    # Count valid methods (exclude errors and statistics)
    valid_methods = [k for k in results.keys() 
                    if k != 'statistics' and 'error' not in results.get(k, {})]
    
    if not valid_methods:
        logger.warning("No valid correlation results to plot")
        return
    
    # Generate separate plots for each method
    for method_name in valid_methods:
        _generate_method_plots(df, results[method_name], method_name, 
                             biodiv_cols, climate_cols, output_dir)


def _generate_method_plots(df: pd.DataFrame, method_result: Dict[str, Any], 
                          method_name: str, biodiv_cols: List[str], 
                          climate_cols: List[str], output_dir: Path):
    """Generate biodiversity-only correlation matrix for each method.
    
    LINUS: No climate cross-correlation needed!
    """
    
    # Get correlation matrix from result
    matrix_key = 'correlation_matrix' if 'correlation_matrix' in method_result else 'mutual_info_matrix'
    if matrix_key not in method_result:
        return  # Skip if no matrix available
    
    full_matrix = pd.DataFrame(method_result[matrix_key])
    
    # Single plot for biodiversity only
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    # Biodiversity-only correlations
    if len(biodiv_cols) > 1:
        biodiv_matrix = full_matrix.loc[biodiv_cols, biodiv_cols]
        
        # Choose colormap based on method
        if method_name == 'spearman':
            cmap, vmin, vmax = 'RdBu_r', -1, 1
            title = 'Spearman Rank Correlation'
        elif method_name == 'distance_correlation':
            cmap, vmin, vmax = 'viridis', 0, 1
            title = 'Distance Correlation'
        else:  # mutual_information
            cmap, vmin, vmax = 'YlOrRd', 0, 1
            title = 'Mutual Information'
        
        im = smart_imshow(ax, biodiv_matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'{title}\n(Biodiversity Features)', fontsize=12, fontweight='bold')
        _add_labels_and_values(ax, biodiv_matrix, im, fmt='.2f')
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'{method_name}_matrix.png')
    plt.close(fig)


def _add_labels_and_values(ax, df, im, fmt='.2f'):
    """Add labels and values to heatmap."""
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticklabels(df.columns)
    
    # Add text annotations
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            value = df.iloc[i, j]
            text_color = 'white' if abs(value - 0.5) > 0.3 else 'black'
            ax.text(j, i, format(value, fmt), ha='center', va='center',
                   color=text_color, fontsize=9)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _generate_comparison_plot(results: Dict[str, Any], output_dir: Path):
    """Generate comparison plot showing all methods side by side."""
    
    valid_methods = [k for k in results.keys() 
                    if k != 'statistics' and 'error' not in results.get(k, {})]
    
    if len(valid_methods) < 2:
        return  # Need at least 2 methods to compare
    
    # Extract key correlations for comparison
    features = list(results.get('statistics', {}).get('features', []))
    if len(features) < 2:
        return
    
    # Create comparison data
    comparison_data = []
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i >= j:  # Skip diagonal and lower triangle
                continue
            row = {'Pair': f'{feat1}-{feat2}'}
            
            # Add values from each method
            if 'spearman' in valid_methods:
                row['Spearman'] = results['spearman']['correlation_matrix'][feat1][feat2]
            if 'distance_correlation' in valid_methods:
                row['Distance'] = results['distance_correlation']['correlation_matrix'][feat1][feat2]
            if 'mutual_information' in valid_methods:
                row['MI'] = results['mutual_information']['mutual_info_matrix'][feat1][feat2]
            if 'partial_correlation' in valid_methods:
                # Partial correlation matrix may only have biodiv variables
                partial_matrix = results['partial_correlation'].get('partial_correlation_matrix', {})
                if feat1 in partial_matrix and feat2 in partial_matrix[feat1]:
                    row['Partial'] = partial_matrix[feat1][feat2]
            
            comparison_data.append(row)
    
    if not comparison_data:
        return
    
    # Create bar plot comparison
    comp_df = pd.DataFrame(comparison_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(comp_df))
    width = 0.25
    offset = 0
    
    colors = {'Spearman': 'steelblue', 'Distance': 'forestgreen', 'MI': 'coral', 'Partial': 'purple'}
    
    for method in ['Spearman', 'Distance', 'MI', 'Partial']:
        if method in comp_df.columns:
            ax.bar(x + offset, comp_df[method], width, label=method, 
                  color=colors.get(method, 'gray'), alpha=0.8)
            offset += width
    
    ax.set_xlabel('Variable Pairs', fontsize=11)
    ax.set_ylabel('Correlation/Dependency Strength', fontsize=11)
    ax.set_title('Comparison of Correlation Methods', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(comp_df['Pair'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'correlation_comparison.png')
    plt.close(fig)

