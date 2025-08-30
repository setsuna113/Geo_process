"""
Partial correlation analysis - Control for confounding variables.

"Climate is MASKING the biodiversity effect!"
This module implements partial correlations to reveal TRUE relationships.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt
from scipy import stats

from src.biodiversity_analysis.statistical.result_types import MethodResult
from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure

logger = logging.getLogger(__name__)


def compute_partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str, 
    control_vars: Union[str, List[str]],
    method: str = 'spearman'
) -> Dict[str, float]:
    """
    Compute partial correlation between x and y, controlling for other variables.
    
    This is THE way to remove confounding effects.
    
    Args:
        df: DataFrame with all variables
        x: First variable name
        y: Second variable name
        control_vars: Variable(s) to control for
        method: 'pearson' or 'spearman'
        
    Returns:
        Dict with correlation coefficient and p-value
    """
    # LINUS: No fallbacks - either it works or it fails
    import pingouin as pg
    
    # Use pingouin's implementation - REQUIRED dependency
    if isinstance(control_vars, str):
        control_vars = [control_vars]
    
    result = pg.partial_corr(
        data=df,
        x=x,
        y=y,
        covar=control_vars,
        method=method
    )
    
    return {
        'r': float(result['r'].values[0]),
        'p': float(result['p-val'].values[0]),
        'n': int(result['n'].values[0])
    }


# REMOVED: _manual_partial_correlation function
# LINUS: "No fallbacks - use the right tool or fail"
# pingouin is a REQUIRED dependency


def analyze_partial_correlations(
    data: Any,
    output_dir: Path,
    **config  # ALL config comes here - NO SPECIAL ARGS
) -> MethodResult:
    """
    Analyze partial correlations between biodiversity variables.
    
    REVEALS the true relationships after removing climate confounding.
    
    Args:
        data: Arrow Table, Dict[str, np.ndarray], or DataFrame
        output_dir: Output directory
        **config: Configuration dict with:
            - target_pairs: List of (var1, var2) pairs
            - control_vars: Variables to control for
            - method: 'pearson' or 'spearman'
            - plot: Generate visualizations
        
    Returns:
        MethodResult comparing simple vs partial correlations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # LINUS: "Direct Arrow → DataFrame conversion. Skip the Dict nonsense."
    if isinstance(data, pa.Table):
        df = data.to_pandas()
        logger.info(f"Converted Arrow Table to DataFrame: {len(df):,} rows")
    elif isinstance(data, dict):
        # Legacy path for compatibility
        df = pd.DataFrame(data)
        logger.info(f"Converted {len(data)} arrays to DataFrame with {len(df):,} rows")
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        # Try to convert BiodiversityData
        if hasattr(data, 'features') and hasattr(data, 'coordinates'):
            df_dict = {
                'x': data.coordinates[:, 0],
                'y': data.coordinates[:, 1]
            }
            # Add features with standardized names
            for i, name in enumerate(data.feature_names):
                df_dict[name] = data.features[:, i]
            df = pd.DataFrame(df_dict)
            logger.info(f"Converted BiodiversityData to DataFrame with {len(df)} rows")
        else:
            raise TypeError(f"partial_correlation expects Arrow Table, Dict[str, array], DataFrame or BiodiversityData, got {type(data).__name__}")
    
    # LINUS: "total_fungi only, 3x3 matrix, control for ALL variables"
    # Ensure total_fungi exists
    if 'total_fungi' not in df.columns:
        logger.info("Creating total_fungi column")
        df['total_fungi'] = df['am_fungi_richness'] + df['ecm_fungi_richness']
    
    # Build 3x3 matrix pairs using ONLY total_fungi
    default_pairs = [
        ('animal_richness', 'plant_richness'),
        ('animal_richness', 'total_fungi'),
        ('plant_richness', 'total_fungi')
    ]
    
    target_pairs = config.get('target_pairs', default_pairs)
    
    # Control for ALL non-biodiversity variables in the arrow file
    # Exclude: coordinates, biodiversity columns, and the total_fungi we just created
    biodiv_cols = ['animal_richness', 'plant_richness', 'am_fungi_richness', 
                   'ecm_fungi_richness', 'total_fungi']
    coord_cols = ['x', 'y']
    
    # Get ALL other columns as control variables
    all_control_vars = [col for col in df.columns 
                       if col not in biodiv_cols and col not in coord_cols]
    
    logger.info(f"Controlling for ALL {len(all_control_vars)} variables: {all_control_vars}")
    control_vars = config.get('control_vars', all_control_vars)
    method = config.get('method', 'spearman')
    plot = config.get('plot', True)
    
    # Results storage
    results = {
        'method': method,
        'control_variables': control_vars,
        'comparisons': {}
    }
    
    # For each pair, compute both simple and partial correlations
    for var1, var2 in target_pairs:
        pair_name = f"{var1}_vs_{var2}"
        
        # Simple correlation (without controlling)
        # Need to align data first - dropna on BOTH columns together
        df_pair = df[[var1, var2]].dropna()
        if len(df_pair) < 10:
            logger.warning(f"Too few samples for {var1} vs {var2}: {len(df_pair)}")
            continue
            
        if method == 'spearman':
            r_simple, p_simple = stats.spearmanr(
                df_pair[var1], 
                df_pair[var2]
            )
        else:
            r_simple, p_simple = stats.pearsonr(
                df_pair[var1],
                df_pair[var2]
            )
        
        # Partial correlation (controlling for confounders)
        partial_result = compute_partial_correlation(
            df, var1, var2, control_vars, method
        )
        
        # Store comparison
        results['comparisons'][pair_name] = {
            'simple': {
                'r': float(r_simple),
                'p': float(p_simple),
                'interpretation': _interpret_correlation(r_simple)
            },
            'partial': {
                'r': partial_result['r'],
                'p': partial_result['p'],
                'n': partial_result['n'],
                'interpretation': _interpret_correlation(partial_result['r'])
            },
            'change': {
                'delta_r': partial_result['r'] - r_simple,
                'effect': 'REVEALED' if abs(partial_result['r']) > abs(r_simple) else 'MASKED'
            }
        }
        
        logger.info(f"{pair_name}: Simple r={r_simple:.3f}, Partial r={partial_result['r']:.3f}")
    
    # Generate visualization
    if plot:
        _plot_partial_comparison(results, output_dir)
    
    # Save results
    import json
    with open(output_dir / 'partial_correlation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    _generate_summary(results, output_dir)
    
    return MethodResult(
        success=True,
        method='partial_correlation',
        output_dir=output_dir,
        metrics=results
    )


# LINUS: Removed _compute_partial_correlation_results - no Ray remote execution needed!


def _interpret_correlation(r: float) -> str:
    """Interpret correlation strength."""
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


def _plot_partial_comparison(results: Dict[str, Any], output_dir: Path):
    """Create comparison plot of simple vs partial correlations.
    
    LINUS: Generate BOTH bar chart AND correlation matrices!
    """
    
    comparisons = results['comparisons']
    n_pairs = len(comparisons)
    
    # First generate matrix visualizations
    _plot_correlation_matrices(results, output_dir)
    
    # Then generate comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    pairs = list(comparisons.keys())
    simple_corrs = [comparisons[p]['simple']['r'] for p in pairs]
    partial_corrs = [comparisons[p]['partial']['r'] for p in pairs]
    
    # Bar plot comparison
    ax = axes[0]
    x = np.arange(n_pairs)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, simple_corrs, width, label='Simple Correlation',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, partial_corrs, width, label='Partial Correlation',
                   color='coral', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9)
    
    ax.set_xlabel('Variable Pairs', fontsize=11)
    ax.set_ylabel('Correlation Coefficient', fontsize=11)
    ax.set_title('Simple vs Partial Correlations\n(Controlling for Climate)', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', ' ') for p in pairs], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Scatter plot: Simple vs Partial
    ax = axes[1]
    ax.scatter(simple_corrs, partial_corrs, s=100, alpha=0.7, color='purple')
    
    # Add diagonal line
    # LINUS: Check if lists are empty before min/max - no special cases!
    if simple_corrs and partial_corrs:
        lims = [min(simple_corrs + partial_corrs) - 0.1,
                max(simple_corrs + partial_corrs) + 0.1]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='No change')
    else:
        # Default range if no data
        lims = [-1.1, 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='No change')
    
    # Add labels
    for i, pair in enumerate(pairs):
        ax.annotate(pair.split('_vs_')[0][:3] + '-' + pair.split('_vs_')[1][:3],
                   (simple_corrs[i], partial_corrs[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8)
    
    ax.set_xlabel('Simple Correlation', fontsize=11)
    ax.set_ylabel('Partial Correlation', fontsize=11)
    ax.set_title('Effect of Controlling for Climate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Color regions (removed 'where' parameter - not supported)
    # ax.axhspan(0, 1, alpha=0.1, color='green', label='Relationship revealed')
    # ax.axhspan(-1, 0, alpha=0.1, color='red', label='Relationship masked')
    
    plt.suptitle(f'Partial Correlation Analysis - {results["method"].title()} Method',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'partial_correlation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_correlation_matrices(results: Dict[str, Any], output_dir: Path):
    """Generate correlation matrices for simple and partial correlations."""
    import seaborn as sns
    
    comparisons = results['comparisons']
    
    # Extract unique variables
    all_vars = set()
    for pair_name in comparisons.keys():
        vars = pair_name.split('_vs_')
        all_vars.update(vars)
    all_vars = sorted(list(all_vars))
    n_vars = len(all_vars)
    
    # Build matrices
    simple_matrix = np.full((n_vars, n_vars), np.nan)
    partial_matrix = np.full((n_vars, n_vars), np.nan)
    
    # Fill diagonal
    np.fill_diagonal(simple_matrix, 1.0)
    np.fill_diagonal(partial_matrix, 1.0)
    
    # Fill off-diagonal
    for pair_name, data in comparisons.items():
        vars = pair_name.split('_vs_')
        if len(vars) == 2:
            i = all_vars.index(vars[0])
            j = all_vars.index(vars[1])
            
            # Fill both triangles
            simple_matrix[i, j] = data['simple']['r']
            simple_matrix[j, i] = data['simple']['r']
            partial_matrix[i, j] = data['partial']['r']
            partial_matrix[j, i] = data['partial']['r']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot simple correlations
    sns.heatmap(simple_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, square=True,
                xticklabels=all_vars, yticklabels=all_vars,
                ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title('Simple Correlations\n(No control variables)', fontweight='bold')
    
    # Plot partial correlations
    sns.heatmap(partial_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, square=True,
                xticklabels=all_vars, yticklabels=all_vars,
                ax=ax2, cbar_kws={'label': 'Correlation'})
    control_str = ', '.join(results['control_variables'][:3])
    if len(results['control_variables']) > 3:
        control_str += '...'
    ax2.set_title(f'Partial Correlations\n(Controlling for {control_str})', fontweight='bold')
    
    plt.suptitle('Biodiversity Correlation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'partial_correlation_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved correlation matrices to {output_dir / 'partial_correlation_matrices.png'}")


def _generate_summary(results: Dict[str, Any], output_dir: Path):
    """Generate text summary of findings."""
    
    summary = []
    summary.append("=" * 60)
    summary.append("PARTIAL CORRELATION ANALYSIS SUMMARY")
    summary.append("=" * 60)
    summary.append("")
    summary.append(f"Method: {results['method'].upper()}")
    summary.append(f"Controlling for: {', '.join(results['control_variables'])}")
    summary.append("")
    
    for pair_name, data in results['comparisons'].items():
        # Format the pair name for display
        display_name = pair_name.replace('_vs_', ' vs ').replace('_', ' ').title()
        summary.append(f"\n{display_name}:")
        summary.append("-" * 40)
        
        simple = data['simple']
        partial = data['partial']
        change = data['change']
        
        summary.append(f"Simple correlation:  r = {simple['r']:+.3f} (p = {simple['p']:.3e})")
        summary.append(f"                     {simple['interpretation']}")
        summary.append("")
        summary.append(f"Partial correlation: r = {partial['r']:+.3f} (p = {partial['p']:.3e})")
        summary.append(f"                     {partial['interpretation']}")
        summary.append("")
        summary.append(f"Change: Δr = {change['delta_r']:+.3f}")
        summary.append(f"Effect: Climate was {change['effect']} the relationship")
        
        # Add interpretation
        if abs(change['delta_r']) > 0.2:
            summary.append(">>> MAJOR CONFOUNDING EFFECT DETECTED!")
            if change['effect'] == 'REVEALED':
                summary.append("    True relationship is STRONGER than observed")
            else:
                summary.append("    True relationship is WEAKER than observed")
    
    summary.append("")
    summary.append("=" * 60)
    summary.append("INTERPRETATION:")
    summary.append("When climate effects are removed, the TRUE biodiversity")
    summary.append("relationships are revealed. Positive Δr means climate was")
    summary.append("masking a stronger relationship.")
    summary.append("=" * 60)
    
    # Save summary
    with open(output_dir / 'partial_correlation_summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    
    # Also print to console
    print('\n'.join(summary))


# Convenience function for biodiversity analysis
def analyze_biodiversity_climate_confounding(
    parquet_path: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Specific analysis for biodiversity-fungi relationships controlling for climate.
    
    This is what you REALLY want - the truth behind the correlations.
    """
    # Define target relationships - REAL production data names!
    target_pairs = [
        ('plant_richness', 'am_fungi_richness'),
        ('plant_richness', 'ecm_fungi_richness'),
        ('animal_richness', 'am_fungi_richness'),
        ('animal_richness', 'ecm_fungi_richness'),
    ]
    
    # Climate variables to control for
    climate_vars = ['bio01', 'bio04', 'bio12', 'bio15', 'dem']
    
    return analyze_partial_correlations(
        data_source=parquet_path,
        output_dir=output_dir,
        target_pairs=target_pairs,
        control_vars=climate_vars,
        method='spearman',
        plot=True
    )