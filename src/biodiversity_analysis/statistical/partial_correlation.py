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
import matplotlib.pyplot as plt
from scipy import stats

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
    try:
        import pingouin as pg
    except ImportError:
        # Fallback to manual implementation
        return _manual_partial_correlation(df, x, y, control_vars, method)
    
    # Use pingouin's efficient implementation
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


def _manual_partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    control_vars: Union[str, List[str]],
    method: str = 'spearman'
) -> Dict[str, float]:
    """
    Manual implementation of partial correlation using residuals.
    
    The REAL algorithm:
    1. Regress x on control variables -> get residuals
    2. Regress y on control variables -> get residuals  
    3. Correlate the residuals
    """
    from sklearn.linear_model import LinearRegression
    
    if isinstance(control_vars, str):
        control_vars = [control_vars]
    
    # Remove rows with any NaN
    cols = [x, y] + control_vars
    df_clean = df[cols].dropna()
    
    # Get control variables as matrix
    Z = df_clean[control_vars].values
    X = df_clean[x].values.reshape(-1, 1)
    Y = df_clean[y].values.reshape(-1, 1)
    
    # Regress X on Z
    reg_x = LinearRegression()
    reg_x.fit(Z, X)
    residuals_x = X - reg_x.predict(Z)
    
    # Regress Y on Z
    reg_y = LinearRegression()
    reg_y.fit(Z, Y)
    residuals_y = Y - reg_y.predict(Z)
    
    # Correlate residuals
    if method == 'spearman':
        r, p = stats.spearmanr(residuals_x.flatten(), residuals_y.flatten())
    else:
        r, p = stats.pearsonr(residuals_x.flatten(), residuals_y.flatten())
    
    return {
        'r': float(r),
        'p': float(p),
        'n': len(df_clean)
    }


def analyze_partial_correlations(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    target_pairs: List[tuple],
    control_vars: List[str],
    method: str = 'spearman',
    plot: bool = True,
    **kwargs  # Ignore extra config
) -> Dict[str, Any]:
    """
    Analyze partial correlations - DataFrame ONLY.
    
    REVEALS the true relationships after removing climate confounding.
    
    Args:
        df: DataFrame with all variables
        output_dir: Output directory
        target_pairs: List of (var1, var2) pairs to analyze
        control_vars: Variables to control for (e.g., climate)
        method: 'pearson' or 'spearman'
        plot: Generate comparison plots
        
    Returns:
        Results comparing simple vs partial correlations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        if method == 'spearman':
            r_simple, p_simple = stats.spearmanr(
                df[var1].dropna(), 
                df[var2].dropna()
            )
        else:
            r_simple, p_simple = stats.pearsonr(
                df[var1].dropna(),
                df[var2].dropna()
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
    
    return results


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
    """Create comparison plot of simple vs partial correlations."""
    
    comparisons = results['comparisons']
    n_pairs = len(comparisons)
    
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
    lims = [min(simple_corrs + partial_corrs) - 0.1,
            max(simple_corrs + partial_corrs) + 0.1]
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
    
    # Color regions
    ax.axhspan(0, 1, where=(ax.get_xlim()[0] < 0), alpha=0.1, color='green',
              label='Relationship revealed')
    ax.axhspan(-1, 0, where=(ax.get_xlim()[0] > 0), alpha=0.1, color='red',
              label='Relationship masked')
    
    plt.suptitle(f'Partial Correlation Analysis - {results["method"].title()} Method',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'partial_correlation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


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
        var1, var2 = pair_name.replace('_vs_', ' vs ')
        summary.append(f"\n{var1} vs {var2}:")
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
    output_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    Specific analysis for biodiversity-fungi relationships controlling for climate.
    
    This is what you REALLY want - the truth behind the correlations.
    """
    # Define target relationships
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