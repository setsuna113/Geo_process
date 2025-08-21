"""Score test visualization for hypothesis testing.

Extracted from hypothesis/score_test.py lines 593-870.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def visualize_score_test(test_results: Dict[str, Any],
                        variable_names: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (14, 5)):
    """Create publication-quality visualization of Score Test results.
    
    EXACT copy from score_test.py lines 593-728.
    
    Args:
        test_results: Results from any of the test methods
        variable_names: Names of variables tested
        figsize: Figure size
        
    Returns:
        matplotlib figure suitable for publication
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Extract statistics
    statistic = test_results.get('score_statistic', test_results.get('test_statistic', 0))
    p_value = test_results.get('p_value', 1)
    df = test_results.get('degrees_of_freedom', test_results.get('df', 1))
    significant = test_results.get('significant_at_0.05', p_value < 0.05)
    
    # Panel 1: Chi-square distribution with test statistic
    ax1 = axes[0]
    # Ensure we have a reasonable x range for visualization
    max_x = max(statistic * 1.5, stats.chi2.ppf(0.999, df), 10)
    # For very large statistics, limit the range for better visualization
    if statistic > 100:
        max_x = min(max_x, statistic * 1.2)
    x = np.linspace(0, max_x, 1000)
    y = stats.chi2.pdf(x, df)
    
    # Plot distribution
    ax1.plot(x, y, 'b-', linewidth=2, label=f'χ²({df}) distribution')
    
    # Shade rejection region
    critical_val = stats.chi2.ppf(0.95, df)
    x_reject = x[x > critical_val]
    y_reject = stats.chi2.pdf(x_reject, df)
    ax1.fill_between(x_reject, y_reject, alpha=0.3, color='red', 
                     label=f'Rejection region (α=0.05)')
    
    # Mark test statistic
    ax1.axvline(statistic, color='darkred', linestyle='--', linewidth=2.5,
               label=f'Observed S = {statistic:.2f}')
    
    # Mark critical value
    ax1.axvline(critical_val, color='orange', linestyle=':', linewidth=2,
               label=f'Critical value = {critical_val:.2f}')
    
    ax1.set_xlabel('Test Statistic Value', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title('Score Test Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max_x)
    
    # Add note if statistic is very large
    if statistic > stats.chi2.ppf(0.9999, df):
        ax1.text(max_x * 0.5, max(y) * 0.8, 
                f'Note: Test statistic ({statistic:.0f})\nis extremely significant',
                ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Panel 2: P-value visualization
    ax2 = axes[1]
    
    # Create significance level bars
    sig_levels = [0.001, 0.01, 0.05, 0.10]
    sig_labels = ['p < 0.001\n(***)', 'p < 0.01\n(**)', 'p < 0.05\n(*)', 'p < 0.10\n(†)']
    colors = ['#8B0000', '#DC143C', '#FF6347', '#FFA07A']
    
    # Plot horizontal bars for significance levels
    for i, (level, label, color) in enumerate(zip(sig_levels, sig_labels, colors)):
        ax2.barh(i, level, height=0.8, color=color, alpha=0.6)
        ax2.text(level + 0.002, i, label, va='center', fontsize=9)
    
    # Mark observed p-value
    if p_value <= 0.15:  # Only show if in reasonable range
        ax2.axvline(p_value, color='black', linestyle='-', linewidth=3,
                   label=f'Observed\np = {p_value:.4f}')
        # Add arrow pointing to p-value
        ax2.annotate('', xy=(p_value, -0.5), xytext=(p_value, -1.2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax2.set_xlim(0, 0.12)
    ax2.set_ylim(-1.5, 4)
    ax2.set_xlabel('P-value', fontsize=11)
    ax2.set_yticks([])
    ax2.set_title('Significance Levels', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    if p_value <= 0.15:
        ax2.legend(loc='upper right', fontsize=9)
    
    # Panel 3: Effect size and power analysis
    ax3 = axes[2]
    
    # Create power curve
    effect_sizes = np.linspace(0, 4, 100)
    ncp = effect_sizes**2 * 50  # Non-centrality parameter (scaled for visualization)
    power_05 = 1 - stats.ncx2.cdf(stats.chi2.ppf(0.95, df), df, ncp)
    power_01 = 1 - stats.ncx2.cdf(stats.chi2.ppf(0.99, df), df, ncp)
    
    ax3.plot(effect_sizes, power_05, 'b-', linewidth=2, label='α = 0.05')
    ax3.plot(effect_sizes, power_01, 'g--', linewidth=2, label='α = 0.01')
    
    # Mark 80% power threshold
    ax3.axhline(0.8, color='red', linestyle=':', alpha=0.5, linewidth=1.5,
               label='80% power')
    
    # Estimate and mark observed effect size (approximate)
    if significant:
        obs_effect = np.sqrt(statistic / 50)  # Rough approximation
        obs_power = 1 - stats.ncx2.cdf(stats.chi2.ppf(0.95, df), df, statistic)
        ax3.plot(obs_effect, obs_power, 'ro', markersize=10, 
                label=f'Observed\n(effect≈{obs_effect:.2f})')
    
    ax3.set_xlabel('Standardized Effect Size', fontsize=11)
    ax3.set_ylabel('Statistical Power', fontsize=11)
    ax3.set_title('Power Analysis', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', fontsize=9)
    
    # Overall title
    decision = "Significant" if significant else "Not Significant"
    if variable_names:
        var_str = f" ({', '.join(variable_names)})"
    else:
        var_str = ""
    
    fig.suptitle(f'Score Test Results{var_str}: {decision}',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig


def create_comparison_plot(score_result: Dict[str, Any],
                          lrt_result: Optional[Dict[str, Any]] = None,
                          wald_result: Optional[Dict[str, Any]] = None,
                          figsize: Tuple[int, int] = (10, 6)):
    """Create comparison plot of Score, LRT, and Wald tests.
    
    Shows that all three tests are asymptotically equivalent.
    NOTE: LRT results should come from Julia's lrtest() function.
    
    Args:
        score_result: Score test results
        lrt_result: Optional LRT results (from Julia)
        wald_result: Optional Wald test results
        figsize: Figure size
        
    Returns:
        Comparison figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect test statistics
    tests = []
    stats_vals = []
    p_vals = []
    colors = []
    
    # Score test
    tests.append('Score\n(LM)')
    stats_vals.append(score_result.get('score_statistic', 0))
    p_vals.append(score_result.get('p_value', 1))
    colors.append('#2E86AB')
    
    # LRT if available
    if lrt_result:
        tests.append('LRT')
        stats_vals.append(lrt_result.get('lr_statistic', 0))
        p_vals.append(lrt_result.get('p_value', 1))
        colors.append('#A23B72')
    
    # Wald if available
    if wald_result:
        tests.append('Wald')
        # Handle both chi2 and z statistics
        if 'chi2_statistic' in wald_result:
            stats_vals.append(wald_result['chi2_statistic'])
        elif 'wald_statistic' in wald_result:
            # If it's a z-statistic, square it for chi2
            z_stat = wald_result['wald_statistic']
            stats_vals.append(z_stat**2 if 'degrees_of_freedom' not in wald_result else z_stat)
        else:
            stats_vals.append(0)
        p_vals.append(wald_result.get('p_value', 1))
        colors.append('#F18F01')
    
    # Create grouped bar chart
    x = np.arange(len(tests))
    width = 0.35
    
    # Plot test statistics
    bars1 = ax.bar(x - width/2, stats_vals, width, label='Test Statistic',
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add p-values as second bars (scaled for visibility)
    p_vals_scaled = [-np.log10(p + 1e-10) for p in p_vals]  # -log10 for visibility
    bars2 = ax.bar(x + width/2, p_vals_scaled, width, label='-log₁₀(p-value)',
                   color=colors, alpha=0.4, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars1, stats_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    for bar, p_val in zip(bars2, p_vals):
        height = bar.get_height()
        if p_val < 0.001:
            p_text = 'p<0.001'
        else:
            p_text = f'p={p_val:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                p_text, ha='center', va='bottom', fontsize=9)
    
    # Add significance threshold line for -log10(0.05)
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5,
               label='Significance threshold (p=0.05)')
    
    # Formatting
    ax.set_xlabel('Test Type', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Comparison of Asymptotically Equivalent Tests', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tests)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add note about asymptotic equivalence
    note_text = "Note: Score, LRT, and Wald tests are asymptotically equivalent.\n"
    note_text += "Differences diminish as sample size increases."
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
            fontsize=9, style='italic', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig