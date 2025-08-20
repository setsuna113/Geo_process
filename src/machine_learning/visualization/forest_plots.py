"""Forest plot visualizations for Wald test results.

Provides comprehensive visualization for Wald tests including:
- Basic forest plot (coefficients with CIs)
- SESOI (Smallest Effect Size of Interest) forest plot
- TOST (Two One-Sided Tests) equivalence forest plot
- P-value significance plot
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def create_wald_visualization(
    wald_results: Dict[str, Any],
    sesoi_threshold: float = 0.1,  # Default SESOI threshold (10% change on response scale)
    tost_bounds: Tuple[float, float] = (-0.05, 0.05),  # Default TOST bounds (±5%)
    plot_type: str = "smart",  # "smart" (filtered+unfiltered), "comprehensive" (4 panels), "compact" (3 panels)
    title: str = "Wald Test: Comprehensive Analysis",
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """Create smart Wald test visualization.
    
    ALWAYS sorts by importance (|z-score|).
    ALWAYS cleans names for readability.
    Default "smart" mode shows both filtered (no MESF) and unfiltered views.
    
    Args:
        wald_results: Results from Wald tests containing coefficients, std_errors, p_values
        sesoi_threshold: Threshold for smallest effect size of interest (on log scale)
        tost_bounds: Equivalence bounds for TOST (on log scale)
        plot_type: "smart" (default - filtered+unfiltered), "comprehensive" (4 panels), "compact" (3 panels)
        title: Main title for the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure with smart layout showing what users actually need
    """
    if plot_type == "smart":
        # The RIGHT default - what users actually want
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, hspace=0.2, wspace=0.3)
        
        # Panel 1: FILTERED view (what users care about)
        ax1 = fig.add_subplot(gs[0])
        _plot_smart_forest(ax1, wald_results, sesoi_threshold, 
                          filter_mesf=True, title="Key Predictors (No MESF)")
        
        # Panel 2: COMPLETE view (for completeness)
        ax2 = fig.add_subplot(gs[1])
        _plot_smart_forest(ax2, wald_results, sesoi_threshold,
                          filter_mesf=False, title="All Predictors")
        
    elif plot_type == "comprehensive":
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: SESOI Forest Plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        _plot_sesoi_forest(ax1, wald_results, sesoi_threshold)
        
        # Panel 2: TOST Forest Plot (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        _plot_tost_forest(ax2, wald_results, tost_bounds)
        
        # Panel 3: P-value Plot (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        _plot_pvalues(ax3, wald_results)
        
        # Panel 4: Combined Summary (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        _plot_wald_summary(ax4, wald_results, sesoi_threshold, tost_bounds)
        
    elif plot_type == "compact":
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.75))
        gs = fig.add_gridspec(1, 3, wspace=0.3)
        
        # Panel 1: Combined Forest Plot with SESOI
        ax1 = fig.add_subplot(gs[0])
        _plot_sesoi_forest(ax1, wald_results, sesoi_threshold)
        
        # Panel 2: TOST Forest Plot
        ax2 = fig.add_subplot(gs[1])
        _plot_tost_forest(ax2, wald_results, tost_bounds)
        
        # Panel 3: P-value Plot
        ax3 = fig.add_subplot(gs[2])
        _plot_pvalues(ax3, wald_results)
    else:
        raise ValueError(f"plot_type must be 'smart', 'comprehensive' or 'compact', got {plot_type}")
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig




def _plot_smart_forest(ax: plt.Axes, wald_results: Dict[str, Any], 
                      sesoi_threshold: float, filter_mesf: bool = True,
                      title: str = "Forest Plot", top_n: Optional[int] = 15):
    """Smart forest plot that ALWAYS sorts by importance and cleans names.
    
    This is what users actually want to see.
    """
    # Clean name function
    def clean_name(name: str) -> str:
        """Make names human-readable."""
        return (name.replace('_log', ' (log)')
                   .replace('_', ' ')
                   .replace('bio', 'Bio')
                   .replace('plants', 'Plants')
                   .replace('vertebrates', 'Vertebrates')
                   .title())
    
    # Calculate importance (|z-score|) for ALL predictors
    importance_dict = {}
    for predictor, result in wald_results.items():
        if predictor == 'overall' or not isinstance(result, dict):
            continue
        coef = result.get('coefficient', 0)
        # Handle both 'std_error' and 'standard_error' keys
        se = result.get('std_error', result.get('standard_error', 1))
        if se > 0:
            importance_dict[predictor] = abs(coef / se)
        else:
            importance_dict[predictor] = 0
    
    # Sort by importance (ALWAYS!)
    sorted_predictors = sorted(importance_dict.keys(), 
                              key=lambda x: importance_dict[x], 
                              reverse=True)
    
    # Filter if requested
    if filter_mesf:
        sorted_predictors = [p for p in sorted_predictors 
                           if not p.lower().startswith('mesf') and p.lower() != '(intercept)']
    
    # Limit to top_n if specified
    if top_n and len(sorted_predictors) > top_n:
        sorted_predictors = sorted_predictors[:top_n]
    
    if not sorted_predictors:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title(title)
        return
    
    # Extract data in sorted order
    coefficients = []
    std_errors = []
    clean_names_list = []
    
    for predictor in sorted_predictors:
        result = wald_results[predictor]
        coefficients.append(result.get('coefficient', 0))
        # Handle both 'std_error' and 'standard_error' keys
        std_err = result.get('std_error', result.get('standard_error', 0))
        std_errors.append(std_err)
        clean_names_list.append(clean_name(predictor))
    
    # Calculate CIs
    coefficients = np.array(coefficients)
    std_errors = np.array(std_errors)
    ci_lower = coefficients - 1.96 * std_errors
    ci_upper = coefficients + 1.96 * std_errors
    
    # Color by interpretation from results
    colors = []
    for predictor in sorted_predictors:
        result = wald_results[predictor]
        # Check both direct interpretation and nested in summary
        interp = result.get('interpretation', '')
        if not interp and 'summary' in result:
            interp = result['summary'].get('interpretation', '')
        
        if 'Practically and statistically significant' in interp:
            # Check if positive or negative
            coef = result.get('coefficient', 0)
            colors.append('darkgreen' if coef > 0 else 'darkred')
        elif 'negligible' in interp.lower():
            colors.append('gray')
        elif 'below meaningful threshold' in interp:
            colors.append('orange')
        else:
            colors.append('lightblue')  # Uncertain/other
    
    # Plot
    y_pos = np.arange(len(sorted_predictors))
    
    for i, (coef, ci_l, ci_u, color) in enumerate(zip(coefficients, ci_lower, ci_upper, colors)):
        ax.errorbar(coef, i, xerr=[[coef - ci_l], [ci_u - coef]], 
                   fmt='o', color=color, markersize=8, 
                   capsize=5, capthick=2, elinewidth=2)
        
        # Add importance score as text
        importance = importance_dict[sorted_predictors[i]]
        ax.text(ci_u + 0.01, i, f'|z|={importance:.1f}', 
               fontsize=8, va='center', alpha=0.7)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=sesoi_threshold, color='blue', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=-sesoi_threshold, color='blue', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add SESOI region
    ax.axvspan(-sesoi_threshold, sesoi_threshold, alpha=0.05, color='gray')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names_list)
    ax.set_xlabel('Coefficient (log scale)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Smart legend
    legend_elements = [
        Patch(facecolor='darkgreen', label='Meaningful (+)'),
        Patch(facecolor='darkred', label='Meaningful (-)'),
        Patch(facecolor='orange', label='Below threshold'),
        Patch(facecolor='gray', label='Negligible')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)
    
    # Add note about sorting
    ax.text(0.98, 0.02, 'Sorted by |z-score|', transform=ax.transAxes,
           ha='right', fontsize=8, style='italic', alpha=0.7)


def _plot_sesoi_forest(ax: plt.Axes, wald_results: Dict[str, Any], sesoi_threshold: float):
    """Create forest plot with SESOI (Smallest Effect Size of Interest) threshold.
    
    Shows whether effects exceed the threshold for practical significance.
    """
    # Extract data (same as basic)
    predictors = []
    coefficients = []
    std_errors = []
    
    for predictor, result in wald_results.items():
        if predictor == 'overall' or not isinstance(result, dict):
            continue
        predictors.append(predictor)
        coefficients.append(result.get('coefficient', 0))
        # Handle both 'std_error' and 'standard_error' keys
        std_err = result.get('std_error', result.get('standard_error', 0))
        std_errors.append(std_err)
    
    if not predictors:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title('SESOI Forest Plot')
        return
    
    # Calculate CIs
    coefficients = np.array(coefficients)
    std_errors = np.array(std_errors)
    ci_lower = coefficients - 1.96 * std_errors
    ci_upper = coefficients + 1.96 * std_errors
    
    # Color by SESOI criteria
    colors = []
    for coef, ci_l, ci_u in zip(coefficients, ci_lower, ci_upper):
        if ci_l > sesoi_threshold:
            colors.append('darkgreen')  # Entire CI above positive SESOI
        elif ci_u < -sesoi_threshold:
            colors.append('darkred')    # Entire CI below negative SESOI
        elif abs(coef) < sesoi_threshold and ci_l > -sesoi_threshold and ci_u < sesoi_threshold:
            colors.append('gray')        # Entirely within SESOI bounds (negligible)
        else:
            colors.append('orange')      # Crosses SESOI threshold (uncertain)
    
    # Add SESOI regions
    ax.axvspan(-sesoi_threshold, sesoi_threshold, alpha=0.1, color='gray', 
              label=f'Negligible (|β| < {sesoi_threshold:.2f})')
    
    # Plot
    y_pos = np.arange(len(predictors))
    
    for i, (coef, ci_l, ci_u, color) in enumerate(zip(coefficients, ci_lower, ci_upper, colors)):
        ax.errorbar(coef, i, xerr=[[coef - ci_l], [ci_u - coef]], 
                   fmt='o', color=color, markersize=8, 
                   capsize=5, capthick=2, elinewidth=2)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=sesoi_threshold, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, 
              label=f'SESOI = ±{sesoi_threshold:.2f}')
    ax.axvline(x=-sesoi_threshold, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors)
    ax.set_xlabel('Coefficient (log scale)', fontsize=10)
    ax.set_title('SESOI Forest Plot (Practical Significance)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    legend_elements = [
        Patch(facecolor='darkgreen', label='Above +SESOI'),
        Patch(facecolor='darkred', label='Below -SESOI'),
        Patch(facecolor='gray', label='Negligible'),
        Patch(facecolor='orange', label='Uncertain')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8, title='SESOI Status')


def _plot_tost_forest(ax: plt.Axes, wald_results: Dict[str, Any], tost_bounds: Tuple[float, float]):
    """Create forest plot with TOST (Two One-Sided Tests) equivalence bounds.
    
    Shows whether effects are practically equivalent to zero.
    """
    # Extract data
    predictors = []
    coefficients = []
    std_errors = []
    
    for predictor, result in wald_results.items():
        if predictor == 'overall' or not isinstance(result, dict):
            continue
        predictors.append(predictor)
        coefficients.append(result.get('coefficient', 0))
        # Handle both 'std_error' and 'standard_error' keys
        std_err = result.get('std_error', result.get('standard_error', 0))
        std_errors.append(std_err)
    
    if not predictors:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title('TOST Forest Plot')
        return
    
    # Calculate CIs
    coefficients = np.array(coefficients)
    std_errors = np.array(std_errors)
    ci_lower = coefficients - 1.96 * std_errors
    ci_upper = coefficients + 1.96 * std_errors
    
    # Color by TOST equivalence test
    colors = []
    for coef, ci_l, ci_u in zip(coefficients, ci_lower, ci_upper):
        # Check if CI is entirely within TOST bounds
        if ci_l > tost_bounds[0] and ci_u < tost_bounds[1]:
            colors.append('green')       # Equivalent to zero
        # Check if CI is entirely outside TOST bounds
        elif ci_l > tost_bounds[1] or ci_u < tost_bounds[0]:
            colors.append('red')         # Not equivalent (meaningful effect)
        else:
            colors.append('orange')      # Inconclusive
    
    # Add TOST equivalence region
    ax.axvspan(tost_bounds[0], tost_bounds[1], alpha=0.15, color='lightgreen', 
              label=f'Equivalence region [{tost_bounds[0]:.2f}, {tost_bounds[1]:.2f}]')
    
    # Plot
    y_pos = np.arange(len(predictors))
    
    for i, (coef, ci_l, ci_u, color) in enumerate(zip(coefficients, ci_lower, ci_upper, colors)):
        ax.errorbar(coef, i, xerr=[[coef - ci_l], [ci_u - coef]], 
                   fmt='o', color=color, markersize=8, 
                   capsize=5, capthick=2, elinewidth=2)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=tost_bounds[0], color='darkgreen', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=tost_bounds[1], color='darkgreen', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors)
    ax.set_xlabel('Coefficient (log scale)', fontsize=10)
    ax.set_title('TOST Forest Plot (Equivalence Testing)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    legend_elements = [
        Patch(facecolor='green', label='Equivalent to zero'),
        Patch(facecolor='red', label='Not equivalent'),
        Patch(facecolor='orange', label='Inconclusive')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8, title='TOST Result')


def _plot_pvalues(ax: plt.Axes, wald_results: Dict[str, Any]):
    """Create p-value plot showing significance levels."""
    # Extract data
    predictors = []
    p_values = []
    wald_stats = []
    
    for predictor, result in wald_results.items():
        if predictor == 'overall' or not isinstance(result, dict):
            continue
        predictors.append(predictor)
        p_values.append(result.get('p_value', 1))
        wald_stats.append(result.get('wald_statistic', 0))
    
    if not predictors:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title('P-value Plot')
        return
    
    y_pos = np.arange(len(predictors))
    
    # Color bars by significance level
    colors = []
    for p in p_values:
        if p < 0.001:
            colors.append('darkgreen')
        elif p < 0.01:
            colors.append('green')
        elif p < 0.05:
            colors.append('orange')
        elif p < 0.1:
            colors.append('yellow')
        else:
            colors.append('lightgray')
    
    # Create horizontal bar plot
    bars = ax.barh(y_pos, p_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add significance thresholds
    ax.axvline(x=0.001, color='darkgreen', linestyle='--', alpha=0.5, label='p = 0.001')
    ax.axvline(x=0.01, color='green', linestyle='--', alpha=0.5, label='p = 0.01')
    ax.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, label='p = 0.05')
    ax.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.5, label='p = 0.10')
    
    # Add exact p-values as text
    for i, (bar, p, w) in enumerate(zip(bars, p_values, wald_stats)):
        width = bar.get_width()
        if p < 0.001:
            text = f'< 0.001 (W={w:.2f})'
        else:
            text = f'{p:.3f} (W={w:.2f})'
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
               text, ha='left', va='center', fontsize=9)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors)
    ax.set_xlabel('P-value', fontsize=10)
    ax.set_xlim(0, max(0.15, max(p_values) * 1.1))
    ax.set_title('P-values and Wald Statistics', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='best', fontsize=8)
    
    # Add interpretation note
    ax.text(0.98, 0.02, 'W = Wald statistic', transform=ax.transAxes,
           ha='right', fontsize=8, style='italic', alpha=0.7)


def _plot_wald_summary(ax: plt.Axes, wald_results: Dict[str, Any], 
                      sesoi_threshold: float, tost_bounds: Tuple[float, float]):
    """Create summary panel showing key findings from SESOI and TOST analyses."""
    ax.axis('off')
    
    # Count results by category
    n_above_sesoi = 0
    n_below_neg_sesoi = 0
    n_negligible = 0
    n_equivalent = 0
    n_not_equivalent = 0
    n_significant = 0
    
    for predictor, result in wald_results.items():
        if predictor == 'overall' or not isinstance(result, dict):
            continue
            
        coef = result.get('coefficient', 0)
        se = result.get('std_error', 0)
        p_val = result.get('p_value', 1)
        
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        
        # SESOI classification
        if ci_lower > sesoi_threshold:
            n_above_sesoi += 1
        elif ci_upper < -sesoi_threshold:
            n_below_neg_sesoi += 1
        elif abs(coef) < sesoi_threshold and ci_lower > -sesoi_threshold and ci_upper < sesoi_threshold:
            n_negligible += 1
            
        # TOST classification
        if ci_lower > tost_bounds[0] and ci_upper < tost_bounds[1]:
            n_equivalent += 1
        elif ci_lower > tost_bounds[1] or ci_upper < tost_bounds[0]:
            n_not_equivalent += 1
            
        # Statistical significance
        if p_val < 0.05:
            n_significant += 1
    
    # Create summary text
    summary_text = """Wald Test Summary
═══════════════════════════

SESOI Analysis (±{:.2f}):
  • Above threshold: {}
  • Below threshold: {}
  • Negligible: {}
  
TOST Equivalence [{:.2f}, {:.2f}]:
  • Equivalent to zero: {}
  • Not equivalent: {}
  
Statistical Significance:
  • Significant (p<0.05): {}/{}
  
Interpretation:
{}""".format(
        sesoi_threshold, n_above_sesoi, n_below_neg_sesoi, n_negligible,
        tost_bounds[0], tost_bounds[1], n_equivalent, n_not_equivalent,
        n_significant, len([k for k in wald_results if k != 'overall']),
        _get_interpretation(n_above_sesoi, n_negligible, n_equivalent, n_significant)
    )
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, va='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    ax.set_title('Summary', fontsize=11, fontweight='bold')


def _get_interpretation(n_above_sesoi, n_negligible, n_equivalent, n_significant):
    """Generate interpretation based on results."""
    if n_negligible > n_above_sesoi:
        return "Most effects are negligible\n(below practical significance)"
    elif n_equivalent > 0:
        return "Some effects are equivalent\nto zero (within bounds)"
    elif n_significant > 0:
        return "Statistically significant\neffects detected"
    else:
        return "No significant effects"


# Keep the old function name for backward compatibility but mark as deprecated
def create_wald_forest_plot_with_sesoi(*args, **kwargs):
    """DEPRECATED: Use create_wald_visualization() instead.
    
    This function is kept for backward compatibility only.
    """
    logger.warning("create_wald_forest_plot_with_sesoi is deprecated. Use create_wald_visualization instead.")
    return create_wald_visualization(*args, **kwargs)