"""Forest plot visualizations for Wald test results.

Provides comprehensive visualization for Wald tests including:
- Basic forest plot (coefficients with CIs)
- SESOI (Smallest Effect Size of Interest) forest plot
- TOST (Two One-Sided Tests) equivalence forest plot

Data processing and visualization in one file following Linus principles.
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


class ForestDataProcessor:
    """Process and prepare data for forest plots.
    
    Separates data manipulation from visualization logic.
    Each method does one thing and does it well.
    """
    
    @staticmethod
    def calculate_importance(wald_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate importance scores (|z-score|) for all predictors."""
        importance = {}
        for predictor, result in wald_results.items():
            if predictor == 'overall' or not isinstance(result, dict):
                continue
                
            coef = result.get('coefficient', 0)
            se = result.get('standard_error', 1)
            
            if se > 0:
                importance[predictor] = abs(coef / se)
            else:
                importance[predictor] = 0
                
        return importance
    
    @staticmethod
    def sort_by_importance(wald_results: Dict[str, Any]) -> List[str]:
        """Sort predictors by importance (|z-score|)."""
        importance = ForestDataProcessor.calculate_importance(wald_results)
        return sorted(importance.keys(), 
                     key=lambda x: importance[x], 
                     reverse=True)
    
    @staticmethod
    def filter_predictors(predictors: List[str], 
                         filter_mesf: bool = True,
                         filter_intercept: bool = True) -> List[str]:
        """Filter out unwanted predictors."""
        filtered = predictors.copy()
        
        if filter_mesf:
            filtered = [p for p in filtered 
                       if not p.lower().startswith('mesf')]
        
        if filter_intercept:
            filtered = [p for p in filtered 
                       if p.lower() not in ['(intercept)', 'intercept']]
        
        return filtered
    
    @staticmethod
    def clean_predictor_names(names: List[str]) -> List[str]:
        """Clean predictor names for display."""
        cleaned = []
        for name in names:
            clean = (name.replace('_log', ' (log)')
                        .replace('_std', ' (std)')
                        .replace('_', ' ')
                        .replace('bio', 'Bio')
                        .replace('plants', 'Plants')
                        .replace('vertebrates', 'Vertebrates')
                        .replace('richness', 'Richness')
                        .title())
            cleaned.append(clean)
        return cleaned
    
    @staticmethod
    def extract_ordered_data(wald_results: Dict[str, Any], 
                            predictor_order: List[str]) -> Tuple[np.ndarray, np.ndarray, List]:
        """Extract coefficients and errors in specified order."""
        coefficients = []
        std_errors = []
        sesoi_results = []
        
        for predictor in predictor_order:
            if predictor not in wald_results:
                continue
                
            result = wald_results[predictor]
            coefficients.append(result.get('coefficient', 0))
            std_errors.append(result.get('standard_error', 0))
            
            # Extract SESOI test result
            sesoi_test = result.get('sesoi_test', {})
            sesoi_results.append(sesoi_test.get('exceeds_threshold', False))
        
        return (np.array(coefficients), 
                np.array(std_errors), 
                sesoi_results)
    
    @staticmethod
    def prepare_smart_forest_data(wald_results: Dict[str, Any],
                                 filter_mesf: bool = True,
                                 top_n: Optional[int] = 15) -> Dict[str, Any]:
        """Prepare data for smart forest plot."""
        # Get sorted predictor order
        sorted_predictors = ForestDataProcessor.sort_by_importance(wald_results)
        
        # Filter if requested
        if filter_mesf:
            sorted_predictors = ForestDataProcessor.filter_predictors(
                sorted_predictors, filter_mesf=True)
        
        # Limit to top N
        if top_n and len(sorted_predictors) > top_n:
            sorted_predictors = sorted_predictors[:top_n]
        
        # Extract data in order
        coefficients, std_errors, sesoi_results = \
            ForestDataProcessor.extract_ordered_data(wald_results, sorted_predictors)
        
        # Clean names
        clean_names = ForestDataProcessor.clean_predictor_names(sorted_predictors)
        
        return {
            'predictors': sorted_predictors,
            'clean_names': clean_names,
            'coefficients': coefficients,
            'std_errors': std_errors,
            'sesoi_results': sesoi_results,
            'importance': ForestDataProcessor.calculate_importance(wald_results)
        }
    
    @staticmethod
    def extract_basic_data(wald_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic data from raw wald_results (no sorting/filtering)."""
        predictors = []
        coefficients = []
        std_errors = []
        sesoi_results = []
        tost_results = []  # Add TOST results extraction
        p_values = []
        wald_stats = []
        
        for predictor, result in wald_results.items():
            if predictor == 'overall' or not isinstance(result, dict):
                continue
            predictors.append(predictor)
            coefficients.append(result.get('coefficient', 0))
            std_errors.append(result.get('standard_error', 0))
            p_values.append(result.get('p_value', 1))
            wald_stats.append(result.get('wald_statistic', 0))
            
            # Extract SESOI test result if available
            sesoi_test = result.get('sesoi_test', {})
            sesoi_results.append(sesoi_test.get('exceeds_threshold', False))
            
            # Extract TOST test result if available
            tost_test = result.get('tost_test', {})
            tost_results.append({
                'is_equivalent': tost_test.get('is_equivalent', False),
                'is_different': tost_test.get('is_different', False),
                'conclusion': tost_test.get('conclusion', 'Inconclusive')
            })
        
        return {
            'predictors': predictors,
            'coefficients': np.array(coefficients),
            'std_errors': np.array(std_errors),
            'sesoi_results': sesoi_results,
            'tost_results': tost_results,  # Include TOST results
            'p_values': np.array(p_values),
            'wald_stats': np.array(wald_stats)
        }


def create_wald_visualization(
    wald_results: Dict[str, Any],
    sesoi_threshold: float = 0.1,  # Default SESOI threshold (10% change on response scale)
    tost_bounds: Tuple[float, float] = (-0.05, 0.05),  # Default TOST bounds (±5%)
    title: str = "Wald Test Analysis",
    figsize: Tuple[int, int] = (10, 8),  # Size for individual plots
    filter_mesf: bool = True  # Whether to filter MESF terms
) -> Tuple[plt.Figure, plt.Figure]:
    """Create TWO SEPARATE Wald test visualizations.
    
    Creates separate figures for SESOI and TOST tests instead of combining them.
    Always sorts by importance (|z-score|) and cleans names.
    
    Args:
        wald_results: Results from Wald tests containing coefficients, std_errors
        sesoi_threshold: Threshold for smallest effect size of interest
        tost_bounds: Equivalence bounds for TOST
        title: Main title for the figures
        figsize: Figure size for each individual plot
        filter_mesf: Whether to filter out MESF spatial terms
        
    Returns:
        Tuple of (sesoi_figure, tost_figure)
    """
    # Create SESOI figure
    fig_sesoi = plt.figure(figsize=figsize)
    ax_sesoi = fig_sesoi.add_subplot(111)
    _plot_smart_forest(ax_sesoi, wald_results, sesoi_threshold, 
                      filter_mesf=filter_mesf, title="SESOI (Smallest Effect Size of Interest) Test")
    fig_sesoi.suptitle(f"{title} - SESOI Analysis", fontsize=14, fontweight='bold')
    plt.figure(fig_sesoi.number)
    plt.tight_layout()
    
    # Create TOST figure
    fig_tost = plt.figure(figsize=figsize)
    ax_tost = fig_tost.add_subplot(111)
    _plot_tost_forest(ax_tost, wald_results, tost_bounds)
    fig_tost.suptitle(f"{title} - TOST (Two One-Sided Tests) Analysis", fontsize=14, fontweight='bold')
    plt.figure(fig_tost.number)
    plt.tight_layout()
    
    return fig_sesoi, fig_tost


def create_method2_comparison(
    continuous_results: Dict[str, Any],
    categorical_results: Dict[str, Any],
    sesoi_threshold: float = 0.1,
    figsize: Tuple[int, int] = (14, 6),
    highlight_interaction: bool = True
) -> plt.Figure:
    """Method 2 specific: Compare continuous vs categorical interaction models.
    
    Simple side-by-side comparison without hardcoded predictor lists.
    
    Args:
        continuous_results: Wald results from Model 3a (continuous interaction)
        categorical_results: Wald results from Model 3b (categorical interaction)
        sesoi_threshold: SESOI threshold for significance
        figsize: Figure size
        highlight_interaction: Whether to highlight interaction terms
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Process data using existing processor
    cont_data = ForestDataProcessor.extract_basic_data(continuous_results)
    cat_data = ForestDataProcessor.extract_basic_data(categorical_results)
    
    # Filter MESF variables manually
    if cont_data['predictors']:
        mesf_indices = [i for i, p in enumerate(cont_data['predictors']) 
                       if not p.lower().startswith('mesf')]
        cont_data = {
            'predictors': [cont_data['predictors'][i] for i in mesf_indices],
            'coefficients': [cont_data['coefficients'][i] for i in mesf_indices],
            'std_errors': [cont_data['std_errors'][i] for i in mesf_indices],
            'sesoi_results': [cont_data['sesoi_results'][i] for i in mesf_indices]
        }
    
    if cat_data['predictors']:
        mesf_indices = [i for i, p in enumerate(cat_data['predictors']) 
                       if not p.lower().startswith('mesf')]
        cat_data = {
            'predictors': [cat_data['predictors'][i] for i in mesf_indices],
            'coefficients': [cat_data['coefficients'][i] for i in mesf_indices],
            'std_errors': [cat_data['std_errors'][i] for i in mesf_indices],
            'sesoi_results': [cat_data['sesoi_results'][i] for i in mesf_indices]
        }
    
    # Plot using existing SESOI forest function  
    _plot_sesoi_forest_base(ax1,
                           cont_data['predictors'],
                           cont_data['coefficients'], 
                           cont_data['std_errors'],
                           cont_data['sesoi_results'],
                           sesoi_threshold)
    
    _plot_sesoi_forest_base(ax2,
                           cat_data['predictors'],
                           cat_data['coefficients'],
                           cat_data['std_errors'],
                           cat_data['sesoi_results'],
                           sesoi_threshold)
    
    # Titles that describe the models
    ax1.set_title("Model 3a: Continuous Interaction", fontweight='bold')
    ax2.set_title("Model 3b: Categorical Interaction", fontweight='bold')
    
    # Highlight interaction terms if requested
    if highlight_interaction:
        # Find and highlight interaction terms
        for ax, data in [(ax1, cont_data), (ax2, cat_data)]:
            labels = ax.get_yticklabels()
            for i, (label, pred) in enumerate(zip(labels, data['predictors'])):
                # Interaction terms contain '_x_' or specific patterns
                if '_x_' in pred or 'temperate' in pred.lower():
                    label.set_weight('bold')
                    label.set_color('darkgoldenrod')
    
    # Main title
    fig.suptitle('Method 2: Climate-Biodiversity Interaction Analysis', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def _plot_smart_forest(ax: plt.Axes, wald_results: Dict[str, Any], 
                      sesoi_threshold: float, filter_mesf: bool = True,
                      title: str = "Forest Plot", top_n: Optional[int] = 15):
    """Smart forest plot with sorting, filtering, and name cleaning.
    
    Can handle both filtered (smart mode) and unfiltered (comprehensive/compact) data.
    Always sorts by importance and cleans names.
    """
    
    # Step 1: Process data - always sort and clean names
    processed_data = ForestDataProcessor.prepare_smart_forest_data(
        wald_results, filter_mesf=filter_mesf, top_n=top_n)
    
    if not processed_data['predictors']:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title(title)
        return
    
    # Step 2: Delegate to base function (no data processing here!)
    _plot_sesoi_forest_base(ax, 
                           processed_data['clean_names'],
                           processed_data['coefficients'],
                           processed_data['std_errors'],
                           processed_data['sesoi_results'],
                           sesoi_threshold)
    
    # Step 3: Add smart-specific features only
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add methodology note (removed |Z| annotations per user request)
    ax.text(0.98, 0.02, 'Sorted by |z-score| • 90% CI shown', transform=ax.transAxes,
           ha='right', fontsize=8, style='italic', alpha=0.7)


def _plot_sesoi_forest_base(ax: plt.Axes, 
                           predictors: List[str],
                           coefficients: np.ndarray,
                           std_errors: np.ndarray,
                           sesoi_results: List[bool],
                           sesoi_threshold: float):
    """Base function for SESOI forest plot (internal use only).
    
    Pure plotting function with 4-layer SESOI judgment logic.
    No data processing - only visualization.
    
    Args:
        ax: Matplotlib axes
        predictors: List of predictor names (for display)
        coefficients: Array of coefficient values
        std_errors: Array of standard errors
        sesoi_results: List of SESOI test results (exceeds_threshold)
        sesoi_threshold: SESOI threshold value
    """
    if not predictors:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title('SESOI Forest Plot')
        return
    
    # Calculate 90% CIs for SESOI testing (single-sided α=0.05)
    coefficients = np.array(coefficients)
    std_errors = np.array(std_errors)
    ci_lower_90 = coefficients - 1.645 * std_errors  # 90% CI
    ci_upper_90 = coefficients + 1.645 * std_errors
    
    # Color based on SESOI test results (two-tailed test)
    colors = []
    markers = []
    for i, (coef, exceeds_sesoi) in enumerate(zip(coefficients, sesoi_results)):
        if exceeds_sesoi:
            # Statistically exceeds SESOI threshold (|β| > threshold with p < 0.05)
            colors.append('darkgreen' if coef > 0 else 'darkred')
            markers.append('o')  # Filled circle
        elif abs(coef) > sesoi_threshold:
            # Magnitude exceeds threshold but not statistically significant
            colors.append('lightgreen' if coef > 0 else 'lightcoral')
            markers.append('^')  # Triangle
        elif abs(coef) < sesoi_threshold * 0.5:
            # Clearly negligible (less than half SESOI)
            colors.append('gray')
            markers.append('s')  # Square
        else:
            # Borderline (between 0.5*SESOI and SESOI)
            colors.append('orange')
            markers.append('d')  # Diamond
    
    # Add SESOI regions
    ax.axvspan(-sesoi_threshold, sesoi_threshold, alpha=0.1, color='gray', 
              label=f'Negligible (|β| < {sesoi_threshold:.2f})')
    
    # Plot with 90% CI
    y_pos = np.arange(len(predictors))
    
    for i, (coef, ci_l, ci_u, color, marker) in enumerate(zip(coefficients, ci_lower_90, ci_upper_90, colors, markers)):
        # Plot error bar with enhanced |-| style caps for better visibility
        ax.errorbar(coef, i, xerr=[[coef - ci_l], [ci_u - coef]], 
                   fmt=marker, color=color, markersize=10, 
                   capsize=8, capthick=2.5, elinewidth=2.5, 
                   markeredgecolor='black', markeredgewidth=0.5,
                   alpha=0.9)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1, label='Zero')
    ax.axvline(x=sesoi_threshold, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
              label=f'SESOI = {sesoi_threshold:.2f}')
    ax.axvline(x=-sesoi_threshold, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors)
    ax.set_xlabel('Coefficient Value', fontsize=10)
    ax.set_title('SESOI Two-Tailed Test: |β| > threshold', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend with proper SESOI test interpretation
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', 
               markersize=8, label='Positive meaningful (β > SESOI, p<0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
               markersize=8, label='Negative meaningful (β < -SESOI, p<0.05)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lightgreen', 
               markersize=8, label='Exceeds threshold but NS'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=8, label='Negligible (|β| < 50% SESOI)'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='orange', 
               markersize=8, label='Borderline (50-100% SESOI)'),
        Line2D([0], [0], color='red', linestyle='--', label=f'SESOI Threshold (±{sesoi_threshold:.2f})')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=7, title='Two-tailed SESOI Test')
    
    # Add note about CI level and test nature
    ax.text(0.02, 0.98, 'Two-tailed test: H₀: |β| ≤ SESOI vs H₁: |β| > SESOI', 
           transform=ax.transAxes, fontsize=8, style='italic', 
           verticalalignment='top', alpha=0.7)




def _plot_tost_forest(ax: plt.Axes, wald_results: Dict[str, Any], tost_bounds: Tuple[float, float]):
    """Create forest plot with TOST (Two One-Sided Tests) equivalence bounds.
    
    Shows whether effects are practically equivalent to zero based on statistical tests.
    Uses actual TOST test results, not visual CI comparison.
    """
    # Use data processor to get all data including TOST results
    data = ForestDataProcessor.extract_basic_data(wald_results)
    
    predictors = data['predictors']
    coefficients = data['coefficients'] 
    std_errors = data['std_errors']
    tost_results = data['tost_results']  # Get TOST test results
    
    if not predictors:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title('TOST Forest Plot')
        return
    
    # Calculate 90% CIs for TOST (correct for equivalence testing)
    coefficients = np.array(coefficients)
    std_errors = np.array(std_errors)
    ci_lower_90 = coefficients - 1.645 * std_errors  # 90% CI for TOST
    ci_upper_90 = coefficients + 1.645 * std_errors
    
    # Color based on actual TOST statistical test results
    colors = []
    markers = []
    for tost_result in tost_results:
        is_equivalent = tost_result['is_equivalent']
        is_different = tost_result['is_different']
        conclusion = tost_result['conclusion']
        
        if is_equivalent and is_different:
            # Different but equivalent (negligible effect)
            colors.append('lightgreen')
            markers.append('s')  # Square
        elif is_equivalent and not is_different:
            # Equivalent to zero
            colors.append('green')
            markers.append('o')  # Circle
        elif not is_equivalent and is_different:
            # Significant and meaningful
            colors.append('red')
            markers.append('d')  # Diamond
        else:
            # Inconclusive
            colors.append('orange')
            markers.append('^')  # Triangle
    
    # Add TOST equivalence region
    ax.axvspan(tost_bounds[0], tost_bounds[1], alpha=0.15, color='lightgreen', 
              label=f'Equivalence region [{tost_bounds[0]:.2f}, {tost_bounds[1]:.2f}]')
    
    # Plot with 90% CI and different markers based on TOST results
    y_pos = np.arange(len(predictors))
    
    for i, (coef, ci_l, ci_u, color, marker) in enumerate(zip(coefficients, ci_lower_90, ci_upper_90, colors, markers)):
        # Plot error bar with enhanced |-| style caps for TOST
        ax.errorbar(coef, i, xerr=[[coef - ci_l], [ci_u - coef]], 
                   fmt=marker, color=color, markersize=10, 
                   capsize=8, capthick=2.5, elinewidth=2.5,
                   markeredgecolor='black', markeredgewidth=0.5,
                   alpha=0.9)
    
    # Reference lines
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=tost_bounds[0], color='darkgreen', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=tost_bounds[1], color='darkgreen', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors)
    ax.set_xlabel('Coefficient Value', fontsize=10)
    ax.set_title('TOST Forest Plot (Equivalence Testing)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend with TOST statistical test interpretations
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=8, label='Equivalent to zero'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', 
               markersize=8, label='Different but negligible'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='red', 
               markersize=8, label='Significant and meaningful'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', 
               markersize=8, label='Inconclusive')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8, title='TOST Statistical Test')
    
    # Add note about CI level
    ax.text(0.02, 0.98, '90% CI shown (for TOST equivalence testing)', 
           transform=ax.transAxes, fontsize=8, style='italic', 
           verticalalignment='top', alpha=0.7)


def _plot_pvalues(ax: plt.Axes, wald_results: Dict[str, Any]):
    """Create p-value plot showing significance levels.
    
    NOTE: This function is kept for backward compatibility but is no longer
    used in main visualization pipelines as p-values plots have been excluded.
    """
    # Use data processor - no more manual extraction!
    data = ForestDataProcessor.extract_basic_data(wald_results)
    
    predictors = data['predictors']
    p_values = data['p_values']
    wald_stats = data['wald_stats']
    
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




# Keep the old function name for backward compatibility but mark as deprecated
def create_wald_forest_plot_with_sesoi(*args, **kwargs):
    """DEPRECATED: Use create_wald_visualization() instead.
    
    This function is kept for backward compatibility only.
    """
    logger.warning("create_wald_forest_plot_with_sesoi is deprecated. Use create_wald_visualization instead.")
    return create_wald_visualization(*args, **kwargs)