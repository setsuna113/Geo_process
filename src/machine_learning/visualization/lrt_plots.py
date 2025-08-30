from src.abstractions import get_logger
"""Likelihood Ratio Test visualization for Julia's lrtest() output.

Publication-standard visualizations for LRT results including:
- Analysis of Deviance tables
- Deviance-df plots for model comparison
- Chi-square distribution plots with test statistics
- Model comparison bar charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

logger = get_logger(__name__)

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def visualize_lrt_results(
    lrt_result: Dict[str, Any],
    model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create publication-standard visualization of LRT results from Julia's lrtest().
    
    Creates a comprehensive 2-panel figure showing:
    1. Analysis of Deviance table (visual)
    2. Chi-square distribution with test statistic
    
    Args:
        lrt_result: Dictionary from Julia's lrtest() containing:
            - lr_statistic: LR test statistic (deviance difference)
            - df: Degrees of freedom difference
            - p_value: P-value from chi-squared test
            - null_deviance: Deviance of null model (optional)
            - full_deviance: Deviance of full model (optional)
        model_names: Names for the models being compared
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    if model_names is None:
        model_names = ['Null Model', 'Full Model']
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    
    # Extract statistics
    lr_stat = lrt_result.get('lr_statistic', 0)
    df = lrt_result.get('df', 1)
    p_value = lrt_result.get('p_value', 1)
    
    # Panel 1: Analysis of Deviance Table
    ax1 = fig.add_subplot(gs[0])
    _plot_deviance_table(ax1, lrt_result, model_names)
    
    # Panel 2: Chi-square Distribution with Test Statistic
    ax2 = fig.add_subplot(gs[1])
    _plot_chi_square_test(ax2, lr_stat, df, p_value)
    
    # Main title
    significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
    color = 'green' if p_value < 0.05 else 'red'
    
    fig.suptitle(
        f'Likelihood Ratio Test Results: {significance} (p = {p_value:.4f})',
        fontsize=16, fontweight='bold', color=color
    )
    
    try:
        plt.tight_layout()
    except:
        # Fallback if tight_layout fails
        pass
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"LRT visualization saved to {save_path}")
        except Exception as e:
            # Try saving without bbox_inches='tight'
            plt.savefig(save_path, dpi=300)
            logger.info(f"LRT visualization saved to {save_path} (without tight bbox)")
    
    return fig


def _plot_deviance_table(ax: plt.Axes, lrt_result: Dict, model_names: List[str]):
    """Create visual representation of Analysis of Deviance table."""
    
    # Create table data
    null_dev = lrt_result.get('null_deviance', lrt_result.get('lr_statistic', 0) + 100)
    full_dev = lrt_result.get('full_deviance', 100)
    lr_stat = lrt_result.get('lr_statistic', null_dev - full_dev)
    df = lrt_result.get('df', 1)
    p_value = lrt_result.get('p_value', 0.05)
    
    # Format p-value
    if p_value < 0.001:
        p_str = "< 0.001***"
    elif p_value < 0.01:
        p_str = f"{p_value:.3f}**"
    elif p_value < 0.05:
        p_str = f"{p_value:.3f}*"
    else:
        p_str = f"{p_value:.3f}"
    
    # Create table
    table_data = [
        [model_names[0], f"{null_dev:.2f}", "-", "-", "-"],
        [model_names[1], f"{full_dev:.2f}", f"{df}", f"{lr_stat:.2f}", p_str]
    ]
    
    columns = ['Model', 'Deviance', 'Δdf', 'χ² Statistic', 'p-value']
    
    # Plot table
    table = ax.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.15, 0.2, 0.2])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code by significance
    if p_value < 0.05:
        table[(2, 4)].set_facecolor('#90EE90')  # Light green for significant
    else:
        table[(2, 4)].set_facecolor('#FFB6C1')  # Light red for non-significant
    
    # Header styling
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.axis('off')
    ax.set_title('Analysis of Deviance Table', fontsize=12, fontweight='bold')
    
    # Add legend for significance
    ax.text(0.5, -0.1, 'Significance: *** p<0.001, ** p<0.01, * p<0.05',
            ha='center', transform=ax.transAxes, fontsize=9, style='italic')



def _plot_chi_square_test(ax: plt.Axes, lr_stat: float, df: int, p_value: float):
    """Plot chi-square distribution with test statistic."""
    
    # Generate chi-square distribution
    x_max = max(lr_stat * 1.5, stats.chi2.ppf(0.999, df))
    x = np.linspace(0, x_max, 1000)
    y = stats.chi2.pdf(x, df)
    
    # Plot distribution
    ax.plot(x, y, 'b-', linewidth=2, label=f'χ²({df}) distribution')
    
    # Fill critical region
    x_crit = stats.chi2.ppf(0.95, df)
    x_fill = x[x >= x_crit]
    y_fill = stats.chi2.pdf(x_fill, df)
    ax.fill_between(x_fill, 0, y_fill, alpha=0.3, color='red', 
                    label='Critical region (α=0.05)')
    
    # Mark test statistic
    ax.axvline(lr_stat, color='green', linewidth=3, linestyle='--',
              label=f'LR statistic = {lr_stat:.2f}')
    
    # Add p-value annotation
    y_stat = stats.chi2.pdf(lr_stat, df)
    ax.plot(lr_stat, y_stat, 'go', markersize=10)
    
    # Annotation with arrow
    if lr_stat > x_crit:
        annotation_text = f'p = {p_value:.4f}\nREJECT H₀'
        color = 'green'
    else:
        annotation_text = f'p = {p_value:.4f}\nFAIL TO REJECT H₀'
        color = 'red'
    
    # Safe annotation positioning - use text without arrow to avoid matplotlib bug
    ax.text(0.7, 0.8, annotation_text, 
            transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.9),
            ha='center', va='center')
    
    # Styling
    ax.set_xlabel('Test Statistic', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Chi-Square Test for LR Statistic', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add critical value line
    ax.axvline(x_crit, color='red', linestyle=':', alpha=0.5)
    ax.text(x_crit, max(y) * 0.1, f'Critical value\n{x_crit:.2f}',
           ha='center', fontsize=9, color='red')



def create_lrt_comparison_matrix(
    lrt_results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create a matrix visualization comparing multiple LRT results.
    
    Useful when comparing multiple nested model pairs.
    
    Args:
        lrt_results: Dictionary of LRT results, keys are comparison names
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    n_comparisons = len(lrt_results)
    
    # Prepare data for heatmap
    comparison_names = list(lrt_results.keys())
    metrics = ['LR Statistic', 'df', 'p-value', 'Significant']
    
    data = []
    for name in comparison_names:
        result = lrt_results[name]
        row = [
            result.get('lr_statistic', 0),
            result.get('df', 1),
            result.get('p_value', 1),
            1 if result.get('p_value', 1) < 0.05 else 0
        ]
        data.append(row)
    
    df = pd.DataFrame(data, index=comparison_names, columns=metrics)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                   gridspec_kw={'width_ratios': [3, 1]})
    
    # Heatmap of statistics
    sns.heatmap(df.iloc[:, :3], annot=True, fmt='.3f', cmap='RdYlGn_r',
               center=0, cbar_kws={'label': 'Value'}, ax=ax1,
               linewidths=1, linecolor='gray')
    ax1.set_title('LRT Statistics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Model Comparisons', fontsize=11)
    
    # Significance plot
    sig_data = df['Significant'].values.reshape(-1, 1)
    sns.heatmap(sig_data, annot=False, cmap=['red', 'green'],
               cbar=False, ax=ax2, linewidths=1, linecolor='gray',
               vmin=0, vmax=1)
    ax2.set_title('Significant?', fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticklabels([])
    
    # Add custom colorbar for significance
    for i, (name, sig) in enumerate(zip(comparison_names, sig_data.flatten())):
        color = 'green' if sig == 1 else 'red'
        text = '✓' if sig == 1 else '✗'
        ax2.text(0.5, i + 0.5, text, ha='center', va='center',
                fontsize=20, color=color, fontweight='bold')
    
    plt.suptitle('Likelihood Ratio Test Comparison Matrix', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"LRT comparison matrix saved to {save_path}")
    
    return fig


def create_analysis_of_deviance_report(
    julia_lrt_output: Union[Dict, pd.DataFrame],
    model_descriptions: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None
) -> str:
    """Generate publication-ready Analysis of Deviance table from Julia lrtest output.
    
    Args:
        julia_lrt_output: Either a dict or DataFrame from Julia's lrtest()
        model_descriptions: Optional descriptions for each model
        save_path: Optional path to save LaTeX table
        
    Returns:
        LaTeX formatted table string
    """
    # Convert to DataFrame if needed
    if isinstance(julia_lrt_output, dict):
        df = pd.DataFrame([julia_lrt_output])
    else:
        df = julia_lrt_output
    
    # Format p-values with significance stars
    def format_pval(p):
        if pd.isna(p):
            return '-'
        elif p < 0.001:
            return '< 0.001***'
        elif p < 0.01:
            return f'{p:.3f}**'
        elif p < 0.05:
            return f'{p:.3f}*'
        else:
            return f'{p:.3f}'
    
    # Create formatted table
    if 'p_value' in df.columns:
        df['p-value'] = df['p_value'].apply(format_pval)
    
    # Generate LaTeX
    latex_table = df.to_latex(
        index=False,
        caption='Analysis of Deviance Table',
        label='tab:anova_deviance',
        column_format='lrrrl',
        escape=False
    )
    
    # Add note about significance
    latex_table = latex_table.replace(
        '\\end{tabular}',
        '\\end{tabular}\n\\footnotesize{Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05}'
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"LaTeX table saved to {save_path}")
    
    return latex_table