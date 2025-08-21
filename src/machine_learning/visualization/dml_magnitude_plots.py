"""DML Magnitude visualization for effect size classification and comparison.

Publication-standard visualizations for DML magnitude analysis including:
- Forest plot with magnitude zones (Cohen's d style)
- Magnitude classification bar chart
- Comparative magnitude plot for multiple effects
- Effect size distribution with overlapping visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def visualize_dml_magnitude(
    magnitude_results: Union[Dict[str, Any], List[Dict[str, Any]]],
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive DML magnitude visualization.
    
    Creates a 4-panel figure showing:
    1. Forest plot with magnitude zones
    2. Magnitude classification chart
    3. Precision vs magnitude plot
    4. Comparative effects visualization
    
    Args:
        magnitude_results: Results from DMLMagnitude.classify_magnitude() or compare_magnitudes()
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Convert single result to list for consistency
    if isinstance(magnitude_results, dict):
        if 'effect_size' in magnitude_results:
            # Single effect result
            results_list = [magnitude_results]
        elif 'rankings' in magnitude_results:
            # This is output from compare_magnitudes or from_julia_results
            results_list = magnitude_results['rankings']
        elif 'individual_magnitudes' in magnitude_results:
            # This is full output from from_julia_results - use rankings if available
            if 'comparison' in magnitude_results and 'rankings' in magnitude_results['comparison']:
                results_list = magnitude_results['comparison']['rankings']
            else:
                # Convert individual_magnitudes to list format
                results_list = []
                for name, data in magnitude_results['individual_magnitudes'].items():
                    result = {'name': name}
                    result.update(data)
                    results_list.append(result)
        else:
            # Unknown dict format, try to use as is
            results_list = [magnitude_results]
    elif isinstance(magnitude_results, list):
        results_list = magnitude_results
    else:
        # Unknown format, wrap in list
        results_list = [magnitude_results] if magnitude_results else []
    
    # Panel 1: Forest plot with magnitude zones
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_magnitude_forest(ax1, results_list)
    
    # Panel 2: Magnitude classification bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_magnitude_classification(ax2, results_list)
    
    # Panel 3: Precision vs Magnitude scatter
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_precision_magnitude(ax3, results_list)
    
    # Panel 4: Effect size distribution/comparison
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_effect_distribution(ax4, results_list)
    
    # Main title
    fig.suptitle('DML Magnitude Analysis: Effect Size Classification',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"DML magnitude visualization saved to {save_path}")
    
    return fig


def _plot_magnitude_forest(ax: plt.Axes, results_list: List[Dict[str, Any]]):
    """Create forest plot with Cohen's d-style magnitude zones."""
    
    # Extract data
    names = []
    effects = []
    ci_lowers = []
    ci_uppers = []
    magnitudes = []
    
    for i, result in enumerate(results_list):
        if 'name' in result:
            names.append(result['name'])
        else:
            names.append(f'Effect {i+1}')
        
        effects.append(result.get('effect_size', result.get('magnitude', 0)))
        ci_lowers.append(result.get('ci_lower', effects[-1] - 0.1))
        ci_uppers.append(result.get('ci_upper', effects[-1] + 0.1))
        magnitudes.append(result.get('magnitude_class', 'unknown'))
    
    if not effects:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title('Forest Plot with Magnitude Zones')
        return
    
    # Define magnitude zones (on log scale, adapted for ecology)
    zones = {
        'negligible': (-0.01, 0.01, 'lightgray'),
        'small': (-0.05, 0.05, 'lightyellow'),
        'medium': (-0.15, 0.15, 'lightblue'),
        'large': (-0.30, 0.30, 'lightgreen'),
        'very_large': (-np.inf, np.inf, 'lightcoral')
    }
    
    # Add magnitude zones as background
    for zone_name, (lower, upper, color) in zones.items():
        if zone_name != 'very_large':
            ax.axvspan(lower, upper, alpha=0.2, color=color, label=zone_name.replace('_', ' ').title())
    
    # Add reference lines for key thresholds
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.axvline(x=0.01, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=-0.01, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=0.05, color='orange', linestyle=':', alpha=0.3)
    ax.axvline(x=-0.05, color='orange', linestyle=':', alpha=0.3)
    ax.axvline(x=0.15, color='blue', linestyle=':', alpha=0.3)
    ax.axvline(x=-0.15, color='blue', linestyle=':', alpha=0.3)
    
    # Plot effects with CIs
    y_pos = np.arange(len(names))
    
    # Color by magnitude class
    colors = []
    for mag in magnitudes:
        if mag == 'negligible':
            colors.append('gray')
        elif mag == 'small':
            colors.append('gold')
        elif mag == 'medium':
            colors.append('blue')
        elif mag == 'large':
            colors.append('green')
        elif mag == 'very large' or mag == 'very_large':
            colors.append('red')
        else:
            colors.append('black')
    
    for i, (effect, ci_l, ci_u, color) in enumerate(zip(effects, ci_lowers, ci_uppers, colors)):
        ax.errorbar(effect, i, xerr=[[effect - ci_l], [ci_u - effect]], 
                   fmt='o', color=color, markersize=10, 
                   capsize=5, capthick=2, elinewidth=2)
        
        # Add effect size value
        ax.text(effect, i + 0.15, f'{effect:.3f}', 
               ha='center', fontsize=8, fontweight='bold')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Effect Size (log scale)', fontsize=11)
    ax.set_title('Forest Plot with Cohen\'s d-style Magnitude Zones', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    # Add interpretation guide
    ax.text(0.02, 0.98, 'Magnitude zones:\n< 1% = Negligible\n1-5% = Small\n5-15% = Medium\n15-30% = Large\n> 30% = Very Large',
           transform=ax.transAxes, fontsize=8, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _plot_magnitude_classification(ax: plt.Axes, results_list: List[Dict[str, Any]]):
    """Create bar chart showing magnitude classifications."""
    
    # Count magnitude classes
    class_counts = {
        'negligible': 0,
        'small': 0,
        'medium': 0,
        'large': 0,
        'very_large': 0
    }
    
    for result in results_list:
        mag_class = result.get('magnitude_class', 'unknown')
        if mag_class in class_counts:
            class_counts[mag_class] += 1
        elif mag_class == 'very large':
            class_counts['very_large'] += 1
    
    # Create bar chart
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['gray', 'gold', 'blue', 'green', 'red']
    
    bars = ax.bar(range(len(classes)), counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   str(count), ha='center', fontweight='bold')
    
    # Add percentage labels
    total = sum(counts)
    if total > 0:
        for i, (bar, count) in enumerate(zip(bars, counts)):
            pct = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                   f'{pct:.0f}%', ha='center', color='white', fontweight='bold')
    
    # Styling
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace('_', ' ').title() for c in classes], rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Effect Magnitude Classification Distribution', 
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.2 if counts else 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation
    dominant_class = classes[counts.index(max(counts))] if max(counts) > 0 else 'none'
    ax.text(0.98, 0.98, f'Dominant: {dominant_class.replace("_", " ").title()}',
           transform=ax.transAxes, ha='right', va='top',
           fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))


def _plot_precision_magnitude(ax: plt.Axes, results_list: List[Dict[str, Any]]):
    """Create scatter plot of precision vs magnitude."""
    
    names = []
    effects = []
    precisions = []
    magnitudes = []
    
    for i, result in enumerate(results_list):
        if 'name' in result:
            names.append(result['name'])
        else:
            names.append(f'Effect {i+1}')
        
        effects.append(abs(result.get('effect_size', result.get('magnitude', 0))))
        precisions.append(result.get('relative_precision', 
                                    result.get('standard_error', 0.5)))
        magnitudes.append(result.get('magnitude_class', 'unknown'))
    
    if not effects:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title('Precision vs Magnitude')
        return
    
    # Color by magnitude class
    colors = []
    for mag in magnitudes:
        if mag == 'negligible':
            colors.append('gray')
        elif mag == 'small':
            colors.append('gold')
        elif mag == 'medium':
            colors.append('blue')
        elif mag == 'large':
            colors.append('green')
        elif mag == 'very large' or mag == 'very_large':
            colors.append('red')
        else:
            colors.append('black')
    
    # Create scatter plot
    scatter = ax.scatter(effects, precisions, c=colors, s=200, alpha=0.6, 
                        edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for i, (x, y, name) in enumerate(zip(effects, precisions, names)):
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    # Add quadrant lines
    median_effect = np.median(effects)
    median_precision = np.median(precisions)
    ax.axvline(x=median_effect, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=median_precision, color='gray', linestyle='--', alpha=0.3)
    
    # Add quadrant labels
    ax.text(0.02, 0.98, 'High Precision\nLow Effect', transform=ax.transAxes,
           fontsize=8, va='top', alpha=0.5)
    ax.text(0.98, 0.98, 'High Precision\nHigh Effect', transform=ax.transAxes,
           fontsize=8, va='top', ha='right', alpha=0.5, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.02, 0.02, 'Low Precision\nLow Effect', transform=ax.transAxes,
           fontsize=8, alpha=0.5)
    ax.text(0.98, 0.02, 'Low Precision\nHigh Effect', transform=ax.transAxes,
           fontsize=8, ha='right', alpha=0.5)
    
    # Styling
    ax.set_xlabel('Absolute Effect Size', fontsize=11)
    ax.set_ylabel('Relative Precision (CV)', fontsize=11)
    ax.set_title('Precision-Magnitude Trade-off', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis so high precision is at top
    ax.invert_yaxis()


def _plot_effect_distribution(ax: plt.Axes, results_list: List[Dict[str, Any]]):
    """Create distribution plot comparing effect sizes."""
    
    if len(results_list) < 2:
        # Single effect - show distribution based on CI
        if results_list:
            result = results_list[0]
            effect = result.get('effect_size', 0)
            se = result.get('standard_error', 0.1)
            
            # Generate normal distribution
            x = np.linspace(effect - 4*se, effect + 4*se, 1000)
            y = stats.norm.pdf(x, effect, se)
            
            ax.fill_between(x, 0, y, alpha=0.3, color='blue')
            ax.plot(x, y, 'b-', linewidth=2)
            
            # Add vertical lines for effect and CIs
            ax.axvline(x=effect, color='red', linestyle='-', linewidth=2, 
                      label=f'Effect = {effect:.3f}')
            ax.axvline(x=effect - 1.96*se, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=effect + 1.96*se, color='red', linestyle='--', alpha=0.5)
            
            # Add magnitude zones
            ax.axvspan(-0.01, 0.01, alpha=0.1, color='gray', label='Negligible')
            ax.axvspan(0.01, 0.05, alpha=0.1, color='yellow')
            ax.axvspan(-0.05, -0.01, alpha=0.1, color='yellow')
            
            ax.set_xlabel('Effect Size', fontsize=11)
            ax.set_ylabel('Probability Density', fontsize=11)
            ax.set_title('Effect Size Distribution', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title('Effect Size Distribution')
    else:
        # Multiple effects - create violin plot
        effects_data = []
        labels = []
        
        for i, result in enumerate(results_list):
            effect = result.get('effect_size', result.get('magnitude', 0))
            se = result.get('standard_error', 0.1)
            
            # Generate samples from distribution
            samples = np.random.normal(effect, se, 1000)
            effects_data.append(samples)
            
            if 'name' in result:
                labels.append(result['name'])
            else:
                labels.append(f'Effect {i+1}')
        
        # Create violin plot
        parts = ax.violinplot(effects_data, positions=range(len(effects_data)),
                             showmeans=True, showmedians=True, showextrema=True)
        
        # Color violins by magnitude
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.5)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add magnitude zones
        ax.axhspan(-0.01, 0.01, alpha=0.1, color='gray')
        ax.axhspan(0.01, 0.05, alpha=0.1, color='yellow')
        ax.axhspan(-0.05, -0.01, alpha=0.1, color='yellow')
        
        # Styling
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Effect Size', fontsize=11)
        ax.set_title('Effect Size Distributions Comparison', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    ax.grid(True, alpha=0.3)


def create_magnitude_comparison_matrix(
    comparison_results: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create matrix visualization for comparing multiple DML effects.
    
    Args:
        comparison_results: Output from DMLMagnitude.compare_magnitudes()
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={'width_ratios': [2, 1]})
    
    # Extract rankings
    rankings = comparison_results.get('rankings', [])
    if not rankings:
        ax1.text(0.5, 0.5, 'No comparison data available', 
                ha='center', va='center')
        return fig
    
    # Create data for heatmap
    names = [r['name'] for r in rankings]
    magnitudes = [r['magnitude'] for r in rankings]
    classes = [r['magnitude_class'] for r in rankings]
    
    # Panel 1: Ranking bar chart
    y_pos = np.arange(len(names))
    colors = []
    for cls in classes:
        if cls == 'negligible':
            colors.append('gray')
        elif cls == 'small':
            colors.append('gold')
        elif cls == 'medium':
            colors.append('blue')
        elif cls == 'large':
            colors.append('green')
        else:
            colors.append('red')
    
    bars = ax1.barh(y_pos, np.abs(magnitudes), color=colors, alpha=0.7,
                    edgecolor='black', linewidth=2)
    
    # Add rank labels
    for i, (bar, rank) in enumerate(zip(bars, range(1, len(bars) + 1))):
        ax1.text(-0.01, bar.get_y() + bar.get_height()/2,
                f'#{rank}', ha='right', va='center', fontweight='bold')
        
        # Add magnitude value
        width = bar.get_width()
        ax1.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{magnitudes[i]:.3f}', ha='left', va='center', fontsize=9)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Absolute Effect Magnitude', fontsize=11)
    ax1.set_title('Effect Magnitude Rankings', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Panel 2: Summary statistics
    ax2.axis('off')
    
    summary = comparison_results.get('summary_stats', {})
    strongest = comparison_results.get('strongest_effect', {})
    weakest = comparison_results.get('weakest_effect', {})
    
    summary_text = f"""Summary Statistics
{'=' * 25}

Strongest Effect:
  {strongest.get('name', 'N/A')}
  Magnitude: {strongest.get('magnitude', 0):.3f}
  Class: {strongest.get('class', 'N/A')}

Weakest Effect:
  {weakest.get('name', 'N/A')}
  Magnitude: {weakest.get('magnitude', 0):.3f}
  Class: {weakest.get('class', 'N/A')}

Overall Statistics:
  Mean: {summary.get('mean_magnitude', 0):.3f}
  Median: {summary.get('median_magnitude', 0):.3f}
  Std Dev: {summary.get('std_magnitude', 0):.3f}
  
Direction Consistency:
  {comparison_results.get('direction_consistency', 'N/A')}

Interpretation:
{comparison_results.get('interpretation', 'N/A')}
"""
    
    ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes,
            fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('DML Magnitude Comparison Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Magnitude comparison saved to {save_path}")
    
    return fig