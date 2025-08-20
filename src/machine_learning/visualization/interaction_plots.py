"""Interaction and contrast visualization for hypothesis testing.

Extracted from hypothesis/contrast_analysis.py lines 441-776.
No logic changes - pure migration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def visualize_contrast_results(
    contrast_analyzer,  # The ContrastAnalysis instance
    contrast_results: Dict[str, Any],
    figsize: Tuple[int, int] = (15, 5)
):
    """Create publication-quality visualization of contrast analysis.
    
    EXACT copy from contrast_analysis.py - requires ContrastAnalysis instance.
    
    Args:
        contrast_analyzer: ContrastAnalysis instance (for self.emmeans_wrapper)
        contrast_results: Results from contrast analysis
        figsize: Figure size
        
    Returns:
        matplotlib figure suitable for publication
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get emmeans and contrast info - EXACTLY as in original
    from ..evaluation.emmeans_core import REMmeansWrapper
    if isinstance(contrast_analyzer.emmeans_wrapper, REMmeansWrapper):
        emmeans = contrast_analyzer.emmeans_wrapper.get_emmeans_array()
    else:
        emmeans_dict = contrast_analyzer.emmeans_wrapper.compute_emmeans_2x2()
        emmeans = np.array([
            emmeans_dict['group_0_0']['emmean'],
            emmeans_dict['group_1_0']['emmean'],
            emmeans_dict['group_0_1']['emmean'],
            emmeans_dict['group_1_1']['emmean']
        ])
    group_names = ['Low Plant\nNon-temp', 'High Plant\nNon-temp',
                  'Low Plant\nTemperate', 'High Plant\nTemperate']
    
    if 'contrast_estimate' in contrast_results:
        estimate = contrast_results['contrast_estimate']
        se = contrast_results['standard_error']
        p_value = contrast_results['p_value']
        contrast_vec = contrast_results['contrast_vector']
    else:
        # Handle multiple contrasts
        first_result = list(contrast_results.values())[0]
        estimate = first_result['estimate']
        se = first_result['standard_error']
        p_value = first_result['p_value']
        contrast_vec = first_result['contrast_vector']
    
    # Panel 1: Emmeans with error bars
    ax1 = axes[0]
    x_pos = np.arange(len(emmeans))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = ax1.bar(x_pos, emmeans, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars (using approximate SE)
    approx_se = [abs(e * 0.05) for e in emmeans]  # Approximate 5% SE
    ax1.errorbar(x_pos, emmeans, yerr=approx_se, fmt='none', 
                color='black', capsize=5, capthick=2)
    
    # Highlight groups involved in contrast
    for i, coef in enumerate(contrast_vec):
        if coef != 0:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(3)
            # Add contrast coefficient label
            ax1.text(i, emmeans[i] + approx_se[i] + 0.5, 
                    f"{coef:+g}", ha='center', fontweight='bold', fontsize=11)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(group_names, rotation=45, ha='right')
    ax1.set_ylabel('Estimated Marginal Mean', fontsize=11)
    ax1.set_title('Group Means with Contrast Weights', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Contrast estimate with CI
    ax2 = axes[1]
    
    # Calculate CI
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    
    # Color based on significance and direction
    if p_value < 0.05:
        color = 'darkgreen' if estimate < 0 else 'darkred'
    else:
        color = 'gray'
    
    # Plot contrast with CI
    ax2.barh(0, estimate, xerr=1.96*se, height=0.5, color=color, alpha=0.7,
            capsize=10, error_kw={'linewidth': 2})
    
    # Add reference line at 0
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    
    # Add text annotations
    ax2.text(estimate/2, 0, f"{estimate:.2f}", ha='center', va='center',
            fontweight='bold', fontsize=12, color='white')
    
    ax2.set_xlim([min(ci_lower*1.2, -1), max(ci_upper*1.2, 1)])
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_xlabel('Contrast Estimate', fontsize=11)
    ax2.set_yticks([])
    ax2.set_title(f'Contrast Effect (p = {p_value:.3f})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add significance annotation
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    ax2.text(0.5, 0.9, sig_text, transform=ax2.transAxes,
            fontsize=16, ha='center', fontweight='bold')
    
    # Panel 3: Pairwise differences visualization
    ax3 = axes[2]
    
    # Create difference matrix
    n_groups = len(emmeans)
    diff_matrix = np.zeros((n_groups, n_groups))
    
    for i in range(n_groups):
        for j in range(n_groups):
            diff_matrix[i, j] = emmeans[j] - emmeans[i]
    
    # Plot heatmap
    im = ax3.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', 
                   vmin=-np.max(np.abs(diff_matrix)), vmax=np.max(np.abs(diff_matrix)))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Difference', rotation=270, labelpad=15, fontsize=10)
    
    # Set ticks and labels
    ax3.set_xticks(range(n_groups))
    ax3.set_yticks(range(n_groups))
    ax3.set_xticklabels([g.split('\n')[0] for g in group_names], rotation=45, ha='right')
    ax3.set_yticklabels([g.split('\n')[0] for g in group_names])
    
    # Add text annotations for values
    for i in range(n_groups):
        for j in range(n_groups):
            if i != j:
                text = ax3.text(j, i, f'{diff_matrix[i, j]:.1f}',
                               ha='center', va='center', fontsize=8,
                               color='white' if abs(diff_matrix[i, j]) > np.max(np.abs(diff_matrix))*0.5 else 'black')
    
    ax3.set_title('Pairwise Differences', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Group (Column - Row)', fontsize=10)
    ax3.set_ylabel('Group', fontsize=10)
    
    # Overall title
    if 'h2_supported' in contrast_results:
        if contrast_results['h2_supported']:
            decision = "H2 Supported"
        else:
            decision = "H2 Not Supported"
    else:
        decision = "Significant" if p_value < 0.05 else "Not Significant"
    
    fig.suptitle(f'Contrast Analysis Results: {decision}',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def create_h2_publication_figure(
    contrast_analyzer,  # The ContrastAnalysis instance
    contrast_results: Dict[str, Any],
    data: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (16, 10)
):
    """Create comprehensive 4-panel figure for H2 hypothesis testing.
    
    METHOD 2 ONLY: This visualization is specifically for Method 2's H2 testing.
    EXACT copy from contrast_analysis.py lines 605-776.
    
    Args:
        contrast_analyzer: ContrastAnalysis instance (for self.emmeans_wrapper)
        contrast_results: Results from H2 contrast test
        data: Optional dataframe with raw data for scatter plot
        figsize: Figure size
        
    Returns:
        Publication-ready 4-panel figure for Method 2 H2 analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get emmeans - EXACTLY as in original
    from ..evaluation.emmeans_core import REMmeansWrapper
    if isinstance(contrast_analyzer.emmeans_wrapper, REMmeansWrapper):
        emmeans = contrast_analyzer.emmeans_wrapper.get_emmeans_array()
    else:
        emmeans_dict = contrast_analyzer.emmeans_wrapper.compute_emmeans_2x2()
        emmeans = np.array([
            emmeans_dict['group_0_0']['emmean'],
            emmeans_dict['group_1_0']['emmean'],
            emmeans_dict['group_0_1']['emmean'],
            emmeans_dict['group_1_1']['emmean']
        ])
    group_names = ['Low Plant\nNon-temp', 'High Plant\nNon-temp',
                  'Low Plant\nTemperate', 'High Plant\nTemperate']
    
    estimate = contrast_results['contrast_estimate']
    se = contrast_results['standard_error']
    p_value = contrast_results['p_value']
    h2_supported = contrast_results.get('h2_supported', False)
    
    # Panel 1: Bar plot of emmeans with CI
    ax1 = axes[0, 0]
    x_pos = np.arange(4)
    colors = ['lightblue', 'darkblue', 'lightcoral', 'darkred']
    
    # Approximate SE for error bars
    emmeans_se = [abs(e * 0.05) for e in emmeans]
    
    bars = ax1.bar(x_pos, emmeans, yerr=emmeans_se, color=colors,
                  capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(group_names, rotation=45, ha='right')
    ax1.set_ylabel('Predicted Fungi Richness', fontsize=11)
    ax1.set_title('Estimated Marginal Means', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add significance annotation
    if p_value < 0.05:
        # Draw line between high-plant groups
        y_max = max(emmeans[1] + emmeans_se[1], emmeans[3] + emmeans_se[3]) + 2
        ax1.plot([1, 3], [y_max, y_max], 'k-', linewidth=1.5)
        ax1.text(2, y_max + 0.5, '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*',
                ha='center', fontsize=14, fontweight='bold')
    
    # Panel 2: Interaction plot
    ax2 = axes[0, 1]
    plant_levels = ['Low', 'High']
    
    # Non-temperate line
    ax2.plot(plant_levels, [emmeans[0], emmeans[1]], 'b-o',
            linewidth=2.5, markersize=10, label='Non-temperate')
    # Temperate line  
    ax2.plot(plant_levels, [emmeans[2], emmeans[3]], 'r--s',
            linewidth=2.5, markersize=10, label='Temperate')
    
    ax2.set_xlabel('Plant Richness Level', fontsize=11)
    ax2.set_ylabel('Predicted Fungi Richness', fontsize=11)
    ax2.set_title('Interaction: Plant × Climate', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Scatter plot with observed data (if available) or effect sizes
    ax3 = axes[1, 0]
    
    if data is not None and all(col in data.columns for col in ['temperate', 'plant_high', 'fungi_richness', 'plants_richness']):
        # Create scatter plot with actual data
        for temp in [0, 1]:
            for plant in [0, 1]:
                mask = (data['temperate'] == temp) & (data['plant_high'] == plant)
                idx = temp * 2 + plant
                ax3.scatter(data.loc[mask, 'plants_richness'],
                          data.loc[mask, 'fungi_richness'],
                          alpha=0.3, s=1, c=colors[idx],
                          label=group_names[idx].replace('\n', ' '))
        
        ax3.set_xlabel('Plant Richness (continuous)', fontsize=11)
        ax3.set_ylabel('Fungi Richness', fontsize=11)
        ax3.set_title('Observed Data by Categories', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8, markerscale=5)
    else:
        # Alternative: Show effect sizes
        effects = [
            ('Plant Effect\n(Low Temp)', emmeans[1] - emmeans[0]),
            ('Plant Effect\n(High Temp)', emmeans[3] - emmeans[2]),
            ('Temp Effect\n(Low Plant)', emmeans[2] - emmeans[0]),
            ('Temp Effect\n(High Plant)', emmeans[3] - emmeans[1])
        ]
        
        labels, values = zip(*effects)
        x_pos = np.arange(len(labels))
        colors_effects = ['blue' if v > 0 else 'red' for v in values]
        
        bars = ax3.bar(x_pos, values, color=colors_effects, alpha=0.7)
        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_ylabel('Effect Size', fontsize=11)
        ax3.set_title('Decomposed Effects', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center',
                    va='bottom' if val > 0 else 'top', fontsize=9)
    
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Contrast visualization and summary
    ax4 = axes[1, 1]
    
    # Create summary text box
    summary_text = f"""H2 Hypothesis Test Summary
    
Contrast: High Plant (Temp - NonTemp)
Estimate: {estimate:.2f} ± {se:.2f} SE
95% CI: [{estimate-1.96*se:.2f}, {estimate+1.96*se:.2f}]
p-value: {p_value:.4f}

H2 Prediction: Negative contrast
(fewer fungi in temperate high-plant)

Result: {'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}
"""
    
    # Add visual representation of contrast
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    # Add contrast bar
    ax_inset = ax4.inset_axes([0.65, 0.3, 0.3, 0.4])
    color = 'green' if h2_supported else 'red' if p_value < 0.05 else 'gray'
    ax_inset.barh(0, estimate, xerr=1.96*se, height=0.5,
                 color=color, alpha=0.7, capsize=8)
    ax_inset.axvline(0, color='black', linestyle='-', linewidth=1)
    ax_inset.set_xlim([min(estimate-2*se, -1), max(estimate+2*se, 1)])
    ax_inset.set_ylim([-0.5, 0.5])
    ax_inset.set_xlabel('Contrast', fontsize=9)
    ax_inset.set_yticks([])
    ax_inset.grid(True, alpha=0.3, axis='x')
    
    ax4.axis('off')
    
    # Overall title
    fig.suptitle('H2 Hypothesis Test: Temperate Mismatch in Plant-Fungi Relationship',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig