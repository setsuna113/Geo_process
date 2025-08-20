"""Simple slopes visualization for hypothesis testing.

Extracted from hypothesis/simple_slopes.py lines 500-701.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def visualize_simple_slopes(
    simple_slopes_results: Dict[str, Any],
    figsize: Tuple[int, int] = (15, 5)
):
    """Create publication-quality visualization of simple slopes analysis.
    
    Takes only the results dictionary - no computation objects needed.
    
    Args:
        simple_slopes_results: Results from simple slopes analysis (must contain 'emmeans')
        figsize: Figure size
        
    Returns:
        matplotlib figure suitable for publication
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get emmeans from results - no analyzer needed!
    if 'emmeans' not in simple_slopes_results:
        raise ValueError("Results must contain 'emmeans' for visualization")
    
    emmeans = np.array(simple_slopes_results['emmeans'])
    
    # Extract results
    slope_low = simple_slopes_results.get('slope_temperate_in_low_plant', {})
    slope_high = simple_slopes_results.get('slope_temperate_in_high_plant', {})
    interaction = simple_slopes_results.get('interaction_effect', {})
    
    # Panel 1: Simple slopes plot
    ax1 = axes[0]
    
    # Define x positions for plant levels
    x_pos = [0, 1]
    plant_labels = ['Low Plant', 'High Plant']
    
    # Plot lines for each temperate level
    # Non-temperate: groups 0 (low plant) and 1 (high plant)
    ax1.plot(x_pos, [emmeans[0], emmeans[1]], 'o-', 
            color='blue', linewidth=2.5, markersize=10,
            label='Non-temperate')
    
    # Temperate: groups 2 (low plant) and 3 (high plant)
    ax1.plot(x_pos, [emmeans[2], emmeans[3]], 's--',
            color='red', linewidth=2.5, markersize=10,
            label='Temperate')
    
    # Add annotations for slopes
    # Low plant slope
    mid_x_low = 0
    mid_y_low = (emmeans[0] + emmeans[2]) / 2
    if slope_low:
        ax1.annotate(f"Slope: {slope_low.get('estimate', 0):.2f}\n"
                    f"p = {slope_low.get('p_value', 1):.3f}",
                    xy=(mid_x_low - 0.15, mid_y_low),
                    fontsize=9, ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='yellow' if slope_low.get('significant', False) else 'white',
                             alpha=0.7))
    
    # High plant slope
    mid_x_high = 1
    mid_y_high = (emmeans[1] + emmeans[3]) / 2
    if slope_high:
        color = 'lightgreen' if slope_high.get('significant', False) and slope_high.get('estimate', 0) < 0 else 'yellow' if slope_high.get('significant', False) else 'white'
        ax1.annotate(f"Slope: {slope_high.get('estimate', 0):.2f}\n"
                    f"p = {slope_high.get('p_value', 1):.3f}",
                    xy=(mid_x_high + 0.15, mid_y_high),
                    fontsize=9, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor=color, alpha=0.7))
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(plant_labels)
    ax1.set_xlabel('Plant Richness Level', fontsize=11)
    ax1.set_ylabel('Predicted Fungi Richness', fontsize=11)
    ax1.set_title('Simple Slopes Analysis', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Slope comparison
    ax2 = axes[1]
    
    slope_estimates = []
    slope_ses = []
    slope_labels = []
    colors = []
    
    if slope_low:
        slope_estimates.append(slope_low.get('estimate', 0))
        slope_ses.append(slope_low.get('standard_error', 0.1))
        slope_labels.append('Low Plant\nRegions')
        colors.append('blue' if slope_low.get('significant', False) else 'gray')
    
    if slope_high:
        slope_estimates.append(slope_high.get('estimate', 0))
        slope_ses.append(slope_high.get('standard_error', 0.1))
        slope_labels.append('High Plant\nRegions')
        colors.append('red' if slope_high.get('significant', False) else 'gray')
    
    if slope_estimates:
        y_pos = np.arange(len(slope_estimates))
        
        # Plot horizontal bars with error bars
        bars = ax2.barh(y_pos, slope_estimates, xerr=slope_ses,
                       color=colors, alpha=0.7, capsize=5)
        
        # Add reference line at 0
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        # Add value labels
        for i, (est, se) in enumerate(zip(slope_estimates, slope_ses)):
            ax2.text(est/2, i, f'{est:.2f}', 
                    ha='center', va='center', fontweight='bold',
                    color='white' if abs(est) > se else 'black')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(slope_labels)
        ax2.set_xlabel('Temperate Effect (Slope)', fontsize=11)
        ax2.set_title('Temperate Effects by Plant Level', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add significance markers
        for i, label in enumerate(slope_labels):
            if 'Low' in label and slope_low and slope_low.get('significant', False):
                ax2.text(ax2.get_xlim()[1] * 0.95, i, '*', fontsize=16, ha='right', fontweight='bold')
            elif 'High' in label and slope_high and slope_high.get('significant', False):
                ax2.text(ax2.get_xlim()[1] * 0.95, i, '***' if slope_high.get('p_value', 1) < 0.001 else '**' if slope_high.get('p_value', 1) < 0.01 else '*',
                        fontsize=16, ha='right', fontweight='bold')
    
    # Panel 3: Interaction effect visualization
    ax3 = axes[2]
    
    if interaction:
        # Create a heatmap showing the 2x2 interaction
        interaction_matrix = np.array([
            [emmeans[0], emmeans[2]],  # Low plant: non-temp, temp
            [emmeans[1], emmeans[3]]   # High plant: non-temp, temp
        ])
        
        im = ax3.imshow(interaction_matrix, cmap='RdBu_r', aspect='auto')
        
        # Add colorbar
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        
        # Set ticks and labels
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Non-temperate', 'Temperate'])
        ax3.set_yticklabels(['Low Plant', 'High Plant'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, f'{interaction_matrix[i, j]:.1f}',
                        ha='center', va='center', fontsize=12,
                        color='white' if abs(interaction_matrix[i, j] - np.mean(interaction_matrix)) > np.std(interaction_matrix) else 'black',
                        fontweight='bold')
        
        # Add interaction statistics
        if interaction.get('estimate') is not None:
            title = f"Interaction: {interaction['estimate']:.2f}"
            if interaction.get('p_value') is not None:
                title += f"\n(p = {interaction['p_value']:.3f})"
                if interaction['p_value'] < 0.05:
                    title += " *"
        else:
            title = "Interaction Effect"
        
        ax3.set_title(title, fontsize=12, fontweight='bold')
        ax3.set_xlabel('Climate Type', fontsize=11)
        ax3.set_ylabel('Plant Richness Level', fontsize=11)
    
    # Overall title
    h2_supported = simple_slopes_results.get('h2_supported', False)
    if 'h2_interpretation' in simple_slopes_results:
        overall_title = simple_slopes_results['h2_interpretation']
    else:
        overall_title = f"Simple Slopes Analysis: H2 {'Supported' if h2_supported else 'Not Supported'}"
    
    fig.suptitle(overall_title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig