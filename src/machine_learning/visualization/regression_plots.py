"""Visualization functions for regression analysis results.

Implements comprehensive visualizations from regression_analysis_decisions.md lines 578-671.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_h2_mismatch_visualization(
    data: pd.DataFrame,
    emmeans_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive 4-panel visualization of the temperate mismatch pattern.
    
    Implements visualization from regression_analysis_decisions.md lines 578-671.
    
    Args:
        data: DataFrame with processed data including temperate and plant_high columns
        emmeans_results: Dictionary with emmeans results from Method 2
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure with 4 panels
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Define colors for each group
    colors = ['lightblue', 'darkblue', 'lightcoral', 'darkred']
    
    # Panel 1: Bar plot of emmeans with confidence intervals
    ax1 = axes[0, 0]
    
    # Extract emmeans values
    emmeans = emmeans_results.get('marginal_means', {})
    
    # Create DataFrame for plotting
    conditions = []
    means = []
    errors = []
    
    for plant in [0, 1]:
        for temp in [0, 1]:
            key = f"plant_{plant}_temp_{temp}"
            if key in emmeans:
                plant_label = "High" if plant else "Low"
                temp_label = "Temperate" if temp else "Non-temp"
                conditions.append(f"{plant_label} Plants\n{temp_label}")
                means.append(emmeans[key]['mean'])
                errors.append(emmeans[key]['se'] * 1.96)  # 95% CI
    
    x = np.arange(len(conditions))
    bars = ax1.bar(x, means, yerr=errors, color=colors[:len(means)], 
                   capsize=5, edgecolor='black', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.set_ylabel('Predicted Fungi Richness')
    ax1.set_title('Estimated Marginal Means by Group')
    
    # Add significance annotation if available
    h2_result = emmeans_results.get('h2_contrast', {})
    if h2_result.get('p_value', 1) < 0.05:
        ax1.annotate('***', xy=(2.5, max(means) * 1.1), 
                    fontsize=16, ha='center', color='red')
    
    # Panel 2: Interaction plot
    ax2 = axes[0, 1]
    
    # Prepare data for interaction plot
    plant_levels = ['Low Plants', 'High Plants']
    
    # Non-temperate line
    non_temp_means = []
    temp_means = []
    
    for plant in [0, 1]:
        key_nontemp = f"plant_{plant}_temp_0"
        key_temp = f"plant_{plant}_temp_1"
        
        if key_nontemp in emmeans:
            non_temp_means.append(emmeans[key_nontemp]['mean'])
        if key_temp in emmeans:
            temp_means.append(emmeans[key_temp]['mean'])
    
    if len(non_temp_means) == 2 and len(temp_means) == 2:
        ax2.plot(plant_levels, non_temp_means, 'o-', label='Non-temperate',
                linewidth=2.5, markersize=10, color='blue')
        ax2.plot(plant_levels, temp_means, 's--', label='Temperate',
                linewidth=2.5, markersize=10, color='red')
        
        ax2.set_xlabel('Plant Richness Category')
        ax2.set_ylabel('Predicted Fungi Richness')
        ax2.set_title('Interaction: Plant Richness × Climate Type')
        ax2.legend(loc='best', frameon=True)
        ax2.grid(True, alpha=0.3)
        
        # Add interpretation text
        if h2_result.get('estimate', 0) < 0:
            ax2.text(0.5, 0.95, 'Negative interaction detected',
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Panel 3: Scatter plot with observed data
    ax3 = axes[1, 0]
    
    # Plot each group with different colors
    group_labels = []
    for temp in [0, 1]:
        for plant in [0, 1]:
            mask = (data['temperate'] == temp) & (data['plant_high'] == plant)
            if mask.sum() > 0:
                plant_label = "High" if plant else "Low"
                temp_label = "Temp" if temp else "Non-temp"
                label = f"{temp_label}, {plant_label} plants"
                color_idx = temp * 2 + plant
                
                ax3.scatter(data.loc[mask, 'plants_richness'], 
                           data.loc[mask, 'fungi_richness'],
                           alpha=0.4, s=10, label=label, 
                           c=[colors[color_idx % len(colors)]])
                group_labels.append(label)
    
    ax3.set_xlabel('Plant Richness (continuous)')
    ax3.set_ylabel('Fungi Richness (observed)')
    ax3.set_title('Observed Data by H2 Categories')
    ax3.legend(fontsize=8, loc='best', frameon=True)
    
    # Add median lines for each category
    for temp in [0, 1]:
        for plant in [0, 1]:
            mask = (data['temperate'] == temp) & (data['plant_high'] == plant)
            if mask.sum() > 0:
                median_plants = data.loc[mask, 'plants_richness'].median()
                median_fungi = data.loc[mask, 'fungi_richness'].median()
                color_idx = temp * 2 + plant
                ax3.plot(median_plants, median_fungi, 'X', 
                        color=colors[color_idx % len(colors)],
                        markersize=12, markeredgecolor='black', markeredgewidth=2)
    
    # Panel 4: Contrast visualization and effect size
    ax4 = axes[1, 1]
    
    if h2_result:
        contrast_val = h2_result.get('estimate', 0)
        contrast_se = h2_result.get('se', 0)
        p_value = h2_result.get('p_value', 1)
        
        # Create horizontal bar plot for contrast
        bar_color = 'darkred' if contrast_val < 0 else 'darkgreen'
        ax4.barh(0, contrast_val, xerr=contrast_se * 1.96,
                color=bar_color, height=0.3, capsize=10,
                edgecolor='black', linewidth=2)
        
        # Add reference line at zero
        ax4.axvline(0, color='black', linestyle='-', linewidth=2)
        
        # Format the plot
        ax4.set_xlabel('H2 Contrast Effect Size\n(High Plants: Temperate - Non-temperate)')
        ax4.set_title(f'H2 Hypothesis Test\np-value = {p_value:.4f}')
        ax4.set_yticks([])
        ax4.set_ylim(-0.5, 0.5)
        
        # Add grid for better readability
        ax4.grid(True, axis='x', alpha=0.3)
        
        # Add interpretation text box
        if contrast_val < 0 and p_value < 0.05:
            interpretation = ('TEMPERATE MISMATCH CONFIRMED\n\n'
                            'Temperate regions show reduced\n'
                            'fungi richness despite high plant\n'
                            'diversity, supporting H2 hypothesis')
            box_color = 'lightcoral'
        elif contrast_val > 0 and p_value < 0.05:
            interpretation = ('UNEXPECTED PATTERN\n\n'
                            'Temperate regions show increased\n'
                            'fungi richness with high plants,\n'
                            'contradicting H2 hypothesis')
            box_color = 'lightgreen'
        else:
            interpretation = ('NO SIGNIFICANT EFFECT\n\n'
                            'No evidence for differential\n'
                            'plant-fungi relationships between\n'
                            'temperate and non-temperate regions')
            box_color = 'lightgray'
        
        ax4.text(0.5, 0.7, interpretation,
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.7))
        
        # Add effect size interpretation
        effect_size_text = f"Effect: {contrast_val:.3f} ± {contrast_se*1.96:.3f}"
        ax4.text(0.5, 0.15, effect_size_text,
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=11)
    
    # Overall title
    fig.suptitle('H2 Temperate Mismatch Analysis: Method 2 (Categorical GLM)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"H2 mismatch visualization saved to {save_path}")
    
    return fig


def create_method1_interaction_plots(
    data: pd.DataFrame,
    model_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create visualization for Method 1 continuous interaction results.
    
    Args:
        data: DataFrame with processed data
        model_results: Dictionary with Method 1 model results
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Scatter plot with interaction visualization
    ax1 = axes[0, 0]
    
    # Create bins for bio4 to show interaction
    bio4_quantiles = data['bio4_std'].quantile([0, 0.33, 0.67, 1.0])
    bio4_labels = ['Low Seasonality', 'Medium Seasonality', 'High Seasonality']
    data['bio4_category'] = pd.cut(data['bio4_std'], bins=bio4_quantiles, 
                                   labels=bio4_labels, include_lowest=True)
    
    # Plot by seasonality category
    for i, category in enumerate(bio4_labels):
        mask = data['bio4_category'] == category
        if mask.sum() > 0:
            color = plt.cm.coolwarm(i / 2)
            ax1.scatter(data.loc[mask, 'plants_richness'], 
                       data.loc[mask, 'fungi_richness'],
                       alpha=0.5, s=10, label=category, c=[color])
    
    ax1.set_xlabel('Plant Richness')
    ax1.set_ylabel('Fungi Richness')
    ax1.set_title('Plant-Fungi Relationship by Temperature Seasonality')
    ax1.legend(loc='best', frameon=True)
    
    # Panel 2: Coefficient plot
    ax2 = axes[0, 1]
    
    interaction_info = model_results.get('interaction_term', {})
    if interaction_info:
        coef = interaction_info.get('coefficient', 0)
        se = interaction_info.get('std_error', 0)
        p_val = interaction_info.get('p_value', 1)
        
        # Create coefficient plot
        ax2.errorbar(0, coef, yerr=se*1.96, fmt='o', markersize=10,
                    color='red' if p_val < 0.05 else 'gray',
                    capsize=10, linewidth=2)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Plants × Bio4\nInteraction'])
        ax2.set_ylabel('Coefficient Value')
        ax2.set_title(f'Interaction Coefficient\np = {p_val:.4f}')
        ax2.grid(True, alpha=0.3)
        
        # Add interpretation
        if coef < 0 and p_val < 0.05:
            ax2.text(0.5, 0.9, 'Negative interaction:\nPlant effect weakens in temperate regions',
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Panel 3: Model comparison (AIC)
    ax3 = axes[1, 0]
    
    models_info = model_results.get('models', {})
    if models_info:
        model_names = ['Null\n(Climate)', 'Full\n(+Biodiversity)', 'Interaction\n(+Plant×Bio4)']
        aic_values = [
            models_info.get('null', {}).get('aic', 0),
            models_info.get('full', {}).get('aic', 0),
            models_info.get('interaction', {}).get('aic', 0)
        ]
        
        colors = ['lightgray', 'lightblue', 'lightgreen']
        bars = ax3.bar(model_names, aic_values, color=colors, edgecolor='black', linewidth=1)
        
        # Highlight best model
        min_aic_idx = np.argmin(aic_values)
        bars[min_aic_idx].set_color('gold')
        
        ax3.set_ylabel('AIC')
        ax3.set_title('Model Comparison')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # Add ΔAIC annotations
        min_aic = min(aic_values)
        for i, (name, aic) in enumerate(zip(model_names, aic_values)):
            delta_aic = aic - min_aic
            ax3.text(i, aic + max(aic_values)*0.01, f'Δ={delta_aic:.1f}',
                    ha='center', va='bottom', fontsize=10)
    
    # Panel 4: Predicted vs Observed
    ax4 = axes[1, 1]
    
    # This would need fitted values from the model
    # For now, create a placeholder
    ax4.text(0.5, 0.5, 'Predicted vs Observed\n(Requires model fitted values)',
            transform=ax4.transAxes, ha='center', va='center',
            fontsize=12, style='italic')
    ax4.set_xlabel('Predicted Fungi Richness')
    ax4.set_ylabel('Observed Fungi Richness')
    ax4.set_title('Model Fit Assessment')
    
    fig.suptitle('Method 1: Continuous Interaction Analysis', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Method 1 visualization saved to {save_path}")
    
    return fig


def create_dml_effects_plot(
    dml_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create visualization for DML causal effects.
    
    Args:
        dml_results: Dictionary with DML results for all hypotheses
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    hypotheses = ['h0', 'h1', 'h2']
    titles = ['H0: Plants → Fungi', 'H1: Vertebrates → Fungi', 'H2: Temperate Mismatch']
    colors = ['green', 'blue', 'red']
    
    for ax, hyp, title, color in zip(axes, hypotheses, titles, colors):
        result = dml_results.get(hyp, {})
        
        if result:
            effect = result.get('causal_effect', 0)
            ci_lower = result.get('ci_lower', 0)
            ci_upper = result.get('ci_upper', 0)
            p_value = result.get('p_value', 1)
            
            # Create forest plot style visualization
            ax.errorbar(effect, 0, xerr=[[effect - ci_lower], [ci_upper - effect]],
                       fmt='o', markersize=12, color=color,
                       capsize=10, linewidth=2.5,
                       markeredgecolor='black', markeredgewidth=1)
            
            # Add reference line at zero
            ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            # Format
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel('Causal Effect')
            ax.set_title(f'{title}\np = {p_value:.4f}')
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add interpretation
            if p_value < 0.05:
                if effect > 0:
                    interpretation = "Positive effect"
                    box_color = 'lightgreen'
                else:
                    interpretation = "Negative effect"
                    box_color = 'lightcoral'
            else:
                interpretation = "No significant effect"
                box_color = 'lightgray'
            
            ax.text(0.5, 0.8, interpretation,
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=11, weight='bold',
                   bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.5))
            
            # Add effect size
            ax.text(0.5, 0.2, f'Effect: {effect:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10)
    
    fig.suptitle('Method 3: Double Machine Learning - Causal Effects', 
                fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"DML effects plot saved to {save_path}")
    
    return fig


def create_regression_synthesis_plot(
    all_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create synthesis visualization comparing all three methods.
    
    Args:
        all_results: Dictionary with results from all three methods
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: H2 results comparison across methods
    ax1 = axes[0, 0]
    
    methods = ['Method 1\n(Continuous)', 'Method 2\n(Categorical)', 'Method 3\n(DML)']
    h2_pvalues = []
    h2_effects = []
    
    # Extract H2 results from each method
    method1 = all_results.get('method1', {})
    if method1:
        interaction = method1.get('interaction_term', {})
        h2_pvalues.append(interaction.get('p_value', 1))
        h2_effects.append(interaction.get('coefficient', 0))
    else:
        h2_pvalues.append(1)
        h2_effects.append(0)
    
    method2 = all_results.get('method2', {})
    if method2:
        h2_contrast = method2.get('h2_contrast', {})
        h2_pvalues.append(h2_contrast.get('p_value', 1))
        h2_effects.append(h2_contrast.get('estimate', 0))
    else:
        h2_pvalues.append(1)
        h2_effects.append(0)
    
    method3 = all_results.get('method3', {})
    if method3:
        h2_result = method3.get('h2', {})
        h2_pvalues.append(h2_result.get('p_value', 1))
        h2_effects.append(h2_result.get('causal_effect', 0))
    else:
        h2_pvalues.append(1)
        h2_effects.append(0)
    
    # Create bar plot with p-values
    colors = ['red' if p < 0.05 else 'gray' for p in h2_pvalues]
    bars = ax1.bar(methods, [-np.log10(p) for p in h2_pvalues], color=colors,
                   edgecolor='black', linewidth=1)
    
    # Add significance line
    ax1.axhline(-np.log10(0.05), color='black', linestyle='--', 
               label='p = 0.05', linewidth=1)
    
    ax1.set_ylabel('-log10(p-value)')
    ax1.set_title('H2 Hypothesis Test Results')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add p-value labels
    for i, (bar, p) in enumerate(zip(bars, h2_pvalues)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'p={p:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Panel 2: Effect sizes comparison
    ax2 = axes[0, 1]
    
    # Normalize effect sizes for comparison
    ax2.bar(methods, h2_effects, color=['lightblue', 'lightgreen', 'lightyellow'],
           edgecolor='black', linewidth=1)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_ylabel('Effect Size')
    ax2.set_title('H2 Effect Sizes Across Methods')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for i, effect in enumerate(h2_effects):
        ax2.text(i, effect + max(abs(min(h2_effects)), abs(max(h2_effects)))*0.05,
                f'{effect:.3f}', ha='center', va='bottom' if effect > 0 else 'top',
                fontsize=10)
    
    # Panel 3: Consensus summary
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    # Count significant results
    significant_count = sum(1 for p in h2_pvalues if p < 0.05)
    
    # Create consensus text
    h2_status = 'H2 SUPPORTED' if significant_count >= 2 else 'H2 NOT SUPPORTED'
    h2_interpretation = ('Evidence suggests temperate regions show\nreduced fungi despite high plant diversity' 
                        if significant_count >= 2 else 
                        'Insufficient evidence for temperate\nmismatch hypothesis')
    
    consensus_text = f"""
    SYNTHESIS SUMMARY
    ═══════════════════════════════
    
    H2 Temperate Mismatch Hypothesis:
    
    • Significant in {significant_count}/3 methods
    • Method 1 (Continuous): p = {h2_pvalues[0]:.3f}
    • Method 2 (Categorical): p = {h2_pvalues[1]:.3f}
    • Method 3 (DML): p = {h2_pvalues[2]:.3f}
    
    Overall Conclusion:
    {h2_status}
    
    {h2_interpretation}
    """
    
    ax3.text(0.5, 0.5, consensus_text, transform=ax3.transAxes,
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='lightgreen' if significant_count >= 2 else 'lightgray',
                     alpha=0.7))
    
    # Panel 4: Method characteristics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    method_comparison = """
    METHOD CHARACTERISTICS
    ═══════════════════════════════
    
    Method 1 - Continuous GLM:
    • Tests continuous interaction
    • Preserves full information
    • Assumes linear relationships
    
    Method 2 - Categorical GLM:
    • Uses top 25% thresholds
    • Clear group comparisons
    • emmeans for marginal effects
    
    Method 3 - DML:
    • Causal inference approach
    • Controls for confounding
    • Machine learning flexibility
    """
    
    ax4.text(0.5, 0.5, method_comparison, transform=ax4.transAxes,
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
    
    fig.suptitle('Regression Analysis Synthesis: Comparing All Methods', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Synthesis plot saved to {save_path}")
    
    return fig