from src.abstractions import get_logger
"""Model Comparison Visualization - ONLY visualization, no calculations.

All metrics are calculated in evaluation/model_comparison.py
This module ONLY creates plots from those results.
NO CALCULATIONS ALLOWED - just display what's in the results dictionary.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = get_logger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def plot_model_comparison(comparison_results: Dict[str, Any],
                         title: str = "Model Comparison",
                         figsize: tuple = (15, 8),
                         save_path: Optional[Path] = None) -> plt.Figure:
    """Create comprehensive model comparison visualization.
    
    This is the ONLY model comparison plotting function.
    Takes results from ModelComparison.compare_models() and visualizes them.
    
    Args:
        comparison_results: Results from ModelComparison.compare_models()
        title: Main title for the plot
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Create figure with subplots - simplified to 1x2 grid (integrated % improvement into AIC/BIC plot)
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Extract data
    models = comparison_results['models']
    weights = comparison_results.get('weights', {})
    best_model = comparison_results.get('best_model', {})
    
    # 1. AIC/BIC Values with % Improvements (left)
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_ic_values(ax1, models)
    
    # 2. R² Values (right) - keep only one R² plot
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_r_squared_values(ax2, models, comparison_results.get('improvement'))
    
    # Add main title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return fig


def _plot_ic_values(ax, models):
    """Plot AIC and BIC values with percentage improvements as third bars."""
    model_names = list(models.keys())
    aics = [m.get('aic', np.nan) for m in models.values()]
    bics = [m.get('bic', np.nan) for m in models.values()]
    
    # Group by IC type with 3 bars each: null, full, %change
    ic_types = ['AIC', 'BIC']
    
    # Get null and full values - handle both Method 1 ('null'/'full') and Method 2 naming
    if 'null' in model_names:
        null_idx = model_names.index('null')
        full_idx = model_names.index('full') if 'full' in model_names else 1
    elif 'Method1_Full' in model_names:
        # For Method 2 comparison: Method1_Full is the baseline, Method2a_Continuous is the comparison
        null_idx = model_names.index('Method1_Full')
        full_idx = model_names.index('Method2a_Continuous') if 'Method2a_Continuous' in model_names else 1
    else:
        # Fallback to first two models
        null_idx = 0
        full_idx = 1
    
    null_values = [aics[null_idx], bics[null_idx]]
    full_values = [aics[full_idx], bics[full_idx]]
    
    # Calculate percentage improvements (scale them to be visible with IC values)
    improvements = []
    improvement_actual = []  # Keep actual values for labels
    for null_val, full_val in zip(null_values, full_values):
        if not np.isnan(null_val) and not np.isnan(full_val) and null_val != 0:
            improvement = ((null_val - full_val) / null_val) * 100
            improvement_actual.append(improvement)
            # Scale the improvement to be visible on the same plot
            # Use a scaling factor based on the IC values
            scale_factor = null_val / 100  # Scale % to be visible relative to IC values
            improvements.append(improvement * scale_factor)
        else:
            improvements.append(0)
            improvement_actual.append(0)
    
    # Create DataFrame for seaborn with all three bar types
    import pandas as pd
    data_list = []
    for ic_type, null_val, full_val, improv_scaled in zip(ic_types, null_values, full_values, improvements):
        data_list.append({'IC Type': ic_type, 'Model': 'Null Model', 'Value': null_val})
        data_list.append({'IC Type': ic_type, 'Model': 'Full Model', 'Value': full_val})
        data_list.append({'IC Type': ic_type, 'Model': '% Improvement (scaled)', 'Value': improv_scaled})
    
    df = pd.DataFrame(data_list)
    
    # Use single seaborn barplot for all bars
    sns.barplot(data=df, x='IC Type', y='Value', hue='Model', ax=ax,
                palette={'Null Model': 'lightcoral', 
                        'Full Model': 'lightgreen', 
                        '% Improvement (scaled)': 'gold'},
                alpha=0.8)
    
    # Add value labels
    patches = ax.patches
    # Seaborn groups patches by hue first (all Null, then all Full, then all %)
    # So order is: Null[AIC], Null[BIC], Full[AIC], Full[BIC], %[AIC], %[BIC]
    
    # Create a mapping of approximate heights to their labels
    height_to_label = {}
    # Map null values
    for i, val in enumerate(null_values):
        if not np.isnan(val):
            height_to_label[val] = f'{val:.1e}' if abs(val) > 1000 else f'{val:.1f}'
    # Map full values  
    for i, val in enumerate(full_values):
        if not np.isnan(val):
            height_to_label[val] = f'{val:.1e}' if abs(val) > 1000 else f'{val:.1f}'
    # Map improvement values (scaled heights to actual % labels)
    for i, (scaled_val, actual_val) in enumerate(zip(improvements, improvement_actual)):
        if scaled_val != 0:
            height_to_label[scaled_val] = f'{actual_val:.1f}%'
    
    # Label each patch
    for patch in patches:
        height = patch.get_height()
        if not np.isnan(height) and height != 0:
            # Find the matching label
            label = None
            for h, l in height_to_label.items():
                if abs(height - h) < 1e-6:  # Use small tolerance for float comparison
                    label = l
                    break
            
            if label:
                if '%' in label:
                    # % improvement labels
                    ax.text(patch.get_x() + patch.get_width()/2., height,
                           label, ha='center', va='bottom', 
                           fontsize=9, fontweight='bold', color='darkorange')
                else:
                    # IC value labels
                    ax.text(patch.get_x() + patch.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Set up dual y-axis to show the actual percentage scale
    ax2 = ax.twinx()
    ax2.set_ylabel('Improvement (%)', color='darkorange', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    # Hide the grid lines to avoid confusion
    ax2.grid(False)
    
    # Set the secondary axis to show actual percentage values
    # But since bars are scaled, we need to adjust the axis accordingly
    if improvement_actual and null_values[0] and not np.isnan(null_values[0]):
        # The scale factor used was null_val / 100
        scale_factor = null_values[0] / 100  # Use AIC's scale factor
        max_improvement = max(abs(i) for i in improvement_actual)
        
        # Set primary axis limit to accommodate scaled values
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max(current_ylim[1], max_improvement * scale_factor * 1.2))
        
        # Set secondary axis to show actual percentage scale
        # This axis shows what the scaled bars actually represent in %
        ax2.set_ylim(current_ylim[0] / scale_factor, current_ylim[1] / scale_factor)
    
    ax.set_xlabel('Information Criterion Type', fontsize=11)
    ax.set_ylabel('IC Value', fontsize=11)
    ax.set_title('Model Comparison: AIC, BIC, and % Improvements', fontsize=12, fontweight='bold')
    
    # Update legend to clarify scaled improvements
    handles, labels = ax.get_legend_handles_labels()
    for i, label in enumerate(labels):
        if '% Improvement (scaled)' in label:
            labels[i] = '% Improvement'
    ax.legend(handles, labels, loc='upper left', fontsize=10)
    
    # Remove grid lines to avoid scale confusion
    ax.grid(False)


def _plot_delta_ic(ax, models, threshold=None):
    """Plot delta AIC and delta BIC with sample-size adjusted threshold."""
    model_names = list(models.keys())
    delta_aics = [m.get('delta_aic', 0) for m in models.values()]
    delta_bics = [m.get('delta_bic', 0) for m in models.values()]
    
    # Filter out NaN values and ensure we have data
    valid_data = [(name, daic, dbic) for name, daic, dbic in zip(model_names, delta_aics, delta_bics)
                  if not np.isnan(daic)]
    
    if not valid_data:
        ax.text(0.5, 0.5, 'No delta IC values available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Delta IC Values')
        ax.axis('off')
        return
    
    names, daic_vals, dbic_vals = zip(*valid_data)
    
    # Create DataFrame for seaborn
    import pandas as pd
    data_list = []
    for i, name in enumerate(names):
        data_list.append({'Model': name, 'IC Type': 'ΔAIC', 'Value': daic_vals[i]})
        data_list.append({'Model': name, 'IC Type': 'ΔBIC', 'Value': dbic_vals[i]})
    
    df = pd.DataFrame(data_list)
    
    # Use seaborn grouped barplot
    sns.barplot(data=df, x='Model', y='Value', hue='IC Type', ax=ax,
                palette={'ΔAIC': 'steelblue', 'ΔBIC': 'coral'}, alpha=0.8)
    
    # Add value labels on bars
    for patch in ax.patches:
        height = patch.get_height()
        if height != 0:  # Don't show label for 0 (best model)
            ax.text(patch.get_x() + patch.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Only show the sample-size adjusted threshold - B&A thresholds are useless for large samples
    if threshold:
        ax.axhline(y=threshold, color='purple', linestyle='-', linewidth=2.5, 
                  label=f'Meaningful difference (n-adjusted: {threshold:.0f})', alpha=0.7)
        
        # Add text annotation for the threshold
        ax.text(0.02, threshold + max(delta_aics + delta_bics) * 0.02, 
               f'Sample-size threshold: {threshold:.0f}', 
               fontsize=10, fontweight='bold', color='purple',
               transform=ax.get_yaxis_transform())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Δ Information Criterion')
    
    # Update title based on whether we have a threshold
    if threshold:
        ax.set_title(f'Delta IC Values (Meaningful threshold: {threshold:.0f})')
    else:
        ax.set_title('Delta IC Values (Difference from Best)')
    
    ax.legend(loc='upper right', fontsize=9)
    # Rotate x-axis labels for better readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    ax.grid(True, alpha=0.3)


def _plot_percentage_improvement(ax, models):
    """Plot percentage improvement of full model over null model."""
    model_names = list(models.keys())
    
    # Get null and full model indices
    null_idx = model_names.index('null') if 'null' in model_names else None
    full_idx = model_names.index('full') if 'full' in model_names else None
    
    if null_idx is None or full_idx is None:
        ax.text(0.5, 0.5, 'Need both null and full models', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Improvement')
        ax.axis('off')
        return
    
    # Get AIC and BIC values
    null_aic = models[model_names[null_idx]].get('aic')
    full_aic = models[model_names[full_idx]].get('aic')
    null_bic = models[model_names[null_idx]].get('bic')
    full_bic = models[model_names[full_idx]].get('bic')
    
    # Calculate percentage improvements
    improvements = []
    labels = []
    colors = []
    
    if null_aic and full_aic:
        aic_improvement = ((null_aic - full_aic) / null_aic) * 100
        improvements.append(aic_improvement)
        labels.append(f'AIC\n{aic_improvement:.2f}%')
        colors.append('steelblue' if aic_improvement > 0 else 'lightcoral')
    
    if null_bic and full_bic:
        bic_improvement = ((null_bic - full_bic) / null_bic) * 100
        improvements.append(bic_improvement)
        labels.append(f'BIC\n{bic_improvement:.2f}%')
        colors.append('steelblue' if bic_improvement > 0 else 'lightcoral')
    
    if improvements:
        # Create DataFrame for seaborn
        import pandas as pd
        df = pd.DataFrame({
            'IC Type': ['AIC', 'BIC'][:len(improvements)],
            'Improvement (%)': improvements,
            'Color': colors
        })
        
        # Use seaborn barplot
        sns.barplot(data=df, x='IC Type', y='Improvement (%)', ax=ax,
                    palette=dict(zip(df['IC Type'], colors)), alpha=0.7)
        
        # Add edge color to bars
        for patch in ax.patches:
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        # Add a horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add value labels on bars
        for i, (patch, val) in enumerate(zip(ax.patches, improvements)):
            height = patch.get_height()
            ax.text(patch.get_x() + patch.get_width()/2., height,
                   f'{val:.1f}%', ha='center', 
                   va='bottom' if height > 0 else 'top', fontweight='bold')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Full Model Improvement over Null Model\n(Percentage Reduction in IC)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation
        avg_improvement = np.mean(improvements)
        if avg_improvement > 10:
            interpretation = f"✓ Strong improvement: {avg_improvement:.1f}%"
            color = 'green'
        elif avg_improvement > 5:
            interpretation = f"✓ Moderate improvement: {avg_improvement:.1f}%"
            color = 'orange'
        else:
            interpretation = f"⚠ Minimal improvement: {avg_improvement:.1f}%"
            color = 'red'
        
        ax.text(0.5, 0.95, interpretation, transform=ax.transAxes,
               ha='center', fontsize=11, fontweight='bold', color=color)
    else:
        ax.text(0.5, 0.5, 'No IC values available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Improvement')
        ax.axis('off')


def _plot_evidence_ratios(ax, models, evidence_ratios):
    """Plot evidence ratios - NO CALCULATIONS, just display."""
    model_names = list(models.keys())
    
    # Get pre-computed evidence ratios from results
    if not evidence_ratios:
        ax.text(0.5, 0.5, 'No evidence ratios available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Evidence Ratios')
        ax.axis('off')
        return
    
    # Extract ratios in same order as models
    ratios = [evidence_ratios.get(name, 1.0) for name in model_names]
    
    # Handle infinite values for plotting
    max_finite = max([r for r in ratios if r != float('inf')] + [1])
    plot_ratios = []
    labels = []
    for r in ratios:
        if r == float('inf'):
            plot_ratios.append(max_finite * 1000)  # Use large value for visualization
            labels.append('∞')
        else:
            plot_ratios.append(r)
            labels.append(f'{r:.1f}')
    
    # Create DataFrame for seaborn
    evidence_df = pd.DataFrame({
        'Model': model_names,
        'Evidence Ratio': plot_ratios,
        'Label': labels
    })
    
    # Plot on log scale using seaborn
    sns.barplot(data=evidence_df, y='Model', x='Evidence Ratio', ax=ax,
                color='teal', alpha=0.7, orient='h')
    
    # Add value labels
    for i, (patch, label) in enumerate(zip(ax.patches, labels)):
        ax.text(patch.get_width(), patch.get_y() + patch.get_height()/2, 
                label, va='center', fontsize=9)
    ax.set_xlabel('Evidence Ratio')
    ax.set_title('Evidence Ratios (vs Best Model)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')


def _plot_r_squared_values(ax, models, improvement=None):
    """Plot R² values comparison - McFadden's, Deviance-based, and Nagelkerke's."""
    model_names = list(models.keys())
    mcfadden_r2s = [m.get('mcfadden_r2', np.nan) for m in models.values()]
    deviance_r2s = [m.get('deviance_r2', np.nan) for m in models.values()]
    nagelkerke_r2s = [m.get('nagelkerke_r2', np.nan) for m in models.values()]
    
    # Check if we have any R² values
    if all(np.isnan(v) for v in mcfadden_r2s) and all(np.isnan(v) for v in deviance_r2s):
        ax.text(0.5, 0.5, 'No R² values available\n(requires loglik and deviance data)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Pseudo R² Comparison')
        ax.axis('off')
        return
    
    # Group by R² type
    r2_types = ['McFadden R²', 'Deviance R²', 'Nagelkerke R²']
    
    # Get null and full values (assuming standard naming)
    null_idx = model_names.index('null') if 'null' in model_names else 0
    full_idx = model_names.index('full') if 'full' in model_names else 1
    
    null_r2_values = [mcfadden_r2s[null_idx], deviance_r2s[null_idx], nagelkerke_r2s[null_idx]]
    full_r2_values = [mcfadden_r2s[full_idx], deviance_r2s[full_idx], nagelkerke_r2s[full_idx]]
    
    # Replace NaN with 0 for plotting (null model typically has R² = 0)
    null_r2_values = [0 if np.isnan(v) else v for v in null_r2_values]
    full_r2_values = [0 if np.isnan(v) else v for v in full_r2_values]
    
    # Create DataFrame for seaborn
    import pandas as pd
    data_list = []
    for r2_type, null_val, full_val in zip(r2_types, null_r2_values, full_r2_values):
        data_list.append({'R² Type': r2_type, 'Model': 'Null Model', 'Value': null_val})
        data_list.append({'R² Type': r2_type, 'Model': 'Full Model', 'Value': full_val})
    
    df = pd.DataFrame(data_list)
    
    # Use seaborn barplot
    sns.barplot(data=df, x='R² Type', y='Value', hue='Model', ax=ax,
                palette={'Null Model': 'lightcoral', 'Full Model': 'lightgreen'},
                alpha=0.8)
    
    # Add value labels on bars
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:  # Only show non-zero values
            ax.text(patch.get_x() + patch.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Removed "Excellent fit" text per user request - just show R² values in bars
    
    # Add improvement metrics if provided (difference between null and full models)
    if improvement:
        improvement_text = "Improvement (Full - Null):\n"
        if 'mcfadden_r2_pct' in improvement:
            improvement_text += f"McFadden: {improvement['mcfadden_r2_pct']:.2f}%\n"
        if 'deviance_r2_pct' in improvement:
            improvement_text += f"Deviance: {improvement['deviance_r2_pct']:.2f}%\n"
        if 'nagelkerke_r2_pct' in improvement:
            improvement_text += f"Nagelkerke: {improvement['nagelkerke_r2_pct']:.2f}%\n"
        improvement_text += "\n(Adding biotic predictors)"
        
        ax.text(0.02, 0.98, improvement_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    ax.set_xlabel('R² Type')
    ax.set_ylabel('R² Value')
    ax.set_title('Pseudo R² Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(full_r2_values), max(null_r2_values)) * 1.2)


def _plot_pseudo_r2_comparison(ax, models, nested_r2=None):
    """Plot pseudo R² comparison between null and full models."""
    model_names = list(models.keys())
    
    # Get null and full model indices
    null_idx = model_names.index('null') if 'null' in model_names else None
    full_idx = model_names.index('full') if 'full' in model_names else None
    
    if null_idx is None or full_idx is None:
        ax.text(0.5, 0.5, 'Need both null and full models', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Pseudo R² Comparison')
        ax.axis('off')
        return
    
    # Extract R² values (these compare to intercept-only model)
    null_mcfadden = models[model_names[null_idx]].get('mcfadden_r2', 0)
    full_mcfadden = models[model_names[full_idx]].get('mcfadden_r2', 0)
    null_deviance_r2 = models[model_names[null_idx]].get('deviance_r2', 0)
    full_deviance_r2 = models[model_names[full_idx]].get('deviance_r2', 0)
    null_nagelkerke = models[model_names[null_idx]].get('nagelkerke_r2', 0)
    full_nagelkerke = models[model_names[full_idx]].get('nagelkerke_r2', 0)
    
    # Create comparison data
    categories = ['McFadden R²', 'Deviance R²', 'Nagelkerke R²']
    null_values = [null_mcfadden, null_deviance_r2, null_nagelkerke]
    full_values = [full_mcfadden, full_deviance_r2, full_nagelkerke]
    
    # Create DataFrame for seaborn
    r2_data = []
    for cat, null_val, full_val in zip(categories, null_values, full_values):
        r2_data.append({'R² Type': cat, 'Model': 'Null Model', 'Value': null_val})
        r2_data.append({'R² Type': cat, 'Model': 'Full Model', 'Value': full_val})
    
    r2_df = pd.DataFrame(r2_data)
    
    # Create bars using seaborn
    sns.barplot(data=r2_df, x='R² Type', y='Value', hue='Model', ax=ax,
                palette={'Null Model': 'lightcoral', 'Full Model': 'lightgreen'},
                alpha=0.7)
    
    # Add edge color to bars
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    # Add value labels
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(patch.get_x() + patch.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Removed reference lines per user request
    
    ax.set_ylabel('Pseudo R² Value')
    ax.set_title('Pseudo R² Comparison')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(max(full_values), 0.5) * 1.1])
    
    # Calculate and display improvement
    improvement_mcfadden = full_mcfadden - null_mcfadden
    improvement_deviance = full_deviance_r2 - null_deviance_r2
    improvement_nagelkerke = full_nagelkerke - null_nagelkerke
    
    improvement_text = f"Improvement (Full - Null):\n"
    improvement_text += f"McFadden: {improvement_mcfadden:.4f} ({improvement_mcfadden*100:.2f}%)\n"
    improvement_text += f"Deviance: {improvement_deviance:.4f} ({improvement_deviance*100:.2f}%)\n"
    improvement_text += f"Nagelkerke: {improvement_nagelkerke:.4f} ({improvement_nagelkerke*100:.2f}%)\n"
    improvement_text += "\n(Adding biotic predictors)"
    
    ax.text(0.02, 0.98, improvement_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    # Removed fit interpretation text per user request


def _plot_summary_table(ax, comparison_results):
    """Plot summary table - NO CALCULATIONS, just display pre-computed table."""
    # Get pre-computed summary table from results
    summary_data = comparison_results.get('summary_table')
    
    if summary_data is None:
        ax.text(0.5, 0.5, 'No summary table available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Comparison Summary')
        ax.axis('off')
        return
    
    # Convert to DataFrame if it's a list of dicts
    if isinstance(summary_data, list):
        df = pd.DataFrame(summary_data)
    else:
        df = summary_data
    
    if df.empty:
        ax.text(0.5, 0.5, 'No summary table available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Comparison Summary')
        ax.axis('off')
        return
    
    # Format numeric columns for display only
    numeric_cols = ['AIC', 'ΔAIC', 'AIC Weight', 'BIC', 'ΔBIC', 'McFadden R²', 'Deviance R²']
    for col in numeric_cols:
        if col in df.columns:
            if 'R²' in col:
                # Special formatting for R² values (3 decimal places)
                df[col] = df[col].apply(lambda x: f'{x:.3f}' if not pd.isna(x) else '-')
            else:
                # Regular formatting for other numeric columns
                df[col] = df[col].apply(lambda x: f'{x:.2f}' if not pd.isna(x) else '-')
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    # Add threshold information if available
    threshold = comparison_results.get('threshold')
    n_obs = comparison_results.get('n_observations')
    
    if threshold and n_obs:
        # Format n_obs for display
        if n_obs >= 1_000_000:
            n_display = f"{n_obs/1_000_000:.0f}M"
        elif n_obs >= 1_000:
            n_display = f"{n_obs/1_000:.0f}K"
        else:
            n_display = str(n_obs)
        title_text = f'Model Comparison Summary\n(Meaningful ΔAIC threshold for n={n_display}: {threshold:.0f})'
    elif threshold:
        title_text = f'Model Comparison Summary\n(Meaningful ΔAIC threshold: {threshold:.0f})'
    else:
        title_text = 'Model Comparison Summary'
    
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)


