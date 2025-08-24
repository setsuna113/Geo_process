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

logger = logging.getLogger(__name__)

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
    _plot_r_squared_values(ax2, models)
    
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
    x = np.arange(len(ic_types))
    width = 0.25  # Narrower bars for 3 per group
    
    # Get null and full values
    null_idx = model_names.index('null') if 'null' in model_names else 0
    full_idx = model_names.index('full') if 'full' in model_names else 1
    
    null_values = [aics[null_idx], bics[null_idx]]
    full_values = [aics[full_idx], bics[full_idx]]
    
    # Calculate percentage improvements
    improvements = []
    for null_val, full_val in zip(null_values, full_values):
        if not np.isnan(null_val) and not np.isnan(full_val) and null_val != 0:
            improvement = ((null_val - full_val) / null_val) * 100
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    # Plot three groups of bars
    bars1 = ax.bar(x - width, null_values, width, label='Null Model', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x, full_values, width, label='Full Model', color='lightgreen', alpha=0.8)
    bars3 = ax.bar(x + width, improvements, width, label='% Improvement', color='gold', alpha=0.8)
    
    # Add value labels for AIC/BIC values (in scientific notation for readability)
    for bars, values in [(bars1, null_values), (bars2, full_values)]:
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.1e}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Add percentage labels
    for bar, val in zip(bars3, improvements):
        if val != 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Set up dual y-axis for percentage values
    ax2 = ax.twinx()
    ax2.set_ylabel('Improvement (%)', color='orange')
    ax2.set_ylim(0, max(improvements) * 1.2 if improvements else 5)
    ax2.tick_params(axis='y', labelcolor='orange')
    
    ax.set_xlabel('Information Criterion Type')
    ax.set_ylabel('IC Value')
    ax.set_title('Model Comparison: AIC, BIC, and % Improvements')
    ax.set_xticks(x)
    ax.set_xticklabels(ic_types)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)


def _plot_delta_ic(ax, models, threshold=None):
    """Plot delta AIC and delta BIC with sample-size adjusted threshold."""
    model_names = list(models.keys())
    delta_aics = [m.get('delta_aic', 0) for m in models.values()]
    delta_bics = [m.get('delta_bic', 0) for m in models.values()]
    
    # Filter out NaN values and ensure we have data
    valid_data = [(name, daic, dbic) for name, daic, dbic in zip(model_names, delta_aics, delta_bics)
                  if not (np.isnan(daic) or daic is None)]
    
    if not valid_data:
        ax.text(0.5, 0.5, 'No delta IC values available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Delta IC Values')
        ax.axis('off')
        return
    
    names, daic_vals, dbic_vals = zip(*valid_data)
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, daic_vals, width, label='ΔAIC', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, dbic_vals, width, label='ΔBIC', color='coral', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height != 0:  # Don't show label for 0 (best model)
                ax.text(bar.get_x() + bar.get_width()/2., height,
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
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
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
        x = np.arange(len(improvements))
        bars = ax.bar(x, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add a horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', 
                   va='bottom' if height > 0 else 'top', fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['AIC', 'BIC'])
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
    
    # Plot on log scale
    y_pos = np.arange(len(model_names))
    bars = ax.barh(y_pos, plot_ratios, color='teal', alpha=0.7)
    
    # Add value labels
    for i, (bar, label) in enumerate(zip(bars, labels)):
        ax.text(bar.get_width(), i, label, va='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Evidence Ratio')
    ax.set_title('Evidence Ratios (vs Best Model)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')


def _plot_r_squared_values(ax, models):
    """Plot R² values comparison - McFadden's and Deviance-based."""
    model_names = list(models.keys())
    mcfadden_r2s = [m.get('mcfadden_r2', np.nan) for m in models.values()]
    deviance_r2s = [m.get('deviance_r2', np.nan) for m in models.values()]
    
    # Check if we have any R² values
    if all(np.isnan(mcfadden_r2s)) and all(np.isnan(deviance_r2s)):
        ax.text(0.5, 0.5, 'No R² values available\n(requires loglik and deviance data)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Pseudo R² Comparison')
        ax.axis('off')
        return
    
    # Group by R² type
    r2_types = ['McFadden R²', 'Deviance R²']
    x = np.arange(len(r2_types))
    width = 0.35
    
    # Get null and full values (assuming standard naming)
    null_idx = model_names.index('null') if 'null' in model_names else 0
    full_idx = model_names.index('full') if 'full' in model_names else 1
    
    null_r2_values = [mcfadden_r2s[null_idx], deviance_r2s[null_idx]]
    full_r2_values = [mcfadden_r2s[full_idx], deviance_r2s[full_idx]]
    
    # Replace NaN with 0 for plotting (null model typically has R² = 0)
    null_r2_values = [0 if np.isnan(v) else v for v in null_r2_values]
    full_r2_values = [0 if np.isnan(v) else v for v in full_r2_values]
    
    # Plot bars
    bars1 = ax.bar(x - width/2, null_r2_values, width, label='Null Model', 
                  color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, full_r2_values, width, label='Full Model', 
                  color='lightgreen', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show non-zero values
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add interpretation text
    max_r2 = max(full_r2_values) if full_r2_values else 0
    if max_r2 > 0:
        if max_r2 >= 0.4:
            interpretation = "Excellent fit"
            color = 'green'
        elif max_r2 >= 0.2:
            interpretation = "Good fit"
            color = 'orange'
        elif max_r2 >= 0.1:
            interpretation = "Moderate fit"
            color = 'blue'
        else:
            interpretation = "Weak fit"
            color = 'red'
        
        ax.text(0.98, 0.95, f'Full model: {interpretation}\n(R² = {max_r2:.3f})',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
               fontsize=10, fontweight='bold')
    
    ax.set_xlabel('R² Type')
    ax.set_ylabel('R² Value')
    ax.set_title('Pseudo R² Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(r2_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(full_r2_values), max(null_r2_values)) * 1.2)


def _plot_pseudo_r2_comparison(ax, models):
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
    
    # Extract R² values
    null_mcfadden = models[model_names[null_idx]].get('mcfadden_r2', 0)
    full_mcfadden = models[model_names[full_idx]].get('mcfadden_r2', 0)
    null_deviance_r2 = models[model_names[null_idx]].get('deviance_r2', 0)
    full_deviance_r2 = models[model_names[full_idx]].get('deviance_r2', 0)
    
    # Create comparison data
    categories = ['McFadden R²', 'Deviance R²']
    null_values = [null_mcfadden, null_deviance_r2]
    full_values = [full_mcfadden, full_deviance_r2]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, null_values, width, label='Null Model', 
                  color='lightcoral', alpha=0.7, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, full_values, width, label='Full Model', 
                  color='lightgreen', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add reference lines for interpretation
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, 
              label='Good fit (0.2)', linewidth=1)
    ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.5, 
              label='Excellent fit (0.4)', linewidth=1)
    
    ax.set_ylabel('Pseudo R² Value')
    ax.set_title('Pseudo R² Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(max(full_values), 0.5) * 1.1])
    
    # Add interpretation
    avg_full_r2 = np.mean(full_values)
    if avg_full_r2 > 0.4:
        interpretation = f"✓ Excellent fit (avg R² = {avg_full_r2:.3f})"
        color = 'darkgreen'
    elif avg_full_r2 > 0.2:
        interpretation = f"✓ Good fit (avg R² = {avg_full_r2:.3f})"
        color = 'green'
    elif avg_full_r2 > 0.1:
        interpretation = f"⚠ Moderate fit (avg R² = {avg_full_r2:.3f})"
        color = 'orange'
    else:
        interpretation = f"⚠ Poor fit (avg R² = {avg_full_r2:.3f})"
        color = 'red'
    
    ax.text(0.98, 0.95, interpretation, transform=ax.transAxes,
           ha='right', fontsize=10, fontweight='bold', color=color,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


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


