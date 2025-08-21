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
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    models = comparison_results['models']
    weights = comparison_results.get('weights', {})
    best_model = comparison_results.get('best_model', {})
    
    # 1. AIC/BIC Values (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_ic_values(ax1, models)
    
    # 2. Delta IC Values (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_delta_ic(ax2, models, comparison_results.get('threshold'))
    
    # 3. Model Weights (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_weights(ax3, models, weights)
    
    # 4. Evidence Ratios (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    _plot_evidence_ratios(ax4, models, comparison_results.get('evidence_ratios', {}))
    
    # 5. Summary Table (bottom middle and right)
    ax5 = fig.add_subplot(gs[1, 1:])
    _plot_summary_table(ax5, comparison_results)
    
    # Add main title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return fig


def _plot_ic_values(ax, models):
    """Plot AIC and BIC values grouped by IC type."""
    model_names = list(models.keys())
    aics = [m.get('aic', np.nan) for m in models.values()]
    bics = [m.get('bic', np.nan) for m in models.values()]
    
    # Group by IC type: (null_aic, full_aic), (null_bic, full_bic)
    ic_types = ['AIC', 'BIC']
    x = np.arange(len(ic_types))
    width = 0.35
    
    # Get null and full values
    null_idx = model_names.index('null') if 'null' in model_names else 0
    full_idx = model_names.index('full') if 'full' in model_names else 1
    
    null_values = [aics[null_idx], bics[null_idx]]
    full_values = [aics[full_idx], bics[full_idx]]
    
    # Plot bars
    bars1 = ax.bar(x - width/2, null_values, width, label='Null Model', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, full_values, width, label='Full Model', color='lightgreen', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Add improvement indicators
    for i, ic_type in enumerate(ic_types):
        null_val = null_values[i]
        full_val = full_values[i]
        if not np.isnan(null_val) and not np.isnan(full_val):
            if full_val < null_val:  # Lower is better
                # Show improvement
                improvement = null_val - full_val
                ax.text(i, max(null_val, full_val) * 1.01, 
                       f'Δ={improvement:.0f}',
                       ha='center', fontsize=9, fontweight='bold', color='green')
    
    ax.set_xlabel('Information Criterion Type')
    ax.set_ylabel('IC Value')
    ax.set_title('Model Comparison: AIC and BIC')
    ax.set_xticks(x)
    ax.set_xticklabels(ic_types)
    ax.legend()
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


def _plot_weights(ax, models, weights):
    """Plot model weights (Akaike weights)."""
    model_names = list(models.keys())
    
    if weights:
        aic_weights = [weights.get(name, {}).get('aic_weight', 0) for name in model_names]
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        wedges, texts, autotexts = ax.pie(aic_weights, labels=model_names, 
                                           colors=colors, autopct='%1.1f%%',
                                           startangle=90)
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title('Model Weights (AIC)')
    else:
        ax.text(0.5, 0.5, 'No weights available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Weights')
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
    numeric_cols = ['AIC', 'ΔAIC', 'AIC Weight', 'BIC', 'ΔBIC']
    for col in numeric_cols:
        if col in df.columns:
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


