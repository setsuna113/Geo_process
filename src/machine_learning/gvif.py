from src.abstractions import get_logger
"""Simplified GVIF multicollinearity diagnostics and essential plots.

Extracted from dharma_diagnostics.py and diagnostic_plots.py.
Provides GVIF calculation interpretation and visualization.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Union
from scipy import stats

logger = get_logger(__name__)


def check_multicollinearity_gvif(gvif_values: Dict[str, float]) -> Dict[str, Any]:
    """Check multicollinearity using GVIF values from Julia.
    
    Extracted from dharma_diagnostics.py _check_multicollinearity_gvif()
    
    Args:
        gvif_values: Dictionary mapping variable names to GVIF values
        
    Returns:
        Dictionary with multicollinearity assessment
    """
    issues = []
    problematic_vars = []
    
    for var_name, gvif in gvif_values.items():
        # For single df predictors, use GVIF^(1/2df) where df=1
        adjusted_gvif = np.sqrt(gvif)
        
        if adjusted_gvif > 2:
            issues.append(f"{var_name}: GVIF^0.5 = {adjusted_gvif:.2f} (>2 suggests collinearity)")
            problematic_vars.append(var_name)
        
        if adjusted_gvif > 5:
            issues.append(f"{var_name}: SEVERE multicollinearity (GVIF^0.5 = {adjusted_gvif:.2f})")
    
    max_gvif = max(gvif_values.values()) if gvif_values else 0
    
    return {
        'max_gvif': max_gvif,
        'max_adjusted_gvif': np.sqrt(max_gvif),
        'problematic_variables': problematic_vars,
        'issues': issues,
        'has_multicollinearity': len(problematic_vars) > 0,
        'interpretation': interpret_gvif(np.sqrt(max_gvif))
    }


def interpret_gvif(adjusted_gvif: float) -> str:
    """Interpret adjusted GVIF value.
    
    Extracted from dharma_diagnostics.py _interpret_gvif()
    """
    if adjusted_gvif < 1.5:
        return "No multicollinearity"
    elif adjusted_gvif < 2:
        return "Low multicollinearity (acceptable)"
    elif adjusted_gvif < 5:
        return "Moderate multicollinearity (monitor coefficients)"
    else:
        return "Severe multicollinearity (consider removing predictors)"


def plot_gvif_only(model_results: Dict[str, Any],
                   output_dir: Optional[Path] = None,
                   filename: str = "gvif_diagnostics.png") -> Optional[Path]:
    """Create GVIF-only diagnostic plot for multicollinearity assessment.
    
    Extracted from diagnostic_plots.py plot_gvif_only()
    
    This is a simplified version that only shows GVIF values, which is
    the most important diagnostic for GLM models.
    
    Args:
        model_results: Model results containing GVIF values
        output_dir: Directory to save plot
        filename: Name of output file
        
    Returns:
        Path to saved plot or None
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract GVIF data
        gvif_data = model_results.get('gvif', {})
        
        # Handle different GVIF data formats
        if isinstance(gvif_data, list):
            # If GVIF is a list, map to coefficient names (excluding intercept)
            coef_names = model_results.get('coef_names', [])
            if len(gvif_data) == len(coef_names) - 1:  # Exclude intercept
                gvif_dict = {}
                for name, value in zip(coef_names[1:], gvif_data):
                    gvif_dict[name] = value
                gvif_data = gvif_dict
            else:
                # Try to match by position
                gvif_dict = {}
                for i, value in enumerate(gvif_data):
                    name = coef_names[i+1] if i+1 < len(coef_names) else f"var_{i+1}"
                    gvif_dict[name] = value
                gvif_data = gvif_dict
        
        if not gvif_data or not isinstance(gvif_data, dict):
            ax.text(0.5, 0.5, 'GVIF values not available', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Multicollinearity Check (GVIF)', fontsize=16, fontweight='bold')
            ax.axis('off')
        else:
            # Extract variables and GVIF values
            vars = list(gvif_data.keys())
            gvif_values = list(gvif_data.values())
            
            # Calculate adjusted GVIF (square root for single df predictors)
            adjusted_gvif = [np.sqrt(v) for v in gvif_values]
            
            # Color coding based on severity
            colors = []
            for v in adjusted_gvif:
                if v > 2:
                    colors.append('#d62728')  # Red - problematic
                elif v > 1.5:
                    colors.append('#ff7f0e')  # Orange - moderate concern
                else:
                    colors.append('#2ca02c')  # Green - no concern
            
            # Create DataFrame for seaborn
            gvif_df = pd.DataFrame({
                'Variable': vars,
                'Adjusted_GVIF': adjusted_gvif,
                'Color': colors
            })
            
            # Use seaborn barplot (horizontal)
            sns.barplot(data=gvif_df, y='Variable', x='Adjusted_GVIF', ax=ax,
                       palette=dict(zip(vars, colors)), alpha=0.7, orient='h')
            
            # Add edge color to bars
            for patch in ax.patches:
                patch.set_edgecolor('black')
                patch.set_linewidth(1)
            
            # Add threshold lines
            ax.axvline(x=1.5, color='#ff7f0e', linestyle='--', alpha=0.5, 
                      label='Low concern (√GVIF = 1.5)', linewidth=2)
            ax.axvline(x=2.0, color='#d62728', linestyle='--', alpha=0.5, 
                      label='High concern (√GVIF = 2.0)', linewidth=2)
            
            # Customize axes
            ax.tick_params(axis='y', labelsize=11)
            ax.set_xlabel('√GVIF (Adjusted GVIF)', fontsize=12, fontweight='bold')
            ax.set_title('Generalized Variance Inflation Factor (GVIF) - Multicollinearity Check', 
                       fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for i, (var, val, orig_val) in enumerate(zip(vars, adjusted_gvif, gvif_values)):
                # Get the bar patch for this variable
                bar = ax.patches[i] if i < len(ax.patches) else None
                if bar:
                    label_x = bar.get_width() + 0.02
                    ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                           f'{val:.2f}\n(GVIF={orig_val:.2f})', 
                           va='center', fontsize=9)
            
            # Add legend
            ax.legend(loc='upper right', framealpha=0.9)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            
            # Add sample size info if available
            n_obs = model_results.get('n_obs', 0)
            n_params = len(model_results.get('coefficients', []))
            if n_obs > 0:
                sample_info = f"Sample size: {n_obs:,} obs, {n_params} params"
                obs_per_param = n_obs / n_params if n_params > 0 else 0
                sample_info += f"\n{obs_per_param:.0f} obs/param"
                
                ax.text(0.02, 0.98, sample_info, transform=ax.transAxes,
                       va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            save_path = output_dir / filename
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"GVIF diagnostics saved to {save_path}")
            return save_path
        else:
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create GVIF diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_coefficient_estimates(model_results: Dict[str, Any],
                              output_dir: Optional[Path] = None,
                              filename: str = "coefficient_estimates.png") -> Optional[Path]:
    """Create coefficient plot with confidence intervals.
    
    Extracted from diagnostic_plots.py plot_coefficient_estimates()
    
    Shows coefficient estimates with 95% CI for statistical inference.
    This is the traditional statistical view of model parameters.
    
    Args:
        model_results: Model results with coefficients, std_errors, coef_names
        output_dir: Directory to save plot
        filename: Name of output file
        
    Returns:
        Path to saved plot or Figure object
    """
    try:
        # Extract coefficient information
        coefs = model_results.get('coefficients', [])
        std_errors = model_results.get('std_errors', [])
        coef_names = model_results.get('coef_names', 
                                     [f"coef_{i}" for i in range(len(coefs))])
        
        if len(coefs) == 0 or len(std_errors) == 0:
            logger.warning("No coefficients or standard errors available")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(coefs) * 0.5)))
        
        # Calculate z-scores and p-values if available
        z_scores = np.array(coefs) / np.array(std_errors)
        p_values = model_results.get('coef_p_values', 
                                    2 * (1 - stats.norm.cdf(np.abs(z_scores))))
        
        # Prepare data for plotting
        y_pos = np.arange(len(coefs))
        
        # Color code by significance
        colors = []
        for i, p in enumerate(p_values):
            if p < 0.001:
                colors.append('darkgreen')  # Highly significant
            elif p < 0.01:
                colors.append('green')       # Very significant
            elif p < 0.05:
                colors.append('orange')      # Significant
            else:
                colors.append('gray')        # Not significant
        
        # Create DataFrame for seaborn
        coef_df = pd.DataFrame({
            'Variable': coef_names,
            'Coefficient': coefs,
            'CI_Lower': [c - 1.96*se for c, se in zip(coefs, std_errors)],
            'CI_Upper': [c + 1.96*se for c, se in zip(coefs, std_errors)],
            'y_pos': y_pos,
            'Color': colors,
            'p_value': p_values
        })
        
        # Plot coefficients as points with error bars
        for i, row in coef_df.iterrows():
            ax.errorbar(row['Coefficient'], row['y_pos'], 
                      xerr=[[row['Coefficient'] - row['CI_Lower']], 
                            [row['CI_Upper'] - row['Coefficient']]],
                      fmt='o', capsize=5, color=row['Color'], markersize=8,
                      label=f'p<0.001' if i == 0 and row['Color'] == 'darkgreen' else '')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(coef_names)
        ax.set_xlabel('Coefficient Value (95% CI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variables', fontsize=12, fontweight='bold')
        ax.set_title('Coefficient Estimates with 95% Confidence Intervals', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add significance legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='p < 0.001'),
            Patch(facecolor='green', label='p < 0.01'),
            Patch(facecolor='orange', label='p < 0.05'),
            Patch(facecolor='gray', label='p ≥ 0.05')
        ]
        ax.legend(handles=legend_elements, loc='best', title='Significance')
        
        # Add text annotations for exact values
        for i, (coef, stderr, p) in enumerate(zip(coefs, std_errors, p_values)):
            # Add coefficient value
            x_offset = max(abs(min(coefs)), abs(max(coefs))) * 0.02
            ax.text(coef + x_offset, y_pos[i], 
                   f'{coef:.3f}', 
                   va='center', fontsize=9)
            
            # Add significance stars
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = ''
            
            if stars:
                ax.text(coef + 1.96*stderr + x_offset, y_pos[i], 
                       stars, va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            save_path = output_dir / filename
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Coefficient estimates plot saved to {save_path}")
            return save_path
        else:
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create coefficient plot: {e}")
        return None


def test_overdispersion(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Test for overdispersion using deviance and residual df.
    
    Extracted from dharma_diagnostics.py _test_overdispersion()
    
    Returns:
        Dictionary with overdispersion test results
    """
    try:
        deviance = model_results.get('deviance', 0)
        df_residual = model_results.get('df_residual', 1)
        
        # Dispersion parameter estimate
        dispersion_ratio = deviance / df_residual if df_residual > 0 else np.inf
        
        # For negative binomial, dispersion > 1 is expected
        # Values > 2 suggest model may still be inadequate
        is_overdispersed = dispersion_ratio > 2.0
        
        # Chi-square test for overdispersion
        p_value = 1 - stats.chi2.cdf(deviance, df_residual) if df_residual > 0 else 0
        
        return {
            'dispersion_ratio': dispersion_ratio,
            'is_overdispersed': is_overdispersed,
            'deviance': deviance,
            'df_residual': df_residual,
            'p_value': p_value,
            'interpretation': interpret_dispersion(dispersion_ratio)
        }
        
    except Exception as e:
        logger.warning(f"Overdispersion test failed: {e}")
        return {
            'dispersion_ratio': None,
            'is_overdispersed': None,
            'error': str(e)
        }


def interpret_dispersion(ratio: float) -> str:
    """Interpret dispersion ratio.
    
    Extracted from dharma_diagnostics.py _interpret_dispersion()
    
    Args:
        ratio: Dispersion ratio (deviance/df_residual)
        
    Returns:
        String interpretation
    """
    if ratio < 0.5:
        return "Underdispersion detected (may indicate model issues)"
    elif ratio < 1.2:
        return "Good dispersion (near 1.0)"
    elif ratio < 2.0:
        return "Mild overdispersion (acceptable for NB model)"
    elif ratio < 5.0:
        return "Moderate overdispersion (model adequate but monitor)"
    else:
        return "Severe overdispersion (consider model revision)"


def check_sample_size(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Check if sample size is adequate for model complexity.
    
    Extracted from dharma_diagnostics.py _check_sample_size()
    
    Returns:
        Dictionary with sample size assessment
    """
    n_obs = model_results.get('n_obs', 0)
    n_parameters = model_results.get('n_parameters', 0)
    df_residual = model_results.get('df_residual', 0)
    
    # Rule of thumb: at least 10-20 observations per parameter
    min_obs_per_param = 10
    recommended_obs = n_parameters * min_obs_per_param
    
    results = {
        'n_obs': n_obs,
        'n_parameters': n_parameters,
        'df_residual': df_residual,
        'obs_per_parameter': n_obs / n_parameters if n_parameters > 0 else np.inf,
        'recommended_min_obs': recommended_obs,
        'adequate': n_obs >= recommended_obs
    }
    
    if not results['adequate']:
        results['message'] = f"Sample size ({n_obs}) may be too small for {n_parameters} parameters"
    else:
        results['message'] = f"Sample size adequate ({n_obs} obs for {n_parameters} parameters)"
    
    return results