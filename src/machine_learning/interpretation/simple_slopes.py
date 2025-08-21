"""
Simple slopes analysis for continuous interactions.

This module provides interpretation and visualization tools for continuous × continuous
interaction effects using emtrends from the emmeans package.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from ..evaluation.emmeans_julia import JuliaEMmeans

logger = logging.getLogger(__name__)


class SimpleSlopesInterpretation:
    """
    Interpret and visualize simple slopes for continuous interactions.
    
    This is a descriptive/visualization tool, not a hypothesis test.
    Uses emtrends to compute slopes of focal variable at different moderator values.
    """
    
    def __init__(self, 
                 emmeans_wrapper: JuliaEMmeans,
                 focal_var: str,
                 moderator_var: str,
                 interaction_term: str):
        """
        Initialize simple slopes interpretation.
        
        Args:
            emmeans_wrapper: JuliaEMmeans wrapper with fitted model
            focal_var: Name of focal variable (e.g., 'plants_richness_log')
            moderator_var: Name of moderator variable (e.g., 'bio04_std')
            interaction_term: Name of interaction term (e.g., 'bio04_std_x_plants_log')
        """
        self.emmeans_wrapper = emmeans_wrapper
        self.focal_var = focal_var
        self.moderator_var = moderator_var
        self.interaction_term = interaction_term
        
        logger.info(f"Initialized simple slopes for {focal_var} × {moderator_var}")
    
    def compute_slopes(self, 
                      moderator_values: Optional[List[float]] = None,
                      n_values: int = 3) -> pd.DataFrame:
        """
        Compute simple slopes using emtrends.
        
        Args:
            moderator_values: Specific moderator values (default: -1SD, mean, +1SD)
            n_values: Number of values if not specified (default: 3)
            
        Returns:
            DataFrame with slopes, SEs, and CIs at each moderator value
        """
        if moderator_values is None:
            # Default to -1SD, mean, +1SD for standardized variables
            if n_values == 3:
                moderator_values = [-1.0, 0.0, 1.0]
            else:
                # Create evenly spaced values
                moderator_values = np.linspace(-2, 2, n_values).tolist()
        
        logger.info(f"Computing slopes at {self.moderator_var} = {moderator_values}")
        
        # MUST use emtrends - no fallbacks allowed
        slopes_df = self.emmeans_wrapper.compute_emtrends(
            var=self.focal_var,
            at={self.moderator_var: moderator_values},
            type='response'
        )
        
        # Add interpretation labels
        if len(moderator_values) == 3:
            slopes_df['label'] = ['Low (-1 SD)', 'Mean', 'High (+1 SD)']
        else:
            slopes_df['label'] = [f"{self.moderator_var} = {v:.2f}" for v in moderator_values]
        
        return slopes_df
    
    def _compute_slopes_manual(self, moderator_values: List[float]) -> pd.DataFrame:
        """Manual slope calculation as fallback."""
        results = []
        
        # Get coefficients
        coef_dict = dict(zip(self.emmeans_wrapper.coef_names, 
                           self.emmeans_wrapper.coefficients))
        
        # Get variance-covariance matrix
        vcov = self.emmeans_wrapper.vcov
        
        # Find indices
        focal_idx = self.emmeans_wrapper.coef_names.index(self.focal_var)
        interaction_idx = self.emmeans_wrapper.coef_names.index(self.interaction_term)
        
        for mod_val in moderator_values:
            # Simple slope = β_focal + β_interaction * moderator_value
            slope = coef_dict[self.focal_var] + coef_dict[self.interaction_term] * mod_val
            
            # SE includes covariance terms
            var_focal = vcov[focal_idx, focal_idx]
            var_interaction = vcov[interaction_idx, interaction_idx]
            cov_focal_interaction = vcov[focal_idx, interaction_idx]
            
            se = np.sqrt(var_focal + 
                        mod_val**2 * var_interaction + 
                        2 * mod_val * cov_focal_interaction)
            
            # 95% CI
            ci_lower = slope - 1.96 * se
            ci_upper = slope + 1.96 * se
            
            results.append({
                self.moderator_var: mod_val,
                'slope': slope,
                'SE': se,
                'lower.CL': ci_lower,
                'upper.CL': ci_upper
            })
        
        return pd.DataFrame(results)
    
    def visualize_slopes(self, 
                        slopes_df: pd.DataFrame,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create visualization of simple slopes.
        
        Creates two plots:
        1. Slope estimates with confidence intervals
        2. Regression lines at different moderator values
        
        Args:
            slopes_df: DataFrame from compute_slopes
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Slope estimates with CIs
        moderator_col = self.moderator_var
        if moderator_col not in slopes_df.columns:
            # Find the moderator column (might have different name)
            for col in slopes_df.columns:
                if self.moderator_var in col:
                    moderator_col = col
                    break
        
        x_vals = slopes_df[moderator_col] if moderator_col in slopes_df.columns else range(len(slopes_df))
        
        # Extract slope and CI columns
        slope_col = 'slope' if 'slope' in slopes_df.columns else slopes_df.columns[1]
        lower_col = 'lower.CL' if 'lower.CL' in slopes_df.columns else slopes_df.columns[-2]
        upper_col = 'upper.CL' if 'upper.CL' in slopes_df.columns else slopes_df.columns[-1]
        
        ax1.errorbar(x_vals, 
                    slopes_df[slope_col],
                    yerr=[slopes_df[slope_col] - slopes_df[lower_col],
                          slopes_df[upper_col] - slopes_df[slope_col]],
                    fmt='o-', markersize=8, capsize=5, capthick=2,
                    color='steelblue', ecolor='gray')
        
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel(f'{self.moderator_var} (Standardized)', fontsize=11)
        ax1.set_ylabel(f'Simple Slope of {self.focal_var}', fontsize=11)
        ax1.set_title('Simple Slopes Analysis', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add significance regions
        for i, row in slopes_df.iterrows():
            if row[lower_col] > 0 or row[upper_col] < 0:
                # Significant (CI doesn't include 0)
                ax1.scatter(x_vals[i] if hasattr(x_vals, '__getitem__') else x_vals.iloc[i],
                          row[slope_col], s=200, alpha=0.3, color='green')
        
        # Plot 2: Regression lines at different moderator values
        x_range = np.linspace(-2, 2, 100)  # Range for focal variable
        
        # Get coefficients
        coef_dict = dict(zip(self.emmeans_wrapper.coef_names, 
                           self.emmeans_wrapper.coefficients))
        intercept = coef_dict.get('(Intercept)', 0)
        
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(slopes_df)))
        
        for i, row in slopes_df.iterrows():
            mod_val = row[moderator_col] if moderator_col in row else x_vals[i]
            slope = row[slope_col]
            
            # y = intercept + slope * focal + moderator_effect * moderator_value
            y_vals = intercept + slope * x_range + coef_dict.get(self.moderator_var, 0) * mod_val
            
            label = row['label'] if 'label' in row else f'{self.moderator_var} = {mod_val:.2f}'
            ax2.plot(x_range, y_vals, color=colors[i], linewidth=2, label=label)
        
        ax2.set_xlabel(f'{self.focal_var} (Standardized)', fontsize=11)
        ax2.set_ylabel('Predicted Outcome', fontsize=11)
        ax2.set_title('Interaction Visualization', fontsize=12, fontweight='bold')
        ax2.legend(title=self.moderator_var, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Simple Slopes: Effect of {self.focal_var} at Different {self.moderator_var} Values',
                    fontsize=13, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved simple slopes visualization to {save_path}")
        
        return fig
    
    def generate_interpretation(self, slopes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate interpretation of simple slopes results.
        
        Args:
            slopes_df: DataFrame from compute_slopes
            
        Returns:
            Dictionary with interpretation details
        """
        interpretation = {
            'focal_variable': self.focal_var,
            'moderator_variable': self.moderator_var,
            'slopes': [],
            'pattern': None,
            'summary': None
        }
        
        # Extract slope information
        slope_col = 'slope' if 'slope' in slopes_df.columns else slopes_df.columns[1]
        lower_col = 'lower.CL' if 'lower.CL' in slopes_df.columns else slopes_df.columns[-2]
        upper_col = 'upper.CL' if 'upper.CL' in slopes_df.columns else slopes_df.columns[-1]
        
        for _, row in slopes_df.iterrows():
            slope_info = {
                'moderator_value': row[self.moderator_var] if self.moderator_var in row else row.iloc[0],
                'slope': row[slope_col],
                'se': row.get('SE', row.get('std.error', np.nan)),
                'ci': [row[lower_col], row[upper_col]],
                'significant': row[lower_col] > 0 or row[upper_col] < 0,
                'label': row.get('label', '')
            }
            interpretation['slopes'].append(slope_info)
        
        # Determine interaction pattern
        slopes = [s['slope'] for s in interpretation['slopes']]
        if len(slopes) >= 3:
            if slopes[0] < slopes[1] < slopes[2]:
                interpretation['pattern'] = 'enhancing'
                interpretation['summary'] = (
                    f"The effect of {self.focal_var} increases as {self.moderator_var} increases. "
                    f"This suggests an enhancing interaction where higher {self.moderator_var} "
                    f"strengthens the positive effect of {self.focal_var}."
                )
            elif slopes[0] > slopes[1] > slopes[2]:
                interpretation['pattern'] = 'buffering'
                interpretation['summary'] = (
                    f"The effect of {self.focal_var} decreases as {self.moderator_var} increases. "
                    f"This suggests a buffering interaction where higher {self.moderator_var} "
                    f"weakens the effect of {self.focal_var}."
                )
            else:
                interpretation['pattern'] = 'crossover'
                interpretation['summary'] = (
                    f"The effect of {self.focal_var} changes direction across {self.moderator_var} values. "
                    f"This suggests a crossover interaction with qualitatively different effects."
                )
        
        return interpretation


def analyze_simple_slopes(julia_model: Dict[str, Any],
                         focal_var: str,
                         moderator_var: str,
                         interaction_term: str,
                         feature_names: List[str],
                         save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Wrapper function for simple slopes analysis.
    
    Args:
        julia_model: Julia model results dictionary
        focal_var: Name of focal variable
        moderator_var: Name of moderator variable
        interaction_term: Name of interaction term
        feature_names: List of all feature names
        save_path: Optional path to save visualization
        
    Returns:
        Dictionary with slopes, interpretation, and figure
    """
    # Create emmeans wrapper
    emmeans_wrapper = JuliaEMmeans(julia_model, feature_names)
    
    # Create interpreter
    interpreter = SimpleSlopesInterpretation(
        emmeans_wrapper=emmeans_wrapper,
        focal_var=focal_var,
        moderator_var=moderator_var,
        interaction_term=interaction_term
    )
    
    # Compute slopes
    slopes_df = interpreter.compute_slopes()
    
    # Generate visualization
    fig = interpreter.visualize_slopes(slopes_df, save_path)
    
    # Generate interpretation
    interpretation = interpreter.generate_interpretation(slopes_df)
    
    return {
        'slopes': slopes_df,
        'interpretation': interpretation,
        'figure': fig
    }