from src.abstractions import get_logger
"""
Marginal means analysis for categorical interactions.

This module provides interpretation and visualization tools for categorical × categorical
interaction effects using emmeans from the R emmeans package.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from ..evaluation.emmeans_julia import JuliaEMmeans

logger = get_logger(__name__)


class MarginalMeansInterpretation:
    """
    Interpret and visualize marginal means for categorical interactions.
    
    This is a descriptive/visualization tool, not a hypothesis test.
    Uses emmeans to compute estimated marginal means for factorial designs.
    """
    
    def __init__(self,
                 emmeans_wrapper: JuliaEMmeans,
                 factor1: str,
                 factor2: str,
                 factor1_levels: List = [0, 1],
                 factor2_levels: List = [0, 1],
                 factor1_labels: Optional[List[str]] = None,
                 factor2_labels: Optional[List[str]] = None):
        """
        Initialize marginal means interpretation.
        
        Args:
            emmeans_wrapper: JuliaEMmeans wrapper with fitted model
            factor1: Name of first factor (e.g., 'temperate')
            factor2: Name of second factor (e.g., 'plant_high')
            factor1_levels: Levels of factor1 (default: [0, 1])
            factor2_levels: Levels of factor2 (default: [0, 1])
            factor1_labels: Labels for factor1 levels
            factor2_labels: Labels for factor2 levels
        """
        self.emmeans_wrapper = emmeans_wrapper
        self.factor1 = factor1
        self.factor2 = factor2
        self.factor1_levels = factor1_levels
        self.factor2_levels = factor2_levels
        
        # Default labels if not provided
        self.factor1_labels = factor1_labels or [f"{factor1}={v}" for v in factor1_levels]
        self.factor2_labels = factor2_labels or [f"{factor2}={v}" for v in factor2_levels]
        
        logger.info(f"Initialized marginal means for {factor1} × {factor2}")
    
    def compute_marginal_means(self) -> pd.DataFrame:
        """
        Compute marginal means using emmeans.
        
        Returns:
            DataFrame with marginal means, SEs, and CIs for each cell
        """
        logger.info(f"Computing marginal means for {self.factor1} × {self.factor2}")
        
        # MUST use emmeans - no fallbacks allowed
        emmeans_df = self.emmeans_wrapper.compute_emmeans(
            factors={
                self.factor1: self.factor1_levels,
                self.factor2: self.factor2_levels
            },
            type='response'  # Back-transformed to response scale
        )
        
        # Add labels for interpretation
        emmeans_df['factor1_label'] = emmeans_df[self.factor1].map(
            dict(zip(self.factor1_levels, self.factor1_labels))
        )
        emmeans_df['factor2_label'] = emmeans_df[self.factor2].map(
            dict(zip(self.factor2_levels, self.factor2_labels))
        )
        
        return emmeans_df
    
    def _compute_marginal_means_manual(self) -> pd.DataFrame:
        """Manual marginal means calculation as fallback."""
        results = []
        
        # Get coefficients
        coef_dict = dict(zip(self.emmeans_wrapper.coef_names, 
                           self.emmeans_wrapper.coefficients))
        
        intercept = coef_dict.get('(Intercept)', 0)
        
        for f1_val, f1_label in zip(self.factor1_levels, self.factor1_labels):
            for f2_val, f2_label in zip(self.factor2_levels, self.factor2_labels):
                # Calculate linear predictor
                linear_pred = intercept
                
                # Add main effects
                if self.factor1 in coef_dict:
                    linear_pred += coef_dict[self.factor1] * f1_val
                if self.factor2 in coef_dict:
                    linear_pred += coef_dict[self.factor2] * f2_val
                
                # Add interaction
                interaction_term = f"{self.factor1}_x_{self.factor2}"
                if interaction_term in coef_dict:
                    linear_pred += coef_dict[interaction_term] * f1_val * f2_val
                
                # Back-transform (assuming log link)
                emmean = np.exp(linear_pred)
                
                # Approximate SE (simplified)
                se = emmean * 0.1  # Placeholder
                
                results.append({
                    self.factor1: f1_val,
                    self.factor2: f2_val,
                    'factor1_label': f1_label,
                    'factor2_label': f2_label,
                    'emmean': emmean,
                    'SE': se,
                    'lower.CL': emmean - 1.96 * se,
                    'upper.CL': emmean + 1.96 * se
                })
        
        return pd.DataFrame(results)
    
    def visualize_marginal_means(self,
                                emmeans_df: pd.DataFrame,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Create visualization of marginal means.
        
        Creates three plots:
        1. Interaction plot (line plot)
        2. Bar plot with error bars
        3. Heatmap of means
        
        Args:
            emmeans_df: DataFrame from compute_marginal_means
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Extract columns
        emmean_col = 'emmean' if 'emmean' in emmeans_df.columns else 'response'
        se_col = 'SE' if 'SE' in emmeans_df.columns else 'std.error'
        
        # Plot 1: Interaction plot using seaborn lineplot
        import seaborn as sns
        
        # Create DataFrame for seaborn
        plot_data = []
        for f2_val, f2_label in zip(self.factor2_levels, self.factor2_labels):
            subset = emmeans_df[emmeans_df[self.factor2] == f2_val]
            for _, row in subset.iterrows():
                plot_data.append({
                    'x': row[self.factor1],
                    'y': row[emmean_col],
                    'factor2': f2_label,
                    'se': row[se_col] if se_col in subset.columns else 0
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Use seaborn lineplot
        sns.lineplot(data=plot_df, x='x', y='y', hue='factor2', 
                    marker='o', markersize=8, linewidth=2, ax=ax1,
                    err_style='bars', errorbar=None)  # We'll add error bars manually
        
        # Add error bars manually for each group
        for f2_val, f2_label in zip(self.factor2_levels, self.factor2_labels):
            subset = emmeans_df[emmeans_df[self.factor2] == f2_val]
            if se_col in subset.columns:
                ax1.errorbar(subset[self.factor1], subset[emmean_col],
                            yerr=subset[se_col], fmt='none', 
                            capsize=5, capthick=2, alpha=0.7)
        
        ax1.set_xlabel(self.factor1_labels[0] + ' vs ' + self.factor1_labels[-1], fontsize=11)
        ax1.set_ylabel('Estimated Marginal Mean', fontsize=11)
        ax1.set_title('Interaction Plot', fontsize=12, fontweight='bold')
        ax1.set_xticks(self.factor1_levels)
        ax1.set_xticklabels(self.factor1_labels)
        ax1.legend(title=self.factor2, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Check for interaction (non-parallel lines)
        if len(emmeans_df) == 4:
            means = emmeans_df[emmean_col].values
            # Interaction = (means[3] - means[2]) - (means[1] - means[0])
            interaction = (means[3] - means[2]) - (means[1] - means[0])
            if abs(interaction) > 0.1:  # Threshold for meaningful interaction
                ax1.text(0.5, 0.95, 'Interaction Present', 
                        transform=ax1.transAxes, ha='center',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Plot 2: Bar plot with error bars using seaborn
        
        # Create labels for x-axis
        x_labels = [f"{row['factor1_label']}\n{row['factor2_label']}" 
                   for _, row in emmeans_df.iterrows()]
        
        # Create a copy of DataFrame with x labels  
        bar_df = emmeans_df.copy()
        bar_df['Group'] = x_labels
        
        # Determine colors based on groups
        colors = ['steelblue', 'coral'] * 2  # Alternating colors
        
        # Use seaborn barplot
        sns.barplot(data=bar_df, x='Group', y=emmean_col, ax=ax2,
                   palette=colors[:len(bar_df)], alpha=0.7,
                   errorbar=None)  # We'll add error bars manually
        
        # Add error bars manually if SE column exists
        if se_col in emmeans_df.columns:
            x_positions = np.arange(len(emmeans_df))
            ax2.errorbar(x_positions, emmeans_df[emmean_col], 
                        yerr=emmeans_df[se_col], 
                        fmt='none', c='black', capsize=5)
        
        ax2.set_xlabel('')  # Remove x label since it's in the tick labels
        ax2.set_ylabel('Estimated Marginal Mean', fontsize=11)
        ax2.set_title('Marginal Means by Group', fontsize=12, fontweight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for patch, val in zip(ax2.patches, emmeans_df[emmean_col]):
            height = patch.get_height()
            ax2.text(patch.get_x() + patch.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Heatmap of means
        # Reshape for heatmap
        pivot_table = emmeans_df.pivot(index=self.factor1, 
                                      columns=self.factor2, 
                                      values=emmean_col)
        
        sns.heatmap(pivot_table, annot=True, fmt='.2f', 
                   cmap='YlOrRd', cbar_kws={'label': 'Marginal Mean'},
                   xticklabels=self.factor2_labels,
                   yticklabels=self.factor1_labels,
                   ax=ax3)
        ax3.set_title('Marginal Means Heatmap', fontsize=12, fontweight='bold')
        ax3.set_xlabel(self.factor2, fontsize=11)
        ax3.set_ylabel(self.factor1, fontsize=11)
        
        plt.suptitle(f'Marginal Means Analysis: {self.factor1} × {self.factor2} Interaction',
                    fontsize=13, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved marginal means visualization to {save_path}")
        
        return fig
    
    def compute_contrasts(self, emmeans_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute contrasts and simple effects.
        
        Args:
            emmeans_df: DataFrame from compute_marginal_means
            
        Returns:
            Dictionary with various contrasts
        """
        contrasts = {}
        
        if len(emmeans_df) == 4:  # 2x2 design
            emmean_col = 'emmean' if 'emmean' in emmeans_df.columns else 'response'
            means = emmeans_df[emmean_col].values
            
            # Assuming order: (0,0), (0,1), (1,0), (1,1)
            
            # Interaction contrast
            interaction = (means[3] - means[2]) - (means[1] - means[0])
            contrasts['interaction'] = {
                'estimate': interaction,
                'interpretation': f'Difference in {self.factor2} effect between {self.factor1} levels'
            }
            
            # Simple effects of factor2 at each level of factor1
            contrasts['simple_effect_factor2_at_factor1_0'] = {
                'estimate': means[1] - means[0],
                'interpretation': f'Effect of {self.factor2} when {self.factor1} = {self.factor1_levels[0]}'
            }
            
            contrasts['simple_effect_factor2_at_factor1_1'] = {
                'estimate': means[3] - means[2],
                'interpretation': f'Effect of {self.factor2} when {self.factor1} = {self.factor1_levels[1]}'
            }
            
            # Simple effects of factor1 at each level of factor2
            contrasts['simple_effect_factor1_at_factor2_0'] = {
                'estimate': means[2] - means[0],
                'interpretation': f'Effect of {self.factor1} when {self.factor2} = {self.factor2_levels[0]}'
            }
            
            contrasts['simple_effect_factor1_at_factor2_1'] = {
                'estimate': means[3] - means[1],
                'interpretation': f'Effect of {self.factor1} when {self.factor2} = {self.factor2_levels[1]}'
            }
            
            # Main effects
            contrasts['main_effect_factor1'] = {
                'estimate': np.mean([means[2], means[3]]) - np.mean([means[0], means[1]]),
                'interpretation': f'Average effect of {self.factor1}'
            }
            
            contrasts['main_effect_factor2'] = {
                'estimate': np.mean([means[1], means[3]]) - np.mean([means[0], means[2]]),
                'interpretation': f'Average effect of {self.factor2}'
            }
        
        return contrasts
    
    def generate_interpretation(self, 
                               emmeans_df: pd.DataFrame,
                               contrasts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate interpretation of marginal means results.
        
        Args:
            emmeans_df: DataFrame from compute_marginal_means
            contrasts: Dictionary from compute_contrasts
            
        Returns:
            Dictionary with interpretation details
        """
        interpretation = {
            'factor1': self.factor1,
            'factor2': self.factor2,
            'marginal_means': emmeans_df.to_dict('records'),
            'contrasts': contrasts,
            'pattern': None,
            'summary': None
        }
        
        # Determine interaction pattern
        if 'interaction' in contrasts:
            interaction_val = contrasts['interaction']['estimate']
            
            if abs(interaction_val) < 0.1:
                interpretation['pattern'] = 'additive'
                interpretation['summary'] = (
                    f"The effects of {self.factor1} and {self.factor2} are largely additive. "
                    f"The effect of one factor does not depend on the level of the other."
                )
            elif interaction_val > 0.1:
                interpretation['pattern'] = 'synergistic'
                interpretation['summary'] = (
                    f"There is a synergistic interaction between {self.factor1} and {self.factor2}. "
                    f"The combined effect is greater than the sum of individual effects."
                )
            else:
                interpretation['pattern'] = 'antagonistic'
                interpretation['summary'] = (
                    f"There is an antagonistic interaction between {self.factor1} and {self.factor2}. "
                    f"The factors partially cancel each other's effects."
                )
            
            # Check for crossover
            se1 = contrasts.get('simple_effect_factor2_at_factor1_0', {}).get('estimate', 0)
            se2 = contrasts.get('simple_effect_factor2_at_factor1_1', {}).get('estimate', 0)
            if np.sign(se1) != np.sign(se2):
                interpretation['pattern'] = 'crossover'
                interpretation['summary'] = (
                    f"There is a crossover interaction between {self.factor1} and {self.factor2}. "
                    f"The effect of {self.factor2} reverses depending on {self.factor1} level."
                )
        
        return interpretation


def analyze_marginal_means(julia_model: Dict[str, Any],
                          factor1: str,
                          factor2: str,
                          feature_names: List[str],
                          factor1_levels: List = [0, 1],
                          factor2_levels: List = [0, 1],
                          factor1_labels: Optional[List[str]] = None,
                          factor2_labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Wrapper function for marginal means analysis.
    
    Args:
        julia_model: Julia model results dictionary
        factor1: Name of first factor
        factor2: Name of second factor
        feature_names: List of all feature names
        factor1_levels: Levels of factor1
        factor2_levels: Levels of factor2
        factor1_labels: Labels for factor1 levels
        factor2_labels: Labels for factor2 levels
        save_path: Optional path to save visualization
        
    Returns:
        Dictionary with marginal means, contrasts, interpretation, and figure
    """
    # Create emmeans wrapper
    emmeans_wrapper = JuliaEMmeans(julia_model, feature_names)
    
    # Create interpreter
    interpreter = MarginalMeansInterpretation(
        emmeans_wrapper=emmeans_wrapper,
        factor1=factor1,
        factor2=factor2,
        factor1_levels=factor1_levels,
        factor2_levels=factor2_levels,
        factor1_labels=factor1_labels,
        factor2_labels=factor2_labels
    )
    
    # Compute marginal means
    emmeans_df = interpreter.compute_marginal_means()
    
    # Compute contrasts
    contrasts = interpreter.compute_contrasts(emmeans_df)
    
    # Generate visualization
    fig = interpreter.visualize_marginal_means(emmeans_df, save_path)
    
    # Generate interpretation
    interpretation = interpreter.generate_interpretation(emmeans_df, contrasts)
    
    return {
        'marginal_means': emmeans_df,
        'contrasts': contrasts,
        'interpretation': interpretation,
        'figure': fig
    }