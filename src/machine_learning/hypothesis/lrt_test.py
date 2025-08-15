"""Likelihood Ratio Test implementation for GLM models."""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LikelihoodRatioTest:
    """Likelihood Ratio Test for nested model comparison.
    
    Tests whether adding parameters significantly improves model fit.
    Reports McFadden's pseudo R² as the primary effect size measure.
    """
    
    @staticmethod
    def test_nested_models(
        null_model: Any,
        full_model: Any,
        df_difference: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform likelihood ratio test between nested models.
        
        Args:
            null_model: Restricted (null) model
            full_model: Unrestricted (full) model  
            df_difference: Difference in degrees of freedom (if not provided, will try to infer)
            
        Returns:
            Dictionary with test statistics, p-value, and McFadden R²
        """
        # Get deviances from models
        if hasattr(null_model, 'residual_deviance'):
            null_deviance = null_model.residual_deviance()
            full_deviance = full_model.residual_deviance()
        elif hasattr(null_model, 'deviance'):
            null_deviance = null_model.deviance() if callable(null_model.deviance) else null_model.deviance
            full_deviance = full_model.deviance() if callable(full_model.deviance) else full_model.deviance
        else:
            raise ValueError("Models must have residual_deviance() or deviance attribute")
        
        # Calculate LR statistic
        lr_statistic = null_deviance - full_deviance
        
        # Calculate McFadden's pseudo R²
        mcfadden_r2 = 1 - (full_deviance / null_deviance) if null_deviance > 0 else 0
        
        # Handle negative LR (full model worse than null)
        if lr_statistic < 0:
            logger.warning(f"Full model has worse fit (LR={lr_statistic:.2f}, R²={mcfadden_r2:.4f})")
            # McFadden R² will be negative when full model is worse
            # Keep negative for interpretation
        
        # Get degrees of freedom difference
        if df_difference is None:
            if hasattr(null_model, 'residual_degrees_of_freedom'):
                null_df = null_model.residual_degrees_of_freedom()
                full_df = full_model.residual_degrees_of_freedom()
                df_difference = null_df - full_df
            else:
                raise ValueError("Cannot determine df difference. Please provide df_difference parameter.")
        
        if df_difference <= 0:
            raise ValueError(f"df_difference must be positive (got {df_difference})")
        
        # Compute p-value from chi-squared distribution
        # Use absolute value of LR for chi-square test
        p_value = 1 - stats.chi2.cdf(abs(lr_statistic), df=df_difference)
        
        # Calculate percent deviance explained
        percent_deviance_explained = (lr_statistic / null_deviance * 100) if null_deviance > 0 else 0
        
        # Interpret McFadden R²
        r2_interpretation = LikelihoodRatioTest._interpret_mcfadden_r2(mcfadden_r2)
        
        # Build clean result
        result = {
            'test_type': 'likelihood_ratio',
            'lr_statistic': float(lr_statistic),
            'degrees_of_freedom': int(df_difference),
            'p_value': float(p_value),
            'mcfadden_r2': float(mcfadden_r2),
            'percent_deviance_explained': float(percent_deviance_explained),
            'null_deviance': float(null_deviance),
            'full_deviance': float(full_deviance),
            'significant': p_value < 0.05,
            'r2_interpretation': r2_interpretation,
            'conclusion': LikelihoodRatioTest._generate_conclusion(p_value, mcfadden_r2)
        }
        
        # Add AIC/BIC if available (for completeness)
        if hasattr(null_model, 'aic'):
            result['null_aic'] = null_model.aic() if callable(null_model.aic) else null_model.aic
        if hasattr(full_model, 'aic'):
            result['full_aic'] = full_model.aic() if callable(full_model.aic) else full_model.aic
            
        logger.info(f"LRT: χ²={lr_statistic:.2f}, df={df_difference}, p={p_value:.4f}, R²={mcfadden_r2:.4f}")
        
        return result
    
    @staticmethod
    def _interpret_mcfadden_r2(r2: float) -> str:
        """Interpret McFadden R² value based on standard guidelines."""
        if r2 < 0:
            return "Model fit worsened"
        elif r2 < 0.05:
            return "Negligible improvement"
        elif r2 < 0.15:
            return "Modest improvement"  
        elif r2 < 0.35:
            return "Good improvement"
        else:
            return "Excellent improvement"
    
    @staticmethod
    def _generate_conclusion(p_value: float, r2: float) -> str:
        """Generate overall conclusion combining significance and effect size."""
        if r2 < 0:
            return f"Full model is worse than null model (R²={r2:.4f})"
        elif p_value < 0.05 and r2 >= 0.05:
            return f"Significant and meaningful improvement (p={p_value:.4f}, R²={r2:.4f})"
        elif p_value < 0.05 and r2 < 0.05:
            return f"Statistically significant but negligible improvement (p={p_value:.4f}, R²={r2:.4f})"
        else:
            return f"No significant improvement (p={p_value:.4f}, R²={r2:.4f})"
    
    @staticmethod
    def generate_publication_report(
        test_results: Dict[str, Any],
        test_name: str = "Likelihood Ratio Test",
        hypothesis: str = "model comparison",
        model_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate publication-ready report for LRT results.
        
        Args:
            test_results: Results from test_nested_models
            test_name: Name of the test for reporting
            hypothesis: Description of what's being tested
            model_description: Optional description of models being compared
            
        Returns:
            Dictionary with formatted text, tables, and summaries
        """
        import pandas as pd
        
        # Extract key values
        lr_stat = test_results['lr_statistic']
        df = test_results['degrees_of_freedom']
        p_value = test_results['p_value']
        r2 = test_results['mcfadden_r2']
        pct_explained = test_results['percent_deviance_explained']
        
        # Format p-value
        if p_value < 0.001:
            p_str = "p < 0.001"
        elif p_value < 0.01:
            p_str = f"p = {p_value:.3f}"
        else:
            p_str = f"p = {p_value:.2f}"
        
        # Main text
        main_text = (
            f"A likelihood ratio test was conducted to evaluate {hypothesis}. "
            f"The test statistic was χ²({df}) = {abs(lr_stat):.2f}, {p_str}. "
            f"McFadden's pseudo-R² = {r2:.4f}, indicating {test_results['r2_interpretation'].lower()}. "
            f"The model explained {abs(pct_explained):.1f}% of the deviance. "
            f"{test_results['conclusion']}"
        )
        
        # Create results table
        results_table = pd.DataFrame([{
            'Test': test_name,
            'χ² Statistic': f"{abs(lr_stat):.2f}",
            'df': df,
            'p-value': p_str.replace('p = ', '').replace('p ', ''),
            'McFadden R²': f"{r2:.4f}",
            '% Deviance Explained': f"{abs(pct_explained):.1f}%"
        }])
        
        # Model comparison details
        comparison_details = pd.DataFrame([
            {'Model': 'Null', 'Deviance': f"{test_results['null_deviance']:.2f}"},
            {'Model': 'Full', 'Deviance': f"{test_results['full_deviance']:.2f}"},
            {'Model': 'Difference', 'Deviance': f"{lr_stat:.2f}"}
        ])
        
        # Methods text
        methods_text = (
            "Model comparison was performed using the likelihood ratio test. "
            "The test statistic follows a chi-squared distribution with degrees of freedom "
            "equal to the difference in parameters between models. McFadden's pseudo-R² "
            "was calculated as 1 - (deviance_full/deviance_null) to quantify the improvement "
            "in model fit. Statistical significance was assessed at α = 0.05."
        )
        
        return {
            'main_text': main_text,
            'methods_text': methods_text,
            'results_table': results_table,
            'comparison_details': comparison_details,
            'statistical_summary': test_results
        }
    
    @staticmethod
    def visualize_lrt(
        test_results: Dict[str, Any],
        null_model_name: str = "Null Model",
        full_model_name: str = "Full Model",
        figsize: tuple = (12, 5)
    ) -> 'plt.Figure':
        """Create visualization of LRT results focusing on McFadden R².
        
        Args:
            test_results: Results from test_nested_models
            null_model_name: Name for null model
            full_model_name: Name for full model
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Extract values
        lr_stat = test_results['lr_statistic']
        p_value = test_results['p_value']
        r2 = test_results['mcfadden_r2']
        null_dev = test_results['null_deviance']
        full_dev = test_results['full_deviance']
        pct_explained = test_results['percent_deviance_explained']
        
        # Panel 1: Deviance comparison
        ax1 = axes[0]
        models = [null_model_name, full_model_name]
        deviances = [null_dev, full_dev]
        colors = ['lightcoral', 'lightgreen' if full_dev < null_dev else 'lightcoral']
        
        bars = ax1.bar(models, deviances, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, deviances):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom')
        
        ax1.set_ylabel('Deviance')
        ax1.set_title('Model Deviance Comparison')
        
        # Add improvement arrow if applicable
        if full_dev < null_dev:
            ax1.annotate('', xy=(1, full_dev), xytext=(0, null_dev),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax1.text(0.5, (null_dev + full_dev)/2, f'Δ = {lr_stat:.1f}',
                    ha='center', va='center', color='green', fontweight='bold')
        
        # Panel 2: McFadden R² visualization
        ax2 = axes[1]
        
        # Create gauge-like visualization for R²
        theta = np.linspace(0, np.pi, 100)
        r_inner, r_outer = 0.7, 1.0
        
        # Background arc segments for interpretation zones
        zones = [
            (0, 0.05, 'lightgray', 'Negligible'),
            (0.05, 0.15, 'lightyellow', 'Modest'),
            (0.15, 0.35, 'lightgreen', 'Good'),
            (0.35, 1.0, 'darkgreen', 'Excellent')
        ]
        
        for z_min, z_max, color, label in zones:
            theta_min = z_min * np.pi
            theta_max = z_max * np.pi
            theta_zone = np.linspace(theta_min, theta_max, 50)
            
            x_inner = r_inner * np.cos(theta_zone)
            y_inner = r_inner * np.sin(theta_zone)
            x_outer = r_outer * np.cos(theta_zone)
            y_outer = r_outer * np.sin(theta_zone)
            
            vertices = list(zip(x_outer, y_outer)) + list(zip(x_inner[::-1], y_inner[::-1]))
            from matplotlib.patches import Polygon
            poly = Polygon(vertices, facecolor=color, alpha=0.3)
            ax2.add_patch(poly)
        
        # Add value indicator
        r2_angle = r2 * np.pi
        arrow_x = 0.85 * np.cos(r2_angle)
        arrow_y = 0.85 * np.sin(r2_angle)
        ax2.arrow(0, 0, arrow_x, arrow_y, head_width=0.05, head_length=0.05,
                 fc='red', ec='red', linewidth=2)
        
        # Add text labels
        ax2.text(0, -0.3, f'McFadden R² = {r2:.4f}', ha='center', fontsize=14, fontweight='bold')
        ax2.text(0, -0.45, f'{test_results["r2_interpretation"]}', ha='center', fontsize=11)
        
        # Add zone labels
        ax2.text(-1.1, 0, '0.0', ha='center')
        ax2.text(1.1, 0, '1.0', ha='center')
        ax2.text(0, 1.1, '0.5', ha='center')
        
        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-0.6, 1.3)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('McFadden Pseudo-R²')
        
        # Panel 3: Statistical significance
        ax3 = axes[2]
        
        # Chi-square distribution visualization
        from scipy import stats as scipy_stats
        df = test_results['degrees_of_freedom']
        x = np.linspace(0, max(abs(lr_stat) * 1.5, scipy_stats.chi2.ppf(0.999, df)), 1000)
        y = scipy_stats.chi2.pdf(x, df)
        
        ax3.fill_between(x, y, alpha=0.3, color='blue', label=f'χ²({df})')
        ax3.plot(x, y, 'b-', linewidth=2)
        
        # Mark observed statistic
        ax3.axvline(abs(lr_stat), color='red', linestyle='--', linewidth=2, 
                   label=f'Observed LR = {abs(lr_stat):.1f}')
        
        # Shade p-value area
        x_crit = np.linspace(abs(lr_stat), x[-1], 100)
        y_crit = scipy_stats.chi2.pdf(x_crit, df)
        ax3.fill_between(x_crit, y_crit, alpha=0.3, color='red')
        
        # Add text
        if p_value < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p_value:.3f}'
        
        ax3.text(abs(lr_stat), max(y) * 0.5, p_text, ha='left', fontsize=11)
        
        ax3.set_xlabel('Test Statistic')
        ax3.set_ylabel('Probability Density')
        ax3.set_title('Likelihood Ratio Test')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Likelihood Ratio Test Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig