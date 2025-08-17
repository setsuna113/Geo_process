"""Likelihood Ratio Test implementation for GLM models."""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LikelihoodRatioTest:
    """Likelihood Ratio Test for nested model comparison.
    
    Tests whether adding parameters significantly improves model fit.
    Uses raw deviance differences: LR = Deviance(null) - Deviance(full).
    Note: McFadden's R² is not calculated as it's invalid when models 
    have different dispersion parameters (theta) in negative binomial GLMs.
    """
    
    @staticmethod
    def test_nested_models(
        null_model: Any,
        full_model: Any,
        df_difference: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform likelihood ratio test between nested models.
        
        Uses raw deviance difference: LR = Deviance(null) - Deviance(full)
        Does not calculate McFadden's R² as it's invalid when models have different theta.
        
        Args:
            null_model: Restricted (null) model
            full_model: Unrestricted (full) model  
            df_difference: Difference in degrees of freedom (if not provided, will try to infer)
            
        Returns:
            Dictionary with test statistics, p-value, and deviance information
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
        
        # Calculate LR statistic (raw deviance difference)
        lr_statistic = null_deviance - full_deviance
        
        # Interpret the deviance change
        if lr_statistic > 0:
            deviance_interpretation = "Full model reduces deviance (better fit)"
            logger.info(f"Full model improves fit: deviance reduced by {lr_statistic:.2f}")
        elif lr_statistic < 0:
            deviance_interpretation = "Full model increases deviance (worse fit)"
            logger.warning(f"Full model worsens fit: deviance increased by {abs(lr_statistic):.2f}")
        else:
            deviance_interpretation = "Models have identical deviance"
        
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
        
        # Calculate deviance change percentage
        deviance_change_pct = (lr_statistic / null_deviance * 100) if null_deviance > 0 else 0
        
        # Build clean result (without McFadden's R²)
        result = {
            'test_type': 'likelihood_ratio',
            'lr_statistic': float(lr_statistic),
            'degrees_of_freedom': int(df_difference),
            'p_value': float(p_value),
            'null_deviance': float(null_deviance),
            'full_deviance': float(full_deviance),
            'deviance_difference': float(lr_statistic),
            'deviance_change_pct': float(deviance_change_pct),
            'significant': p_value < 0.05,
            'deviance_interpretation': deviance_interpretation,
            'conclusion': LikelihoodRatioTest._generate_conclusion(p_value, lr_statistic)
        }
        
        # Add AIC/BIC if available (for completeness)
        if hasattr(null_model, 'aic'):
            result['null_aic'] = null_model.aic() if callable(null_model.aic) else null_model.aic
        if hasattr(full_model, 'aic'):
            result['full_aic'] = full_model.aic() if callable(full_model.aic) else full_model.aic
            
        logger.info(f"LRT: χ²={lr_statistic:.2f}, df={df_difference}, p={p_value:.4f}")
        
        return result
    
    @staticmethod
    def _generate_conclusion(p_value: float, lr_statistic: float) -> str:
        """Generate overall conclusion based on p-value and deviance change.
        
        Args:
            p_value: P-value from chi-squared test
            lr_statistic: LR statistic (deviance difference)
            
        Returns:
            Conclusion string
        """
        if lr_statistic < 0:
            return f"Full model has significantly worse fit (deviance increased by {abs(lr_statistic):.2f}, p={p_value:.4f})"
        elif p_value < 0.05 and lr_statistic > 0:
            return f"Full model has significantly better fit (deviance reduced by {lr_statistic:.2f}, p={p_value:.4f})"
        elif p_value >= 0.05 and lr_statistic > 0:
            return f"Full model shows non-significant improvement (deviance reduced by {lr_statistic:.2f}, p={p_value:.4f})"
        else:
            return f"No meaningful difference between models (p={p_value:.4f})"
    
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
        deviance_change_pct = test_results.get('deviance_change_pct', 0)
        deviance_interpretation = test_results.get('deviance_interpretation', '')
        
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
            f"The deviance {'decreased' if lr_stat > 0 else 'increased'} by {abs(lr_stat):.2f} "
            f"({abs(deviance_change_pct):.1f}% {'reduction' if lr_stat > 0 else 'increase'}). "
            f"{test_results['conclusion']}"
        )
        
        # Create results table
        results_table = pd.DataFrame([{
            'Test': test_name,
            'χ² Statistic': f"{abs(lr_stat):.2f}",
            'df': df,
            'p-value': p_str.replace('p = ', '').replace('p ', ''),
            'Deviance Change': f"{lr_stat:.2f}",
            '% Change': f"{deviance_change_pct:.1f}%"
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
            "The test statistic is calculated as the difference in deviances: "
            "LR = Deviance(null) - Deviance(full), which follows a chi-squared distribution "
            "with degrees of freedom equal to the difference in parameters between models. "
            "Statistical significance was assessed at α = 0.05. Note: When models have "
            "different dispersion parameters (theta), deviance comparisons should be "
            "interpreted with caution."
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
        """Create visualization of LRT results focusing on deviance comparison.
        
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
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Extract values
        lr_stat = test_results['lr_statistic']
        p_value = test_results['p_value']
        null_dev = test_results['null_deviance']
        full_dev = test_results['full_deviance']
        deviance_change_pct = test_results.get('deviance_change_pct', 0)
        
        # Panel 1: Deviance comparison
        ax1 = axes[0]
        models = [null_model_name, full_model_name]
        deviances = [null_dev, full_dev]
        colors = ['lightcoral', 'lightgreen' if full_dev < null_dev else 'salmon']
        
        bars = ax1.bar(models, deviances, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, deviances):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=11)
        
        ax1.set_ylabel('Deviance', fontsize=12)
        ax1.set_title('Model Deviance Comparison', fontsize=13, fontweight='bold')
        
        # Add change indicators
        if lr_stat != 0:
            # Arrow showing direction of change
            if full_dev < null_dev:
                ax1.annotate('', xy=(1, full_dev), xytext=(0, null_dev),
                            arrowprops=dict(arrowstyle='->', color='green', lw=2))
                change_color = 'green'
                change_text = 'IMPROVED'
            else:
                ax1.annotate('', xy=(1, full_dev), xytext=(0, null_dev),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2))
                change_color = 'red'
                change_text = 'WORSENED'
            
            # Add change statistics
            ax1.text(0.5, (null_dev + full_dev)/2, 
                    f'Δ = {lr_stat:.1f}\n({deviance_change_pct:.1f}%)',
                    ha='center', va='center', color=change_color, 
                    fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add interpretation
            ax1.text(0.5, min(null_dev, full_dev) * 0.9,
                    f'Fit {change_text}', ha='center', fontsize=12,
                    color=change_color, fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Statistical significance
        ax2 = axes[1]
        
        # Chi-square distribution visualization
        from scipy import stats as scipy_stats
        df = test_results['degrees_of_freedom']
        
        # Ensure we have valid x range
        max_x = max(abs(lr_stat) * 1.5, scipy_stats.chi2.ppf(0.999, df), 20)
        x = np.linspace(0, max_x, 1000)
        y = scipy_stats.chi2.pdf(x, df)
        
        # Plot the chi-squared distribution
        ax3.fill_between(x, y, alpha=0.3, color='blue', label=f'χ²({df}) distribution')
        ax3.plot(x, y, 'b-', linewidth=2)
        
        # Mark critical value at alpha=0.05
        critical_val = scipy_stats.chi2.ppf(0.95, df)
        ax3.axvline(critical_val, color='orange', linestyle=':', linewidth=2, 
                   label=f'Critical value = {critical_val:.2f}')
        
        # Mark observed statistic
        ax3.axvline(abs(lr_stat), color='red', linestyle='--', linewidth=2, 
                   label=f'Observed LR = {abs(lr_stat):.1f}')
        
        # Shade rejection region (beyond critical value)
        x_reject = x[x >= critical_val]
        y_reject = scipy_stats.chi2.pdf(x_reject, df)
        ax3.fill_between(x_reject, y_reject, alpha=0.3, color='red', label='Rejection region')
        
        # Add text with proper comparison
        if abs(lr_stat) > critical_val:
            decision_text = f'REJECT H₀\n({abs(lr_stat):.1f} > {critical_val:.1f})'
            decision_color = 'red'
        else:
            decision_text = f'FAIL TO REJECT H₀\n({abs(lr_stat):.1f} ≤ {critical_val:.1f})'
            decision_color = 'gray'
        
        # Add p-value text
        if p_value < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p_value:.3f}'
        
        # Position text appropriately
        text_y = max(y) * 0.6
        ax3.text(abs(lr_stat), text_y, p_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax3.text(max_x * 0.7, text_y * 0.8, decision_text, ha='center', fontsize=11,
                fontweight='bold', color=decision_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Test Statistic')
        ax3.set_ylabel('Probability Density')
        ax3.set_title('Chi-squared Test of Significance')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max_x)
        
        plt.suptitle('Likelihood Ratio Test Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig