"""Hypothesis tests for Julia GLM results.

Simplified implementations that work with Julia result dictionaries
instead of H2O model objects.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class JuliaHypothesisTests:
    """Hypothesis tests for Julia GLM results."""
    
    @staticmethod
    def wald_test_single_coefficient(
        coefficient: float,
        standard_error: float,
        null_value: float = 0.0,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """Perform Wald test on a single coefficient.
        
        Args:
            coefficient: Coefficient estimate
            standard_error: Standard error
            null_value: Value under null hypothesis
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Test results dictionary
        """
        # Compute test statistic
        z_stat = (coefficient - null_value) / standard_error if standard_error > 0 else 0
        
        # Compute p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(z_stat)
        else:  # less
            p_value = stats.norm.cdf(z_stat)
        
        # Confidence interval
        ci_lower = coefficient - 1.96 * standard_error
        ci_upper = coefficient + 1.96 * standard_error
        
        return {
            'coefficient': coefficient,
            'standard_error': standard_error,
            'z_statistic': z_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant_at_0.05': p_value < 0.05,
            'null_value': null_value,
            'alternative': alternative
        }
    
    @staticmethod
    def sesoi_test(
        coefficient: float,
        standard_error: float,
        sesoi_threshold: float = 0.1,
        alternative: str = 'greater'
    ) -> Dict[str, Any]:
        """Test if effect exceeds smallest effect size of interest (SESOI).
        
        Args:
            coefficient: Coefficient estimate
            standard_error: Standard error
            sesoi_threshold: SESOI threshold (e.g., 0.1 on log scale)
            alternative: 'greater' (effect > threshold) or 'less'
            
        Returns:
            Test results with practical significance
        """
        # Test against SESOI threshold
        if alternative == 'greater':
            z_stat = (coefficient - sesoi_threshold) / standard_error
            p_value = 1 - stats.norm.cdf(z_stat)
            exceeds_sesoi = coefficient > sesoi_threshold and p_value < 0.05
        else:
            z_stat = (coefficient - (-sesoi_threshold)) / standard_error
            p_value = stats.norm.cdf(z_stat)
            exceeds_sesoi = coefficient < -sesoi_threshold and p_value < 0.05
        
        # Confidence interval
        ci_lower = coefficient - 1.96 * standard_error
        ci_upper = coefficient + 1.96 * standard_error
        
        # Response scale interpretation (for log link)
        response_scale_effect = (np.exp(coefficient) - 1) * 100  # Percent change
        response_scale_threshold = (np.exp(sesoi_threshold) - 1) * 100
        
        return {
            'coefficient': coefficient,
            'standard_error': standard_error,
            'sesoi_threshold': sesoi_threshold,
            'z_statistic': z_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'exceeds_sesoi': exceeds_sesoi,
            'response_scale_effect': response_scale_effect,
            'response_scale_threshold': response_scale_threshold,
            'response_scale_interpretation': f"{response_scale_effect:.1f}% change (threshold: {response_scale_threshold:.1f}%)",
            'practical_significance': 'Yes' if exceeds_sesoi else 'No'
        }
    
    @staticmethod
    def tost_test(
        coefficient: float,
        standard_error: float,
        lower_bound: float = -0.1,
        upper_bound: float = 0.1,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Two One-Sided Tests (TOST) for equivalence testing.
        
        Args:
            coefficient: Coefficient estimate
            standard_error: Standard error
            lower_bound: Lower equivalence bound
            upper_bound: Upper equivalence bound
            alpha: Significance level
            
        Returns:
            TOST test results
        """
        # Test 1: coefficient > lower_bound
        z1 = (coefficient - lower_bound) / standard_error
        p1 = 1 - stats.norm.cdf(z1)
        
        # Test 2: coefficient < upper_bound
        z2 = (coefficient - upper_bound) / standard_error
        p2 = stats.norm.cdf(z2)
        
        # TOST p-value is the maximum of the two
        tost_p_value = max(p1, p2)
        is_equivalent = tost_p_value < alpha
        
        # Confidence interval
        ci_lower = coefficient - 1.96 * standard_error
        ci_upper = coefficient + 1.96 * standard_error
        
        return {
            'coefficient': coefficient,
            'standard_error': standard_error,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'test1_p_value': p1,
            'test2_p_value': p2,
            'tost_p_value': tost_p_value,
            'is_equivalent': is_equivalent,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'interpretation': 'Effect is practically equivalent to zero' if is_equivalent else 'Effect is not negligible'
        }
    
    @staticmethod
    def joint_wald_test(
        coefficients: np.ndarray,
        covariance_matrix: np.ndarray,
        null_values: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Joint Wald test for multiple coefficients.
        
        Args:
            coefficients: Vector of coefficient estimates
            covariance_matrix: Variance-covariance matrix
            null_values: Values under null hypothesis (default: zeros)
            
        Returns:
            Joint test results
        """
        k = len(coefficients)
        if null_values is None:
            null_values = np.zeros(k)
        
        # Compute test statistic: (β - β0)' V^{-1} (β - β0)
        diff = coefficients - null_values
        try:
            chi2_stat = diff @ np.linalg.inv(covariance_matrix) @ diff
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            chi2_stat = diff @ np.linalg.pinv(covariance_matrix) @ diff
        
        # P-value from chi-square distribution
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=k)
        
        return {
            'chi2_statistic': chi2_stat,
            'df': k,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'n_coefficients': k,
            'interpretation': f"Joint test of {k} coefficients"
        }
    
    @staticmethod
    def likelihood_ratio_test(
        null_loglik: float,
        full_loglik: float,
        df_difference: int
    ) -> Dict[str, Any]:
        """Likelihood ratio test for nested models.
        
        Args:
            null_loglik: Log-likelihood of null model
            full_loglik: Log-likelihood of full model
            df_difference: Difference in degrees of freedom
            
        Returns:
            LRT results
        """
        # LR statistic
        lr_statistic = 2 * (full_loglik - null_loglik)
        
        # P-value from chi-square distribution
        p_value = 1 - stats.chi2.cdf(lr_statistic, df=df_difference)
        
        # McFadden pseudo R-squared (for interpretation)
        # Note: Only valid for models with same theta in NB
        mcfadden_r2 = 1 - (full_loglik / null_loglik) if null_loglik != 0 else 0
        
        return {
            'lr_statistic': lr_statistic,
            'df': df_difference,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'null_loglik': null_loglik,
            'full_loglik': full_loglik,
            'mcfadden_r2': mcfadden_r2,
            'interpretation': 'Full model significantly better' if p_value < 0.05 else 'No significant improvement'
        }
    
    @staticmethod
    def score_test_from_deviance(
        null_deviance: float,
        full_deviance: float,
        df_difference: int
    ) -> Dict[str, Any]:
        """Score test approximation using deviance difference.
        
        Args:
            null_deviance: Deviance of null model
            full_deviance: Deviance of full model
            df_difference: Difference in parameters
            
        Returns:
            Score test results
        """
        # Score statistic approximation
        score_statistic = null_deviance - full_deviance
        
        # P-value from chi-square distribution
        p_value = 1 - stats.chi2.cdf(score_statistic, df=df_difference)
        
        return {
            'score_statistic': score_statistic,
            'df': df_difference,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'deviance_improvement': score_statistic,
            'interpretation': 'Significant improvement' if p_value < 0.05 else 'No significant improvement'
        }
    
    @staticmethod
    def generate_hypothesis_report(
        test_results: Dict[str, Any],
        test_name: str,
        hypothesis: str
    ) -> Dict[str, Any]:
        """Generate publication-ready report for hypothesis test.
        
        Args:
            test_results: Test results dictionary
            test_name: Name of the test
            hypothesis: Hypothesis being tested
            
        Returns:
            Formatted report
        """
        report = {
            'test': test_name,
            'hypothesis': hypothesis,
            'results': {}
        }
        
        # Format based on test type
        if 'lr_statistic' in test_results:
            # LRT
            report['results'] = {
                'statistic': f"χ² = {test_results['lr_statistic']:.2f}",
                'df': test_results['df'],
                'p_value': f"p = {test_results['p_value']:.4f}",
                'decision': 'Reject H0' if test_results['significant_at_0.05'] else 'Fail to reject H0',
                'interpretation': test_results.get('interpretation', '')
            }
        elif 'score_statistic' in test_results:
            # Score test
            report['results'] = {
                'statistic': f"S = {test_results['score_statistic']:.2f}",
                'df': test_results['df'],
                'p_value': f"p = {test_results['p_value']:.4f}",
                'decision': 'Reject H0' if test_results['significant_at_0.05'] else 'Fail to reject H0',
                'interpretation': test_results.get('interpretation', '')
            }
        elif 'z_statistic' in test_results:
            # Wald test
            report['results'] = {
                'coefficient': f"β = {test_results['coefficient']:.4f}",
                'se': f"SE = {test_results['standard_error']:.4f}",
                'statistic': f"z = {test_results['z_statistic']:.2f}",
                'p_value': f"p = {test_results['p_value']:.4f}",
                'ci': f"95% CI: [{test_results['ci_lower']:.4f}, {test_results['ci_upper']:.4f}]",
                'decision': 'Significant' if test_results['significant_at_0.05'] else 'Not significant'
            }
        elif 'chi2_statistic' in test_results:
            # Joint Wald
            report['results'] = {
                'statistic': f"χ² = {test_results['chi2_statistic']:.2f}",
                'df': test_results['df'],
                'p_value': f"p = {test_results['p_value']:.4f}",
                'decision': 'Reject H0' if test_results['significant_at_0.05'] else 'Fail to reject H0',
                'interpretation': test_results.get('interpretation', '')
            }
        
        return report
    
    @staticmethod
    def visualize_hypothesis_test(
        test_results: Dict[str, Any],
        test_name: str,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create visualization for hypothesis test results.
        
        Args:
            test_results: Test results dictionary
            test_name: Name of the test
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'coefficient' in test_results and 'standard_error' in test_results:
            # Single coefficient plot
            coef = test_results['coefficient']
            se = test_results['standard_error']
            ci_lower = test_results.get('ci_lower', coef - 1.96*se)
            ci_upper = test_results.get('ci_upper', coef + 1.96*se)
            
            # Plot coefficient with CI
            ax.errorbar([0], [coef], yerr=[[coef-ci_lower], [ci_upper-coef]], 
                       fmt='o', markersize=10, capsize=10, capthick=2)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='H0: β = 0')
            
            # Add SESOI threshold if present
            if 'sesoi_threshold' in test_results:
                threshold = test_results['sesoi_threshold']
                ax.axhline(y=threshold, color='orange', linestyle='--', alpha=0.5, 
                          label=f'SESOI: {threshold}')
                ax.axhline(y=-threshold, color='orange', linestyle='--', alpha=0.5)
            
            # Add equivalence bounds if present
            if 'lower_bound' in test_results and 'upper_bound' in test_results:
                ax.axhspan(test_results['lower_bound'], test_results['upper_bound'],
                          alpha=0.2, color='green', label='Equivalence region')
            
            ax.set_xlabel('Parameter')
            ax.set_ylabel('Coefficient Value')
            ax.set_title(f'{test_name}\np = {test_results.get("p_value", 0):.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif 'lr_statistic' in test_results or 'score_statistic' in test_results:
            # Chi-square test visualization
            stat_name = 'lr_statistic' if 'lr_statistic' in test_results else 'score_statistic'
            statistic = test_results[stat_name]
            df = test_results['df']
            p_value = test_results['p_value']
            
            # Plot chi-square distribution
            x = np.linspace(0, max(statistic * 1.5, stats.chi2.ppf(0.999, df)), 1000)
            y = stats.chi2.pdf(x, df)
            
            ax.plot(x, y, 'b-', label=f'χ²({df}) distribution')
            ax.fill_between(x[x >= statistic], 0, stats.chi2.pdf(x[x >= statistic], df),
                           alpha=0.3, color='red', label=f'p = {p_value:.4f}')
            ax.axvline(x=statistic, color='red', linestyle='--', 
                      label=f'Test statistic = {statistic:.2f}')
            
            ax.set_xlabel('Test Statistic')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'{test_name}\nχ²({df}) = {statistic:.2f}, p = {p_value:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hypothesis test visualization saved to {save_path}")
        
        return fig