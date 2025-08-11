"""
Classical Hypothesis Testing for DML (Option 1).

Standard significance testing using t-statistics and p-values
for Double Machine Learning causal parameters.
"""

import logging
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class DMLClassicalTest:
    """Classical hypothesis testing for DML causal effects.
    
    Tests:
    - H0: θ = 0 (no causal effect)
    - H1: θ > 0 (positive causal effect) or θ ≠ 0 (two-sided)
    
    Uses t-statistics and p-values for inference.
    """
    
    @staticmethod
    def test_causal_effect(
        theta: float,
        se_theta: float,
        alpha: float = 0.05,
        alternative: str = 'greater'
    ) -> Dict[str, Any]:
        """Test single causal parameter using classical hypothesis testing.
        
        Args:
            theta: Causal effect estimate
            se_theta: Standard error of theta
            alpha: Significance level (default 0.05)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with test results:
            - t_statistic: Test statistic
            - p_value: P-value
            - reject_h0: Whether to reject null hypothesis
            - ci_lower: Lower bound of confidence interval
            - ci_upper: Upper bound of confidence interval
            - interpretation: Text interpretation
        """
        if se_theta <= 0:
            raise ValueError("Standard error must be positive")
        
        # Calculate t-statistic
        t_stat = theta / se_theta
        
        # Calculate p-value based on alternative hypothesis
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            z_critical = stats.norm.ppf(1 - alpha/2)
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(t_stat)
            z_critical = stats.norm.ppf(1 - alpha)
        elif alternative == 'less':
            p_value = stats.norm.cdf(t_stat)
            z_critical = -stats.norm.ppf(1 - alpha)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        
        # Confidence interval (always two-sided)
        ci_lower = theta - stats.norm.ppf(1 - alpha/2) * se_theta
        ci_upper = theta + stats.norm.ppf(1 - alpha/2) * se_theta
        
        # Decision
        if alternative == 'two-sided':
            reject_h0 = abs(t_stat) > z_critical
        elif alternative == 'greater':
            reject_h0 = t_stat > z_critical
        else:  # less
            reject_h0 = t_stat < z_critical
        
        # Generate interpretation
        interpretation = DMLClassicalTest._generate_interpretation(
            theta, se_theta, t_stat, p_value, reject_h0, alternative
        )
        
        result = {
            'theta': float(theta),
            'se': float(se_theta),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'reject_h0': bool(reject_h0),
            'alpha': float(alpha),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'alternative': alternative,
            'interpretation': interpretation
        }
        
        logger.info(f"Classical test: θ={theta:.4f}, SE={se_theta:.4f}, "
                   f"t={t_stat:.2f}, p={p_value:.4f}")
        
        return result
    
    @staticmethod
    def test_biodiversity_hypothesis(
        theta_plants: float,
        se_plants: float,
        theta_vertebrates: float,
        se_vertebrates: float,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Test biodiversity begets biodiversity hypothesis (H1).
        
        Tests whether both plant and vertebrate richness have positive
        causal effects on fungi richness.
        
        Args:
            theta_plants: Causal effect of plants on fungi
            se_plants: Standard error for plants effect
            theta_vertebrates: Causal effect of vertebrates on fungi
            se_vertebrates: Standard error for vertebrates effect
            alpha: Significance level
            
        Returns:
            Dictionary with comprehensive test results
        """
        # Test each effect
        plants_test = DMLClassicalTest.test_causal_effect(
            theta_plants, se_plants, alpha, 'greater'
        )
        vertebrates_test = DMLClassicalTest.test_causal_effect(
            theta_vertebrates, se_vertebrates, alpha, 'greater'
        )
        
        # Joint hypothesis: both effects positive
        both_positive = plants_test['reject_h0'] and vertebrates_test['reject_h0']
        either_positive = plants_test['reject_h0'] or vertebrates_test['reject_h0']
        
        # Combined interpretation
        if both_positive:
            h1_conclusion = "STRONG SUPPORT for H1: Both plant and vertebrate richness cause increased fungi richness"
        elif either_positive:
            if plants_test['reject_h0']:
                h1_conclusion = "PARTIAL SUPPORT for H1: Only plant richness shows significant positive effect"
            else:
                h1_conclusion = "PARTIAL SUPPORT for H1: Only vertebrate richness shows significant positive effect"
        else:
            h1_conclusion = "NO SUPPORT for H1: Neither biodiversity measure shows significant positive effect"
        
        result = {
            'plants_effect': plants_test,
            'vertebrates_effect': vertebrates_test,
            'h1_supported': both_positive,
            'h1_partial_support': either_positive,
            'h1_conclusion': h1_conclusion,
            'summary': {
                'theta_plants': float(theta_plants),
                'p_plants': plants_test['p_value'],
                'theta_vertebrates': float(theta_vertebrates),
                'p_vertebrates': vertebrates_test['p_value'],
                'both_significant': both_positive
            }
        }
        
        logger.info(f"Biodiversity hypothesis test: {h1_conclusion}")
        
        return result
    
    @staticmethod
    def _generate_interpretation(
        theta: float,
        se: float,
        t_stat: float,
        p_value: float,
        reject_h0: bool,
        alternative: str
    ) -> str:
        """Generate human-readable interpretation of test results.
        
        Args:
            theta: Effect estimate
            se: Standard error
            t_stat: T-statistic
            p_value: P-value
            reject_h0: Whether H0 is rejected
            alternative: Type of test
            
        Returns:
            Interpretation string
        """
        # Rule of thumb interpretation
        if abs(t_stat) > 3:
            strength = "very strong"
        elif abs(t_stat) > 2:
            strength = "strong"
        elif abs(t_stat) > 1.5:
            strength = "moderate"
        else:
            strength = "weak"
        
        if reject_h0:
            if alternative == 'greater' and theta > 0:
                interpretation = (
                    f"Significant positive causal effect detected (θ={theta:.3f}, "
                    f"t={t_stat:.2f}, p={p_value:.4f}). "
                    f"Each unit increase causes {theta:.3f} more in outcome. "
                    f"Evidence strength: {strength}."
                )
            elif alternative == 'less' and theta < 0:
                interpretation = (
                    f"Significant negative causal effect detected (θ={theta:.3f}, "
                    f"t={t_stat:.2f}, p={p_value:.4f}). "
                    f"Each unit increase causes {abs(theta):.3f} less in outcome. "
                    f"Evidence strength: {strength}."
                )
            else:
                interpretation = (
                    f"Significant causal effect detected (θ={theta:.3f}, "
                    f"t={t_stat:.2f}, p={p_value:.4f}). "
                    f"Evidence strength: {strength}."
                )
        else:
            interpretation = (
                f"No significant causal effect detected (θ={theta:.3f}, "
                f"t={t_stat:.2f}, p={p_value:.4f}). "
                f"Cannot reject null hypothesis at α={p_value:.2f} level."
            )
        
        return interpretation
    
    @staticmethod
    def test_joint_significance(
        thetas: list,
        ses: list,
        labels: Optional[list] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Test joint significance of multiple parameters.
        
        Uses Bonferroni correction for multiple testing.
        
        Args:
            thetas: List of causal effect estimates
            ses: List of standard errors
            labels: Optional labels for each parameter
            alpha: Overall significance level
            
        Returns:
            Dictionary with joint test results
        """
        n_tests = len(thetas)
        if len(ses) != n_tests:
            raise ValueError("thetas and ses must have same length")
        
        if labels is None:
            labels = [f"Parameter {i+1}" for i in range(n_tests)]
        
        # Bonferroni correction
        alpha_corrected = alpha / n_tests
        
        # Test each parameter
        individual_tests = []
        for i, (theta, se, label) in enumerate(zip(thetas, ses, labels)):
            test = DMLClassicalTest.test_causal_effect(
                theta, se, alpha_corrected, 'two-sided'
            )
            test['label'] = label
            individual_tests.append(test)
        
        # Count significant effects
        n_significant = sum(test['reject_h0'] for test in individual_tests)
        all_significant = n_significant == n_tests
        any_significant = n_significant > 0
        
        result = {
            'individual_tests': individual_tests,
            'n_parameters': n_tests,
            'n_significant': n_significant,
            'all_significant': all_significant,
            'any_significant': any_significant,
            'bonferroni_alpha': float(alpha_corrected),
            'interpretation': (
                f"{n_significant}/{n_tests} parameters significant "
                f"after Bonferroni correction (α={alpha_corrected:.4f})"
            )
        }
        
        return result