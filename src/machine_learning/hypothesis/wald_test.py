"""Wald test implementation for GLM models - SIMPLIFIED VERSION.

No abstractions. Just statistics.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class WaldTest:
    """Wald test for hypothesis testing in GLM models.
    
    The Wald test evaluates: W = (beta - beta_0) / SE(beta) ~ N(0,1)
    """
    
    @staticmethod
    def test(coefficient: float,
             standard_error: float,
             null_value: float = 0.0,
             alternative: str = 'two-sided') -> Dict[str, Any]:
        """Basic Wald test - pure statistics, no frills.
        
        Args:
            coefficient: Estimated coefficient value
            standard_error: Standard error of the coefficient
            null_value: Null hypothesis value (default: 0)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with test statistics and p-value
        """
        if standard_error <= 0:
            raise ValueError("Standard error must be positive")
        
        # Core Wald statistic
        wald_statistic = (coefficient - null_value) / standard_error
        
        # Compute p-value based on alternative
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(wald_statistic)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(wald_statistic)
        elif alternative == 'less':
            p_value = stats.norm.cdf(wald_statistic)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        # 95% CI
        z_critical = stats.norm.ppf(0.975)
        ci_lower = coefficient - z_critical * standard_error
        ci_upper = coefficient + z_critical * standard_error
        
        return {
            'coefficient': coefficient,
            'standard_error': standard_error,
            'wald_statistic': wald_statistic,
            'p_value': p_value,
            'null_value': null_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05,
            'alternative': alternative
        }
    
    @staticmethod
    def test_equivalence(coefficient: float,
                        standard_error: float,
                        lower_bound: float,
                        upper_bound: float,
                        alpha: float = 0.05) -> Dict[str, Any]:
        """TOST (Two One-Sided Tests) for equivalence testing.
        
        Tests if effect is within equivalence bounds.
        This is just 2 Wald tests - no need for 340-line wrapper!
        
        Args:
            coefficient: Estimated coefficient
            standard_error: Standard error
            lower_bound: Lower equivalence bound
            upper_bound: Upper equivalence bound
            alpha: Significance level
            
        Returns:
            Dictionary with TOST results
        """
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be less than upper bound")
        
        # Test 1: H0: θ ≤ lower, H1: θ > lower
        test1 = WaldTest.test(coefficient, standard_error, lower_bound, 'greater')
        
        # Test 2: H0: θ ≥ upper, H1: θ < upper
        test2 = WaldTest.test(coefficient, standard_error, upper_bound, 'less')
        
        # TOST p-value is the maximum of the two
        p_tost = max(test1['p_value'], test2['p_value'])
        is_equivalent = p_tost < alpha
        
        # Also test vs zero for comparison
        test0 = WaldTest.test(coefficient, standard_error, 0, 'two-sided')
        is_different = test0['p_value'] < alpha
        
        # Interpret results
        if is_equivalent and is_different:
            conclusion = "Different but equivalent (negligible effect)"
        elif is_equivalent and not is_different:
            conclusion = "Equivalent to zero"
        elif not is_equivalent and is_different:
            conclusion = "Significant and meaningful"
        else:
            conclusion = "Inconclusive"
        
        return {
            'coefficient': coefficient,
            'standard_error': standard_error,
            'bounds': [lower_bound, upper_bound],
            'p_tost': p_tost,
            'p_zero': test0['p_value'],
            'is_equivalent': is_equivalent,
            'is_different': is_different,
            'conclusion': conclusion,
            # Keep individual test results for transparency
            'test_lower': test1,
            'test_upper': test2,
            'test_zero': test0
        }
    
    @staticmethod
    def test_sesoi(coefficient: float,
                   standard_error: float,
                   threshold: float,
                   direction: str = 'greater') -> Dict[str, Any]:
        """SESOI (Smallest Effect Size of Interest) test.
        
        Tests if effect exceeds meaningful threshold.
        
        Args:
            coefficient: Estimated coefficient
            standard_error: Standard error
            threshold: SESOI threshold (smallest meaningful effect)
            direction: 'greater' (exceeds threshold), 'less', or 'two-sided'
            
        Returns:
            Dictionary with SESOI test results
        """
        # Adjust null value based on direction
        if direction == 'greater':
            null_value = threshold
            alternative = 'greater'
            exceeds = coefficient > threshold
        elif direction == 'less':
            null_value = -threshold
            alternative = 'less'
            exceeds = coefficient < -threshold
        else:  # two-sided
            null_value = 0
            alternative = 'two-sided'
            exceeds = abs(coefficient) > threshold
        
        # Run test with adjusted null
        result = WaldTest.test(coefficient, standard_error, null_value, alternative)
        
        # Add SESOI-specific fields
        result['threshold'] = threshold
        result['exceeds_threshold'] = exceeds and result['significant']
        
        # Add interpretation
        if result['exceeds_threshold']:
            result['interpretation'] = f"Effect exceeds meaningful threshold ({threshold:.3f})"
        else:
            result['interpretation'] = f"Effect below meaningful threshold ({threshold:.3f})"
        
        return result
    
    @staticmethod
    def test_multiple(coefficients: np.ndarray,
                     covariance_matrix: np.ndarray,
                     null_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Joint Wald test for multiple coefficients.
        
        Tests: W = (beta - beta_0)' * Cov^(-1) * (beta - beta_0) ~ χ²(k)
        
        Args:
            coefficients: Vector of coefficient estimates
            covariance_matrix: Covariance matrix of coefficients
            null_values: Null hypothesis values (default: zeros)
            
        Returns:
            Dictionary with chi-squared test results
        """
        coefficients = np.asarray(coefficients)
        k = len(coefficients)
        
        if null_values is None:
            null_values = np.zeros(k)
        else:
            null_values = np.asarray(null_values)
        
        if covariance_matrix.shape != (k, k):
            raise ValueError(f"Covariance matrix shape mismatch")
        
        # Compute Wald statistic
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, using pseudo-inverse")
            cov_inv = np.linalg.pinv(covariance_matrix)
        
        diff = coefficients - null_values
        wald_statistic = float(diff.T @ cov_inv @ diff)
        
        # Chi-squared p-value
        p_value = 1 - stats.chi2.cdf(wald_statistic, df=k)
        
        return {
            'wald_statistic': wald_statistic,
            'chi2_statistic': wald_statistic,  # Same thing
            'degrees_of_freedom': k,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def to_response_scale(coefficient: float, 
                         standard_error: float,
                         link: str = 'log') -> Dict[str, Any]:
        """Convert coefficient to response scale.
        
        For log link: percent change = (exp(coef) - 1) * 100
        
        Args:
            coefficient: Coefficient on link scale
            standard_error: SE on link scale
            link: Link function ('log' or 'identity')
            
        Returns:
            Dictionary with response scale values
        """
        if link == 'identity':
            return {
                'effect': coefficient,
                'ci_lower': coefficient - 1.96 * standard_error,
                'ci_upper': coefficient + 1.96 * standard_error
            }
        elif link == 'log':
            # Percent change for log link
            pct_change = (np.exp(coefficient) - 1) * 100
            pct_lower = (np.exp(coefficient - 1.96 * standard_error) - 1) * 100
            pct_upper = (np.exp(coefficient + 1.96 * standard_error) - 1) * 100
            
            return {
                'percent_change': pct_change,
                'ci_lower_pct': pct_lower,
                'ci_upper_pct': pct_upper,
                'interpretation': f"{pct_change:+.1f}% change [95% CI: {pct_lower:.1f}%, {pct_upper:.1f}%]"
            }
        else:
            raise ValueError(f"Unknown link function: {link}")


# Convenience function for backward compatibility
def wald_test_all_methods(coefficient: float,
                          standard_error: float,
                          sesoi_threshold: float = 0.1,
                          tost_bounds: tuple = (-0.05, 0.05)) -> Dict[str, Any]:
    """Run all three Wald test variants at once.
    
    Returns combined results from traditional, SESOI, and TOST tests.
    """
    return {
        'traditional': WaldTest.test(coefficient, standard_error),
        'sesoi': WaldTest.test_sesoi(coefficient, standard_error, sesoi_threshold),
        'tost': WaldTest.test_equivalence(coefficient, standard_error, 
                                          tost_bounds[0], tost_bounds[1])
    }