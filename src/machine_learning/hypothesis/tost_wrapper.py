"""
TOST (Two One-Sided Tests) Wrapper for Equivalence Testing.

This wrapper calls any hypothesis test twice to perform equivalence testing,
ensuring the original test remains pure and unmodified.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np
from scipy import stats


class TOSTWrapper:
    """
    Wrapper that converts any hypothesis test into a TOST equivalence test.
    
    The wrapper:
    1. Calls the test twice with different H0 values
    2. Combines results for equivalence conclusion
    3. Preserves original test purity
    """
    
    @staticmethod
    def perform_tost(
        test_function: Callable,
        effect_estimate: float,
        standard_error: float,
        lower_bound: float,
        upper_bound: float,
        alpha: float = 0.05,
        **test_kwargs
    ) -> Dict[str, Any]:
        """
        Perform TOST equivalence test using any hypothesis test function.
        
        Args:
            test_function: The hypothesis test function to use (e.g., LRT.test)
            effect_estimate: The estimated effect
            standard_error: Standard error of the estimate
            lower_bound: Lower equivalence bound
            upper_bound: Upper equivalence bound
            alpha: Significance level
            **test_kwargs: Additional arguments for the test function
            
        Returns:
            TOST results with both individual tests and combined conclusion
        """
        # Validate bounds
        if lower_bound >= upper_bound:
            raise ValueError(f"Lower bound ({lower_bound}) must be less than upper bound ({upper_bound})")
        if lower_bound > 0 or upper_bound < 0:
            raise ValueError("Equivalence bounds should typically include zero")
        
        # Test 1: H0: θ ≤ lower_bound, H1: θ > lower_bound
        # This tests if the effect is significantly ABOVE the lower bound
        test1_h0 = {
            'null_value': lower_bound,
            'alternative': 'greater',
            'description': f'Effect ≤ {lower_bound}'
        }
        
        # Test 2: H0: θ ≥ upper_bound, H1: θ < upper_bound  
        # This tests if the effect is significantly BELOW the upper bound
        test2_h0 = {
            'null_value': upper_bound,
            'alternative': 'less',
            'description': f'Effect ≥ {upper_bound}'
        }
        
        # Run both one-sided tests
        # Note: WaldTest uses 'coefficient' not 'effect_estimate'
        test1_result = test_function(
            coefficient=effect_estimate,  # Map to correct parameter name
            standard_error=standard_error,
            h0_definition=test1_h0,
            **test_kwargs
        )
        
        test2_result = test_function(
            coefficient=effect_estimate,  # Map to correct parameter name
            standard_error=standard_error,
            h0_definition=test2_h0,
            **test_kwargs
        )
        
        # TOST logic: equivalence is established if BOTH tests reject their H0
        p1 = test1_result.get('p_value', 1.0)
        p2 = test2_result.get('p_value', 1.0)
        
        # The TOST p-value is the maximum of the two p-values
        p_tost = max(p1, p2)
        
        # Equivalence is established if p_tost < alpha
        is_equivalent = p_tost < alpha
        
        # Also run standard test against zero for comparison
        test0_h0 = {
            'null_value': 0,
            'alternative': 'two-sided',
            'description': 'Effect = 0'
        }
        
        test0_result = test_function(
            coefficient=effect_estimate,  # Map to correct parameter name
            standard_error=standard_error,
            h0_definition=test0_h0,
            **test_kwargs
        )
        
        is_different_from_zero = test0_result.get('p_value', 1.0) < alpha
        
        # Determine conclusion based on four possible outcomes
        if is_equivalent and is_different_from_zero:
            conclusion = 'Statistically different but practically equivalent'
            interpretation = (
                f"The effect ({effect_estimate:.3f}) is statistically different from zero "
                f"but lies within the equivalence bounds [{lower_bound}, {upper_bound}]. "
                "This suggests a real but negligible effect."
            )
        elif is_equivalent and not is_different_from_zero:
            conclusion = 'Statistically and practically equivalent'
            interpretation = (
                f"The effect ({effect_estimate:.3f}) is both statistically non-significant "
                f"and within the equivalence bounds [{lower_bound}, {upper_bound}]. "
                "Strong evidence for no meaningful effect."
            )
        elif not is_equivalent and is_different_from_zero:
            conclusion = 'Statistically and practically significant'
            interpretation = (
                f"The effect ({effect_estimate:.3f}) is statistically significant "
                f"and outside the equivalence bounds [{lower_bound}, {upper_bound}]. "
                "Clear evidence of a meaningful effect."
            )
        else:  # not equivalent and not different
            conclusion = 'Inconclusive'
            interpretation = (
                f"The effect ({effect_estimate:.3f}) is neither statistically significant "
                f"nor demonstrably within the equivalence bounds [{lower_bound}, {upper_bound}]. "
                "Insufficient evidence to conclude equivalence or difference."
            )
        
        return {
            'method': 'TOST',
            'effect_estimate': effect_estimate,
            'standard_error': standard_error,
            'bounds': {
                'lower': lower_bound,
                'upper': upper_bound
            },
            'test1_above_lower': {
                'h0': f'θ ≤ {lower_bound}',
                'p_value': p1,
                'reject': p1 < alpha,
                'result': test1_result
            },
            'test2_below_upper': {
                'h0': f'θ ≥ {upper_bound}',
                'p_value': p2,
                'reject': p2 < alpha,
                'result': test2_result
            },
            'test0_vs_zero': {
                'h0': 'θ = 0',
                'p_value': test0_result.get('p_value', 1.0),
                'reject': is_different_from_zero,
                'result': test0_result
            },
            'tost_p_value': p_tost,
            'alpha': alpha,
            'is_equivalent': is_equivalent,
            'is_different_from_zero': is_different_from_zero,
            'conclusion': conclusion,
            'interpretation': interpretation
        }
    
    @staticmethod
    def perform_tost_with_scale(
        test_function: Callable,
        effect_estimate: float,
        standard_error: float,
        lower_bound: float,
        upper_bound: float,
        alpha: float = 0.05,
        link: str = None,
        **test_kwargs
    ) -> Dict[str, Any]:
        """
        Perform TOST with optional scale transformations for GLM models.
        
        This ENHANCES the existing TOST by adding scale transformations when link='log'.
        It doesn't duplicate the test logic, just adds scale interpretation.
        
        Args:
            test_function: The hypothesis test function to use
            effect_estimate: The estimated effect
            standard_error: Standard error of the estimate
            lower_bound: Lower equivalence bound (on link scale)
            upper_bound: Upper equivalence bound (on link scale)
            alpha: Significance level
            link: Link function ('log', 'identity', None). If 'log', adds scale transformations
            **test_kwargs: Additional arguments for the test function
            
        Returns:
            TOST results with optional scale transformations
        """
        # First, run the standard TOST
        result = TOSTWrapper.perform_tost(
            test_function=test_function,
            effect_estimate=effect_estimate,
            standard_error=standard_error,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            alpha=alpha,
            **test_kwargs
        )
        
        # If log link, add scale transformations
        if link == 'log':
            from ..hypothesis.scale_utils import ScaleTransformation
            
            # Transform bounds to response scale
            lower_pct = (np.exp(lower_bound) - 1) * 100
            upper_pct = (np.exp(upper_bound) - 1) * 100
            
            result['bounds_response_scale'] = {
                'lower_percent': lower_pct,
                'upper_percent': upper_pct,
                'interpretation': f"Equivalence bounds: [{lower_pct:.1f}%, {upper_pct:.1f}%]"
            }
            
            # Transform effect to response scale
            from .scale_utils import ScaleTransformation
            scale_info = ScaleTransformation.log_to_response_scale(
                coefficient=effect_estimate,
                standard_error=standard_error,
                confidence_level=1 - alpha
            )
            result['effect_response_scale'] = scale_info['percent_change']
            result['scale_transformation'] = scale_info
            result['link_function'] = 'log'
            
            # Check equivalence on response scale
            pct_change = scale_info['percent_change']['estimate']
            within_bounds_response = lower_pct <= pct_change <= upper_pct
            result['equivalent_response_scale'] = within_bounds_response
            
            # Enhanced interpretation
            if result['is_equivalent'] and within_bounds_response:
                result['scale_interpretation'] = (
                    f"Effect ({pct_change:.1f}%) is within equivalence bounds "
                    f"[{lower_pct:.1f}%, {upper_pct:.1f}%] on response scale"
                )
            elif not result['is_equivalent'] and not within_bounds_response:
                result['scale_interpretation'] = (
                    f"Effect ({pct_change:.1f}%) exceeds equivalence bounds "
                    f"[{lower_pct:.1f}%, {upper_pct:.1f}%] on response scale"
                )
            else:
                result['scale_interpretation'] = (
                    f"Inconsistent results between scales - check model assumptions"
                )
        
        return result
    
    @staticmethod
    def validate_tost_setup(
        lower_bound: float,
        upper_bound: float,
        effect_estimate: float,
        standard_error: float,
        n_obs: int
    ) -> Dict[str, Any]:
        """
        Validate TOST setup and provide recommendations.
        
        Args:
            lower_bound: Lower equivalence bound
            upper_bound: Upper equivalence bound
            effect_estimate: Current effect estimate
            standard_error: Standard error
            n_obs: Sample size
            
        Returns:
            Validation results and recommendations
        """
        issues = []
        recommendations = []
        
        # Check bound validity
        if lower_bound >= upper_bound:
            issues.append("Lower bound must be less than upper bound")
            
        if lower_bound > 0:
            issues.append("Lower bound is positive - typically should be negative")
            recommendations.append(f"Consider using lower_bound = {-abs(upper_bound)}")
            
        if upper_bound < 0:
            issues.append("Upper bound is negative - typically should be positive")
            recommendations.append(f"Consider using upper_bound = {abs(lower_bound)}")
            
        # Check if bounds are symmetric
        if abs(lower_bound + upper_bound) > 1e-10:
            recommendations.append(
                f"Bounds are asymmetric. Consider symmetric bounds: "
                f"[{-max(abs(lower_bound), abs(upper_bound))}, {max(abs(lower_bound), abs(upper_bound))}]"
            )
        
        # Check if bounds are too narrow given precision
        bound_width = upper_bound - lower_bound
        min_detectable_effect = 2 * standard_error  # Rough approximation
        
        if bound_width < min_detectable_effect:
            issues.append(
                f"Bounds may be too narrow. Width ({bound_width:.3f}) < "
                f"minimum detectable effect (~{min_detectable_effect:.3f})"
            )
            recommendations.append(
                f"Consider wider bounds or increase sample size"
            )
        
        # Check power considerations
        if n_obs < 100:
            recommendations.append(
                "Small sample size may limit power for equivalence testing"
            )
        
        # Check if effect is already within bounds
        if lower_bound < effect_estimate < upper_bound:
            info = f"Current estimate ({effect_estimate:.3f}) is within bounds"
        else:
            info = f"Current estimate ({effect_estimate:.3f}) is outside bounds"
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'info': info,
            'bounds_width': bound_width,
            'relative_precision': bound_width / (2 * standard_error) if standard_error > 0 else np.inf
        }