#!/usr/bin/env python3
"""
Fix the redundancy by integrating scale transformations into existing hypothesis tests.
"""

print("""
REDUNDANCY ANALYSIS
==================

CURRENT PROBLEMS:
1. ScaleAwareReporter RE-IMPLEMENTS WaldTest instead of enhancing it
2. ScaleAwareReporter creates its own TOST instead of using TOSTWrapper
3. Each hypothesis test would need its own "scale-aware" version = massive duplication

THE RIGHT FIX:
=============

Instead of ScaleAwareReporter replacing tests, we should:

1. Keep scale_utils.py as pure transformation utilities
2. Enhance EXISTING tests to optionally use scale_utils
3. Delete ScaleAwareReporter or make it a thin wrapper

Here's the proper integration:
""")

# Generate the fix code
fix_code = '''
# ==============================================================================
# FIX 1: Enhance WaldTest to use scale transformations
# ==============================================================================
# In wald_test.py, add this method to WaldTest class:

@staticmethod
def test_single_coefficient_with_scale(
    coefficient: float,
    standard_error: float,
    null_value: float = 0.0,
    h0_definition: Optional[Dict[str, Any]] = None,
    alternative: str = 'two-sided',
    link: str = None,  # NEW: specify link function
    report_scale: bool = True  # NEW: whether to add scale transformations
) -> Dict[str, Any]:
    """Test coefficient with optional scale transformations.
    
    This enhances the existing test, doesn't replace it!
    """
    # First, run the standard test
    result = WaldTest.test_single_coefficient(
        coefficient=coefficient,
        standard_error=standard_error,
        null_value=null_value,
        h0_definition=h0_definition,
        alternative=alternative
    )
    
    # If link function specified, add scale transformations
    if link == 'log' and report_scale:
        from ..hypothesis.scale_utils import ScaleTransformation
        
        # Add scale transformation to existing result
        scale_info = ScaleTransformation.log_to_response_scale(
            coefficient=coefficient,
            standard_error=standard_error,
            confidence_level=0.95
        )
        result['scale_transformation'] = scale_info
        
        # Add interpretation on response scale
        pct_change = scale_info['percent_change']
        result['response_scale_interpretation'] = (
            f"{pct_change['estimate']:+.1f}% change "
            f"(95% CI: [{pct_change['ci_lower']:.1f}%, {pct_change['ci_upper']:.1f}%])"
        )
        
        # If SESOI test, also transform the threshold
        if h0_definition and h0_definition.get('type') == 'sesoi':
            sesoi_log = h0_definition.get('null_value', 0.1)
            sesoi_info = ScaleTransformation.adjust_sesoi_for_scale(sesoi_log, link)
            result['sesoi_response_scale'] = sesoi_info
    
    return result

# ==============================================================================
# FIX 2: Enhance TOSTWrapper to handle scale transformations
# ==============================================================================
# In tost_wrapper.py, add scale parameter:

@staticmethod
def perform_tost_with_scale(
    test_function: Callable,
    effect_estimate: float,
    standard_error: float,
    lower_bound: float,
    upper_bound: float,
    alpha: float = 0.05,
    link: str = None,  # NEW: specify link function
    **test_kwargs
) -> Dict[str, Any]:
    """TOST with optional scale transformations."""
    
    # Run standard TOST
    result = TOSTWrapper.perform_tost(
        test_function=test_function,
        effect_estimate=effect_estimate,
        standard_error=standard_error,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        alpha=alpha,
        **test_kwargs
    )
    
    # Add scale transformations if link specified
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
        scale_info = ScaleTransformation.log_to_response_scale(
            coefficient=effect_estimate,
            standard_error=standard_error
        )
        result['effect_response_scale'] = scale_info['percent_change']
    
    return result

# ==============================================================================
# FIX 3: Update Method 1 to use enhanced tests (not ScaleAwareReporter)
# ==============================================================================
# In method1_continuous_refactored.py:

# Instead of:
scale_report = ScaleAwareReporter.report_multiple_coefficients(...)

# Use:
for coef_name in biotic_cols:
    if coef_name in coef_dict:
        # Use enhanced WaldTest with scale
        wald_result = WaldTest.test_single_coefficient_with_scale(
            coefficient=coef_dict[coef_name],
            standard_error=se_dict[coef_name],
            h0_definition=H0Definition.sesoi(0.1, 'greater'),
            alternative='greater',
            link='log',  # Enables scale transformations
            report_scale=True
        )
        
        # Use enhanced TOST with scale
        tost_result = TOSTWrapper.perform_tost_with_scale(
            test_function=WaldTest.test_single_coefficient,
            effect_estimate=coef_dict[coef_name],
            standard_error=se_dict[coef_name],
            lower_bound=-0.1,
            upper_bound=0.1,
            link='log'  # Enables scale transformations
        )
        
        wald_results[coef_name] = {
            'wald': wald_result,
            'tost': tost_result
        }
'''

print(fix_code)

print("""
# ==============================================================================
# BENEFITS OF THIS APPROACH:
# ==============================================================================

1. NO DUPLICATION: Each test has ONE implementation with optional scale support
2. BACKWARD COMPATIBLE: Existing calls still work (link=None means no scale)
3. CONSISTENT: All tests use same scale_utils for transformations
4. MAINTAINABLE: Scale logic in one place (scale_utils), used by all tests
5. EXTENSIBLE: Easy to add scale support to other tests (LRT, Score, etc.)

# ==============================================================================
# WHAT TO DO WITH ScaleAwareReporter?
# ==============================================================================

Option 1: DELETE IT (it's redundant now)
Option 2: Make it a thin convenience wrapper that just calls enhanced tests
Option 3: Rename it to ScaleReportFormatter for creating publication tables

The key insight: Don't REPLACE existing tests, ENHANCE them!
""")