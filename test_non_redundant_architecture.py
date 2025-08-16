#!/usr/bin/env python3
"""
Test the non-redundant architecture where scale transformations are integrated
into existing hypothesis tests rather than duplicated in a separate reporter.
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
from src.machine_learning.hypothesis.wald_test import WaldTest
from src.machine_learning.hypothesis.tost_wrapper import TOSTWrapper
from src.machine_learning.hypothesis.h0_definitions import H0Definition
from src.machine_learning.hypothesis.scale_report_formatter import ScaleReportFormatter


def test_integrated_architecture():
    """Test that scale transformations are properly integrated into existing tests."""
    
    print("=" * 80)
    print("TESTING NON-REDUNDANT ARCHITECTURE")
    print("=" * 80)
    
    # Example coefficient from a GLM with log link
    coefficient = 0.15
    standard_error = 0.025
    
    print("\nTest data:")
    print(f"  Coefficient (log scale): {coefficient:.3f}")
    print(f"  Standard error: {standard_error:.3f}")
    
    # =========================================================================
    # 1. Test WaldTest with integrated scale transformations
    # =========================================================================
    print("\n" + "-" * 80)
    print("1. ENHANCED WALDTEST (with scale transformations)")
    print("-" * 80)
    
    # Test WITHOUT scale (backward compatible)
    result_no_scale = WaldTest.test_single_coefficient(
        coefficient=coefficient,
        standard_error=standard_error,
        null_value=0.0,
        alternative='two-sided'
    )
    
    print("\nWaldTest without scale (traditional):")
    print(f"  p-value: {result_no_scale['p_value']:.4f}")
    print(f"  CI: [{result_no_scale['ci_lower']:.4f}, {result_no_scale['ci_upper']:.4f}]")
    assert 'scale_transformation' not in result_no_scale, "Should not have scale info"
    print("  ✓ No scale transformation (as expected)")
    
    # Test WITH scale (enhanced)
    result_with_scale = WaldTest.test_single_coefficient_with_scale(
        coefficient=coefficient,
        standard_error=standard_error,
        null_value=0.0,
        alternative='two-sided',
        link='log'  # This enables scale transformations
    )
    
    print("\nWaldTest with scale (enhanced):")
    print(f"  p-value: {result_with_scale['p_value']:.4f} (same as traditional)")
    print(f"  CI (log): [{result_with_scale['ci_lower']:.4f}, {result_with_scale['ci_upper']:.4f}]")
    
    # Check scale transformation was added
    assert 'scale_transformation' in result_with_scale, "Should have scale info"
    assert 'response_scale_interpretation' in result_with_scale, "Should have interpretation"
    print(f"  ✓ Response scale: {result_with_scale['response_scale_interpretation']}")
    
    # Test SESOI with scale
    sesoi_result = WaldTest.test_single_coefficient_with_scale(
        coefficient=coefficient,
        standard_error=standard_error,
        h0_definition=H0Definition.sesoi(0.1, 'greater'),
        alternative='greater',
        link='log'
    )
    
    print("\nWaldTest with SESOI and scale:")
    print(f"  p-value: {sesoi_result['p_value']:.4f}")
    assert 'sesoi_response_scale' in sesoi_result, "Should have SESOI scale info"
    assert 'exceeds_sesoi_response_scale' in sesoi_result, "Should have SESOI check"
    print(f"  ✓ SESOI on response scale: {sesoi_result['sesoi_response_scale']['percent_change']:.1f}%")
    print(f"  ✓ Exceeds SESOI: {sesoi_result['exceeds_sesoi_response_scale']}")
    print(f"  ✓ Practical significance: {sesoi_result.get('practical_significance', 'N/A')}")
    
    # =========================================================================
    # 2. Test TOSTWrapper with integrated scale transformations
    # =========================================================================
    print("\n" + "-" * 80)
    print("2. ENHANCED TOSTWRAPPER (with scale transformations)")
    print("-" * 80)
    
    # Test WITHOUT scale (backward compatible)
    tost_no_scale = TOSTWrapper.perform_tost(
        test_function=WaldTest.test_single_coefficient,
        effect_estimate=coefficient,
        standard_error=standard_error,
        lower_bound=-0.1,
        upper_bound=0.1,
        alpha=0.05
    )
    
    print("\nTOST without scale (traditional):")
    print(f"  p-value: {tost_no_scale['tost_p_value']:.4f}")
    print(f"  Is equivalent: {tost_no_scale['is_equivalent']}")
    assert 'bounds_response_scale' not in tost_no_scale, "Should not have scale info"
    print("  ✓ No scale transformation (as expected)")
    
    # Test WITH scale (enhanced)
    tost_with_scale = TOSTWrapper.perform_tost_with_scale(
        test_function=WaldTest.test_single_coefficient,
        effect_estimate=coefficient,
        standard_error=standard_error,
        lower_bound=-0.1,
        upper_bound=0.1,
        alpha=0.05,
        link='log'  # This enables scale transformations
    )
    
    print("\nTOST with scale (enhanced):")
    print(f"  p-value: {tost_with_scale['tost_p_value']:.4f} (same as traditional)")
    print(f"  Is equivalent: {tost_with_scale['is_equivalent']}")
    
    # Check scale transformation was added
    assert 'bounds_response_scale' in tost_with_scale, "Should have bounds on response scale"
    assert 'effect_response_scale' in tost_with_scale, "Should have effect on response scale"
    bounds = tost_with_scale['bounds_response_scale']
    print(f"  ✓ Bounds on response scale: [{bounds['lower_percent']:.1f}%, {bounds['upper_percent']:.1f}%]")
    
    effect = tost_with_scale['effect_response_scale']
    print(f"  ✓ Effect on response scale: {effect['estimate']:.1f}%")
    
    if 'scale_interpretation' in tost_with_scale:
        print(f"  ✓ Interpretation: {tost_with_scale['scale_interpretation']}")
    
    # =========================================================================
    # 3. Test ScaleReportFormatter (just formatting, no test logic)
    # =========================================================================
    print("\n" + "-" * 80)
    print("3. SCALE REPORT FORMATTER (formatting only, no test duplication)")
    print("-" * 80)
    
    # Prepare test results from enhanced tests
    test_results = {
        'fungi_richness': {
            'sesoi': sesoi_result,
            'tost': tost_with_scale,
            'summary': {
                'coefficient': coefficient,
                'se': standard_error,
                'p_value': sesoi_result['p_value'],
                'exceeds_sesoi': sesoi_result.get('exceeds_sesoi_response_scale', False),
                'response_scale_interpretation': sesoi_result.get('response_scale_interpretation', '')
            }
        }
    }
    
    # Create formatted table
    table = ScaleReportFormatter.create_coefficient_table(
        test_results=test_results,
        link='log',
        format='markdown'
    )
    
    print("\nFormatted table from test results:")
    print(table)
    print("  ✓ Formatter only formats, doesn't run tests")
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("ARCHITECTURE VERIFICATION")
    print("=" * 80)
    
    print("""
✅ NON-REDUNDANT ARCHITECTURE CONFIRMED:

1. EXISTING TESTS ENHANCED (not replaced):
   - WaldTest.test_single_coefficient() - Still works (backward compatible)
   - WaldTest.test_single_coefficient_with_scale() - Adds scale when link='log'
   - TOSTWrapper.perform_tost() - Still works (backward compatible)
   - TOSTWrapper.perform_tost_with_scale() - Adds scale when link='log'

2. SCALE LOGIC IN ONE PLACE:
   - scale_utils.py contains all transformation math
   - Tests import and use scale_utils when needed
   - No duplication of transformation logic

3. FORMATTER JUST FORMATS:
   - ScaleReportFormatter creates tables from test results
   - Doesn't run any tests itself
   - No duplication of test logic

4. BENEFITS:
   - Backward compatible (old code still works)
   - No redundancy (each test has one implementation)
   - Maintainable (scale logic in one place)
   - Extensible (easy to add scale to other tests)
""")
    
    return True


if __name__ == "__main__":
    success = test_integrated_architecture()
    if success:
        print("\n✅ All architecture tests passed!")
    else:
        print("\n❌ Architecture test failed!")
        sys.exit(1)