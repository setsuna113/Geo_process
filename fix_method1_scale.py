#!/usr/bin/env python3
"""
Fix Method 1 to properly handle scale transformations for SESOI/TOST tests.
"""

import sys
import os
sys.path.insert(0, '.')

from src.machine_learning.hypothesis.scale_reporter import ScaleAwareReporter
from src.machine_learning.hypothesis.scale_utils import ScaleTransformation
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def patch_method1_wald_results():
    """
    Patch Method 1's Wald test results to include proper scale transformations.
    
    This demonstrates how Method 1 should handle coefficients on log scale.
    """
    
    # Example coefficients from Method 1 (these would come from the actual model)
    # These are on LOG SCALE since Method 1 uses negative binomial GLM with log link
    example_coefficients = {
        'fungi_richness': 0.12,  # 12% increase on log scale
        'bacteria_richness': 0.08,  # 8% increase on log scale
        'protist_richness': -0.05,  # 5% decrease on log scale
    }
    
    example_standard_errors = {
        'fungi_richness': 0.03,
        'bacteria_richness': 0.02,
        'protist_richness': 0.04,
    }
    
    # SESOI threshold on log scale (as Method 1 currently uses)
    sesoi_log_scale = 0.1  # 0.1 on log scale ≈ 10.5% change on response scale
    
    # Generate proper scale-aware report
    report = ScaleAwareReporter.report_multiple_coefficients(
        coefficients=example_coefficients,
        standard_errors=example_standard_errors,
        sesoi_log_scale=sesoi_log_scale,
        link='log',  # Method 1 uses log link for negative binomial
        perform_tost=True,
        equivalence_bounds=(-0.1, 0.1)  # Symmetric bounds on log scale
    )
    
    # Print summary
    print("=" * 80)
    print("METHOD 1 SCALE-AWARE REPORTING DEMONSTRATION")
    print("=" * 80)
    
    # Show SESOI interpretation on both scales
    print("\nSESOI THRESHOLD INTERPRETATION:")
    print(f"  Log scale: {sesoi_log_scale}")
    sesoi_pct = report['sesoi_info']['percent_change']
    print(f"  Response scale: {sesoi_pct:.1f}% change")
    print(f"  Interpretation: Effects need >{sesoi_pct:.1f}% change to be practically meaningful")
    
    print("\n" + "=" * 80)
    print("COEFFICIENT RESULTS WITH PROPER SCALE HANDLING:")
    print("=" * 80)
    
    for var_name, results in report['coefficients'].items():
        print(f"\n{var_name}:")
        print("-" * 40)
        
        # Traditional test
        tt = results['traditional_test']
        print(f"  Statistical significance:")
        print(f"    p-value: {tt['p_value']:.4f}")
        print(f"    Significant at α=0.05: {'Yes' if tt['p_value'] < 0.05 else 'No'}")
        
        # Scale transformation
        if 'scale_transformation' in results:
            st = results['scale_transformation']
            pct = st['percent_change']
            print(f"\n  Effect on response scale:")
            print(f"    Multiplicative effect: {st['response_scale']['multiplicative_effect']:.3f}")
            print(f"    Percent change: {pct['estimate']:+.1f}%")
            print(f"    95% CI: [{pct['ci_lower']:.1f}%, {pct['ci_upper']:.1f}%]")
        
        # Practical significance
        ps = results['practical_significance']
        print(f"\n  Practical significance (SESOI):")
        print(f"    Log scale: {ps['log_scale']['interpretation']}")
        print(f"    Response scale: {ps['response_scale']['interpretation']}")
        print(f"    Exceeds threshold: {'Yes' if ps['response_scale']['exceeds'] else 'No'}")
        
        # TOST results
        if 'tost' in results:
            tost = results['tost']
            print(f"\n  Equivalence test (TOST):")
            print(f"    p-value: {tost['tost_p_value']:.4f}")
            print(f"    Conclusion: {tost['conclusion']}")
        
        # Overall conclusion
        print(f"\n  Overall: {results['conclusion']}")
    
    # Create publication table
    print("\n" + "=" * 80)
    print("PUBLICATION-READY TABLE:")
    print("=" * 80)
    table = ScaleAwareReporter.create_publication_table(report, format='markdown')
    print(table)
    
    # Save enhanced report
    output_path = '/home/yl998/Geo_process/method1_scale_aware_report.json'
    with open(output_path, 'w') as f:
        # Convert to serializable format
        def make_serializable(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        json.dump(make_serializable(report), f, indent=2)
    
    print(f"\n✓ Enhanced report saved to: {output_path}")
    
    # Show how Method 1 SHOULD interpret its results
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR METHOD 1:")
    print("=" * 80)
    print("""
1. SESOI of 0.1 on log scale means ~10.5% change on response scale
   - This is the threshold for "meaningful" biodiversity effects
   
2. With log-link negative binomial GLM:
   - Coefficient β = 0.1 means exp(0.1) = 1.105 = 10.5% increase
   - Coefficient β = 0.2 means exp(0.2) = 1.221 = 22.1% increase
   
3. Method 1 currently tests if |β| > 0.1 on log scale
   - This is correct but needs proper interpretation
   - Should report BOTH log scale and % change
   
4. Confidence intervals should be transformed:
   - Log scale CI: [β ± 1.96*SE]
   - Response scale CI: [exp(lower), exp(upper)]
   - Percent change CI: [(exp(lower)-1)*100, (exp(upper)-1)*100]
   
5. TOST with bounds [-0.1, 0.1] on log scale means:
   - Testing if effect is between -9.5% and +10.5% change
   - If TOST passes, effect is "practically equivalent to zero"
""")
    
    return report


def generate_method1_fix_code():
    """Generate the actual code changes needed for Method 1."""
    
    fix_code = '''
# Add this import at the top of method1_continuous_refactored.py
from src.machine_learning.hypothesis.scale_reporter import ScaleAwareReporter

# Replace the Wald test section (around line 320-340) with:

# Wald tests with proper scale handling
logger.info("Performing Wald tests with scale-aware reporting...")
biotic_cols = ['fungi_richness', 'bacteria_richness', 'protist_richness']

# Extract coefficients and SEs
coefficients_dict = {}
se_dict = {}
for coef_name in biotic_cols:
    if coef_name in coef_table.columns:
        coefficients_dict[coef_name] = float(coef_table[coef_name].values[0])
        # Get SE from model (this needs proper extraction from H2O)
        se_dict[coef_name] = 0.1  # Placeholder - extract from vcov

# Generate scale-aware report
scale_report = ScaleAwareReporter.report_multiple_coefficients(
    coefficients=coefficients_dict,
    standard_errors=se_dict,
    sesoi_log_scale=0.1,  # SESOI on log scale
    link='log',  # Negative binomial uses log link
    perform_tost=True,
    equivalence_bounds=(-0.1, 0.1)
)

# Update wald_results with scale-aware results
wald_results = {}
for var_name, var_results in scale_report['coefficients'].items():
    wald_results[f'{var_name}_sesoi'] = var_results['sesoi_test']
    wald_results[f'{var_name}_tost'] = var_results.get('tost', {})
    wald_results[f'{var_name}_scale'] = var_results.get('scale_transformation', {})

# In the interpretation section, add:
logger.info("Scale transformation summary:")
for var_name, var_results in scale_report['coefficients'].items():
    if 'scale_transformation' in var_results:
        pct = var_results['scale_transformation']['percent_change']
        logger.info(f"  {var_name}: {pct['estimate']:+.1f}% change "
                   f"(95% CI: [{pct['ci_lower']:.1f}%, {pct['ci_upper']:.1f}%])")
'''
    
    print("\n" + "=" * 80)
    print("CODE CHANGES FOR METHOD 1:")
    print("=" * 80)
    print(fix_code)
    
    # Save fix code
    fix_path = '/home/yl998/Geo_process/method1_scale_fix.txt'
    with open(fix_path, 'w') as f:
        f.write(fix_code)
    
    print(f"\n✓ Fix code saved to: {fix_path}")


if __name__ == "__main__":
    print("Demonstrating proper scale handling for Method 1...")
    report = patch_method1_wald_results()
    generate_method1_fix_code()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("""
1. Apply the fix code to method1_continuous_refactored.py
2. Extract proper standard errors from H2O model's vcov
3. Update reporting to show both log scale and response scale
4. Fix Method 2 to use the same scale-aware reporting
5. Ensure CI computation uses proper scale transformations
""")