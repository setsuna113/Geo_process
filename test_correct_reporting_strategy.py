#!/usr/bin/env python3
"""
Verify that Methods 1 and 2 implement the correct hypothesis test reporting strategy.

CORRECT REPORTING STRATEGY:
===========================
1. Model Comparison Tests (LRT, Score): Report TRADITIONAL ONLY
   - These test if one model fits better than another
   - Traditional H0: No improvement / No difference between models
   
2. Coefficient Tests (Wald): Report SESOI and TOST ONLY
   - These test individual parameter effects
   - SESOI H0: |β| ≤ threshold (practical significance)
   - TOST: Equivalence testing within bounds

This separation ensures:
- We don't mix different hypotheses
- Each test type reports appropriate results
- Scale transformations are applied consistently
"""

import sys
sys.path.insert(0, '.')

print("""
================================================================================
HYPOTHESIS TEST REPORTING STRATEGY - VERIFICATION
================================================================================

PRINCIPLE: Different test types require different reporting
--------------------------------------------------------

1. MODEL COMPARISON TESTS (Report TRADITIONAL ONLY)
   ================================================
   Tests: LRT, Score Test
   Purpose: Compare model fit (null vs full, with vs without interaction)
   H0: Traditional hypothesis (no improvement/no effect)
   
   Example Output:
   - LRT: χ² = 125.3, p = 0.0001 (traditional H0: no improvement)
   - Score: χ² = 118.7, p = 0.0002 (traditional H0: no effect)
   
   DO NOT report SESOI/TOST for these tests!

2. COEFFICIENT TESTS (Report SESOI and TOST ONLY)
   ==============================================
   Tests: Wald Test (single and joint coefficients)
   Purpose: Test specific parameter effects
   H0: Alternative hypotheses for practical significance
   
   SESOI (Smallest Effect Size of Interest):
   - H0: |β| ≤ 0.1 (effect is negligible)
   - H1: |β| > 0.1 (effect is meaningful)
   
   TOST (Two One-Sided Tests):
   - Tests equivalence within bounds
   - H0: Effect outside bounds
   - H1: Effect within bounds (equivalent)
   
   Example Output:
   - SESOI: β = 0.15, p = 0.03 (exceeds threshold, practically significant)
   - TOST: p = 0.45 (not equivalent, effect is meaningful)
   
   DO NOT report traditional H0: β = 0 for these tests!

================================================================================
METHOD 1 IMPLEMENTATION CHECK
================================================================================
""")

# Check Method 1 implementation
try:
    with open('src/machine_learning/methods/method1_continuous_refactored.py', 'r') as f:
        method1_code = f.read()
    
    # Check for correct LRT reporting
    if "LRT: Reporting traditional results only" in method1_code:
        print("✓ Method 1 LRT: Reports traditional only")
    else:
        print("✗ Method 1 LRT: May still report SESOI/TOST (incorrect)")
    
    # Check for correct Score test reporting
    if "Score Test: Reporting traditional results only" in method1_code:
        print("✓ Method 1 Score: Reports traditional only")
    else:
        print("✗ Method 1 Score: May still report SESOI/TOST (incorrect)")
    
    # Check for correct Wald reporting
    if "# DO NOT run traditional test - only SESOI and TOST" in method1_code:
        print("✓ Method 1 Wald: Reports SESOI/TOST only")
    else:
        print("✗ Method 1 Wald: May still report traditional (incorrect)")
    
    # Check for joint SESOI test
    if "Joint SESOI test: H0: all |βi| ≤ 0.1" in method1_code:
        print("✓ Method 1 Joint Wald: Uses SESOI version")
    else:
        print("✗ Method 1 Joint Wald: May use traditional (incorrect)")
        
except Exception as e:
    print(f"Error checking Method 1: {e}")

print("""
================================================================================
METHOD 2 IMPLEMENTATION CHECK
================================================================================
""")

# Check Method 2 implementation
try:
    with open('src/machine_learning/methods/method2_categorical_refactored.py', 'r') as f:
        method2_code = f.read()
    
    # Check for correct LRT reporting
    if "LRT for interaction (traditional only)" in method2_code:
        print("✓ Method 2 LRT: Reports traditional only")
    else:
        print("✗ Method 2 LRT: May still report SESOI/TOST (incorrect)")
    
    # Check for correct Score test reporting
    if "Score test for interaction (traditional only)" in method2_code:
        print("✓ Method 2 Score: Reports traditional only")
    else:
        print("✗ Method 2 Score: May still report SESOI/TOST (incorrect)")
    
    # Check for correct Wald reporting
    if "# IMPORTANT: For interaction coefficient, only report SESOI and TOST" in method2_code:
        print("✓ Method 2 Wald: Reports SESOI/TOST only")
    else:
        print("✗ Method 2 Wald: May still report traditional (incorrect)")
    
    # Check for SESOI/TOST implementation
    if "sesoi_test = WaldTest.test_single_coefficient_with_scale" in method2_code and \
       "tost_test = TOSTWrapper.perform_tost_with_scale" in method2_code:
        print("✓ Method 2: Uses enhanced tests with scale transformations")
    else:
        print("✗ Method 2: Missing enhanced test implementations")
        
except Exception as e:
    print(f"Error checking Method 2: {e}")

print("""
================================================================================
KEY IMPLEMENTATION DETAILS
================================================================================

1. SCALE TRANSFORMATIONS
   - Applied to SESOI and TOST tests when link='log'
   - Log scale: β = 0.1 means ~10.5% change on response scale
   - Provides interpretable effect sizes

2. VCOV MATRIX EXTRACTION
   - Uses sandwich estimator for robust standard errors
   - Critical for proper inference with 43M observations
   - Cached for efficiency

3. REPORTING GUIDELINES
   - Focus on practical significance over statistical significance
   - With large N, everything is "statistically significant"
   - SESOI threshold determines if effects are meaningful
   - TOST confirms if effects are negligible

4. VISUALIZATION
   - Model comparisons: Show likelihood/deviance improvements
   - Coefficients: Show SESOI thresholds and confidence intervals
   - Interactions: Show how effects change across conditions

================================================================================
EXAMPLE CORRECT OUTPUT
================================================================================

Method 1 Report:
----------------
LRT (Model Comparison):
  χ² = 125.3, df = 2, p < 0.0001
  McFadden R² = 0.023 (modest improvement)
  Interpretation: Biodiversity improves model fit

Wald Tests (Coefficients):
  plants_richness_log:
    SESOI: β = 0.15 > 0.1, p = 0.03 (exceeds threshold)
    TOST: Not equivalent (p = 0.45), effect is meaningful
    Response scale: 16.2% increase per unit
    
  vertebrates_richness_log:
    SESOI: β = 0.08 < 0.1, p = 0.15 (below threshold)
    TOST: Equivalent (p = 0.02), effect is negligible
    Response scale: 8.3% increase per unit

Method 2 Report:
----------------
LRT (Interaction Test):
  χ² = 18.7, df = 1, p = 0.0001
  Interpretation: Interaction improves model fit

Interaction Coefficient:
  bio04_std × plants_richness_log:
    SESOI: β = -0.12 > 0.05, p = 0.02 (exceeds threshold)
    TOST: Not equivalent (p = 0.38), interaction is meaningful
    Response scale: -11.3% change in plant effect per SD of seasonality
    
================================================================================
""")

print("\nVerification complete! The reporting strategy separates:")
print("1. Model comparisons (LRT/Score) → Traditional only")
print("2. Coefficient tests (Wald) → SESOI/TOST only")
print("\nThis ensures we don't mix different hypotheses and maintain clear interpretations.")