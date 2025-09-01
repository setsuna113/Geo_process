from src.abstractions import get_logger
"""Publication-ready report generation for hypothesis tests.

Extracted from various hypothesis test files.
Consolidates all report generation in one place.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
import logging

logger = get_logger(__name__)


def generate_wald_publication_report(
    test_results: Dict[str, Any],
    test_name: str = "Wald Test",
    hypothesis: str = "coefficient significance",
    variable_names: Optional[List[str]] = None,
    method_number: Optional[int] = None,
    sesoi_threshold: Optional[float] = None,
    tost_bounds: Optional[tuple] = None
) -> Dict[str, Any]:
    """Generate publication-ready report for Wald test results.
    
    Supports standard Wald, SESOI (practical significance), and TOST (equivalence) tests.
    Following ecology journal standards for statistical reporting.
    
    Args:
        test_results: Wald test results dictionary (can be from WaldMethod with SESOI/TOST)
        test_name: Name of the test for reporting
        hypothesis: Description of hypothesis being tested
        variable_names: Names of variables tested (or extract from results)
        method_number: Method number for cross-reference
        sesoi_threshold: SESOI threshold if using practical significance
        tost_bounds: TOST bounds if using equivalence testing
        
    Returns:
        Dictionary with formatted text, tables, and LaTeX code
    """
    # Handle both old format (flat) and new format (nested with SESOI/TOST)
    if 'wald_tests' in test_results:
        # New format from WaldMethod with SESOI/TOST
        wald_tests = test_results['wald_tests']
        has_sesoi = any('sesoi_test' in v for v in wald_tests.values() if isinstance(v, dict))
        has_tost = any('tost_test' in v for v in wald_tests.values() if isinstance(v, dict))
    else:
        # Old format - flat structure
        wald_tests = test_results
        has_sesoi = False
        has_tost = False
        chi2_stat = test_results.get('chi2_statistic', test_results.get('statistic'))
        p_value = test_results.get('p_value')
        df = test_results.get('df', test_results.get('degrees_of_freedom'))
        coefficients = test_results.get('coefficients', {})
        std_errors = test_results.get('std_errors', {})
    
    # Create results table - handle SESOI/TOST format
    rows = []
    
    if has_sesoi or has_tost:
        # New format with SESOI/TOST
        for var_name, var_result in wald_tests.items():
            if not isinstance(var_result, dict) or var_name == 'overall':
                continue
                
            row = {
                'Variable': var_name.replace('_', ' ').title(),
                'Coefficient': f"{var_result.get('coefficient', 0):.4f}",
                'SE': f"{var_result.get('std_error', 0):.4f}",
                'Wald χ²': f"{var_result.get('wald_statistic', 0):.2f}",
                'p-value': f"{var_result.get('p_value', 1):.4f}"
            }
            
            # Add SESOI results if available
            if 'sesoi_test' in var_result:
                sesoi = var_result['sesoi_test']
                row['Exceeds SESOI'] = 'Yes' if sesoi.get('exceeds_threshold', False) else 'No'
                row['Practical Sig.'] = sesoi.get('interpretation', 'N/A')
            elif 'summary' in var_result and 'exceeds_sesoi' in var_result['summary']:
                row['Exceeds SESOI'] = 'Yes' if var_result['summary']['exceeds_sesoi'] else 'No'
            
            # Add TOST results if available
            if 'tost_test' in var_result:
                tost = var_result['tost_test']
                row['Equivalent'] = 'Yes' if tost.get('is_equivalent', False) else 'No'
                row['TOST Result'] = tost.get('conclusion', 'N/A')
            elif 'summary' in var_result and 'is_equivalent' in var_result['summary']:
                row['Equivalent'] = 'Yes' if var_result['summary']['is_equivalent'] else 'No'
            
            rows.append(row)
    else:
        # Old format - traditional Wald test
        if variable_names and coefficients:
            for var in variable_names:
                if var in coefficients:
                    row = {
                        'Variable': var,
                        'Coefficient': f"{coefficients[var]:.4f}",
                        'SE': f"{std_errors.get(var, 'NA'):.4f}" if var in std_errors else 'NA',
                        'Wald χ²': f"{test_results.get(f'{var}_wald_stat', 'NA'):.2f}" 
                                  if f'{var}_wald_stat' in test_results else 'NA',
                        'p-value': f"{test_results.get(f'{var}_p_value', 'NA'):.4f}"
                                  if f'{var}_p_value' in test_results else 'NA'
                    }
                    rows.append(row)
    
    results_table = pd.DataFrame(rows) if rows else None
    
    # Generate main text
    if has_sesoi and has_tost:
        main_text = f"Wald tests were conducted with both SESOI (practical significance, threshold={sesoi_threshold}) and TOST (equivalence testing, bounds={tost_bounds}) to test {hypothesis}. "
    elif has_sesoi:
        main_text = f"Wald tests with SESOI (practical significance threshold={sesoi_threshold}) were conducted to test {hypothesis}. "
    elif has_tost:
        main_text = f"Wald tests with TOST (equivalence bounds={tost_bounds}) were conducted to test {hypothesis}. "
    else:
        main_text = f"A {test_name} was conducted to test {hypothesis}. "
    
    # Add statistical significance for traditional Wald
    if not has_sesoi and not has_tost:
        if 'chi2_stat' in locals() and chi2_stat is not None and p_value is not None:
            main_text += f"The overall test statistic was χ²({df}) = {chi2_stat:.2f}, p = {p_value:.4f}. "
            
            if p_value < 0.001:
                main_text += "This provides very strong evidence against the null hypothesis. "
            elif p_value < 0.01:
                main_text += "This provides strong evidence against the null hypothesis. "
            elif p_value < 0.05:
                main_text += "This provides moderate evidence against the null hypothesis. "
            else:
                main_text += "There is insufficient evidence to reject the null hypothesis. "
    
    # Add individual coefficient results
    if has_sesoi or has_tost:
        # Report SESOI/TOST results
        exceeds_sesoi = []
        equivalent_to_zero = []
        
        for var_name, var_result in wald_tests.items():
            if not isinstance(var_result, dict) or var_name == 'overall':
                continue
            
            clean_name = var_name.replace('_', ' ').title()
            
            # Check SESOI
            if 'summary' in var_result and var_result['summary'].get('exceeds_sesoi'):
                exceeds_sesoi.append(f"{clean_name} (β={var_result.get('coefficient', 0):.3f})")
            
            # Check TOST
            if 'summary' in var_result and var_result['summary'].get('is_equivalent'):
                equivalent_to_zero.append(clean_name)
        
        if exceeds_sesoi:
            main_text += f"Predictors exceeding SESOI threshold: {', '.join(exceeds_sesoi)}. "
        if equivalent_to_zero:
            main_text += f"Predictors equivalent to zero: {', '.join(equivalent_to_zero)}. "
            
    else:
        # Traditional reporting
        if variable_names and coefficients:
            sig_vars = []
            for var in variable_names:
                if var in coefficients:
                    var_p = test_results.get(f'{var}_p_value')
                    if var_p and var_p < 0.05:
                        sig_vars.append(f"{var} (β={coefficients[var]:.3f}, p={var_p:.3f})")
            
            if sig_vars:
                main_text += f"Significant predictors include: {', '.join(sig_vars)}."
    
    # Methods text
    methods_text = (
        "Hypothesis testing was performed using the Wald test, which evaluates whether "
        "regression coefficients are significantly different from zero. The test statistic "
        "follows a chi-square distribution under the null hypothesis. For each coefficient, "
        "the Wald statistic is calculated as (β/SE)², where β is the coefficient estimate "
        "and SE is its standard error. P-values < 0.05 were considered statistically significant."
    )
    
    # Supplementary information
    supplementary_info = {
        'test_type': 'Wald Chi-Square Test',
        'null_hypothesis': 'Coefficients equal to zero',
        'alternative': 'Coefficients differ from zero',
        'assumptions': [
            'Large sample size for asymptotic validity',
            'Correct specification of variance-covariance matrix',
            'Model assumptions (e.g., linearity, independence) are met'
        ],
        'interpretation': {
            'p < 0.001': 'Very strong evidence',
            'p < 0.01': 'Strong evidence',
            'p < 0.05': 'Moderate evidence',
            'p ≥ 0.05': 'Insufficient evidence'
        }
    }
    
    # LaTeX table generation
    from .latex_tables import generate_wald_latex_table
    latex_table = None
    if results_table is not None:
        latex_table = generate_wald_latex_table(results_table, test_name)
    
    return {
        'main_text': main_text,
        'methods_text': methods_text,
        'results_table': results_table,
        'supplementary_info': supplementary_info,
        'latex_table': latex_table
    }


def generate_contrast_publication_report(
    contrast_results: Dict[str, Any],
    emmeans_results: Optional[Dict[str, Any]] = None,
    hypothesis_name: str = "H2 (Temperate Mismatch)",
    response_type: str = "fungi"
) -> Dict[str, Any]:
    """Generate publication-ready report for contrast analysis.
    
    Extracted from contrast_analysis.py lines 289-437.
    Following ecology journal standards for reporting emmeans contrasts.
    
    Args:
        contrast_results: Results from contrast analysis
        emmeans_results: Optional emmeans results for context
        hypothesis_name: Name of hypothesis being tested
        
    Returns:
        Dictionary with formatted text, tables, and visualizations
    """
    # Extract statistics
    if 'contrast_estimate' in contrast_results:
        # Single contrast result
        estimate = contrast_results['contrast_estimate']
        se = contrast_results['standard_error']
        z_stat = contrast_results['z_statistic']
        p_value = contrast_results['p_value']
        contrast_vec = contrast_results['contrast_vector']
        h2_supported = contrast_results.get('h2_supported', False)
    else:
        # Multiple contrasts - use first or specified one
        first_key = list(contrast_results.keys())[0]
        first_result = contrast_results[first_key]
        estimate = first_result['estimate']
        se = first_result['standard_error']
        z_stat = first_result['z_statistic']
        p_value = first_result['p_value']
        contrast_vec = first_result['contrast_vector']
        h2_supported = False
    
    # Format p-value
    if p_value < 0.001:
        p_str = "p < 0.001"
        p_formatted = "< 0.001"
    elif p_value < 0.01:
        p_str = f"p = {p_value:.3f}"
        p_formatted = f"{p_value:.3f}"
    else:
        p_str = f"p = {p_value:.2f}"
        p_formatted = f"{p_value:.2f}"
    
    # Generate main results text
    main_text = (
        f"Contrast analysis was performed to test {hypothesis_name}. "
        f"The contrast estimate was {estimate:.2f} (SE = {se:.2f}, z = {z_stat:.2f}, {p_str}). "
    )
    
    if hypothesis_name == "H2 (Temperate Mismatch)":
        if h2_supported:
            main_text += (
                f"This result supports H2, indicating that high-plant temperate regions "
                f"have {abs(estimate):.2f} fewer {response_type} species than high-plant non-temperate "
                f"regions (95% CI [{estimate - 1.96*se:.2f}, {estimate + 1.96*se:.2f}]). "
                f"This confirms the biodiversity decoupling pattern in temperate zones."
            )
        elif p_value < 0.05 and estimate > 0:
            main_text += (
                f"This result contradicts H2, showing that high-plant temperate regions "
                f"have {estimate:.2f} more {response_type} species than expected."
            )
        else:
            main_text += (
                f"The contrast was not statistically significant, providing no support "
                f"for the temperate mismatch hypothesis."
            )
    else:
        if p_value < 0.05:
            main_text += f"The contrast is statistically significant."
        else:
            main_text += f"The contrast is not statistically significant."
    
    # Generate methods text
    methods_text = (
        f"Estimated marginal means (emmeans) were computed for the 2×2 factorial design "
        f"(plant richness × climate type). Contrast analysis was performed using the "
        f"contrast vector {contrast_vec} to test specific hypotheses about group differences. "
        f"Standard errors were computed using the delta method, and significance was "
        f"assessed using z-tests at α = 0.05."
    )
    
    # Create results table
    results_table = pd.DataFrame({
        'Hypothesis': [hypothesis_name],
        'Contrast': [str(contrast_vec)],
        'Estimate': [f"{estimate:.3f}"],
        'SE': [f"{se:.3f}"],
        'z': [f"{z_stat:.2f}"],
        'p-value': [p_formatted],
        'Decision': ['Supported' if h2_supported else 'Not Supported' if 'H2' in hypothesis_name else 
                    'Significant' if p_value < 0.05 else 'Not Significant']
    })
    
    # Confidence intervals
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    
    # Supplementary statistics
    supplementary_stats = pd.DataFrame({
        'Statistic': ['Contrast Estimate', '95% CI Lower', '95% CI Upper', 
                     'Standard Error', 'z-statistic', 'p-value'],
        'Value': [f"{estimate:.3f}", f"{ci_lower:.3f}", f"{ci_upper:.3f}",
                 f"{se:.3f}", f"{z_stat:.2f}", p_formatted]
    })
    
    # Generate LaTeX table
    from .latex_tables import generate_contrast_latex_table
    latex_table = generate_contrast_latex_table(results_table)
    
    return {
        'main_text': main_text,
        'methods_text': methods_text,
        'results_table': results_table,
        'supplementary_stats': supplementary_stats,
        'latex_table': latex_table,
        'statistical_summary': {
            'estimate': estimate,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    }


def generate_lrt_publication_report(
    test_results: Dict[str, Any],
    test_name: str = "Likelihood Ratio Test",
    hypothesis: str = "model comparison",
    model_description: Optional[str] = None
) -> Dict[str, Any]:
    """Generate publication-ready report for LRT results.
    
    Extracted from lrt_test.py lines 126-220.
    
    Args:
        test_results: Results from LRT
        test_name: Name of the test for reporting
        hypothesis: Description of what's being tested
        model_description: Optional description of models being compared
        
    Returns:
        Dictionary with formatted text, tables, and summaries
    """
    # Extract key values
    lr_stat = test_results['lr_statistic']
    df = test_results['degrees_of_freedom']
    p_value = test_results['p_value']
    # Use percent_deviance_explained (they're the same thing)
    deviance_change_pct = test_results.get('percent_deviance_explained', test_results.get('deviance_change_pct', 0))
    
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
        "Statistical significance was assessed at α = 0.05."
    )
    
    # Generate LaTeX table
    from .latex_tables import generate_lrt_latex_table
    latex_table = generate_lrt_latex_table(
        lrt_result=test_results
    )
    
    return {
        'main_text': main_text,
        'methods_text': methods_text,
        'results_table': results_table,
        'comparison_details': comparison_details,
        'latex_table': latex_table
    }

def generate_simple_slopes_publication_report(
    simple_slopes_results: Dict[str, Any],
    hypothesis_name: str = "H2 (Temperate Mismatch)",
    response_type: str = "fungi"
) -> Dict[str, Any]:
    """Generate publication-ready report for simple slopes analysis.
    
    METHOD 2 ONLY: For H2 hypothesis testing.
    No longer requires analyzer instance - works with results dict only.
    
    Args:
        simple_slopes_results: Results from SimpleSlopesAnalysis.test_simple_slopes_h2()
        hypothesis_name: Name of hypothesis being tested (H2 for Method 2)
        
    Returns:
        Dictionary with formatted text, tables, and visualizations for Method 2
    """
    # Extract key results
    high_plant_result = simple_slopes_results.get('slope_temperate_in_high_plant', {})
    low_plant_result = simple_slopes_results.get('slope_temperate_in_low_plant', {})
    interaction_result = simple_slopes_results.get('interaction_effect')
    h2_supported = simple_slopes_results.get('h2_supported', False)
    
    # Format p-values helper
    def format_p(p):
        if p < 0.001:
            return "< 0.001", "p < 0.001"
        elif p < 0.01:
            return f"{p:.3f}", f"p = {p:.3f}"
        else:
            return f"{p:.2f}", f"p = {p:.2f}"
    
    hp_p_formatted, hp_p_str = format_p(high_plant_result.get('p_value', 1))
    lp_p_formatted, lp_p_str = format_p(low_plant_result.get('p_value', 1))
    
    # Generate main results text
    main_text = (
        f"Simple slopes analysis was performed to test {hypothesis_name}. "
        f"The effect of temperate climate in high-plant regions was "
        f"{high_plant_result.get('estimate', 0):.2f} (SE = {high_plant_result.get('standard_error', 0):.2f}, "
        f"z = {high_plant_result.get('z_statistic', 0):.2f}, {hp_p_str}). "
    )
    
    if h2_supported:
        main_text += (
            f"This supports H2, showing that high-plant temperate regions have "
            f"{abs(high_plant_result.get('estimate', 0)):.2f} fewer {response_type} species than "
            f"high-plant non-temperate regions. "
        )
    else:
        main_text += (
            f"The temperate effect in high-plant regions was "
            f"{'not significant' if high_plant_result.get('p_value', 1) >= 0.05 else 'significant but positive'}, "
            f"providing no support for H2. "
        )
    
    # Add interaction information if available
    if interaction_result:
        int_p_formatted, int_p_str = format_p(interaction_result.get('p_value', 1))
        main_text += (
            f"The interaction between plant level and climate was "
            f"{'significant' if interaction_result.get('significant', False) else 'not significant'} "
            f"(difference = {interaction_result.get('estimate', 0):.2f}, {int_p_str})."
        )
    
    # Generate methods text
    methods_text = (
        f"Simple slopes analysis was conducted to examine the temperate effect at different "
        f"levels of plant richness. This approach tests whether the effect of one factor "
        f"(temperate climate) varies across levels of another factor (plant richness). "
        f"Simple slopes were computed using estimated marginal means from the 2×2 factorial design. "
        f"Standard errors were estimated using the delta method, and significance was assessed "
        f"using z-tests at α = 0.05. Johnson-Neyman intervals were not computed as both "
        f"factors are categorical."
    )
    
    # Create results table
    results_table = pd.DataFrame([
        {
            'Condition': 'Low Plant Regions',
            'Temperate Effect': f"{low_plant_result.get('estimate', 0):.3f}",
            'SE': f"{low_plant_result.get('standard_error', 0):.3f}",
            'z': f"{low_plant_result.get('z_statistic', 0):.2f}",
            'p-value': lp_p_formatted,
            'Significant': 'Yes' if low_plant_result.get('significant', False) else 'No'
        },
        {
            'Condition': 'High Plant Regions',
            'Temperate Effect': f"{high_plant_result.get('estimate', 0):.3f}",
            'SE': f"{high_plant_result.get('standard_error', 0):.3f}",
            'z': f"{high_plant_result.get('z_statistic', 0):.2f}",
            'p-value': hp_p_formatted,
            'Significant': 'Yes' if high_plant_result.get('significant', False) else 'No'
        }
    ])
    
    # Add interaction row if available
    if interaction_result:
        int_p_formatted, _ = format_p(interaction_result.get('p_value', 1))
        interaction_row = pd.DataFrame([{
            'Condition': 'Interaction (Difference)',
            'Temperate Effect': f"{interaction_result.get('estimate', 0):.3f}",
            'SE': f"{interaction_result.get('standard_error', 0):.3f}",
            'z': f"{interaction_result.get('z_statistic', 0):.2f}",
            'p-value': int_p_formatted,
            'Significant': 'Yes' if interaction_result.get('significant', False) else 'No'
        }])
        results_table = pd.concat([results_table, interaction_row], ignore_index=True)
    
    # Confidence intervals
    hp_ci_lower = high_plant_result.get('estimate', 0) - 1.96 * high_plant_result.get('standard_error', 0)
    hp_ci_upper = high_plant_result.get('estimate', 0) + 1.96 * high_plant_result.get('standard_error', 0)
    lp_ci_lower = low_plant_result.get('estimate', 0) - 1.96 * low_plant_result.get('standard_error', 0)
    lp_ci_upper = low_plant_result.get('estimate', 0) + 1.96 * low_plant_result.get('standard_error', 0)
    
    # Supplementary statistics
    supplementary_stats = pd.DataFrame({
        'Statistic': ['Slope (Low Plant)', '95% CI (Low Plant)', 
                     'Slope (High Plant)', '95% CI (High Plant)',
                     'H2 Test Result'],
        'Value': [
            f"{low_plant_result.get('estimate', 0):.3f}",
            f"[{lp_ci_lower:.3f}, {lp_ci_upper:.3f}]",
            f"{high_plant_result.get('estimate', 0):.3f}",
            f"[{hp_ci_lower:.3f}, {hp_ci_upper:.3f}]",
            'Supported' if h2_supported else 'Not Supported'
        ]
    })
    
    # Generate LaTeX table
    from .latex_tables import generate_simple_slopes_latex_table
    latex_table = generate_simple_slopes_latex_table(results_table) if hasattr(generate_simple_slopes_latex_table, '__call__') else None
    
    return {
        'main_text': main_text,
        'methods_text': methods_text,
        'results_table': results_table,
        'supplementary_stats': supplementary_stats,
        'latex_table': latex_table,
        'statistical_summary': {
            'high_plant_slope': high_plant_result.get('estimate', 0),
            'high_plant_se': high_plant_result.get('standard_error', 0),
            'high_plant_p': high_plant_result.get('p_value', 1),
            'low_plant_slope': low_plant_result.get('estimate', 0),
            'low_plant_se': low_plant_result.get('standard_error', 0),
            'low_plant_p': low_plant_result.get('p_value', 1),
            'h2_supported': h2_supported
        }
    }


def generate_score_test_publication_report(
    test_results: Dict[str, Any],
    test_name: str = "Score Test",
    hypothesis: str = "model specification",
    variable_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate publication-ready report for Score test results.
    
    The Score test (Lagrange Multiplier test) is asymptotically equivalent 
    to Wald and LRT tests but only requires fitting the null model.
    
    Args:
        test_results: Score test results dictionary (from ScoreTest or Method1)
        test_name: Name of the test for reporting
        hypothesis: Description of hypothesis being tested
        variable_names: Names of variables tested
        
    Returns:
        Dictionary with formatted text, tables, and LaTeX code
    """
    # Extract statistics - handle both naming conventions
    score_stat = test_results.get('score_statistic', test_results.get('statistic', 0))
    p_value = test_results.get('p_value', 1)
    df = test_results.get('degrees_of_freedom', test_results.get('df', 1))
    reject_h0 = test_results.get('reject_h0', p_value < 0.05)
    interpretation = test_results.get('interpretation', '')
    
    # Generate main text
    main_text = f"A {test_name} (Lagrange Multiplier test) was conducted to test {hypothesis}. "
    main_text += f"The test statistic was χ²({df}) = {score_stat:.2f}, p = {p_value:.4e}. "
    
    if p_value < 0.001:
        main_text += "This provides very strong evidence against the null hypothesis. "
    elif p_value < 0.01:
        main_text += "This provides strong evidence against the null hypothesis. "
    elif p_value < 0.05:
        main_text += "This provides moderate evidence against the null hypothesis. "
    else:
        main_text += "There is insufficient evidence to reject the null hypothesis. "
    
    if interpretation:
        main_text += interpretation
    
    # Methods text
    methods_text = (
        "The Score test (Lagrange Multiplier test) evaluates model specification by testing "
        "whether excluded variables would significantly improve model fit. The test only requires "
        "fitting the restricted (null) model, making it computationally efficient. The test statistic "
        "follows a chi-square distribution with degrees of freedom equal to the number of restrictions. "
        "The Score test is asymptotically equivalent to the Wald and Likelihood Ratio tests."
    )
    
    # Create results summary
    results_summary = {
        'Test': test_name,
        'χ² Statistic': f"{score_stat:.2f}",
        'Degrees of Freedom': df,
        'P-value': f"{p_value:.4e}" if p_value < 0.0001 else f"{p_value:.4f}",
        'Decision': 'Reject H₀' if reject_h0 else 'Fail to reject H₀'
    }
    
    # LaTeX code for table
    from .latex_tables import generate_score_test_latex_table
    latex_table = generate_score_test_latex_table(test_results)
    
    return {
        'main_text': main_text,
        'methods_text': methods_text,
        'latex_table': latex_table
    }
