"""
Practical significance testing utilities for large-scale hypothesis testing.

This module provides modern approaches to hypothesis testing that address 
the problem of "everything is significant" with large sample sizes.

Methods:
1. SESOI (Smallest Effect Size of Interest) - Redefines null hypothesis
2. TOST (Two One-Sided Tests) - Tests for practical equivalence
5. Effect Size Reporting - Standardized effect measures
6. Confidence Intervals on Effect Sizes - Precision assessment

Author: Linus Torvalds (channeled)
Note: Because p-values are garbage with 43 million observations
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, Union, List, Any
import logging

logger = logging.getLogger(__name__)


class PracticalSignificance:
    """
    Utility class for practical significance testing with large datasets.
    
    Addresses the fundamental problem: with millions of observations,
    everything becomes "statistically significant" even if meaningless.
    """
    
    @staticmethod
    def calculate_effect_sizes(
        null_deviance: float,
        full_deviance: float, 
        null_df: int,
        full_df: int,
        n_obs: int,
        null_aic: Optional[float] = None,
        full_aic: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive effect size measures for GLM models.
        
        Always applicable - reports magnitude independent of significance.
        
        Args:
            null_deviance: Deviance of null model
            full_deviance: Deviance of full model
            null_df: Degrees of freedom for null model
            full_df: Degrees of freedom for full model
            n_obs: Number of observations
            null_aic: AIC of null model (optional)
            full_aic: AIC of full model (optional)
            
        Returns:
            Dictionary of effect size measures
        """
        effect_sizes = {}
        
        # 1. Percent deviance explained (like R² for GLM)
        deviance_explained = (null_deviance - full_deviance) / null_deviance
        effect_sizes['percent_deviance_explained'] = deviance_explained * 100
        
        # 2. Mean deviance reduction
        mean_null_dev = null_deviance / null_df if null_df > 0 else np.inf
        mean_full_dev = full_deviance / full_df if full_df > 0 else np.inf
        effect_sizes['mean_deviance_null'] = mean_null_dev
        effect_sizes['mean_deviance_full'] = mean_full_dev
        effect_sizes['mean_deviance_reduction'] = mean_null_dev - mean_full_dev
        effect_sizes['percent_mean_deviance_reduction'] = (
            ((mean_null_dev - mean_full_dev) / mean_null_dev * 100) 
            if mean_null_dev > 0 else 0
        )
        
        # 3. Per-observation effect (LRT/n) 
        lrt_statistic = null_deviance - full_deviance
        effect_sizes['per_observation_lrt'] = lrt_statistic / n_obs
        
        # 4. McFadden's Pseudo R² (if log-likelihoods available)
        # Deviance = -2 * log-likelihood (up to constant)
        # So we can approximate: LL ≈ -deviance/2
        ll_null = -null_deviance / 2
        ll_full = -full_deviance / 2
        if ll_null != 0:
            mcfadden_r2 = 1 - (ll_full / ll_null)
            effect_sizes['mcfadden_pseudo_r2'] = mcfadden_r2
            
        # 5. Nagelkerke's Pseudo R² (bounded between 0 and 1)
        # Based on likelihood ratio
        lr = np.exp((ll_full - ll_null) * 2 / n_obs)
        cox_snell_r2 = 1 - lr
        max_cox_snell = 1 - np.exp(ll_null * 2 / n_obs)
        if max_cox_snell > 0:
            nagelkerke_r2 = cox_snell_r2 / max_cox_snell
            effect_sizes['nagelkerke_pseudo_r2'] = nagelkerke_r2
            
        # 6. AIC/BIC differences (if provided)
        if null_aic is not None and full_aic is not None:
            effect_sizes['aic_difference'] = null_aic - full_aic
            effect_sizes['aic_improvement'] = null_aic > full_aic
            
        # 7. Overdispersion assessment
        effect_sizes['overdispersion_null'] = mean_null_dev
        effect_sizes['overdispersion_full'] = mean_full_dev
        effect_sizes['overdispersion_improved'] = mean_full_dev < mean_null_dev
        
        # Interpretation
        effect_sizes['interpretation'] = PracticalSignificance._interpret_effect_sizes(
            deviance_explained, mean_full_dev
        )
        
        return effect_sizes
    
    @staticmethod
    def _interpret_effect_sizes(deviance_explained: float, mean_deviance: float) -> str:
        """Provide interpretation of effect sizes."""
        interpretation = []
        
        # Deviance explained interpretation (like R²)
        if deviance_explained > 0.5:
            interpretation.append("Very large effect (>50% deviance explained)")
        elif deviance_explained > 0.3:
            interpretation.append("Large effect (>30% deviance explained)")
        elif deviance_explained > 0.1:
            interpretation.append("Moderate effect (>10% deviance explained)")
        elif deviance_explained > 0.01:
            interpretation.append("Small effect (>1% deviance explained)")
        else:
            interpretation.append("Trivial effect (<1% deviance explained)")
            
        # Overdispersion check
        if mean_deviance > 2:
            interpretation.append("Model shows overdispersion (mean deviance > 2)")
        elif mean_deviance > 1.5:
            interpretation.append("Model shows mild overdispersion")
        elif mean_deviance < 1.2:
            interpretation.append("Model fits well (mean deviance ≈ 1)")
            
        return "; ".join(interpretation)
    
    @staticmethod
    def calculate_effect_size_ci(
        effect_estimate: float,
        standard_error: float,
        n_obs: int,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence intervals for effect sizes.
        
        Always applicable - provides precision assessment.
        
        Args:
            effect_estimate: Point estimate of effect
            standard_error: Standard error of estimate
            n_obs: Number of observations
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Dictionary with CI bounds and interpretation
        """
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Standard CI
        ci_lower = effect_estimate - z_critical * standard_error
        ci_upper = effect_estimate + z_critical * standard_error
        ci_width = ci_upper - ci_lower
        
        # Relative precision (CV of the estimate)
        if effect_estimate != 0:
            relative_precision = (standard_error / abs(effect_estimate)) * 100
        else:
            relative_precision = np.inf
            
        result = {
            'estimate': effect_estimate,
            'se': standard_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'confidence_level': confidence_level,
            'relative_precision_pct': relative_precision,
            'n_obs': n_obs
        }
        
        # Interpretation
        if relative_precision < 10:
            result['precision'] = 'Very precise (<10% relative SE)'
        elif relative_precision < 25:
            result['precision'] = 'Precise (<25% relative SE)'
        elif relative_precision < 50:
            result['precision'] = 'Moderate precision'
        else:
            result['precision'] = 'Low precision (>50% relative SE)'
            
        # Check if CI includes zero
        result['includes_zero'] = (ci_lower <= 0 <= ci_upper)
        result['all_positive'] = (ci_lower > 0)
        result['all_negative'] = (ci_upper < 0)
        
        return result
    
    @staticmethod
    def sesoi_test(
        effect_estimate: float,
        standard_error: float,
        sesoi: float,
        alternative: str = 'two-sided'
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Test against Smallest Effect Size of Interest (SESOI).
        
        Redefines null hypothesis from "effect = 0" to "effect < SESOI".
        
        Args:
            effect_estimate: Observed effect size
            standard_error: Standard error of effect
            sesoi: Smallest Effect Size of Interest (threshold)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Test results with p-value against SESOI
        """
        if alternative == 'two-sided':
            # Test if |effect| > SESOI
            if abs(effect_estimate) > sesoi:
                # Calculate how many SEs away from SESOI
                z_stat = (abs(effect_estimate) - sesoi) / standard_error
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                exceeds_sesoi = True
                interpretation = f"Effect ({effect_estimate:.3f}) exceeds SESOI ({sesoi:.3f})"
            else:
                z_stat = (sesoi - abs(effect_estimate)) / standard_error
                p_value = 2 * stats.norm.cdf(z_stat)
                exceeds_sesoi = False
                interpretation = f"Effect ({effect_estimate:.3f}) below SESOI ({sesoi:.3f})"
                
        elif alternative == 'greater':
            # Test if effect > SESOI
            z_stat = (effect_estimate - sesoi) / standard_error
            p_value = 1 - stats.norm.cdf(z_stat)
            exceeds_sesoi = effect_estimate > sesoi
            interpretation = (
                f"Effect ({effect_estimate:.3f}) {'exceeds' if exceeds_sesoi else 'below'} "
                f"SESOI ({sesoi:.3f})"
            )
            
        elif alternative == 'less':
            # Test if effect < -SESOI
            z_stat = (effect_estimate + sesoi) / standard_error
            p_value = stats.norm.cdf(z_stat)
            exceeds_sesoi = effect_estimate < -sesoi
            interpretation = (
                f"Effect ({effect_estimate:.3f}) {'exceeds' if exceeds_sesoi else 'above'} "
                f"-SESOI ({-sesoi:.3f})"
            )
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
            
        return {
            'effect_estimate': effect_estimate,
            'sesoi': sesoi,
            'z_statistic': z_stat,
            'p_value': p_value,
            'exceeds_sesoi': exceeds_sesoi,
            'alternative': alternative,
            'interpretation': interpretation,
            'practically_significant': exceeds_sesoi and p_value < 0.05
        }
    
    @staticmethod
    def tost_equivalence_test(
        effect_estimate: float,
        standard_error: float,
        lower_bound: float,
        upper_bound: float,
        alpha: float = 0.05
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Two One-Sided Tests (TOST) for equivalence testing.
        
        Tests if effect is practically equivalent to zero (within bounds).
        
        Args:
            effect_estimate: Observed effect
            standard_error: Standard error 
            lower_bound: Lower equivalence bound (negative)
            upper_bound: Upper equivalence bound (positive)
            alpha: Significance level
            
        Returns:
            TOST test results
        """
        # Test 1: Effect > lower_bound
        z1 = (effect_estimate - lower_bound) / standard_error
        p1 = 1 - stats.norm.cdf(z1)  # One-sided p-value
        
        # Test 2: Effect < upper_bound  
        z2 = (upper_bound - effect_estimate) / standard_error
        p2 = 1 - stats.norm.cdf(z2)  # One-sided p-value
        
        # TOST p-value is the maximum of the two
        p_tost = max(p1, p2)
        
        # Equivalence established if both tests reject
        is_equivalent = p_tost < alpha
        
        # Also check if effect is significantly different from zero (standard test)
        z_diff = effect_estimate / standard_error
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
        is_different = p_diff < alpha
        
        # Four possible outcomes
        if is_equivalent and is_different:
            conclusion = "Statistically different from zero but practically equivalent"
            interpretation = "Effect is real but too small to matter"
        elif is_equivalent and not is_different:
            conclusion = "Practically and statistically equivalent to zero"
            interpretation = "No meaningful effect detected"
        elif not is_equivalent and is_different:
            conclusion = "Practically and statistically significant"
            interpretation = "Effect is both real and meaningful"
        else:
            conclusion = "Inconclusive"
            interpretation = "Insufficient evidence (increase sample size or tighten bounds)"
            
        return {
            'effect_estimate': effect_estimate,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'z_lower': z1,
            'z_upper': z2,
            'p_lower': p1,
            'p_upper': p2,
            'p_tost': p_tost,
            'p_difference': p_diff,
            'is_equivalent': is_equivalent,
            'is_different': is_different,
            'alpha': alpha,
            'conclusion': conclusion,
            'interpretation': interpretation
        }
    
    @staticmethod
    def determine_testing_approach(
        n_obs: int,
        sesoi: Optional[float] = None,
        equivalence_bounds: Optional[Tuple[float, float]] = None
    ) -> str:
        """
        Determine which testing approach to use based on sample size and parameters.
        
        Args:
            n_obs: Number of observations
            sesoi: SESOI threshold if specified
            equivalence_bounds: Equivalence bounds if specified
            
        Returns:
            Recommended approach: 'sesoi', 'tost', or 'both'
        """
        # With very large samples, practical significance is crucial
        large_sample = n_obs > 100000
        
        if sesoi is not None and equivalence_bounds is not None:
            return 'both'  # Can do both tests
        elif sesoi is not None:
            return 'sesoi'  # Test against minimum meaningful effect
        elif equivalence_bounds is not None:
            return 'tost'  # Test for practical equivalence
        elif large_sample:
            logger.warning(
                f"Large sample (n={n_obs:,}) but no SESOI or equivalence bounds specified. "
                "Consider defining practical significance thresholds."
            )
            return 'effect_sizes_only'
        else:
            return 'traditional'  # Small enough sample for traditional testing
    
    @staticmethod
    def interpret_lrt_with_practical_significance(
        lrt_result: Dict[str, Any],
        null_deviance: float,
        full_deviance: float,
        null_df: int,
        full_df: int,
        n_obs: int,
        sesoi_pct: Optional[float] = None,
        equivalence_bounds_pct: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Interpret traditional LRT results with practical significance context.
        
        This is a PURE INTERPRETATION layer - it doesn't change the LRT test itself,
        just adds practical context to understand what the results mean.
        
        Args:
            lrt_result: Result from standard LRT.test_nested_models()
            null_deviance: Null model deviance
            full_deviance: Full model deviance
            null_df: Null model degrees of freedom
            full_df: Full model degrees of freedom
            n_obs: Number of observations
            sesoi_pct: SESOI as percent deviance reduction (e.g., 10.0 for 10%)
            equivalence_bounds_pct: Tuple of (lower, upper) bounds as percent
            
        Returns:
            Enhanced LRT result with practical significance interpretation
        """
        # Start with original LRT result
        enhanced_result = lrt_result.copy()
        
        # Always add effect sizes (Method 5) 
        effect_sizes = PracticalSignificance.calculate_effect_sizes(
            null_deviance, full_deviance, null_df, full_df, n_obs
        )
        enhanced_result['effect_sizes'] = effect_sizes
        
        # Determine which interpretation approach to use
        approach = PracticalSignificance.determine_testing_approach(
            n_obs, sesoi_pct, equivalence_bounds_pct
        )
        enhanced_result['practical_approach'] = approach
        
        # Method 1: SESOI interpretation (if threshold provided)
        if sesoi_pct is not None:
            pct_reduction = effect_sizes['percent_deviance_explained']
            enhanced_result['sesoi_interpretation'] = {
                'threshold_pct': sesoi_pct,
                'observed_pct': pct_reduction,
                'exceeds_threshold': pct_reduction > sesoi_pct,
                'interpretation': (
                    f"Effect ({pct_reduction:.1f}%) "
                    f"{'exceeds' if pct_reduction > sesoi_pct else 'below'} "
                    f"practical threshold ({sesoi_pct}%)"
                ),
                'practically_significant': (
                    pct_reduction > sesoi_pct and lrt_result.get('p_value', 1) < 0.05
                )
            }
            
        # Method 2: TOST interpretation (if bounds provided)
        if equivalence_bounds_pct is not None:
            pct_reduction = effect_sizes['percent_deviance_explained']
            within_bounds = (
                equivalence_bounds_pct[0] <= pct_reduction <= equivalence_bounds_pct[1]
            )
            enhanced_result['equivalence_interpretation'] = {
                'bounds_pct': equivalence_bounds_pct,
                'observed_pct': pct_reduction,
                'within_bounds': within_bounds,
                'interpretation': (
                    f"Effect ({pct_reduction:.1f}%) is "
                    f"{'within' if within_bounds else 'outside'} "
                    f"equivalence bounds [{equivalence_bounds_pct[0]}, {equivalence_bounds_pct[1]}]%"
                ),
                'practically_equivalent': within_bounds
            }
            
        # Add large sample warning if appropriate
        if n_obs > 100000:
            enhanced_result['large_sample_context'] = {
                'n_observations': n_obs,
                'warning': (
                    f"With {n_obs:,} observations, even tiny effects will be statistically significant. "
                    "Focus on effect sizes and practical significance thresholds."
                ),
                'per_observation_effect': lrt_result.get('lr_statistic', 0) / n_obs
            }
            
        # Generate overall practical conclusion
        enhanced_result['practical_conclusion'] = PracticalSignificance._generate_practical_conclusion(
            enhanced_result, n_obs
        )
        
        return enhanced_result
    
    @staticmethod
    def interpret_score_with_practical_significance(
        score_result: Dict[str, Any],
        null_deviance: float,
        full_deviance: float,
        null_df: int,
        full_df: int,
        n_obs: int,
        sesoi_pct: Optional[float] = None,
        equivalence_bounds_pct: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Interpret traditional Score test results with practical significance context.
        
        Score test has the same large-N problem as LRT - it tests if adding
        parameters improves the model, which with 43M observations will always
        be "significant" even for tiny improvements.
        
        Args:
            score_result: Result from standard ScoreTest
            null_deviance: Null model deviance
            full_deviance: Full model deviance (estimated)
            null_df: Null model degrees of freedom
            full_df: Full model degrees of freedom
            n_obs: Number of observations
            sesoi_pct: SESOI as percent deviance reduction
            equivalence_bounds_pct: Equivalence bounds as percent
            
        Returns:
            Enhanced Score test result with practical significance
        """
        # Start with original Score test result
        enhanced_result = score_result.copy()
        
        # Always add effect sizes (Method 5)
        if null_deviance is not None and full_deviance is not None:
            effect_sizes = PracticalSignificance.calculate_effect_sizes(
                null_deviance, full_deviance, null_df, full_df, n_obs
            )
            enhanced_result['effect_sizes'] = effect_sizes
            
            # Method 1: SESOI interpretation
            if sesoi_pct is not None:
                pct_reduction = effect_sizes['percent_deviance_explained']
                enhanced_result['sesoi_interpretation'] = {
                    'threshold_pct': sesoi_pct,
                    'observed_pct': pct_reduction,
                    'exceeds_threshold': pct_reduction > sesoi_pct,
                    'interpretation': (
                        f"Score test indicates {pct_reduction:.1f}% improvement, "
                        f"{'exceeding' if pct_reduction > sesoi_pct else 'below'} "
                        f"{sesoi_pct}% threshold"
                    ),
                    'practically_significant': (
                        pct_reduction > sesoi_pct and score_result.get('p_value', 1) < 0.05
                    )
                }
                
        # Large sample warning
        if n_obs > 100000:
            enhanced_result['large_sample_context'] = {
                'n_observations': n_obs,
                'warning': (
                    f"Score test with {n_obs:,} observations will detect tiny improvements. "
                    "Focus on effect size not just p-value."
                ),
                'per_observation_effect': score_result.get('statistic', 0) / n_obs
            }
            
        # Generate conclusion
        enhanced_result['practical_conclusion'] = PracticalSignificance._generate_score_conclusion(
            enhanced_result, n_obs
        )
        
        return enhanced_result
    
    @staticmethod
    def _generate_score_conclusion(result: Dict, n_obs: int) -> str:
        """Generate practical conclusion for Score test."""
        conclusions = []
        
        # Statistical significance
        p_val = result.get('p_value', 1)
        if p_val < 0.05:
            conclusions.append(f"Score test significant (p={p_val:.2e})")
        else:
            conclusions.append(f"Score test not significant (p={p_val:.3f})")
            
        # Effect size if available
        if 'effect_sizes' in result:
            pct = result['effect_sizes'].get('percent_deviance_explained', 0)
            conclusions.append(f"{pct:.1f}% deviance improvement")
            
        # SESOI check
        if 'sesoi_interpretation' in result:
            if result['sesoi_interpretation']['exceeds_threshold']:
                conclusions.append("Practically meaningful")
            else:
                conclusions.append("Not practically meaningful")
                
        # Large sample note
        if n_obs > 100000:
            conclusions.append(f"Large sample (n={n_obs:,})")
            
        return "; ".join(conclusions)
    
    @staticmethod
    def interpret_wald_with_practical_significance(
        wald_result: Dict[str, Any],
        coefficient_estimate: float,
        standard_error: float,
        n_obs: int,
        sesoi: Optional[float] = None,
        equivalence_bounds: Optional[Tuple[float, float]] = None,
        parameter_name: str = "Coefficient"
    ) -> Dict[str, Any]:
        """
        Interpret traditional Wald test results with practical significance context.
        
        Pure interpretation layer for Wald tests.
        
        Args:
            wald_result: Result from standard WaldTest
            coefficient_estimate: Point estimate
            standard_error: Standard error
            n_obs: Number of observations
            sesoi: SESOI threshold for coefficient
            equivalence_bounds: Equivalence bounds for coefficient
            parameter_name: Name of parameter
            
        Returns:
            Enhanced Wald result with practical significance
        """
        # Start with original Wald result
        enhanced_result = wald_result.copy()
        
        # Always add confidence intervals (Method 6)
        ci_result = PracticalSignificance.calculate_effect_size_ci(
            coefficient_estimate, standard_error, n_obs
        )
        enhanced_result['confidence_intervals'] = ci_result
        
        # Method 1: SESOI test for coefficient
        if sesoi is not None:
            sesoi_result = PracticalSignificance.sesoi_test(
                coefficient_estimate, standard_error, sesoi
            )
            enhanced_result['sesoi_interpretation'] = sesoi_result
            
        # Method 2: TOST for coefficient  
        if equivalence_bounds is not None:
            tost_result = PracticalSignificance.tost_equivalence_test(
                coefficient_estimate, standard_error,
                equivalence_bounds[0], equivalence_bounds[1]
            )
            enhanced_result['equivalence_interpretation'] = tost_result
            
        # Large sample context
        if n_obs > 100000:
            enhanced_result['large_sample_context'] = {
                'n_observations': n_obs,
                'warning': f"Large sample (n={n_obs:,}) - focus on effect size not just p-value",
                'precision': ci_result['precision']
            }
            
        enhanced_result['parameter_name'] = parameter_name
        enhanced_result['practical_conclusion'] = PracticalSignificance._generate_wald_conclusion(
            enhanced_result, parameter_name
        )
        
        return enhanced_result
    
    @staticmethod
    def _generate_practical_conclusion(result: Dict, n_obs: int) -> str:
        """Generate practical conclusion for LRT results."""
        conclusions = []
        
        # Statistical significance
        p_val = result.get('p_value', 1)
        if p_val < 0.05:
            conclusions.append(f"Statistically significant (p={p_val:.2e})")
        else:
            conclusions.append(f"Not statistically significant (p={p_val:.3f})")
            
        # Effect size interpretation
        if 'effect_sizes' in result:
            pct = result['effect_sizes']['percent_deviance_explained']
            if pct > 30:
                conclusions.append(f"Large effect ({pct:.1f}%)")
            elif pct > 10:
                conclusions.append(f"Moderate effect ({pct:.1f}%)")
            else:
                conclusions.append(f"Small effect ({pct:.1f}%)")
                
        # SESOI interpretation
        if 'sesoi_interpretation' in result:
            if result['sesoi_interpretation']['exceeds_threshold']:
                conclusions.append("Exceeds practical threshold")
            else:
                conclusions.append("Below practical threshold")
                
        # TOST interpretation
        if 'equivalence_interpretation' in result:
            if result['equivalence_interpretation']['practically_equivalent']:
                conclusions.append("Practically equivalent to null")
            else:
                conclusions.append("Not equivalent to null")
                
        # Large sample note
        if n_obs > 100000:
            conclusions.append(f"Large sample (n={n_obs:,})")
            
        return "; ".join(conclusions)
    
    @staticmethod
    def interpret_dml_with_practical_significance(
        dml_result: Dict[str, Any],
        treatment_effect: float,
        standard_error: float,
        n_obs: int,
        sesoi: Optional[float] = None,
        equivalence_bounds: Optional[Tuple[float, float]] = None,
        effect_type: str = "ATE"
    ) -> Dict[str, Any]:
        """
        Interpret DML causal effect estimates with practical significance.
        
        DML tests if treatment effects ≠ 0, which with 43M observations
        will detect even economically meaningless effects as "significant".
        
        Args:
            dml_result: Result from DML classical test
            treatment_effect: Estimated causal effect (ATE/CATE)
            standard_error: Standard error of effect
            n_obs: Number of observations
            sesoi: Smallest Effect Size of Interest (e.g., 0.1 SD change)
            equivalence_bounds: Bounds for practical equivalence
            effect_type: Type of effect ("ATE", "CATE", etc.)
            
        Returns:
            Enhanced DML result with practical significance
        """
        # Start with original DML result
        enhanced_result = dml_result.copy()
        
        # Always add confidence intervals (Method 6)
        ci_result = PracticalSignificance.calculate_effect_size_ci(
            treatment_effect, standard_error, n_obs, confidence_level=0.95
        )
        enhanced_result['confidence_intervals'] = ci_result
        
        # Method 1: SESOI test for causal effect
        if sesoi is not None:
            sesoi_result = PracticalSignificance.sesoi_test(
                treatment_effect, standard_error, sesoi, alternative='two-sided'
            )
            enhanced_result['sesoi_interpretation'] = {
                'threshold': sesoi,
                'observed_effect': treatment_effect,
                'exceeds_threshold': sesoi_result['exceeds_sesoi'],
                'interpretation': (
                    f"{effect_type} = {treatment_effect:.3f} "
                    f"{'exceeds' if sesoi_result['exceeds_sesoi'] else 'below'} "
                    f"threshold ({sesoi:.3f})"
                ),
                'practically_significant': sesoi_result['practically_significant']
            }
            
        # Method 2: TOST for practical equivalence
        if equivalence_bounds is not None:
            tost_result = PracticalSignificance.tost_equivalence_test(
                treatment_effect, standard_error,
                equivalence_bounds[0], equivalence_bounds[1]
            )
            enhanced_result['equivalence_interpretation'] = tost_result
            
        # Large sample warning for causal inference
        if n_obs > 100000:
            enhanced_result['large_sample_context'] = {
                'n_observations': n_obs,
                'warning': (
                    f"With {n_obs:,} observations, DML will detect tiny causal effects. "
                    "Economic/practical significance matters more than statistical significance."
                ),
                'precision': ci_result['precision'],
                'ci_width': ci_result['ci_width']
            }
            
        # Generate conclusion
        enhanced_result['practical_conclusion'] = PracticalSignificance._generate_dml_conclusion(
            enhanced_result, effect_type
        )
        
        return enhanced_result
    
    @staticmethod
    def _generate_dml_conclusion(result: Dict, effect_type: str) -> str:
        """Generate practical conclusion for DML causal effects."""
        conclusions = [f"{effect_type}:"]
        
        # Point estimate and CI
        if 'confidence_intervals' in result:
            ci = result['confidence_intervals']
            conclusions.append(
                f"{ci['estimate']:.3f} "
                f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
            )
            
        # Statistical significance
        p_val = result.get('p_value', 1)
        if p_val < 0.05:
            conclusions.append(f"p={p_val:.2e}")
        else:
            conclusions.append(f"p={p_val:.3f} (NS)")
            
        # SESOI check
        if 'sesoi_interpretation' in result:
            if result['sesoi_interpretation']['practically_significant']:
                conclusions.append("Economically significant")
            else:
                conclusions.append("Not economically significant")
                
        # TOST check
        if 'equivalence_interpretation' in result:
            tost = result['equivalence_interpretation']
            conclusions.append(tost['conclusion'])
            
        return "; ".join(conclusions)
    
    @staticmethod
    def _generate_wald_conclusion(result: Dict, param_name: str) -> str:
        """Generate practical conclusion for Wald test results."""
        conclusions = [f"{param_name}:"]
        
        # Statistical significance
        p_val = result.get('p_value', 1)
        if p_val < 0.05:
            conclusions.append(f"p={p_val:.2e}")
        else:
            conclusions.append(f"p={p_val:.3f} (NS)")
            
        # Confidence interval
        if 'confidence_intervals' in result:
            ci = result['confidence_intervals']
            conclusions.append(f"95% CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
            conclusions.append(ci['precision'])
            
        # SESOI test
        if 'sesoi_interpretation' in result:
            sesoi = result['sesoi_interpretation']
            if sesoi['practically_significant']:
                conclusions.append("Practically significant")
            else:
                conclusions.append("Not practically significant")
                
        return "; ".join(conclusions)
    
    @staticmethod
    def comprehensive_report(
        effect_estimate: float,
        standard_error: float,
        null_deviance: float,
        full_deviance: float,
        null_df: int,
        full_df: int,
        n_obs: int,
        sesoi: Optional[float] = None,
        equivalence_bounds: Optional[Tuple[float, float]] = None,
        parameter_name: str = "Effect"
    ) -> Dict[str, any]:
        """
        Generate comprehensive practical significance report.
        
        Combines all methods (1, 2, 5, 6) into unified output.
        
        Args:
            effect_estimate: Point estimate of effect
            standard_error: Standard error
            null_deviance: Null model deviance
            full_deviance: Full model deviance
            null_df: Null model degrees of freedom
            full_df: Full model degrees of freedom
            n_obs: Number of observations
            sesoi: SESOI threshold (optional)
            equivalence_bounds: Tuple of (lower, upper) bounds (optional)
            parameter_name: Name of parameter for reporting
            
        Returns:
            Comprehensive report dictionary
        """
        report = {
            'parameter': parameter_name,
            'n_observations': n_obs
        }
        
        # Always include effect sizes (Method 5)
        report['effect_sizes'] = PracticalSignificance.calculate_effect_sizes(
            null_deviance, full_deviance, null_df, full_df, n_obs
        )
        
        # Always include confidence intervals (Method 6)
        report['confidence_intervals'] = PracticalSignificance.calculate_effect_size_ci(
            effect_estimate, standard_error, n_obs
        )
        
        # Determine testing approach
        approach = PracticalSignificance.determine_testing_approach(
            n_obs, sesoi, equivalence_bounds
        )
        report['testing_approach'] = approach
        
        # Add appropriate tests based on approach
        if approach in ['sesoi', 'both'] and sesoi is not None:
            report['sesoi_test'] = PracticalSignificance.sesoi_test(
                effect_estimate, standard_error, sesoi
            )
            
        if approach in ['tost', 'both'] and equivalence_bounds is not None:
            report['tost_test'] = PracticalSignificance.tost_equivalence_test(
                effect_estimate, standard_error, 
                equivalence_bounds[0], equivalence_bounds[1]
            )
            
        # Overall conclusion
        report['overall_conclusion'] = PracticalSignificance._synthesize_conclusion(report)
        
        return report
    
    @staticmethod
    def _synthesize_conclusion(report: Dict) -> str:
        """Synthesize overall conclusion from all tests."""
        conclusions = []
        
        # Effect size conclusion
        effect_pct = report['effect_sizes'].get('percent_deviance_explained', 0)
        if effect_pct > 30:
            conclusions.append(f"Large effect ({effect_pct:.1f}% deviance explained)")
        elif effect_pct > 10:
            conclusions.append(f"Moderate effect ({effect_pct:.1f}% deviance explained)")
        else:
            conclusions.append(f"Small effect ({effect_pct:.1f}% deviance explained)")
            
        # SESOI conclusion
        if 'sesoi_test' in report:
            if report['sesoi_test']['practically_significant']:
                conclusions.append("Exceeds minimum meaningful threshold")
            else:
                conclusions.append("Below practical significance threshold")
                
        # TOST conclusion
        if 'tost_test' in report:
            conclusions.append(report['tost_test']['interpretation'])
            
        # Precision conclusion
        if 'confidence_intervals' in report:
            conclusions.append(report['confidence_intervals']['precision'])
            
        return "; ".join(conclusions)


# Convenience functions for quick testing
def is_practically_significant(
    effect: float, 
    se: float, 
    sesoi: float,
    p_value: float = None
) -> bool:
    """Quick check if effect is practically significant."""
    result = PracticalSignificance.sesoi_test(effect, se, sesoi)
    if p_value is not None:
        # Both statistically and practically significant
        return result['exceeds_sesoi'] and p_value < 0.05
    return result['exceeds_sesoi']


def report_with_large_n_context(
    test_name: str,
    p_value: float,
    effect_size: float,
    n_obs: int,
    sesoi: float = None
) -> str:
    """
    Generate appropriate interpretation for large-N testing.
    
    Example:
        "LRT highly significant (p<0.0001) with 43M observations.
         Effect size: 30% deviance reduction (exceeds 10% SESOI).
         Conclusion: Both statistically and practically significant."
    """
    lines = [f"{test_name}: p = {p_value:.4e}"]
    
    if n_obs > 100000:
        lines.append(f"Note: With {n_obs:,} observations, even tiny effects are significant")
        
    lines.append(f"Effect size: {effect_size:.1f}%")
    
    if sesoi is not None:
        if effect_size > sesoi:
            lines.append(f"Exceeds SESOI threshold ({sesoi:.1f}%)")
            lines.append("Conclusion: Practically significant")
        else:
            lines.append(f"Below SESOI threshold ({sesoi:.1f}%)")
            lines.append("Conclusion: Statistically significant but practically negligible")
    else:
        if n_obs > 100000:
            lines.append("Warning: No SESOI defined for large-sample test")
            
    return "\n".join(lines)