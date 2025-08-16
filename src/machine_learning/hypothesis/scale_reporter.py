"""
Scale-aware reporting for GLM hypothesis tests.

Integrates scale transformations with hypothesis testing to provide
proper interpretation on both log and response scales.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from .scale_utils import ScaleTransformation
from .wald_test import WaldTest
from .h0_definitions import H0Definition
from .tost_wrapper import TOSTWrapper
import logging

logger = logging.getLogger(__name__)


class ScaleAwareReporter:
    """Reporter that properly handles scale transformations in GLM tests."""
    
    @staticmethod
    def test_with_scale_aware_sesoi(
        coefficient: float,
        standard_error: float,
        sesoi_log_scale: float = 0.1,
        link: str = 'log',
        variable_name: str = "Effect"
    ) -> Dict[str, Any]:
        """
        Test coefficient with SESOI, reporting on both scales.
        
        Args:
            coefficient: Coefficient estimate (on link scale)
            standard_error: Standard error (on link scale)
            sesoi_log_scale: SESOI threshold on log scale (default 0.1)
            link: Link function used
            variable_name: Name of variable for reporting
            
        Returns:
            Comprehensive test results with scale transformations
        """
        results = {
            'variable': variable_name,
            'link_function': link
        }
        
        # 1. Traditional test (H0: β = 0)
        traditional = WaldTest.test_single_coefficient(
            coefficient=coefficient,
            standard_error=standard_error,
            null_value=0.0,
            alternative='two-sided'
        )
        results['traditional_test'] = traditional
        
        # 2. SESOI test on log scale
        sesoi_definition = H0Definition.sesoi(sesoi_log_scale, 'greater')
        sesoi_test = WaldTest.test_single_coefficient(
            coefficient=abs(coefficient),  # Test absolute value
            standard_error=standard_error,
            h0_definition=sesoi_definition,
            alternative='greater'
        )
        results['sesoi_test'] = sesoi_test
        
        # 3. Scale transformations (if log link)
        if link == 'log':
            scale_transform = ScaleTransformation.log_to_response_scale(
                coefficient=coefficient,
                standard_error=standard_error,
                confidence_level=0.95
            )
            results['scale_transformation'] = scale_transform
            
            # SESOI interpretation on response scale
            sesoi_scales = ScaleTransformation.adjust_sesoi_for_scale(
                sesoi_log_scale=sesoi_log_scale,
                link=link
            )
            results['sesoi_scales'] = sesoi_scales
            
            # Check if effect exceeds SESOI on response scale
            pct_change = scale_transform['percent_change']['estimate']
            sesoi_pct = sesoi_scales['percent_change']
            exceeds_on_response = abs(pct_change) > sesoi_pct
            
            results['practical_significance'] = {
                'log_scale': {
                    'coefficient': coefficient,
                    'sesoi': sesoi_log_scale,
                    'exceeds': abs(coefficient) > sesoi_log_scale,
                    'interpretation': f"|β| = {abs(coefficient):.3f} {'>' if abs(coefficient) > sesoi_log_scale else '≤'} {sesoi_log_scale}"
                },
                'response_scale': {
                    'percent_change': pct_change,
                    'sesoi_percent': sesoi_pct,
                    'exceeds': exceeds_on_response,
                    'interpretation': f"{abs(pct_change):.1f}% change {'>' if exceeds_on_response else '≤'} {sesoi_pct:.1f}% threshold"
                }
            }
        else:
            # For identity link, no transformation needed
            results['practical_significance'] = {
                'original_scale': {
                    'coefficient': coefficient,
                    'sesoi': sesoi_log_scale,
                    'exceeds': abs(coefficient) > sesoi_log_scale,
                    'interpretation': f"|β| = {abs(coefficient):.3f} {'>' if abs(coefficient) > sesoi_log_scale else '≤'} {sesoi_log_scale}"
                }
            }
        
        # 4. Overall conclusion
        stat_sig = traditional['p_value'] < 0.05
        prac_sig = results['practical_significance'].get(
            'response_scale' if link == 'log' else 'original_scale'
        )['exceeds']
        
        if stat_sig and prac_sig:
            conclusion = "Both statistically and practically significant"
        elif stat_sig and not prac_sig:
            conclusion = "Statistically significant but not practically meaningful"
        elif not stat_sig and prac_sig:
            conclusion = "Large effect but not statistically significant (may need more data)"
        else:
            conclusion = "Neither statistically nor practically significant"
        
        results['conclusion'] = conclusion
        results['summary'] = ScaleAwareReporter._generate_summary(results, link)
        
        return results
    
    @staticmethod
    def test_with_scale_aware_tost(
        coefficient: float,
        standard_error: float,
        equivalence_bounds_log: tuple = (-0.1, 0.1),
        link: str = 'log',
        variable_name: str = "Effect",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform TOST with proper scale awareness.
        
        Args:
            coefficient: Coefficient estimate (on link scale)
            standard_error: Standard error (on link scale)
            equivalence_bounds_log: Bounds on log scale
            link: Link function
            variable_name: Variable name
            alpha: Significance level
            
        Returns:
            TOST results with scale transformations
        """
        results = {
            'variable': variable_name,
            'link_function': link
        }
        
        # 1. Perform TOST on link scale
        tost_result = TOSTWrapper.perform_tost(
            test_function=lambda **kwargs: WaldTest.test_single_coefficient(
                coefficient=kwargs.get('effect_estimate'),
                standard_error=kwargs.get('standard_error'),
                null_value=kwargs.get('h0_definition', {}).get('null_value', 0),
                alternative=kwargs.get('h0_definition', {}).get('alternative', 'two-sided')
            ),
            effect_estimate=coefficient,
            standard_error=standard_error,
            lower_bound=equivalence_bounds_log[0],
            upper_bound=equivalence_bounds_log[1],
            alpha=alpha
        )
        results['tost_test'] = tost_result
        
        # 2. Scale transformations
        if link == 'log':
            # Transform coefficient to response scale
            scale_transform = ScaleTransformation.log_to_response_scale(
                coefficient=coefficient,
                standard_error=standard_error,
                confidence_level=1 - alpha
            )
            results['scale_transformation'] = scale_transform
            
            # Transform equivalence bounds to response scale
            lower_pct = (np.exp(equivalence_bounds_log[0]) - 1) * 100
            upper_pct = (np.exp(equivalence_bounds_log[1]) - 1) * 100
            
            results['equivalence_bounds'] = {
                'log_scale': equivalence_bounds_log,
                'percent_change': (lower_pct, upper_pct),
                'interpretation': (
                    f"Log scale: [{equivalence_bounds_log[0]:.2f}, {equivalence_bounds_log[1]:.2f}] = "
                    f"Response scale: [{lower_pct:.1f}%, {upper_pct:.1f}%]"
                )
            }
            
            # Check if percent change is within bounds
            pct_change = scale_transform['percent_change']['estimate']
            within_bounds_response = lower_pct <= pct_change <= upper_pct
            
            results['equivalence_on_scales'] = {
                'log_scale': {
                    'within_bounds': tost_result['is_equivalent'],
                    'coefficient': coefficient,
                    'bounds': equivalence_bounds_log
                },
                'response_scale': {
                    'within_bounds': within_bounds_response,
                    'percent_change': pct_change,
                    'bounds_percent': (lower_pct, upper_pct)
                }
            }
        else:
            results['equivalence_on_scales'] = {
                'original_scale': {
                    'within_bounds': tost_result['is_equivalent'],
                    'coefficient': coefficient,
                    'bounds': equivalence_bounds_log
                }
            }
        
        results['summary'] = ScaleAwareReporter._generate_tost_summary(results, link)
        
        return results
    
    @staticmethod
    def report_multiple_coefficients(
        coefficients: Dict[str, float],
        standard_errors: Dict[str, float],
        sesoi_log_scale: float = 0.1,
        link: str = 'log',
        perform_tost: bool = False,
        equivalence_bounds: tuple = (-0.1, 0.1)
    ) -> Dict[str, Any]:
        """
        Report multiple coefficients with scale transformations.
        
        Args:
            coefficients: Dictionary of coefficient estimates
            standard_errors: Dictionary of standard errors
            sesoi_log_scale: SESOI threshold on log scale
            link: Link function
            perform_tost: Whether to perform TOST tests
            equivalence_bounds: Bounds for TOST
            
        Returns:
            Comprehensive report for all coefficients
        """
        report = {
            'link_function': link,
            'sesoi_threshold': sesoi_log_scale,
            'coefficients': {}
        }
        
        # Transform SESOI to response scale for reporting
        if link == 'log':
            sesoi_info = ScaleTransformation.adjust_sesoi_for_scale(
                sesoi_log_scale=sesoi_log_scale,
                link=link
            )
            report['sesoi_info'] = sesoi_info
        
        # Test each coefficient
        for var_name, coef in coefficients.items():
            se = standard_errors.get(var_name, 0.1)  # Default SE if missing
            
            # SESOI test with scale awareness
            var_result = ScaleAwareReporter.test_with_scale_aware_sesoi(
                coefficient=coef,
                standard_error=se,
                sesoi_log_scale=sesoi_log_scale,
                link=link,
                variable_name=var_name
            )
            
            # Add TOST if requested
            if perform_tost:
                tost_result = ScaleAwareReporter.test_with_scale_aware_tost(
                    coefficient=coef,
                    standard_error=se,
                    equivalence_bounds_log=equivalence_bounds,
                    link=link,
                    variable_name=var_name
                )
                var_result['tost'] = tost_result['tost_test']
            
            report['coefficients'][var_name] = var_result
        
        # Summary statistics
        n_significant = sum(
            1 for v in report['coefficients'].values()
            if v['traditional_test']['p_value'] < 0.05
        )
        n_practical = sum(
            1 for v in report['coefficients'].values()
            if v['practical_significance'].get(
                'response_scale' if link == 'log' else 'original_scale'
            )['exceeds']
        )
        
        report['summary'] = {
            'n_coefficients': len(coefficients),
            'n_statistically_significant': n_significant,
            'n_practically_significant': n_practical,
            'interpretation': ScaleAwareReporter._interpret_multiple(
                n_significant, n_practical, len(coefficients)
            )
        }
        
        return report
    
    @staticmethod
    def _generate_summary(results: Dict[str, Any], link: str) -> str:
        """Generate readable summary of test results."""
        lines = []
        
        # Variable name
        lines.append(f"Variable: {results['variable']}")
        
        # Statistical significance
        p_val = results['traditional_test']['p_value']
        lines.append(f"Statistical: p = {p_val:.4f} {'(significant)' if p_val < 0.05 else '(NS)'}")
        
        # Effect size and CI on relevant scale
        if link == 'log' and 'scale_transformation' in results:
            st = results['scale_transformation']
            pct = st['percent_change']
            lines.append(
                f"Effect: {pct['estimate']:+.1f}% "
                f"(95% CI: [{pct['ci_lower']:.1f}%, {pct['ci_upper']:.1f}%])"
            )
        else:
            tt = results['traditional_test']
            lines.append(
                f"Effect: {tt['coefficient']:.3f} "
                f"(95% CI: [{tt['ci_lower']:.3f}, {tt['ci_upper']:.3f}])"
            )
        
        # Practical significance
        if 'practical_significance' in results:
            ps = results['practical_significance']
            scale_key = 'response_scale' if link == 'log' else 'original_scale'
            if scale_key in ps:
                lines.append(f"Practical: {ps[scale_key]['interpretation']}")
        
        # Overall conclusion
        lines.append(f"Conclusion: {results['conclusion']}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _generate_tost_summary(results: Dict[str, Any], link: str) -> str:
        """Generate summary for TOST results."""
        lines = []
        
        tost = results['tost_test']
        lines.append(f"Variable: {results['variable']}")
        lines.append(f"TOST p-value: {tost['tost_p_value']:.4f}")
        lines.append(f"Conclusion: {tost['conclusion']}")
        
        if link == 'log' and 'equivalence_on_scales' in results:
            eq = results['equivalence_on_scales']
            lines.append(
                f"Response scale: {eq['response_scale']['percent_change']:.1f}% change "
                f"within [{eq['response_scale']['bounds_percent'][0]:.1f}%, "
                f"{eq['response_scale']['bounds_percent'][1]:.1f}%]"
            )
        
        lines.append(tost['interpretation'])
        
        return "\n".join(lines)
    
    @staticmethod
    def _interpret_multiple(n_sig: int, n_prac: int, n_total: int) -> str:
        """Interpret results for multiple coefficients."""
        if n_sig == n_prac and n_sig == n_total:
            return "All coefficients are both statistically and practically significant"
        elif n_sig == n_total and n_prac == 0:
            return "All statistically significant but none practically meaningful (large N artifact)"
        elif n_prac > n_sig:
            return f"{n_prac} practically meaningful effects, but only {n_sig} statistically significant"
        elif n_sig > 0 and n_prac > 0:
            return f"{n_sig}/{n_total} statistically significant, {n_prac}/{n_total} practically meaningful"
        else:
            return "No significant or meaningful effects detected"
    
    @staticmethod
    def create_publication_table(
        report: Dict[str, Any],
        include_response_scale: bool = True,
        format: str = 'markdown'
    ) -> str:
        """
        Create publication-ready table from report.
        
        Args:
            report: Report from report_multiple_coefficients
            include_response_scale: Whether to include response scale columns
            format: 'markdown' or 'latex'
            
        Returns:
            Formatted table string
        """
        if format == 'markdown':
            lines = []
            
            # Header
            if include_response_scale and report['link_function'] == 'log':
                lines.append("| Variable | Coefficient (log) | SE | p-value | % Change | 95% CI (%) | Practical |")
                lines.append("|----------|------------------|-----|---------|----------|------------|-----------|")
            else:
                lines.append("| Variable | Coefficient | SE | p-value | 95% CI | Practical |")
                lines.append("|----------|-------------|-----|---------|--------|-----------|")
            
            # Data rows
            for var, results in report['coefficients'].items():
                tt = results['traditional_test']
                ps = results['practical_significance']
                
                if include_response_scale and report['link_function'] == 'log':
                    st = results['scale_transformation']
                    pct = st['percent_change']
                    prac = "Yes" if ps['response_scale']['exceeds'] else "No"
                    
                    lines.append(
                        f"| {var} | {tt['coefficient']:.3f} | {tt['standard_error']:.3f} | "
                        f"{tt['p_value']:.4f} | {pct['estimate']:+.1f}% | "
                        f"[{pct['ci_lower']:.1f}, {pct['ci_upper']:.1f}] | {prac} |"
                    )
                else:
                    scale_key = 'response_scale' if report['link_function'] == 'log' else 'original_scale'
                    prac = "Yes" if ps.get(scale_key, ps.get('original_scale'))['exceeds'] else "No"
                    
                    lines.append(
                        f"| {var} | {tt['coefficient']:.3f} | {tt['standard_error']:.3f} | "
                        f"{tt['p_value']:.4f} | [{tt['ci_lower']:.3f}, {tt['ci_upper']:.3f}] | {prac} |"
                    )
            
            # Footer note
            sesoi_info = report.get('sesoi_info', {})
            if sesoi_info:
                lines.append("")
                lines.append(f"Note: Practical significance threshold = {sesoi_info.get('percent_change', 10):.1f}% change")
            
            return "\n".join(lines)
        
        elif format == 'latex':
            # LaTeX table format
            lines = ["\\begin{table}[htbp]"]
            lines.append("\\centering")
            lines.append("\\caption{GLM Coefficients with Scale Transformations}")
            
            if include_response_scale and report['link_function'] == 'log':
                lines.append("\\begin{tabular}{lcccccc}")
                lines.append("\\toprule")
                lines.append("Variable & $\\beta$ (log) & SE & p-value & \\% Change & 95\\% CI (\\%) & Practical \\\\")
            else:
                lines.append("\\begin{tabular}{lccccc}")
                lines.append("\\toprule")
                lines.append("Variable & $\\beta$ & SE & p-value & 95\\% CI & Practical \\\\")
            
            lines.append("\\midrule")
            
            # Data rows
            for var, results in report['coefficients'].items():
                tt = results['traditional_test']
                ps = results['practical_significance']
                
                # Format p-value
                if tt['p_value'] < 0.001:
                    p_str = "$<$0.001"
                else:
                    p_str = f"{tt['p_value']:.3f}"
                
                if include_response_scale and report['link_function'] == 'log':
                    st = results['scale_transformation']
                    pct = st['percent_change']
                    prac = "$\\checkmark$" if ps['response_scale']['exceeds'] else ""
                    
                    lines.append(
                        f"{var} & {tt['coefficient']:.3f} & {tt['standard_error']:.3f} & "
                        f"{p_str} & {pct['estimate']:+.1f} & "
                        f"[{pct['ci_lower']:.1f}, {pct['ci_upper']:.1f}] & {prac} \\\\"
                    )
                else:
                    scale_key = 'response_scale' if report['link_function'] == 'log' else 'original_scale'
                    prac = "$\\checkmark$" if ps.get(scale_key, ps.get('original_scale'))['exceeds'] else ""
                    
                    lines.append(
                        f"{var} & {tt['coefficient']:.3f} & {tt['standard_error']:.3f} & "
                        f"{p_str} & [{tt['ci_lower']:.3f}, {tt['ci_upper']:.3f}] & {prac} \\\\"
                    )
            
            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            
            # Footer note
            sesoi_info = report.get('sesoi_info', {})
            if sesoi_info:
                lines.append("\\begin{tablenotes}")
                lines.append("\\small")
                lines.append(f"\\item Note: Practical significance threshold = {sesoi_info.get('percent_change', 10):.1f}\\% change.")
                lines.append("\\end{tablenotes}")
            
            lines.append("\\end{table}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unknown format: {format}")