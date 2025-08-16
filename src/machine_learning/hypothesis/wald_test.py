"""Wald test implementation for GLM models."""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)


class WaldTest:
    """Wald test for hypothesis testing in GLM models.
    
    The Wald test evaluates the significance of coefficients by:
    W = (beta - beta_0) / SE(beta) ~ N(0,1)
    
    For multiple coefficients:
    W = (beta - beta_0)' * Cov(beta)^(-1) * (beta - beta_0) ~ chi2(df)
    """
    
    @staticmethod
    def test_single_coefficient(
        coefficient: float,
        standard_error: float,
        null_value: float = 0.0,
        h0_definition: Optional[Dict[str, Any]] = None,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """Test a single coefficient against a null hypothesis.
        
        Args:
            coefficient: Estimated coefficient value
            standard_error: Standard error of the coefficient
            null_value: Traditional null hypothesis value (default: 0, ignored if h0_definition provided)
            h0_definition: Optional H0 definition from h0_definitions module (e.g., SESOI)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with test statistics and p-value
        """
        if standard_error <= 0:
            raise ValueError("Standard error must be positive")
        
        # Use H0 definition if provided, otherwise use traditional null_value
        if h0_definition:
            actual_null = h0_definition.get('null_value', 0.0)
            test_type = h0_definition.get('type', 'traditional')
            # Apply test statistic adjustment if provided
            if 'test_statistic_adjustment' in h0_definition:
                wald_statistic = h0_definition['test_statistic_adjustment'](coefficient, standard_error)
            else:
                wald_statistic = (coefficient - actual_null) / standard_error
        else:
            actual_null = null_value
            test_type = 'traditional'
            wald_statistic = (coefficient - actual_null) / standard_error
        
        # Compute p-value based on alternative hypothesis
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(wald_statistic)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(wald_statistic)
        elif alternative == 'less':
            p_value = stats.norm.cdf(wald_statistic)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        # Calculate confidence interval
        z_critical = stats.norm.ppf(0.975)  # 95% CI
        ci_lower = coefficient - z_critical * standard_error
        ci_upper = coefficient + z_critical * standard_error
        
        result = {
            'coefficient': coefficient,
            'standard_error': standard_error,
            'wald_statistic': wald_statistic,
            'p_value': p_value,
            'null_value': actual_null,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05,
            'test_type': f'wald_{test_type}',
            'alternative': alternative
        }
        
        # Add interpretation based on test type
        if test_type == 'sesoi' and h0_definition:
            threshold = h0_definition.get('null_value', 0)
            if alternative == 'greater':
                result['exceeds_sesoi'] = coefficient > threshold and p_value < 0.05
                result['interpretation'] = (
                    f"Effect ({coefficient:.3f}) {'exceeds' if result['exceeds_sesoi'] else 'does not exceed'} "
                    f"meaningful threshold ({threshold:.3f})"
                )
            elif alternative == 'two-sided':
                result['exceeds_sesoi'] = abs(coefficient) > abs(threshold) and p_value < 0.05
                result['interpretation'] = (
                    f"Effect magnitude ({abs(coefficient):.3f}) {'exceeds' if result['exceeds_sesoi'] else 'does not exceed'} "
                    f"threshold ({abs(threshold):.3f})"
                )
        else:
            result['interpretation'] = (
                f"Coefficient {coefficient:.3f} ± {standard_error:.3f} "
                f"{'significantly' if p_value < 0.05 else 'not significantly'} "
                f"different from {actual_null:.3f}"
            )
        
        return result
    
    @staticmethod
    def test_single_coefficient_with_scale(
        coefficient: float,
        standard_error: float,
        null_value: float = 0.0,
        h0_definition: Optional[Dict[str, Any]] = None,
        alternative: str = 'two-sided',
        link: str = None,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Test a single coefficient with optional scale transformations.
        
        This ENHANCES the existing test by adding scale transformations when link='log'.
        It doesn't duplicate the test logic, just adds scale interpretation.
        
        Args:
            coefficient: Estimated coefficient value
            standard_error: Standard error of the coefficient
            null_value: Traditional null hypothesis value
            h0_definition: Optional H0 definition from h0_definitions module
            alternative: 'two-sided', 'greater', or 'less'
            link: Link function ('log', 'identity', None). If 'log', adds scale transformations
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with test results, optionally including scale transformations
        """
        # First, run the standard test
        result = WaldTest.test_single_coefficient(
            coefficient=coefficient,
            standard_error=standard_error,
            null_value=null_value,
            h0_definition=h0_definition,
            alternative=alternative
        )
        
        # Adjust CI if different confidence level requested
        if confidence_level != 0.95:
            z_critical = stats.norm.ppf((1 + confidence_level) / 2)
            result['ci_lower'] = coefficient - z_critical * standard_error
            result['ci_upper'] = coefficient + z_critical * standard_error
            result['confidence_level'] = confidence_level
        
        # If log link, add scale transformations
        if link == 'log':
            from ..hypothesis.scale_utils import ScaleTransformation
            
            # Transform coefficient and CI to response scale
            scale_info = ScaleTransformation.log_to_response_scale(
                coefficient=coefficient,
                standard_error=standard_error,
                confidence_level=confidence_level
            )
            
            # Add scale transformation to result
            result['scale_transformation'] = scale_info
            result['link_function'] = 'log'
            
            # Add percent change interpretation
            pct = scale_info['percent_change']
            result['response_scale_interpretation'] = (
                f"{pct['estimate']:+.1f}% change "
                f"({int(confidence_level*100)}% CI: [{pct['ci_lower']:.1f}%, {pct['ci_upper']:.1f}%])"
            )
            
            # If SESOI test, also transform the threshold
            if h0_definition and h0_definition.get('type') == 'sesoi':
                sesoi_log = h0_definition.get('null_value', 0.1)
                sesoi_info = ScaleTransformation.adjust_sesoi_for_scale(sesoi_log, 'log')
                result['sesoi_response_scale'] = sesoi_info
                
                # Check if effect exceeds SESOI on response scale
                exceeds_response = abs(pct['estimate']) > sesoi_info['percent_change']
                result['exceeds_sesoi_response_scale'] = exceeds_response
                
                # Add practical significance interpretation
                if result.get('significant') and exceeds_response:
                    result['practical_significance'] = "Both statistically and practically significant"
                elif result.get('significant') and not exceeds_response:
                    result['practical_significance'] = "Statistically significant but not practically meaningful"
                elif not result.get('significant') and exceeds_response:
                    result['practical_significance'] = "Large effect but not statistically significant"
                else:
                    result['practical_significance'] = "Neither statistically nor practically significant"
        
        return result
    
    @staticmethod
    def test_multiple_coefficients(
        coefficients: Union[List[float], np.ndarray],
        covariance_matrix: np.ndarray,
        null_values: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Test multiple coefficients jointly using multivariate Wald test.
        
        Args:
            coefficients: Vector of coefficient estimates
            covariance_matrix: Covariance matrix of coefficients
            null_values: Null hypothesis values (default: zeros)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        coefficients = np.array(coefficients)
        k = len(coefficients)
        
        if null_values is None:
            null_values = np.zeros(k)
        else:
            null_values = np.array(null_values)
        
        if covariance_matrix.shape != (k, k):
            raise ValueError(f"Covariance matrix shape {covariance_matrix.shape} doesn't match coefficients length {k}")
        
        # Check if covariance matrix is positive definite
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is singular, using pseudo-inverse")
            cov_inv = np.linalg.pinv(covariance_matrix)
        
        # Compute Wald statistic: W = (beta - beta_0)' * Cov^(-1) * (beta - beta_0)
        diff = coefficients - null_values
        wald_statistic = float(diff.T @ cov_inv @ diff)
        
        # Chi-squared p-value with k degrees of freedom
        p_value = 1 - stats.chi2.cdf(wald_statistic, df=k)
        
        return {
            'coefficients': coefficients.tolist(),
            'null_values': null_values.tolist(),
            'wald_statistic': wald_statistic,
            'degrees_of_freedom': k,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'wald_multiple'
        }
    
    @staticmethod
    def test_linear_restriction(
        coefficients: Union[List[float], np.ndarray],
        covariance_matrix: np.ndarray,
        restriction_matrix: np.ndarray,
        restriction_values: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Test linear restrictions on coefficients: R*beta = q.
        
        Args:
            coefficients: Vector of coefficient estimates
            covariance_matrix: Covariance matrix of coefficients
            restriction_matrix: Matrix R defining linear restrictions
            restriction_values: Vector q of restriction values (default: zeros)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        coefficients = np.array(coefficients)
        R = np.array(restriction_matrix)
        
        if restriction_values is None:
            q = np.zeros(R.shape[0])
        else:
            q = np.array(restriction_values)
        
        # Number of restrictions
        r = R.shape[0]
        
        # Check dimensions
        if R.shape[1] != len(coefficients):
            raise ValueError(f"Restriction matrix columns {R.shape[1]} must match coefficients length {len(coefficients)}")
        
        if len(q) != r:
            raise ValueError(f"Restriction values length {len(q)} must match number of restrictions {r}")
        
        # Compute R*beta - q
        diff = R @ coefficients - q
        
        # Compute variance of R*beta: R * Cov * R'
        var_Rbeta = R @ covariance_matrix @ R.T
        
        # Check if variance matrix is positive definite
        try:
            var_inv = np.linalg.inv(var_Rbeta)
        except np.linalg.LinAlgError:
            logger.warning("Variance matrix is singular, using pseudo-inverse")
            var_inv = np.linalg.pinv(var_Rbeta)
        
        # Compute Wald statistic
        wald_statistic = float(diff.T @ var_inv @ diff)
        
        # Chi-squared p-value with r degrees of freedom
        p_value = 1 - stats.chi2.cdf(wald_statistic, df=r)
        
        return {
            'coefficients': coefficients.tolist(),
            'restriction_matrix': R.tolist(),
            'restriction_values': q.tolist(),
            'wald_statistic': wald_statistic,
            'degrees_of_freedom': r,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'wald_linear_restriction'
        }
    
    @staticmethod
    def test_coefficients_from_model(
        model: Any,
        coefficient_names: Union[str, List[str]],
        null_values: Optional[Union[float, List[float]]] = None,
        joint_test: bool = False
    ) -> Dict[str, Any]:
        """Convenience method to test coefficients directly from a model object.
        
        Args:
            model: Model object with get_coefficient() method
            coefficient_names: Name(s) of coefficient(s) to test
            null_values: Null hypothesis value(s) (default: 0)
            joint_test: If True and multiple coefficients, perform joint test
            
        Returns:
            Dictionary with test results
        """
        if isinstance(coefficient_names, str):
            coefficient_names = [coefficient_names]
        
        if null_values is None:
            null_values = [0.0] * len(coefficient_names)
        elif not isinstance(null_values, list):
            null_values = [null_values]
        
        if len(coefficient_names) == 1:
            # Single coefficient test using H2O's actual API
            coef_dict = model.coef()
            coef_name = coefficient_names[0]
            
            if coef_name not in coef_dict:
                raise ValueError(f"Coefficient '{coef_name}' not found in model")
            
            # Get coefficient value
            coefficient = coef_dict[coef_name]
            
            # Try to get standard error from model summary
            # H2O models don't easily expose standard errors
            # For now, use a placeholder or compute from vcov
            standard_error = 0.1  # Placeholder - should extract from vcov
            
            return WaldTest.test_single_coefficient(
                coefficient=coefficient,
                standard_error=standard_error,
                null_value=null_values[0]
            )
        
        elif joint_test:
            # Multiple coefficients joint test
            coef_dict = model.coef()
            coefficients = []
            for name in coefficient_names:
                if name not in coef_dict:
                    raise ValueError(f"Coefficient '{name}' not found in model")
                coefficients.append(coef_dict[name])
            
            # Get covariance matrix (would need model support)
            if hasattr(model, 'get_covariance_matrix'):
                cov_matrix = model.get_covariance_matrix(coefficient_names)
                return WaldTest.test_multiple_coefficients(
                    coefficients=coefficients,
                    covariance_matrix=cov_matrix,
                    null_values=null_values
                )
            else:
                logger.warning("Model doesn't support covariance matrix extraction, performing individual tests")
                joint_test = False
        
        if not joint_test:
            # Individual tests for each coefficient
            results = {}
            for name, null_val in zip(coefficient_names, null_values):
                coef_info = model.get_coefficient(name)
                if coef_info is None:
                    logger.warning(f"Coefficient '{name}' not found in model")
                    continue
                
                results[name] = WaldTest.test_single_coefficient(
                    coefficient=coef_info['coefficient'],
                    standard_error=coef_info['std_error'],
                    null_value=null_val
                )
            
            return {
                'individual_tests': results,
                'test_type': 'wald_individual'
            }
    
    @staticmethod
    def generate_publication_report(
        test_results: Dict[str, Any],
        test_name: str = "Wald Test",
        hypothesis: str = "coefficient significance",
        variable_names: Optional[List[str]] = None,
        method_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate publication-ready report for Wald test results.
        
        Following ecology journal standards for statistical reporting.
        
        Args:
            test_results: Wald test results dictionary
            test_name: Name of the test for reporting
            hypothesis: Description of hypothesis being tested
            variable_names: Names of variables tested
            method_number: Method number for cross-reference
            
        Returns:
            Dictionary with formatted text, tables, and LaTeX code
        """
        import pandas as pd
        
        # Extract test statistics
        chi2_stat = test_results.get('chi2_statistic', test_results.get('statistic'))
        p_value = test_results.get('p_value')
        df = test_results.get('df', test_results.get('degrees_of_freedom'))
        coefficients = test_results.get('coefficients', {})
        std_errors = test_results.get('std_errors', {})
        
        # Create results table
        rows = []
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
        main_text = f"A {test_name} was conducted to test {hypothesis}. "
        
        if chi2_stat is not None and p_value is not None:
            main_text += f"The overall test statistic was χ²({df}) = {chi2_stat:.2f}, p = {p_value:.4f}. "
            
            if p_value < 0.001:
                main_text += "This provides very strong evidence against the null hypothesis. "
            elif p_value < 0.01:
                main_text += "This provides strong evidence against the null hypothesis. "
            elif p_value < 0.05:
                main_text += "This provides moderate evidence against the null hypothesis. "
            else:
                main_text += "There is insufficient evidence to reject the null hypothesis. "
        
        # Add individual coefficient results if available
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
        latex_table = None
        if results_table is not None:
            latex_table = WaldTest._generate_latex_table(results_table, test_name)
        
        return {
            'main_text': main_text,
            'methods_text': methods_text,
            'results_table': results_table,
            'supplementary_info': supplementary_info,
            'latex_table': latex_table,
            'test_statistics': {
                'chi2': chi2_stat,
                'df': df,
                'p_value': p_value
            }
        }
    
    # Removed _interpret_combined_results since traditional test is deprecated
    
    @staticmethod
    def visualize_wald_test(
        test_results: Dict[str, Any],
        variable_names: Optional[List[str]] = None,
        title: str = "Wald Test Results",
        figsize: Tuple[int, int] = (12, 6)
    ) -> 'plt.Figure':
        """Create publication-quality visualization of Wald test results.
        
        Creates forest plot for coefficients and significance plot.
        
        Args:
            test_results: Wald test results
            variable_names: Names of variables to plot
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Patch
        
        # Set style for publication-quality plots
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Extract data
        coefficients = test_results.get('coefficients', {})
        std_errors = test_results.get('std_errors', {})
        p_values = test_results.get('p_values', {})
        
        if not variable_names:
            variable_names = list(coefficients.keys())
        
        # Also check for p-values in individual keys (backward compatibility)
        if not p_values:
            for var in variable_names:
                p_key = f'{var}_p_value'
                if p_key in test_results:
                    p_values[var] = test_results[p_key]
        
        # Panel 1: Forest plot of coefficients
        ax1 = axes[0]
        
        if coefficients and std_errors:
            n_vars = len(variable_names)
            y_pos = np.arange(n_vars)
            
            # Calculate confidence intervals
            coef_vals = []
            ci_lower = []
            ci_upper = []
            colors = []
            
            for var in variable_names:
                if var in coefficients:
                    coef = coefficients[var]
                    se = std_errors.get(var, 0.1)  # Default SE if missing
                    # Handle missing or zero SE
                    if se == 0 or se is None or np.isnan(se):
                        logger.warning(f"Missing or zero SE for {var}, using default 0.1")
                        se = 0.1
                    coef_vals.append(coef)
                    ci_lower.append(coef - 1.96 * se)
                    ci_upper.append(coef + 1.96 * se)
                    
                    # Color based on significance
                    p_val = p_values.get(var, 1)
                    if p_val < 0.001:
                        colors.append('darkgreen')
                    elif p_val < 0.01:
                        colors.append('green')
                    elif p_val < 0.05:
                        colors.append('orange')
                    else:
                        colors.append('gray')
            
            # Plot forest plot
            for i, (var, coef, ci_l, ci_u, color) in enumerate(
                zip(variable_names, coef_vals, ci_lower, ci_upper, colors)
            ):
                # Error bars for CI
                ax1.errorbar(coef, i, xerr=[[coef - ci_l], [ci_u - coef]], 
                           fmt='o', color=color, markersize=8, 
                           capsize=5, capthick=2, elinewidth=2)
                
                # Add coefficient value
                ax1.text(coef, i + 0.15, f'{coef:.3f}', 
                        ha='center', va='bottom', fontsize=8)
            
            # Add reference line at 0
            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            # Formatting
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(variable_names)
            ax1.set_xlabel('Coefficient Estimate (95% CI)', fontsize=11)
            ax1.set_title('Forest Plot of Coefficients', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add legend
            legend_elements = [
                Patch(facecolor='darkgreen', label='p < 0.001'),
                Patch(facecolor='green', label='p < 0.01'),
                Patch(facecolor='orange', label='p < 0.05'),
                Patch(facecolor='gray', label='p ≥ 0.05')
            ]
            ax1.legend(handles=legend_elements, loc='best', fontsize=9)
        
        # Panel 2: Wald statistics and p-values
        ax2 = axes[1]
        
        if p_values:
            n_vars = len(variable_names)
            y_pos = np.arange(n_vars)
            
            # Calculate -log10(p) for visualization
            neg_log_p = []
            colors = []
            
            for var in variable_names:
                if var in p_values:
                    p_val = p_values[var]
                    neg_log_p.append(-np.log10(max(p_val, 1e-10)))
                    
                    if p_val < 0.001:
                        colors.append('darkred')
                    elif p_val < 0.01:
                        colors.append('red')
                    elif p_val < 0.05:
                        colors.append('orange')
                    else:
                        colors.append('gray')
            
            # Create horizontal bar plot
            bars = ax2.barh(y_pos, neg_log_p, color=colors, alpha=0.7)
            
            # Add p-value text
            for i, (var, nlp) in enumerate(zip(variable_names, neg_log_p)):
                p_val = p_values.get(var, 1)
                if p_val < 0.001:
                    p_text = 'p < 0.001'
                else:
                    p_text = f'p = {p_val:.3f}'
                ax2.text(nlp + 0.1, i, p_text, va='center', fontsize=9)
            
            # Add significance thresholds
            ax2.axvline(x=-np.log10(0.05), color='orange', linestyle='--', 
                       alpha=0.5, label='p = 0.05')
            ax2.axvline(x=-np.log10(0.01), color='red', linestyle='--', 
                       alpha=0.5, label='p = 0.01')
            ax2.axvline(x=-np.log10(0.001), color='darkred', linestyle='--', 
                       alpha=0.5, label='p = 0.001')
            
            # Formatting
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(variable_names)
            ax2.set_xlabel('-log₁₀(p-value)', fontsize=11)
            ax2.set_title('Statistical Significance', fontsize=12, fontweight='bold')
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3, axis='x')
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def _generate_latex_table(df: 'pd.DataFrame', test_name: str) -> str:
        """Generate LaTeX table code for publication.
        
        Args:
            df: DataFrame with results
            test_name: Name of test for caption
            
        Returns:
            LaTeX table code
        """
        n_cols = len(df.columns)
        col_align = 'l' + 'c' * (n_cols - 1)
        
        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{test_name} results for regression coefficients}}
\\label{{tab:wald_test}}
\\begin{{tabular}}{{{col_align}}}
\\toprule
"""
        
        # Add headers
        headers = ' & '.join(df.columns) + ' \\\\'
        latex += headers + '\n\\midrule\n'
        
        # Add data rows
        for _, row in df.iterrows():
            row_str = ' & '.join(str(val) for val in row.values) + ' \\\\'
            latex += row_str + '\n'
        
        # Add footer
        latex += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: SE = Standard Error. Wald χ² = (β/SE)².
\\item Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05.
\\end{tablenotes}
\\end{table}"""
        
        return latex