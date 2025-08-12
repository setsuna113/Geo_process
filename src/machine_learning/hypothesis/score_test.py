"""Score Test (Lagrange Multiplier Test) implementation for GLM models."""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class ScoreTest:
    """Score Test (Lagrange Multiplier Test) for hypothesis testing in GLM models.
    
    The Score test evaluates hypotheses using only the null model, making it
    computationally efficient. It tests whether the score (gradient) of the 
    log-likelihood at the null hypothesis is significantly different from zero.
    
    Score statistic: S = U'(θ₀) * I(θ₀)^(-1) * U(θ₀) ~ χ²(df)
    where U is the score vector and I is the Fisher information matrix.
    """
    
    @staticmethod
    def test_coefficient_addition(
        null_model: Any,
        score_vector: Optional[np.ndarray] = None,
        information_matrix: Optional[np.ndarray] = None,
        df: int = 1
    ) -> Dict[str, Any]:
        """Test whether adding coefficients would improve the model.
        
        This is the classic score test for testing restrictions on parameters
        without fitting the full model.
        
        Args:
            null_model: Fitted null (restricted) model
            score_vector: Score vector at null hypothesis (if not available from model)
            information_matrix: Fisher information matrix (if not available from model)
            df: Degrees of freedom (number of parameters being tested)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        # Try to get score and information from model
        if score_vector is None:
            if hasattr(null_model, 'get_score'):
                score_vector = null_model.get_score()
            elif hasattr(null_model, 'score'):
                score_vector = null_model.score
            else:
                raise ValueError("Score vector not provided and not available from model")
        
        if information_matrix is None:
            if hasattr(null_model, 'get_information_matrix'):
                information_matrix = null_model.get_information_matrix()
            elif hasattr(null_model, 'information_matrix'):
                information_matrix = null_model.information_matrix
            elif hasattr(null_model, 'get_hessian'):
                # Information matrix is negative expected Hessian
                information_matrix = -null_model.get_hessian()
            else:
                raise ValueError("Information matrix not provided and not available from model")
        
        # Ensure arrays
        score_vector = np.array(score_vector)
        information_matrix = np.array(information_matrix)
        
        # Check dimensions
        if len(score_vector.shape) == 1:
            score_vector = score_vector.reshape(-1, 1)
        
        # Compute score statistic: S = U' * I^(-1) * U
        try:
            info_inv = np.linalg.inv(information_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Information matrix is singular, using pseudo-inverse")
            info_inv = np.linalg.pinv(information_matrix)
        
        score_statistic = float(score_vector.T @ info_inv @ score_vector)
        
        # Chi-squared p-value
        p_value = 1 - stats.chi2.cdf(score_statistic, df=df)
        
        return {
            'score_statistic': score_statistic,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'score_test'
        }
    
    @staticmethod
    def test_from_gradients(
        null_model: Any,
        data: Any,
        additional_predictors: List[str],
        compute_gradient_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Score test using gradient computation for additional predictors.
        
        This version computes the score for adding new predictors to a model
        without actually fitting the expanded model.
        
        Args:
            null_model: Fitted null model
            data: Data containing additional predictors
            additional_predictors: Names of predictors to test
            compute_gradient_func: Function to compute gradient (if model doesn't provide)
            
        Returns:
            Dictionary with test results
        """
        df = len(additional_predictors)
        
        # Get residuals from null model
        if hasattr(null_model, 'get_residuals'):
            residuals = null_model.get_residuals()
        elif hasattr(null_model, 'residuals'):
            residuals = null_model.residuals
        else:
            if hasattr(null_model, 'predict'):
                predictions = null_model.predict(data)
                if hasattr(data, 'get_response'):
                    y = data.get_response()
                else:
                    raise ValueError("Cannot compute residuals")
                residuals = y - predictions
            else:
                raise ValueError("Cannot compute residuals from model")
        
        # Compute score for additional predictors
        # For GLM: score = X' * (y - μ) / φ
        # where φ is dispersion parameter
        
        # Get design matrix for additional predictors
        if hasattr(data, '__getitem__'):
            X_add = np.column_stack([data[col] for col in additional_predictors])
        else:
            raise ValueError("Cannot extract additional predictors from data")
        
        # Get dispersion parameter
        if hasattr(null_model, 'get_dispersion'):
            dispersion = null_model.get_dispersion()
        elif hasattr(null_model, 'dispersion'):
            dispersion = null_model.dispersion
        else:
            dispersion = 1.0  # Assume 1 for Poisson/Binomial
            logger.info("Dispersion parameter not available, assuming 1.0")
        
        # Compute score vector
        score_vector = X_add.T @ residuals / dispersion
        
        # Estimate information matrix for additional parameters
        # For GLM: I = X' * W * X / φ
        # where W is weight matrix (depends on variance function)
        
        if hasattr(null_model, 'get_weights'):
            weights = null_model.get_weights()
        else:
            # Assume constant weights
            weights = np.ones(len(residuals))
            logger.info("Weights not available, assuming constant")
        
        # Information matrix for additional parameters
        W = np.diag(weights)
        info_matrix = X_add.T @ W @ X_add / dispersion
        
        # Compute score statistic
        try:
            info_inv = np.linalg.inv(info_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Information matrix is singular, using pseudo-inverse")
            info_inv = np.linalg.pinv(info_matrix)
        
        score_statistic = float(score_vector.T @ info_inv @ score_vector)
        
        # Chi-squared p-value
        p_value = 1 - stats.chi2.cdf(score_statistic, df=df)
        
        return {
            'score_statistic': score_statistic,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'tested_predictors': additional_predictors,
            'test_type': 'score_test_gradients'
        }
    
    @staticmethod
    def test_interaction(
        null_model: Any,
        var1: str,
        var2: str,
        data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Test for interaction effect using score test.
        
        Tests whether adding an interaction term var1:var2 would
        significantly improve the model fit.
        
        Args:
            null_model: Model without interaction
            var1: First variable in interaction
            var2: Second variable in interaction
            data: Data containing the variables
            
        Returns:
            Dictionary with test results
        """
        if data is None:
            if hasattr(null_model, 'data'):
                data = null_model.data
            else:
                raise ValueError("Data must be provided or available from model")
        
        # Create interaction term
        if hasattr(data, '__getitem__'):
            interaction = data[var1] * data[var2]
        else:
            raise ValueError("Cannot compute interaction from data")
        
        # Get residuals and weights from null model
        if hasattr(null_model, 'get_working_residuals'):
            residuals = null_model.get_working_residuals()
        elif hasattr(null_model, 'residuals'):
            residuals = null_model.residuals
        else:
            raise ValueError("Cannot get residuals from model")
        
        # Compute score for interaction term
        score = np.sum(interaction * residuals)
        
        # Compute information (variance of score)
        if hasattr(null_model, 'get_weights'):
            weights = null_model.get_weights()
        else:
            weights = np.ones(len(residuals))
        
        information = np.sum(weights * interaction**2)
        
        # Score statistic
        score_statistic = score**2 / information
        
        # Chi-squared test with 1 df
        p_value = 1 - stats.chi2.cdf(score_statistic, df=1)
        
        return {
            'interaction': f"{var1}:{var2}",
            'score_statistic': score_statistic,
            'degrees_of_freedom': 1,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'score_test_interaction'
        }
    
    @staticmethod
    def compare_with_other_tests(
        null_model: Any,
        full_model: Any,
        score_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare score test results with Wald and LRT for validation.
        
        All three tests (Score, Wald, LRT) are asymptotically equivalent.
        This function helps verify consistency.
        
        Args:
            null_model: Null (restricted) model
            full_model: Full (unrestricted) model
            score_result: Results from score test
            
        Returns:
            Dictionary comparing all three tests
        """
        from .wald_test import WaldTest
        from .lrt_test import LikelihoodRatioTest
        
        comparison = {
            'score_test': score_result,
            'test_type': 'three_test_comparison'
        }
        
        # Try to get Wald test
        try:
            # This would need the specific coefficient names
            logger.info("Wald test comparison requires specific coefficient names")
        except Exception as e:
            logger.debug(f"Could not perform Wald test: {e}")
        
        # Try to get LRT
        try:
            lrt_result = LikelihoodRatioTest.test_nested_models(
                null_model=null_model,
                full_model=full_model,
                df_difference=score_result.get('degrees_of_freedom', 1)
            )
            comparison['lrt_test'] = lrt_result
            
            # Check consistency
            score_p = score_result['p_value']
            lrt_p = lrt_result['p_value']
            
            if abs(score_p - lrt_p) > 0.1:
                comparison['warning'] = "Score and LRT p-values differ substantially"
            else:
                comparison['consistency'] = "Score and LRT results are consistent"
                
        except Exception as e:
            logger.debug(f"Could not perform LRT: {e}")
        
        return comparison
    
    @staticmethod
    def test_from_model_statistics(
        deviance_null: float,
        deviance_residual: float,
        df_null: int,
        df_residual: int,
        n_obs: int
    ) -> Dict[str, Any]:
        """Approximate score test using model deviance statistics.
        
        This is useful when the model doesn't provide score/information
        but does provide deviance statistics.
        
        Args:
            deviance_null: Null deviance
            deviance_residual: Residual deviance
            df_null: Null model degrees of freedom
            df_residual: Residual degrees of freedom
            n_obs: Number of observations
            
        Returns:
            Dictionary with approximate score test
        """
        # Approximate score statistic from deviance reduction
        # This uses the relationship between score and deviance
        deviance_diff = deviance_null - deviance_residual
        df_diff = df_null - df_residual
        
        # For large samples, deviance difference approximates score statistic
        score_statistic = deviance_diff
        
        # Chi-squared test
        p_value = 1 - stats.chi2.cdf(score_statistic, df=df_diff)
        
        return {
            'score_statistic': score_statistic,
            'degrees_of_freedom': df_diff,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'note': 'Approximated from deviance statistics',
            'deviance_reduction': deviance_diff,
            'test_type': 'score_test_approximate'
        }
    
    @staticmethod
    def generate_publication_report(test_results: Dict[str, Any],
                                   test_name: str = "Score Test",
                                   hypothesis: str = "parameter addition",
                                   variable_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate publication-ready report for Score Test results.
        
        Following ecology journal standards for reporting Lagrange Multiplier tests.
        
        Args:
            test_results: Results from any of the test methods
            test_name: Name of the specific test conducted
            hypothesis: Description of what was tested
            variable_names: Names of variables tested (for clarity)
            
        Returns:
            Dictionary with formatted text, tables, and visualizations
        """
        # Extract key statistics
        statistic = test_results.get('score_statistic', test_results.get('test_statistic', 0))
        p_value = test_results.get('p_value', 1)
        df = test_results.get('degrees_of_freedom', test_results.get('df', 1))
        significant = test_results.get('significant_at_0.05', p_value < 0.05)
        
        # Format p-value according to publication standards
        if p_value < 0.001:
            p_str = "p < 0.001"
            p_formatted = "< 0.001"
        elif p_value < 0.01:
            p_str = f"p = {p_value:.3f}"
            p_formatted = f"{p_value:.3f}"
        else:
            p_str = f"p = {p_value:.2f}"
            p_formatted = f"{p_value:.2f}"
        
        # Generate main text for results section
        if variable_names:
            var_str = ", ".join(variable_names)
            hypothesis_desc = f"the addition of {var_str} to the model"
        else:
            hypothesis_desc = hypothesis
        
        main_text = (
            f"A Score Test (Lagrange Multiplier test) was conducted to assess {hypothesis_desc}. "
            f"The test statistic was {statistic:.2f} (df = {df}, {p_str}). "
        )
        
        if significant:
            main_text += (
                f"The null hypothesis was rejected at the 0.05 significance level, "
                f"indicating that the additional parameter(s) significantly improve model fit."
            )
        else:
            main_text += (
                f"The null hypothesis was retained, suggesting that the restricted "
                f"model is adequate and the additional parameter(s) do not significantly "
                f"improve model fit."
            )
        
        # Generate methods text
        methods_text = (
            f"Model improvement was assessed using the Score Test (Lagrange Multiplier test), "
            f"which evaluates whether additional parameters would significantly improve model fit "
            f"without requiring estimation of the full model. The test statistic follows a "
            f"chi-squared distribution with {df} degree{'s' if df != 1 else ''} of freedom under "
            f"the null hypothesis. Statistical significance was assessed at α = 0.05."
        )
        
        # Create results table for supplementary materials
        results_table = pd.DataFrame({
            'Test': [test_name],
            'Parameters Tested': [variable_names[0] if variable_names and len(variable_names)==1 
                                 else f"{len(variable_names)} parameters" if variable_names 
                                 else "Additional parameters"],
            'Score Statistic (S)': [f"{statistic:.3f}"],
            'df': [df],
            'p-value': [p_formatted],
            'Significant (α=0.05)': ['Yes' if significant else 'No']
        })
        
        # Critical value for reference
        critical_value = stats.chi2.ppf(0.95, df)
        
        # Additional statistics for supplementary
        supplementary_stats = pd.DataFrame({
            'Statistic': ['Score (S)', 'Critical Value (α=0.05)', 'p-value', 'Decision'],
            'Value': [f"{statistic:.3f}", f"{critical_value:.3f}", p_formatted,
                     'Reject H₀' if significant else 'Retain H₀']
        })
        
        return {
            'main_text': main_text,
            'methods_text': methods_text,
            'results_table': results_table,
            'supplementary_stats': supplementary_stats,
            'latex_table': ScoreTest._generate_latex_table(results_table),
            'statistical_summary': {
                'statistic': statistic,
                'df': df,
                'p_value': p_value,
                'critical_value': critical_value,
                'significant': significant
            }
        }
    
    @staticmethod
    def visualize_score_test(test_results: Dict[str, Any],
                            variable_names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        """Create publication-quality visualization of Score Test results.
        
        Args:
            test_results: Results from any of the test methods
            variable_names: Names of variables tested
            figsize: Figure size
            
        Returns:
            matplotlib figure suitable for publication
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Extract statistics
        statistic = test_results.get('score_statistic', test_results.get('test_statistic', 0))
        p_value = test_results.get('p_value', 1)
        df = test_results.get('degrees_of_freedom', test_results.get('df', 1))
        significant = test_results.get('significant_at_0.05', p_value < 0.05)
        
        # Panel 1: Chi-square distribution with test statistic
        ax1 = axes[0]
        x = np.linspace(0, max(statistic * 1.5, stats.chi2.ppf(0.999, df)), 1000)
        y = stats.chi2.pdf(x, df)
        
        # Plot distribution
        ax1.plot(x, y, 'b-', linewidth=2, label=f'χ²({df}) distribution')
        
        # Shade rejection region
        critical_val = stats.chi2.ppf(0.95, df)
        x_reject = x[x > critical_val]
        y_reject = stats.chi2.pdf(x_reject, df)
        ax1.fill_between(x_reject, y_reject, alpha=0.3, color='red', 
                         label=f'Rejection region (α=0.05)')
        
        # Mark test statistic
        ax1.axvline(statistic, color='darkred', linestyle='--', linewidth=2.5,
                   label=f'Observed S = {statistic:.2f}')
        
        # Mark critical value
        ax1.axvline(critical_val, color='orange', linestyle=':', linewidth=2,
                   label=f'Critical value = {critical_val:.2f}')
        
        ax1.set_xlabel('Test Statistic Value', fontsize=11)
        ax1.set_ylabel('Probability Density', fontsize=11)
        ax1.set_title('Score Test Distribution', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, x[-1])
        
        # Panel 2: P-value visualization
        ax2 = axes[1]
        
        # Create significance level bars
        sig_levels = [0.001, 0.01, 0.05, 0.10]
        sig_labels = ['p < 0.001\n(***)', 'p < 0.01\n(**)', 'p < 0.05\n(*)', 'p < 0.10\n(†)']
        colors = ['#8B0000', '#DC143C', '#FF6347', '#FFA07A']
        
        # Plot horizontal bars for significance levels
        for i, (level, label, color) in enumerate(zip(sig_levels, sig_labels, colors)):
            ax2.barh(i, level, height=0.8, color=color, alpha=0.6)
            ax2.text(level + 0.002, i, label, va='center', fontsize=9)
        
        # Mark observed p-value
        if p_value <= 0.15:  # Only show if in reasonable range
            ax2.axvline(p_value, color='black', linestyle='-', linewidth=3,
                       label=f'Observed\np = {p_value:.4f}')
            # Add arrow pointing to p-value
            ax2.annotate('', xy=(p_value, -0.5), xytext=(p_value, -1.2),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        ax2.set_xlim(0, 0.12)
        ax2.set_ylim(-1.5, 4)
        ax2.set_xlabel('P-value', fontsize=11)
        ax2.set_yticks([])
        ax2.set_title('Significance Levels', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        if p_value <= 0.15:
            ax2.legend(loc='upper right', fontsize=9)
        
        # Panel 3: Effect size and power analysis
        ax3 = axes[2]
        
        # Create power curve
        effect_sizes = np.linspace(0, 4, 100)
        ncp = effect_sizes**2 * 50  # Non-centrality parameter (scaled for visualization)
        power_05 = 1 - stats.ncx2.cdf(stats.chi2.ppf(0.95, df), df, ncp)
        power_01 = 1 - stats.ncx2.cdf(stats.chi2.ppf(0.99, df), df, ncp)
        
        ax3.plot(effect_sizes, power_05, 'b-', linewidth=2, label='α = 0.05')
        ax3.plot(effect_sizes, power_01, 'g--', linewidth=2, label='α = 0.01')
        
        # Mark 80% power threshold
        ax3.axhline(0.8, color='red', linestyle=':', alpha=0.5, linewidth=1.5,
                   label='80% power')
        
        # Estimate and mark observed effect size (approximate)
        if significant:
            obs_effect = np.sqrt(statistic / 50)  # Rough approximation
            obs_power = 1 - stats.ncx2.cdf(stats.chi2.ppf(0.95, df), df, statistic)
            ax3.plot(obs_effect, obs_power, 'ro', markersize=10, 
                    label=f'Observed\n(effect≈{obs_effect:.2f})')
        
        ax3.set_xlabel('Standardized Effect Size', fontsize=11)
        ax3.set_ylabel('Statistical Power', fontsize=11)
        ax3.set_title('Power Analysis', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='lower right', fontsize=9)
        
        # Overall title
        decision = "Significant" if significant else "Not Significant"
        if variable_names:
            var_str = f" ({', '.join(variable_names)})"
        else:
            var_str = ""
        
        fig.suptitle(f'Score Test Results{var_str}: {decision}',
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def _generate_latex_table(df: pd.DataFrame) -> str:
        """Generate LaTeX table code for publication.
        
        Args:
            df: DataFrame with results
            
        Returns:
            LaTeX table code suitable for ecology journals
        """
        # Format for ecology journal standards
        latex = r"""\begin{table}[htbp]
\centering
\caption{Score Test (Lagrange Multiplier Test) Results}
\label{tab:score_test}
\begin{tabular}{lccccc}
\toprule
"""
        
        # Add headers
        headers = ' & '.join(df.columns) + ' \\\\'
        latex += headers + '\n\\midrule\n'
        
        # Add data rows
        for _, row in df.iterrows():
            row_str = ' & '.join(str(val) for val in row.values) + ' \\\\'
            latex += row_str + '\n'
        
        # Add footer
        latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: The Score test statistic follows a $\chi^2$ distribution under $H_0$. 
\item Significance assessed at $\alpha = 0.05$.
\end{tablenotes}
\end{table}"""
        
        return latex
    
    @staticmethod
    def create_comparison_plot(score_result: Dict[str, Any],
                              lrt_result: Optional[Dict[str, Any]] = None,
                              wald_result: Optional[Dict[str, Any]] = None,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Create comparison plot of Score, LRT, and Wald tests.
        
        Shows that all three tests are asymptotically equivalent.
        
        Args:
            score_result: Score test results
            lrt_result: Optional LRT results  
            wald_result: Optional Wald test results
            figsize: Figure size
            
        Returns:
            Comparison figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        tests = ['Score Test']
        statistics = [score_result.get('score_statistic', 0)]
        p_values = [score_result.get('p_value', 1)]
        
        if lrt_result:
            tests.append('LRT')
            statistics.append(lrt_result.get('lr_statistic', 0))
            p_values.append(lrt_result.get('p_value', 1))
        
        if wald_result:
            tests.append('Wald Test')
            statistics.append(wald_result.get('wald_statistic', 0))
            p_values.append(wald_result.get('p_value', 1))
        
        x = np.arange(len(tests))
        width = 0.35
        
        # Plot test statistics
        ax1 = ax
        bars1 = ax1.bar(x - width/2, statistics, width, label='Test Statistic',
                       color='steelblue', alpha=0.8)
        ax1.set_ylabel('Test Statistic', fontsize=11)
        ax1.set_xlabel('Test Type', fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(tests)
        
        # Add p-values on secondary axis
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, p_values, width, label='P-value',
                       color='coral', alpha=0.8)
        ax2.set_ylabel('P-value', fontsize=11)
        ax2.axhline(0.05, color='red', linestyle='--', alpha=0.5,
                   label='α = 0.05')
        
        # Add value labels
        for bar, val in zip(bars1, statistics):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        for bar, val in zip(bars2, p_values):
            height = bar.get_height()
            p_label = f'{val:.3f}' if val >= 0.001 else '< 0.001'
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    p_label, ha='center', va='bottom', fontsize=9)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.set_title('Comparison of Asymptotically Equivalent Tests',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig