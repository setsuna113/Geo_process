"""
Classical Hypothesis Testing for DML (Option 1).

Standard significance testing using t-statistics and p-values
for Double Machine Learning causal parameters.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class DMLClassicalTest:
    """Classical hypothesis testing for DML causal effects.
    
    Tests:
    - H0: θ = 0 (no causal effect)
    - H1: θ > 0 (positive causal effect) or θ ≠ 0 (two-sided)
    
    Uses t-statistics and p-values for inference.
    """
    
    @staticmethod
    def test_causal_effect(
        theta: float,
        se_theta: float,
        alpha: float = 0.05,
        alternative: str = 'greater'
    ) -> Dict[str, Any]:
        """Test single causal parameter using classical hypothesis testing.
        
        Args:
            theta: Causal effect estimate
            se_theta: Standard error of theta
            alpha: Significance level (default 0.05)
            alternative: 'two-sided', 'greater', or 'less'
            
        Returns:
            Dictionary with test results:
            - t_statistic: Test statistic
            - p_value: P-value
            - reject_h0: Whether to reject null hypothesis
            - ci_lower: Lower bound of confidence interval
            - ci_upper: Upper bound of confidence interval
            - interpretation: Text interpretation
        """
        if se_theta <= 0:
            raise ValueError("Standard error must be positive")
        
        # Calculate t-statistic
        t_stat = theta / se_theta
        
        # Calculate p-value based on alternative hypothesis
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            z_critical = stats.norm.ppf(1 - alpha/2)
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(t_stat)
            z_critical = stats.norm.ppf(1 - alpha)
        elif alternative == 'less':
            p_value = stats.norm.cdf(t_stat)
            z_critical = -stats.norm.ppf(1 - alpha)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        
        # Confidence interval (always two-sided)
        ci_lower = theta - stats.norm.ppf(1 - alpha/2) * se_theta
        ci_upper = theta + stats.norm.ppf(1 - alpha/2) * se_theta
        
        # Decision
        if alternative == 'two-sided':
            reject_h0 = abs(t_stat) > z_critical
        elif alternative == 'greater':
            reject_h0 = t_stat > z_critical
        else:  # less
            reject_h0 = t_stat < z_critical
        
        # Generate interpretation
        interpretation = DMLClassicalTest._generate_interpretation(
            theta, se_theta, t_stat, p_value, reject_h0, alternative
        )
        
        result = {
            'theta': float(theta),
            'se': float(se_theta),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'reject_h0': bool(reject_h0),
            'alpha': float(alpha),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'alternative': alternative,
            'interpretation': interpretation
        }
        
        logger.info(f"Classical test: θ={theta:.4f}, SE={se_theta:.4f}, "
                   f"t={t_stat:.2f}, p={p_value:.4f}")
        
        return result
    
    @staticmethod
    def test_biodiversity_hypothesis(
        theta_plants: float,
        se_plants: float,
        theta_vertebrates: float,
        se_vertebrates: float,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Test biodiversity begets biodiversity hypothesis (H1).
        
        Tests whether both plant and vertebrate richness have positive
        causal effects on fungi richness.
        
        Args:
            theta_plants: Causal effect of plants on fungi
            se_plants: Standard error for plants effect
            theta_vertebrates: Causal effect of vertebrates on fungi
            se_vertebrates: Standard error for vertebrates effect
            alpha: Significance level
            
        Returns:
            Dictionary with comprehensive test results
        """
        # Test each effect
        plants_test = DMLClassicalTest.test_causal_effect(
            theta_plants, se_plants, alpha, 'greater'
        )
        vertebrates_test = DMLClassicalTest.test_causal_effect(
            theta_vertebrates, se_vertebrates, alpha, 'greater'
        )
        
        # Joint hypothesis: both effects positive
        both_positive = plants_test['reject_h0'] and vertebrates_test['reject_h0']
        either_positive = plants_test['reject_h0'] or vertebrates_test['reject_h0']
        
        # Combined interpretation
        if both_positive:
            h1_conclusion = "STRONG SUPPORT for H1: Both plant and vertebrate richness cause increased fungi richness"
        elif either_positive:
            if plants_test['reject_h0']:
                h1_conclusion = "PARTIAL SUPPORT for H1: Only plant richness shows significant positive effect"
            else:
                h1_conclusion = "PARTIAL SUPPORT for H1: Only vertebrate richness shows significant positive effect"
        else:
            h1_conclusion = "NO SUPPORT for H1: Neither biodiversity measure shows significant positive effect"
        
        result = {
            'plants_effect': plants_test,
            'vertebrates_effect': vertebrates_test,
            'h1_supported': both_positive,
            'h1_partial_support': either_positive,
            'h1_conclusion': h1_conclusion,
            'summary': {
                'theta_plants': float(theta_plants),
                'p_plants': plants_test['p_value'],
                'theta_vertebrates': float(theta_vertebrates),
                'p_vertebrates': vertebrates_test['p_value'],
                'both_significant': both_positive
            }
        }
        
        logger.info(f"Biodiversity hypothesis test: {h1_conclusion}")
        
        return result
    
    @staticmethod
    def _generate_interpretation(
        theta: float,
        se: float,
        t_stat: float,
        p_value: float,
        reject_h0: bool,
        alternative: str
    ) -> str:
        """Generate human-readable interpretation of test results.
        
        Args:
            theta: Effect estimate
            se: Standard error
            t_stat: T-statistic
            p_value: P-value
            reject_h0: Whether H0 is rejected
            alternative: Type of test
            
        Returns:
            Interpretation string
        """
        # Rule of thumb interpretation
        if abs(t_stat) > 3:
            strength = "very strong"
        elif abs(t_stat) > 2:
            strength = "strong"
        elif abs(t_stat) > 1.5:
            strength = "moderate"
        else:
            strength = "weak"
        
        if reject_h0:
            if alternative == 'greater' and theta > 0:
                interpretation = (
                    f"Significant positive causal effect detected (θ={theta:.3f}, "
                    f"t={t_stat:.2f}, p={p_value:.4f}). "
                    f"Each unit increase causes {theta:.3f} more in outcome. "
                    f"Evidence strength: {strength}."
                )
            elif alternative == 'less' and theta < 0:
                interpretation = (
                    f"Significant negative causal effect detected (θ={theta:.3f}, "
                    f"t={t_stat:.2f}, p={p_value:.4f}). "
                    f"Each unit increase causes {abs(theta):.3f} less in outcome. "
                    f"Evidence strength: {strength}."
                )
            else:
                interpretation = (
                    f"Significant causal effect detected (θ={theta:.3f}, "
                    f"t={t_stat:.2f}, p={p_value:.4f}). "
                    f"Evidence strength: {strength}."
                )
        else:
            interpretation = (
                f"No significant causal effect detected (θ={theta:.3f}, "
                f"t={t_stat:.2f}, p={p_value:.4f}). "
                f"Cannot reject null hypothesis at α={p_value:.2f} level."
            )
        
        return interpretation
    
    @staticmethod
    def test_joint_significance(
        thetas: list,
        ses: list,
        labels: Optional[list] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Test joint significance of multiple parameters.
        
        Uses Bonferroni correction for multiple testing.
        
        Args:
            thetas: List of causal effect estimates
            ses: List of standard errors
            labels: Optional labels for each parameter
            alpha: Overall significance level
            
        Returns:
            Dictionary with joint test results
        """
        n_tests = len(thetas)
        if len(ses) != n_tests:
            raise ValueError("thetas and ses must have same length")
        
        if labels is None:
            labels = [f"Parameter {i+1}" for i in range(n_tests)]
        
        # Bonferroni correction
        alpha_corrected = alpha / n_tests
        
        # Test each parameter
        individual_tests = []
        for i, (theta, se, label) in enumerate(zip(thetas, ses, labels)):
            test = DMLClassicalTest.test_causal_effect(
                theta, se, alpha_corrected, 'two-sided'
            )
            test['label'] = label
            individual_tests.append(test)
        
        # Count significant effects
        n_significant = sum(test['reject_h0'] for test in individual_tests)
        all_significant = n_significant == n_tests
        any_significant = n_significant > 0
        
        result = {
            'individual_tests': individual_tests,
            'n_parameters': n_tests,
            'n_significant': n_significant,
            'all_significant': all_significant,
            'any_significant': any_significant,
            'bonferroni_alpha': float(alpha_corrected),
            'interpretation': (
                f"{n_significant}/{n_tests} parameters significant "
                f"after Bonferroni correction (α={alpha_corrected:.4f})"
            )
        }
        
        return result
    
    @staticmethod
    def generate_publication_report(
        test_results: Dict[str, Any],
        test_name: str = "DML Causal Effect Test",
        hypothesis: str = "H1 (Biodiversity Begets Biodiversity)",
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate publication-ready report for DML classical test results.
        
        Following ecology journal standards for reporting causal inference.
        Focuses on H0 (no effect) vs H1 (biodiversity begets biodiversity).
        
        Args:
            test_results: Results from test methods (single or biodiversity test)
            test_name: Name of the specific test conducted
            hypothesis: Description of what was tested (H0 or H1)
            variable_names: Names of variables tested
            
        Returns:
            Dictionary with formatted text, tables, and visualizations
        """
        # Handle both single effect and biodiversity hypothesis results
        if 'plants_effect' in test_results:
            # Biodiversity hypothesis test result
            plants_result = test_results['plants_effect']
            vert_result = test_results['vertebrates_effect']
            h1_supported = test_results['h1_supported']
            h1_conclusion = test_results['h1_conclusion']
            
            # Format p-values
            def format_p(p):
                if p < 0.001:
                    return "< 0.001", "p < 0.001"
                elif p < 0.01:
                    return f"{p:.3f}", f"p = {p:.3f}"
                else:
                    return f"{p:.2f}", f"p = {p:.2f}"
            
            p_plants_fmt, p_plants_str = format_p(plants_result['p_value'])
            p_vert_fmt, p_vert_str = format_p(vert_result['p_value'])
            
            # Generate main text
            main_text = (
                f"Double Machine Learning was used to test the biodiversity begets biodiversity hypothesis (H1). "
                f"The causal effect of plant richness on fungi richness was "
                f"{plants_result['theta']:.3f} (SE = {plants_result['se']:.3f}, "
                f"t = {plants_result['t_statistic']:.2f}, {p_plants_str}). "
                f"The causal effect of vertebrate richness on fungi richness was "
                f"{vert_result['theta']:.3f} (SE = {vert_result['se']:.3f}, "
                f"t = {vert_result['t_statistic']:.2f}, {p_vert_str}). "
                f"{h1_conclusion}"
            )
            
            # Create results table
            results_table = pd.DataFrame([
                {
                    'Predictor': 'Plant Richness',
                    'Causal Effect (θ)': f"{plants_result['theta']:.3f}",
                    'SE': f"{plants_result['se']:.3f}",
                    't-statistic': f"{plants_result['t_statistic']:.2f}",
                    'p-value': p_plants_fmt,
                    '95% CI': f"[{plants_result['ci_lower']:.3f}, {plants_result['ci_upper']:.3f}]"
                },
                {
                    'Predictor': 'Vertebrate Richness',
                    'Causal Effect (θ)': f"{vert_result['theta']:.3f}",
                    'SE': f"{vert_result['se']:.3f}",
                    't-statistic': f"{vert_result['t_statistic']:.2f}",
                    'p-value': p_vert_fmt,
                    '95% CI': f"[{vert_result['ci_lower']:.3f}, {vert_result['ci_upper']:.3f}]"
                }
            ])
            
        else:
            # Single effect test result
            theta = test_results['theta']
            se = test_results['se']
            t_stat = test_results['t_statistic']
            p_value = test_results['p_value']
            ci_lower = test_results['ci_lower']
            ci_upper = test_results['ci_upper']
            
            # Format p-value
            if p_value < 0.001:
                p_str = "p < 0.001"
                p_fmt = "< 0.001"
            elif p_value < 0.01:
                p_str = f"p = {p_value:.3f}"
                p_fmt = f"{p_value:.3f}"
            else:
                p_str = f"p = {p_value:.2f}"
                p_fmt = f"{p_value:.2f}"
            
            var_str = variable_names[0] if variable_names else "the treatment"
            
            main_text = (
                f"Double Machine Learning was used to estimate the causal effect of {var_str}. "
                f"The estimated causal effect was θ = {theta:.3f} (SE = {se:.3f}, "
                f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}], t = {t_stat:.2f}, {p_str}). "
            )
            
            if test_results['reject_h0']:
                main_text += f"The null hypothesis of no causal effect was rejected at α = {test_results['alpha']:.2f}."
            else:
                main_text += f"The null hypothesis of no causal effect was not rejected at α = {test_results['alpha']:.2f}."
            
            # Create results table
            results_table = pd.DataFrame([{
                'Predictor': var_str,
                'Causal Effect (θ)': f"{theta:.3f}",
                'SE': f"{se:.3f}",
                't-statistic': f"{t_stat:.2f}",
                'p-value': p_fmt,
                '95% CI': f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            }])
        
        # Generate methods text
        methods_text = (
            "Causal effects were estimated using Double Machine Learning (DML), which provides "
            "valid statistical inference for treatment effects in high-dimensional settings. "
            "The DML framework uses cross-fitting to avoid overfitting bias and Neyman orthogonality "
            "to achieve √n-consistency. Machine learning models were used to control for confounding "
            "in the first stage, while the second stage estimates the causal parameter of interest. "
            "Standard errors were computed using the influence function approach, and significance "
            "was assessed using t-tests at α = 0.05."
        )
        
        # Supplementary statistics
        if 'plants_effect' in test_results:
            supplementary_stats = pd.DataFrame({
                'Test': ['H1: Biodiversity Hypothesis'],
                'Both Significant': [test_results['h1_supported']],
                'Either Significant': [test_results['h1_partial_support']],
                'Conclusion': [test_results['h1_conclusion']]
            })
        else:
            supplementary_stats = pd.DataFrame({
                'Statistic': ['Causal Effect', 'Standard Error', 't-statistic', 
                             'p-value', 'Decision'],
                'Value': [f"{test_results['theta']:.3f}", 
                         f"{test_results['se']:.3f}",
                         f"{test_results['t_statistic']:.2f}",
                         f"{test_results['p_value']:.4f}",
                         'Reject H₀' if test_results['reject_h0'] else 'Retain H₀']
            })
        
        return {
            'main_text': main_text,
            'methods_text': methods_text,
            'results_table': results_table,
            'supplementary_stats': supplementary_stats,
            'latex_table': DMLClassicalTest._generate_latex_table(results_table),
            'statistical_summary': test_results
        }
    
    @staticmethod
    def visualize_causal_effects(
        test_results: Dict[str, Any],
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """Create publication-quality visualization of DML causal effect tests.
        
        Creates forest plots and distribution visualizations for causal effects.
        
        Args:
            test_results: Results from test methods
            figsize: Figure size
            
        Returns:
            matplotlib figure suitable for publication
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Check if biodiversity hypothesis or single effect
        if 'plants_effect' in test_results:
            # Panel 1: Forest plot of causal effects
            ax1 = axes[0]
            
            effects = [
                ('Plant Richness', test_results['plants_effect']),
                ('Vertebrate Richness', test_results['vertebrates_effect'])
            ]
            
            y_pos = np.arange(len(effects))
            estimates = [e[1]['theta'] for e in effects]
            ses = [e[1]['se'] for e in effects]
            ci_lower = [e[1]['ci_lower'] for e in effects]
            ci_upper = [e[1]['ci_upper'] for e in effects]
            p_values = [e[1]['p_value'] for e in effects]
            
            # Plot confidence intervals
            for i, (name, result) in enumerate(effects):
                color = 'green' if result['reject_h0'] else 'gray'
                ax1.plot([ci_lower[i], ci_upper[i]], [i, i], 
                        color=color, linewidth=2, alpha=0.7)
                ax1.plot(estimates[i], i, 'o', color=color, markersize=10)
                
                # Add p-value annotation
                p_text = f"p < 0.001" if p_values[i] < 0.001 else f"p = {p_values[i]:.3f}"
                ax1.text(ci_upper[i] + 0.01, i, p_text, 
                        va='center', fontsize=9)
            
            ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([e[0] for e in effects])
            ax1.set_xlabel('Causal Effect (θ) with 95% CI', fontsize=11)
            ax1.set_title('Forest Plot of Causal Effects', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Panel 2: t-statistics comparison
            ax2 = axes[1]
            
            t_stats = [e[1]['t_statistic'] for e in effects]
            colors = ['green' if e[1]['reject_h0'] else 'gray' for e in effects]
            
            bars = ax2.bar([e[0] for e in effects], t_stats, color=colors, alpha=0.7)
            
            # Add critical values
            ax2.axhline(1.96, color='red', linestyle='--', alpha=0.5, 
                       label='Critical value (α=0.05)')
            ax2.axhline(-1.96, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            
            ax2.set_ylabel('t-statistic', fontsize=11)
            ax2.set_title('Test Statistics', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Panel 3: Effect size interpretation
            ax3 = axes[2]
            
            # H1 hypothesis support visualization
            support_text = test_results['h1_conclusion']
            
            # Create summary box
            props = dict(boxstyle='round,pad=1', facecolor='lightgreen' if test_results['h1_supported'] 
                        else 'lightcoral' if not test_results['h1_partial_support'] 
                        else 'lightyellow', alpha=0.8)
            
            summary = f"""H1 Biodiversity Hypothesis Test
            
Plant Effect: {test_results['plants_effect']['theta']:.3f} 
              (p = {test_results['plants_effect']['p_value']:.4f})
              
Vertebrate Effect: {test_results['vertebrates_effect']['theta']:.3f}
                  (p = {test_results['vertebrates_effect']['p_value']:.4f})

{support_text}"""
            
            ax3.text(0.5, 0.5, summary, transform=ax3.transAxes,
                    fontsize=10, ha='center', va='center',
                    bbox=props)
            ax3.axis('off')
            
        else:
            # Single effect visualization
            ax1 = axes[0]
            
            # Panel 1: Distribution with observed effect
            theta = test_results['theta']
            se = test_results['se']
            t_stat = test_results['t_statistic']
            p_value = test_results['p_value']
            
            # Create normal distribution
            x = np.linspace(-4, 4, 1000)
            y = stats.norm.pdf(x, 0, 1)
            
            ax1.plot(x, y, 'b-', linewidth=2, label='Null distribution')
            ax1.fill_between(x[np.abs(x) > 1.96], y[np.abs(x) > 1.96], 
                            alpha=0.3, color='red', label='Rejection region')
            ax1.axvline(t_stat, color='darkgreen', linestyle='--', linewidth=2.5,
                       label=f'Observed t = {t_stat:.2f}')
            
            ax1.set_xlabel('t-statistic', fontsize=11)
            ax1.set_ylabel('Probability Density', fontsize=11)
            ax1.set_title('Hypothesis Test', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Panel 2: Confidence interval
            ax2 = axes[1]
            
            ci_lower = test_results['ci_lower']
            ci_upper = test_results['ci_upper']
            
            ax2.plot([ci_lower, ci_upper], [0, 0], 'b-', linewidth=3)
            ax2.plot(theta, 0, 'ro', markersize=12, label=f'θ = {theta:.3f}')
            ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
            
            ax2.set_xlim([min(ci_lower*1.2, -0.1), max(ci_upper*1.2, 0.1)])
            ax2.set_ylim([-0.5, 0.5])
            ax2.set_xlabel('Causal Effect', fontsize=11)
            ax2.set_yticks([])
            ax2.set_title(f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]', 
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Panel 3: Summary
            ax3 = axes[2]
            
            color = 'lightgreen' if test_results['reject_h0'] else 'lightgray'
            props = dict(boxstyle='round,pad=1', facecolor=color, alpha=0.8)
            
            summary = f"""DML Causal Effect Test
            
Effect: θ = {theta:.3f} ± {se:.3f}
t-statistic: {t_stat:.2f}
p-value: {p_value:.4f}

Decision: {'Reject H₀' if test_results['reject_h0'] else 'Retain H₀'}
Interpretation: {test_results['interpretation']}"""
            
            ax3.text(0.5, 0.5, summary, transform=ax3.transAxes,
                    fontsize=10, ha='center', va='center',
                    bbox=props)
            ax3.axis('off')
        
        # Overall title
        fig.suptitle('DML Causal Inference Results: H1 Biodiversity Hypothesis',
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
        latex = r"""\begin{table}[htbp]
\centering
\caption{Double Machine Learning Causal Effect Estimates}
\label{tab:dml_effects}
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
\item Note: Causal effects estimated using Double Machine Learning with cross-fitting.
\item Standard errors computed using influence functions. Significance assessed at $\alpha = 0.05$.
\end{tablenotes}
\end{table}"""
        
        return latex