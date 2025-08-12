"""Likelihood Ratio Test implementation for GLM models."""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class LikelihoodRatioTest:
    """Likelihood Ratio Test for nested model comparison.
    
    The LRT compares nested models by:
    LR = -2 * (log L_null - log L_full) ~ chi2(df)
    where df = difference in number of parameters
    """
    
    @staticmethod
    def test_nested_models(
        null_model: Any,
        full_model: Any,
        df_difference: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform likelihood ratio test between nested models.
        
        Args:
            null_model: Restricted (null) model
            full_model: Unrestricted (full) model
            df_difference: Difference in degrees of freedom (if not provided, will try to infer)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        # Get log-likelihoods or deviances
        if hasattr(null_model, 'get_log_likelihood') and hasattr(full_model, 'get_log_likelihood'):
            null_ll = null_model.get_log_likelihood()
            full_ll = full_model.get_log_likelihood()
            lr_statistic = 2 * (full_ll - null_ll)
        elif hasattr(null_model, 'get_deviance') and hasattr(full_model, 'get_deviance'):
            # Deviance = -2 * log-likelihood (up to constant)
            null_deviance = null_model.get_deviance()
            full_deviance = full_model.get_deviance()
            lr_statistic = null_deviance - full_deviance
        else:
            raise ValueError("Models must have either get_log_likelihood() or get_deviance() methods")
        
        # Get degrees of freedom difference
        if df_difference is None:
            if hasattr(null_model, 'get_num_parameters') and hasattr(full_model, 'get_num_parameters'):
                null_params = null_model.get_num_parameters()
                full_params = full_model.get_num_parameters()
                df_difference = full_params - null_params
            elif hasattr(null_model, 'get_degrees_of_freedom') and hasattr(full_model, 'get_degrees_of_freedom'):
                null_df = null_model.get_degrees_of_freedom()
                full_df = full_model.get_degrees_of_freedom()
                df_difference = null_df - full_df  # Note: df decreases as params increase
            else:
                raise ValueError("Cannot determine degrees of freedom difference. Please provide df_difference parameter.")
        
        if df_difference <= 0:
            raise ValueError(f"Degrees of freedom difference must be positive (got {df_difference}). Ensure full model has more parameters than null model.")
        
        # Compute p-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(lr_statistic, df=df_difference)
        
        # Get additional model information if available
        result = {
            'lr_statistic': lr_statistic,
            'degrees_of_freedom': df_difference,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'likelihood_ratio'
        }
        
        # Add model fit statistics if available
        if hasattr(null_model, 'get_aic'):
            result['null_model_aic'] = null_model.get_aic()
        if hasattr(full_model, 'get_aic'):
            result['full_model_aic'] = full_model.get_aic()
        if hasattr(null_model, 'get_bic'):
            result['null_model_bic'] = null_model.get_bic()
        if hasattr(full_model, 'get_bic'):
            result['full_model_bic'] = full_model.get_bic()
        if hasattr(null_model, 'get_deviance'):
            result['null_model_deviance'] = null_model.get_deviance()
        if hasattr(full_model, 'get_deviance'):
            result['full_model_deviance'] = full_model.get_deviance()
        
        return result
    
    @staticmethod
    def test_coefficient_subset(
        full_model: Any,
        null_model: Any,
        coefficient_names: Optional[Union[str, list]] = None
    ) -> Dict[str, Any]:
        """Test significance of a subset of coefficients.
        
        This is a convenience wrapper for testing whether specific coefficients
        significantly improve model fit.
        
        Args:
            full_model: Model with all coefficients
            null_model: Model without the tested coefficients
            coefficient_names: Names of coefficients being tested (for documentation)
            
        Returns:
            Dictionary with test results
        """
        result = LikelihoodRatioTest.test_nested_models(null_model, full_model)
        
        if coefficient_names:
            if isinstance(coefficient_names, str):
                coefficient_names = [coefficient_names]
            result['tested_coefficients'] = coefficient_names
            
            # Try to get coefficient values from full model
            if hasattr(full_model, 'get_coefficient'):
                coef_values = {}
                for name in coefficient_names:
                    coef_info = full_model.get_coefficient(name)
                    if coef_info:
                        coef_values[name] = {
                            'estimate': coef_info.get('coefficient'),
                            'std_error': coef_info.get('std_error'),
                            'p_value': coef_info.get('p_value')
                        }
                if coef_values:
                    result['coefficient_estimates'] = coef_values
        
        return result
    
    @staticmethod
    def test_deviance_difference(
        null_deviance: float,
        full_deviance: float,
        df_difference: int
    ) -> Dict[str, Any]:
        """Perform LRT using deviance values directly.
        
        Useful when you have deviance values but not model objects.
        
        Args:
            null_deviance: Deviance of null model
            full_deviance: Deviance of full model
            df_difference: Difference in degrees of freedom
            
        Returns:
            Dictionary with test statistics and p-value
        """
        if df_difference <= 0:
            raise ValueError(f"Degrees of freedom difference must be positive (got {df_difference})")
        
        lr_statistic = null_deviance - full_deviance
        
        if lr_statistic < 0:
            logger.warning(f"Negative LR statistic ({lr_statistic}). Full model may have higher deviance than null model.")
        
        # Compute p-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(max(0, lr_statistic), df=df_difference)
        
        return {
            'null_deviance': null_deviance,
            'full_deviance': full_deviance,
            'lr_statistic': lr_statistic,
            'degrees_of_freedom': df_difference,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'likelihood_ratio_deviance'
        }
    
    @staticmethod
    def sequential_test(
        models: list,
        model_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """Perform sequential likelihood ratio tests on a series of nested models.
        
        Args:
            models: List of models, each nested within the next
            model_names: Optional names for the models
            
        Returns:
            Dictionary with sequential test results
        """
        if len(models) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        if model_names and len(model_names) != len(models):
            raise ValueError("Number of model names must match number of models")
        
        if not model_names:
            model_names = [f"Model_{i}" for i in range(len(models))]
        
        results = {
            'n_models': len(models),
            'model_names': model_names,
            'comparisons': [],
            'test_type': 'sequential_likelihood_ratio'
        }
        
        # Perform pairwise comparisons
        for i in range(len(models) - 1):
            comparison = {
                'null_model': model_names[i],
                'full_model': model_names[i + 1],
            }
            
            try:
                test_result = LikelihoodRatioTest.test_nested_models(
                    null_model=models[i],
                    full_model=models[i + 1]
                )
                comparison.update(test_result)
            except Exception as e:
                logger.warning(f"Failed to compare {model_names[i]} vs {model_names[i+1]}: {e}")
                comparison['error'] = str(e)
            
            results['comparisons'].append(comparison)
        
        # Overall test (first vs last)
        if len(models) > 2:
            try:
                overall_test = LikelihoodRatioTest.test_nested_models(
                    null_model=models[0],
                    full_model=models[-1]
                )
                results['overall_test'] = {
                    'null_model': model_names[0],
                    'full_model': model_names[-1],
                    **overall_test
                }
            except Exception as e:
                logger.warning(f"Failed to perform overall test: {e}")
                results['overall_test'] = {'error': str(e)}
        
        return results
    
    @staticmethod
    def generate_publication_report(
        test_results: Dict[str, Any],
        test_name: str = "Likelihood Ratio Test",
        hypothesis: str = "model comparison",
        null_model_name: str = "Null Model",
        full_model_name: str = "Full Model",
        method_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate publication-ready report for LRT results.
        
        Following ecology journal standards, particularly Methods in Ecology and Evolution.
        
        Args:
            test_results: LRT results dictionary
            test_name: Name of the test for reporting
            hypothesis: Description of hypothesis being tested
            null_model_name: Name of simpler model
            full_model_name: Name of complex model
            method_number: Method number for cross-reference
            
        Returns:
            Dictionary with formatted text, tables, and LaTeX code
        """
        import pandas as pd
        
        # Extract test statistics
        lr_stat = test_results.get('lr_statistic', test_results.get('statistic'))
        p_value = test_results.get('p_value')
        df = test_results.get('df', test_results.get('degrees_of_freedom'))
        
        # Model fit statistics
        null_ll = test_results.get('null_log_likelihood')
        full_ll = test_results.get('full_log_likelihood')
        null_aic = test_results.get('null_aic')
        full_aic = test_results.get('full_aic')
        null_bic = test_results.get('null_bic')
        full_bic = test_results.get('full_bic')
        
        # Create comparison table
        rows = []
        
        # Null model row
        null_row = {'Model': null_model_name}
        if null_ll is not None:
            null_row['Log-Likelihood'] = f"{null_ll:.2f}"
        if null_aic is not None:
            null_row['AIC'] = f"{null_aic:.1f}"
        if null_bic is not None:
            null_row['BIC'] = f"{null_bic:.1f}"
        null_row['Parameters'] = test_results.get('null_parameters', 'NA')
        rows.append(null_row)
        
        # Full model row
        full_row = {'Model': full_model_name}
        if full_ll is not None:
            full_row['Log-Likelihood'] = f"{full_ll:.2f}"
        if full_aic is not None:
            full_row['AIC'] = f"{full_aic:.1f}"
            if null_aic is not None:
                full_row['ΔAIC'] = f"{null_aic - full_aic:.1f}"
        if full_bic is not None:
            full_row['BIC'] = f"{full_bic:.1f}"
            if null_bic is not None:
                full_row['ΔBIC'] = f"{null_bic - full_bic:.1f}"
        full_row['Parameters'] = test_results.get('full_parameters', 'NA')
        rows.append(full_row)
        
        results_table = pd.DataFrame(rows)
        
        # Generate main text
        main_text = f"Model selection was performed using the {test_name} to test {hypothesis}. "
        
        if lr_stat is not None and p_value is not None:
            main_text += (
                f"The {full_model_name} was compared against the {null_model_name}. "
                f"The likelihood ratio test statistic was LR = {lr_stat:.2f} "
                f"(χ²({df}) = {lr_stat:.2f}, p = {p_value:.4f}). "
            )
            
            if p_value < 0.001:
                main_text += f"This provides very strong evidence that the {full_model_name} "
                main_text += f"significantly improves model fit compared to the {null_model_name}. "
            elif p_value < 0.01:
                main_text += f"This provides strong evidence favoring the {full_model_name}. "
            elif p_value < 0.05:
                main_text += f"The {full_model_name} provides significantly better fit (p < 0.05). "
            else:
                main_text += (
                    f"There is insufficient evidence to prefer the {full_model_name} "
                    f"over the simpler {null_model_name}. "
                )
        
        # Add AIC/BIC comparison if available
        if null_aic is not None and full_aic is not None:
            delta_aic = null_aic - full_aic
            if delta_aic > 10:
                main_text += f"The AIC strongly supports the {full_model_name} (ΔAIC = {delta_aic:.1f}). "
            elif delta_aic > 2:
                main_text += f"The AIC favors the {full_model_name} (ΔAIC = {delta_aic:.1f}). "
            elif delta_aic < -2:
                main_text += f"The AIC favors the simpler {null_model_name}. "
        
        # Methods text following ecology standards
        methods_text = (
            "Model comparison was conducted using the likelihood ratio test (LRT), which "
            "compares the goodness-of-fit between nested models. The test statistic is "
            "calculated as LR = 2(ln L_full - ln L_null), where L represents the likelihood "
            "of each model. Under the null hypothesis, the LR statistic follows a chi-square "
            "distribution with degrees of freedom equal to the difference in parameters "
            "between models. "
        )
        
        # Supplementary information
        supplementary_info = {
            'test_type': 'Likelihood Ratio Test (LRT)',
            'null_hypothesis': f'{null_model_name} is adequate',
            'alternative': f'{full_model_name} provides better fit',
            'assumptions': [
                'Models are nested (null is special case of full)',
                'Large sample size for chi-square approximation',
                'Models correctly specified',
                'Independence of observations'
            ],
            'interpretation_guidelines': {
                'p < 0.001': 'Very strong evidence for complex model',
                'p < 0.01': 'Strong evidence for complex model',
                'p < 0.05': 'Moderate evidence for complex model',
                'p ≥ 0.05': 'Insufficient evidence; prefer simpler model',
                'ΔAIC > 10': 'Essentially no support for simpler model',
                'ΔAIC 4-7': 'Considerably less support for simpler model',
                'ΔAIC 2-4': 'Some support for complex model',
                'ΔAIC < 2': 'Models are competitive'
            }
        }
        
        return {
            'main_text': main_text,
            'methods_text': methods_text,
            'results_table': results_table,
            'supplementary_info': supplementary_info,
            'test_statistics': {
                'lr_statistic': lr_stat,
                'df': df,
                'p_value': p_value,
                'delta_aic': null_aic - full_aic if null_aic and full_aic else None,
                'delta_bic': null_bic - full_bic if null_bic and full_bic else None
            }
        }
    
    @staticmethod
    def visualize_lrt(
        test_results: Dict[str, Any],
        null_model_name: str = "Null",
        full_model_name: str = "Full",
        title: str = "Likelihood Ratio Test Results",
        figsize: Tuple[int, int] = (15, 5)
    ) -> 'plt.Figure':
        """Create publication-quality visualization of LRT results.
        
        Creates model comparison plots and chi-square distribution visualization.
        
        Args:
            test_results: LRT results
            null_model_name: Name of null model
            full_model_name: Name of full model
            title: Plot title
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        
        # Set style for publication-quality plots
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Extract statistics
        lr_stat = test_results.get('lr_statistic', 0)
        p_value = test_results.get('p_value', 1)
        df = test_results.get('df', 1)
        null_ll = test_results.get('null_log_likelihood', 0)
        full_ll = test_results.get('full_log_likelihood', 0)
        
        # Panel 1: Log-likelihood comparison
        ax1 = axes[0]
        
        models = [null_model_name, full_model_name]
        lls = [null_ll, full_ll]
        colors = ['gray', 'green' if p_value < 0.05 else 'orange']
        
        bars = ax1.bar(models, lls, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, ll in zip(bars, lls):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ll:.1f}', ha='center', va='bottom' if ll > 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        # Add LR statistic annotation
        if len(lls) == 2 and lls[0] != 0 and lls[1] != 0:
            mid_y = (lls[0] + lls[1]) / 2
            ax1.annotate('', xy=(1, lls[1]), xytext=(0, lls[0]),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax1.text(0.5, mid_y, f'LR = {lr_stat:.2f}', 
                    ha='center', va='center', color='red',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_ylabel('Log-Likelihood', fontsize=11)
        ax1.set_title('Model Log-Likelihoods', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Chi-square distribution with test statistic
        ax2 = axes[1]
        
        # Generate chi-square distribution
        x = np.linspace(0, max(lr_stat * 1.5, stats.chi2.ppf(0.999, df)), 1000)
        y = stats.chi2.pdf(x, df)
        
        # Plot distribution
        ax2.plot(x, y, 'b-', linewidth=2, label=f'χ²({df}) distribution')
        ax2.fill_between(x, 0, y, alpha=0.3)
        
        # Mark critical value (α = 0.05)
        crit_val = stats.chi2.ppf(0.95, df)
        ax2.axvline(crit_val, color='orange', linestyle='--', 
                   label=f'Critical value (α=0.05): {crit_val:.2f}')
        
        # Mark observed test statistic
        ax2.axvline(lr_stat, color='red', linewidth=3,
                   label=f'Observed LR: {lr_stat:.2f}')
        
        # Shade rejection region
        x_reject = x[x > crit_val]
        y_reject = stats.chi2.pdf(x_reject, df)
        ax2.fill_between(x_reject, 0, y_reject, alpha=0.3, color='red')
        
        # Add p-value annotation
        ax2.text(lr_stat, max(y) * 0.5,
                f'p = {p_value:.4f}', ha='left', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax2.set_xlabel('Test Statistic', fontsize=11)
        ax2.set_ylabel('Probability Density', fontsize=11)
        ax2.set_title('Chi-Square Test', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Information criteria comparison if available
        ax3 = axes[2]
        
        null_aic = test_results.get('null_aic')
        full_aic = test_results.get('full_aic')
        null_bic = test_results.get('null_bic')
        full_bic = test_results.get('full_bic')
        
        if null_aic and full_aic:
            # Prepare data for grouped bar plot
            criteria = ['AIC', 'BIC']
            null_values = [null_aic, null_bic if null_bic else null_aic]
            full_values = [full_aic, full_bic if full_bic else full_aic]
            
            x = np.arange(len(criteria))
            width = 0.35
            
            # Create bars
            bars1 = ax3.bar(x - width/2, null_values, width, 
                          label=null_model_name, color='gray', alpha=0.7)
            bars2 = ax3.bar(x + width/2, full_values, width,
                          label=full_model_name, color='green', alpha=0.7)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom',
                           fontsize=9)
            
            # Add delta annotations
            for i, criterion in enumerate(criteria):
                if i == 0:  # AIC
                    delta = null_aic - full_aic
                else:  # BIC
                    delta = (null_bic - full_bic) if null_bic and full_bic else 0
                
                if delta != 0:
                    ax3.text(i, max(null_values[i], full_values[i]) * 1.05,
                           f'Δ = {delta:.1f}',
                           ha='center', fontsize=9, fontweight='bold',
                           color='green' if delta > 2 else 'orange' if delta > 0 else 'red')
            
            ax3.set_ylabel('Information Criterion Value', fontsize=11)
            ax3.set_title('Model Selection Criteria', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(criteria)
            ax3.legend(loc='upper right', fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add interpretation note
            ax3.text(0.5, -0.15, 'Lower values indicate better fit',
                    ha='center', transform=ax3.transAxes,
                    fontsize=9, style='italic')
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig