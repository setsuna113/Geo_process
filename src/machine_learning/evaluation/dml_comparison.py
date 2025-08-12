"""
Model Comparison for DML (Option 4).

Compares models with and without biodiversity predictors
using residual-based metrics and pseudo R-squared.
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


class DMLModelComparison:
    """Model comparison for DML using residual analysis.
    
    Evaluates whether biodiversity predictors improve model fit
    by comparing residuals and calculating improvement metrics.
    """
    
    @staticmethod
    def compare_models(
        residuals_with_biodiv: np.ndarray,
        residuals_without_biodiv: np.ndarray,
        residuals_A1: Optional[np.ndarray] = None,
        residuals_A2: Optional[np.ndarray] = None,
        Y_observed: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compare DML models with and without biodiversity predictors.
        
        Args:
            residuals_with_biodiv: Residuals from model with biodiversity
            residuals_without_biodiv: Residuals from model without biodiversity
            residuals_A1: Plants residuals (for correlation analysis)
            residuals_A2: Vertebrates residuals (for correlation analysis)
            Y_observed: Original outcome values (for R² calculation)
            
        Returns:
            Dictionary with model comparison results
        """
        n = len(residuals_with_biodiv)
        
        if len(residuals_without_biodiv) != n:
            raise ValueError("Residuals must have same length")
        
        # Calculate RSS for each model
        rss_with = np.sum(residuals_with_biodiv ** 2)
        rss_without = np.sum(residuals_without_biodiv ** 2)
        
        # Pseudo R-squared improvement
        r2_improvement = 1 - (rss_with / rss_without) if rss_without > 0 else 0
        
        # F-statistic for model comparison
        # F = [(RSS_reduced - RSS_full) / p] / [RSS_full / (n - p - 1)]
        p = 2 if residuals_A2 is not None else 1  # Number of biodiversity predictors
        f_stat = ((rss_without - rss_with) / p) / (rss_with / (n - p - 1))
        f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
        
        # Calculate R² if original outcome provided
        r2_with = None
        r2_without = None
        if Y_observed is not None:
            tss = np.sum((Y_observed - np.mean(Y_observed)) ** 2)
            r2_with = 1 - (rss_with / tss) if tss > 0 else 0
            r2_without = 1 - (rss_without / tss) if tss > 0 else 0
        
        # RMSE comparison
        rmse_with = np.sqrt(rss_with / n)
        rmse_without = np.sqrt(rss_without / n)
        rmse_reduction = (rmse_without - rmse_with) / rmse_without if rmse_without > 0 else 0
        
        # MAE comparison
        mae_with = np.mean(np.abs(residuals_with_biodiv))
        mae_without = np.mean(np.abs(residuals_without_biodiv))
        mae_reduction = (mae_without - mae_with) / mae_without if mae_without > 0 else 0
        
        # Likelihood ratio test approximation
        # LR = n * log(RSS_reduced / RSS_full)
        lr_stat = n * np.log(rss_without / rss_with) if rss_with > 0 and rss_without > 0 else 0
        lr_p_value = 1 - stats.chi2.cdf(lr_stat, p)
        
        # Classification of improvement
        classification = DMLModelComparison._classify_improvement(
            r2_improvement, f_p_value, rmse_reduction
        )
        
        # Generate interpretation
        interpretation = DMLModelComparison._generate_interpretation(
            r2_improvement, f_p_value, rmse_reduction, classification
        )
        
        result = {
            'rss_with_biodiv': float(rss_with),
            'rss_without_biodiv': float(rss_without),
            'r2_improvement': float(r2_improvement),
            'f_statistic': float(f_stat),
            'f_p_value': float(f_p_value),
            'lr_statistic': float(lr_stat),
            'lr_p_value': float(lr_p_value),
            'rmse_with': float(rmse_with),
            'rmse_without': float(rmse_without),
            'rmse_reduction': float(rmse_reduction),
            'mae_with': float(mae_with),
            'mae_without': float(mae_without),
            'mae_reduction': float(mae_reduction),
            'classification': classification,
            'interpretation': interpretation,
            'n_observations': n,
            'n_biodiv_predictors': p
        }
        
        # Add R² values if available
        if r2_with is not None:
            result['r2_with_biodiv'] = float(r2_with)
            result['r2_without_biodiv'] = float(r2_without)
            result['r2_absolute_improvement'] = float(r2_with - r2_without)
        
        # Add correlation analysis if residuals provided
        if residuals_A1 is not None:
            correlations = DMLModelComparison._analyze_residual_correlations(
                residuals_with_biodiv, residuals_A1, residuals_A2
            )
            result['residual_correlations'] = correlations
        
        logger.info(f"Model comparison: R² improvement={r2_improvement:.4f}, "
                   f"F-test p={f_p_value:.4f}, Classification: {classification}")
        
        return result
    
    @staticmethod
    def visualize_residual_relationships(
        residuals_Y: np.ndarray,
        residuals_A1: np.ndarray,
        residuals_A2: Optional[np.ndarray] = None,
        sample_size: int = 5000
    ) -> plt.Figure:
        """Create visualization of residual relationships.
        
        Args:
            residuals_Y: Outcome residuals
            residuals_A1: First treatment residuals
            residuals_A2: Second treatment residuals
            sample_size: Number of points to plot (for large datasets)
            
        Returns:
            Matplotlib figure
        """
        # Sample if dataset is large
        n = len(residuals_Y)
        if n > sample_size:
            idx = np.random.choice(n, sample_size, replace=False)
            residuals_Y_plot = residuals_Y[idx]
            residuals_A1_plot = residuals_A1[idx]
            residuals_A2_plot = residuals_A2[idx] if residuals_A2 is not None else None
        else:
            residuals_Y_plot = residuals_Y
            residuals_A1_plot = residuals_A1
            residuals_A2_plot = residuals_A2
        
        # Create figure
        n_plots = 2 if residuals_A2 is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot 1: Plants vs Fungi residuals
        ax1 = axes[0]
        ax1.scatter(residuals_A1_plot, residuals_Y_plot, alpha=0.3, s=1)
        
        # Add regression line
        z = np.polyfit(residuals_A1_plot, residuals_Y_plot, 1)
        p = np.poly1d(z)
        x_line = np.linspace(residuals_A1_plot.min(), residuals_A1_plot.max(), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Slope: {z[0]:.3f}')
        
        # Calculate correlation
        corr1 = np.corrcoef(residuals_A1_plot, residuals_Y_plot)[0, 1]
        
        ax1.set_xlabel('Unexpected Plant Richness')
        ax1.set_ylabel('Unexpected Fungi Richness')
        ax1.set_title(f'Biodiversity Coupling (r={corr1:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Vertebrates vs Fungi residuals (if provided)
        if residuals_A2_plot is not None:
            ax2 = axes[1]
            ax2.scatter(residuals_A2_plot, residuals_Y_plot, alpha=0.3, s=1, color='green')
            
            # Add regression line
            z2 = np.polyfit(residuals_A2_plot, residuals_Y_plot, 1)
            p2 = np.poly1d(z2)
            x_line2 = np.linspace(residuals_A2_plot.min(), residuals_A2_plot.max(), 100)
            ax2.plot(x_line2, p2(x_line2), 'r-', linewidth=2, label=f'Slope: {z2[0]:.3f}')
            
            # Calculate correlation
            corr2 = np.corrcoef(residuals_A2_plot, residuals_Y_plot)[0, 1]
            
            ax2.set_xlabel('Unexpected Vertebrate Richness')
            ax2.set_ylabel('Unexpected Fungi Richness')
            ax2.set_title(f'Biodiversity Coupling (r={corr2:.3f})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add overall interpretation
        if corr1 > 0 and (residuals_A2 is None or corr2 > 0):
            fig.suptitle('Positive slopes indicate biodiversity begets biodiversity', 
                        y=1.02, fontsize=12)
        
        return fig
    
    @staticmethod
    def _classify_improvement(
        r2_improvement: float,
        f_p_value: float,
        rmse_reduction: float
    ) -> str:
        """Classify the level of model improvement.
        
        Args:
            r2_improvement: Pseudo R² improvement
            f_p_value: P-value from F-test
            rmse_reduction: Relative RMSE reduction
            
        Returns:
            Classification string
        """
        if f_p_value < 0.001 and r2_improvement > 0.05:
            return "substantial_improvement"
        elif f_p_value < 0.01 and r2_improvement > 0.02:
            return "moderate_improvement"
        elif f_p_value < 0.05 and r2_improvement > 0.01:
            return "modest_improvement"
        elif f_p_value < 0.05:
            return "small_improvement"
        else:
            return "negligible_improvement"
    
    @staticmethod
    def _generate_interpretation(
        r2_improvement: float,
        f_p_value: float,
        rmse_reduction: float,
        classification: str
    ) -> str:
        """Generate interpretation of model comparison.
        
        Args:
            r2_improvement: Pseudo R² improvement
            f_p_value: P-value from F-test
            rmse_reduction: Relative RMSE reduction
            classification: Improvement classification
            
        Returns:
            Interpretation string
        """
        interpretations = {
            "substantial_improvement": (
                f"Biodiversity predictors SUBSTANTIALLY improve model fit "
                f"(R² improvement: {r2_improvement:.3f}, p<0.001). "
                f"Strong support for H1 - biodiversity begets biodiversity."
            ),
            "moderate_improvement": (
                f"Biodiversity predictors provide MODERATE improvement "
                f"(R² improvement: {r2_improvement:.3f}, p={f_p_value:.4f}). "
                f"Good evidence for H1."
            ),
            "modest_improvement": (
                f"Biodiversity predictors provide MODEST improvement "
                f"(R² improvement: {r2_improvement:.3f}, p={f_p_value:.4f}). "
                f"Some support for H1."
            ),
            "small_improvement": (
                f"Biodiversity predictors provide SMALL but significant improvement "
                f"(R² improvement: {r2_improvement:.3f}, p={f_p_value:.4f}). "
                f"Weak support for H1."
            ),
            "negligible_improvement": (
                f"Biodiversity predictors provide NEGLIGIBLE improvement "
                f"(R² improvement: {r2_improvement:.3f}, p={f_p_value:.4f}). "
                f"No support for H1 - cannot reject H0."
            )
        }
        
        base_interpretation = interpretations.get(classification, "")
        
        # Add RMSE information
        if rmse_reduction > 0:
            base_interpretation += f" RMSE reduced by {rmse_reduction:.1%}."
        
        return base_interpretation
    
    @staticmethod
    def _analyze_residual_correlations(
        residuals_Y: np.ndarray,
        residuals_A1: np.ndarray,
        residuals_A2: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Analyze correlations between residuals.
        
        Args:
            residuals_Y: Outcome residuals
            residuals_A1: First treatment residuals
            residuals_A2: Second treatment residuals
            
        Returns:
            Dictionary with correlation analysis
        """
        # Basic correlations
        corr_Y_A1 = np.corrcoef(residuals_Y, residuals_A1)[0, 1]
        
        analysis = {
            'fungi_plants_correlation': float(corr_Y_A1),
            'fungi_plants_interpretation': (
                "Positive biodiversity coupling" if corr_Y_A1 > 0.1 
                else "No clear coupling" if corr_Y_A1 > -0.1 
                else "Negative relationship"
            )
        }
        
        if residuals_A2 is not None:
            corr_Y_A2 = np.corrcoef(residuals_Y, residuals_A2)[0, 1]
            corr_A1_A2 = np.corrcoef(residuals_A1, residuals_A2)[0, 1]
            
            analysis.update({
                'fungi_vertebrates_correlation': float(corr_Y_A2),
                'plants_vertebrates_correlation': float(corr_A1_A2),
                'fungi_vertebrates_interpretation': (
                    "Positive biodiversity coupling" if corr_Y_A2 > 0.1 
                    else "No clear coupling" if corr_Y_A2 > -0.1 
                    else "Negative relationship"
                ),
                'overall_coupling': (
                    "Strong biodiversity coupling" if corr_Y_A1 > 0.2 and corr_Y_A2 > 0.2
                    else "Moderate coupling" if corr_Y_A1 > 0.1 or corr_Y_A2 > 0.1
                    else "Weak or no coupling"
                )
            })
        
        return analysis
    
    @staticmethod
    def calculate_information_criteria(
        rss: float,
        n: int,
        k: int,
        log_likelihood: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate AIC and BIC for model comparison.
        
        Args:
            rss: Residual sum of squares
            n: Number of observations
            k: Number of parameters
            log_likelihood: Optional log-likelihood value
            
        Returns:
            Dictionary with AIC and BIC
        """
        if log_likelihood is None:
            # Approximate log-likelihood from RSS (assuming normal errors)
            sigma2 = rss / n
            log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        
        # AIC = 2k - 2*log(L)
        aic = 2 * k - 2 * log_likelihood
        
        # BIC = k*log(n) - 2*log(L)
        bic = k * np.log(n) - 2 * log_likelihood
        
        return {
            'AIC': float(aic),
            'BIC': float(bic),
            'log_likelihood': float(log_likelihood)
        }
    
    @staticmethod
    def generate_publication_report(
        comparison_results: Dict[str, Any],
        hypothesis: str = "H1 (Biodiversity Begets Biodiversity)",
        variable_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate publication-ready report for DML model comparison.
        
        Following ecology journal standards for reporting model improvement tests.
        
        Args:
            comparison_results: Results from compare_models
            hypothesis: Hypothesis being tested
            variable_names: Names of biodiversity variables
            
        Returns:
            Dictionary with formatted text, tables, and visualizations
        """
        if variable_names is None:
            variable_names = ['plant richness', 'vertebrate richness']
        
        # Extract key statistics
        r2_improvement = comparison_results['r2_improvement']
        f_stat = comparison_results['f_statistic']
        f_p = comparison_results['f_p_value']
        lr_stat = comparison_results['lr_statistic']
        lr_p = comparison_results['lr_p_value']
        rmse_reduction = comparison_results['rmse_reduction']
        mae_reduction = comparison_results['mae_reduction']
        classification = comparison_results['classification']
        
        # Format p-values
        def format_p(p):
            if p < 0.001:
                return "< 0.001", "p < 0.001"
            elif p < 0.01:
                return f"{p:.3f}", f"p = {p:.3f}"
            else:
                return f"{p:.2f}", f"p = {p:.2f}"
        
        f_p_fmt, f_p_str = format_p(f_p)
        lr_p_fmt, lr_p_str = format_p(lr_p)
        
        # Main results text
        main_text = (
            f"Model comparison was performed to test {hypothesis} using Double Machine Learning. "
            f"Adding {' and '.join(variable_names)} to the model resulted in a pseudo-R² improvement "
            f"of {r2_improvement:.3f}. The F-test for model improvement yielded F = {f_stat:.2f} "
            f"(df = {comparison_results['n_biodiv_predictors']}, {comparison_results['n_observations'] - comparison_results['n_biodiv_predictors'] - 1}), "
            f"{f_p_str}. "
        )
        
        if comparison_results['classification'] in ['substantial_improvement', 'moderate_improvement']:
            main_text += (
                f"This represents a {classification.replace('_', ' ')} in model fit, "
                f"providing {'strong' if 'substantial' in classification else 'good'} support for H1. "
                f"RMSE was reduced by {rmse_reduction:.1%} and MAE by {mae_reduction:.1%}."
            )
        elif comparison_results['classification'] == 'modest_improvement':
            main_text += (
                f"The improvement is modest but statistically significant, providing some support for H1. "
                f"RMSE was reduced by {rmse_reduction:.1%}."
            )
        else:
            main_text += (
                f"The improvement is {'negligible' if 'negligible' in classification else 'small'}, "
                f"{'providing no' if 'negligible' in classification else 'providing only weak'} support for H1."
            )
        
        # Add R² values if available
        if 'r2_with_biodiv' in comparison_results:
            main_text += (
                f" The R² increased from {comparison_results['r2_without_biodiv']:.3f} "
                f"to {comparison_results['r2_with_biodiv']:.3f}."
            )
        
        # Methods text
        methods_text = (
            "Model comparison was conducted using residual-based metrics from Double Machine Learning. "
            "Models with and without biodiversity predictors were compared using F-tests for nested models "
            "and likelihood ratio tests. The pseudo-R² improvement quantifies the proportional reduction "
            "in residual sum of squares. Residual correlations between outcome and treatment residuals "
            "were analyzed to assess biodiversity coupling patterns. Model improvement was classified as "
            "substantial (p < 0.001, R² improvement > 0.05), moderate (p < 0.01, R² improvement > 0.02), "
            "modest (p < 0.05, R² improvement > 0.01), or negligible."
        )
        
        # Create results table
        results_table = pd.DataFrame([
            {
                'Metric': 'RSS with biodiversity',
                'Value': f"{comparison_results['rss_with_biodiv']:.1f}"
            },
            {
                'Metric': 'RSS without biodiversity',
                'Value': f"{comparison_results['rss_without_biodiv']:.1f}"
            },
            {
                'Metric': 'Pseudo-R² improvement',
                'Value': f"{r2_improvement:.3f}"
            },
            {
                'Metric': 'F-statistic',
                'Value': f"{f_stat:.2f}"
            },
            {
                'Metric': 'F-test p-value',
                'Value': f_p_fmt
            },
            {
                'Metric': 'LR statistic',
                'Value': f"{lr_stat:.2f}"
            },
            {
                'Metric': 'LR test p-value',
                'Value': lr_p_fmt
            },
            {
                'Metric': 'RMSE reduction',
                'Value': f"{rmse_reduction:.1%}"
            },
            {
                'Metric': 'MAE reduction',
                'Value': f"{mae_reduction:.1%}"
            },
            {
                'Metric': 'Classification',
                'Value': classification.replace('_', ' ').title()
            }
        ])
        
        # Add correlation results if available
        correlation_table = None
        if 'residual_correlations' in comparison_results:
            corr_data = comparison_results['residual_correlations']
            corr_rows = [
                {
                    'Relationship': 'Fungi ~ Plants',
                    'Correlation': f"{corr_data['fungi_plants_correlation']:.3f}",
                    'Interpretation': corr_data['fungi_plants_interpretation']
                }
            ]
            
            if 'fungi_vertebrates_correlation' in corr_data:
                corr_rows.append({
                    'Relationship': 'Fungi ~ Vertebrates',
                    'Correlation': f"{corr_data['fungi_vertebrates_correlation']:.3f}",
                    'Interpretation': corr_data['fungi_vertebrates_interpretation']
                })
                corr_rows.append({
                    'Relationship': 'Plants ~ Vertebrates',
                    'Correlation': f"{corr_data['plants_vertebrates_correlation']:.3f}",
                    'Interpretation': 'Inter-predictor correlation'
                })
            
            correlation_table = pd.DataFrame(corr_rows)
        
        # Supplementary statistics
        supplementary_stats = pd.DataFrame({
            'Test': ['F-test', 'Likelihood Ratio'],
            'Statistic': [f"{f_stat:.3f}", f"{lr_stat:.3f}"],
            'df': [f"{comparison_results['n_biodiv_predictors']}",
                  f"{comparison_results['n_biodiv_predictors']}"],
            'p-value': [f_p_fmt, lr_p_fmt],
            'Decision': ['Reject H₀' if f_p < 0.05 else 'Retain H₀',
                        'Reject H₀' if lr_p < 0.05 else 'Retain H₀']
        })
        
        return {
            'main_text': main_text,
            'methods_text': methods_text,
            'results_table': results_table,
            'correlation_table': correlation_table,
            'supplementary_stats': supplementary_stats,
            'latex_table': DMLModelComparison._generate_latex_table(results_table),
            'statistical_summary': comparison_results
        }
    
    @staticmethod
    def visualize_model_comparison(
        comparison_results: Dict[str, Any],
        residuals_with: Optional[np.ndarray] = None,
        residuals_without: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """Create comprehensive visualization of DML model comparison.
        
        Args:
            comparison_results: Results from compare_models
            residuals_with: Residuals from model with biodiversity
            residuals_without: Residuals from model without biodiversity
            figsize: Figure size
            
        Returns:
            matplotlib figure suitable for publication
        """
        fig = plt.figure(figsize=figsize)
        
        # Create grid specification for uneven layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Model fit comparison bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        
        metrics = ['RSS', 'RMSE', 'MAE']
        without_vals = [
            comparison_results['rss_without_biodiv'],
            comparison_results['rmse_without'],
            comparison_results['mae_without']
        ]
        with_vals = [
            comparison_results['rss_with_biodiv'],
            comparison_results['rmse_with'],
            comparison_results['mae_with']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, without_vals, width, label='Without biodiversity',
                       color='lightcoral', alpha=0.7)
        bars2 = ax1.bar(x + width/2, with_vals, width, label='With biodiversity',
                       color='lightgreen', alpha=0.7)
        
        # Add percentage improvement labels
        improvements = [
            (without_vals[i] - with_vals[i]) / without_vals[i] * 100
            for i in range(len(metrics))
        ]
        
        for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvements)):
            if imp > 0:
                y_pos = max(bar1.get_height(), bar2.get_height()) * 1.05
                ax1.text(i, y_pos, f'-{imp:.1f}%', ha='center', fontsize=9,
                        color='green', fontweight='bold')
        
        ax1.set_xlabel('Metric', fontsize=11)
        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title('Model Fit Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: R² and improvement visualization
        ax2 = fig.add_subplot(gs[0, 1])
        
        if 'r2_with_biodiv' in comparison_results:
            r2_without = comparison_results['r2_without_biodiv']
            r2_with = comparison_results['r2_with_biodiv']
            
            # Stacked bar showing improvement
            ax2.bar(0, r2_without, width=0.5, color='lightcoral', alpha=0.7,
                   label=f'Base R² = {r2_without:.3f}')
            ax2.bar(0, r2_with - r2_without, width=0.5, bottom=r2_without,
                   color='green', alpha=0.7,
                   label=f'Improvement = {r2_with - r2_without:.3f}')
            
            ax2.set_xlim(-0.5, 0.5)
            ax2.set_ylim(0, min(1, r2_with * 1.2))
            ax2.set_xticks([])
            ax2.set_ylabel('R²', fontsize=11)
            ax2.set_title('R² Improvement', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper left', fontsize=9)
        else:
            # Show pseudo-R² improvement
            r2_imp = comparison_results['r2_improvement']
            color = 'green' if r2_imp > 0.02 else 'yellow' if r2_imp > 0.01 else 'red'
            
            ax2.bar(0, r2_imp, width=0.5, color=color, alpha=0.7)
            ax2.text(0, r2_imp/2, f'{r2_imp:.3f}', ha='center', va='center',
                    fontsize=12, fontweight='bold')
            
            ax2.set_xlim(-0.5, 0.5)
            ax2.set_ylim(0, max(0.1, r2_imp * 1.5))
            ax2.set_xticks([])
            ax2.set_ylabel('Pseudo-R² Improvement', fontsize=11)
            ax2.set_title('Model Improvement', fontsize=12, fontweight='bold')
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Statistical tests
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Create test statistics visualization
        tests = ['F-test', 'LR test']
        p_values = [comparison_results['f_p_value'], comparison_results['lr_p_value']]
        statistics = [comparison_results['f_statistic'], comparison_results['lr_statistic']]
        
        # Convert to -log10(p) for better visualization
        neg_log_p = [-np.log10(p) if p > 0 else 10 for p in p_values]
        
        colors = ['green' if p < 0.001 else 'yellow' if p < 0.05 else 'red' 
                 for p in p_values]
        
        bars = ax3.bar(tests, neg_log_p, color=colors, alpha=0.7)
        
        # Add significance thresholds
        ax3.axhline(-np.log10(0.05), color='orange', linestyle='--', 
                   label='p = 0.05', linewidth=1.5)
        ax3.axhline(-np.log10(0.001), color='red', linestyle='--', 
                   label='p = 0.001', linewidth=1.5)
        
        # Add p-value labels
        for bar, p, stat in zip(bars, p_values, statistics):
            p_text = f'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    p_text, ha='center', fontsize=9)
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                    f'{stat:.2f}', ha='center', va='center', fontsize=10,
                    color='white', fontweight='bold')
        
        ax3.set_ylabel('-log₁₀(p-value)', fontsize=11)
        ax3.set_title('Statistical Tests', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Panel 4: Residual distributions (if provided)
        if residuals_with is not None and residuals_without is not None:
            ax4 = fig.add_subplot(gs[1, :])
            
            # Sample for visualization if large
            n = len(residuals_with)
            if n > 10000:
                idx = np.random.choice(n, 10000, replace=False)
                res_with = residuals_with[idx]
                res_without = residuals_without[idx]
            else:
                res_with = residuals_with
                res_without = residuals_without
            
            # Create violin plots
            parts1 = ax4.violinplot([res_without], positions=[0], widths=0.7,
                                   showmeans=True, showmedians=True)
            parts2 = ax4.violinplot([res_with], positions=[1], widths=0.7,
                                   showmeans=True, showmedians=True)
            
            # Color the violin plots
            for pc in parts1['bodies']:
                pc.set_facecolor('lightcoral')
                pc.set_alpha(0.7)
            for pc in parts2['bodies']:
                pc.set_facecolor('lightgreen')
                pc.set_alpha(0.7)
            
            # Add labels and statistics
            ax4.set_xticks([0, 1])
            ax4.set_xticklabels(['Without biodiversity', 'With biodiversity'])
            ax4.set_ylabel('Residuals', fontsize=11)
            ax4.set_title('Residual Distributions', fontsize=12, fontweight='bold')
            
            # Add variance reduction annotation
            var_without = np.var(res_without)
            var_with = np.var(res_with)
            var_reduction = (var_without - var_with) / var_without * 100
            
            ax4.text(0.5, ax4.get_ylim()[1] * 0.9,
                    f'Variance reduction: {var_reduction:.1f}%',
                    ha='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
            
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            # Panel 4: Summary text
            ax4 = fig.add_subplot(gs[1, :])
            
            classification = comparison_results['classification']
            interpretation = comparison_results['interpretation']
            
            # Color based on classification
            if 'substantial' in classification:
                facecolor = 'lightgreen'
            elif 'moderate' in classification:
                facecolor = 'yellow'
            elif 'modest' in classification:
                facecolor = 'lightyellow'
            else:
                facecolor = 'lightcoral'
            
            summary_text = f"""Model Comparison Summary
            
Classification: {classification.replace('_', ' ').title()}

Interpretation:
{interpretation}

Key Statistics:
• F-statistic: {comparison_results['f_statistic']:.2f} (p = {comparison_results['f_p_value']:.4f})
• Pseudo-R² improvement: {comparison_results['r2_improvement']:.3f}
• RMSE reduction: {comparison_results['rmse_reduction']:.1%}
• MAE reduction: {comparison_results['mae_reduction']:.1%}

Conclusion: {'Strong' if 'substantial' in classification else 'Moderate' if 'moderate' in classification else 'Weak' if 'modest' in classification else 'No'} support for H1
"""
            
            ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=1', facecolor=facecolor, alpha=0.8))
            ax4.axis('off')
        
        # Overall title
        decision = 'Biodiversity Improves Model' if comparison_results['f_p_value'] < 0.05 else 'No Significant Improvement'
        fig.suptitle(f'DML Model Comparison: {decision}',
                    fontsize=14, fontweight='bold', y=0.98)
        
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
\caption{Double Machine Learning Model Comparison}
\label{tab:dml_comparison}
\begin{tabular}{lc}
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
\item Note: Model comparison using residual-based metrics from DML.
\item RSS = Residual Sum of Squares, RMSE = Root Mean Squared Error, MAE = Mean Absolute Error.
\item F-test and likelihood ratio test assess significance of biodiversity predictors.
\end{tablenotes}
\end{table}"""
        
        return latex