"""
Model Comparison for DML (Option 4).

Compares models with and without biodiversity predictors
using residual-based metrics and pseudo R-squared.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


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