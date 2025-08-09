"""Statistical tests for model comparison and evaluation.

Implements LRT and AIC comparisons as specified in regression_analysis_decisions.md.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, Optional
import logging
from sklearn.metrics import r2_score, mean_squared_error

logger = logging.getLogger(__name__)


class LikelihoodRatioTest:
    """Perform likelihood ratio tests for nested linear models."""
    
    @staticmethod
    def calculate_log_likelihood(y_true: np.ndarray, y_pred: np.ndarray, 
                                n_params: int) -> float:
        """
        Calculate log-likelihood for linear regression model.
        
        Assumes normally distributed errors.
        """
        n = len(y_true)
        residuals = y_true - y_pred
        rss = np.sum(residuals ** 2)
        
        # Maximum likelihood estimate of variance
        sigma2 = rss / n
        
        # Log-likelihood
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        
        return log_likelihood
    
    @staticmethod
    def test(model1_results: Dict[str, Any], 
             model2_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform likelihood ratio test between two nested models.
        
        Args:
            model1_results: Dict with 'y_true', 'y_pred', 'n_params'
            model2_results: Dict with 'y_true', 'y_pred', 'n_params' (fuller model)
            
        Returns:
            Dict with test statistics and p-value
        """
        # Validate inputs
        if not np.array_equal(model1_results['y_true'], model2_results['y_true']):
            raise ValueError("Models must be fitted on same target data")
        
        if model1_results['n_params'] >= model2_results['n_params']:
            raise ValueError("Model 2 must have more parameters than Model 1")
        
        # Calculate log-likelihoods
        ll1 = LikelihoodRatioTest.calculate_log_likelihood(
            model1_results['y_true'],
            model1_results['y_pred'],
            model1_results['n_params']
        )
        
        ll2 = LikelihoodRatioTest.calculate_log_likelihood(
            model2_results['y_true'],
            model2_results['y_pred'],
            model2_results['n_params']
        )
        
        # LR statistic
        lr_statistic = 2 * (ll2 - ll1)
        
        # Degrees of freedom
        df = model2_results['n_params'] - model1_results['n_params']
        
        # P-value from chi-square distribution
        p_value = 1 - stats.chi2.cdf(lr_statistic, df)
        
        # Calculate R-squared for context
        r2_model1 = r2_score(model1_results['y_true'], model1_results['y_pred'])
        r2_model2 = r2_score(model2_results['y_true'], model2_results['y_pred'])
        
        return {
            'lr_statistic': lr_statistic,
            'df': df,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'log_likelihood_model1': ll1,
            'log_likelihood_model2': ll2,
            'r2_model1': r2_model1,
            'r2_model2': r2_model2,
            'r2_improvement': r2_model2 - r2_model1
        }


class NestedModelComparison:
    """Compare a sequence of nested models."""
    
    def __init__(self):
        self.comparisons = []
        
    def compare_sequential(self, model_results: list) -> list:
        """
        Compare models sequentially (1 vs 2, 2 vs 3, etc.).
        
        Args:
            model_results: List of dicts with model results
            
        Returns:
            List of comparison results
        """
        comparisons = []
        
        for i in range(len(model_results) - 1):
            comparison = LikelihoodRatioTest.test(
                model_results[i],
                model_results[i + 1]
            )
            comparison['models'] = f"Model {i+1} vs Model {i+2}"
            comparisons.append(comparison)
            
        self.comparisons = comparisons
        return comparisons
    
    def summary_table(self) -> pd.DataFrame:
        """Create summary table of all comparisons."""
        if not self.comparisons:
            return pd.DataFrame()
        
        summary_data = []
        for comp in self.comparisons:
            summary_data.append({
                'Comparison': comp['models'],
                'LR Statistic': f"{comp['lr_statistic']:.3f}",
                'DF': comp['df'],
                'P-value': f"{comp['p_value']:.4f}",
                'Significant': '***' if comp['p_value'] < 0.001 else
                              '**' if comp['p_value'] < 0.01 else
                              '*' if comp['p_value'] < 0.05 else 'ns',
                'R² Model 1': f"{comp['r2_model1']:.3f}",
                'R² Model 2': f"{comp['r2_model2']:.3f}",
                'R² Improvement': f"{comp['r2_improvement']:.3f}"
            })
        
        return pd.DataFrame(summary_data)


class ModelDiagnostics:
    """Additional diagnostic tests for model evaluation."""
    
    @staticmethod
    def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute residual diagnostics."""
        residuals = y_true - y_pred
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        # Homoscedasticity test (Breusch-Pagan)
        # Simplified version - regress squared residuals on predicted values
        squared_resid = residuals ** 2
        bp_slope, bp_intercept, bp_r, bp_p, bp_se = stats.linregress(y_pred, squared_resid)
        
        return {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'residuals_normal': shapiro_p > 0.05,
            'bp_statistic': bp_r ** 2 * len(y_true),
            'bp_p_value': bp_p,
            'homoscedastic': bp_p > 0.05
        }
    
    @staticmethod
    def calculate_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, 
                         n_params: int) -> Dict[str, float]:
        """Calculate AIC and BIC for model selection."""
        n = len(y_true)
        rss = np.sum((y_true - y_pred) ** 2)
        
        # Log-likelihood
        ll = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1)
        
        # AIC = 2k - 2ln(L)
        aic = 2 * n_params - 2 * ll
        
        # BIC = k*ln(n) - 2ln(L)
        bic = n_params * np.log(n) - 2 * ll
        
        return {
            'aic': aic,
            'bic': bic,
            'log_likelihood': ll
        }


class GLMComparison:
    """Model comparison methods specifically for GLMs (Negative Binomial/Poisson).
    
    Implements the comparison methods from regression_analysis_decisions.md.
    """
    
    @staticmethod
    def likelihood_ratio_test_glm(model_null, model_full) -> Dict[str, Any]:
        """Perform LRT for GLM models (equivalent to anova(..., test='Chisq') in R).
        
        Args:
            model_null: Null GLM model (H2O or statsmodels)
            model_full: Full GLM model with additional predictors
            
        Returns:
            Dictionary with LRT results
        """
        # Get deviances or log-likelihoods
        if hasattr(model_null, 'get_log_likelihood'):
            ll_null = model_null.get_log_likelihood()
            ll_full = model_full.get_log_likelihood()
        elif hasattr(model_null, 'deviance'):
            # For H2O models, use deviance
            ll_null = -model_null.deviance() / 2
            ll_full = -model_full.deviance() / 2
        else:
            raise ValueError("Models must have get_log_likelihood or deviance method")
        
        # LR statistic
        lr_statistic = 2 * (ll_full - ll_null)
        
        # Get degrees of freedom
        if hasattr(model_null, 'get_num_parameters'):
            df_null = model_null.get_num_parameters()
            df_full = model_full.get_num_parameters()
        elif hasattr(model_null, 'dof'):
            df_null = model_null.dof()
            df_full = model_full.dof()
        else:
            # Fallback: count coefficients
            df_null = len(model_null.coef())
            df_full = len(model_full.coef())
        
        df_diff = df_full - df_null
        
        # P-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(lr_statistic, df_diff)
        
        logger.info(f"GLM LRT: χ² = {lr_statistic:.4f}, df = {df_diff}, p = {p_value:.6f}")
        
        return {
            'lr_statistic': lr_statistic,
            'df': df_diff,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': (
                "Biodiversity predictors significantly improve model" if p_value < 0.05
                else "No significant improvement from biodiversity predictors"
            )
        }
    
    @staticmethod
    def aic_comparison_glm(model_null, model_full) -> Dict[str, Any]:
        """Compare GLMs using AIC.
        
        Decision rules from regression_analysis_decisions.md:
        - ΔAIC > 2: Substantial support for full model
        - ΔAIC > 4: Strong support
        - ΔAIC > 10: Essentially no support for null model
        
        Args:
            model_null: Null GLM model
            model_full: Full GLM model
            
        Returns:
            Dictionary with AIC comparison
        """
        # Get AIC values
        if hasattr(model_null, 'get_aic'):
            aic_null = model_null.get_aic()
            aic_full = model_full.get_aic()
        elif hasattr(model_null, 'aic'):
            aic_null = model_null.aic()
            aic_full = model_full.aic()
        else:
            raise ValueError("Models must have get_aic or aic method")
        
        # Calculate delta AIC
        delta_aic = aic_null - aic_full
        
        # Interpret based on thresholds from decisions document
        if delta_aic > 10:
            interpretation = "Essentially no support for null model (H1 strongly supported)"
        elif delta_aic > 4:
            interpretation = "Strong support for full model (H1 supported)"
        elif delta_aic > 2:
            interpretation = "Substantial support for full model (H1 weakly supported)"
        elif delta_aic > -2:
            interpretation = "Models roughly equivalent"
        else:
            interpretation = "Support for null model (H0 supported)"
        
        logger.info(f"AIC: Null={aic_null:.2f}, Full={aic_full:.2f}, ΔAIC={delta_aic:.2f}")
        
        return {
            'aic_null': aic_null,
            'aic_full': aic_full,
            'delta_aic': delta_aic,
            'preferred_model': 'full' if delta_aic > 0 else 'null',
            'h1_supported': delta_aic > 2,
            'interpretation': interpretation
        }
    
    @staticmethod
    def comprehensive_glm_comparison(model_null, model_full) -> Dict[str, Any]:
        """Perform both LRT and AIC comparison for GLMs.
        
        This is the main method to use for hypothesis testing.
        
        Args:
            model_null: Null model (climate only)
            model_full: Full model (climate + biodiversity)
            
        Returns:
            Dictionary with all comparison results
        """
        results = {}
        
        # Perform LRT
        try:
            results['lrt'] = GLMComparison.likelihood_ratio_test_glm(model_null, model_full)
        except Exception as e:
            logger.warning(f"LRT failed: {e}")
            results['lrt'] = {'error': str(e)}
        
        # Perform AIC comparison
        try:
            results['aic'] = GLMComparison.aic_comparison_glm(model_null, model_full)
        except Exception as e:
            logger.warning(f"AIC comparison failed: {e}")
            results['aic'] = {'error': str(e)}
        
        # Overall H1 support determination
        lrt_significant = results.get('lrt', {}).get('significant', False)
        aic_support = results.get('aic', {}).get('h1_supported', False)
        
        results['h1_supported'] = lrt_significant and aic_support
        results['summary'] = (
            "H1 supported: Biodiversity begets biodiversity" if results['h1_supported']
            else "H0 supported: No biodiversity effect beyond climate"
        )
        
        return results