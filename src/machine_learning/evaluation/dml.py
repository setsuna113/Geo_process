"""
Double Machine Learning (DML) parameter calculations.

Provides functions to calculate statistical parameters needed
for hypothesis testing from DML model outputs.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class DMLParameterCalculator:
    """Calculate statistical parameters from DML outputs.
    
    This class computes standard errors, t-statistics, p-values,
    and other parameters needed for hypothesis testing from
    DML residuals and coefficient estimates.
    """
    
    @staticmethod
    def calculate_standard_errors(
        residuals_Y: np.ndarray,
        residuals_A: np.ndarray,
        theta: float,
        robust: bool = True
    ) -> float:
        """Calculate standard error for DML estimator.
        
        Uses either homoscedastic or heteroscedasticity-robust
        standard errors.
        
        Args:
            residuals_Y: Outcome residuals from nuisance model
            residuals_A: Treatment residuals from nuisance model
            theta: Estimated causal effect
            robust: Whether to use robust standard errors
            
        Returns:
            Standard error of theta
        """
        n = len(residuals_Y)
        
        if len(residuals_A) != n:
            raise ValueError("Residuals must have same length")
        
        # Residuals from final regression
        final_residuals = residuals_Y - theta * residuals_A
        
        if robust:
            # Heteroscedasticity-robust (HC1) standard errors
            # Variance = (1/n) * sum(e_i^2 * A_i^2) / [sum(A_i^2)]^2
            numerator = np.sum((final_residuals ** 2) * (residuals_A ** 2))
            denominator = np.sum(residuals_A ** 2) ** 2
            
            # HC1 correction factor
            correction = n / (n - 1)
            variance = correction * numerator / denominator
        else:
            # Homoscedastic standard errors
            sigma_squared = np.sum(final_residuals ** 2) / (n - 1)
            variance = sigma_squared / np.sum(residuals_A ** 2)
        
        se = np.sqrt(variance)
        
        logger.debug(f"Standard error calculated: {se:.6f} (robust={robust})")
        
        return se
    
    @staticmethod
    def calculate_t_statistic(theta: float, se_theta: float) -> float:
        """Calculate t-statistic for hypothesis testing.
        
        Args:
            theta: Causal effect estimate
            se_theta: Standard error of theta
            
        Returns:
            T-statistic
        """
        if se_theta <= 0:
            raise ValueError("Standard error must be positive")
        
        t_stat = theta / se_theta
        return t_stat
    
    @staticmethod
    def calculate_p_value(
        theta: float,
        se_theta: float,
        alternative: str = 'two-sided',
        df: Optional[int] = None
    ) -> float:
        """Calculate p-value for hypothesis test.
        
        Args:
            theta: Causal effect estimate
            se_theta: Standard error of theta
            alternative: 'two-sided', 'greater', or 'less'
            df: Degrees of freedom (None for asymptotic normal)
            
        Returns:
            P-value
        """
        t_stat = theta / se_theta
        
        # Use normal or t-distribution
        if df is None:
            # Asymptotic normal distribution
            if alternative == 'two-sided':
                p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            elif alternative == 'greater':
                p_value = 1 - stats.norm.cdf(t_stat)
            elif alternative == 'less':
                p_value = stats.norm.cdf(t_stat)
            else:
                raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        else:
            # T-distribution with specified df
            if alternative == 'two-sided':
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            elif alternative == 'greater':
                p_value = 1 - stats.t.cdf(t_stat, df)
            elif alternative == 'less':
                p_value = stats.t.cdf(t_stat, df)
            else:
                raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
        
        return p_value
    
    @staticmethod
    def calculate_confidence_interval(
        theta: float,
        se_theta: float,
        confidence_level: float = 0.95,
        df: Optional[int] = None
    ) -> Tuple[float, float]:
        """Calculate confidence interval for causal effect.
        
        Args:
            theta: Causal effect estimate
            se_theta: Standard error of theta
            confidence_level: Confidence level (e.g., 0.95)
            df: Degrees of freedom (None for asymptotic normal)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - confidence_level
        
        if df is None:
            # Asymptotic normal
            z_critical = stats.norm.ppf(1 - alpha/2)
        else:
            # T-distribution
            z_critical = stats.t.ppf(1 - alpha/2, df)
        
        ci_lower = theta - z_critical * se_theta
        ci_upper = theta + z_critical * se_theta
        
        return ci_lower, ci_upper
    
    @staticmethod
    def calculate_r_squared(
        Y: np.ndarray,
        Y_pred: np.ndarray,
        adjusted: bool = False,
        n_features: Optional[int] = None
    ) -> float:
        """Calculate R-squared for nuisance models.
        
        Args:
            Y: True values
            Y_pred: Predicted values
            adjusted: Whether to calculate adjusted R-squared
            n_features: Number of features (for adjusted R²)
            
        Returns:
            R-squared value
        """
        n = len(Y)
        
        # Total sum of squares
        ss_total = np.sum((Y - np.mean(Y)) ** 2)
        
        # Residual sum of squares
        ss_residual = np.sum((Y - Y_pred) ** 2)
        
        # R-squared
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        if adjusted and n_features is not None:
            # Adjusted R-squared
            if n - n_features - 1 > 0:
                r2_adj = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
                return r2_adj
        
        return r2
    
    @staticmethod
    def calculate_residual_correlation(
        residuals_Y: np.ndarray,
        residuals_A1: np.ndarray,
        residuals_A2: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate correlations between residuals.
        
        Useful for understanding biodiversity coupling after
        controlling for environmental factors.
        
        Args:
            residuals_Y: Fungi residuals
            residuals_A1: Plants residuals
            residuals_A2: Vertebrates residuals (optional)
            
        Returns:
            Dictionary of correlation coefficients
        """
        correlations = {}
        
        # Correlation between fungi and plants
        corr_Y_A1 = np.corrcoef(residuals_Y, residuals_A1)[0, 1]
        correlations['fungi_plants'] = float(corr_Y_A1)
        
        if residuals_A2 is not None:
            # Correlation between fungi and vertebrates
            corr_Y_A2 = np.corrcoef(residuals_Y, residuals_A2)[0, 1]
            correlations['fungi_vertebrates'] = float(corr_Y_A2)
            
            # Correlation between plants and vertebrates
            corr_A1_A2 = np.corrcoef(residuals_A1, residuals_A2)[0, 1]
            correlations['plants_vertebrates'] = float(corr_A1_A2)
        
        return correlations
    
    @staticmethod
    def calculate_effective_sample_size(
        residuals_A: np.ndarray,
        threshold: float = 0.01
    ) -> int:
        """Calculate effective sample size for treatment variation.
        
        Estimates how much independent variation remains in
        treatment after partialling out confounders.
        
        Args:
            residuals_A: Treatment residuals
            threshold: Minimum variance threshold
            
        Returns:
            Effective sample size
        """
        # Variance of residuals
        var_residuals = np.var(residuals_A)
        
        # Original variance estimate (using MAD as robust estimator)
        mad = np.median(np.abs(residuals_A - np.median(residuals_A)))
        original_var_estimate = (1.4826 * mad) ** 2
        
        # Proportion of variance retained
        if original_var_estimate > 0:
            prop_variance = var_residuals / original_var_estimate
        else:
            prop_variance = 0
        
        # Effective sample size
        n = len(residuals_A)
        n_eff = int(n * min(prop_variance, 1.0))
        
        return n_eff
    
    @staticmethod
    def compute_all_parameters(
        residuals_Y: np.ndarray,
        residuals_A1: np.ndarray,
        residuals_A2: Optional[np.ndarray] = None,
        theta1: Optional[float] = None,
        theta2: Optional[float] = None,
        robust_se: bool = True
    ) -> Dict[str, Any]:
        """Compute all DML parameters for hypothesis testing.
        
        Args:
            residuals_Y: Outcome (fungi) residuals
            residuals_A1: First treatment (plants) residuals
            residuals_A2: Second treatment (vertebrates) residuals
            theta1: Estimated effect of first treatment
            theta2: Estimated effect of second treatment
            robust_se: Whether to use robust standard errors
            
        Returns:
            Dictionary with all computed parameters
        """
        results = {}
        
        # If thetas not provided, estimate from residuals
        if theta1 is None:
            # Simple OLS on residuals
            theta1 = np.dot(residuals_A1, residuals_Y) / np.dot(residuals_A1, residuals_A1)
        
        # Standard error for theta1
        se1 = DMLParameterCalculator.calculate_standard_errors(
            residuals_Y, residuals_A1, theta1, robust_se
        )
        
        # T-statistic and p-value for theta1
        t1 = DMLParameterCalculator.calculate_t_statistic(theta1, se1)
        p1 = DMLParameterCalculator.calculate_p_value(theta1, se1)
        ci1_lower, ci1_upper = DMLParameterCalculator.calculate_confidence_interval(theta1, se1)
        
        results['theta1'] = {
            'estimate': float(theta1),
            'standard_error': float(se1),
            't_statistic': float(t1),
            'p_value': float(p1),
            'ci_lower': float(ci1_lower),
            'ci_upper': float(ci1_upper)
        }
        
        # If second treatment provided
        if residuals_A2 is not None and theta2 is not None:
            se2 = DMLParameterCalculator.calculate_standard_errors(
                residuals_Y, residuals_A2, theta2, robust_se
            )
            
            t2 = DMLParameterCalculator.calculate_t_statistic(theta2, se2)
            p2 = DMLParameterCalculator.calculate_p_value(theta2, se2)
            ci2_lower, ci2_upper = DMLParameterCalculator.calculate_confidence_interval(theta2, se2)
            
            results['theta2'] = {
                'estimate': float(theta2),
                'standard_error': float(se2),
                't_statistic': float(t2),
                'p_value': float(p2),
                'ci_lower': float(ci2_lower),
                'ci_upper': float(ci2_upper)
            }
        
        # Residual correlations
        correlations = DMLParameterCalculator.calculate_residual_correlation(
            residuals_Y, residuals_A1, residuals_A2
        )
        results['residual_correlations'] = correlations
        
        # Effective sample sizes
        results['effective_sample_sizes'] = {
            'treatment1': DMLParameterCalculator.calculate_effective_sample_size(residuals_A1)
        }
        if residuals_A2 is not None:
            results['effective_sample_sizes']['treatment2'] = (
                DMLParameterCalculator.calculate_effective_sample_size(residuals_A2)
            )
        
        logger.info(f"DML parameters computed: θ1={theta1:.4f} (SE={se1:.4f}, p={p1:.4f})")
        
        return results