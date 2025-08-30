from src.abstractions import get_logger
"""
Utilities for handling scale transformations in hypothesis testing.

Ensures proper handling of log-scale coefficients and their interpretation
on the response scale for ecological models.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = get_logger(__name__)


class ScaleTransformation:
    """Handle transformations between link and response scales."""
    
    @staticmethod
    def log_to_response_scale(
        coefficient: float,
        standard_error: float,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Transform log-scale coefficient to response scale (multiplicative effect).
        
        For log-link GLMs:
        - Log scale: E[log(Y)] = β₀ + β₁X
        - Response scale: E[Y] = exp(β₀) * exp(β₁X)
        - Interpretation: exp(β₁) is the multiplicative effect
        
        Args:
            coefficient: Coefficient on log scale
            standard_error: Standard error on log scale
            confidence_level: Confidence level for CI
            
        Returns:
            Dictionary with response-scale interpretation
        """
        from scipy import stats
        
        # Critical value for CI
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Log-scale CI
        log_ci_lower = coefficient - z_critical * standard_error
        log_ci_upper = coefficient + z_critical * standard_error
        
        # Transform to response scale (multiplicative effects)
        multiplicative_effect = np.exp(coefficient)
        mult_ci_lower = np.exp(log_ci_lower)
        mult_ci_upper = np.exp(log_ci_upper)
        
        # Percent change interpretation
        percent_change = (multiplicative_effect - 1) * 100
        percent_ci_lower = (mult_ci_lower - 1) * 100
        percent_ci_upper = (mult_ci_upper - 1) * 100
        
        # Delta method for SE on response scale
        # Var[exp(β)] ≈ exp(2β) * Var[β]
        response_se = multiplicative_effect * standard_error
        
        return {
            'log_scale': {
                'coefficient': coefficient,
                'se': standard_error,
                'ci_lower': log_ci_lower,
                'ci_upper': log_ci_upper
            },
            'response_scale': {
                'multiplicative_effect': multiplicative_effect,
                'se_approx': response_se,
                'ci_lower': mult_ci_lower,
                'ci_upper': mult_ci_upper
            },
            'percent_change': {
                'estimate': percent_change,
                'ci_lower': percent_ci_lower,
                'ci_upper': percent_ci_upper
            },
            'interpretation': ScaleTransformation._interpret_effect(
                percent_change, percent_ci_lower, percent_ci_upper
            )
        }
    
    @staticmethod
    def _interpret_effect(
        percent_change: float,
        ci_lower: float,
        ci_upper: float
    ) -> str:
        """Generate interpretation of percent change effect."""
        
        if ci_lower > 0:
            direction = "increase"
            magnitude = percent_change
        elif ci_upper < 0:
            direction = "decrease"
            magnitude = abs(percent_change)
        else:
            return f"No significant effect (CI includes 0: [{ci_lower:.1f}%, {ci_upper:.1f}%])"
        
        # Categorize magnitude
        if abs(magnitude) < 5:
            size = "negligible"
        elif abs(magnitude) < 10:
            size = "small"
        elif abs(magnitude) < 25:
            size = "moderate"
        elif abs(magnitude) < 50:
            size = "large"
        else:
            size = "very large"
        
        return (
            f"{size.capitalize()} {direction} of {abs(percent_change):.1f}% "
            f"(95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%])"
        )
    
    @staticmethod
    def adjust_sesoi_for_scale(
        sesoi_log_scale: float,
        link: str = 'log'
    ) -> Dict[str, float]:
        """
        Adjust SESOI threshold between scales.
        
        Args:
            sesoi_log_scale: SESOI on log scale
            link: Link function used
            
        Returns:
            SESOI thresholds on different scales
        """
        if link == 'log':
            # Log scale SESOI -> Response scale
            # SESOI of 0.1 on log scale = ~10.5% change on response scale
            sesoi_multiplicative = np.exp(sesoi_log_scale)
            sesoi_percent = (sesoi_multiplicative - 1) * 100
            
            return {
                'log_scale': sesoi_log_scale,
                'multiplicative': sesoi_multiplicative,
                'percent_change': sesoi_percent,
                'interpretation': (
                    f"SESOI: {sesoi_log_scale:.2f} on log scale = "
                    f"{sesoi_percent:.1f}% change on response scale"
                )
            }
        else:
            # For identity link, no transformation needed
            return {
                'original_scale': sesoi_log_scale,
                'interpretation': f"SESOI: {sesoi_log_scale:.2f} on original scale"
            }
    
    @staticmethod
    def compute_standardized_effect(
        coefficient: float,
        predictor_sd: float,
        response_sd: float,
        link: str = 'log'
    ) -> Dict[str, float]:
        """
        Compute standardized effect size for comparison across variables.
        
        For GLMs, this is analogous to correlation coefficient.
        
        Args:
            coefficient: Unstandardized coefficient
            predictor_sd: Standard deviation of predictor
            response_sd: Standard deviation of response (on link scale)
            link: Link function
            
        Returns:
            Standardized effect sizes
        """
        # Basic standardization
        standardized_coef = coefficient * predictor_sd / response_sd
        
        # For log link, also compute elasticity
        if link == 'log':
            # Elasticity = percent change in Y for 1% change in X
            # For log-log model, elasticity = coefficient
            # For log-linear model, need to adjust
            elasticity = coefficient  # Simplified - adjust based on model form
        else:
            elasticity = None
        
        return {
            'standardized_coefficient': standardized_coef,
            'elasticity': elasticity,
            'interpretation': ScaleTransformation._interpret_standardized(standardized_coef)
        }
    
    @staticmethod
    def _interpret_standardized(std_coef: float) -> str:
        """Interpret standardized coefficient magnitude."""
        abs_coef = abs(std_coef)
        
        if abs_coef < 0.1:
            return "Negligible standardized effect"
        elif abs_coef < 0.3:
            return "Small standardized effect"
        elif abs_coef < 0.5:
            return "Medium standardized effect"
        else:
            return "Large standardized effect"
    
    @staticmethod
    def report_all_scales(
        coefficient: float,
        standard_error: float,
        predictor_name: str,
        response_name: str,
        link: str = 'log',
        sesoi: Optional[float] = None,
        predictor_sd: Optional[float] = None,
        response_sd: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report with all scale transformations.
        
        Args:
            coefficient: Coefficient estimate
            standard_error: Standard error
            predictor_name: Name of predictor variable
            response_name: Name of response variable
            link: Link function
            sesoi: SESOI threshold (optional)
            predictor_sd: SD of predictor (for standardization)
            response_sd: SD of response (for standardization)
            
        Returns:
            Comprehensive report with all transformations
        """
        report = {
            'variable': predictor_name,
            'response': response_name,
            'link_function': link
        }
        
        # Add scale transformations for log link
        if link == 'log':
            report['scales'] = ScaleTransformation.log_to_response_scale(
                coefficient, standard_error
            )
        else:
            report['scales'] = {
                'original_scale': {
                    'coefficient': coefficient,
                    'se': standard_error
                }
            }
        
        # Add SESOI interpretation if provided
        if sesoi is not None:
            report['sesoi'] = ScaleTransformation.adjust_sesoi_for_scale(sesoi, link)
            
            # Check if effect exceeds SESOI
            if link == 'log':
                exceeds_sesoi = abs(coefficient) > sesoi
                report['sesoi']['exceeds_threshold'] = exceeds_sesoi
            else:
                exceeds_sesoi = abs(coefficient) > sesoi
                report['sesoi']['exceeds_threshold'] = exceeds_sesoi
        
        # Add standardized effect if SDs provided
        if predictor_sd is not None and response_sd is not None:
            report['standardized'] = ScaleTransformation.compute_standardized_effect(
                coefficient, predictor_sd, response_sd, link
            )
        
        # Create summary interpretation
        if link == 'log':
            pct_change = report['scales']['percent_change']['estimate']
            ci_lower = report['scales']['percent_change']['ci_lower']
            ci_upper = report['scales']['percent_change']['ci_upper']
            
            report['summary'] = (
                f"{predictor_name} effect on {response_name}: "
                f"{pct_change:+.1f}% change (95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%])"
            )
        else:
            report['summary'] = (
                f"{predictor_name} effect on {response_name}: "
                f"{coefficient:.3f} ± {standard_error:.3f} SE"
            )
        
        return report