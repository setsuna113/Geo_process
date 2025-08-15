"""
H0 (Null Hypothesis) Definitions for Hypothesis Testing.

This module defines different null hypothesis formulations that can be
used with any hypothesis test (LRT, Wald, Score, etc.).
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


class H0Definition:
    """Base class for null hypothesis definitions."""
    
    @staticmethod
    def traditional() -> Dict[str, Any]:
        """
        Traditional null hypothesis: θ = 0
        
        Returns:
            Dictionary with H0 specification
        """
        return {
            'type': 'traditional',
            'null_value': 0.0,
            'description': 'No effect (θ = 0)',
            'test_statistic_adjustment': lambda theta, se: theta / se,
            'interpretation': 'Testing if effect is different from zero'
        }
    
    @staticmethod
    def sesoi(threshold: float, direction: str = 'greater') -> Dict[str, Any]:
        """
        SESOI (Smallest Effect Size of Interest) null hypothesis.
        
        Args:
            threshold: The smallest effect size considered meaningful
            direction: 'greater' (H0: θ ≤ threshold), 
                      'less' (H0: θ ≥ -threshold),
                      'two-sided' (H0: |θ| ≤ threshold)
        
        Returns:
            Dictionary with H0 specification
        """
        if direction == 'greater':
            return {
                'type': 'sesoi',
                'null_value': threshold,
                'direction': direction,
                'description': f'Effect ≤ {threshold}',
                'test_statistic_adjustment': lambda theta, se: (theta - threshold) / se,
                'interpretation': f'Testing if effect exceeds meaningful threshold ({threshold})'
            }
        elif direction == 'less':
            return {
                'type': 'sesoi',
                'null_value': -threshold,
                'direction': direction,
                'description': f'Effect ≥ {-threshold}',
                'test_statistic_adjustment': lambda theta, se: (theta + threshold) / se,
                'interpretation': f'Testing if effect is below meaningful threshold ({-threshold})'
            }
        else:  # two-sided
            return {
                'type': 'sesoi',
                'null_value': threshold,
                'direction': direction,
                'description': f'|Effect| ≤ {threshold}',
                'test_statistic_adjustment': lambda theta, se: (abs(theta) - threshold) / se,
                'interpretation': f'Testing if absolute effect exceeds threshold ({threshold})'
            }
    
    @staticmethod
    def equivalence_bounds(lower: float, upper: float) -> Dict[str, Any]:
        """
        Equivalence bounds for TOST testing.
        
        Note: This returns bounds, not a single H0. The TOST wrapper
        will use these to construct two one-sided tests.
        
        Args:
            lower: Lower equivalence bound
            upper: Upper equivalence bound
            
        Returns:
            Dictionary with equivalence bounds specification
        """
        return {
            'type': 'equivalence',
            'lower_bound': lower,
            'upper_bound': upper,
            'description': f'Effect within [{lower}, {upper}]',
            'interpretation': f'Testing if effect is practically equivalent to zero'
        }
    
    @staticmethod
    def non_inferiority(margin: float) -> Dict[str, Any]:
        """
        Non-inferiority hypothesis: θ ≥ -margin
        
        Args:
            margin: Non-inferiority margin
            
        Returns:
            Dictionary with H0 specification
        """
        return {
            'type': 'non_inferiority',
            'null_value': -margin,
            'description': f'Effect < {-margin}',
            'test_statistic_adjustment': lambda theta, se: (theta + margin) / se,
            'interpretation': f'Testing if effect is not inferior by more than {margin}'
        }
    
    @staticmethod
    def superiority(margin: float = 0) -> Dict[str, Any]:
        """
        Superiority hypothesis: θ > margin
        
        Args:
            margin: Superiority margin (0 for simple superiority)
            
        Returns:
            Dictionary with H0 specification
        """
        return {
            'type': 'superiority',
            'null_value': margin,
            'description': f'Effect ≤ {margin}',
            'test_statistic_adjustment': lambda theta, se: (theta - margin) / se,
            'interpretation': f'Testing if effect is superior{"" if margin == 0 else f" by more than {margin}"}'
        }


class AICDefinition:
    """Definitions for AIC/BIC adjustments based on sample size."""
    
    @staticmethod
    def traditional(aic: float, bic: float, n_obs: int) -> Dict[str, float]:
        """
        Traditional AIC/BIC without adjustments.
        
        Args:
            aic: Akaike Information Criterion
            bic: Bayesian Information Criterion  
            n_obs: Number of observations
            
        Returns:
            Dictionary with AIC/BIC values
        """
        return {
            'aic': aic,
            'bic': bic,
            'aic_threshold': 2.0,  # Difference > 2 is meaningful
            'bic_threshold': 2.0,
            'interpretation': 'Standard AIC/BIC comparison'
        }
    
    @staticmethod
    def sample_adjusted(aic: float, bic: float, n_obs: int) -> Dict[str, float]:
        """
        Sample-size adjusted AIC/BIC for large datasets.
        
        With large samples, even tiny improvements show as "better" in AIC/BIC.
        This adjusts thresholds based on sample size.
        
        Args:
            aic: Akaike Information Criterion
            bic: Bayesian Information Criterion
            n_obs: Number of observations
            
        Returns:
            Dictionary with adjusted thresholds
        """
        # Adjust thresholds based on sample size
        # With millions of observations, require larger differences
        if n_obs < 1000:
            aic_threshold = 2.0
            bic_threshold = 2.0
        elif n_obs < 10000:
            aic_threshold = 10.0
            bic_threshold = 10.0
        elif n_obs < 100000:
            aic_threshold = 50.0
            bic_threshold = 50.0
        elif n_obs < 1000000:
            aic_threshold = 100.0
            bic_threshold = 100.0
        else:  # > 1 million
            # Scale logarithmically with sample size
            aic_threshold = 100 * np.log10(n_obs / 1000000 + 1)
            bic_threshold = 100 * np.log10(n_obs / 1000000 + 1)
        
        return {
            'aic': aic,
            'bic': bic,
            'aic_threshold': aic_threshold,
            'bic_threshold': bic_threshold,
            'n_obs': n_obs,
            'interpretation': f'Sample-adjusted thresholds (n={n_obs:,}): AIC>{aic_threshold:.1f}, BIC>{bic_threshold:.1f}'
        }
    
    @staticmethod
    def aicc_corrected(aic: float, n_obs: int, n_params: int) -> Dict[str, float]:
        """
        Corrected AIC (AICc) for small samples.
        
        AICc = AIC + 2k(k+1)/(n-k-1) where k is number of parameters.
        
        Args:
            aic: Standard AIC
            n_obs: Number of observations
            n_params: Number of parameters
            
        Returns:
            Dictionary with AICc
        """
        if n_obs - n_params - 1 > 0:
            correction = (2 * n_params * (n_params + 1)) / (n_obs - n_params - 1)
            aicc = aic + correction
        else:
            aicc = np.inf
            
        return {
            'aic': aic,
            'aicc': aicc,
            'correction': aicc - aic if aicc != np.inf else np.inf,
            'use_aicc': n_obs / n_params < 40,  # Rule of thumb
            'interpretation': f'AICc recommended when n/k < 40 (here: {n_obs}/{n_params} = {n_obs/n_params:.1f})'
        }