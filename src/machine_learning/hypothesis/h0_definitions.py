"""H0 (Null Hypothesis) Definitions for Hypothesis Testing.

This module defines different null hypothesis formulations.
AIC definitions moved to model_selection/aic_definitions.py where they belong.
"""

from typing import Dict, Any


class H0Definition:
    """Null hypothesis definitions for statistical tests."""
    
    @staticmethod
    def traditional() -> Dict[str, Any]:
        """Traditional null hypothesis: θ = 0
        
        Returns:
            Dictionary with H0 specification
        """
        return {
            'type': 'traditional',
            'null_value': 0.0,
            'description': 'No effect (θ = 0)',
            'interpretation': 'Testing if effect is different from zero'
        }
    
    @staticmethod
    def sesoi(threshold: float, direction: str = 'greater') -> Dict[str, Any]:
        """SESOI (Smallest Effect Size of Interest) null hypothesis.
        
        Args:
            threshold: The smallest effect size considered meaningful
            direction: 'greater', 'less', or 'two-sided'
        
        Returns:
            Dictionary with H0 specification
        """
        if direction == 'greater':
            return {
                'type': 'sesoi',
                'null_value': threshold,
                'direction': direction,
                'description': f'Effect ≤ {threshold}',
                'interpretation': f'Testing if effect exceeds meaningful threshold ({threshold})'
            }
        elif direction == 'less':
            return {
                'type': 'sesoi',
                'null_value': -threshold,
                'direction': direction,
                'description': f'Effect ≥ {-threshold}',
                'interpretation': f'Testing if effect is below meaningful threshold ({-threshold})'
            }
        else:  # two-sided
            return {
                'type': 'sesoi',
                'null_value': threshold,
                'direction': direction,
                'description': f'|Effect| ≤ {threshold}',
                'interpretation': f'Testing if absolute effect exceeds threshold ({threshold})'
            }
    
    @staticmethod
    def equivalence_bounds(lower: float, upper: float) -> Dict[str, Any]:
        """Equivalence bounds for TOST testing.
        
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
            'interpretation': 'Testing if effect is practically equivalent to zero'
        }
    
    @staticmethod
    def non_inferiority(margin: float) -> Dict[str, Any]:
        """Non-inferiority hypothesis: θ ≥ -margin
        
        Args:
            margin: Non-inferiority margin
            
        Returns:
            Dictionary with H0 specification
        """
        return {
            'type': 'non_inferiority',
            'null_value': -margin,
            'description': f'Effect < {-margin}',
            'interpretation': f'Testing if effect is not inferior by more than {margin}'
        }
    
    @staticmethod
    def superiority(margin: float = 0) -> Dict[str, Any]:
        """Superiority hypothesis: θ > margin
        
        Args:
            margin: Superiority margin (0 for simple superiority)
            
        Returns:
            Dictionary with H0 specification
        """
        return {
            'type': 'superiority',
            'null_value': margin,
            'description': f'Effect ≤ {margin}',
            'interpretation': f'Testing if effect is superior{"" if margin == 0 else f" by more than {margin}"}'
        }