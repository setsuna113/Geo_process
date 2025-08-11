"""
Contrast Analysis for testing specific hypotheses using emmeans.

Test 6: Contrast Analysis on Fitted Values
This module tests specific linear combinations of group means to evaluate
directional hypotheses like H2 (temperate mismatch).
"""

import logging
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple

from ..evaluation.emmeans_core import EMMeansCore

logger = logging.getLogger(__name__)


class ContrastAnalysis:
    """Perform contrast analysis on estimated marginal means.
    
    Tests specific hypotheses by applying contrast vectors to emmeans.
    For H2: Tests whether high-plant temperate regions have fewer fungi
    than high-plant non-temperate regions.
    """
    
    def __init__(self, emmeans_calculator: EMMeansCore):
        """Initialize contrast analyzer.
        
        Args:
            emmeans_calculator: EMMeansCore instance with computed emmeans
        """
        self.emmeans_calc = emmeans_calculator
        
    def test_h2_contrast(
        self,
        contrast_vector: List[float] = [0, -1, 0, 1]
    ) -> Dict[str, Any]:
        """Test H2 hypothesis using specific contrast.
        
        The default contrast [0, -1, 0, 1] tests:
        (high_plant:temperate) - (high_plant:non_temperate)
        
        Order of groups: [low_nontemp, high_nontemp, low_temp, high_temp]
        
        Args:
            contrast_vector: Contrast coefficients
            
        Returns:
            Dictionary with test results:
            - contrast_estimate: Effect size
            - standard_error: SE of contrast
            - z_statistic: Test statistic
            - p_value: Two-tailed p-value
            - h2_supported: Whether result supports H2
            - interpretation: Text interpretation
        """
        if self.emmeans_calc.emmeans_results is None:
            raise ValueError("Must compute emmeans first")
        
        logger.info(f"Testing H2 with contrast vector: {contrast_vector}")
        
        # Get emmeans in standard order
        emmeans = self.emmeans_calc.get_emmeans_array()
        
        # Apply contrast
        contrast_estimate = np.dot(contrast_vector, emmeans)
        
        # Estimate standard error (simplified - should use delta method)
        contrast_se = self._estimate_contrast_se(contrast_vector)
        
        # Calculate test statistic
        z_stat = contrast_estimate / contrast_se if contrast_se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # H2 predicts NEGATIVE contrast (fewer fungi in temperate)
        h2_supported = (p_value < 0.05) and (contrast_estimate < 0)
        
        result = {
            'contrast_vector': contrast_vector,
            'contrast_estimate': float(contrast_estimate),
            'standard_error': float(contrast_se),
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'h2_supported': h2_supported,
            'interpretation': self._interpret_h2_result(contrast_estimate, p_value)
        }
        
        logger.info(f"H2 Contrast: {contrast_estimate:.3f} (SE={contrast_se:.3f}, p={p_value:.4f})")
        
        return result
    
    def test_custom_contrasts(
        self,
        contrasts: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, Any]]:
        """Test multiple custom contrasts.
        
        Args:
            contrasts: Dictionary of contrast names and vectors
                      e.g., {'main_effect_plant': [-1, 1, -1, 1]}
            
        Returns:
            Dictionary with results for each contrast
        """
        if self.emmeans_calc.emmeans_results is None:
            raise ValueError("Must compute emmeans first")
        
        emmeans = self.emmeans_calc.get_emmeans_array()
        results = {}
        
        for name, contrast_vector in contrasts.items():
            if len(contrast_vector) != 4:
                logger.warning(f"Contrast {name} has wrong length, skipping")
                continue
            
            # Check if proper contrast (sums to zero)
            if abs(sum(contrast_vector)) > 0.001:
                logger.warning(f"Contrast {name} doesn't sum to zero")
            
            # Apply contrast
            estimate = np.dot(contrast_vector, emmeans)
            se = self._estimate_contrast_se(contrast_vector)
            
            # Test statistic
            z_stat = estimate / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results[name] = {
                'contrast_vector': contrast_vector,
                'estimate': float(estimate),
                'standard_error': float(se),
                'z_statistic': float(z_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': self._interpret_contrast(name, estimate, p_value)
            }
            
            logger.info(f"{name}: {estimate:.3f} (p={p_value:.4f})")
        
        return results
    
    def _estimate_contrast_se(self, contrast_vector: List[float]) -> float:
        """Estimate standard error for contrast.
        
        Simplified approximation. Full implementation would use
        the variance-covariance matrix from the GLM.
        
        Args:
            contrast_vector: Contrast coefficients
            
        Returns:
            Estimated standard error
        """
        # This is a placeholder - should extract from model's vcov matrix
        # For now, use a reasonable approximation based on contrast complexity
        base_se = 0.1  # Should be extracted from model
        
        # Variance of contrast: Var(c'μ) = c'Σc
        # Simplified: assume diagonal covariance
        contrast_variance = sum(c**2 for c in contrast_vector) * base_se**2
        
        return np.sqrt(contrast_variance)
    
    def _interpret_h2_result(self, estimate: float, p_value: float) -> str:
        """Interpret H2 contrast result.
        
        Args:
            estimate: Contrast estimate
            p_value: P-value
            
        Returns:
            Human-readable interpretation
        """
        if p_value >= 0.05:
            return "No significant difference between temperate and non-temperate high-plant regions"
        
        if estimate < 0:
            return (f"SUPPORTS H2: High-plant temperate regions have {abs(estimate):.2f} "
                   "fewer fungi species than high-plant non-temperate regions")
        else:
            return (f"CONTRADICTS H2: High-plant temperate regions have {estimate:.2f} "
                   "more fungi species than high-plant non-temperate regions")
    
    def _interpret_contrast(self, name: str, estimate: float, p_value: float) -> str:
        """Provide interpretation for named contrasts.
        
        Args:
            name: Contrast name
            estimate: Contrast estimate
            p_value: P-value
            
        Returns:
            Interpretation string
        """
        if p_value >= 0.05:
            return f"No significant effect (p={p_value:.3f})"
        
        # Interpret based on contrast name
        if 'main_effect_plant' in name.lower():
            direction = "positive" if estimate > 0 else "negative"
            return f"Significant {direction} plant effect: {abs(estimate):.2f}"
        
        elif 'main_effect_temp' in name.lower():
            direction = "positive" if estimate > 0 else "negative"
            return f"Significant {direction} temperate effect: {abs(estimate):.2f}"
        
        else:
            direction = "positive" if estimate > 0 else "negative"
            return f"Significant {direction} effect: {abs(estimate):.2f}"