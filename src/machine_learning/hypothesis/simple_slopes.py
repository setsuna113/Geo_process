"""
Simple Slopes Analysis for testing interactions in 2x2 factorial designs.

Test 7: Simple Slopes Analysis
This module tests whether the effect of one factor differs across levels
of another factor, specifically for H2 testing the temperate effect within
plant diversity levels.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, Optional, List, Union

from ..evaluation.emmeans_core import REMmeansWrapper, SimpleEMmeansApproximation

logger = logging.getLogger(__name__)


class SimpleSlopesAnalysis:
    """Perform simple slopes analysis on estimated marginal means.
    
    IMPORTANT: This class is specifically for METHOD 2 ONLY, which tests H2.
    Methods 1 and 3 test H0/H1 and should not use this class.
    
    Tests whether the effect of one factor changes across levels of another.
    For H2: Tests if temperate effect differs between high and low plant regions.
    """
    
    def __init__(self, emmeans_wrapper: Union[REMmeansWrapper, SimpleEMmeansApproximation],
                 vcov_matrix: Optional[np.ndarray] = None):
        """Initialize simple slopes analyzer.
        
        Args:
            emmeans_wrapper: REMmeansWrapper or SimpleEMmeansApproximation instance
            vcov_matrix: Optional variance-covariance matrix for proper SE calculation
        """
        self.emmeans_wrapper = emmeans_wrapper
        self.vcov_matrix = vcov_matrix
        
        # For backward compatibility - check if wrapper has old interface
        if hasattr(emmeans_wrapper, 'vcov_matrix'):
            self.vcov_matrix = emmeans_wrapper.vcov_matrix
        
        # Extract emmeans ONCE at initialization
        self.emmeans = self._extract_emmeans()
    
    def _extract_emmeans(self) -> np.ndarray:
        """Extract emmeans array from wrapper - single source of truth.
        
        Returns:
            Array of 4 emmeans in order: [low_nontemp, high_nontemp, low_temp, high_temp]
        """
        if isinstance(self.emmeans_wrapper, REMmeansWrapper):
            return self.emmeans_wrapper.get_emmeans_array()
        else:
            # For SimpleEMmeansApproximation
            emmeans_dict = self.emmeans_wrapper.compute_emmeans_2x2()
            return np.array([
                emmeans_dict['group_0_0']['emmean'],
                emmeans_dict['group_1_0']['emmean'],
                emmeans_dict['group_0_1']['emmean'],
                emmeans_dict['group_1_1']['emmean']
            ])
        
    def test_simple_slopes_h2(self) -> Dict[str, Any]:
        """Test H2 using simple slopes analysis.
        
        Computes:
        1. Effect of temperate in low-plant regions
        2. Effect of temperate in high-plant regions
        3. Difference between these effects (interaction)
        
        For H2, we focus on #2: whether temperate has a negative effect
        specifically in high-plant regions.
        
        Returns:
            Dictionary with simple slopes results
        """
        logger.info("Performing simple slopes analysis for H2")
        
        # Use pre-extracted emmeans
        emmeans = self.emmeans
        # Order: [low_nontemp, high_nontemp, low_temp, high_temp]
        
        # Simple slope 1: Effect of temperate in LOW plant regions
        slope_low_plant = emmeans[2] - emmeans[0]  # group_0_1 - group_0_0
        se_low = self._estimate_slope_se([1, 0, -1, 0])
        
        # Simple slope 2: Effect of temperate in HIGH plant regions (KEY FOR H2)
        slope_high_plant = emmeans[3] - emmeans[1]  # group_1_1 - group_1_0
        se_high = self._estimate_slope_se([0, 1, 0, -1])
        
        # Test each slope
        z_low = slope_low_plant / se_low if se_low > 0 else 0
        p_low = 2 * (1 - stats.norm.cdf(abs(z_low)))
        
        z_high = slope_high_plant / se_high if se_high > 0 else 0
        p_high = 2 * (1 - stats.norm.cdf(abs(z_high)))
        
        # Interaction: difference between slopes
        interaction = slope_high_plant - slope_low_plant
        se_interaction = self._estimate_slope_se([1, -1, -1, 1])
        z_interaction = interaction / se_interaction if se_interaction > 0 else 0
        p_interaction = 2 * (1 - stats.norm.cdf(abs(z_interaction)))
        
        # H2 is supported if temperate effect is negative in high-plant regions
        h2_supported = (p_high < 0.05) and (slope_high_plant < 0)
        
        result = {
            'emmeans': emmeans.tolist(),  # Add emmeans for visualization
            'slope_temperate_in_low_plant': {
                'estimate': float(slope_low_plant),
                'standard_error': float(se_low),
                'z_statistic': float(z_low),
                'p_value': float(p_low),
                'significant': p_low < 0.05,
                'interpretation': self._interpret_slope(slope_low_plant, p_low, 'low plant')
            },
            'slope_temperate_in_high_plant': {
                'estimate': float(slope_high_plant),
                'standard_error': float(se_high),
                'z_statistic': float(z_high),
                'p_value': float(p_high),
                'significant': p_high < 0.05,
                'interpretation': self._interpret_slope(slope_high_plant, p_high, 'high plant')
            },
            'interaction_effect': {
                'estimate': float(interaction),
                'standard_error': float(se_interaction),
                'z_statistic': float(z_interaction),
                'p_value': float(p_interaction),
                'significant': p_interaction < 0.05,
                'interpretation': self._interpret_interaction(interaction, p_interaction)
            },
            'h2_supported': h2_supported,
            'h2_interpretation': self._interpret_h2_simple_slopes(
                slope_high_plant, p_high, h2_supported
            )
        }
        
        logger.info(f"Temperate effect in high-plant: {slope_high_plant:.3f} (p={p_high:.4f})")
        logger.info(f"H2 supported: {h2_supported}")
        
        return result
    
    def test_all_simple_effects(self) -> Dict[str, Any]:
        """Test all simple effects in the 2x2 design.
        
        Computes effects of each factor at each level of the other:
        - Temperate effect at each plant level
        - Plant effect at each temperate level
        
        Returns:
            Dictionary with all simple effects
        """
        # Use pre-extracted emmeans
        emmeans = self.emmeans
        
        results = {
            'emmeans': emmeans.tolist(),  # Standardize: always include emmeans
            'temperate_effects': {},
            'plant_effects': {}
        }
        
        # Temperate effects at each plant level
        results['temperate_effects']['at_low_plant'] = self._compute_simple_effect(
            emmeans, [1, 0, -1, 0], 'temperate', 'plant_high=0'
        )
        results['temperate_effects']['at_high_plant'] = self._compute_simple_effect(
            emmeans, [0, 1, 0, -1], 'temperate', 'plant_high=1'
        )
        
        # Plant effects at each temperate level
        results['plant_effects']['at_non_temperate'] = self._compute_simple_effect(
            emmeans, [1, -1, 0, 0], 'plant', 'temperate=0'
        )
        results['plant_effects']['at_temperate'] = self._compute_simple_effect(
            emmeans, [0, 0, 1, -1], 'plant', 'temperate=1'
        )
        
        return results
    
    def _compute_simple_effect(
        self,
        emmeans: np.ndarray,
        contrast: list,
        effect_name: str,
        condition: str
    ) -> Dict[str, Any]:
        """Compute a single simple effect.
        
        Args:
            emmeans: Array of 4 emmeans
            contrast: Contrast vector for this effect
            effect_name: Name of the effect being tested
            condition: Condition under which effect is tested
            
        Returns:
            Dictionary with simple effect results
        """
        estimate = np.dot(contrast, emmeans)
        se = self._estimate_slope_se(contrast)
        z_stat = estimate / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'effect': effect_name,
            'condition': condition,
            'estimate': float(estimate),
            'standard_error': float(se),
            'z_statistic': float(z_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    def _estimate_slope_se(self, contrast: list) -> float:
        """Estimate standard error for a simple slope using variance-covariance matrix.
        
        Args:
            contrast: Contrast vector for the slope
            
        Returns:
            Estimated standard error
        """
        contrast = np.array(contrast)
        
        # Primary method: Use vcov matrix if available
        if self.vcov_matrix is not None and len(contrast) == self.vcov_matrix.shape[0]:
            contrast_variance = contrast @ self.vcov_matrix @ contrast
            return np.sqrt(max(0, contrast_variance))
        
        # Fallback: Use individual SEs if available
        if hasattr(self.emmeans_wrapper, 'emmeans_results'):
            emmeans_results = self.emmeans_wrapper.emmeans_results
            if emmeans_results and isinstance(emmeans_results, dict):
                # Extract SEs for each group
                ses = []
                for key in sorted(emmeans_results.keys()):
                    if 'group' in key and 'SE' in emmeans_results[key]:
                        se = emmeans_results[key]['SE']
                        if se is not None:
                            ses.append(se)
                
                if len(ses) == len(contrast):
                    # SE(contrast) ≈ sqrt(sum(c_i^2 * SE_i^2))
                    contrast_variance = sum(c**2 * se**2 for c, se in zip(contrast, ses))
                    return np.sqrt(max(0, contrast_variance))
        
        # If nothing available, raise error - don't guess
        raise ValueError("Cannot estimate SE: no variance-covariance matrix or individual SEs available")
    
    def _interpret_slope(self, slope: float, p_value: float, condition: str) -> str:
        """Interpret a simple slope.
        
        Args:
            slope: Slope estimate
            p_value: P-value
            condition: Condition description
            
        Returns:
            Interpretation string
        """
        if p_value >= 0.05:
            return f"No significant temperate effect in {condition} regions"
        
        direction = "positive" if slope > 0 else "negative"
        return (f"Significant {direction} temperate effect in {condition} regions: "
                f"{abs(slope):.2f} {'more' if slope > 0 else 'fewer'} fungi")
    
    def _interpret_interaction(self, interaction: float, p_value: float) -> str:
        """Interpret the interaction effect.
        
        Args:
            interaction: Interaction estimate (difference of slopes)
            p_value: P-value
            
        Returns:
            Interpretation string
        """
        if p_value >= 0.05:
            return "No significant interaction: temperate effect similar across plant levels"
        
        return (f"Significant interaction: temperate effect differs by {abs(interaction):.2f} "
                "between high and low plant regions")
    
    def _interpret_h2_simple_slopes(
        self,
        slope_high: float,
        p_high: float,
        h2_supported: bool
    ) -> str:
        """Interpret simple slopes results for H2.
        
        Args:
            slope_high: Temperate effect in high-plant regions
            p_high: P-value for high-plant slope
            h2_supported: Whether H2 is supported
            
        Returns:
            H2-specific interpretation
        """
        if h2_supported:
            return (f"H2 SUPPORTED: In high-plant regions, temperate areas have "
                   f"{abs(slope_high):.2f} fewer fungi species (p={p_high:.4f})")
        elif p_high < 0.05 and slope_high > 0:
            return (f"H2 CONTRADICTED: In high-plant regions, temperate areas have "
                   f"{slope_high:.2f} MORE fungi species (p={p_high:.4f})")
        else:
            return "H2 NOT SUPPORTED: No significant temperate effect in high-plant regions"
    
    # Report generation has been moved to src/machine_learning/reporting/publication_reports.py
    # Use: from ..reporting.publication_reports import generate_simple_slopes_publication_report
    
    # Visualization moved to src/machine_learning/visualization/simple_slopes_plots.py
    # Use: from ..visualization.simple_slopes_plots import visualize_simple_slopes