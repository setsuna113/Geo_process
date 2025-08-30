from src.abstractions import get_logger
"""
Contrast Analysis for testing specific hypotheses using emmeans.

Test 6: Contrast Analysis on Fitted Values
This module tests specific linear combinations of group means to evaluate
directional hypotheses like H2 (temperate mismatch).
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple, Union

from ..evaluation.emmeans_core import REMmeansWrapper

logger = get_logger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class ContrastAnalysis:
    """Perform contrast analysis on estimated marginal means.
    
    IMPORTANT: This class is specifically for METHOD 2 ONLY, which tests H2.
    Methods 1 and 3 test H0/H1 and should not use this class.
    
    Tests specific hypotheses by applying contrast vectors to emmeans.
    For H2: Tests whether high-plant temperate regions have fewer fungi
    than high-plant non-temperate regions.
    """
    
    def __init__(self, emmeans_wrapper: REMmeansWrapper, 
                 vcov_matrix: Optional[np.ndarray] = None):
        """Initialize contrast analyzer.
        
        Args:
            emmeans_wrapper: REMmeansWrapper instance
            vcov_matrix: Optional variance-covariance matrix for proper SE calculation
        """
        self.emmeans_wrapper = emmeans_wrapper
        self.vcov_matrix = vcov_matrix
        
        # For backward compatibility - check if wrapper has old interface
        if hasattr(emmeans_wrapper, 'vcov_matrix'):
            self.vcov_matrix = emmeans_wrapper.vcov_matrix
        
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
        logger.info(f"Testing H2 with contrast vector: {contrast_vector}")
        
        # Get emmeans in standard order - R is REQUIRED
        emmeans = self.emmeans_wrapper.get_emmeans_array()
        
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
        # Get emmeans in standard order - R is REQUIRED
        emmeans = self.emmeans_wrapper.get_emmeans_array()
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
        """计算对比的标准误差
        
        Args:
            contrast_vector: Contrast coefficients
            
        Returns:
            Estimated standard error
        """
        contrast_vector = np.array(contrast_vector)
        
        # 优先使用方差-协方差矩阵（最准确）
        if self.vcov_matrix is not None:
            variance = contrast_vector @ self.vcov_matrix @ contrast_vector
            return np.sqrt(max(0, variance))
        
        # 否则要求必须提供
        raise ValueError("Variance-covariance matrix required for SE calculation")
    
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
    
    def generate_publication_report(
        self,
        contrast_results: Dict[str, Any],
        emmeans_results: Optional[Dict[str, Any]] = None,
        hypothesis_name: str = "H2 (Temperate Mismatch)"
    ) -> Dict[str, Any]:
        """Generate publication-ready report for contrast analysis.
        
        Following ecology journal standards for reporting emmeans contrasts.
        
        Args:
            contrast_results: Results from test_h2_contrast or test_custom_contrasts
            emmeans_results: Optional emmeans results for context
            hypothesis_name: Name of hypothesis being tested
            
        Returns:
            Dictionary with formatted text, tables, and visualizations
        """
        # Extract statistics
        if 'contrast_estimate' in contrast_results:
            # Single contrast result
            estimate = contrast_results['contrast_estimate']
            se = contrast_results['standard_error']
            z_stat = contrast_results['z_statistic']
            p_value = contrast_results['p_value']
            contrast_vec = contrast_results['contrast_vector']
            h2_supported = contrast_results.get('h2_supported', False)
        else:
            # Multiple contrasts - use first or specified one
            first_key = list(contrast_results.keys())[0]
            first_result = contrast_results[first_key]
            estimate = first_result['estimate']
            se = first_result['standard_error']
            z_stat = first_result['z_statistic']
            p_value = first_result['p_value']
            contrast_vec = first_result['contrast_vector']
            h2_supported = False
        
        # Format p-value
        if p_value < 0.001:
            p_str = "p < 0.001"
            p_formatted = "< 0.001"
        elif p_value < 0.01:
            p_str = f"p = {p_value:.3f}"
            p_formatted = f"{p_value:.3f}"
        else:
            p_str = f"p = {p_value:.2f}"
            p_formatted = f"{p_value:.2f}"
        
        # Generate main results text
        main_text = (
            f"Contrast analysis was performed to test {hypothesis_name}. "
            f"The contrast estimate was {estimate:.2f} (SE = {se:.2f}, z = {z_stat:.2f}, {p_str}). "
        )
        
        if hypothesis_name == "H2 (Temperate Mismatch)":
            if h2_supported:
                main_text += (
                    f"This result supports H2, indicating that high-plant temperate regions "
                    f"have {abs(estimate):.2f} fewer fungi species than high-plant non-temperate "
                    f"regions (95% CI [{estimate - 1.96*se:.2f}, {estimate + 1.96*se:.2f}]). "
                    f"This confirms the biodiversity decoupling pattern in temperate zones."
                )
            elif p_value < 0.05 and estimate > 0:
                main_text += (
                    f"This result contradicts H2, showing that high-plant temperate regions "
                    f"have {estimate:.2f} more fungi species than expected."
                )
            else:
                main_text += (
                    f"The contrast was not statistically significant, providing no support "
                    f"for the temperate mismatch hypothesis."
                )
        else:
            if p_value < 0.05:
                main_text += f"The contrast is statistically significant."
            else:
                main_text += f"The contrast is not statistically significant."
        
        # Generate methods text
        methods_text = (
            f"Estimated marginal means (emmeans) were computed for the 2×2 factorial design "
            f"(plant richness × climate type). Contrast analysis was performed using the "
            f"contrast vector {contrast_vec} to test specific hypotheses about group differences. "
            f"Standard errors were computed using the delta method, and significance was "
            f"assessed using z-tests at α = 0.05."
        )
        
        # Create results table
        results_table = pd.DataFrame({
            'Hypothesis': [hypothesis_name],
            'Contrast': [str(contrast_vec)],
            'Estimate': [f"{estimate:.3f}"],
            'SE': [f"{se:.3f}"],
            'z': [f"{z_stat:.2f}"],
            'p-value': [p_formatted],
            'Decision': ['Supported' if h2_supported else 'Not Supported' if 'H2' in hypothesis_name else 
                        'Significant' if p_value < 0.05 else 'Not Significant']
        })
        
        # Confidence intervals
        ci_lower = estimate - 1.96 * se
        ci_upper = estimate + 1.96 * se
        
        # Supplementary statistics
        supplementary_stats = pd.DataFrame({
            'Statistic': ['Contrast Estimate', '95% CI Lower', '95% CI Upper', 
                         'Standard Error', 'z-statistic', 'p-value'],
            'Value': [f"{estimate:.3f}", f"{ci_lower:.3f}", f"{ci_upper:.3f}",
                     f"{se:.3f}", f"{z_stat:.2f}", p_formatted]
        })
        
        # Get emmeans if available
        emmeans_table = None
        if hasattr(self.emmeans_wrapper, 'emmeans_results') and self.emmeans_wrapper.emmeans_results:
            if isinstance(self.emmeans_wrapper, REMmeansWrapper):
                emmeans = self.emmeans_wrapper.get_emmeans_array()
            else:
                emmeans_dict = self.emmeans_wrapper.compute_emmeans_2x2()
                emmeans = np.array([
                    emmeans_dict['group_0_0']['emmean'],
                    emmeans_dict['group_1_0']['emmean'],
                    emmeans_dict['group_0_1']['emmean'],
                    emmeans_dict['group_1_1']['emmean']
                ])
            group_names = ['Low Plant\nNon-temp', 'High Plant\nNon-temp',
                          'Low Plant\nTemperate', 'High Plant\nTemperate']
            emmeans_table = pd.DataFrame({
                'Group': group_names,
                'Estimated Mean': [f"{m:.2f}" for m in emmeans]
            })
        
        return {
            'main_text': main_text,
            'methods_text': methods_text,
            'results_table': results_table,
            'supplementary_stats': supplementary_stats,
            'emmeans_table': emmeans_table,
            # 'latex_table': self._generate_latex_table(results_table),  # Moved to reporting module
            'statistical_summary': {
                'estimate': estimate,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        }
    

    # Visualization methods moved to src/machine_learning/visualization/interaction_plots.py
    # Use: from ..visualization.interaction_plots import visualize_contrast_results
    # Note: Pass self (ContrastAnalysis instance) as first parameter
    
    # LaTeX generation moved to src/machine_learning/reporting/latex_tables.py
    # Use: from ..reporting.latex_tables import generate_contrast_latex_table
