from src.abstractions import get_logger
"""
Large Sample Effect Size Interpretations and Warnings.

This module provides practical significance guidance for large-N datasets
where p-values become meaningless (everything is "significant").

Key functions:
1. Large sample warnings and thresholds
2. Practical significance interpretations
3. Effect size context for hypothesis tests
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = get_logger(__name__)


class LargeSampleEffects:
    """Handle effect size interpretations and warnings for large samples."""
    
    @staticmethod
    def add_large_sample_context(
        results: Dict[str, Any],
        n_obs: int,
        test_type: str = "hypothesis_test"
    ) -> Dict[str, Any]:
        """
        Add warnings and context for large sample sizes to test results.
        
        With millions of observations, statistical significance becomes
        meaningless. This adds appropriate warnings and focuses attention
        on effect sizes and practical significance.
        
        Args:
            results: Test results dictionary to enhance
            n_obs: Number of observations
            test_type: Type of test ("lrt", "wald", "contrast", etc.)
            
        Returns:
            Enhanced results with large sample context
        """
        if n_obs > 100000:
            context = {
                'is_large_sample': True,
                'n_observations': n_obs,
                'warning_message': (
                    f"With {n_obs:,} observations, even tiny effects will be "
                    f"statistically significant. Focus on effect sizes, not p-values."
                )
            }
            
            # Add severity based on sample size
            if n_obs > 10000000:  # > 10 million
                context['severity'] = 'extreme'
                context['recommendation'] = (
                    "P-values are meaningless at this scale. "
                    "Use only effect sizes and practical thresholds."
                )
            elif n_obs > 1000000:  # > 1 million
                context['severity'] = 'high'
                context['recommendation'] = (
                    "Interpret p-values with extreme caution. "
                    "Effect sizes are much more informative."
                )
            else:  # > 100k
                context['severity'] = 'moderate'
                context['recommendation'] = (
                    "Consider effect sizes alongside p-values."
                )
            
            # Test-specific warnings
            if test_type == "lrt" and 'percent_deviance_explained' in results:
                threshold = LargeSampleEffects.get_practical_threshold(n_obs, "deviance")
                pct = results['percent_deviance_explained']
                if pct < threshold:
                    context['practical_warning'] = (
                        f"Effect size ({pct:.1f}%) below suggested threshold "
                        f"({threshold}%) for n={n_obs:,}"
                    )
            
            elif test_type == "wald" and 'coefficient' in results:
                threshold = LargeSampleEffects.get_practical_threshold(n_obs, "coefficient")
                coef = abs(results.get('coefficient', 0))
                if coef < threshold:
                    context['practical_warning'] = (
                        f"Coefficient magnitude ({coef:.3f}) below suggested "
                        f"threshold ({threshold:.3f}) for n={n_obs:,}"
                    )
            
            results['large_sample_context'] = context
        
        return results
    
    @staticmethod
    def get_practical_threshold(
        n_obs: int,
        effect_type: str = "deviance"
    ) -> float:
        """
        Get suggested practical significance thresholds based on sample size.
        
        Larger samples need stricter thresholds to avoid detecting
        meaningless effects.
        
        Args:
            n_obs: Number of observations
            effect_type: Type of effect ("deviance", "coefficient", "aic_difference")
            
        Returns:
            Suggested threshold value
        """
        if effect_type == "deviance":
            # Percent deviance explained thresholds for LRT
            if n_obs < 1000:
                return 1.0  # 1% for small samples
            elif n_obs < 10000:
                return 2.0  # 2% for medium samples
            elif n_obs < 100000:
                return 5.0  # 5% for large samples
            elif n_obs < 1000000:
                return 10.0  # 10% for very large samples
            else:
                return 15.0  # 15% for massive samples
        
        elif effect_type == "coefficient":
            # Standardized coefficient magnitude thresholds
            if n_obs < 10000:
                return 0.1  # Small effect
            elif n_obs < 100000:
                return 0.2  # Medium effect
            elif n_obs < 1000000:
                return 0.3  # Large effect
            else:
                return 0.5  # Very large effect only
        
        elif effect_type == "aic_difference":
            # AIC difference thresholds (delegated to ModelComparison)
            if n_obs < 1000:
                return 2.0
            elif n_obs < 10000:
                return 10.0
            elif n_obs < 100000:
                return 50.0
            elif n_obs < 1000000:
                return 100.0
            else:
                return 100 * np.log10(n_obs / 1000000 + 1)
        
        else:
            # Default conservative threshold
            logger.warning(f"Unknown effect type: {effect_type}")
            return 0.1
    
    @staticmethod
    def interpret_with_sample_size(
        p_value: float,
        effect_size: float,
        n_obs: int,
        effect_type: str = "general",
        effect_name: str = "Effect"
    ) -> str:
        """
        Generate interpretation that considers both significance and sample size.
        
        Args:
            p_value: Statistical p-value
            effect_size: Magnitude of effect
            n_obs: Number of observations
            effect_type: Type of effect for threshold selection
            effect_name: Name of the effect for reporting
            
        Returns:
            Contextual interpretation string
        """
        lines = []
        
        # Statistical significance
        if p_value < 0.001:
            lines.append(f"Highly significant (p < 0.001)")
        elif p_value < 0.01:
            lines.append(f"Significant (p = {p_value:.3f})")
        elif p_value < 0.05:
            lines.append(f"Marginally significant (p = {p_value:.3f})")
        else:
            lines.append(f"Not significant (p = {p_value:.3f})")
        
        # Effect size interpretation
        lines.append(f"{effect_name}: {effect_size:.3f}")
        
        # Large sample context
        if n_obs > 100000:
            threshold = LargeSampleEffects.get_practical_threshold(n_obs, effect_type)
            lines.append(f"Large sample (n={n_obs:,})")
            
            if abs(effect_size) < threshold:
                lines.append(f"⚠️ Below practical threshold ({threshold:.3f})")
                if p_value < 0.05:
                    lines.append("Statistically significant but practically negligible")
            else:
                lines.append(f"✓ Exceeds practical threshold ({threshold:.3f})")
                if p_value < 0.05:
                    lines.append("Both statistically and practically significant")
        
        return " | ".join(lines)
    
    @staticmethod
    def should_focus_on_effect_size(n_obs: int) -> bool:
        """
        Determine if effect sizes should be prioritized over p-values.
        
        Args:
            n_obs: Number of observations
            
        Returns:
            True if sample is large enough that p-values are unreliable
        """
        return n_obs > 100000
    
    @staticmethod
    def get_reporting_recommendation(n_obs: int) -> str:
        """
        Get recommendation for how to report results based on sample size.
        
        Args:
            n_obs: Number of observations
            
        Returns:
            Recommendation string
        """
        if n_obs < 1000:
            return (
                "Report p-values and confidence intervals. "
                "Effect sizes optional but recommended."
            )
        elif n_obs < 10000:
            return (
                "Report p-values, effect sizes, and confidence intervals. "
                "Emphasize practical significance."
            )
        elif n_obs < 100000:
            return (
                "Report effect sizes prominently alongside p-values. "
                "Include practical significance thresholds."
            )
        elif n_obs < 1000000:
            return (
                "Focus primarily on effect sizes and practical significance. "
                "P-values should be de-emphasized or omitted."
            )
        else:
            return (
                "Report only effect sizes and practical significance. "
                "P-values are meaningless at this scale and should be omitted."
            )