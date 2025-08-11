"""
Effect Size Evaluation for DML (Option 2).

Magnitude-based inference without p-values, focusing on
confidence intervals and practical significance.
"""

import logging
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class DMLEffectSize:
    """Effect size evaluation for DML causal parameters.
    
    Focuses on magnitude and precision of effects rather than
    statistical significance. Uses confidence intervals and
    practical significance thresholds.
    """
    
    @staticmethod
    def evaluate_effect(
        theta: float,
        se_theta: float,
        confidence_level: float = 0.95,
        practical_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluate causal effect based on magnitude and precision.
        
        Args:
            theta: Causal effect estimate
            se_theta: Standard error of theta
            confidence_level: Confidence level for interval (default 0.95)
            practical_threshold: Minimum effect size for practical significance
            
        Returns:
            Dictionary with effect size evaluation:
            - effect_size: Point estimate
            - ci_lower: Lower confidence bound
            - ci_upper: Upper confidence bound
            - ci_width: Width of confidence interval (precision)
            - classification: Effect size classification
            - interpretation: Text interpretation
        """
        if se_theta <= 0:
            raise ValueError("Standard error must be positive")
        
        # Calculate confidence interval
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = theta - z_critical * se_theta
        ci_upper = theta + z_critical * se_theta
        ci_width = ci_upper - ci_lower
        
        # Relative precision (coefficient of variation)
        if theta != 0:
            relative_precision = abs(se_theta / theta)
        else:
            relative_precision = float('inf')
        
        # Classify effect size
        classification = DMLEffectSize._classify_effect(
            theta, ci_lower, ci_upper, practical_threshold
        )
        
        # Generate interpretation
        interpretation = DMLEffectSize._generate_interpretation(
            theta, ci_lower, ci_upper, ci_width, classification, relative_precision
        )
        
        result = {
            'effect_size': float(theta),
            'standard_error': float(se_theta),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_width': float(ci_width),
            'confidence_level': float(confidence_level),
            'relative_precision': float(relative_precision),
            'classification': classification,
            'interpretation': interpretation
        }
        
        if practical_threshold is not None:
            result['practical_threshold'] = float(practical_threshold)
            result['exceeds_threshold'] = abs(ci_lower) > practical_threshold
        
        logger.info(f"Effect size: θ={theta:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], "
                   f"Classification: {classification}")
        
        return result
    
    @staticmethod
    def evaluate_biodiversity_effects(
        theta_plants: float,
        se_plants: float,
        theta_vertebrates: float,
        se_vertebrates: float,
        confidence_level: float = 0.95,
        ecological_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Evaluate biodiversity effects for ecological significance.
        
        Args:
            theta_plants: Causal effect of plants on fungi
            se_plants: Standard error for plants effect
            theta_vertebrates: Causal effect of vertebrates on fungi
            se_vertebrates: Standard error for vertebrates effect
            confidence_level: Confidence level for intervals
            ecological_threshold: Minimum effect for ecological relevance
            
        Returns:
            Dictionary with comprehensive effect size evaluation
        """
        # Evaluate each effect
        plants_eval = DMLEffectSize.evaluate_effect(
            theta_plants, se_plants, confidence_level, ecological_threshold
        )
        vertebrates_eval = DMLEffectSize.evaluate_effect(
            theta_vertebrates, se_vertebrates, confidence_level, ecological_threshold
        )
        
        # Compare effect sizes
        effect_ratio = theta_plants / theta_vertebrates if theta_vertebrates != 0 else float('inf')
        
        # Determine which effect is stronger
        if abs(theta_plants) > abs(theta_vertebrates):
            stronger_effect = "plants"
            strength_ratio = abs(theta_plants / theta_vertebrates) if theta_vertebrates != 0 else float('inf')
        else:
            stronger_effect = "vertebrates"
            strength_ratio = abs(theta_vertebrates / theta_plants) if theta_plants != 0 else float('inf')
        
        # Overall assessment
        both_positive = plants_eval['ci_lower'] > 0 and vertebrates_eval['ci_lower'] > 0
        both_meaningful = (plants_eval.get('exceeds_threshold', False) and 
                          vertebrates_eval.get('exceeds_threshold', False))
        
        if both_positive and both_meaningful:
            overall_assessment = "Strong evidence for biodiversity begets biodiversity"
        elif both_positive:
            overall_assessment = "Positive effects detected but may be small"
        elif plants_eval['ci_lower'] > 0 or vertebrates_eval['ci_lower'] > 0:
            overall_assessment = "Mixed evidence - only one biodiversity measure shows clear positive effect"
        else:
            overall_assessment = "No clear evidence for positive biodiversity effects"
        
        result = {
            'plants_effect': plants_eval,
            'vertebrates_effect': vertebrates_eval,
            'effect_comparison': {
                'stronger_effect': stronger_effect,
                'strength_ratio': float(strength_ratio) if strength_ratio != float('inf') else None,
                'effect_ratio': float(effect_ratio) if effect_ratio != float('inf') else None
            },
            'both_positive': both_positive,
            'both_meaningful': both_meaningful,
            'overall_assessment': overall_assessment,
            'summary': {
                'plants': f"{theta_plants:.3f} [{plants_eval['ci_lower']:.3f}, {plants_eval['ci_upper']:.3f}]",
                'vertebrates': f"{theta_vertebrates:.3f} [{vertebrates_eval['ci_lower']:.3f}, {vertebrates_eval['ci_upper']:.3f}]"
            }
        }
        
        logger.info(f"Biodiversity effects: {overall_assessment}")
        
        return result
    
    @staticmethod
    def _classify_effect(
        theta: float,
        ci_lower: float,
        ci_upper: float,
        practical_threshold: Optional[float] = None
    ) -> str:
        """Classify effect size based on magnitude and CI.
        
        Args:
            theta: Effect estimate
            ci_lower: Lower CI bound
            ci_upper: Upper CI bound
            practical_threshold: Optional practical significance threshold
            
        Returns:
            Classification string
        """
        # Check if CI excludes zero
        if ci_lower > 0:
            direction = "positive"
            excludes_zero = True
        elif ci_upper < 0:
            direction = "negative"
            excludes_zero = True
        else:
            direction = "uncertain"
            excludes_zero = False
        
        # Magnitude classification (Cohen's d-like)
        abs_theta = abs(theta)
        if abs_theta < 0.2:
            magnitude = "negligible"
        elif abs_theta < 0.5:
            magnitude = "small"
        elif abs_theta < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        # Combine classifications
        if not excludes_zero:
            return f"{magnitude} but uncertain direction"
        
        if practical_threshold is not None:
            if abs(ci_lower) > practical_threshold:
                return f"{magnitude} {direction} (practically significant)"
            else:
                return f"{magnitude} {direction} (below practical threshold)"
        
        return f"{magnitude} {direction}"
    
    @staticmethod
    def _generate_interpretation(
        theta: float,
        ci_lower: float,
        ci_upper: float,
        ci_width: float,
        classification: str,
        relative_precision: float
    ) -> str:
        """Generate interpretation of effect size.
        
        Args:
            theta: Effect estimate
            ci_lower: Lower CI bound
            ci_upper: Upper CI bound
            ci_width: CI width
            classification: Effect classification
            relative_precision: Coefficient of variation
            
        Returns:
            Interpretation string
        """
        # Precision assessment
        if relative_precision < 0.1:
            precision_desc = "very precise"
        elif relative_precision < 0.3:
            precision_desc = "precise"
        elif relative_precision < 0.5:
            precision_desc = "moderately precise"
        else:
            precision_desc = "imprecise"
        
        # Direction and magnitude
        if ci_lower > 0:
            direction_interp = f"positive effect of {theta:.3f} units"
            confidence_interp = f"We can be confident the true effect is between {ci_lower:.3f} and {ci_upper:.3f}"
        elif ci_upper < 0:
            direction_interp = f"negative effect of {theta:.3f} units"
            confidence_interp = f"We can be confident the true effect is between {ci_lower:.3f} and {ci_upper:.3f}"
        else:
            direction_interp = f"uncertain effect (estimate: {theta:.3f})"
            confidence_interp = f"The confidence interval [{ci_lower:.3f}, {ci_upper:.3f}] includes zero"
        
        interpretation = (
            f"Effect size analysis reveals a {direction_interp} ({classification}). "
            f"{confidence_interp}. "
            f"The estimate is {precision_desc} (CI width: {ci_width:.3f})."
        )
        
        return interpretation
    
    @staticmethod
    def compare_effect_sizes(
        effects: List[Tuple[str, float, float]],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Compare multiple effect sizes.
        
        Args:
            effects: List of (label, theta, se) tuples
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with effect size comparisons
        """
        evaluations = []
        for label, theta, se in effects:
            eval_result = DMLEffectSize.evaluate_effect(theta, se, confidence_level)
            eval_result['label'] = label
            evaluations.append(eval_result)
        
        # Sort by absolute effect size
        evaluations.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        
        # Create ranking
        ranking = [
            {
                'rank': i + 1,
                'label': eval_result['label'],
                'effect_size': eval_result['effect_size'],
                'ci': f"[{eval_result['ci_lower']:.3f}, {eval_result['ci_upper']:.3f}]",
                'classification': eval_result['classification']
            }
            for i, eval_result in enumerate(evaluations)
        ]
        
        result = {
            'evaluations': evaluations,
            'ranking': ranking,
            'strongest_effect': evaluations[0]['label'],
            'weakest_effect': evaluations[-1]['label'],
            'n_positive': sum(1 for e in evaluations if e['ci_lower'] > 0),
            'n_negative': sum(1 for e in evaluations if e['ci_upper'] < 0),
            'n_uncertain': sum(1 for e in evaluations if e['ci_lower'] <= 0 and e['ci_upper'] >= 0)
        }
        
        return result