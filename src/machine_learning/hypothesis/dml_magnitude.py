from src.abstractions import get_logger
"""
Pure Magnitude Classification for DML Effects.

Focuses on effect size interpretation without threshold testing.
Threshold testing is handled by WaldMethod with SESOI and TOST.

This module provides Cohen's d style magnitude classification and
relative effect comparisons without redundant significance testing.
"""

import logging
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple

logger = get_logger(__name__)


class DMLMagnitude:
    """Effect magnitude classification for DML causal parameters.
    
    Provides pure magnitude-based interpretation without threshold
    testing (which is handled by WaldMethod SESOI/TOST).
    """
    
    @staticmethod
    def classify_magnitude(
        theta: float,
        se_theta: float,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Classify effect magnitude without threshold testing.
        
        Uses Cohen's d style classification for standardized effects.
        
        Args:
            theta: Causal effect estimate
            se_theta: Standard error of theta
            confidence_level: Confidence level for interval (default 0.95)
            
        Returns:
            Dictionary with magnitude classification:
            - magnitude_class: negligible/small/medium/large/very_large
            - relative_precision: coefficient of variation
            - confidence_interval: CI bounds
            - interpretation: Text description
        """
        if se_theta <= 0:
            raise ValueError("Standard error must be positive")
        
        # Calculate confidence interval
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = theta - z_critical * se_theta
        ci_upper = theta + z_critical * se_theta
        ci_width = ci_upper - ci_lower
        
        # Relative precision (coefficient of variation)
        if abs(theta) < 0.001:  # Effect too close to zero for reliable CV
            relative_precision = float('inf')
            precision_class = "undefined (effect too close to zero)"
        elif theta != 0:
            relative_precision = abs(se_theta / theta)
            precision_class = DMLMagnitude._classify_precision(relative_precision, theta)
        else:
            relative_precision = float('inf')
            precision_class = "undefined (effect is zero)"
        
        # Classify magnitude (Cohen's d style, adapted for log scale)
        abs_theta = abs(theta)
        magnitude_class = DMLMagnitude._get_magnitude_class(abs_theta)
        
        # Direction
        if ci_lower > 0:
            direction = "positive"
            direction_certainty = "clear"
        elif ci_upper < 0:
            direction = "negative"
            direction_certainty = "clear"
        else:
            direction = "uncertain"
            direction_certainty = "unclear (CI includes zero)"
        
        # Generate interpretation
        interpretation = (
            f"Effect magnitude is {magnitude_class} "
            f"(|θ| = {abs_theta:.4f}). "
        )
        
        if direction != "uncertain":
            interpretation += f"Direction is clearly {direction}. "
        else:
            interpretation += f"Direction is uncertain (CI includes zero). "
        
        interpretation += f"Precision is {precision_class} (CV = {relative_precision:.2f})."
        
        result = {
            'effect_size': float(theta),
            'standard_error': float(se_theta),
            'magnitude_class': magnitude_class,
            'direction': direction,
            'direction_certainty': direction_certainty,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_width': float(ci_width),
            'confidence_level': float(confidence_level),
            'relative_precision': float(relative_precision),
            'precision_class': precision_class,
            'interpretation': interpretation
        }
        
        logger.info(f"Magnitude: θ={theta:.4f}, class={magnitude_class}, precision={precision_class}")
        
        return result
    
    @staticmethod
    def _get_magnitude_class(abs_theta: float) -> str:
        """Classify absolute magnitude using standardized thresholds.
        
        Thresholds adapted for log-scale effects in ecology:
        - Negligible: < 0.01 (< 1% change)
        - Small: 0.01-0.05 (1-5% change)
        - Medium: 0.05-0.15 (5-15% change)
        - Large: 0.15-0.30 (15-30% change)
        - Very large: > 0.30 (> 30% change)
        """
        if abs_theta < 0.01:
            return "negligible"
        elif abs_theta < 0.05:
            return "small"
        elif abs_theta < 0.15:
            return "medium"
        elif abs_theta < 0.30:
            return "large"
        else:
            return "very large"
    
    @staticmethod
    def _classify_precision(cv: float, theta: float = None) -> str:
        """Classify precision based on coefficient of variation.
        
        Args:
            cv: Coefficient of variation (SE/theta)
            theta: Effect estimate (for checking if too close to zero)
            
        Returns:
            Precision classification string
        """
        # Check if effect is too close to zero for reliable CV
        if theta is not None and abs(theta) < 0.001:
            return "undefined (effect too close to zero)"
        
        # Updated thresholds based on evidence from manufacturing and analytical chemistry
        if cv < 0.05:
            return "excellent"
        elif cv < 0.10:
            return "very good"
        elif cv < 0.20:
            return "acceptable"
        elif cv < 0.30:
            return "marginal"
        else:
            return "poor"
    
    @staticmethod
    def compare_magnitudes(
        effects: List[Tuple[str, float, float]],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Compare magnitudes across multiple effects.
        
        Args:
            effects: List of (name, theta, se) tuples
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with comparative analysis
        """
        if len(effects) < 2:
            raise ValueError("Need at least 2 effects to compare")
        
        # Classify each effect
        classifications = []
        for name, theta, se in effects:
            result = DMLMagnitude.classify_magnitude(theta, se, confidence_level)
            result['name'] = name
            classifications.append(result)
        
        # Sort by absolute magnitude
        classifications.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        
        # Find strongest and weakest
        strongest = classifications[0]
        weakest = classifications[-1]
        
        # Calculate relative strengths
        if weakest['effect_size'] != 0:
            relative_strength = abs(strongest['effect_size'] / weakest['effect_size'])
        else:
            relative_strength = float('inf')
        
        # Determine dominant effect
        dominant_effect = strongest['name']
        
        # Check consistency of direction
        directions = [c['direction'] for c in classifications if c['direction'] != 'uncertain']
        if len(set(directions)) == 1 and directions:
            direction_consistency = f"consistent ({directions[0]})"
        elif len(directions) == 0:
            direction_consistency = "all uncertain"
        else:
            direction_consistency = "mixed"
        
        # Summary statistics
        magnitudes = [abs(c['effect_size']) for c in classifications]
        
        comparison = {
            'n_effects': len(effects),
            'rankings': [
                {
                    'rank': i + 1,
                    'name': c['name'],
                    'magnitude': c['effect_size'],
                    'magnitude_class': c['magnitude_class'],
                    'precision_class': c['precision_class']
                }
                for i, c in enumerate(classifications)
            ],
            'strongest_effect': {
                'name': strongest['name'],
                'magnitude': strongest['effect_size'],
                'class': strongest['magnitude_class']
            },
            'weakest_effect': {
                'name': weakest['name'],
                'magnitude': weakest['effect_size'],
                'class': weakest['magnitude_class']
            },
            'relative_strength_ratio': float(relative_strength),
            'direction_consistency': direction_consistency,
            'summary_stats': {
                'mean_magnitude': float(np.mean(magnitudes)),
                'median_magnitude': float(np.median(magnitudes)),
                'std_magnitude': float(np.std(magnitudes)),
                'range_magnitude': [float(min(magnitudes)), float(max(magnitudes))]
            },
            'interpretation': DMLMagnitude._interpret_comparison(
                classifications, relative_strength, direction_consistency
            )
        }
        
        return comparison
    
    @staticmethod
    def _interpret_comparison(
        classifications: List[Dict],
        relative_strength: float,
        direction_consistency: str
    ) -> str:
        """Generate interpretation of effect comparison."""
        strongest = classifications[0]
        weakest = classifications[-1]
        
        interpretation = (
            f"The strongest effect is {strongest['name']} "
            f"({strongest['magnitude_class']} magnitude), "
        )
        
        if relative_strength == float('inf'):
            interpretation += f"while {weakest['name']} has zero effect. "
        elif relative_strength > 10:
            interpretation += (
                f"which is {relative_strength:.1f}x stronger than "
                f"the weakest effect ({weakest['name']}). "
            )
        elif relative_strength > 2:
            interpretation += (
                f"which is {relative_strength:.1f}x stronger than "
                f"{weakest['name']}. "
            )
        else:
            interpretation += f"with all effects having similar magnitudes. "
        
        # Direction consistency
        if direction_consistency == "consistent (positive)":
            interpretation += "All effects are consistently positive."
        elif direction_consistency == "consistent (negative)":
            interpretation += "All effects are consistently negative."
        elif direction_consistency == "mixed":
            interpretation += "Effects have mixed directions (both positive and negative)."
        else:
            interpretation += "Effect directions are uncertain."
        
        return interpretation
    
    @staticmethod
    def from_julia_results(julia_dml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze magnitudes directly from Julia DML results.
        
        Args:
            julia_dml_results: Julia method3_dml results dictionary
            
        Returns:
            Complete magnitude analysis including comparisons
        """
        # Extract coefficients and std errors
        coefficients = julia_dml_results.get('coefficients', [])
        std_errors = julia_dml_results.get('std_errors', [])
        coef_names = julia_dml_results.get('coef_names', [])
        
        if not coefficients or not std_errors or not coef_names:
            raise ValueError("Julia results missing required fields")
        
        # Individual magnitude classifications
        magnitudes = {}
        effects_list = []
        
        for i, name in enumerate(coef_names):
            coef = coefficients[i]
            se = std_errors[i]
            
            # Classify each effect
            mag_result = DMLMagnitude.classify_magnitude(coef, se)
            magnitudes[name] = mag_result
            effects_list.append((name, coef, se))
        
        # Compare all effects
        comparison = DMLMagnitude.compare_magnitudes(effects_list) if len(effects_list) > 1 else None
        
        # Special biodiversity evaluation if applicable
        biodiversity_eval = None
        # Detect biodiversity predictors dynamically (looking for _std suffix)
        biodiv_predictors = [name for name in coef_names if name.endswith('_richness_std')]
        
        if len(biodiv_predictors) >= 2:
            # Use first two biodiversity predictors found
            pred1, pred2 = biodiv_predictors[:2]
            pred1_idx = coef_names.index(pred1)
            pred2_idx = coef_names.index(pred2)
            
            # Look for interaction term
            interaction_idx = None
            pred1_base = pred1.replace('_richness_std', '')
            pred2_base = pred2.replace('_richness_std', '')
            possible_interactions = [
                f'{pred1_base}_x_{pred2_base}',
                f'{pred2_base}_x_{pred1_base}',
                f'{pred1}_x_{pred2}',
                f'{pred2}_x_{pred1}'
            ]
            for interaction_name in possible_interactions:
                if interaction_name in coef_names:
                    interaction_idx = coef_names.index(interaction_name)
                    break
            
            biodiversity_eval = DMLMagnitude.evaluate_biodiversity_magnitudes(
                theta_plants=coefficients[pred1_idx],
                se_plants=std_errors[pred1_idx],
                theta_vertebrates=coefficients[pred2_idx],
                se_vertebrates=std_errors[pred2_idx],
                theta_interaction=coefficients[interaction_idx] if interaction_idx is not None else None,
                se_interaction=std_errors[interaction_idx] if interaction_idx is not None else None
            )
        
        return {
            'individual_magnitudes': magnitudes,
            'comparison': comparison,
            'biodiversity_evaluation': biodiversity_eval,
            'n_obs': julia_dml_results.get('n_obs'),
            'method': julia_dml_results.get('method_note', 'DML')
        }
    
    @staticmethod
    def evaluate_biodiversity_magnitudes(
        theta_plants: float,
        se_plants: float,
        theta_vertebrates: float,
        se_vertebrates: float,
        theta_interaction: Optional[float] = None,
        se_interaction: Optional[float] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Specialized magnitude evaluation for biodiversity effects.
        
        Args:
            theta_plants: Plant effect estimate
            se_plants: Standard error for plants
            theta_vertebrates: Vertebrate effect estimate
            se_vertebrates: Standard error for vertebrates
            theta_interaction: Optional interaction effect
            se_interaction: Optional interaction standard error
            confidence_level: Confidence level for intervals
            
        Returns:
            Comprehensive magnitude evaluation for biodiversity
        """
        # Build effects list
        effects = [
            ('plants', theta_plants, se_plants),
            ('vertebrates', theta_vertebrates, se_vertebrates)
        ]
        
        if theta_interaction is not None and se_interaction is not None:
            effects.append(('interaction', theta_interaction, se_interaction))
        
        # Get individual classifications
        plants_mag = DMLMagnitude.classify_magnitude(theta_plants, se_plants, confidence_level)
        vertebrates_mag = DMLMagnitude.classify_magnitude(theta_vertebrates, se_vertebrates, confidence_level)
        
        result = {
            'plants': plants_mag,
            'vertebrates': vertebrates_mag
        }
        
        if theta_interaction is not None:
            interaction_mag = DMLMagnitude.classify_magnitude(
                theta_interaction, se_interaction, confidence_level
            )
            result['interaction'] = interaction_mag
        
        # Compare all effects
        result['comparison'] = DMLMagnitude.compare_magnitudes(effects, confidence_level)
        
        # Overall biodiversity assessment
        both_positive = (plants_mag['direction'] == 'positive' and 
                        vertebrates_mag['direction'] == 'positive')
        both_meaningful = (plants_mag['magnitude_class'] not in ['negligible', 'small'] and
                          vertebrates_mag['magnitude_class'] not in ['negligible', 'small'])
        
        if both_positive and both_meaningful:
            overall = "Strong biodiversity effects"
        elif both_positive:
            overall = "Positive but weak biodiversity effects"
        elif plants_mag['direction'] == 'positive' or vertebrates_mag['direction'] == 'positive':
            overall = "Mixed biodiversity effects"
        else:
            overall = "No clear biodiversity effects"
        
        result['overall_assessment'] = overall
        result['interpretation'] = (
            f"{overall}. "
            f"Plant effect is {plants_mag['magnitude_class']} and {plants_mag['direction']}. "
            f"Vertebrate effect is {vertebrates_mag['magnitude_class']} and {vertebrates_mag['direction']}."
        )
        
        if theta_interaction is not None:
            if interaction_mag['magnitude_class'] != 'negligible':
                result['interpretation'] += (
                    f" Interaction is {interaction_mag['magnitude_class']} "
                    f"and {interaction_mag['direction']}, suggesting effects are interdependent."
                )
            else:
                result['interpretation'] += " Interaction is negligible, suggesting independent effects."
        
        return result