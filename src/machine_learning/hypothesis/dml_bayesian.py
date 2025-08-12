"""
Bayesian-Style Interpretation for DML (Option 3).

Probability-based inference using posterior distributions
simulated from DML estimates.
"""

import logging
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class DMLBayesian:
    """Bayesian-style interpretation of DML causal effects.
    
    Uses asymptotic normality of DML estimators to simulate
    posterior distributions and calculate probabilities.
    """
    
    @staticmethod
    def calculate_probabilities(
        theta: float,
        se_theta: float,
        n_simulations: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculate probability statements for causal effect.
        
        Args:
            theta: Causal effect estimate
            se_theta: Standard error of theta
            n_simulations: Number of posterior samples
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with probability-based inference:
            - prob_positive: P(θ > 0)
            - prob_negative: P(θ < 0)
            - prob_magnitude: P(|θ| > threshold) for various thresholds
            - credible_interval: 95% credible interval
            - interpretation: Text interpretation
        """
        if se_theta <= 0:
            raise ValueError("Standard error must be positive")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Simulate posterior distribution
        posterior_samples = np.random.normal(theta, se_theta, n_simulations)
        
        # Calculate probabilities
        prob_positive = (posterior_samples > 0).mean()
        prob_negative = (posterior_samples < 0).mean()
        
        # Probability of exceeding various effect sizes
        thresholds = [0.1, 0.2, 0.5, 1.0]
        prob_exceeds = {}
        for threshold in thresholds:
            prob_exceeds[f'exceeds_{threshold}'] = (np.abs(posterior_samples) > threshold).mean()
        
        # Credible intervals
        credible_interval_95 = np.percentile(posterior_samples, [2.5, 97.5])
        credible_interval_90 = np.percentile(posterior_samples, [5, 95])
        credible_interval_80 = np.percentile(posterior_samples, [10, 90])
        
        # Most probable direction
        if prob_positive > prob_negative:
            direction = "positive"
            direction_prob = prob_positive
        else:
            direction = "negative"
            direction_prob = prob_negative
        
        # Generate interpretation
        interpretation = DMLBayesian._generate_interpretation(
            theta, prob_positive, prob_exceeds, credible_interval_95
        )
        
        # Evidence strength classification
        evidence_strength = DMLBayesian._classify_evidence(prob_positive)
        
        result = {
            'posterior_mean': float(theta),
            'posterior_sd': float(se_theta),
            'prob_positive': float(prob_positive),
            'prob_negative': float(prob_negative),
            'most_probable_direction': direction,
            'direction_probability': float(direction_prob),
            'prob_exceeds_thresholds': {k: float(v) for k, v in prob_exceeds.items()},
            'credible_interval_95': [float(credible_interval_95[0]), float(credible_interval_95[1])],
            'credible_interval_90': [float(credible_interval_90[0]), float(credible_interval_90[1])],
            'credible_interval_80': [float(credible_interval_80[0]), float(credible_interval_80[1])],
            'evidence_strength': evidence_strength,
            'interpretation': interpretation,
            'n_simulations': n_simulations
        }
        
        logger.info(f"Bayesian analysis: P(θ>0)={prob_positive:.3f}, "
                   f"Evidence: {evidence_strength}")
        
        return result
    
    @staticmethod
    def test_biodiversity_hypothesis_bayesian(
        theta_plants: float,
        se_plants: float,
        theta_vertebrates: float,
        se_vertebrates: float,
        n_simulations: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Bayesian evaluation of biodiversity hypothesis.
        
        Args:
            theta_plants: Causal effect of plants on fungi
            se_plants: Standard error for plants effect
            theta_vertebrates: Causal effect of vertebrates on fungi
            se_vertebrates: Standard error for vertebrates effect
            n_simulations: Number of posterior samples
            seed: Random seed
            
        Returns:
            Dictionary with Bayesian hypothesis evaluation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Simulate joint posterior
        plants_posterior = np.random.normal(theta_plants, se_plants, n_simulations)
        vertebrates_posterior = np.random.normal(theta_vertebrates, se_vertebrates, n_simulations)
        
        # Individual probabilities
        plants_prob = DMLBayesian.calculate_probabilities(
            theta_plants, se_plants, n_simulations, seed
        )
        vertebrates_prob = DMLBayesian.calculate_probabilities(
            theta_vertebrates, se_vertebrates, n_simulations, seed
        )
        
        # Joint probabilities
        prob_both_positive = ((plants_posterior > 0) & (vertebrates_posterior > 0)).mean()
        prob_either_positive = ((plants_posterior > 0) | (vertebrates_posterior > 0)).mean()
        prob_plants_stronger = (plants_posterior > vertebrates_posterior).mean()
        
        # Probability both exceed ecological threshold
        ecological_threshold = 0.1
        prob_both_meaningful = (
            (plants_posterior > ecological_threshold) & 
            (vertebrates_posterior > ecological_threshold)
        ).mean()
        
        # Generate overall assessment
        if prob_both_positive > 0.95:
            h1_assessment = f"Very strong evidence for H1 ({prob_both_positive:.1%} probability both positive)"
        elif prob_both_positive > 0.80:
            h1_assessment = f"Strong evidence for H1 ({prob_both_positive:.1%} probability both positive)"
        elif prob_either_positive > 0.95:
            h1_assessment = f"Partial evidence for H1 ({prob_either_positive:.1%} probability at least one positive)"
        else:
            h1_assessment = f"Weak evidence for H1 ({prob_both_positive:.1%} probability both positive)"
        
        result = {
            'plants_effect': plants_prob,
            'vertebrates_effect': vertebrates_prob,
            'joint_probabilities': {
                'both_positive': float(prob_both_positive),
                'either_positive': float(prob_either_positive),
                'both_meaningful': float(prob_both_meaningful),
                'plants_stronger': float(prob_plants_stronger)
            },
            'h1_assessment': h1_assessment,
            'posterior_correlation': float(np.corrcoef(plants_posterior, vertebrates_posterior)[0, 1]),
            'summary': {
                'plants_prob_positive': float(plants_prob['prob_positive']),
                'vertebrates_prob_positive': float(vertebrates_prob['prob_positive']),
                'joint_prob_positive': float(prob_both_positive)
            }
        }
        
        logger.info(f"Bayesian H1 test: {h1_assessment}")
        
        return result
    
    @staticmethod
    def _classify_evidence(prob: float) -> str:
        """Classify strength of evidence based on probability.
        
        Uses Jeffreys' scale adapted for probabilities.
        
        Args:
            prob: Probability of positive effect
            
        Returns:
            Evidence classification
        """
        # Convert to Bayes factor approximation (assuming equal priors)
        if prob > 0.5:
            bf_approx = prob / (1 - prob)
        else:
            bf_approx = (1 - prob) / prob
        
        # Classify based on probability
        if prob > 0.99 or prob < 0.01:
            return "decisive evidence"
        elif prob > 0.97 or prob < 0.03:
            return "very strong evidence"
        elif prob > 0.95 or prob < 0.05:
            return "strong evidence"
        elif prob > 0.90 or prob < 0.10:
            return "moderate evidence"
        elif prob > 0.75 or prob < 0.25:
            return "weak evidence"
        else:
            return "no clear evidence"
    
    @staticmethod
    def _generate_interpretation(
        theta: float,
        prob_positive: float,
        prob_exceeds: Dict[str, float],
        credible_interval: np.ndarray
    ) -> str:
        """Generate Bayesian interpretation.
        
        Args:
            theta: Point estimate
            prob_positive: Probability of positive effect
            prob_exceeds: Probabilities of exceeding thresholds
            credible_interval: 95% credible interval
            
        Returns:
            Interpretation string
        """
        # Evidence classification
        evidence = DMLBayesian._classify_evidence(prob_positive)
        
        # Direction interpretation
        if prob_positive > 0.5:
            direction = "positive"
            direction_prob = prob_positive
        else:
            direction = "negative"
            direction_prob = 1 - prob_positive
        
        # Magnitude interpretation
        if prob_exceeds.get('exceeds_0.5', 0) > 0.8:
            magnitude_desc = "large effect"
        elif prob_exceeds.get('exceeds_0.2', 0) > 0.8:
            magnitude_desc = "moderate effect"
        elif prob_exceeds.get('exceeds_0.1', 0) > 0.8:
            magnitude_desc = "small effect"
        else:
            magnitude_desc = "negligible effect"
        
        interpretation = (
            f"Bayesian analysis shows {evidence} for a {direction} effect "
            f"({direction_prob:.1%} probability). "
            f"The posterior mean is {theta:.3f} with 95% credible interval "
            f"[{credible_interval[0]:.3f}, {credible_interval[1]:.3f}]. "
            f"This represents a {magnitude_desc} "
            f"({prob_exceeds.get('exceeds_0.1', 0):.1%} probability |θ| > 0.1)."
        )
        
        return interpretation
    
    @staticmethod
    def calculate_bayes_factors(
        theta: float,
        se_theta: float,
        null_value: float = 0,
        alternative_values: Optional[List[float]] = None,
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """Calculate Bayes factors for various hypotheses.
        
        Args:
            theta: Causal effect estimate
            se_theta: Standard error
            null_value: Value under null hypothesis
            alternative_values: Specific alternative values to test
            n_simulations: Number of simulations
            
        Returns:
            Dictionary with Bayes factors
        """
        if alternative_values is None:
            alternative_values = [0.1, 0.2, 0.5]
        
        # Simulate posterior
        posterior = np.random.normal(theta, se_theta, n_simulations)
        
        # Calculate densities
        null_density = stats.norm.pdf(null_value, theta, se_theta)
        
        bayes_factors = {}
        for alt_value in alternative_values:
            alt_density = stats.norm.pdf(alt_value, theta, se_theta)
            bf = alt_density / null_density if null_density > 0 else float('inf')
            bayes_factors[f'BF_{alt_value}_vs_{null_value}'] = float(bf)
        
        # Point null vs composite alternative
        prob_positive = (posterior > null_value).mean()
        bf_composite = prob_positive / (1 - prob_positive) if prob_positive < 1 else float('inf')
        
        result = {
            'null_value': float(null_value),
            'bayes_factors': bayes_factors,
            'bf_positive_vs_null': float(bf_composite),
            'interpretation': DMLBayesian._interpret_bayes_factor(bf_composite)
        }
        
        return result
    
    @staticmethod
    def _interpret_bayes_factor(bf: float) -> str:
        """Interpret Bayes factor using Jeffreys' scale.
        
        Args:
            bf: Bayes factor
            
        Returns:
            Interpretation string
        """
        if bf > 100:
            return "Decisive evidence for alternative"
        elif bf > 30:
            return "Very strong evidence for alternative"
        elif bf > 10:
            return "Strong evidence for alternative"
        elif bf > 3:
            return "Moderate evidence for alternative"
        elif bf > 1:
            return "Weak evidence for alternative"
        elif bf > 1/3:
            return "No clear evidence"
        elif bf > 1/10:
            return "Moderate evidence for null"
        else:
            return "Strong evidence for null"