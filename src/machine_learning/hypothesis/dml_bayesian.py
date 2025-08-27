"""
Proper Bayesian Inference for DML (Bayesian Double Machine Learning).

Based on recent research (2024-2025) on Bayesian DML methods that provide
valid posterior inference with credible intervals that have correct coverage.

References:
- "Bayesian Double Machine Learning for Causal Inference" (2024)
- "Double Robust Bayesian Inference on Average Treatment Effects" (2025)
"""

import logging
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple
import warnings

logger = logging.getLogger(__name__)


class BayesianDML:
    """Proper Bayesian inference for DML causal effects.
    
    This implements a principled Bayesian approach that:
    1. Uses informative priors based on domain knowledge
    2. Corrects for regularization-induced confounding
    3. Provides valid credible intervals with correct coverage
    4. Respects the likelihood principle
    """
    
    @staticmethod
    def bayesian_inference_with_prior(
        theta: float,
        se_theta: float,
        prior_mean: float = 0.0,
        prior_sd: Optional[float] = None,
        n_posterior_samples: int = 10000,
        credible_level: float = 0.95,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Proper Bayesian inference with informative prior.
        
        Uses conjugate normal prior to obtain posterior distribution.
        Unlike the pseudo-Bayesian approach, this properly combines
        prior information with the likelihood.
        
        Args:
            theta: DML estimate (MLE from orthogonal score)
            se_theta: Standard error from DML
            prior_mean: Prior mean for the effect (default: 0 = no effect)
            prior_sd: Prior standard deviation (None = weakly informative)
            n_posterior_samples: Number of posterior samples to draw
            credible_level: Level for credible interval (default: 0.95)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with proper Bayesian inference results
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Set weakly informative prior if not specified
        if prior_sd is None:
            # Weakly informative: 3x the standard error
            prior_sd = 3 * se_theta
            logger.info(f"Using weakly informative prior: N({prior_mean}, {prior_sd})")
        
        # Compute posterior parameters (conjugate normal-normal)
        # Posterior precision = prior precision + likelihood precision
        prior_precision = 1 / (prior_sd ** 2)
        likelihood_precision = 1 / (se_theta ** 2)
        posterior_precision = prior_precision + likelihood_precision
        posterior_variance = 1 / posterior_precision
        
        # Posterior mean = weighted average of prior and likelihood
        posterior_mean = (
            (prior_precision * prior_mean + likelihood_precision * theta) /
            posterior_precision
        )
        posterior_sd = np.sqrt(posterior_variance)
        
        # Draw from posterior
        posterior_samples = np.random.normal(
            posterior_mean, posterior_sd, n_posterior_samples
        )
        
        # Compute posterior probabilities
        prob_positive = (posterior_samples > 0).mean()
        prob_negative = (posterior_samples < 0).mean()
        
        # Credible intervals
        alpha = (1 - credible_level) / 2
        credible_interval = np.percentile(
            posterior_samples, [alpha * 100, (1 - alpha) * 100]
        )
        
        # Bayes factor against null (theta = 0)
        # BF = p(data|H1) / p(data|H0)
        # Under normal model: BF = exp(-0.5 * (z1^2 - z0^2))
        z_null = (theta - 0) / se_theta  # z-score under null
        z_prior = (theta - prior_mean) / np.sqrt(se_theta**2 + prior_sd**2)
        log_bf = 0.5 * (z_null**2 - z_prior**2)
        bayes_factor = np.exp(log_bf)
        
        # Posterior probability of null (using Savage-Dickey ratio)
        posterior_density_at_zero = stats.norm.pdf(0, posterior_mean, posterior_sd)
        prior_density_at_zero = stats.norm.pdf(0, prior_mean, prior_sd)
        savage_dickey_bf = prior_density_at_zero / posterior_density_at_zero
        
        # Shrinkage factor (how much prior pulls estimate toward prior mean)
        shrinkage = prior_precision / posterior_precision
        
        result = {
            # Prior specification
            'prior': {
                'mean': float(prior_mean),
                'sd': float(prior_sd),
                'type': 'Normal (conjugate)'
            },
            # Posterior summary
            'posterior': {
                'mean': float(posterior_mean),
                'sd': float(posterior_sd),
                'median': float(np.median(posterior_samples)),
                'mode': float(posterior_mean),  # For normal, mode = mean
            },
            # Probabilities
            'probabilities': {
                'positive': float(prob_positive),
                'negative': float(prob_negative),
                'max_probability': float(max(prob_positive, prob_negative)),
                'direction': 'positive' if prob_positive > prob_negative else 'negative'
            },
            # Credible intervals
            'credible_intervals': {
                f'{int(credible_level*100)}%': [
                    float(credible_interval[0]), 
                    float(credible_interval[1])
                ],
                '90%': [
                    float(np.percentile(posterior_samples, 5)),
                    float(np.percentile(posterior_samples, 95))
                ],
                '80%': [
                    float(np.percentile(posterior_samples, 10)),
                    float(np.percentile(posterior_samples, 90))
                ]
            },
            # Model comparison
            'bayes_factors': {
                'bf_vs_null': float(bayes_factor),
                'savage_dickey_bf': float(savage_dickey_bf),
                'log_bf': float(log_bf)
            },
            # Shrinkage and influence
            'shrinkage_factor': float(shrinkage),
            'data_influence': float(1 - shrinkage),
            # Original data
            'data': {
                'theta_mle': float(theta),
                'se_mle': float(se_theta)
            },
            'n_samples': n_posterior_samples
        }
        
        # Add interpretation
        result['interpretation'] = BayesianDML._interpret_bayesian_results(result)
        
        return result
    
    @staticmethod
    def empirical_bayes_inference(
        theta: float,
        se_theta: float,
        historical_effects: Optional[List[float]] = None,
        n_posterior_samples: int = 10000,
        credible_level: float = 0.95,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Empirical Bayes inference using data to inform prior.
        
        This approach estimates the prior from historical data or
        uses a data-driven approach to set hyperparameters.
        
        Args:
            theta: DML estimate
            se_theta: Standard error from DML
            historical_effects: Previous effect estimates (for empirical prior)
            n_posterior_samples: Number of posterior samples
            credible_level: Level for credible interval
            seed: Random seed
            
        Returns:
            Empirical Bayes inference results
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Estimate prior from historical effects or use default
        if historical_effects and len(historical_effects) > 1:
            # Empirical prior from historical data
            prior_mean = np.mean(historical_effects)
            prior_sd = np.std(historical_effects)
            prior_type = 'Empirical (data-driven)'
            logger.info(f"Using empirical prior from {len(historical_effects)} historical effects")
        else:
            # Default weakly informative prior centered at small positive effect
            # Based on biodiversity literature suggesting positive relationships
            prior_mean = 0.05  # Small positive effect expected
            prior_sd = 0.2     # Moderate uncertainty
            prior_type = 'Weakly informative (domain knowledge)'
            logger.info("Using weakly informative prior based on domain knowledge")
        
        # Use the proper Bayesian inference with estimated prior
        result = BayesianDML.bayesian_inference_with_prior(
            theta=theta,
            se_theta=se_theta,
            prior_mean=prior_mean,
            prior_sd=prior_sd,
            n_posterior_samples=n_posterior_samples,
            credible_level=credible_level,
            seed=seed
        )
        
        # Update prior type
        result['prior']['type'] = prior_type
        if historical_effects:
            result['prior']['n_historical'] = len(historical_effects)
        
        return result
    
    @staticmethod
    def proper_bayesian_dml(
        Y_residuals: np.ndarray,
        A_residuals: np.ndarray,
        prior_alpha_mean: float = 0.0,
        prior_alpha_var: float = 1.0,
        n_posterior_samples: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Proper Bayesian DML using DiTraglia & Gao (2024) approach.
        
        This method uses the correct BDML implementation that learns
        causal effects through the covariance matrix to avoid
        regularization-induced confounding.
        
        Args:
            Y_residuals: Outcome residuals from DML cross-fitting
            A_residuals: Treatment residuals from DML cross-fitting 
            prior_alpha_mean: Prior mean for causal effect
            prior_alpha_var: Prior variance for causal effect
            n_posterior_samples: Number of posterior samples
            seed: Random seed
            
        Returns:
            Proper Bayesian DML inference results
        """
        # Import the correct implementation
        from .bdml_correct import CorrectBayesianDML
        
        # Use the proper BDML algorithm
        result = CorrectBayesianDML.bdml_posterior(
            Y_residuals=Y_residuals,
            A_residuals=A_residuals,
            prior_alpha_mean=prior_alpha_mean,
            prior_alpha_var=prior_alpha_var,
            n_posterior_samples=n_posterior_samples,
            seed=seed
        )
        
        return result
    
    @staticmethod
    def proper_multivariate_bdml(
        Y_residuals: np.ndarray,
        A_residuals: np.ndarray,
        prior_R2: float = 0.5,
        n_posterior_samples: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Proper multivariate BDML for multiple treatments.
        
        Uses the R2D2 prior for better finite-sample performance.
        
        Args:
            Y_residuals: Outcome residuals (n×1)
            A_residuals: Treatment residuals (n×p) for p treatments
            prior_R2: Prior expected R-squared (0 to 1)
            n_posterior_samples: Number of posterior samples
            seed: Random seed
            
        Returns:
            Multivariate BDML results
        """
        from .bdml_correct import CorrectBayesianDML
        
        return CorrectBayesianDML.bdml_multivariate(
            Y_residuals=Y_residuals,
            A_residuals=A_residuals,
            prior_R2=prior_R2,
            n_posterior_samples=n_posterior_samples,
            seed=seed
        )
    
    @staticmethod
    def _interpret_bayesian_results(result: Dict[str, Any]) -> str:
        """Generate interpretation of Bayesian results."""
        prob_positive = result['probabilities']['positive']
        posterior_mean = result['posterior']['mean']
        ci_95 = result['credible_intervals']['95%']
        bf = result['bayes_factors']['bf_vs_null']
        
        # Interpret Bayes factor (Jeffreys' scale)
        if bf < 1/10:
            bf_interpretation = "strong evidence against effect"
        elif bf < 1/3:
            bf_interpretation = "moderate evidence against effect"
        elif bf < 1:
            bf_interpretation = "weak evidence against effect"
        elif bf < 3:
            bf_interpretation = "weak evidence for effect"
        elif bf < 10:
            bf_interpretation = "moderate evidence for effect"
        else:
            bf_interpretation = "strong evidence for effect"
        
        # Overall interpretation
        interpretation = (
            f"Posterior mean: {posterior_mean:.4f}, "
            f"95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]. "
            f"Probability of positive effect: {prob_positive:.1%}. "
            f"Bayes factor: {bf:.2f} ({bf_interpretation}). "
        )
        
        # Add shrinkage interpretation
        shrinkage = result['shrinkage_factor']
        if shrinkage > 0.5:
            interpretation += f"Prior has strong influence (shrinkage: {shrinkage:.1%})."
        elif shrinkage > 0.2:
            interpretation += f"Prior has moderate influence (shrinkage: {shrinkage:.1%})."
        else:
            interpretation += f"Data dominates prior (shrinkage: {shrinkage:.1%})."
        
        return interpretation
    
    @staticmethod
    def test_biodiversity_hypothesis_bayesian(
        theta_plants: float,
        se_plants: float,
        theta_vertebrates: float,
        se_vertebrates: float,
        prior_correlation: float = 0.3,
        ecological_prior: bool = True,
        n_simulations: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Bayesian test of biodiversity hypothesis with ecological priors.
        
        Uses domain knowledge about biodiversity relationships to set
        informative priors for more efficient inference.
        
        Args:
            theta_plants: Plant effect estimate
            se_plants: Standard error for plants
            theta_vertebrates: Vertebrate effect estimate
            se_vertebrates: Standard error for vertebrates
            prior_correlation: Expected correlation between effects
            ecological_prior: Use ecological knowledge for priors
            n_simulations: Number of posterior samples
            seed: Random seed
            
        Returns:
            Joint Bayesian inference results
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Set priors based on ecological knowledge
        if ecological_prior:
            # Based on biodiversity literature
            prior_mean_plants = 0.1      # Positive expectation
            prior_sd_plants = 0.2        # Moderate uncertainty
            prior_mean_vertebrates = 0.08  # Slightly weaker expectation
            prior_sd_vertebrates = 0.2
            logger.info("Using informative ecological priors")
        else:
            # Weakly informative
            prior_mean_plants = 0
            prior_sd_plants = 1
            prior_mean_vertebrates = 0
            prior_sd_vertebrates = 1
            logger.info("Using weakly informative priors")
        
        # Individual posteriors
        plants_posterior = BayesianDML.bayesian_inference_with_prior(
            theta_plants, se_plants, prior_mean_plants, prior_sd_plants,
            n_simulations, seed=seed
        )
        
        vertebrates_posterior = BayesianDML.bayesian_inference_with_prior(
            theta_vertebrates, se_vertebrates, prior_mean_vertebrates, prior_sd_vertebrates,
            n_simulations, seed=seed if seed is None else seed + 1
        )
        
        # Joint posterior (accounting for correlation)
        # Create correlated samples
        from scipy.stats import multivariate_normal
        
        # Posterior parameters
        post_means = np.array([
            plants_posterior['posterior']['mean'],
            vertebrates_posterior['posterior']['mean']
        ])
        
        # Construct covariance matrix with correlation
        post_sds = np.array([
            plants_posterior['posterior']['sd'],
            vertebrates_posterior['posterior']['sd']
        ])
        
        # Correlation matrix
        corr_matrix = np.array([
            [1.0, prior_correlation],
            [prior_correlation, 1.0]
        ])
        
        # Covariance matrix
        cov_matrix = np.outer(post_sds, post_sds) * corr_matrix
        
        # Draw joint samples
        joint_samples = multivariate_normal.rvs(
            mean=post_means,
            cov=cov_matrix,
            size=n_simulations
        )
        
        # Joint probabilities
        both_positive = ((joint_samples[:, 0] > 0) & (joint_samples[:, 1] > 0)).mean()
        either_positive = ((joint_samples[:, 0] > 0) | (joint_samples[:, 1] > 0)).mean()
        
        # Test H1: Biodiversity begets biodiversity
        h1_probability = both_positive  # Both effects positive
        
        result = {
            'plants': plants_posterior,
            'vertebrates': vertebrates_posterior,
            'joint': {
                'both_positive': float(both_positive),
                'either_positive': float(either_positive),
                'correlation': float(prior_correlation),
                'h1_probability': float(h1_probability),
                'h1_supported': h1_probability > 0.95
            },
            'interpretation': (
                f"P(both effects > 0) = {both_positive:.1%}. "
                f"H1 {'supported' if h1_probability > 0.95 else 'not strongly supported'} "
                f"(requires P > 95% for strong evidence)."
            )
        }
        
        return result
    
    @staticmethod
    def from_julia_results(julia_dml_results: Dict[str, Any], 
                          n_simulations: int = 10000) -> Dict[str, Any]:
        """Run Bayesian DML analysis directly from Julia results.
        
        Args:
            julia_dml_results: Julia method3_dml results with bdml_stats
            n_simulations: Number of posterior samples
            
        Returns:
            Bayesian DML analysis results
        """
        # Check for BDML stats
        bdml_stats = julia_dml_results.get('bdml_stats')
        if not bdml_stats:
            raise ValueError("Julia results missing bdml_stats")
        
        # Extract covariance matrix and metadata
        S = np.array(bdml_stats['covariance_matrix'])
        n = bdml_stats['n_obs']
        treatment_names = bdml_stats.get('treatment_names', [])
        
        # Use proper BDML implementation
        from .bdml_correct import CorrectBayesianDML
        
        bdml_result = CorrectBayesianDML.bdml_posterior(
            S=S,
            n=n,
            n_posterior_samples=n_simulations
        )
        
        # For multiple treatments, extract alpha samples
        # The covariance matrix structure for k treatments is:
        # S = [[Var(Y), Cov(Y,A1), ..., Cov(Y,Ak)],
        #      [Cov(A1,Y), Var(A1), ..., Cov(A1,Ak)],
        #      ...]
        # We get k causal effects from the first row
        
        if 'alpha_posterior_samples' in bdml_result:
            # Single treatment case
            alpha_samples = bdml_result['alpha_posterior_samples']
            prob_positive = (alpha_samples > 0).mean()
            
            bdml_result['biodiversity_analysis'] = {
                'both_positive': float(prob_positive),  # For single effect
                'either_positive': float(prob_positive),
                'h1_probability': float(prob_positive),
                'h1_supported': prob_positive > 0.95,
                'interpretation': (
                    f"P(effect > 0) = {prob_positive:.1%}. "
                    f"H1 {'strongly supported' if prob_positive > 0.95 else 'not strongly supported'}."
                )
            }
        else:
            # Multiple treatments - simplified interpretation
            bdml_result['biodiversity_analysis'] = {
                'note': 'Multiple treatment BDML requires specialized implementation',
                'h1_supported': None,
                'interpretation': 'BDML analysis completed. See posterior summaries for details.'
            }
        
        return bdml_result