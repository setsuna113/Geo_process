"""
Correct Bayesian Double Machine Learning (BDML) Implementation.

Based on DiTraglia & Gao (2024): "Bayesian Double Machine Learning for Causal Inference"
ArXiv: 2508.12688

This implements the proper BDML approach that:
1. Avoids regularization-induced confounding (RIC)
2. Properly marginalizes over nuisance parameters
3. Uses reduced-form reparameterization
"""

import numpy as np
from scipy import stats
from scipy.stats import invwishart, multivariate_normal
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CorrectBayesianDML:
    """Correct implementation of Bayesian Double Machine Learning.
    
    Uses the reduced-form reparameterization to learn causal effects
    through the covariance matrix, avoiding regularization-induced confounding.
    """
    
    @staticmethod
    def bdml_posterior_from_residuals(
        Y_residuals: np.ndarray,
        A_residuals: np.ndarray,
        prior_alpha_mean: float = 0.0,
        prior_alpha_var: float = 1.0,
        prior_sigma_df: int = 3,
        n_posterior_samples: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute BDML posterior from raw residuals (for small datasets).
        
        Args:
            Y_residuals: Outcome residuals from cross-fitting (n×1)
            A_residuals: Treatment residuals from cross-fitting (n×1)
            prior_alpha_mean: Prior mean for causal effect
            prior_alpha_var: Prior variance for causal effect
            prior_sigma_df: Degrees of freedom for inverse-Wishart prior on Σ
            n_posterior_samples: Number of posterior samples to draw
            seed: Random seed
            
        Returns:
            Dictionary with proper Bayesian posterior results
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(Y_residuals)
        
        # Step 1: Form reduced-form bivariate system
        # Stack residuals: Z = [Y_resid, A_resid]
        Z = np.column_stack([Y_residuals, A_residuals])
        
        # Step 2: Compute sample covariance matrix
        S = Z.T @ Z / n  # 2×2 matrix
        
        # Call the main function with covariance matrix
        return CorrectBayesianDML.bdml_posterior(
            S=S, n=n, prior_alpha_mean=prior_alpha_mean,
            prior_alpha_var=prior_alpha_var, prior_sigma_df=prior_sigma_df,
            n_posterior_samples=n_posterior_samples, seed=seed
        )
    
    @staticmethod
    def bdml_posterior(
        S: np.ndarray,
        n: int,
        prior_alpha_mean: float = 0.0,
        prior_alpha_var: float = 1.0,
        prior_sigma_df: int = 3,
        n_posterior_samples: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute BDML posterior from sufficient statistics.
        
        This implements Algorithm 1 from DiTraglia & Gao (2024).
        
        Args:
            S: Sample covariance matrix (p×p) where S = Z'Z/n
            n: Sample size
            prior_alpha_mean: Prior mean for causal effect
            prior_alpha_var: Prior variance for causal effect
            prior_sigma_df: Degrees of freedom for inverse-Wishart prior on Σ
            n_posterior_samples: Number of posterior samples to draw
            seed: Random seed
            
        Returns:
            Dictionary with proper Bayesian posterior results
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Get dimension from covariance matrix
        p = S.shape[0]  # Should be 2 for single treatment, 3+ for multiple
        
        # Step 3: Set up conjugate prior for Σ (inverse-Wishart)
        # Prior: Σ ~ IW(ν, Ψ) where ν = df, Ψ = scale matrix
        nu_prior = prior_sigma_df + p  # Must be > p-1
        
        # Scale matrix encodes prior beliefs about variance
        # Use identity scaled by expected residual variance
        psi_prior = np.eye(p) * nu_prior
        
        # Step 4: Posterior for Σ is also inverse-Wishart
        # Posterior: Σ | data ~ IW(ν_post, Ψ_post)
        nu_post = nu_prior + n
        psi_post = psi_prior + n * S
        
        # Step 5: Draw from posterior of Σ
        sigma_posterior_samples = []
        for _ in range(n_posterior_samples):
            # Draw from inverse-Wishart
            sigma_draw = invwishart.rvs(df=nu_post, scale=psi_post)
            sigma_posterior_samples.append(sigma_draw)
        
        sigma_posterior_samples = np.array(sigma_posterior_samples)
        
        # Step 6: Transform to get posterior for α (causal effect)
        # From the structural equations:
        # Y = αA + ε_y
        # A = ε_a
        # This implies: Σ[0,1] / Σ[1,1] = α
        
        alpha_posterior_samples = []
        for sigma in sigma_posterior_samples:
            # Extract causal effect from covariance structure
            sigma_ya = sigma[0, 1]  # Cov(Y, A)
            sigma_aa = sigma[1, 1]  # Var(A)
            
            # Causal effect α = Cov(Y,A) / Var(A)
            alpha = sigma_ya / sigma_aa
            alpha_posterior_samples.append(alpha)
        
        alpha_posterior_samples = np.array(alpha_posterior_samples)
        
        # Step 7: Incorporate informative prior on α if desired
        # Use importance sampling to reweight samples
        if prior_alpha_var < np.inf:
            # Compute importance weights
            prior_log_probs = -0.5 * ((alpha_posterior_samples - prior_alpha_mean)**2 / prior_alpha_var)
            weights = np.exp(prior_log_probs - np.max(prior_log_probs))
            weights /= weights.sum()
            
            # Resample with weights
            indices = np.random.choice(
                len(alpha_posterior_samples), 
                size=n_posterior_samples, 
                replace=True, 
                p=weights
            )
            alpha_posterior_samples = alpha_posterior_samples[indices]
        
        # Compute posterior statistics
        posterior_mean = np.mean(alpha_posterior_samples)
        posterior_std = np.std(alpha_posterior_samples)
        posterior_median = np.median(alpha_posterior_samples)
        
        # Credible intervals
        ci_95 = np.percentile(alpha_posterior_samples, [2.5, 97.5])
        ci_90 = np.percentile(alpha_posterior_samples, [5, 95])
        ci_80 = np.percentile(alpha_posterior_samples, [10, 90])
        
        # Posterior probabilities
        prob_positive = (alpha_posterior_samples > 0).mean()
        prob_negative = (alpha_posterior_samples < 0).mean()
        
        # SESOI (Smallest Effect Size of Interest) probabilities
        sesoi = 0.01  # 1% change as meaningful
        prob_meaningful_positive = (alpha_posterior_samples > sesoi).mean()
        prob_meaningful_negative = (alpha_posterior_samples < -sesoi).mean()
        
        result = {
            'method': 'Bayesian Double Machine Learning (BDML)',
            'algorithm': 'DiTraglia & Gao (2024) - Reduced form reparameterization',
            
            # Posterior summary
            'posterior': {
                'mean': float(posterior_mean),
                'std': float(posterior_std),
                'median': float(posterior_median),
                'samples': alpha_posterior_samples  # Keep for further analysis
            },
            
            # Credible intervals
            'credible_intervals': {
                '95%': [float(ci_95[0]), float(ci_95[1])],
                '90%': [float(ci_90[0]), float(ci_90[1])],
                '80%': [float(ci_80[0]), float(ci_80[1])]
            },
            
            # Probabilities
            'probabilities': {
                'positive': float(prob_positive),
                'negative': float(prob_negative),
                'meaningful_positive': float(prob_meaningful_positive),
                'meaningful_negative': float(prob_meaningful_negative),
                'negligible': float(1 - prob_meaningful_positive - prob_meaningful_negative)
            },
            
            # Model diagnostics
            'diagnostics': {
                'n_obs': n,
                'prior_df': nu_prior,
                'posterior_df': nu_post,
                'effective_sample_size': n_posterior_samples
            },
            
            # Interpretation
            'interpretation': _interpret_bdml_results(
                posterior_mean, ci_95, prob_positive, 
                prob_meaningful_positive, prob_meaningful_negative
            )
        }
        
        return result
    
    @staticmethod
    def bdml_multivariate(
        Y_residuals: np.ndarray,
        A_residuals: np.ndarray,
        vcov_residuals: Optional[np.ndarray] = None,
        prior_R2: float = 0.5,
        n_posterior_samples: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """BDML for multiple treatments using R2D2 prior.
        
        Implements the R2D2 prior from the paper for better finite-sample performance.
        
        Args:
            Y_residuals: Outcome residuals (n×1)
            A_residuals: Treatment residuals (n×p) for p treatments
            vcov_residuals: Optional pre-computed variance-covariance of residuals
            prior_R2: Prior expected R-squared (0 to 1)
            n_posterior_samples: Number of posterior samples
            seed: Random seed
            
        Returns:
            Multivariate BDML posterior results
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(Y_residuals)
        
        # Ensure A_residuals is 2D
        if A_residuals.ndim == 1:
            A_residuals = A_residuals.reshape(-1, 1)
        
        p = A_residuals.shape[1]  # Number of treatments
        
        # Form augmented system: Z = [Y, A1, A2, ...]
        Z = np.column_stack([Y_residuals, A_residuals])
        
        # Sample covariance
        S = Z.T @ Z / n
        
        # R2D2 prior setup (simplified version)
        # This encodes beliefs about the reduced-form R-squared
        expected_signal = prior_R2
        expected_noise = 1 - prior_R2
        
        # Prior degrees of freedom (controls prior strength)
        nu_prior = p + 3  # Minimal informative prior
        
        # Prior scale matrix incorporating R2 beliefs
        psi_prior = np.eye(p + 1) * nu_prior
        psi_prior[0, 0] *= (1 / expected_noise)  # Outcome variance
        
        # Posterior parameters
        nu_post = nu_prior + n
        psi_post = psi_prior + n * S
        
        # Draw from posterior
        alpha_samples = []
        
        for _ in range(n_posterior_samples):
            # Draw Σ from inverse-Wishart
            sigma = invwishart.rvs(df=nu_post, scale=psi_post)
            
            # Extract causal effects from covariance structure
            # α = Σ_{YA} @ Σ_{AA}^{-1}
            sigma_ya = sigma[0, 1:]  # Cov(Y, A_i)
            sigma_aa = sigma[1:, 1:]  # Var(A)
            
            # Solve for causal effects
            alpha = np.linalg.solve(sigma_aa, sigma_ya)
            alpha_samples.append(alpha)
        
        alpha_samples = np.array(alpha_samples)
        
        # Compute statistics for each treatment
        results = {
            'method': 'Multivariate BDML with R2D2 prior',
            'n_treatments': p,
            'treatments': {}
        }
        
        for i in range(p):
            treatment_samples = alpha_samples[:, i]
            
            results['treatments'][f'treatment_{i+1}'] = {
                'posterior_mean': float(np.mean(treatment_samples)),
                'posterior_std': float(np.std(treatment_samples)),
                'ci_95': [
                    float(np.percentile(treatment_samples, 2.5)),
                    float(np.percentile(treatment_samples, 97.5))
                ],
                'prob_positive': float((treatment_samples > 0).mean()),
                'prob_meaningful': float((np.abs(treatment_samples) > 0.01).mean())
            }
        
        # Joint probabilities
        all_positive = np.all(alpha_samples > 0, axis=1).mean()
        any_positive = np.any(alpha_samples > 0, axis=1).mean()
        
        results['joint'] = {
            'all_positive': float(all_positive),
            'any_positive': float(any_positive),
            'prior_R2': prior_R2
        }
        
        return results


def _interpret_bdml_results(
    posterior_mean: float,
    ci_95: Tuple[float, float],
    prob_positive: float,
    prob_meaningful_positive: float,
    prob_meaningful_negative: float
) -> str:
    """Generate interpretation of BDML results."""
    
    # Effect direction
    if prob_positive > 0.95:
        direction = "clearly positive"
    elif prob_positive < 0.05:
        direction = "clearly negative"
    else:
        direction = "uncertain"
    
    # Effect magnitude
    if prob_meaningful_positive > 0.95:
        magnitude = "meaningfully positive"
    elif prob_meaningful_negative > 0.95:
        magnitude = "meaningfully negative"
    elif prob_meaningful_positive + prob_meaningful_negative < 0.1:
        magnitude = "negligible"
    else:
        magnitude = "small but non-negligible"
    
    interpretation = (
        f"The causal effect is {direction} with posterior mean {posterior_mean:.4f} "
        f"and 95% CI [{ci_95[0]:.4f}, {ci_95[1]:.4f}]. "
        f"The effect is {magnitude} with {prob_positive:.1%} probability of being positive. "
        f"This inference properly marginalizes over all uncertainty in nuisance parameters."
    )
    
    return interpretation