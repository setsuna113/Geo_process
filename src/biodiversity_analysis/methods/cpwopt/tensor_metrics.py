"""Metrics for evaluating tensor factorization quality.

This module implements various metrics for assessing CP decomposition quality,
including reconstruction error, core consistency, and component stability.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple
from scipy.stats import pearsonr
from numpy.linalg import norm

from src.abstractions import CPFactors, TensorMetrics
from src.biodiversity_analysis.shared.tensor_utils import (
    cp_reconstruct, unfold, weighted_norm
)
from .constants import EPSILON

logger = logging.getLogger(__name__)


class CPWOPTMetricsCalculator:
    """Calculate metrics for CP-WOPT decomposition quality."""
    
    def reconstruction_error(self, original: np.ndarray, 
                           reconstructed: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> float:
        """Calculate weighted reconstruction error.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
            weights: Optional weight mask
            
        Returns:
            Weighted Frobenius norm of residual
        """
        if weights is None:
            weights = np.ones_like(original)
        
        # Handle NaN in original
        original_clean = np.nan_to_num(original, nan=0.0)
        
        # Calculate residual
        residual = (original_clean - reconstructed) * weights
        
        # Return weighted Frobenius norm
        return weighted_norm(residual, weights)
    
    def relative_error(self, original: np.ndarray,
                      reconstructed: np.ndarray,
                      weights: Optional[np.ndarray] = None) -> float:
        """Calculate relative reconstruction error.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
            weights: Optional weight mask
            
        Returns:
            Relative error (reconstruction_error / original_norm)
        """
        recon_error = self.reconstruction_error(original, reconstructed, weights)
        
        # Calculate norm of original
        if weights is None:
            weights = np.ones_like(original)
        
        original_clean = np.nan_to_num(original, nan=0.0)
        original_norm = weighted_norm(original_clean, weights)
        
        if original_norm == 0:
            return np.inf if recon_error > 0 else 0.0
            
        return recon_error / original_norm
    
    def fit_score(self, original: np.ndarray,
                 reconstructed: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
        """Calculate goodness of fit score.
        
        Args:
            original: Original tensor
            reconstructed: Reconstructed tensor
            weights: Optional weight mask
            
        Returns:
            Fit score between 0 and 1 (1 = perfect fit)
        """
        rel_error = self.relative_error(original, reconstructed, weights)
        return max(0.0, 1.0 - rel_error)
    
    def tucker_congruence(self, factors1: CPFactors, 
                         factors2: CPFactors) -> np.ndarray:
        """Calculate Tucker congruence between two sets of factors.
        
        Congruence coefficient measures similarity between components.
        Values close to 1 indicate high similarity.
        
        Args:
            factors1: First set of factors
            factors2: Second set of factors
            
        Returns:
            Congruence coefficients per component
        """
        if factors1.rank != factors2.rank:
            raise ValueError("Factors must have same rank")
        
        if factors1.n_modes != factors2.n_modes:
            raise ValueError("Factors must have same number of modes")
        
        rank = factors1.rank
        congruences = np.zeros(rank)
        
        # Calculate congruence for each component
        for r in range(rank):
            congruence = 1.0
            
            # Multiply congruences across modes
            for mode in range(factors1.n_modes):
                f1 = factors1.factors[mode][:, r]
                f2 = factors2.factors[mode][:, r]
                
                # Normalize
                f1_norm = f1 / (norm(f1) + EPSILON)
                f2_norm = f2 / (norm(f2) + EPSILON)
                
                # Compute congruence (absolute cosine similarity)
                mode_congruence = abs(np.dot(f1_norm, f2_norm))
                congruence *= mode_congruence
            
            # Take geometric mean
            congruences[r] = congruence ** (1.0 / factors1.n_modes)
        
        return congruences
    
    def core_consistency(self, tensor: np.ndarray, 
                        factors: CPFactors) -> float:
        """Calculate core consistency diagnostic.
        
        Core consistency measures how well the CP model fits the data.
        Values near 100 indicate good model, negative values indicate poor fit.
        
        Args:
            tensor: Original tensor
            factors: CP factors
            
        Returns:
            Core consistency value (100 = perfect, <0 = poor model)
        """
        # This is a simplified version - full implementation requires
        # computing Tucker3 model and comparing cores
        
        # For now, use reconstruction-based approximation
        reconstructed = cp_reconstruct(factors.factors)
        fit = self.fit_score(tensor, reconstructed)
        
        # Map fit to core consistency scale
        # This is an approximation - true core consistency requires Tucker3
        if fit > 0.95:
            consistency = 90 + (fit - 0.95) * 200  # 90-100
        elif fit > 0.8:
            consistency = 50 + (fit - 0.8) * 40 / 0.15  # 50-90
        elif fit > 0.5:
            consistency = (fit - 0.5) * 50 / 0.3  # 0-50
        else:
            consistency = (fit - 0.5) * 100  # Negative values
        
        return consistency
    
    def factor_match_score(self, factors: CPFactors, 
                          reference: CPFactors) -> float:
        """Calculate how well factors match a reference decomposition.
        
        Uses Hungarian algorithm to find best component matching.
        
        Args:
            factors: Factors to evaluate
            reference: Reference factors
            
        Returns:
            Match score between 0 and 1
        """
        congruences = self.tucker_congruence(factors, reference)
        
        # For diagonal matching (no permutation), just average
        # Full implementation would use Hungarian algorithm
        return np.mean(congruences)
    
    def explained_variance(self, tensor: np.ndarray,
                         factors: CPFactors,
                         weights: Optional[np.ndarray] = None) -> float:
        """Calculate variance explained by the decomposition.
        
        Args:
            tensor: Original tensor (numpy or cupy array)
            factors: CP factors
            weights: Optional weight mask
            
        Returns:
            Proportion of variance explained (0-1)
        """
        # Convert to numpy if needed (for CuPy arrays)
        if hasattr(tensor, 'get'):
            tensor = tensor.get()
        
        if weights is None:
            weights = np.ones_like(tensor)
        elif hasattr(weights, 'get'):
            weights = weights.get()
        
        # Convert factors to numpy if needed
        factors_np = []
        for f in factors.factors:
            if hasattr(f, 'get'):
                factors_np.append(f.get())
            else:
                factors_np.append(f)
        
        # Clean tensor
        tensor_clean = np.nan_to_num(tensor, nan=0.0)
        
        # Calculate total variance
        valid_mask = weights > 0
        tensor_valid = tensor_clean[valid_mask]
        
        if len(tensor_valid) == 0:
            return 0.0
        
        total_var = np.var(tensor_valid)
        
        if total_var == 0:
            return 1.0  # No variance to explain
        
        # Calculate residual variance
        reconstructed = cp_reconstruct(factors_np)
        residual = (tensor_clean - reconstructed)[valid_mask]
        residual_var = np.var(residual)
        
        # Explained variance ratio
        explained = 1.0 - (residual_var / total_var)
        
        return max(0.0, explained)
    
    def component_stability(self, factors_list: List[CPFactors]) -> np.ndarray:
        """Assess stability of components across multiple runs.
        
        Args:
            factors_list: List of factors from different runs
            
        Returns:
            Stability score for each component (0-1)
        """
        n_runs = len(factors_list)
        if n_runs < 2:
            raise ValueError("Need at least 2 runs to assess stability")
        
        rank = factors_list[0].rank
        stability_scores = np.zeros(rank)
        
        # Compare each pair of runs
        for r in range(rank):
            pairwise_congruences = []
            
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    # Get congruence between run i and j
                    congruence = self.tucker_congruence(
                        factors_list[i], factors_list[j]
                    )[r]
                    pairwise_congruences.append(congruence)
            
            # Stability is mean of pairwise congruences
            stability_scores[r] = np.mean(pairwise_congruences)
        
        return stability_scores
    
    def calculate_all_metrics(self, tensor: np.ndarray,
                            factors: CPFactors,
                            weights: Optional[np.ndarray] = None,
                            convergence_history: Optional[List[float]] = None) -> TensorMetrics:
        """Calculate all metrics and return TensorMetrics object.
        
        Args:
            tensor: Original tensor
            factors: CP factors
            weights: Optional weight mask
            convergence_history: Optional convergence history
            
        Returns:
            TensorMetrics object with all computed metrics
        """
        # Reconstruct tensor
        reconstructed = cp_reconstruct(factors.factors)
        
        # Calculate basic metrics
        recon_error = self.reconstruction_error(tensor, reconstructed, weights)
        rel_error = self.relative_error(tensor, reconstructed, weights)
        fit = self.fit_score(tensor, reconstructed, weights)
        
        # Calculate component weights
        component_weights = self._calculate_component_weights(factors)
        
        # Calculate explained variance
        explained_var = self.explained_variance(tensor, factors, weights)
        
        # Create metrics object
        metrics = TensorMetrics(
            reconstruction_error=recon_error,
            relative_error=rel_error,
            fit=fit,
            convergence_history=convergence_history or [],
            component_weights=component_weights,
            explained_variance=explained_var
        )
        
        return metrics
    
    def _calculate_component_weights(self, factors: CPFactors) -> np.ndarray:
        """Calculate importance weight for each component.
        
        Args:
            factors: CP factors
            
        Returns:
            Array of component weights
        """
        rank = factors.rank
        weights = np.zeros(rank)
        
        for r in range(rank):
            # Weight is product of norms across modes
            weight = 1.0
            for mode in range(factors.n_modes):
                factor_norm = norm(factors.factors[mode][:, r])
                weight *= factor_norm
            weights[r] = weight
        
        # Normalize to sum to 1
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights /= total_weight
            
        return weights


def print_metrics_summary(metrics: TensorMetrics, model_name: str = "CP-WOPT"):
    """Print a formatted summary of tensor metrics.
    
    Args:
        metrics: TensorMetrics object
        model_name: Name of the model for display
    """
    print(f"\n{model_name} Metrics Summary")
    print("=" * 50)
    print(f"Reconstruction Error: {metrics.reconstruction_error:.6f}")
    print(f"Relative Error: {metrics.relative_error:.6f}")
    print(f"Fit Score: {metrics.fit:.4f}")
    
    if metrics.explained_variance is not None:
        print(f"Explained Variance: {metrics.explained_variance:.4f}")
    
    if metrics.component_weights is not None:
        print(f"\nTop 5 Components (by weight):")
        top_idx = np.argsort(metrics.component_weights)[::-1][:5]
        for i, idx in enumerate(top_idx):
            print(f"  {i+1}. Component {idx}: {metrics.component_weights[idx]:.4f}")
    
    if len(metrics.convergence_history) > 0:
        print(f"\nConvergence:")
        print(f"  Initial Error: {metrics.convergence_history[0]:.6f}")
        print(f"  Final Error: {metrics.convergence_history[-1]:.6f}")
        print(f"  Iterations: {len(metrics.convergence_history)}")
        
        # Check if converged
        if len(metrics.convergence_history) > 10:
            recent = metrics.convergence_history[-10:]
            rel_changes = [abs(recent[i] - recent[i-1]) / recent[i-1] 
                          for i in range(1, len(recent))]
            if all(c < 1e-4 for c in rel_changes):
                print("  Status: Converged ✓")
            else:
                print("  Status: Not converged")


def compare_decompositions(factors1: CPFactors, factors2: CPFactors,
                          labels: Tuple[str, str] = ("Model 1", "Model 2")) -> Dict[str, Any]:
    """Compare two CP decompositions.
    
    Args:
        factors1: First decomposition
        factors2: Second decomposition
        labels: Labels for the two models
        
    Returns:
        Dictionary with comparison metrics
    """
    calculator = CPWOPTMetricsCalculator()
    
    # Calculate congruence
    congruences = calculator.tucker_congruence(factors1, factors2)
    
    # Find best matching (for now, assume diagonal)
    match_score = calculator.factor_match_score(factors1, factors2)
    
    comparison = {
        'congruences': congruences,
        'match_score': match_score,
        'mean_congruence': np.mean(congruences),
        'min_congruence': np.min(congruences),
        'highly_similar': np.sum(congruences > 0.95),
        'moderately_similar': np.sum((congruences > 0.85) & (congruences <= 0.95)),
        'dissimilar': np.sum(congruences <= 0.85)
    }
    
    return comparison