"""Rank selection methods for CP-WOPT tensor decomposition.

This module implements various methods for automatic rank determination including:
- Greedy rank expansion
- Information criteria (AIC/BIC)
- Elbow method
- Cross-validation
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .core import CPWOPT
from src.abstractions import TensorData

logger = logging.getLogger(__name__)


@dataclass
class RankSelectionResult:
    """Results from rank selection process."""
    optimal_rank: int
    ranks_tested: List[int]
    fits: List[float]
    aic_scores: Optional[List[float]] = None
    bic_scores: Optional[List[float]] = None
    method_used: str = "greedy"
    

def compute_aic(reconstruction_error: float, rank: int, tensor_shape: Tuple[int, ...]) -> float:
    """Compute Akaike Information Criterion for CP decomposition.
    
    Args:
        reconstruction_error: Squared Frobenius norm of error
        rank: CP rank used
        tensor_shape: Shape of the tensor
        
    Returns:
        AIC score (lower is better)
    """
    # Number of parameters in CP decomposition
    n_params = rank * sum(tensor_shape)
    
    # Number of observations
    n_obs = np.prod(tensor_shape)
    
    # AIC = n*log(RSS/n) + 2k
    # Where RSS is residual sum of squares
    aic = n_obs * np.log(reconstruction_error / n_obs) + 2 * n_params
    
    return aic


def compute_bic(reconstruction_error: float, rank: int, tensor_shape: Tuple[int, ...]) -> float:
    """Compute Bayesian Information Criterion for CP decomposition.
    
    Args:
        reconstruction_error: Squared Frobenius norm of error
        rank: CP rank used
        tensor_shape: Shape of the tensor
        
    Returns:
        BIC score (lower is better)
    """
    # Number of parameters
    n_params = rank * sum(tensor_shape)
    
    # Number of observations
    n_obs = np.prod(tensor_shape)
    
    # BIC = n*log(RSS/n) + k*log(n)
    bic = n_obs * np.log(reconstruction_error / n_obs) + n_params * np.log(n_obs)
    
    return bic


def fit_cp_with_rank(tensor_data: TensorData, rank: int, 
                     max_iterations: int = 500,  # Reduced for rank selection
                     convergence_threshold: float = 1e-5,  # Looser for rank selection
                     **kwargs) -> Tuple[float, float, float, float]:
    """Fit CP decomposition with specified rank and return metrics.
    
    Args:
        tensor_data: Input tensor data
        rank: CP rank to use
        max_iterations: Maximum iterations for CP-ALS
        convergence_threshold: Convergence threshold
        **kwargs: Additional arguments for CPWOPT
        
    Returns:
        Tuple of (fit, reconstruction_error, aic, bic)
    """
    logger.info(f"Fitting CP with rank={rank}")
    
    # Extract use_gpu if provided, otherwise use 'auto'
    use_gpu = kwargs.pop('use_gpu', 'auto')
    
    # Create and fit model
    model = CPWOPT(
        rank=rank,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        verbose=False,  # Quiet for multiple runs
        use_gpu=use_gpu,  # Pass GPU config
        **kwargs
    )
    
    # Fit the model
    model.fit(tensor_data)
    
    # Get metrics
    fit = model.metrics_.fit
    reconstruction_error = model.metrics_.reconstruction_error
    
    # Compute information criteria
    aic = compute_aic(reconstruction_error, rank, tensor_data.shape)
    bic = compute_bic(reconstruction_error, rank, tensor_data.shape)
    
    logger.info(f"  Rank {rank}: fit={fit:.4f}, AIC={aic:.1f}, BIC={bic:.1f}")
    
    return fit, reconstruction_error, aic, bic


def greedy_rank_expansion(tensor_data: TensorData,
                         min_rank: int = 2,
                         max_rank: int = 30,
                         fit_improvement_threshold: float = 0.01,
                         max_iterations: int = 1000,
                         plot: bool = False,  # Extract plot argument
                         use_gpu: str = 'auto',  # GPU configuration
                         **kwargs) -> RankSelectionResult:
    """Select rank using greedy expansion with early stopping.
    
    Starts with low rank and increases until improvement is minimal.
    
    Args:
        tensor_data: Input tensor data
        min_rank: Starting rank
        max_rank: Maximum rank to test
        fit_improvement_threshold: Stop if improvement < this
        max_iterations: Max iterations per CP fit
        **kwargs: Additional arguments for CPWOPT
        
    Returns:
        RankSelectionResult with optimal rank and metrics
    """
    logger.info(f"Starting greedy rank expansion from {min_rank} to {max_rank}")
    
    ranks_tested = []
    fits = []
    aic_scores = []
    bic_scores = []
    
    prev_fit = 0.0
    optimal_rank = min_rank
    
    for rank in range(min_rank, max_rank + 1):
        # Fit model with this rank (filter out non-CPWOPT kwargs)
        cpwopt_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['plot', 'output_dir', 'fit_improvement_threshold']}
        fit, error, aic, bic = fit_cp_with_rank(
            tensor_data, rank, max_iterations, use_gpu=use_gpu, **cpwopt_kwargs
        )
        
        ranks_tested.append(rank)
        fits.append(fit)
        aic_scores.append(aic)
        bic_scores.append(bic)
        
        # Check improvement
        improvement = fit - prev_fit
        logger.info(f"Rank {rank}: fit={fit:.4f}, improvement={improvement:.4f}")
        
        # Update optimal based on BIC (more conservative)
        if len(bic_scores) == 1 or bic < min(bic_scores[:-1]):
            optimal_rank = rank
        
        # Early stopping if improvement is small
        if rank > min_rank and improvement < fit_improvement_threshold:
            logger.info(f"Stopping early - improvement below threshold")
            break
            
        prev_fit = fit
    
    return RankSelectionResult(
        optimal_rank=optimal_rank,
        ranks_tested=ranks_tested,
        fits=fits,
        aic_scores=aic_scores,
        bic_scores=bic_scores,
        method_used="greedy_expansion"
    )


def elbow_method(tensor_data: TensorData,
                 min_rank: int = 2,
                 max_rank: int = 30,
                 rank_step: int = 2,
                 plot: bool = True,
                 output_dir: Optional[Path] = None,
                 use_gpu: str = 'auto',
                 **kwargs) -> RankSelectionResult:
    """Select rank using elbow method on fit vs rank curve.
    
    Args:
        tensor_data: Input tensor data
        min_rank: Minimum rank to test
        max_rank: Maximum rank to test
        rank_step: Step size for rank testing
        plot: Whether to generate plot
        output_dir: Directory to save plot
        **kwargs: Additional arguments for CPWOPT
        
    Returns:
        RankSelectionResult with optimal rank
    """
    logger.info(f"Running elbow method from rank {min_rank} to {max_rank}")
    
    ranks_tested = []
    fits = []
    aic_scores = []
    bic_scores = []
    
    # Test different ranks
    for rank in range(min_rank, max_rank + 1, rank_step):
        # Filter out non-CPWOPT kwargs
        cpwopt_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['plot', 'output_dir', 'rank_step']}
        fit, error, aic, bic = fit_cp_with_rank(
            tensor_data, rank, use_gpu=use_gpu, **cpwopt_kwargs
        )
        
        ranks_tested.append(rank)
        fits.append(fit)
        aic_scores.append(aic)
        bic_scores.append(bic)
    
    # Find elbow using second derivative
    if len(fits) > 2:
        # Compute second derivative of fit curve
        first_deriv = np.diff(fits)
        second_deriv = np.diff(first_deriv)
        
        # Find where improvement starts diminishing
        elbow_idx = np.argmin(second_deriv) + 1  # +1 because of diff
        optimal_rank = ranks_tested[min(elbow_idx, len(ranks_tested)-1)]
    else:
        # Use BIC if not enough points
        optimal_rank = ranks_tested[np.argmin(bic_scores)]
    
    # Generate plot if requested
    if plot and output_dir:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Fit vs Rank
        ax = axes[0, 0]
        ax.plot(ranks_tested, fits, 'b-o')
        ax.axvline(optimal_rank, color='r', linestyle='--', label=f'Selected: {optimal_rank}')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Fit')
        ax.set_title('Fit Quality vs Rank')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # AIC vs Rank
        ax = axes[0, 1]
        ax.plot(ranks_tested, aic_scores, 'g-s')
        ax.axvline(ranks_tested[np.argmin(aic_scores)], color='r', linestyle='--')
        ax.set_xlabel('Rank')
        ax.set_ylabel('AIC')
        ax.set_title('AIC vs Rank (lower is better)')
        ax.grid(True, alpha=0.3)
        
        # BIC vs Rank
        ax = axes[1, 0]
        ax.plot(ranks_tested, bic_scores, 'm-^')
        ax.axvline(ranks_tested[np.argmin(bic_scores)], color='r', linestyle='--')
        ax.set_xlabel('Rank')
        ax.set_ylabel('BIC')
        ax.set_title('BIC vs Rank (lower is better)')
        ax.grid(True, alpha=0.3)
        
        # Improvement vs Rank
        ax = axes[1, 1]
        improvements = np.diff(fits)
        ax.plot(ranks_tested[1:], improvements, 'c-d')
        ax.axhline(0.01, color='orange', linestyle=':', label='Threshold')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Fit Improvement')
        ax.set_title('Marginal Improvement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rank_selection_analysis.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved rank selection plot to {output_dir / 'rank_selection_analysis.png'}")
    
    return RankSelectionResult(
        optimal_rank=optimal_rank,
        ranks_tested=ranks_tested,
        fits=fits,
        aic_scores=aic_scores,
        bic_scores=bic_scores,
        method_used="elbow"
    )


def auto_select_rank(tensor_data: TensorData,
                    method: str = 'greedy',
                    min_rank: int = 2,
                    max_rank: int = 30,
                    plot: bool = True,
                    output_dir: Optional[Path] = None,
                    use_gpu: str = 'auto',
                    **kwargs) -> int:
    """Automatically select optimal CP rank.
    
    This is the main entry point called by analyzer_adapter.
    
    Args:
        tensor_data: Input tensor data
        method: Selection method ('greedy', 'elbow', 'bic')
        min_rank: Minimum rank to consider
        max_rank: Maximum rank to consider
        plot: Whether to generate diagnostic plots
        output_dir: Directory for output plots
        **kwargs: Additional arguments passed to methods
        
    Returns:
        Optimal rank value
    """
    logger.info(f"Auto-selecting rank using method: {method}")
    logger.info(f"Tensor shape: {tensor_data.shape}, sparsity: {tensor_data.sparsity:.2%}")
    
    # Ensure output directory exists
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run appropriate method
    if method == 'greedy':
        result = greedy_rank_expansion(
            tensor_data, min_rank, max_rank, 
            plot=plot, use_gpu=use_gpu, **kwargs
        )
    elif method == 'elbow':
        result = elbow_method(
            tensor_data, min_rank, max_rank,
            plot=plot, output_dir=output_dir, use_gpu=use_gpu, **kwargs
        )
    elif method == 'bic':
        # Use elbow method but select based on BIC
        result = elbow_method(
            tensor_data, min_rank, max_rank,
            plot=plot, output_dir=output_dir, use_gpu=use_gpu, **kwargs
        )
        # Override with BIC selection
        if result.bic_scores:
            result.optimal_rank = result.ranks_tested[np.argmin(result.bic_scores)]
    else:
        logger.warning(f"Unknown method {method}, using greedy")
        result = greedy_rank_expansion(
            tensor_data, min_rank, max_rank, use_gpu=use_gpu, **kwargs
        )
    
    logger.info(f"Selected optimal rank: {result.optimal_rank}")
    logger.info(f"Tested ranks: {result.ranks_tested}")
    logger.info(f"Final fits: {[f'{f:.4f}' for f in result.fits]}")
    
    # Save results summary
    if output_dir:
        import json
        summary = {
            'optimal_rank': result.optimal_rank,
            'method': result.method_used,
            'ranks_tested': result.ranks_tested,
            'fits': result.fits,
            'aic_scores': result.aic_scores,
            'bic_scores': result.bic_scores
        }
        with open(output_dir / 'rank_selection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    return result.optimal_rank