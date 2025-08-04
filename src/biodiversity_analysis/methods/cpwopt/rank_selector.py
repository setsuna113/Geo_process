"""Automatic rank selection for CP-WOPT tensor decomposition.

This module implements various methods for automatically determining the optimal
rank for CP decomposition, including elbow method, cross-validation, and
core consistency diagnostics.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path

from src.abstractions.types.tensor_types import TensorData
from .core import CPWOPT

logger = logging.getLogger(__name__)


@dataclass
class RankSelectionResult:
    """Result of rank selection process."""
    optimal_rank: int
    rank_scores: Dict[int, float]
    selection_method: str
    metadata: Dict[str, Any]


class CPWOPTRankSelector:
    """Automatic rank selection for CP-WOPT decomposition."""
    
    def __init__(self, 
                 min_rank: int = 2,
                 max_rank: int = 20,
                 n_trials: int = 3,
                 selection_method: str = 'elbow',
                 convergence_threshold: float = 1e-6,
                 max_iterations: int = 100):
        """Initialize rank selector.
        
        Args:
            min_rank: Minimum rank to test
            max_rank: Maximum rank to test
            n_trials: Number of trials per rank (for stability)
            selection_method: Method for selection ('elbow', 'cross_validation', 'core_consistency')
            convergence_threshold: Convergence threshold for CP-WOPT
            max_iterations: Maximum iterations for each decomposition
        """
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.n_trials = n_trials
        self.selection_method = selection_method
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        
    def select_rank(self, tensor_data: TensorData, 
                   verbose: bool = True) -> RankSelectionResult:
        """Automatically select optimal rank for tensor decomposition.
        
        Args:
            tensor_data: Input tensor data
            verbose: Whether to log progress
            
        Returns:
            RankSelectionResult with optimal rank and diagnostics
        """
        if self.selection_method == 'elbow':
            return self._elbow_method(tensor_data, verbose)
        elif self.selection_method == 'cross_validation':
            return self._cross_validation_method(tensor_data, verbose)
        elif self.selection_method == 'core_consistency':
            return self._core_consistency_method(tensor_data, verbose)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _elbow_method(self, tensor_data: TensorData, 
                     verbose: bool = True) -> RankSelectionResult:
        """Select rank using elbow method on reconstruction error.
        
        The elbow method looks for the "elbow" in the curve of reconstruction
        error vs rank, where adding more components provides diminishing returns.
        """
        if verbose:
            logger.info(f"Running elbow method for ranks {self.min_rank}-{self.max_rank}")
        
        rank_scores = {}
        rank_times = {}
        
        # Test each rank
        # OPTIMIZATION: Use fewer iterations for initial screening
        screening_iterations = min(50, self.max_iterations // 2)
        
        for rank in range(self.min_rank, self.max_rank + 1):
            if verbose:
                logger.info(f"Testing rank {rank}...")
            
            # Run multiple trials for stability
            fits = []
            times = []
            
            # OPTIMIZATION: Use only 1 trial for initial screening if many ranks
            n_trials_to_use = 1 if (self.max_rank - self.min_rank) > 10 else self.n_trials
            
            for trial in range(n_trials_to_use):
                model = CPWOPT(
                    rank=rank,
                    max_iterations=screening_iterations,  # Reduced iterations
                    convergence_threshold=self.convergence_threshold * 10,  # Looser convergence
                    random_state=trial,
                    verbose=False
                )
                
                try:
                    model.fit(tensor_data)
                    fits.append(model.metrics_.fit)
                    times.append(model.training_time_)
                except Exception as e:
                    logger.warning(f"Failed to fit rank {rank}, trial {trial}: {e}")
                    fits.append(0.0)
                    times.append(np.inf)
            
            # Store average fit
            rank_scores[rank] = np.mean(fits)
            rank_times[rank] = np.mean(times)
            
            if verbose:
                logger.info(f"Rank {rank}: fit={rank_scores[rank]:.4f}, time={rank_times[rank]:.2f}s")
        
        # Find elbow using second derivative
        optimal_rank = self._find_elbow(list(rank_scores.keys()), 
                                       list(rank_scores.values()))
        
        # Create result
        result = RankSelectionResult(
            optimal_rank=optimal_rank,
            rank_scores=rank_scores,
            selection_method='elbow',
            metadata={
                'rank_times': rank_times,
                'n_trials': self.n_trials
            }
        )
        
        if verbose:
            logger.info(f"Optimal rank selected: {optimal_rank}")
        
        return result
    
    def _cross_validation_method(self, tensor_data: TensorData,
                                verbose: bool = True) -> RankSelectionResult:
        """Select rank using cross-validation on tensor entries.
        
        This method randomly holds out entries of the tensor and evaluates
        how well different ranks can predict the held-out values.
        """
        if verbose:
            logger.info("Running cross-validation rank selection")
        
        # Create random mask for validation (hold out 20% of non-zero entries)
        validation_mask = self._create_validation_mask(tensor_data, fraction=0.2)
        
        rank_scores = {}
        
        for rank in range(self.min_rank, self.max_rank + 1):
            if verbose:
                logger.info(f"Cross-validating rank {rank}...")
            
            # Create training data with held-out entries
            train_tensor = tensor_data.tensor.copy()
            train_weights = tensor_data.weights.copy()
            train_weights[validation_mask] = 0
            
            train_data = TensorData(
                tensor=train_tensor,
                weights=train_weights,
                shape=tensor_data.shape,
                sparsity=tensor_data.sparsity,
                ndim=tensor_data.ndim
            )
            
            # Fit model
            model = CPWOPT(
                rank=rank,
                max_iterations=self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                verbose=False
            )
            
            try:
                model.fit(train_data)
                
                # Evaluate on held-out entries
                reconstructed = model.predict()
                validation_error = np.sqrt(np.mean(
                    (tensor_data.tensor[validation_mask] - reconstructed[validation_mask])**2
                ))
                
                rank_scores[rank] = -validation_error  # Negative because we maximize
                
            except Exception as e:
                logger.warning(f"Cross-validation failed for rank {rank}: {e}")
                rank_scores[rank] = -np.inf
        
        # Select rank with best validation score
        optimal_rank = max(rank_scores, key=rank_scores.get)
        
        return RankSelectionResult(
            optimal_rank=optimal_rank,
            rank_scores=rank_scores,
            selection_method='cross_validation',
            metadata={'validation_fraction': 0.2}
        )
    
    def _core_consistency_method(self, tensor_data: TensorData,
                               verbose: bool = True) -> RankSelectionResult:
        """Select rank using core consistency diagnostic (CORCONDIA).
        
        Core consistency measures how well the CP model approximates the
        data compared to a Tucker model with the same rank.
        """
        if verbose:
            logger.info("Running core consistency diagnostic")
        
        rank_scores = {}
        
        for rank in range(self.min_rank, self.max_rank + 1):
            if verbose:
                logger.info(f"Computing core consistency for rank {rank}...")
            
            # Fit CP model
            model = CPWOPT(
                rank=rank,
                max_iterations=self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                verbose=False
            )
            
            try:
                model.fit(tensor_data)
                
                # Compute core consistency
                consistency = self._compute_core_consistency(
                    tensor_data.tensor,
                    model.factors_.factors,
                    tensor_data.weights
                )
                
                rank_scores[rank] = consistency
                
                if verbose:
                    logger.info(f"Rank {rank}: consistency={consistency:.2f}%")
                
                # Stop if consistency drops below threshold
                if consistency < 50:
                    logger.info(f"Core consistency dropped below 50% at rank {rank}")
                    break
                    
            except Exception as e:
                logger.warning(f"Core consistency failed for rank {rank}: {e}")
                rank_scores[rank] = 0.0
        
        # Select highest rank with good consistency (>80%)
        good_ranks = [r for r, c in rank_scores.items() if c > 80]
        if good_ranks:
            optimal_rank = max(good_ranks)
        else:
            # If no rank has good consistency, use the one with highest score
            optimal_rank = max(rank_scores, key=rank_scores.get)
        
        return RankSelectionResult(
            optimal_rank=optimal_rank,
            rank_scores=rank_scores,
            selection_method='core_consistency',
            metadata={'consistency_threshold': 80}
        )
    
    def _find_elbow(self, ranks: List[int], scores: List[float]) -> int:
        """Find elbow point in curve using angle-based method.
        
        Args:
            ranks: List of rank values
            scores: List of corresponding scores
            
        Returns:
            Optimal rank at elbow point
        """
        if len(ranks) < 3:
            return ranks[len(ranks)//2]
        
        # Normalize scores to [0, 1]
        scores = np.array(scores)
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        # Calculate angles between consecutive segments
        angles = []
        for i in range(1, len(ranks) - 1):
            # Vectors from point i-1 to i and from i to i+1
            v1 = np.array([ranks[i] - ranks[i-1], scores_norm[i] - scores_norm[i-1]])
            v2 = np.array([ranks[i+1] - ranks[i], scores_norm[i+1] - scores_norm[i]])
            
            # Angle between vectors
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            angles.append(angle)
        
        # Find maximum angle (sharpest elbow)
        if angles:
            elbow_idx = np.argmax(angles) + 1
            return ranks[elbow_idx]
        else:
            # Default to middle rank
            return ranks[len(ranks)//2]
    
    def _create_validation_mask(self, tensor_data: TensorData, 
                              fraction: float = 0.2) -> np.ndarray:
        """Create random mask for cross-validation.
        
        Args:
            tensor_data: Input tensor
            fraction: Fraction of entries to hold out
            
        Returns:
            Boolean mask with True for validation entries
        """
        # Only hold out non-zero entries
        non_zero_mask = tensor_data.weights > 0
        non_zero_indices = np.where(non_zero_mask)
        
        n_non_zero = len(non_zero_indices[0])
        n_validate = int(n_non_zero * fraction)
        
        # Random selection
        validate_idx = np.random.choice(n_non_zero, n_validate, replace=False)
        
        # Create mask
        validation_mask = np.zeros_like(tensor_data.weights, dtype=bool)
        for idx in validate_idx:
            coords = tuple(indices[idx] for indices in non_zero_indices)
            validation_mask[coords] = True
        
        return validation_mask
    
    def _compute_core_consistency(self, tensor: np.ndarray,
                                factors: List[np.ndarray],
                                weights: np.ndarray) -> float:
        """Compute CORCONDIA core consistency diagnostic.
        
        Core consistency measures how well a CP model explains the data
        compared to an unconstrained Tucker model.
        
        Args:
            tensor: Original tensor
            factors: CP factor matrices
            weights: Weight mask
            
        Returns:
            Core consistency percentage (0-100)
        """
        # This is a simplified version - full CORCONDIA requires Tucker decomposition
        # For now, return a proxy based on reconstruction quality
        
        # Reconstruct tensor
        reconstructed = factors[0]
        for i in range(1, len(factors)):
            reconstructed = np.multiply.outer(reconstructed, factors[i])
        
        # Compute weighted reconstruction error
        error = np.sum(weights * (tensor - reconstructed)**2)
        total = np.sum(weights * tensor**2)
        
        # Convert to consistency score (100% = perfect)
        consistency = 100 * (1 - error / (total + 1e-10))
        
        return max(0, min(100, consistency))
    
    def plot_rank_selection(self, result: RankSelectionResult, 
                          save_path: Optional[Path] = None) -> None:
        """Plot rank selection diagnostic.
        
        Args:
            result: RankSelectionResult from selection process
            save_path: Optional path to save plot
        """
        ranks = sorted(result.rank_scores.keys())
        scores = [result.rank_scores[r] for r in ranks]
        
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, scores, 'b.-', markersize=10)
        plt.axvline(result.optimal_rank, color='r', linestyle='--', 
                   label=f'Optimal rank: {result.optimal_rank}')
        
        plt.xlabel('Rank')
        if result.selection_method == 'elbow':
            plt.ylabel('Fit Score')
            plt.title('Rank Selection: Elbow Method')
        elif result.selection_method == 'cross_validation':
            plt.ylabel('Validation Score')
            plt.title('Rank Selection: Cross-Validation')
        elif result.selection_method == 'core_consistency':
            plt.ylabel('Core Consistency (%)')
            plt.title('Rank Selection: CORCONDIA')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def auto_select_rank(tensor_data: TensorData,
                    method: str = 'elbow',
                    min_rank: int = 2,
                    max_rank: int = 15,
                    plot: bool = True,
                    output_dir: Optional[Path] = None) -> int:
    """Convenience function for automatic rank selection.
    
    Args:
        tensor_data: Input tensor
        method: Selection method
        min_rank: Minimum rank to test
        max_rank: Maximum rank to test  
        plot: Whether to create diagnostic plot
        output_dir: Directory for saving plot
        
    Returns:
        Optimal rank
    """
    # Adjust max_rank based on tensor dimensions
    max_possible_rank = min(tensor_data.shape)
    if max_rank > max_possible_rank:
        logger.warning(f"Reducing max_rank from {max_rank} to {max_possible_rank} "
                      f"(limited by tensor dimensions)")
        max_rank = max_possible_rank
    
    # Create selector
    selector = CPWOPTRankSelector(
        min_rank=min_rank,
        max_rank=max_rank,
        selection_method=method
    )
    
    # Run selection
    result = selector.select_rank(tensor_data)
    
    # Create plot if requested
    if plot and output_dir:
        plot_path = output_dir / f'rank_selection_{method}.png'
        selector.plot_rank_selection(result, plot_path)
    
    return result.optimal_rank