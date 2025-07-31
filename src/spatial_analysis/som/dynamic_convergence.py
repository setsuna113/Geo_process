"""Dynamic convergence detection for SOM training."""

import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import deque

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConvergenceMetrics:
    """Container for convergence metrics."""
    iteration: int
    weight_change: float
    quantization_error: float
    topographic_error: float
    convergence_score: float
    is_converged: bool


class DynamicConvergenceDetector:
    """Dynamic convergence detection for SOM training.
    
    Implements multiple convergence criteria:
    1. Weight stability (change in neuron weights)
    2. Quantization error (mapping accuracy)
    3. Topographic error (topology preservation)
    4. Convergence index (combined metric)
    """
    
    def __init__(
        self,
        weight_threshold: float = 1e-6,
        quantization_error_threshold: float = 1e-6,
        topographic_error_threshold: float = 0.01,
        check_interval: int = 100,
        min_iterations: int = 500,
        patience: int = 5,
        convergence_window: int = 10
    ):
        """Initialize convergence detector.
        
        Args:
            weight_threshold: Maximum allowed weight change for convergence
            quantization_error_threshold: Maximum QE change for convergence
            topographic_error_threshold: Maximum topographic error for convergence
            check_interval: Check convergence every N iterations
            min_iterations: Minimum iterations before checking convergence
            patience: Number of checks without improvement before stopping
            convergence_window: Window size for moving averages
        """
        self.weight_threshold = weight_threshold
        self.quantization_error_threshold = quantization_error_threshold
        self.topographic_error_threshold = topographic_error_threshold
        self.check_interval = check_interval
        self.min_iterations = min_iterations
        self.patience = patience
        self.convergence_window = convergence_window
        
        # State tracking
        self.previous_weights: Optional[np.ndarray] = None
        self.convergence_history: List[ConvergenceMetrics] = []
        self.weight_changes: deque = deque(maxlen=convergence_window)
        self.quantization_errors: deque = deque(maxlen=convergence_window)
        self.topographic_errors: deque = deque(maxlen=convergence_window)
        
        # Convergence tracking
        self.converged_checks = 0
        self.best_convergence_score = float('inf')
        self.iterations_without_improvement = 0
        
        logger.info(f"Initialized dynamic convergence detector:")
        logger.info(f"  Weight threshold: {weight_threshold}")
        logger.info(f"  QE threshold: {quantization_error_threshold}")
        logger.info(f"  Topographic threshold: {topographic_error_threshold}")
        logger.info(f"  Check interval: {check_interval}")
        logger.info(f"  Patience: {patience}")
    
    def should_check_convergence(self, iteration: int) -> bool:
        """Check if convergence should be evaluated at this iteration."""
        return (iteration >= self.min_iterations and 
                iteration % self.check_interval == 0)
    
    def check_convergence(
        self,
        som,
        data: np.ndarray,
        iteration: int
    ) -> Tuple[bool, ConvergenceMetrics]:
        """Check convergence based on multiple criteria.
        
        Args:
            som: MiniSom instance
            data: Training data
            iteration: Current iteration
            
        Returns:
            Tuple of (is_converged, convergence_metrics)
        """
        # Get current weights
        current_weights = som.get_weights().copy()
        
        # Calculate weight change
        weight_change = self._calculate_weight_change(current_weights)
        
        # Calculate quantization error
        quantization_error = self._calculate_quantization_error(som, data)
        
        # Calculate topographic error
        topographic_error = self._calculate_topographic_error(som, data)
        
        # Calculate convergence score (combined metric)
        convergence_score = self._calculate_convergence_score(
            weight_change, quantization_error, topographic_error
        )
        
        # Update moving averages
        self.weight_changes.append(weight_change)
        self.quantization_errors.append(quantization_error)
        self.topographic_errors.append(topographic_error)
        
        # Check convergence criteria
        is_converged = self._evaluate_convergence(
            weight_change, quantization_error, topographic_error, convergence_score
        )
        
        # Create metrics object
        metrics = ConvergenceMetrics(
            iteration=iteration,
            weight_change=weight_change,
            quantization_error=quantization_error,
            topographic_error=topographic_error,
            convergence_score=convergence_score,
            is_converged=is_converged
        )
        
        # Update convergence history
        self.convergence_history.append(metrics)
        
        # Update previous weights for next check
        self.previous_weights = current_weights
        
        # Log convergence status
        if iteration % (self.check_interval * 5) == 0:  # Log every 5th check
            logger.info(
                f"Iteration {iteration}: Weight Δ={weight_change:.2e}, "
                f"QE={quantization_error:.4f}, TE={topographic_error:.4f}, "
                f"Score={convergence_score:.4f}"
            )
        
        return is_converged, metrics
    
    def _calculate_weight_change(self, current_weights: np.ndarray) -> float:
        """Calculate the change in weights since last check."""
        if self.previous_weights is None:
            return float('inf')  # First check, no previous weights
        
        # Calculate mean absolute difference
        weight_diff = np.abs(current_weights - self.previous_weights)
        return np.mean(weight_diff)
    
    def _calculate_quantization_error(self, som, data: np.ndarray) -> float:
        """Calculate quantization error (average distance to BMU)."""
        total_error = 0.0
        n_samples = len(data)
        
        for sample in data:
            # Find best matching unit
            distances = np.linalg.norm(som.get_weights() - sample, axis=2)
            min_distance = np.min(distances)
            total_error += min_distance
        
        return total_error / n_samples
    
    def _calculate_topographic_error(self, som, data: np.ndarray) -> float:
        """Calculate topographic error (topology preservation)."""
        topographic_errors = 0
        n_samples = len(data)
        
        for sample in data:
            # Find two best matching units
            distances = np.linalg.norm(som.get_weights() - sample, axis=2)
            flat_distances = distances.flatten()
            
            # Get indices of two smallest distances
            bmu_indices = np.argpartition(flat_distances, 2)[:2]
            bmu1_2d = np.unravel_index(bmu_indices[0], distances.shape)
            bmu2_2d = np.unravel_index(bmu_indices[1], distances.shape)
            
            # Check if BMUs are neighbors (distance <= sqrt(2) for 8-connectivity)
            distance_between_bmus = np.sqrt(
                (bmu1_2d[0] - bmu2_2d[0])**2 + (bmu1_2d[1] - bmu2_2d[1])**2
            )
            
            if distance_between_bmus > np.sqrt(2):  # Not neighbors
                topographic_errors += 1
        
        return topographic_errors / n_samples
    
    def _calculate_convergence_score(
        self,
        weight_change: float,
        quantization_error: float,
        topographic_error: float
    ) -> float:
        """Calculate combined convergence score.
        
        Lower scores indicate better convergence.
        """
        # Normalize metrics to [0, 1] range approximately
        normalized_weight_change = min(weight_change / self.weight_threshold, 1.0)
        normalized_qe = min(quantization_error, 1.0)  # QE is already in reasonable range
        normalized_te = min(topographic_error / self.topographic_error_threshold, 1.0)
        
        # Weighted combination (weights can be tuned)
        convergence_score = (
            0.4 * normalized_weight_change +
            0.4 * normalized_qe +
            0.2 * normalized_te
        )
        
        return convergence_score
    
    def _evaluate_convergence(
        self,
        weight_change: float,
        quantization_error: float,
        topographic_error: float,
        convergence_score: float
    ) -> bool:
        """Evaluate if convergence criteria are met."""
        
        # Primary convergence criteria
        weight_converged = weight_change < self.weight_threshold
        qe_stable = len(self.quantization_errors) >= 3 and self._is_stable(self.quantization_errors)
        te_acceptable = topographic_error < self.topographic_error_threshold
        
        # Check if all primary criteria are met
        primary_converged = weight_converged and qe_stable and te_acceptable
        
        if primary_converged:
            self.converged_checks += 1
            logger.debug(f"Convergence criteria met ({self.converged_checks}/{self.patience})")
        else:
            self.converged_checks = 0
        
        # Check improvement in convergence score
        if convergence_score < self.best_convergence_score:
            self.best_convergence_score = convergence_score
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        
        # Converged if primary criteria met for patience number of checks
        # OR if no improvement for too long (early stopping)
        converged = (
            self.converged_checks >= self.patience or
            self.iterations_without_improvement >= self.patience * 2
        )
        
        if converged:
            reason = "criteria met" if self.converged_checks >= self.patience else "no improvement"
            logger.info(f"Convergence detected: {reason}")
        
        return converged
    
    def _is_stable(self, values: deque, threshold: float = 0.01) -> bool:
        """Check if values in deque are stable (low variance)."""
        if len(values) < 3:
            return False
        
        values_array = np.array(values)
        relative_std = np.std(values_array) / (np.mean(values_array) + 1e-8)
        return relative_std < threshold
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get summary of convergence analysis."""
        if not self.convergence_history:
            return {"status": "no_convergence_checks"}
        
        latest = self.convergence_history[-1]
        
        return {
            "total_checks": len(self.convergence_history),
            "final_iteration": latest.iteration,
            "final_weight_change": latest.weight_change,
            "final_quantization_error": latest.quantization_error,
            "final_topographic_error": latest.topographic_error,
            "final_convergence_score": latest.convergence_score,
            "converged": latest.is_converged,
            "best_convergence_score": self.best_convergence_score,
            "convergence_trend": self._analyze_convergence_trend()
        }
    
    def _analyze_convergence_trend(self) -> str:
        """Analyze the trend in convergence metrics."""
        if len(self.convergence_history) < 3:
            return "insufficient_data"
        
        # Look at last few convergence scores
        recent_scores = [m.convergence_score for m in self.convergence_history[-3:]]
        
        if recent_scores[-1] < recent_scores[0]:
            return "improving"
        elif recent_scores[-1] > recent_scores[0]:
            return "degrading"
        else:
            return "stable"