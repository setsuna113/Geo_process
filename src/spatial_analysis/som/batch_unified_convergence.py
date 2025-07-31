"""Combined batch SOM convergence using weight stabilization + unified convergence index."""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from src.infrastructure.logging import get_logger
from .advanced_convergence import AdvancedConvergenceDetector, UnifiedConvergenceIndex

logger = get_logger(__name__)


@dataclass
class BatchConvergenceResult:
    """Result of batch SOM convergence check."""
    is_converged: bool
    weight_stabilized: bool
    quality_converged: bool
    weight_change: float
    unified_result: UnifiedConvergenceIndex
    convergence_reason: str


class BatchUnifiedConvergenceDetector(AdvancedConvergenceDetector):
    """Convergence detector optimized for batch SOM training.
    
    Combines:
    1. Weight stabilization (primary for batch SOM)
    2. Unified convergence index (quality assurance)
    3. Both must be satisfied for true convergence
    """
    
    def __init__(
        self,
        weight_stabilization_threshold: float = 1e-6,
        require_both_criteria: bool = True,
        weight_stabilization_patience: int = 3,
        **kwargs
    ):
        """Initialize batch convergence detector.
        
        Args:
            weight_stabilization_threshold: Threshold for weight change
            require_both_criteria: Require both weight stabilization AND quality
            weight_stabilization_patience: Epochs of stable weights required
            **kwargs: Parameters for unified convergence index
        """
        super().__init__(**kwargs)
        
        self.weight_stabilization_threshold = weight_stabilization_threshold
        self.require_both_criteria = require_both_criteria
        self.weight_stabilization_patience = weight_stabilization_patience
        
        # Batch-specific tracking
        self.previous_batch_weights: Optional[np.ndarray] = None
        self.weight_changes: List[float] = []
        self.stable_weight_epochs = 0
        
        logger.info(f"Initialized batch unified convergence detector:")
        logger.info(f"  Weight stabilization threshold: {weight_stabilization_threshold}")
        logger.info(f"  Require both criteria: {require_both_criteria}")
        logger.info(f"  Weight stabilization patience: {weight_stabilization_patience}")
    
    def check_batch_convergence(
        self,
        som,
        data: np.ndarray,
        epoch: int,
        max_epochs: int
    ) -> BatchConvergenceResult:
        """Check convergence for batch SOM training.
        
        Args:
            som: MiniSom instance
            data: Training data
            epoch: Current epoch
            max_epochs: Maximum epochs
            
        Returns:
            BatchConvergenceResult with detailed convergence information
        """
        # Get current weights
        current_weights = som.get_weights().copy()
        
        # 1. Check weight stabilization
        weight_change = self._calculate_weight_change_batch(current_weights)
        weight_stabilized = self._check_weight_stabilization(weight_change)
        
        # 2. Check unified convergence index (quality)
        unified_result = self.calculate_unified_convergence_index(som, data, epoch)
        quality_converged = unified_result.is_converged
        
        # 3. Determine overall convergence
        if self.require_both_criteria:
            is_converged = weight_stabilized and quality_converged
            convergence_reason = self._get_convergence_reason(weight_stabilized, quality_converged)
        else:
            # Either criterion is sufficient
            is_converged = weight_stabilized or quality_converged
            if weight_stabilized and quality_converged:
                convergence_reason = "both_criteria_met"
            elif weight_stabilized:
                convergence_reason = "weight_stabilized"
            elif quality_converged:
                convergence_reason = "quality_converged"
            else:
                convergence_reason = "not_converged"
        
        # Update state for next epoch
        self.previous_batch_weights = current_weights
        self.weight_changes.append(weight_change)
        
        # Log progress
        if epoch % 10 == 0:  # Log every 10 epochs
            logger.info(
                f"Epoch {epoch}: Weight Δ={weight_change:.2e}, "
                f"Unified={unified_result.convergence_index:.3f}, "
                f"Stable={weight_stabilized}, Quality={quality_converged}"
            )
        
        return BatchConvergenceResult(
            is_converged=is_converged,
            weight_stabilized=weight_stabilized,
            quality_converged=quality_converged,
            weight_change=weight_change,
            unified_result=unified_result,
            convergence_reason=convergence_reason
        )
    
    def _calculate_weight_change_batch(self, current_weights: np.ndarray) -> float:
        """Calculate weight change for batch SOM."""
        if self.previous_batch_weights is None:
            return float('inf')  # First epoch
        
        # Calculate mean absolute change across all weights
        weight_diff = np.abs(current_weights - self.previous_batch_weights)
        return np.mean(weight_diff)
    
    def _check_weight_stabilization(self, weight_change: float) -> bool:
        """Check if weights have stabilized."""
        if weight_change < self.weight_stabilization_threshold:
            self.stable_weight_epochs += 1
        else:
            self.stable_weight_epochs = 0
        
        # Require stable weights for multiple epochs
        return self.stable_weight_epochs >= self.weight_stabilization_patience
    
    def _get_convergence_reason(self, weight_stabilized: bool, quality_converged: bool) -> str:
        """Get human-readable convergence reason."""
        if weight_stabilized and quality_converged:
            return "both_criteria_met"
        elif weight_stabilized and not quality_converged:
            return "weights_stable_quality_insufficient"
        elif not weight_stabilized and quality_converged:
            return "quality_good_weights_changing"
        else:
            return "neither_criterion_met"
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary of batch convergence analysis."""
        base_summary = self.get_advanced_summary()
        
        batch_summary = {
            'training_method': 'batch_som',
            'weight_stabilization_threshold': self.weight_stabilization_threshold,
            'stable_weight_epochs': self.stable_weight_epochs,
            'final_weight_change': self.weight_changes[-1] if self.weight_changes else None,
            'weight_change_trend': self._analyze_weight_change_trend(),
            'require_both_criteria': self.require_both_criteria
        }
        
        return {**base_summary, **batch_summary}
    
    def _analyze_weight_change_trend(self) -> str:
        """Analyze trend in weight changes."""
        if len(self.weight_changes) < 3:
            return "insufficient_data"
        
        recent_changes = self.weight_changes[-3:]
        if recent_changes[-1] < recent_changes[0]:
            return "decreasing"
        elif recent_changes[-1] > recent_changes[0]:
            return "increasing"
        else:
            return "stable"


def create_batch_convergence_detector(
    weight_threshold: float = 1e-6,
    quality_threshold: float = 0.95,
    require_both: bool = True,
    **kwargs
) -> BatchUnifiedConvergenceDetector:
    """Factory function for batch SOM convergence detector.
    
    Args:
        weight_threshold: Weight stabilization threshold
        quality_threshold: Unified convergence index threshold
        require_both: Require both weight stabilization AND quality
        **kwargs: Additional parameters
        
    Returns:
        Configured batch convergence detector
    """
    return BatchUnifiedConvergenceDetector(
        weight_stabilization_threshold=weight_threshold,
        require_both_criteria=require_both,
        convergence_threshold=quality_threshold,
        unified_convergence=True,
        **kwargs
    )