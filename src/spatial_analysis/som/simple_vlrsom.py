"""
Simple VLRSOM Implementation Following Real Research Pattern

Based on actual VLRSOM papers (2020-2024):
- Primary loss: Quantization Error (QE)
- Secondary monitoring: Topographic Error (TE)  
- Simple dual-threshold convergence: QE < threshold AND TE < threshold
- Variable learning rate based on QE improvement trend
- No complex regularization or invented metrics
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class VLRSOMResult:
    """Simple result container for VLRSOM training."""
    converged: bool
    final_quantization_error: float
    final_topographic_error: float
    total_iterations: int
    training_time: float
    learning_rate_history: List[float]
    qe_history: List[float]
    te_history: List[float]


class SimpleVLRSOM:
    """
    Simple Variable Learning Rate SOM following real research pattern.
    
    Key principles from actual VLRSOM papers:
    1. Quantization Error as primary objective
    2. Topographic Error for monitoring topology preservation
    3. Simple dual-threshold convergence check
    4. Adaptive learning rate based on QE improvement
    5. Manhattan distance for biodiversity data
    """
    
    def __init__(self,
                 som,  # Manhattan SOM instance
                 initial_learning_rate: float = 0.5,
                 min_learning_rate: float = 0.01,
                 max_learning_rate: float = 0.9,
                 qe_threshold: float = 1e-6,
                 te_threshold: float = 0.05,
                 max_iterations: int = 2000,
                 patience: int = 50,
                 learning_rate_factor: float = 0.1):
        """
        Initialize Simple VLRSOM trainer.
        
        Args:
            som: Manhattan SOM instance to train
            initial_learning_rate: Starting learning rate
            min_learning_rate: Minimum learning rate
            max_learning_rate: Maximum learning rate  
            qe_threshold: Quantization error convergence threshold
            te_threshold: Topographic error threshold
            max_iterations: Maximum training iterations
            patience: Iterations without improvement before stopping
            learning_rate_factor: Factor for learning rate adaptation
        """
        self.som = som
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.qe_threshold = qe_threshold
        self.te_threshold = te_threshold
        self.max_iterations = max_iterations
        self.patience = patience
        self.learning_rate_factor = learning_rate_factor
        
        # Training state
        self.current_learning_rate = initial_learning_rate
        self.qe_history = []
        self.te_history = []
        self.learning_rate_history = []
        
        logger.info(f"Initialized Simple VLRSOM:")
        logger.info(f"  QE threshold: {qe_threshold}")
        logger.info(f"  TE threshold: {te_threshold}")
        logger.info(f"  Max iterations: {max_iterations}")
        logger.info(f"  Learning rate range: [{min_learning_rate}, {max_learning_rate}]")
    
    def _adapt_learning_rate(self, current_qe: float, previous_qe: Optional[float]) -> float:
        """
        Adapt learning rate based on quantization error improvement.
        
        Core VLRSOM logic from research papers:
        - Increase rate if QE improving
        - Decrease rate if QE worsening
        - Simple multiplicative adaptation
        
        Args:
            current_qe: Current quantization error
            previous_qe: Previous quantization error
            
        Returns:
            Adapted learning rate
        """
        if previous_qe is None:
            return self.current_learning_rate
        
        # Calculate improvement ratio
        if previous_qe > 0:
            improvement_ratio = (previous_qe - current_qe) / previous_qe
        else:
            improvement_ratio = 0
        
        # Adapt learning rate based on improvement
        if improvement_ratio > 0.001:  # Good improvement
            # Increase learning rate (but not too much)
            new_rate = self.current_learning_rate * (1 + self.learning_rate_factor * improvement_ratio)
        elif improvement_ratio < -0.001:  # Getting worse
            # Decrease learning rate more aggressively
            new_rate = self.current_learning_rate * (1 - self.learning_rate_factor * abs(improvement_ratio))
        else:  # Minimal change
            # Slight decrease (natural cooling)
            new_rate = self.current_learning_rate * 0.999
        
        # Clamp to valid range
        new_rate = max(self.min_learning_rate, min(self.max_learning_rate, new_rate))
        
        return new_rate
    
    def _check_convergence(self, current_qe: float, current_te: float) -> Tuple[bool, str]:
        """
        Check convergence using simple dual-threshold approach from VLRSOM papers.
        
        Args:
            current_qe: Current quantization error
            current_te: Current topographic error
            
        Returns:
            Tuple of (converged, reason)
        """
        qe_converged = current_qe < self.qe_threshold
        te_converged = current_te < self.te_threshold
        
        if qe_converged and te_converged:
            return True, "both_thresholds_met"
        elif qe_converged:
            return False, "qe_converged_te_high"
        elif te_converged:
            return False, "te_converged_qe_high"
        else:
            return False, "neither_threshold_met"
    
    def train(self, 
              train_data: np.ndarray,
              validation_data: Optional[np.ndarray] = None,
              progress_callback: Optional[Any] = None) -> VLRSOMResult:
        """
        Train SOM using simple VLRSOM approach.
        
        Args:
            train_data: Training data (n_samples, n_features)
            validation_data: Optional validation data for monitoring
            progress_callback: Optional progress callback
            
        Returns:
            VLRSOMResult with training results
        """
        logger.info(f"Starting Simple VLRSOM training on {len(train_data):,} samples")
        start_time = time.time()
        
        # Reset training state
        self.current_learning_rate = self.initial_learning_rate
        self.qe_history = []
        self.te_history = []
        self.learning_rate_history = []
        
        best_qe = float('inf')
        patience_counter = 0
        
        for iteration in range(self.max_iterations):
            # Perform one training iteration
            # Use stochastic training (one random sample per iteration)
            sample_idx = np.random.randint(0, len(train_data))
            sample = train_data[sample_idx]
            
            # Train SOM with current learning rate
            self.som.train_single(sample, self.current_learning_rate, self.som.sigma)
            
            # Calculate errors every 10 iterations (efficiency)
            if iteration % 10 == 0:
                # Use validation data if available, otherwise training data
                eval_data = validation_data if validation_data is not None else train_data
                
                current_qe = self.som.quantization_error(eval_data)
                current_te = self.som.topographic_error(eval_data)
                
                # Store history
                self.qe_history.append(current_qe)
                self.te_history.append(current_te)
                self.learning_rate_history.append(self.current_learning_rate)
                
                # Adapt learning rate based on QE trend
                previous_qe = self.qe_history[-2] if len(self.qe_history) > 1 else None
                self.current_learning_rate = self._adapt_learning_rate(current_qe, previous_qe)
                
                # Check convergence
                converged, reason = self._check_convergence(current_qe, current_te)
                
                # Early stopping based on validation QE
                if current_qe < best_qe:
                    best_qe = current_qe
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Progress callback
                if progress_callback and iteration % 100 == 0:
                    progress_callback(iteration, self.max_iterations, 
                                    f"Iter {iteration}: QE={current_qe:.6f}, TE={current_te:.4f}, LR={self.current_learning_rate:.4f}")
                
                # Convergence check
                if converged:
                    logger.info(f"VLRSOM converged at iteration {iteration}: {reason}")
                    logger.info(f"Final QE: {current_qe:.6f}, Final TE: {current_te:.4f}")
                    break
                
                # Patience check
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at iteration {iteration} (patience exceeded)")
                    break
                
                # Log progress
                if iteration % 200 == 0:
                    logger.info(f"Iteration {iteration}: QE={current_qe:.6f}, TE={current_te:.4f}, LR={self.current_learning_rate:.4f}")
        
        # Final evaluation
        eval_data = validation_data if validation_data is not None else train_data
        final_qe = self.som.quantization_error(eval_data)
        final_te = self.som.topographic_error(eval_data)
        training_time = time.time() - start_time
        
        result = VLRSOMResult(
            converged=converged,
            final_quantization_error=final_qe,
            final_topographic_error=final_te,
            total_iterations=iteration + 1,
            training_time=training_time,
            learning_rate_history=self.learning_rate_history.copy(),
            qe_history=self.qe_history.copy(),
            te_history=self.te_history.copy()
        )
        
        logger.info(f"VLRSOM training completed in {training_time:.2f}s")
        logger.info(f"Final metrics: QE={final_qe:.6f}, TE={final_te:.4f}")
        
        return result


def create_simple_vlrsom(som, **kwargs) -> SimpleVLRSOM:
    """Factory function for creating Simple VLRSOM trainer."""
    return SimpleVLRSOM(som, **kwargs)