"""Advanced cutting-edge convergence methods for SOM training."""

import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import math

from src.infrastructure.logging import get_logger
from .dynamic_convergence import ConvergenceMetrics, DynamicConvergenceDetector

logger = get_logger(__name__)


# Manhattan distance is always used for biodiversity data (objectively better)


class LearningRateSchedule(Enum):
    """Learning rate scheduling strategies."""
    FIXED = "fixed"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    VLRSOM = "vlrsom"  # Variable Learning Rate SOM
    ADAPTIVE = "adaptive"


@dataclass
class UnifiedConvergenceIndex:
    """Unified convergence index combining multiple quality measures."""
    embedding_accuracy: float  # Similar to quantization error
    topographic_accuracy: float  # Topology preservation
    convergence_index: float  # Combined metric
    is_converged: bool
    confidence: float  # Statistical confidence


class AdvancedConvergenceDetector(DynamicConvergenceDetector):
    """Advanced convergence detector with cutting-edge methods.
    
    Implements:
    1. Unified Convergence Index (Tatoian & Hamel, 2018)
    2. Variable Learning Rate SOM (VLRSOM, 2022)
    3. Batch SOM convergence criteria
    4. Alternative distance metrics
    """
    
    def __init__(
        self,
        learning_rate_schedule: LearningRateSchedule = LearningRateSchedule.VLRSOM,
        unified_convergence: bool = True,
        alpha: float = 0.6,  # Weight for embedding accuracy
        beta: float = 0.4,   # Weight for topographic accuracy
        convergence_threshold: float = 0.95,  # Unified index threshold
        confidence_level: float = 0.95,  # Statistical confidence
        **kwargs
    ):
        """Initialize advanced convergence detector.
        
        Args:
            learning_rate_schedule: Learning rate adaptation strategy
            unified_convergence: Use unified convergence index
            alpha: Weight for embedding accuracy in unified index
            beta: Weight for topographic accuracy in unified index
            convergence_threshold: Threshold for unified convergence index
            confidence_level: Statistical confidence level
            **kwargs: Additional parameters for base class
        """
        super().__init__(**kwargs)
        
        # Always use Manhattan distance (matches SOM training)
        self.distance_metric = "manhattan"
        self.learning_rate_schedule = learning_rate_schedule
        self.unified_convergence = unified_convergence
        self.alpha = alpha
        self.beta = beta
        self.convergence_threshold = convergence_threshold
        self.confidence_level = confidence_level
        
        # Advanced state tracking
        self.embedding_accuracies: List[float] = []
        self.topographic_accuracies: List[float] = []
        self.unified_indices: List[float] = []
        self.learning_rates: List[float] = []
        
        # VLRSOM parameters
        self.initial_learning_rate = 0.5
        self.final_learning_rate = 0.01
        self.learning_rate_factor = 0.95
        
        logger.info(f"Initialized advanced convergence detector:")
        logger.info(f"  Distance metric: {self.distance_metric} (optimized for biodiversity data)")
        logger.info(f"  Learning rate schedule: {learning_rate_schedule.value}")
        logger.info(f"  Unified convergence: {unified_convergence}")
        logger.info(f"  Convergence threshold: {convergence_threshold}")
    
    def calculate_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Manhattan distance (optimized for biodiversity data)."""
        return np.sum(np.abs(a - b))
    
    def calculate_adaptive_learning_rate(
        self,
        iteration: int,
        max_iterations: int,
        current_error: float,
        previous_error: Optional[float] = None
    ) -> float:
        """Calculate adaptive learning rate using various strategies."""
        
        if self.learning_rate_schedule == LearningRateSchedule.FIXED:
            return self.initial_learning_rate
        
        elif self.learning_rate_schedule == LearningRateSchedule.LINEAR_DECAY:
            # Linear decay from initial to final
            decay_factor = iteration / max_iterations
            return self.initial_learning_rate * (1 - decay_factor) + self.final_learning_rate * decay_factor
        
        elif self.learning_rate_schedule == LearningRateSchedule.EXPONENTIAL_DECAY:
            # Exponential decay
            return self.initial_learning_rate * (self.learning_rate_factor ** iteration)
        
        elif self.learning_rate_schedule == LearningRateSchedule.VLRSOM:
            # Variable Learning Rate SOM (VLRSOM) - 2022 research
            return self._calculate_vlrsom_rate(iteration, max_iterations, current_error)
        
        elif self.learning_rate_schedule == LearningRateSchedule.ADAPTIVE:
            # Adaptive based on error improvement
            if previous_error is not None and previous_error > 0:
                improvement_ratio = (previous_error - current_error) / previous_error
                # Increase rate if improving well, decrease if not
                adaptation = 1.0 + 0.1 * improvement_ratio
                base_rate = self.initial_learning_rate * (self.learning_rate_factor ** iteration)
                return max(self.final_learning_rate, min(self.initial_learning_rate, base_rate * adaptation))
            else:
                return self.initial_learning_rate * (self.learning_rate_factor ** iteration)
        
        return self.initial_learning_rate
    
    def _calculate_vlrsom_rate(self, iteration: int, max_iterations: int, current_error: float) -> float:
        """Calculate Variable Learning Rate SOM (VLRSOM) rate.
        
        Based on: "A faster dynamic convergency approach for self-organizing maps" (2022)
        """
        # Phase-based learning rate adaptation
        phase_ratio = iteration / max_iterations
        
        if phase_ratio < 0.3:  # Initial phase - higher learning rate
            base_rate = self.initial_learning_rate
            phase_factor = 1.0
        elif phase_ratio < 0.7:  # Middle phase - adaptive rate
            base_rate = self.initial_learning_rate * 0.7
            # Adapt based on error
            error_factor = max(0.5, min(1.5, 1.0 / (1.0 + current_error)))
            phase_factor = error_factor
        else:  # Final phase - fine-tuning
            base_rate = self.final_learning_rate * 2
            phase_factor = 0.8
        
        # Apply smooth decay
        decay = math.exp(-iteration / (max_iterations * 0.3))
        vlr_rate = base_rate * phase_factor * decay
        
        # Ensure minimum rate
        return max(self.final_learning_rate, vlr_rate)
    
    def calculate_unified_convergence_index(
        self,
        som,
        data: np.ndarray,
        iteration: int
    ) -> UnifiedConvergenceIndex:
        """Calculate unified convergence index (Tatoian & Hamel, 2018).
        
        Combines embedding accuracy and topographic accuracy into single metric.
        """
        # Calculate embedding accuracy (inverse of quantization error)
        embedding_accuracy = self._calculate_embedding_accuracy(som, data)
        
        # Calculate topographic accuracy (inverse of topographic error)
        topographic_accuracy = self._calculate_topographic_accuracy(som, data)
        
        # Calculate unified convergence index
        convergence_index = (
            self.alpha * embedding_accuracy + 
            self.beta * topographic_accuracy
        )
        
        # Statistical confidence calculation
        confidence = self._calculate_statistical_confidence(
            embedding_accuracy, topographic_accuracy, len(data)
        )
        
        # Determine convergence
        is_converged = (
            convergence_index >= self.convergence_threshold and
            confidence >= self.confidence_level
        )
        
        # Store for history
        self.embedding_accuracies.append(embedding_accuracy)
        self.topographic_accuracies.append(topographic_accuracy)
        self.unified_indices.append(convergence_index)
        
        return UnifiedConvergenceIndex(
            embedding_accuracy=embedding_accuracy,
            topographic_accuracy=topographic_accuracy,
            convergence_index=convergence_index,
            is_converged=is_converged,
            confidence=confidence
        )
    
    def _calculate_embedding_accuracy(self, som, data: np.ndarray) -> float:
        """Calculate embedding accuracy (map fitting quality)."""
        total_distance = 0.0
        max_distance = 0.0
        
        for sample in data:
            # Find BMU using selected distance metric
            min_distance = float('inf')
            for i in range(som.get_weights().shape[0]):
                for j in range(som.get_weights().shape[1]):
                    distance = self.calculate_distance(sample, som.get_weights()[i, j])
                    min_distance = min(min_distance, distance)
            
            total_distance += min_distance
            max_distance = max(max_distance, min_distance)
        
        # Calculate average quantization error
        avg_qe = total_distance / len(data)
        
        # Convert to accuracy (higher is better)
        # Normalize by maximum possible distance
        if max_distance > 0:
            embedding_accuracy = 1.0 - (avg_qe / max_distance)
        else:
            embedding_accuracy = 1.0
        
        return max(0.0, min(1.0, embedding_accuracy))
    
    def _calculate_topographic_accuracy(self, som, data: np.ndarray) -> float:
        """Calculate topographic accuracy (topology preservation)."""
        correct_topology = 0
        total_samples = len(data)
        
        for sample in data:
            # Find two best matching units using selected distance metric
            distances = []
            for i in range(som.get_weights().shape[0]):
                for j in range(som.get_weights().shape[1]):
                    distance = self.calculate_distance(sample, som.get_weights()[i, j])
                    distances.append((distance, i, j))
            
            # Sort by distance
            distances.sort(key=lambda x: x[0])
            
            if len(distances) >= 2:
                _, bmu1_i, bmu1_j = distances[0]
                _, bmu2_i, bmu2_j = distances[1]
                
                # Check if BMUs are neighbors (8-connectivity)
                distance_between_bmus = max(abs(bmu1_i - bmu2_i), abs(bmu1_j - bmu2_j))
                
                if distance_between_bmus <= 1:  # Are neighbors
                    correct_topology += 1
        
        return correct_topology / total_samples if total_samples > 0 else 0.0
    
    def _calculate_statistical_confidence(
        self,
        embedding_accuracy: float,
        topographic_accuracy: float,
        sample_size: int
    ) -> float:
        """Calculate statistical confidence for convergence decision."""
        # Use confidence interval calculation
        # Simplified version - in practice, you'd use proper statistical tests
        
        # Standard error approximation
        se_embedding = math.sqrt(embedding_accuracy * (1 - embedding_accuracy) / sample_size)
        se_topographic = math.sqrt(topographic_accuracy * (1 - topographic_accuracy) / sample_size)
        
        # Combined standard error
        combined_se = math.sqrt(
            (self.alpha * se_embedding) ** 2 + 
            (self.beta * se_topographic) ** 2
        )
        
        # Z-score for 95% confidence
        z_score = 1.96  # For 95% confidence
        
        # Calculate confidence
        confidence = max(0.0, min(1.0, 1.0 - (z_score * combined_se)))
        
        return confidence
    
    def check_advanced_convergence(
        self,
        som,
        data: np.ndarray,
        iteration: int,
        max_iterations: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check convergence using advanced methods."""
        
        # Calculate adaptive learning rate
        current_qe = self._calculate_quantization_error(som, data)
        previous_qe = self.quantization_errors[-1] if self.quantization_errors else None
        
        learning_rate = self.calculate_adaptive_learning_rate(
            iteration, max_iterations, current_qe, previous_qe
        )
        self.learning_rates.append(learning_rate)
        
        # Use unified convergence index if enabled
        if self.unified_convergence:
            unified_result = self.calculate_unified_convergence_index(som, data, iteration)
            
            convergence_info = {
                'method': 'unified_convergence_index',
                'embedding_accuracy': unified_result.embedding_accuracy,
                'topographic_accuracy': unified_result.topographic_accuracy,
                'convergence_index': unified_result.convergence_index,
                'confidence': unified_result.confidence,
                'learning_rate': learning_rate,
                'distance_metric': self.distance_metric,
                'iteration': iteration
            }
            
            return unified_result.is_converged, convergence_info
        
        else:
            # Fall back to standard multi-criteria approach
            is_converged, metrics = self.check_convergence(som, data, iteration)
            
            convergence_info = {
                'method': 'multi_criteria',
                'weight_change': metrics.weight_change,
                'quantization_error': metrics.quantization_error,
                'topographic_error': metrics.topographic_error,
                'convergence_score': metrics.convergence_score,
                'learning_rate': learning_rate,
                'distance_metric': self.distance_metric,
                'iteration': iteration
            }
            
            return is_converged, convergence_info
    
    def get_advanced_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of advanced convergence analysis."""
        base_summary = self.get_convergence_summary()
        
        advanced_summary = {
            'distance_metric': self.distance_metric,
            'learning_rate_schedule': self.learning_rate_schedule.value,
            'unified_convergence_enabled': self.unified_convergence,
            'final_learning_rate': self.learning_rates[-1] if self.learning_rates else None,
            'learning_rate_trend': self._analyze_learning_rate_trend(),
        }
        
        if self.unified_convergence:
            advanced_summary.update({
                'final_embedding_accuracy': self.embedding_accuracies[-1] if self.embedding_accuracies else None,
                'final_topographic_accuracy': self.topographic_accuracies[-1] if self.topographic_accuracies else None,
                'final_unified_index': self.unified_indices[-1] if self.unified_indices else None,
                'unified_convergence_trend': self._analyze_unified_trend()
            })
        
        # Combine with base summary
        return {**base_summary, **advanced_summary}
    
    def _analyze_learning_rate_trend(self) -> str:
        """Analyze learning rate adaptation trend."""
        if len(self.learning_rates) < 3:
            return "insufficient_data"
        
        recent_rates = self.learning_rates[-3:]
        if recent_rates[-1] < recent_rates[0]:
            return "decreasing"
        elif recent_rates[-1] > recent_rates[0]:
            return "increasing"
        else:
            return "stable"
    
    def _analyze_unified_trend(self) -> str:
        """Analyze unified convergence index trend."""
        if len(self.unified_indices) < 3:
            return "insufficient_data"
        
        recent_indices = self.unified_indices[-3:]
        if recent_indices[-1] > recent_indices[0]:
            return "improving"
        elif recent_indices[-1] < recent_indices[0]:
            return "degrading"
        else:
            return "stable"


def create_advanced_convergence_detector(
    method: str = "unified",
    learning_rate_schedule: str = "vlrsom",
    **kwargs
) -> AdvancedConvergenceDetector:
    """Factory function for creating advanced convergence detector.
    
    Args:
        method: Convergence detection method ("unified" or "multi_criteria")
        learning_rate_schedule: Learning rate adaptation strategy
        **kwargs: Additional parameters
        
    Returns:
        Configured advanced convergence detector
    """
    return AdvancedConvergenceDetector(
        learning_rate_schedule=LearningRateSchedule(learning_rate_schedule),
        unified_convergence=(method == "unified"),
        **kwargs
    )