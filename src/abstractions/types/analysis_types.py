"""Type definitions for analysis results."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class AnalysisResult:
    """Result of an analysis operation."""
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None
    warnings: Optional[list] = None
    processing_time: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize optional fields."""
        if self.warnings is None:
            self.warnings = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass 
class ModelMetrics:
    """Metrics for a trained model."""
    fit: float
    relative_error: float
    iterations: int
    convergence_history: list
    final_error: float
    component_weights: Any  # numpy array or list