# src/pipelines/stages/base_stage.py
"""Base class for pipeline stages."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result from stage execution."""
    success: bool
    data: Dict[str, Any]
    metrics: Dict[str, Any]
    warnings: List[str] = []
    execution_time: float = 0.0
    output_size: int = 0  # bytes
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Each stage represents a discrete processing step with:
    - Dependencies on other stages
    - Resource requirements
    - Validation logic
    - Execution logic
    - Quality checks
    """
    
    def __init__(self):
        self.status = StageStatus.PENDING
        self.error: Optional[str] = None
        self.result: Optional[StageResult] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this stage."""
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """List of stage names this stage depends on."""
        pass
    
    @property
    def memory_requirements(self) -> Optional[float]:
        """Estimated memory requirements in GB."""
        return None
    
    @property
    def disk_requirements(self) -> Optional[float]:
        """Estimated disk space requirements in GB."""
        return None
    
    @property
    def estimated_operations(self) -> int:
        """Estimated number of operations for progress tracking."""
        return 100
    
    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate stage configuration and inputs.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass
    
    @abstractmethod
    def execute(self, context) -> StageResult:
        """
        Execute the stage.
        
        Args:
            context: PipelineContext with shared data and configuration
            
        Returns:
            StageResult with outputs and metrics
        """
        pass
    
    def pre_execute(self, context):
        """Hook called before execution."""
        pass
    
    def post_execute(self, context, result: StageResult):
        """Hook called after successful execution."""
        pass
    
    def on_failure(self, context, error: Exception):
        """Hook called on execution failure."""
        pass
    
    def cleanup(self, context):
        """Cleanup resources used by this stage."""
        pass