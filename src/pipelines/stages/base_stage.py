# src/pipelines/stages/base_stage.py
"""Base class for pipeline stages."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import logging

# Import ProcessingConfig from config module
from src.config.processing_config import ProcessingConfig

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PAUSED = "paused"


@dataclass
class StageResult:
    """Result from stage execution."""
    success: bool
    data: Dict[str, Any]
    metrics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    output_size: int = 0  # bytes
    memory_peak_mb: float = 0.0  # Peak memory usage
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages with memory-aware processing.
    
    Each stage represents a discrete processing step with:
    - Dependencies on other stages
    - Resource requirements
    - Validation logic
    - Execution logic
    - Quality checks
    - Memory-aware processing support
    """
    
    def __init__(self):
        self.status = StageStatus.PENDING
        self.error: Optional[str] = None
        self.result: Optional[StageResult] = None
        self.processing_config: Optional[ProcessingConfig] = None
        self._pause_requested = False
        self._cancel_requested = False
    
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
    
    @property
    def supports_chunking(self) -> bool:
        """Whether this stage supports chunked processing."""
        return False
    
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
    
    def set_processing_config(self, config: ProcessingConfig):
        """Set memory-aware processing configuration."""
        self.processing_config = config
    
    def pause(self):
        """Request pause of stage execution."""
        self._pause_requested = True
        self.status = StageStatus.PAUSED
    
    def resume(self):
        """Resume stage execution."""
        self._pause_requested = False
        if self.status == StageStatus.PAUSED:
            self.status = StageStatus.RUNNING
    
    def cancel(self):
        """Cancel stage execution."""
        self._cancel_requested = True
    
    def is_paused(self) -> bool:
        """Check if stage is paused."""
        return self._pause_requested
    
    def is_cancelled(self) -> bool:
        """Check if stage is cancelled."""
        return self._cancel_requested
    
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
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data for checkpointing stage state."""
        return {
            'status': self.status.value,
            'error': self.error
        }
    
    def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Restore stage state from checkpoint."""
        self.status = StageStatus(checkpoint_data.get('status', 'pending'))
        self.error = checkpoint_data.get('error')