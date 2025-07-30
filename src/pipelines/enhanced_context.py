"""Enhanced pipeline context with integrated monitoring and logging."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.infrastructure.logging import LoggingContext
from src.infrastructure.monitoring import UnifiedMonitor
from src.core.enhanced_progress_manager import get_enhanced_progress_manager


@dataclass
class EnhancedPipelineContext:
    """Enhanced pipeline context with monitoring and logging integration.
    
    Extends the basic PipelineContext with:
    - Structured logging context
    - Unified monitoring (progress + metrics)
    - Enhanced progress manager with persistence
    """
    # Core fields (same as PipelineContext)
    config: Config
    db: DatabaseManager
    experiment_id: str
    checkpoint_dir: Path
    output_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced fields
    logging_context: Optional[LoggingContext] = field(default=None, init=False)
    monitor: Optional[UnifiedMonitor] = field(default=None, init=False)
    progress_manager: Optional[Any] = field(default=None, init=False)
    memory_monitor: Optional[Any] = field(default=None)
    
    def __post_init__(self):
        """Initialize monitoring and logging components."""
        # Setup logging context
        job_id = self.metadata.get('job_id')
        self.logging_context = LoggingContext(self.experiment_id, job_id)
        
        # Setup unified monitor
        self.monitor = UnifiedMonitor(self.config, self.db)
        
        # Setup enhanced progress manager
        self.progress_manager = get_enhanced_progress_manager(
            experiment_id=self.experiment_id,
            db_manager=self.db
        )
    
    def start_monitoring(self):
        """Start monitoring for the pipeline execution."""
        if self.monitor:
            self.monitor.start(self.experiment_id, self.metadata.get('job_id'))
    
    def stop_monitoring(self):
        """Stop monitoring."""
        if self.monitor:
            self.monitor.stop()
    
    # Compatibility methods (same as PipelineContext)
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from shared data."""
        return self.shared_data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in shared data."""
        self.shared_data[key] = value
    
    def update_metadata(self, **kwargs):
        """Update metadata."""
        self.metadata.update(kwargs)
    
    def add_quality_metric(self, name: str, value: Any):
        """Add quality metric."""
        self.quality_metrics[name] = value
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get all quality metrics."""
        return self.quality_metrics.copy()
    
    # New monitoring methods
    def log_progress(self, completed: int, total: int, message: Optional[str] = None):
        """Log progress for current operation.
        
        Args:
            completed: Completed units
            total: Total units
            message: Optional progress message
        """
        if self.logging_context:
            self.logging_context.log_progress(completed, total, message)
    
    def record_metrics(self, **metrics):
        """Record performance metrics.
        
        Args:
            **metrics: Metric name-value pairs
        """
        if self.monitor:
            node_id = self.logging_context.current_node if self.logging_context else None
            self.monitor.record_metrics(metrics, node_id)
    
    @classmethod
    def from_pipeline_context(cls, context: Any) -> 'EnhancedPipelineContext':
        """Create enhanced context from existing PipelineContext.
        
        Args:
            context: Existing PipelineContext instance
            
        Returns:
            EnhancedPipelineContext with same data
        """
        enhanced = cls(
            config=context.config,
            db=context.db,
            experiment_id=context.experiment_id,
            checkpoint_dir=context.checkpoint_dir,
            output_dir=context.output_dir,
            metadata=context.metadata.copy(),
            shared_data=context.shared_data.copy(),
            quality_metrics=context.quality_metrics.copy(),
            memory_monitor=getattr(context, 'memory_monitor', None)
        )
        return enhanced