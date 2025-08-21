"""Logging context management for pipeline execution correlation."""

from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import uuid
import time
from datetime import datetime

from .structured_logger import (
    experiment_context, node_context, stage_context, job_context,
    get_logger
)


class LoggingContext:
    """Manages logging context throughout pipeline execution.
    
    Provides hierarchical context management for:
    - Experiments
    - Pipeline stages  
    - Operations
    - Sub-operations
    
    Context is automatically propagated to all log messages within scope.
    """
    
    def __init__(self, experiment_id: Optional[str] = None, job_id: Optional[str] = None):
        """Initialize logging context.
        
        Args:
            experiment_id: Experiment UUID (generated if not provided)
            job_id: Job UUID for tracking
        """
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.job_id = job_id
        self.node_stack: List[str] = []
        self.stage_stack: List[str] = []
        self.timings: Dict[str, Dict[str, Any]] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    @contextmanager
    def pipeline(self, name: str, **metadata):
        """Context for pipeline execution.
        
        Args:
            name: Pipeline name
            **metadata: Additional metadata to log
            
        Example:
            with ctx.pipeline('biodiversity_analysis'):
                # Pipeline code here
        """
        node_id = f"pipeline_{name}"
        start_time = time.time()
        
        # Set context vars
        experiment_context.set(self.experiment_id)
        if self.job_id:
            job_context.set(self.job_id)
        node_context.set(node_id)
        
        self.node_stack.append(node_id)
        
        # Log pipeline start
        self.logger.info(
            f"Pipeline started: {name}",
            extra={'context': {'pipeline_name': name, **metadata}}
        )
        
        try:
            yield self
        finally:
            # Log pipeline completion
            duration = time.time() - start_time
            self.logger.log_performance(
                f"pipeline_{name}",
                duration,
                status='completed'
            )
            
            # Clean up context
            self.node_stack.pop()
            if self.node_stack:
                node_context.set(self.node_stack[-1])
            else:
                node_context.set(None)
    
    @contextmanager
    def stage(self, name: str, **metadata):
        """Context for stage execution.
        
        Args:
            name: Stage name
            **metadata: Additional metadata
            
        Example:
            with ctx.stage('data_loading'):
                # Stage code here
        """
        stage_context.set(name)
        self.stage_stack.append(name)
        
        # Create hierarchical node ID
        parent = self.node_stack[-1] if self.node_stack else "unknown"
        node_id = f"{parent}/{name}"
        node_context.set(node_id)
        self.node_stack.append(node_id)
        
        start_time = time.time()
        
        # Log stage start
        self.logger.info(
            f"Stage started: {name}",
            extra={'context': {'stage_name': name, **metadata}}
        )
        
        try:
            yield self
            status = 'completed'
        except Exception as e:
            status = 'failed'
            self.logger.log_error_with_context(e, operation=f"stage_{name}")
            raise
        finally:
            # Log stage completion
            duration = time.time() - start_time
            self.timings[node_id] = {
                'duration': duration,
                'status': status,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.log_performance(
                f"stage_{name}",
                duration,
                status=status
            )
            
            # Clean up context
            self.stage_stack.pop()
            self.node_stack.pop()
            
            if self.stage_stack:
                stage_context.set(self.stage_stack[-1])
            else:
                stage_context.set(None)
                
            if self.node_stack:
                node_context.set(self.node_stack[-1])
    
    @contextmanager
    def operation(self, name: str, **metadata):
        """Context for specific operations within stages.
        
        Args:
            name: Operation name
            **metadata: Additional metadata
            
        Example:
            with ctx.operation('validate_data', record_count=1000):
                # Operation code here
        """
        parent = self.node_stack[-1] if self.node_stack else "unknown"
        node_id = f"{parent}/{name}"
        node_context.set(node_id)
        self.node_stack.append(node_id)
        
        start_time = time.time()
        
        # Log operation start with metadata
        self.logger.debug(
            f"Operation started: {name}",
            extra={'context': metadata}
        )
        
        try:
            yield self
            status = 'success'
        except Exception as e:
            status = 'failed'
            self.logger.log_error_with_context(
                e, 
                operation=name,
                **metadata
            )
            raise
        finally:
            # Log operation completion
            duration = time.time() - start_time
            self.logger.log_performance(
                name,
                duration,
                status=status,
                **metadata
            )
            
            # Clean up
            self.node_stack.pop()
            if self.node_stack:
                node_context.set(self.node_stack[-1])
    
    @contextmanager
    def checkpoint(self, name: str):
        """Context for checkpoint operations.
        
        Args:
            name: Checkpoint name
        """
        with self.operation(f"checkpoint_{name}", checkpoint_type=name):
            yield
    
    def log_progress(self, completed: int, total: int, message: str = None):
        """Log progress update for current operation.
        
        Args:
            completed: Number of completed items
            total: Total number of items
            message: Optional progress message
        """
        percent = (completed / total * 100) if total > 0 else 0
        current_node = self.node_stack[-1] if self.node_stack else "unknown"
        
        log_msg = f"Progress: {percent:.1f}% ({completed}/{total})"
        if message:
            log_msg += f" - {message}"
            
        self.logger.info(
            log_msg,
            extra={
                'context': {
                    'progress_percent': percent,
                    'completed_units': completed,
                    'total_units': total
                }
            }
        )
    
    def get_timings(self) -> Dict[str, Dict[str, Any]]:
        """Get timing information for all operations.
        
        Returns:
            Dict mapping node_id to timing info
        """
        return self.timings.copy()
    
    @property
    def current_node(self) -> Optional[str]:
        """Get current node ID."""
        return self.node_stack[-1] if self.node_stack else None
    
    @property
    def current_stage(self) -> Optional[str]:
        """Get current stage name."""
        return self.stage_stack[-1] if self.stage_stack else None


# Global context instance for easy access
_global_context: Optional[LoggingContext] = None


def get_global_context() -> Optional[LoggingContext]:
    """Get the global logging context if set."""
    return _global_context


def set_global_context(context: LoggingContext):
    """Set the global logging context."""
    global _global_context
    _global_context = context


@contextmanager
def temporary_context(**fields):
    """Temporarily add context fields.
    
    Args:
        **fields: Context fields to add
        
    Example:
        with temporary_context(user_id='123', request_id='abc'):
            logger.info("Processing request")  # Will include user_id and request_id
    """
    logger = get_logger('context')
    logger.add_context(**fields)
    
    try:
        yield
    finally:
        logger.remove_context(*fields.keys())